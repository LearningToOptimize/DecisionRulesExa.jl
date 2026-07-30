# runtests_phase2.jl — Phase-2 suite for the REPAIRED implementation.
#
# Covers: shared-ACP parity with the accepted Phase-1 builder; the demand process
# and paired protocol; the reachable policy (stretchedsigmoid / hardsigmoidsafe,
# safe upper margin, non-differentiable bounds); the two-sided active nodal
# recourse (unbounded deficit d⁺ + surplus d⁻ at every bus, hard reactive KCL,
# recourse pricing and inclusion in physical cost) that gives STRICT complete
# recourse; strict absolute recourse across cases and target classes reported by
# DIRECTION; strict and soft target modes with the split-slack soft form; the
# multiplier finite-difference check; checkpointing; and CPU/GPU structural parity.
#
# Run (from examples/BatteryStorageOPF):
#   module load julia
#   julia --pkgimages=no --project=. test/runtests_phase2.jl
#
# GPU tests are gated on CUDA.functional() and reported as skipped on CPU nodes.

include(joinpath(@__DIR__, "..", "src", "BatteryStorageOPF.jl"))
using .BatteryStorageOPF
using Test
using MadNLP
using JSON
using Flux
using Zygote
using CUDA
using DecisionRulesExa
using LinearAlgebra
using Statistics
using Random

const SMALL = "case14_ieee"

# DECLARED numerical tolerances for "zero active recourse". An interior-point
# solution leaves the recourse power at barrier tolerance rather than exactly 0,
# so "zero" is asserted at these declared levels. They are ~4-6 orders of
# magnitude below any material recourse (a genuinely recoursing path carries
# O(1-10) MWh), so they cannot mask a real failure. Deficit d⁺ and surplus d⁻ are
# each held below RECOURSE_ENERGY_TOL_MWH; their sum below 2×.
const RECOURSE_ENERGY_TOL_MWH = 1e-3
const RECOURSE_PU_TOL         = 1e-4

# Fresh-solver solve used throughout (avoids CommonSolve.solve! ambiguity).
solve_tsddr(de) = MadNLP.madnlp(de.model; print_level = MadNLP.ERROR, tol = 1e-8)

# Default (stronger) fixture, used where demand stress is irrelevant.
function fixture(; nbat = 2, nregion = 2, seed = 11)
    case = make_battery_case(SMALL; number_of_batteries = nbat, seed = seed,
                             fleet_power_fraction = 0.3, duration_hours = 4.0)
    process = make_load_process(case; nregion = nregion, period = 6,
                                train_seed = 101, eval_seed = 202)
    return case, process
end

# Mild, demand-feasible fixture (peak ≈1.1×) so an accepted run sheds no load.
function mild_fixture(; nbat = 2, nregion = 2, seed = 11)
    case = make_battery_case(SMALL; number_of_batteries = nbat, seed = seed,
                             fleet_power_fraction = 0.3, duration_hours = 4.0)
    atoms = [LoadAtom(1.0, [1.0, 1.0]), LoadAtom(1.03, [1.05, 0.99]), LoadAtom(1.03, [0.99, 1.05])]
    process = make_load_process(case; nregion = nregion, period = 6, base_amplitude = 0.04,
                                atoms = atoms, probs = [0.5, 0.25, 0.25],
                                train_seed = 101, eval_seed = 202)
    return case, process
end

# Roll the policy forward to build a reachable target trajectory for scenario w.
function policy_targets(pol, process, w, e0, T)
    Flux.reset!(pol); prev = e0; xh = Float64[]
    nw = n_uncertainty(process)
    for t in 1:T
        wt = w[(t-1)*nw+1 : t*nw]
        xt = pol(vcat(wt, prev)); append!(xh, xt); prev = xt
    end
    return xh
end

@testset "BatteryStorageOPF Phase 2 (repaired)" begin

    # ── REPAIR 1: shared ACP source of truth ─────────────────────────────────
    @testset "shared-ACP parity: stochastic builder reproduces Phase 1" begin
        case, process = mild_fixture()
        T = 2
        # Realized demand for a chosen scenario.
        w = materialize_scenario(process, [1, 1]; horizon = T)
        nw = n_uncertainty(process); nd = case.network
        # Phase-1 builder with the SAME realized demand: atom 1 is the calm atom
        # (L=1, R=1), so the realized multiplier is exactly the daily shape h_t.
        prof = [process.base_shape[((t - 1) % process.period) + 1] for t in 1:T]
        p1 = build_battery_de(case, T; stage_hours = 1.0, demand_profile = prof)
        r1 = solve_de!(p1; print_level = MadNLP.ERROR, tol = 1e-8)
        @test BatteryStorageOPF.solve_succeeded(r1.status)

        # Stochastic builder with targets DISABLED (soft, zero penalty, targets
        # pinned to the realized Phase-1 SoC) and active recourse FIXED TO ZERO by
        # bounds — not by a zero price, which would make recourse free and is the
        # opposite of disabling it.
        p2 = build_battery_tsddr_de(case, process; reporting_horizon = T, lookahead = 0,
                                    mode = :soft, rho1 = 0.0, rho2 = 0.0,
                                    allow_active_recourse = false, stage_hours = 1.0)
        set_tsddr_initial_soc!(p2, policy_initial_state(case; float_type = Float64))
        set_tsddr_uncertainty!(p2, w)
        # Realized demand must match Phase-1's baked profile exactly.
        for t in 1:T, b in 1:nbus(nd)
            @test isapprox(p2.realized_pd[t, b], nd.bus_pd[b] * prof[t]; rtol = 1e-12)
            @test isapprox(p2.realized_qd[t, b], nd.bus_qd[b] * prof[t]; rtol = 1e-12)
        end
        # Pin the target to Phase-1's realized SoC path and forbid recourse, so
        # the two models describe the same optimization problem.
        s1 = battery_solution(p1, r1)
        set_tsddr_targets!(p2, Float64[s1.soc[k, t+1] for t in 1:T for k in 1:p2.nBat])
        r2 = solve_tsddr(p2)
        @test DecisionRulesExa.solve_succeeded(r2)
        d2 = decompose_costs(p2, r2)
        # Public recourse energy is projected (≥ 0) — assert nonnegativity directly,
        # never via abs() on a value that must not be negative in the first place.
        @test 0.0 <= d2.active_deficit_energy_mwh < RECOURSE_ENERGY_TOL_MWH  # recourse fixed to zero
        @test 0.0 <= d2.active_surplus_energy_mwh < RECOURSE_ENERGY_TOL_MWH
        # Same physical operating cost as the Phase-1 objective.
        @test isapprox(d2.physical_operating_cost, r1.objective; rtol = 1e-5)
    end

    # ── Demand process + paired protocol ─────────────────────────────────────
    @testset "exact demand replay and distinct train/eval protocols" begin
        case, process = fixture()
        H, P = 5, 4
        m1 = scenario_index_matrix(process, H, P; seed = process.train_seed)
        m2 = scenario_index_matrix(process, H, P; seed = process.train_seed)
        @test m1 == m2
        @test index_matrix_hash(m1) == index_matrix_hash(m2)
        @test size(m1) == (H, P)
        @test all(1 .<= m1 .<= natom(process))
        me = scenario_index_matrix(process, H, P; seed = process.eval_seed)
        @test m1 != me
        @test process.train_seed != process.eval_seed
    end

    @testset "process validation" begin
        case, _ = fixture()
        @test_throws ErrorException make_load_process(case; nregion = 0)
        @test_throws ErrorException make_load_process(case; train_seed = 5, eval_seed = 5)
        @test_throws ErrorException make_load_process(case; nregion = 2,
            atoms = [LoadAtom(1.0, [1.0, 1.0]), LoadAtom(1.1, [1.2, 0.9])], probs = [0.3, 0.3])
        @test_throws ErrorException make_load_process(case; nregion = 2,
            atoms = [LoadAtom(1.0, [1.0])], probs = [1.0])
    end

    @testset "deterministic region assignment" begin
        case, _ = fixture(nregion = 3)
        r1, a1 = assign_regions(case.network, 3)
        r2, a2 = assign_regions(case.network, 3)
        @test r1 == r2 && a1 == a2
        @test length(r1) == nbus(case.network)
        @test all(1 .<= r1 .<= 3)
        @test length(unique(a1)) == 3
        @test length(Set(r1)) == 3
    end

    @testset "power factor preserved under demand scaling" begin
        case, process = fixture()
        nd = case.network
        de = build_battery_tsddr_de(case, process; reporting_horizon = 3, lookahead = 0,
                                    mode = :soft, stage_hours = 1.0)
        set_tsddr_uncertainty!(de, materialize_scenario(process, [2, 3, 1]; horizon = 3))
        for t in 1:3, b in 1:nbus(nd)
            nd.bus_pd[b] > 0 || continue
            @test isapprox(de.realized_qd[t, b] / de.realized_pd[t, b],
                           nd.bus_qd[b] / nd.bus_pd[b]; rtol = 1e-12)
        end
    end

    @testset "protocol write, reconstruct, tamper" begin
        case, process = fixture()
        H, P = 5, 6
        mat = scenario_index_matrix(process, H, P; seed = process.eval_seed)
        dir = mktempdir(); path = joinpath(dir, "eval_protocol.json")
        write_scenario_protocol(path, process, mat; kind = "eval", seed = process.eval_seed)
        pr2, mat2, meta = reconstruct_scenario_protocol(path)
        @test mat2 == mat
        @test process_hash(pr2) == process_hash(process)
        @test meta.kind == "eval" && meta.horizon == H && meta.paths == P
        @test materialize_all(pr2, mat2) == materialize_all(process, mat)
        doc = JSON.parsefile(path); doc["index_matrix_hash_sha256"] = repeat("0", 64)
        bad = joinpath(dir, "bad.json"); open(io -> JSON.print(io, doc, 2), bad, "w")
        @test_throws ErrorException reconstruct_scenario_protocol(bad)
    end

    @testset "stochastic manifest reconstruct" begin
        case, process = fixture()
        dir = mktempdir(); path = joinpath(dir, "manifest.json")
        write_stochastic_manifest(path, case, process;
            reporting_horizon = 4, lookahead = 2, mode = :soft, stage_hours = 1.0,
            active_recourse_cost_per_mwh = DEFAULT_ACTIVE_RECOURSE_COST_PER_MWH, rho1 = 0.0, rho2 = 12.5,
            activation = string(stretchedsigmoid), policy_layers = [8],
            train_index_matrix_hash = repeat("a", 64),
            eval_index_matrix_hash = repeat("b", 64),
            train_protocol_file_sha256 = repeat("c", 64),
            eval_protocol_file_sha256 = repeat("d", 64),
            train_paths = 8, eval_paths = 16)
        case2, process2, meta = reconstruct_stochastic_manifest(path)
        @test manifest_hash(case2) == manifest_hash(case)
        @test process_hash(process2) == process_hash(process)
        @test meta.reporting_horizon == 4 && meta.lookahead == 2 && meta.horizon == 6
        @test meta.active_recourse_cost_per_mwh == 10_000.0
        doc = JSON.parsefile(path)
        # The misleading load_shedding section is gone; active_recourse replaces it.
        @test !haskey(doc, "load_shedding")
        ar = doc["active_recourse"]
        @test ar["active_recourse_cost_per_mwh"] == 10_000.0
        @test ar["formulation"] == "two-sided active nodal slack"
        @test ar["unbounded_above"] == true
        @test ar["active_only"] == true
        @test ar["included_in_physical_operating_cost"] == true
        @test occursin("both directions", ar["accepted_scientific_requirement"])
        @test doc["target_penalty"]["included_in_physical_operating_cost"] == false
        @test doc["policy"]["safe_upper_margin"] == 1e-3
    end

    # ── REPAIR 2: reachable policy ───────────────────────────────────────────
    @testset "reachability bounds at interior and boundary" begin
        case, _ = fixture()
        b = case.batteries[1]; dt = 1.0
        a = 1 - b.sigma * dt
        dd = (dt / b.eta_dis) * b.p_discharge_max
        cg = b.eta_ch * dt * b.p_charge_max
        e = (b.e_min + b.e_max) / 2
        lo, up = battery_reachable_bounds([e], [a], [dd], [cg], [b.e_min], [b.e_max])
        @test lo[1] ≈ max(b.e_min, a*e - dd)
        @test up[1] ≈ min(b.e_max, a*e + cg)
        @test lo[1] <= up[1]
        _, up2 = battery_reachable_bounds([b.e_max], [a], [dd], [cg], [b.e_min], [b.e_max])
        @test up2[1] ≈ b.e_max
        lo3, _ = battery_reachable_bounds([b.e_min], [a], [dd], [cg], [b.e_min], [b.e_max])
        @test lo3[1] ≈ b.e_min
    end

    @testset "activations: safe upper margin, exact lower edge" begin
        # stretchedsigmoid: 0 at large negative, 1-1e-3 (NEVER 1) at large positive.
        @test stretchedsigmoid(-50.0) == 0.0
        @test stretchedsigmoid(50.0) == 1.0 - 1e-3
        @test stretchedsigmoid(50.0) < 1.0
        @test 0.4 < stretchedsigmoid(0.0) < 0.6
        # hardsigmoidsafe: same safe margin.
        @test hardsigmoidsafe(-50.0) == 0.0
        @test hardsigmoidsafe(50.0) == 1.0 - 1e-3
        @test hardsigmoidsafe(50.0) < 1.0
        @test hardsigmoidsafe(0.0) == 0.5
        # Only bounded activations are admissible.
        case, process = fixture()
        @test_throws ArgumentError battery_reachable_policy(case, process; layers = [4],
                                                           activation = Flux.sigmoid)
    end

    @testset "policy targets within reachability; default output range" begin
        case, process = fixture()
        for act in (stretchedsigmoid, hardsigmoidsafe)
            pol = battery_reachable_policy(case, process; dt = 1.0, layers = [8],
                                           activation = act, float_type = Float32)
            nw = n_uncertainty(process); nB = nbattery(case)
            Random.seed!(1)
            for _ in 1:15
                Flux.reset!(pol)
                e = Float32.(rand(nB) .* [b.e_max for b in case.batteries])
                y = pol(vcat(Float32.(rand(nw)), e))
                lo, up = battery_reachable_bounds(e, pol.a, pol.discharge_drop,
                                                  pol.charge_gain, pol.e_min, pol.e_max)
                @test all(lo .- 1f-4 .<= y .<= up .+ 1f-4)
                # Never at the exact upper edge (safe margin).
                @test all(y .<= up .- (up .- lo) .* 1f-4 .+ 1f-5)
            end
        end
    end

    @testset "no gradient through reachable bounds; gradients reach parameters" begin
        case, process = fixture()
        pol = battery_reachable_policy(case, process; dt = 1.0, layers = [8],
                                       combiner_layers = Int[], float_type = Float64)
        nw = n_uncertainty(process); nB = nbattery(case)
        w = Float64.(rand(nw)); e0 = policy_initial_state(case; float_type = Float64)
        # Gradients reach encoder and combiner parameters.
        g = Zygote.gradient(m -> (Flux.reset!(m); sum(m(vcat(w, e0)))), pol)[1]
        @test g.combiner !== nothing && all(isfinite, g.combiner.weight)
        @test g.encoder !== nothing
        # The bound computation itself carries NO gradient (physical data).
        gb = Zygote.gradient(e -> sum(sum(battery_reachable_bounds(e, pol.a, pol.discharge_drop,
                                                                  pol.charge_gain, pol.e_min, pol.e_max))),
                             e0)[1]
        @test gb === nothing || all(iszero, gb)
    end

    @testset "recurrent reset and determinism" begin
        case, process = fixture()
        pol = battery_reachable_policy(case, process; dt = 1.0, layers = [8], float_type = Float32)
        nw = n_uncertainty(process); nB = nbattery(case)
        inp = vcat(Float32.(rand(nw)), Float32.(fill(0.5, nB)))
        Flux.reset!(pol); y1 = copy(pol(inp))
        Flux.reset!(pol); y2 = copy(pol(inp))
        @test y1 ≈ y2
        Flux.reset!(pol); a1 = copy(pol(inp)); b1 = copy(pol(inp))
        Flux.reset!(pol); a2 = copy(pol(inp)); b2 = copy(pol(inp))
        @test a1 ≈ a2 && b1 ≈ b2
    end

    @testset "finite gradients + finite-difference" begin
        case, process = fixture()
        pol = battery_reachable_policy(case, process; dt = 1.0, layers = [8],
                                       combiner_layers = Int[], float_type = Float64)
        nw = n_uncertainty(process); T = 3
        wflat = Float64.(rand(T * nw)); x0 = policy_initial_state(case; float_type = Float64)

        # (a) Multi-stage rollout gradients are finite. NOTE: a finite difference
        # of THIS loss will NOT equal the analytic gradient, and that is correct:
        # feeding each target forward makes the next stage's reachable bounds
        # depend on the parameters, and the gradient through ℓ and u is
        # deliberately stopped (physical projection data, canonical spec).
        function rollout_loss(m)
            Flux.reset!(m); prev = x0; s = 0.0
            for t in 1:T
                xt = m(vcat(view(wflat, (t-1)*nw+1 : t*nw), prev)); s += sum(xt); prev = xt
            end
            return s
        end
        gr = Zygote.gradient(rollout_loss, pol)[1]
        @test all(isfinite, gr.combiner.weight) && all(isfinite, gr.combiner.bias)
        @test gr.encoder !== nothing

        # (b) SINGLE-stage loss with a CONSTANT previous state: no parameter path
        # runs through the (stopped) bounds, so the analytic gradient and a finite
        # difference must agree.
        e_fixed = copy(x0)
        w1 = Float64.(wflat[1:nw])
        stage_loss(m) = (Flux.reset!(m); sum(m(vcat(w1, e_fixed))))
        g = Zygote.gradient(stage_loss, pol)[1]
        @test all(isfinite, g.combiner.weight) && all(isfinite, g.combiner.bias)
        W = pol.combiner.weight; i, j = 1, 1; ε = 1e-6
        w0 = W[i, j]
        W[i, j] = w0 + ε; Lp = stage_loss(pol)
        W[i, j] = w0 - ε; Lm = stage_loss(pol)
        W[i, j] = w0
        @test isapprox((Lp - Lm) / (2ε), g.combiner.weight[i, j]; rtol = 1e-4, atol = 1e-8)
    end

    # ── REPAIR 3: two-sided active recourse = complete-recourse slack ─────────
    # The recourse device is a per-bus two-sided pair d⁺ (deficit / injection) and
    # d⁻ (surplus / absorption), each ≥ 0 and UNBOUNDED above, present at EVERY bus
    # (an artificial active-balance recourse, NOT a fraction of local demand: d⁺
    # may exceed local demand or be positive where p^d = 0). They enter the active
    # balance as `− d⁺ + d⁻`, guaranteeing the stage subproblem is feasible for
    # every incoming state and every reachable target; the reactive balance stays a
    # HARD equality (recourse is active-only). Priced at VOLL, so accepted runs ~0
    # in BOTH directions.
    @testset "two-sided active recourse: unbounded nodal slack, priced, hard reactive KCL" begin
        case, process = mild_fixture()
        nd = case.network
        de = build_battery_tsddr_de(case, process; reporting_horizon = 2, lookahead = 0,
                                    mode = :soft, stage_hours = 1.0)
        # Recourse bounds: 0 ≤ d⁺,d⁻ < ∞ at EVERY bus (complete recourse everywhere).
        lv = Array(de.model.meta.lvar); uv = Array(de.model.meta.uvar)
        T = de.horizon; nB = de.nBus; nG = de.nGen; nBR = de.nBranch; nK = de.nBat
        off = variable_offsets(de)
        for t in 1:T, b in 1:nB
            idx = off.active_deficit + (t-1)*nB + b       # deficit block
            @test lv[idx] == 0.0
            @test uv[idx] == Inf                          # unbounded at every bus
            sidx = off.active_surplus + (t-1)*nB + b      # surplus block
            @test lv[sidx] == 0.0
            @test uv[sidx] == Inf
        end
        # allow_active_recourse=false fixes BOTH slack blocks to zero.
        dff = build_battery_tsddr_de(case, process; reporting_horizon = 2, lookahead = 0,
                                     mode = :soft, stage_hours = 1.0, allow_active_recourse = false)
        uvf = Array(dff.model.meta.uvar)
        for t in 1:T, b in 1:nB
            @test uvf[off.active_deficit + (t-1)*nB + b] == 0.0
            @test uvf[off.active_surplus + (t-1)*nB + b] == 0.0
        end
        # A feasible instance uses ~0 recourse (both directions) and reactive is hard.
        e0 = policy_initial_state(case; float_type = Float64)
        w = materialize_scenario(process, [1, 1]; horizon = T)
        set_tsddr_initial_soc!(de, e0); set_tsddr_uncertainty!(de, w)
        set_tsddr_targets!(de, Float64[e for _ in 1:T for e in e0])
        r = solve_tsddr(de)
        @test DecisionRulesExa.solve_succeeded(r)
        d = decompose_costs(de, r)
        # Zero-recourse path: public values are projected (≥ 0) and below tolerance —
        # asserted WITHOUT abs (a negative value would itself be a defect).
        @test 0.0 <= d.active_deficit_energy_mwh < RECOURSE_ENERGY_TOL_MWH  # ~zero deficit
        @test 0.0 <= d.active_surplus_energy_mwh < RECOURSE_ENERGY_TOL_MWH  # ~zero surplus
        @test 0.0 <= d.max_active_deficit_pu < RECOURSE_PU_TOL
        @test 0.0 <= d.max_active_surplus_pu < RECOURSE_PU_TOL
        # Raw per-variable bound noise stayed inside the declared tolerance.
        @test d.maximum_active_recourse_lower_bound_violation_pu <= ACTIVE_RECOURSE_LB_TOL_PU
        @test tsddr_max_primal_residual(de, r) < 1e-5           # hard reactive KCL satisfied

        # The two-sided slack provides recourse for BATTERY targets that are NOT
        # network-deliverable — the whole point of complete recourse. On case300
        # a strict maximal-CHARGE target (upper reachable endpoint) needs power the
        # congested network cannot import: the DEFICIT (d⁺) engages and the strict
        # stage STILL SOLVES. A strict maximal-DISCHARGE target (lower endpoint)
        # needs power the network cannot export: the SURPLUS (d⁻) engages instead.
        c300 = make_battery_case("case300_ieee"; number_of_batteries = 20, seed = 20260722)
        p300 = make_load_process(c300; preset = DEFAULT_DEMAND_PRESET, nregion = 3, period = 4)
        e300 = Float64.(policy_initial_state(c300; float_type = Float64))
        w300 = materialize_scenario(p300, [1]; horizon = 1)
        st300 = build_battery_stage_problem(c300, p300; mode = :strict, stage_hours = 1.0)
        set_tsddr_initial_soc!(st300, e300); set_tsddr_uncertainty!(st300, w300)
        up = [min(b.e_max, (1 - b.sigma)*e300[k] + b.eta_ch*b.p_charge_max)
              for (k, b) in enumerate(c300.batteries)]
        lo = [max(b.e_min, (1 - b.sigma)*e300[k] - (1/b.eta_dis)*b.p_discharge_max)
              for (k, b) in enumerate(c300.batteries)]
        # Maximal charge → the DEFICIT direction provides recourse.
        set_tsddr_targets!(st300, up)
        rc = solve_tsddr(st300)
        @test DecisionRulesExa.solve_succeeded(rc)          # recourse ⇒ feasible
        dc = decompose_costs(st300, rc); solc = tsddr_solution(st300, rc)
        @test dc.active_deficit_pu > 1e-3                   # projected deficit engaged
        # PUBLIC recourse quantities are ALL finite and ≥ 0 (never negative energy/cost).
        for x in (dc.active_deficit_pu, dc.active_surplus_pu, dc.active_deficit_energy_mwh,
                  dc.active_surplus_energy_mwh, dc.total_active_recourse_energy_mwh,
                  dc.max_active_deficit_pu, dc.max_active_surplus_pu, dc.active_recourse_cost)
            @test isfinite(x) && x >= 0.0
        end
        # Raw per-variable bound noise stayed inside the declared tolerance …
        @test dc.maximum_active_recourse_lower_bound_violation_pu <= ACTIVE_RECOURSE_LB_TOL_PU
        # … and projection is ELEMENTWISE: public pu == Σ max(raw, 0) (not max of the sum).
        @test isapprox(dc.active_deficit_pu, sum(x -> max(Float64(x), 0.0), solc.active_deficit_pu); rtol = 1e-12)
        @test isapprox(dc.active_surplus_pu, sum(x -> max(Float64(x), 0.0), solc.active_surplus_pu); rtol = 1e-12)
        # Raw diagnostics keep the un-projected sums (used for objective reproduction).
        @test isapprox(dc.raw_active_deficit_pu, sum(Float64, solc.active_deficit_pu); rtol = 1e-12)
        @test isapprox(dc.raw_active_surplus_pu, sum(Float64, solc.active_surplus_pu); rtol = 1e-12)
        # Public accounting identities.
        @test isapprox(dc.total_active_recourse_energy_mwh,
                       dc.active_deficit_energy_mwh + dc.active_surplus_energy_mwh; rtol = 1e-12)
        @test isapprox(dc.active_recourse_cost,
                       st300.active_recourse_cost_per_mwh * dc.total_active_recourse_energy_mwh; rtol = 1e-12)
        @test isapprox(dc.active_deficit_energy_mwh, st300.baseMVA * st300.dt * dc.active_deficit_pu; rtol = 1e-12)
        @test isapprox(dc.active_surplus_energy_mwh, st300.baseMVA * st300.dt * dc.active_surplus_pu; rtol = 1e-12)
        @test isapprox(dc.physical_operating_cost,
                       dc.generator_cost + dc.battery_throughput_cost + dc.active_recourse_cost; rtol = 1e-12)
        # Projection correction is EXACTLY the projected-minus-raw recourse cost.
        @test isapprox(dc.active_recourse_projection_correction,
                       dc.active_recourse_cost - dc.raw_active_recourse_cost; rtol = 1e-10, atol = 1e-12)
        @test dc.active_recourse_projection_correction >= -1e-12   # projection only removes negatives
        # RAW diagnostics reproduce the solver objective (which uses raw primal).
        @test isapprox(dc.raw_total_check, dc.total_solver_objective; rtol = 1e-6)
        @test dc.solver_objective_recompute_residual < 1e-6
        # Maximal discharge → the SURPLUS direction provides recourse.
        set_tsddr_targets!(st300, lo)
        rd = solve_tsddr(st300)
        @test DecisionRulesExa.solve_succeeded(rd)          # recourse ⇒ feasible
        dd = decompose_costs(st300, rd)
        @test dd.active_surplus_pu > 1e-3                   # projected surplus engaged
        @test dd.active_deficit_pu < dd.active_surplus_pu   # discharge target ⇒ surplus dominates
        for x in (dd.active_deficit_pu, dd.active_surplus_pu, dd.active_deficit_energy_mwh,
                  dd.active_surplus_energy_mwh, dd.active_recourse_cost)
            @test isfinite(x) && x >= 0.0
        end
        @test dd.maximum_active_recourse_lower_bound_violation_pu <= ACTIVE_RECOURSE_LB_TOL_PU
    end

    # ── Reporting correctness: raw vs projected, elementwise, fail-loud ───────
    # The reporting layer must never export a negative recourse quantity: it
    # validates the raw per-variable bound noise, fails loudly beyond tolerance,
    # and projects the tolerated noise ELEMENTWISE (never on the aggregate, where
    # a negative bus could cancel a positive one).
    @testset "reporting: elementwise projection, fail-loud, raw/projected split" begin
        case, process = mild_fixture()
        de = build_battery_tsddr_de(case, process; reporting_horizon = 2, lookahead = 0,
                                    mode = :soft, rho1 = 1.0, rho2 = 10.0, stage_hours = 1.0)
        e0 = policy_initial_state(case; float_type = Float64)
        w = materialize_scenario(process, [1, 1]; horizon = 2)
        set_tsddr_initial_soc!(de, e0); set_tsddr_uncertainty!(de, w)
        set_tsddr_targets!(de, Float64[e for _ in 1:2 for e in e0])
        r = solve_tsddr(de)
        @test DecisionRulesExa.solve_succeeded(r)

        # (a) ELEMENTWISE projection ≠ aggregate projection. Inject a deficit bus at
        # −0.4 and another at +1.0 (both within the SAME stage). Aggregate = +0.6 → a
        # naive max(Σ,0) keeps 0.6; the correct elementwise Σ max(·,0) keeps 1.0.
        sol = tsddr_solution(de, r)
        sol.active_deficit_pu .= 0.0; sol.active_surplus_pu .= 0.0
        sol.active_deficit_pu[1, 1] = -0.4
        sol.active_deficit_pu[2, 1] =  1.0
        d = decompose_costs(de, r; sol = sol, lb_tol = 1.0)   # permit the injected −0.4
        @test isapprox(d.active_deficit_pu, 1.0; rtol = 1e-12)        # Σ max(·,0), NOT 0.6
        @test isapprox(d.raw_active_deficit_pu, 0.6; rtol = 1e-12)    # raw keeps the sum
        @test d.active_deficit_pu >= 0.0 && d.active_surplus_pu >= 0.0
        @test d.active_deficit_energy_mwh >= 0.0 && d.active_surplus_energy_mwh >= 0.0
        @test d.active_recourse_cost >= 0.0
        # projection correction = projected − raw cost (removes the −0.4 bus).
        @test isapprox(d.active_recourse_projection_correction,
                       d.active_recourse_cost - d.raw_active_recourse_cost; rtol = 1e-12)
        @test d.active_recourse_projection_correction > 0.0          # a negative was removed

        # (b) FAIL LOUD: a value further below zero than the declared tolerance is a
        # defect, not noise — decompose_costs must raise rather than report it.
        solbad = tsddr_solution(de, r)
        solbad.active_surplus_pu[1, 1] = -10 * ACTIVE_RECOURSE_LB_TOL_PU
        @test_throws ErrorException decompose_costs(de, r; sol = solbad)
        # A target-slack value below tolerance also fails loudly.
        solbad2 = tsddr_solution(de, r)
        solbad2.target_slack_pos[1, 1] = -10 * ACTIVE_RECOURSE_LB_TOL_PU
        @test_throws ErrorException decompose_costs(de, r; sol = solbad2)

        # (c) On the genuine solve, everything public is ≥ 0, raw reproduces the
        # objective, and the declared per-variable tolerance holds.
        dg = decompose_costs(de, r)
        for x in (dg.active_deficit_pu, dg.active_surplus_pu, dg.active_deficit_energy_mwh,
                  dg.active_surplus_energy_mwh, dg.total_active_recourse_energy_mwh,
                  dg.active_recourse_cost, dg.target_penalty, dg.target_violation,
                  dg.max_active_deficit_pu, dg.max_active_surplus_pu)
            @test isfinite(x) && x >= 0.0
        end
        @test dg.maximum_active_recourse_lower_bound_violation_pu <= ACTIVE_RECOURSE_LB_TOL_PU
        @test isapprox(dg.raw_total_check, r.objective; rtol = 1e-5)
        @test dg.solver_objective_recompute_residual < 1e-4
        @test isapprox(dg.active_recourse_cost,
                       de.active_recourse_cost_per_mwh * dg.total_active_recourse_energy_mwh; rtol = 1e-12)
    end

    # ── Targetless physical diagnostic (no targets at all) ───────────────────
    @testset "targetless diagnostic: no target constraints, slacks, or penalty" begin
        case, process = mild_fixture()
        T = 2
        tl   = build_targetless_diagnostic_de(case, process; reporting_horizon = T,
                                              lookahead = 0, stage_hours = 1.0)
        soft = build_battery_tsddr_de(case, process; reporting_horizon = T, lookahead = 0,
                                      mode = :soft, rho1 = 0.0, rho2 = 0.0, stage_hours = 1.0)
        @test is_targetless(tl)
        @test !is_targetless(soft)

        # (a) NO target constraints and NO target parameter.
        @test isempty(tl.target_con_range)
        @test tl.p_target === nothing
        # Soft carries T*nBat target rows AND 2*T*nBat slack variables; the
        # targetless model has neither, so it is strictly smaller by exactly that.
        @test length(soft.target_con_range) == T * tl.nBat
        @test soft.model.meta.ncon - tl.model.meta.ncon == T * tl.nBat
        @test soft.model.meta.nvar - tl.model.meta.nvar == 2 * T * tl.nBat

        # (b) Production API refuses to build it, and target ops reject it.
        @test_throws ErrorException build_battery_tsddr_de(case, process;
                                        reporting_horizon = T, mode = :none)
        @test_throws ErrorException set_tsddr_targets!(tl, zeros(T * tl.nBat))
        @test_throws ErrorException train_battery_tsddr(nothing, tl, process,
                                        scenario_index_matrix(process, T, 1; seed = 1))

        e0 = policy_initial_state(case; float_type = Float64)
        w = materialize_scenario(process, [1, 1]; horizon = T)
        set_tsddr_initial_soc!(tl, e0); set_tsddr_uncertainty!(tl, w)
        r = solve_tsddr(tl)
        @test DecisionRulesExa.solve_succeeded(r)
        @test_throws ErrorException target_multipliers(tl, r)

        # (c) PROJECTED physical = generator + throughput + active-recourse cost, and
        # every public recourse quantity is finite and ≥ 0.
        d = decompose_costs(tl, r)
        @test d.target_penalty == 0.0 && d.target_violation == 0.0
        for x in (d.active_deficit_pu, d.active_surplus_pu, d.active_deficit_energy_mwh,
                  d.active_surplus_energy_mwh, d.total_active_recourse_energy_mwh,
                  d.active_recourse_cost)
            @test isfinite(x) && x >= 0.0
        end
        @test isapprox(d.physical_operating_cost,
                       d.generator_cost + d.battery_throughput_cost + d.active_recourse_cost;
                       rtol = 1e-12)
        # Public reported total: physical + projected penalty (exact, projected side).
        @test isapprox(d.total_check, d.physical_operating_cost + d.target_penalty; rtol = 1e-12)
        # RAW diagnostics reproduce the solver objective; projection correction explained.
        @test isapprox(d.raw_total_check, r.objective; rtol = 1e-6)
        @test d.solver_objective_recompute_residual < 1e-6
        @test isapprox(d.active_recourse_projection_correction,
                       d.active_recourse_cost - d.raw_active_recourse_cost; rtol = 1e-10, atol = 1e-12)
        @test d.maximum_active_recourse_lower_bound_violation_pu <= ACTIVE_RECOURSE_LB_TOL_PU

        # (d) Physics residuals within the existing tolerances.
        sol = tsddr_solution(tl, r)
        @test tsddr_max_primal_residual(tl, r) < 1e-5
        @test maximum(abs, tsddr_balance_residuals(tl, sol)) < 1e-6

        # (e) Rebuilding with different (unused) target data cannot change it —
        # there is no target data to change, so an identical rebuild reproduces
        # the same objective bit-for-bit.
        tl2 = build_targetless_diagnostic_de(case, process; reporting_horizon = T,
                                             lookahead = 0, stage_hours = 1.0)
        set_tsddr_initial_soc!(tl2, e0); set_tsddr_uncertainty!(tl2, w)
        r2 = solve_tsddr(tl2)
        @test isapprox(r2.objective, r.objective; rtol = 1e-10)

        # (f) recourse-enabled and recourse-forbidden share every physical
        # equation/datum: they differ ONLY in the two-sided nodal-slack upper
        # bounds (deficit d⁺ and surplus d⁻), with the same var/con counts.
        tlf = build_targetless_diagnostic_de(case, process; reporting_horizon = T,
                                             lookahead = 0, stage_hours = 1.0,
                                             allow_active_recourse = false)
        @test tlf.model.meta.nvar == tl.model.meta.nvar
        @test tlf.model.meta.ncon == tl.model.meta.ncon
        @test tlf.active_recourse_cost_per_mwh == tl.active_recourse_cost_per_mwh
        off = variable_offsets(tl)
        uv, uvf = Array(tl.model.meta.uvar), Array(tlf.model.meta.uvar)
        slack_rng = vcat((off.active_deficit + 1):(off.active_deficit + T * tl.nBus),
                         (off.active_surplus + 1):(off.active_surplus + T * tl.nBus))
        @test all(uvf[slack_rng] .== 0.0)                      # forbidden: both fixed to 0
        @test all(uv[slack_rng] .== Inf)                       # enabled: both unbounded
        # every OTHER bound is identical
        others = setdiff(1:length(uv), slack_rng)
        @test uv[others] == uvf[others]
    end

    # ── REPAIR 4: strict / soft target modes ─────────────────────────────────
    @testset "strict equality, soft split slacks, cost separation" begin
        case, process = mild_fixture()
        T = 3
        w = materialize_scenario(process, [2, 1, 3]; horizon = T)
        e0 = policy_initial_state(case; float_type = Float64)
        pol = battery_reachable_policy(case, process; dt = 1.0, layers = [8], float_type = Float64)

        # Strict: hard equality, no slack, no penalty. With the two-sided active
        # recourse providing complete recourse, the strict solve ALWAYS succeeds —
        # this is asserted unconditionally (a skipped/logged strict failure is a
        # test failure).
        de_s = build_battery_tsddr_de(case, process; reporting_horizon = T, lookahead = 0,
                                      mode = :strict, stage_hours = 1.0)
        xhat = policy_targets(pol, process, w, e0, T)
        set_tsddr_initial_soc!(de_s, e0); set_tsddr_uncertainty!(de_s, w); set_tsddr_targets!(de_s, xhat)
        rs = solve_tsddr(de_s)
        @test DecisionRulesExa.solve_succeeded(rs)              # unconditional
        sol_s = tsddr_solution(de_s, rs)
        @test maximum(abs, vec(sol_s.soc[:, 2:T+1]) .- xhat) < 1e-5   # strict target residual
        @test tsddr_max_primal_residual(de_s, rs) < 1e-6             # primal residual
        dS = decompose_costs(de_s, rs)
        @test dS.target_penalty == 0.0 && dS.target_violation == 0.0 # no penalty/slack
        @test all(iszero, sol_s.target_slack_pos) && all(iszero, sol_s.target_slack_neg)
        # Public: physical = gen + throughput + projected recourse; total_check is the
        # projected reported total; every public recourse value ≥ 0.
        @test isapprox(dS.physical_operating_cost,                    # strict cost decomposition
                       dS.generator_cost + dS.battery_throughput_cost + dS.active_recourse_cost;
                       rtol = 1e-12)
        @test isapprox(dS.total_check, dS.physical_operating_cost + dS.target_penalty; rtol = 1e-12)
        @test dS.active_recourse_cost >= 0.0 && dS.active_deficit_energy_mwh >= 0.0 &&
              dS.active_surplus_energy_mwh >= 0.0
        # RAW diagnostics reproduce the solver objective (strict ⇒ no penalty).
        @test isapprox(dS.raw_total_check, rs.objective; rtol = 1e-6)
        @test dS.solver_objective_recompute_residual < 1e-6
        @test dS.maximum_active_recourse_lower_bound_violation_pu <= ACTIVE_RECOURSE_LB_TOL_PU
        @test maximum(abs, tsddr_balance_residuals(de_s, sol_s)) < 1e-6  # battery-balance residual

        # Soft: split slacks, L1+L2 training-only penalty excluded from physical.
        de_f = build_battery_tsddr_de(case, process; reporting_horizon = T, lookahead = 0,
                                      mode = :soft, rho1 = 1.0, rho2 = 10.0, stage_hours = 1.0)
        set_tsddr_initial_soc!(de_f, e0); set_tsddr_uncertainty!(de_f, w)
        set_tsddr_targets!(de_f, policy_targets(pol, process, w, e0, T))
        rf = solve_tsddr(de_f)
        @test DecisionRulesExa.solve_succeeded(rf)
        sol_f = tsddr_solution(de_f, rf)
        dF = decompose_costs(de_f, rf)
        @test all(sol_f.target_slack_pos .>= -ACTIVE_RECOURSE_LB_TOL_PU)   # raw slacks within tol
        @test all(sol_f.target_slack_neg .>= -ACTIVE_RECOURSE_LB_TOL_PU)
        # Public target penalty/violation are projected (≥ 0), and the reported total
        # is physical + projected penalty.
        @test dF.target_penalty >= 0.0 && dF.target_violation >= 0.0
        @test isapprox(dF.total_check, dF.physical_operating_cost + dF.target_penalty; rtol = 1e-12)
        # Physical cost = generator + throughput + active recourse (NO target penalty).
        @test isapprox(dF.physical_operating_cost,
                       dF.generator_cost + dF.battery_throughput_cost + dF.active_recourse_cost;
                       rtol = 1e-12)
        @test dF.active_recourse_cost >= 0.0
        # RAW diagnostics reproduce the solver objective WITH the raw penalty term.
        @test isapprox(dF.raw_total_check, rf.objective; rtol = 1e-5)
        @test dF.solver_objective_recompute_residual < 1e-4
        @test isapprox(dF.reporting_physical_cost + dF.lookahead_physical_cost,
                       dF.physical_operating_cost; rtol = 1e-10)
        @test maximum(min.(sol_f.p_ch, sol_f.p_dis)) < 1e-5   # no simultaneous ch/dis
    end

    # ── Strict absolute recourse: EVERY reachable target solves ──────────────
    # The invariant: with the two-sided active recourse, the strict operational
    # model has complete recourse — for every supported PGLib case and every
    # dynamically reachable target (INCLUDING the exact reachable endpoints and
    # aggressive charging targets that are NOT network-deliverable), the strict
    # stage NLP solves and reproduces the target exactly. Network congestion may
    # raise the deficit/cost; it can never make the strict solve fail.
    @testset "strict absolute recourse across cases and target classes" begin
        function reach1(case, e0, dt)
            lo = similar(e0); up = similar(e0)
            for (k, b) in enumerate(case.batteries)
                a = 1 - b.sigma * dt
                lo[k] = max(b.e_min, a*e0[k] - (dt/b.eta_dis)*b.p_discharge_max)
                up[k] = min(b.e_max, a*e0[k] + b.eta_ch*dt*b.p_charge_max)
            end
            lo, up
        end
        for (cn, nb) in (("case14_ieee", 3), ("case118_ieee", 10), ("case300_ieee", 20))
            case = make_battery_case(cn; number_of_batteries = nb, seed = 20260722)
            proc = make_load_process(case; preset = DEFAULT_DEMAND_PRESET, nregion = 3, period = 4)
            e0 = Float64.(policy_initial_state(case; float_type = Float64))
            lo, up = reach1(case, e0, 1.0)
            w = materialize_scenario(proc, [1]; horizon = 1)
            st = build_battery_stage_problem(case, proc; mode = :strict, stage_hours = 1.0)
            set_tsddr_initial_soc!(st, e0); set_tsddr_uncertainty!(st, w)
            classes = Dict(
                "hold"     => [(1 - case.batteries[k].sigma)*e0[k] for k in 1:nb],
                "midpoint" => (lo .+ up) ./ 2,
                "lower"    => copy(lo),
                "upper"    => copy(up),
                "interior" => lo .+ 0.5 .* (up .- lo),
                "charge90" => lo .+ 0.9 .* (up .- lo),
            )
            for (nm, tg) in classes
                set_tsddr_targets!(st, tg)
                r = solve_tsddr(st)
                @test DecisionRulesExa.solve_succeeded(r)             # ALWAYS feasible
                sol = tsddr_solution(st, r)
                @test maximum(abs, vec(sol.soc[:, 2]) .- tg) < 1e-5   # target residual ≤ 1e-5
                @test tsddr_max_primal_residual(st, r) < 1e-5         # primal residual ≤ 1e-5
                @test maximum(abs, tsddr_balance_residuals(st, sol)) < 1e-6
                d = decompose_costs(st, r)
                @test d.target_penalty == 0.0                        # strict: no penalty
                # Every PUBLIC recourse quantity is finite and STRICTLY nonnegative —
                # the reporting layer projects bound noise elementwise, so a negative
                # public value is impossible (and would fail here, not be tolerated).
                for x in (d.active_deficit_pu, d.active_surplus_pu, d.active_deficit_energy_mwh,
                          d.active_surplus_energy_mwh, d.total_active_recourse_energy_mwh,
                          d.max_active_deficit_pu, d.max_active_surplus_pu, d.active_recourse_cost)
                    @test isfinite(x) && x >= 0.0
                end
                # Raw per-variable noise stayed inside the declared tolerance, and the
                # raw diagnostics reproduce the solver objective.
                @test d.maximum_active_recourse_lower_bound_violation_pu <= ACTIVE_RECOURSE_LB_TOL_PU
                @test isapprox(d.raw_total_check, d.total_solver_objective; rtol = 1e-6)
                @test isapprox(d.total_active_recourse_energy_mwh,
                               d.active_deficit_energy_mwh + d.active_surplus_energy_mwh; rtol = 1e-12)
                @test isfinite(d.physical_operating_cost)
            end
        end
    end

    @testset "target multiplier finite-difference (sign and magnitude)" begin
        case, process = mild_fixture()
        T = 2
        de = build_battery_tsddr_de(case, process; reporting_horizon = T, lookahead = 0,
                                    mode = :soft, rho1 = 0.0, rho2 = 10.0, stage_hours = 1.0)
        w = materialize_scenario(process, [1, 2]; horizon = T)
        e0 = policy_initial_state(case; float_type = Float64)
        pol = battery_reachable_policy(case, process; dt = 1.0, layers = [8], float_type = Float64)
        xhat = policy_targets(pol, process, w, e0, T)
        set_tsddr_initial_soc!(de, e0); set_tsddr_uncertainty!(de, w); set_tsddr_targets!(de, xhat)
        r0 = solve_tsddr(de)
        @test DecisionRulesExa.solve_succeeded(r0)
        λ = target_multipliers(de, r0)
        i = 1; ε = 1e-4
        xp = copy(xhat); xp[i] += ε; set_tsddr_targets!(de, xp); rp = solve_tsddr(de)
        xm = copy(xhat); xm[i] -= ε; set_tsddr_targets!(de, xm); rm = solve_tsddr(de)
        @test DecisionRulesExa.solve_succeeded(rp) && DecisionRulesExa.solve_succeeded(rm)
        fd = (rp.objective - rm.objective) / (2ε)
        @test sign(fd) == sign(λ[i]) || abs(λ[i]) < 1e-6
        @test isapprox(fd, λ[i]; rtol = 5e-2, atol = 1e-2)
    end

    @testset "reporting/lookahead horizons recorded" begin
        case, process = fixture()
        de = build_battery_tsddr_de(case, process; reporting_horizon = 2, lookahead = 2,
                                    mode = :soft, stage_hours = 1.0)
        @test de.horizon == 4 && de.reporting_horizon == 2 && de.lookahead == 2
    end

    # ── REPAIR 5: no unapproved case modification ────────────────────────────
    @testset "no generator/network modification" begin
        case, process = fixture()
        nd = case.network
        de = build_battery_tsddr_de(case, process; reporting_horizon = 2, lookahead = 0,
                                    mode = :soft, stage_hours = 1.0)
        # gen_capacity_scale must not exist as a keyword any more.
        @test_throws MethodError build_battery_tsddr_de(case, process; reporting_horizon = 2,
                                                        gen_capacity_scale = 2.0)
        # Generator bounds in the model equal the untouched PGLib per-unit values.
        lv = Array(de.model.meta.lvar); uv = Array(de.model.meta.uvar)
        T = de.horizon; nB = de.nBus; nG = de.nGen
        pg_off = 2*T*nB
        for t in 1:T, g in 1:nG
            idx = pg_off + (t-1)*nG + g
            @test lv[idx] == nd.gens[g].pmin
            @test uv[idx] == nd.gens[g].pmax
        end
    end

    # ── Checkpointing ────────────────────────────────────────────────────────
    @testset "checkpoint exact reload" begin
        case, process = fixture()
        de = build_battery_tsddr_de(case, process; reporting_horizon = 2, lookahead = 0,
                                    mode = :soft, stage_hours = 1.0)
        pol = battery_reachable_policy(case, process; dt = 1.0, layers = [8],
                                       combiner_layers = [8], float_type = Float32)
        nw = n_uncertainty(process); nB = nbattery(case)
        inp = vcat(Float32.(rand(nw)), Float32.(fill(0.4, nB)))
        Flux.reset!(pol); y_before = copy(pol(inp))
        dir = mktempdir(); ckpt = joinpath(dir, "ckpt.jls")
        save_checkpoint(ckpt, pol, de; case = case, process = process)
        pol2, meta = load_checkpoint(ckpt, case, process)
        Flux.reset!(pol2); y_after = copy(pol2(inp))
        @test y_after == y_before
        @test meta["architecture"]["target_mode"] == "soft"
        @test meta["architecture"]["activation"] == string(stretchedsigmoid)
        @test meta["architecture"]["safe_upper_margin"] == 1e-3
        @test meta["active_recourse_cost_per_mwh"] == DEFAULT_ACTIVE_RECOURSE_COST_PER_MWH
        @test meta["hashes"]["load_process_hash"] == process_hash(process)
    end

    # ── CPU/GPU structural parity ────────────────────────────────────────────
    @testset "CPU/GPU model structural parity" begin
        case, process = fixture()
        cpu = build_battery_tsddr_de(case, process; reporting_horizon = 2, lookahead = 0,
                                     mode = :soft, stage_hours = 1.0, backend = nothing)
        if CUDA.functional()
            gpu = build_battery_tsddr_de(case, process; reporting_horizon = 2, lookahead = 0,
                                         mode = :soft, stage_hours = 1.0, backend = CUDABackend())
            @test cpu.model.meta.nvar == gpu.model.meta.nvar
            @test cpu.model.meta.ncon == gpu.model.meta.ncon
            @test cpu.target_con_range == gpu.target_con_range
            @info "GPU structural parity checked" CUDA.name(CUDA.device())
        else
            @info "CUDA not functional; GPU structural-parity check SKIPPED (not run)"
            @test cpu.model.meta.ncon > 0
        end
    end
end
