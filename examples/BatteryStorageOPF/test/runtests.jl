# Consolidated regression suite for the ExaModels/GPU battery engine.
#
# One file, grouped by the property being protected. The manual ACP formulation
# is checked against the physics (residuals recomputed from the reported
# solution, independently of the model that produced it) and against the case
# contract; its agreement with the actual PowerModels ACP model is checked by a
# separate cross-engine gate outside this package, because only a gate that
# depends on both packages at once can load both engines.
#
#   julia --project=. test/runtests.jl
#   DR_BAT_DEVICE=gpu julia --project=. test/runtests.jl   # additionally on a GPU

using Test
using Flux
using Zygote
using Random
using Statistics
using LinearAlgebra
using MadNLP
using JLD2

const EXAMPLE = dirname(@__DIR__)
include(joinpath(EXAMPLE, "train_battery_exa_strict.jl"))   # pulls in the whole engine

const CASE_DIR = get(ENV, "DR_BAT_CASE_DIR",
                     joinpath(EXAMPLE, "case", "pglib_opf_case14_ieee"))

# Cases are CONSTRUCTED by the JuMP package's builder and MIRRORED here; nothing
# under `case/` is committed, because a committed artifact is a second source of
# truth that can drift from the builder that defines it. This engine cannot build
# one — it has no PGLib dependency by design — so it says exactly what to run.
isfile(joinpath(CASE_DIR, "case_manifest.json")) || error("""
no case at $CASE_DIR. Build and mirror it from the JuMP package first:

  cd /path/to/DecisionRules.jl/examples/BatteryStorageOPF
  DR_BAT_MIRROR=$(dirname(EXAMPLE))/$(basename(EXAMPLE)) julia --project=. build_battery_case.jl
""")
const DEVICE = get(ENV, "DR_BAT_DEVICE", "cpu")

@testset "battery storage OPF (Exa engine)" begin

    case = read_battery_case(CASE_DIR)
    net = exa_network(case)
    Δt = stage_hours(case)
    bats = sort(collect(case.batteries); by = b -> b.index)
    nBat = length(bats)
    backend, to_device, solver_kwargs = configure_device(DEVICE)

    # ── The network parse ────────────────────────────────────────────────────
    @testset "network parse" begin
        @test nbus(net) == length(case.network["bus"])
        @test ngen(net) == count(g -> Int(get(g[2], "gen_status", 1)) != 0, case.network["gen"])
        @test nbranch(net) == count(b -> Int(get(b[2], "br_status", 1)) != 0, case.network["branch"])
        @test !isempty(net.ref_bus_positions)
        @test sum(net.nominal_pd) ≈ sum(Float64(l["pd"]) for (_, l) in case.network["load"])
        @test sum(net.nominal_qd) ≈ sum(Float64(l["qd"]) for (_, l) in case.network["load"])
        # Shunts live in their own table and must reach the bus they belong to.
        for (_, sh) in get(case.network, "shunt", Dict{String,Any}())
            p = net.bus_pos[Int(sh["shunt_bus"])]
            @test net.buses[p].bs != 0.0 || Float64(get(sh, "bs", 0.0)) == 0.0
        end

        # Component identifiers are IDENTITIES, not positions: a case whose bus
        # ids are relabelled to a sparse set must parse to the same physics.
        relabelled = deepcopy(case.network)
        remap = Dict(Int(b["index"]) => 100 * Int(b["index"]) + 7 for (_, b) in case.network["bus"])
        newbus = Dict{String,Any}()
        for (_, b) in relabelled["bus"]
            b["index"] = remap[Int(b["bus_i"])]
            b["bus_i"] = b["index"]
            newbus[string(b["index"])] = b
        end
        relabelled["bus"] = newbus
        for (_, g) in relabelled["gen"];    g["gen_bus"] = remap[Int(g["gen_bus"])];   end
        for (_, l) in relabelled["load"];   l["load_bus"] = remap[Int(l["load_bus"])]; end
        for (_, s) in relabelled["shunt"];  s["shunt_bus"] = remap[Int(s["shunt_bus"])]; end
        for (_, br) in relabelled["branch"]
            br["f_bus"] = remap[Int(br["f_bus"])]
            br["t_bus"] = remap[Int(br["t_bus"])]
        end
        rcase = BatteryCase(case.dir, case.name, relabelled,
                            [BatterySpec(b.index, remap[b.bus], b.energy_min, b.energy_max,
                                         b.energy_initial, b.charge_max, b.discharge_max,
                                         b.charge_efficiency, b.discharge_efficiency,
                                         b.self_discharge, b.throughput_cost) for b in bats],
                            case.recourse, case.demand, case.manifest)
        rnet = exa_network(rcase)
        @test nbus(rnet) == nbus(net) && ngen(rnet) == ngen(net) && nbranch(rnet) == nbranch(net)
        @test rnet.nominal_pd ≈ net.nominal_pd
        @test [b.f_pos for b in rnet.branches] == [b.f_pos for b in net.branches]
        @test rnet.ref_bus_positions == net.ref_bus_positions
    end

    # ── Model structure ──────────────────────────────────────────────────────
    @testset "model structure" begin
        T = 3
        prob = build_battery_exa(case, T; backend = backend)
        nB, nG, nBR = prob.nBus, prob.nGen, prob.nBranch
        @test prob.model.meta.nvar == 2T * nB + 2T * nG + 4T * nBR + 2T * nB + 2T * nBat
        @test length(prob.transition_range) == T * nBat
        @test last(prob.transition_range) == prob.model.meta.ncon   # transitions are LAST

        lv = Array(prob.model.meta.lvar); uv = Array(prob.model.meta.uvar)
        off = 2T * nB + 2T * nG + 4T * nBR
        rng_d = (off + 1):(off + T * nB)
        rng_s = (off + T * nB + 1):(off + 2T * nB)
        # Two-sided recourse: nonnegative, and uncapped in both directions.
        @test all(==(0.0), lv[rng_d]) && all(isinf, uv[rng_d])
        @test all(==(0.0), lv[rng_s]) && all(isinf, uv[rng_s])
        # Charge/discharge respect the case's power ratings.
        rng_ch = (off + 2T * nB + 1):(off + 2T * nB + T * nBat)
        @test all(==(0.0), lv[rng_ch])
        @test maximum(uv[rng_ch]) ≈ maximum(b.charge_max for b in bats)
        @test eltype(prob.model.meta.x0) === Float64
    end

    # ── The strict solve and its multipliers ─────────────────────────────────
    @testset "strict solve" begin
        T = 3
        prob = build_battery_exa(case, T; backend = backend)
        atoms = [3, 1, 2]
        pd, qd = realized_demand(case, net, collect(1:T), atoms)
        set_demand!(prob, pd, qd)
        e0 = initial_energy(case)

        function targets(frac)
            xs = Float64[]; prev = copy(e0)
            for _ in 1:T
                nxt = similar(prev)
                for (k, b) in enumerate(bats)
                    lo, hi = reachable_interval(b, prev[k], Δt)
                    nxt[k] = lo + frac * (hi - lo)
                end
                append!(xs, nxt); prev = nxt
            end
            return xs
        end

        x = targets(0.45)
        set_energy_path!(prob, e0, x)
        # The strict-mode invariant: the parameter's first block IS the initial
        # state, because the model has no initial-condition row.
        @test prob.energy_values[1:nBat] ≈ e0
        res = solve!(prob; solver_kwargs...)
        @test solve_succeeded(res)
        sol = battery_solution(prob, res)
        cost = stage_costs(prob, sol)
        @test sum(cost.total) ≈ res.objective atol = 1e-6
        @test maximum(sol.deficit) < 1e-6 && maximum(sol.surplus) < 1e-6
        @test maximum(min.(sol.p_ch, sol.p_dis)) < 1e-6        # no simultaneous operation

        # The reported trajectory IS the target, exactly: energy is a parameter.
        @test vec(sol.energy[:, 2:end]) ≈ x atol = 1e-12

        # Residuals recomputed independently of the model that produced them.
        for t in 1:T
            s = (vm = Dict(net.buses[i].id => sol.vm[i, t] for i in eachindex(net.buses)),
                 va = Dict(net.buses[i].id => sol.va[i, t] for i in eachindex(net.buses)),
                 pg = Dict(net.gens[i].id => sol.pg[i, t] for i in eachindex(net.gens)),
                 qg = Dict(net.gens[i].id => sol.qg[i, t] for i in eachindex(net.gens)),
                 p_fr = Dict(net.branches[i].id => sol.p_fr[i, t] for i in eachindex(net.branches)),
                 q_fr = Dict(net.branches[i].id => sol.q_fr[i, t] for i in eachindex(net.branches)),
                 p_to = Dict(net.branches[i].id => sol.p_to[i, t] for i in eachindex(net.branches)),
                 q_to = Dict(net.branches[i].id => sol.q_to[i, t] for i in eachindex(net.branches)),
                 deficit = Dict(net.buses[i].id => sol.deficit[i, t] for i in eachindex(net.buses)),
                 surplus = Dict(net.buses[i].id => sol.surplus[i, t] for i in eachindex(net.buses)),
                 pd = Dict(net.buses[i].id => pd[t, i] for i in eachindex(net.buses)),
                 qd = Dict(net.buses[i].id => qd[t, i] for i in eachindex(net.buses)),
                 p_ch = Dict(bats[k].index => sol.p_ch[k, t] for k in 1:nBat),
                 p_dis = Dict(bats[k].index => sol.p_dis[k, t] for k in 1:nBat),
                 energy_in = Dict(bats[k].index => sol.energy[k, t] for k in 1:nBat),
                 energy_out = Dict(bats[k].index => sol.energy[k, t + 1] for k in 1:nBat))
            r = physical_residuals(case.network, bats, Δt, s)
            @test r.branch_flow < 1e-6
            @test r.active_balance < 1e-8
            @test r.reactive_balance < 1e-8
            @test r.transition < TRANSITION_RESIDUAL_TOL
            @test r.thermal < 1e-6
            @test r.voltage < 1e-6
        end

        # The multiplier is the derivative of the solved value in the target,
        # including the self-discharge factor that couples adjacent rows.
        λ = target_multipliers(prob, res)
        @test length(λ) == T * nBat
        h = 1e-5
        for i in (1, nBat + 2, (T - 1) * nBat + 3)
            xp = copy(x); xp[i] += h
            xm = copy(x); xm[i] -= h
            set_energy_path!(prob, e0, xp)
            vp = solve!(prob; solver_kwargs...).objective
            set_energy_path!(prob, e0, xm)
            vm_ = solve!(prob; solver_kwargs...).objective
            @test (vp - vm_) / (2h) ≈ λ[i] rtol = 1e-4
        end
    end

    # ── A generator that exists in some stages and not others ────────────────
    # The case may declare a per-stage availability schedule for any generator
    # (`STAGE_AVAILABILITY_KEY`). The JuMP engine applies it to the parsed network
    # before PowerModels instantiates each stage; this engine builds every stage
    # at once, so it applies it to each stage's generator BOUNDS instead. The
    # failure this guards against is silence: a schedule that is written into a
    # case, hashed and mirrored here, and then ignored by the model builder, would
    # give this engine a generator that is always available while the JuMP engine,
    # the case record and the digest all say otherwise — two engines solving two
    # different problems, which is the one thing this suite exists to prevent.
    @testset "per-stage generator availability" begin
        # ── Additivity: the frozen case declares no schedule ──────────────────
        # A case built before this convention existed must build EXACTLY as it
        # did then: one `repeat` of the case's own limits per stage, compared
        # bit-for-bit rather than approximately.
        @test all(isempty(g.availability) for g in net.gens)
        T = 2
        plain = build_battery_exa(case, T; backend = backend)
        off_p = 2T * nbus(net)
        off_q = off_p + T * ngen(net)
        plv = Array(plain.model.meta.lvar); puv = Array(plain.model.meta.uvar)
        # An EXACTLY degenerate box is not a box in this model: `_pin_exact_boxes!`
        # opens it and pins the variable with the equality row `x − v = 0`, which
        # is the same feasible set without the zero-width bound an interior-point
        # method mishandles. The expectation therefore carries that transform.
        opened(lo, hi) = lo == hi ? (-Inf, Inf) : (lo, hi)
        want_p = [opened(g.pmin, g.pmax) for g in net.gens]
        want_q = [opened(isfinite(g.qmin) ? g.qmin : -1e4,
                         isfinite(g.qmax) ? g.qmax : 1e4) for g in net.gens]
        @test plv[(off_p + 1):(off_p + T * ngen(net))] == repeat(first.(want_p), T)
        @test puv[(off_p + 1):(off_p + T * ngen(net))] == repeat(last.(want_p), T)
        @test plv[(off_q + 1):(off_q + T * ngen(net))] == repeat(first.(want_q), T)
        @test puv[(off_q + 1):(off_q + T * ngen(net))] == repeat(last.(want_q), T)
        # No zero-width box survives anywhere in the model, which is the property
        # the reformulation exists to guarantee.
        @test !any(i -> plv[i] == puv[i], eachindex(plv))

        # ── The fixture ───────────────────────────────────────────────────────
        # An in-memory case whose generator `gid` is FREE to run and carries the
        # schedule `av`. Nothing is written to disk: the schedule is network data,
        # and a test that had to run the JuMP builder could not run in this
        # environment at all.
        #
        # The unit is made free rather than picked for being dispatched, so that
        # the always-available twin further down is a decisive null control by
        # construction: a zero-cost unit runs unless something stops it, so a
        # stage-2 output of exactly zero can only be the schedule. Picking a unit
        # by its dispatch would make the control depend on the frozen case's
        # economics, which is not what is being tested here.
        refbuses = Set(net.buses[p].id for p in net.ref_bus_positions)
        cands = [i for (i, g) in enumerate(net.gens)
                 if !(net.buses[g.bus_pos].id in refbuses)]
        @test !isempty(cands)
        gpos = cands[argmax([net.gens[i].pmax for i in cands])]
        GID = net.gens[gpos].id

        function fixture(av; free::Bool = true)
            netj = deepcopy(case.network)
            g = netj["gen"][string(GID)]
            if free
                g["model"] = 2; g["ncost"] = 2; g["cost"] = [0.0, 0.0]
            end
            av === nothing ? delete!(g, STAGE_AVAILABILITY_KEY) :
                             (g[STAGE_AVAILABILITY_KEY] = av)
            return BatteryCase(case.dir, case.name, netj, case.batteries,
                               case.recourse, case.demand, case.manifest)
        end

        base = build_battery_exa(fixture(nothing), T; backend = backend)
        fnet = exa_network(fixture(nothing))
        gsel = fnet.gens[gpos]
        @test gsel.id == GID
        lv = Array(base.model.meta.lvar); uv = Array(base.model.meta.uvar)

        # ── What the parser refuses ───────────────────────────────────────────
        @test_throws ErrorException exa_network(fixture(Float64[]))
        @test_throws ErrorException exa_network(fixture(1.0))
        @test_throws ErrorException exa_network(fixture([1.0, 1.5]))
        @test_throws ErrorException exa_network(fixture([1.0, -0.5]))
        @test_throws ErrorException exa_network(fixture([1.0, NaN]))
        # Inconsistent bounds are checked where the new data is read.
        badb = fixture([1.0, 0.0])
        badb.network["gen"][string(GID)]["pmin"] = Float64(gsel.pmax) + 1.0
        @test_throws ErrorException exa_network(badb)

        # ── Scaling is per stage, and covers reactive as well as active ───────
        sched = fixture([1.0, 0.0])
        snet = exa_network(sched)
        @test snet.gens[gpos].availability == [1.0, 0.0]
        @test availability_at(snet.gens[gpos], 1) == 1.0
        @test availability_at(snet.gens[gpos], 2) == 0.0
        sprob = build_battery_exa(sched, T; backend = backend)
        # The model's VARIABLE shape does not depend on the schedule.
        @test sprob.model.meta.nvar == base.model.meta.nvar
        slv = Array(sprob.model.meta.lvar); suv = Array(sprob.model.meta.uvar)
        nG = ngen(net)
        p1 = off_p + gpos; p2 = off_p + nG + gpos
        q1 = off_q + gpos; q2 = off_q + nG + gpos
        @test slv[p1] == gsel.pmin && suv[p1] == gsel.pmax     # stage 1: untouched
        # Stage 2 is exactly out. The unit is taken out by an EQUALITY row now,
        # not by a zero-width box, so its box is OPEN and the pin is what holds
        # it at zero — the same feasible set, and the dispatch below proves it.
        @test slv[p2] == -Inf && suv[p2] == Inf
        @test slv[q2] == -Inf && suv[q2] == Inf
        # The scheduled build carries one pin per fully-unavailable p and q entry
        # more than the unscheduled one.
        @test sprob.model.meta.ncon == base.model.meta.ncon + 2
        @test slv[q1] == lv[q1] && suv[q1] == uv[q1]
        # Every other generator, in both stages, is exactly as it was.
        for i in 1:(T * nG)
            (i == gpos || i == nG + gpos) && continue
            @test slv[off_p + i] == lv[off_p + i] && suv[off_p + i] == uv[off_p + i]
            @test slv[off_q + i] == lv[off_q + i] && suv[off_q + i] == uv[off_q + i]
        end
        # A schedule of all ones is the null control for that zero: same case,
        # same unit, and bounds identical to the unscheduled build.
        ones_prob = build_battery_exa(fixture([1.0, 1.0]), T; backend = backend)
        @test Array(ones_prob.model.meta.lvar) == lv
        @test Array(ones_prob.model.meta.uvar) == uv

        # ── A stage the schedule does not cover is an error, never the last ───
        @test_throws ErrorException build_battery_exa(sched, 3; backend = backend)
        @test_throws ErrorException build_battery_exa(sched, 1; backend = backend,
                                                      stages = [3])

        # ── A WINDOW of the horizon reads that window's entry ─────────────────
        # A continuation problem is the case's stage 2 solved on its own, and it
        # must see stage 2's availability, not the first entry of the schedule.
        w2 = build_battery_exa(sched, 1; backend = backend, stages = [2])
        @test w2.stages == [2]
        @test Array(w2.model.meta.lvar)[2 * nbus(net) + gpos] == -Inf
        @test Array(w2.model.meta.uvar)[2 * nbus(net) + gpos] == Inf
        w1 = build_battery_exa(sched, 1; backend = backend, stages = [1])
        @test Array(w1.model.meta.uvar)[2 * nbus(net) + gpos] == gsel.pmax
        # And the window is enforced at the solve site rather than assumed: a
        # scheduled problem refuses an offset it was not built for, while an
        # unscheduled one accepts every offset exactly as it always did.
        @test assert_stage_window(sprob, [1, 2]) === nothing
        @test_throws ErrorException assert_stage_window(sprob, [2, 3])
        @test assert_stage_window(base, [2, 3]) === nothing
        @test !has_stage_schedule(base) && has_stage_schedule(sprob)

        # ── What it DISPATCHES ────────────────────────────────────────────────
        # One demand realization, one strict trajectory, imposed on the scheduled
        # problem and on its always-available twin. The schedule is the only
        # difference between the two.
        atoms = [1, 2]
        pd2, qd2 = realized_demand(sched, snet, collect(1:T), atoms)
        e0 = initial_energy(sched)
        xs = Float64[]; prev = copy(e0)
        for _ in 1:T
            nxt = similar(prev)
            for (k, b) in enumerate(bats)
                lo, hi = reachable_interval(b, prev[k], Δt)
                nxt[k] = lo + 0.5 * (hi - lo)
            end
            append!(xs, nxt); prev = nxt
        end
        set_demand!(sprob, pd2, qd2); set_energy_path!(sprob, e0, xs)
        sres = solve!(sprob; solver_kwargs...)
        @test solve_succeeded(sres)
        ssol = battery_solution(sprob, sres)
        @test abs(ssol.pg[gpos, 2]) <= 1e-9          # the schedule's own property
        @test abs(ssol.qg[gpos, 2]) <= 1e-9
        # The null control dispatches the same unit in the same stage without it.
        set_demand!(ones_prob, pd2, qd2); set_energy_path!(ones_prob, e0, xs)
        ores = solve!(ones_prob; solver_kwargs...)
        @test solve_succeeded(ores)
        osol = battery_solution(ones_prob, ores)
        @test abs(osol.pg[gpos, 2]) > 1e-6

        # ── The schedule is DATA, not a differentiable path ───────────────────
        # It reaches the model through variable bounds only, so the trajectory
        # derivative the trainer consumes is still exactly the derivative of the
        # solved value — checked here against central differences on the case
        # that carries the schedule.
        λ = target_multipliers(sprob, sres)
        @test length(λ) == T * nBat
        @test all(isfinite, λ)
        h = 1e-5
        for i in (1, nBat + 1)
            xp = copy(xs); xp[i] += h
            xm = copy(xs); xm[i] -= h
            set_energy_path!(sprob, e0, xp)
            vp = solve!(sprob; solver_kwargs...).objective
            set_energy_path!(sprob, e0, xm)
            vm_ = solve!(sprob; solver_kwargs...).objective
            @test (vp - vm_) / (2h) ≈ λ[i] rtol = 1e-4
        end
    end

    # ── The strict reachable policy ──────────────────────────────────────────
    @testset "reachable policy" begin
        Random.seed!(11)
        prob = build_battery_exa(case, 4; backend = backend)
        policy = to_device(battery_reachable_policy(case, [8];
                           n_observation = prob.nBus, n_context = N_CONTEXT,
                           head_layers = [12]))
        assert_device(policy, DEVICE)
        @test_throws ArgumentError battery_reachable_policy(case, [8];
                        n_observation = prob.nBus, activation = tanh)

        like = _policy_array(policy)
        e0 = initial_energy(case)
        e0d = _to_like(like, e0)
        atoms = [1, 3, 2, 2]
        features = rollout_features(case, prob.net, atoms; like = like)

        # The vectorized bounds agree with the shared case contract's scalar form.
        lo, hi = reachable_bounds(policy, e0d, e0d)
        for (k, b) in enumerate(bats)
            l, u = reachable_interval(b, e0[k], Δt)
            @test Float64(Array(lo)[k]) ≈ l atol = 1e-5
            @test Float64(Array(hi)[k]) ≈ u atol = 1e-5
        end

        # Every emitted target lies inside its own reachable interval.
        x = Float64.(vec(Array(rollout_targets(policy, features, e0d))))
        prev = copy(e0)
        for t in 1:4
            for (k, b) in enumerate(bats)
                l, u = reachable_interval(b, prev[k], Δt)
                @test l - 1e-5 <= x[(t - 1) * nBat + k] <= u + 1e-5
            end
            prev = x[((t - 1) * nBat + 1):(t * nBat)]
        end

        # The recurrent state really advances, and `reset!` really resets it.
        Flux.reset!(policy)
        s0 = deepcopy(policy.state)
        policy(vcat(features[:, 1], e0d))
        @test !isapprox(Float64.(vec(Array(policy.state[1][1]))),
                        Float64.(vec(Array(s0[1][1]))); atol = 1e-12)
        Flux.reset!(policy)
        @test Float64.(vec(Array(policy.state[1][1]))) ≈ Float64.(vec(Array(s0[1][1])))

        # A repeated rollout of the SAME scenario reproduces itself exactly,
        # which is only true if the boundary reset actually happens.
        x2 = Float64.(vec(Array(rollout_targets(policy, features, e0d))))
        @test x2 == x
        # A DIFFERENT scenario must produce a different trajectory — otherwise
        # the encoder is memoryless and the policy is not reading the demand.
        alt = rollout_features(case, prob.net, [3, 1, 1, 3]; like = like)
        @test !isapprox(Float64.(vec(Array(rollout_targets(policy, alt, e0d)))), x; atol = 1e-8)

        # The gradient flows through the reachable bounds. Holding the head
        # output fixed, the target still moves with the incoming energy, so a
        # derivative taken with the bounds detached is strictly smaller.
        λ = _to_like(like, ones(4 * nBat))
        _, g_full = actor_gradient(policy, features, λ, e0d)
        gvec = Float64[]
        Flux.fmap(x -> (x isa AbstractArray && append!(gvec, Float64.(vec(Array(x)))); x), g_full)
        @test any(!iszero, gvec)
        @test all(isfinite, gvec)
    end

    # ── One shared ACP bound relaxation, stated not inherited ───────────────
    @testset "ACP bound-relaxation parity" begin
        @test ACP_BOUND_RELAX_FACTOR == 1e-8
        @test haskey(DEFAULT_SOLVER_OPTIONS, :bound_relax_factor)
        @test DEFAULT_SOLVER_OPTIONS.bound_relax_factor === ACP_BOUND_RELAX_FACTOR
        # Zero is what broke CUDSS; it must never be reintroduced silently.
        @test DEFAULT_SOLVER_OPTIONS.bound_relax_factor > 0
        # The constant arrives from the byte-identical shared contract file, so
        # the two engines cannot drift apart on it.
        @test occursin("ACP_BOUND_RELAX_FACTOR",
                       read(joinpath(EXAMPLE, "battery_solution_schema.jl"), String))
    end

    # ── Which protocol a policy may be SELECTED on ──────────────────────────
    #
    # The final protocol exists to be fresh. A trainer that scored checkpoints on
    # it would destroy that property silently — the columns solve, the costs are
    # finite, and the printed panel looks exactly like a screening panel. An
    # earlier revision of the trainer did precisely this, because
    # `manifest["protocol"]` IS the final protocol and `manifest["screening"]` is
    # the screening one.
    @testset "selection protocol" begin
        # The correctness fixture declares one protocol and no split at all.
        @test get(case.manifest, "screening", nothing) === nothing
        msole, ksole = evaluation_protocol(case)
        @test ksole === :sole
        @test msole == scenario_index_matrix(case.demand,
                                             Int(case.manifest["protocol"]["num_stages"]),
                                             Int(case.manifest["protocol"]["num_scenarios"]))

        # A case that DOES declare a screening protocol must be evaluated on it,
        # and the panel it produces must share no scenario with the final one.
        # The stage count matches the final protocol's, as it does on every panel
        # case, because the exclusion set is that protocol's columns.
        pstages = Int(case.manifest["protocol"]["num_stages"])
        sseed, sscen = 424242, 6
        ex = protocol_columns(scenario_index_matrix(case.demand, pstages,
                                                    Int(case.manifest["protocol"]["num_scenarios"])))
        function with_screening(sha)
            man = deepcopy(case.manifest)
            man["screening"] = Dict{String,Any}("seed" => sseed, "num_stages" => pstages,
                                                "num_scenarios" => sscen,
                                                "excludes" => "protocol", "sha256" => sha)
            return BatteryCase(case.dir, case.name, case.network, case.batteries,
                               case.recourse, case.demand, man)
        end
        good = protocol_digest(case.demand, pstages, sscen; seed = sseed, exclude = ex)
        mscr, kscr = evaluation_protocol(with_screening(good))
        @test kscr === :screening
        @test mscr == scenario_index_matrix(case.demand, pstages, sscen;
                                            seed = sseed, exclude = ex)
        @test mscr != msole[1:pstages, 1:sscen]
        # Disjoint by construction, not by luck.
        @test isempty(intersect(Set(protocol_columns(mscr)), Set(ex)))
        # A screening record whose digest does not regenerate is refused, so a
        # tampered or stale manifest cannot quietly select a different panel.
        @test_throws ErrorException evaluation_protocol(with_screening(repeat("0", 64)))

        # ── The panel mean is the CORRECTED cost, not the raw objective ─────
        # This engine parks the two nodal recourse injections a bound-relaxation
        # BELOW zero. At the case's recourse price that is a visible negative
        # penalty in the raw objective, and a selection metric carrying it would
        # rank checkpoints partly on the solver's barrier parameter. The panel
        # therefore goes through `physical_stage_cost`, exactly as every reported
        # cost in the study does.
        pprob = build_battery_exa(case, 3; backend = backend)
        ppol = to_device(battery_reachable_policy(case, [6]; n_observation = pprob.nBus,
                         n_context = N_CONTEXT, head_layers = [6]))
        ev = evaluate_panel(ppol, pprob, case, [1]; solver_kwargs = solver_kwargs)
        @test ev.complete
        # A raw "worst recourse" could be NEGATIVE and pass a positive tolerance
        # for the wrong reason; the contract's is an absolute value.
        @test ev.worst_recourse >= 0
        @test ev.worst_recourse < PHYSICAL_RECOURSE_TOL
        # Recompute the same column's RAW total, and its generation-plus-
        # throughput. With no recourse used, the corrected cost is exactly the
        # latter — and the raw objective is neither.
        matrix, _ = evaluation_protocol(case)
        atoms = matrix[1:3, 1]
        lk = _policy_array(ppol)
        tg = rollout_targets(ppol, rollout_features(case, pprob.net, atoms; like = lk),
                             _to_like(lk, initial_energy(case)))
        _, psol, _ = strict_solve!(pprob, case, atoms, tg; solver_kwargs = solver_kwargs)
        pc = stage_costs(pprob, psol)
        @test ev.mean_cost ≈ sum(pc.generation) + sum(pc.throughput) atol = 1e-9
        @test ev.mean_cost != sum(pc.total)
        @test abs(sum(pc.total) - ev.mean_cost) ≈ abs(sum(pc.deficit) + sum(pc.surplus)) atol = 1e-9

        # The trainer resolves it once, reports it, and records it in the
        # checkpoint — so which panel selected a policy is on the artifact.
        ck = joinpath(mktempdir(), "prot.jld2")
        out = train_strict(; case_dir = CASE_DIR, num_stages = 3, epochs = 1, batches = 1,
                             trajectories = 1, encoder_layers = [6], head_layers = [6],
                             eval_every = 1, eval_columns = [1], device = DEVICE,
                             checkpoint = ck, verbose = false)
        @test out.protocol === :sole
        @test isfile(ck)
        @test JLD2.load(ck)["meta"]["panel_protocol"] == "sole"
    end

    # ── The recurrent linear decision rule (TSLDR) ───────────────────────────
    #
    # What is being protected here is a NAME. "Time-series linear decision rule"
    # is a claim about a function class, and a recurrent network that quietly
    # kept one gate, one squashing output or one read of the incoming energy
    # would train, reduce loss, and be reported under a name that is false. Each
    # group below closes one way that could happen, and each is stated about the
    # RAW target — the object the claim is about — rather than about the emitted
    # target, which is deliberately a nonlinear (bounded, state-dependent)
    # function of it.
    @testset "recurrent linear TSLDR" begin
        Random.seed!(4711)
        T = 4
        prob = build_battery_exa(case, T; backend = backend)
        mkpolicy(; enc = [6], head = [5], arch = :tsldr_recurrent_linear) =
            to_device(battery_reachable_policy(case, enc; n_observation = prob.nBus,
                      n_context = N_CONTEXT, head_layers = head, architecture = arch))
        policy = mkpolicy()
        assert_device(policy, DEVICE)
        like = _policy_array(policy)
        e0 = initial_energy(case)
        e0d = _to_like(like, e0)
        features = rollout_features(case, prob.net, [1, 3, 2, 2]; like = like)

        # ── What the constructor refuses ─────────────────────────────────────
        @test_throws ArgumentError battery_reachable_policy(case, [6];
                        n_observation = prob.nBus, architecture = :tsldr_linear)
        # A recurrent layer that is not the architecture's own affine one cannot
        # be smuggled in under the linear name.
        @test_throws ArgumentError battery_reachable_policy(case, [6];
                        n_observation = prob.nBus, encoder_type = Flux.GRU,
                        architecture = :tsldr_recurrent_linear)
        # Neither can a different squashing: it belongs to the feasibility layer.
        @test_throws ArgumentError battery_reachable_policy(case, [6];
                        n_observation = prob.nBus, activation = NNlib.sigmoid,
                        architecture = :tsldr_recurrent_linear)

        # ── Activation audit ─────────────────────────────────────────────────
        # Every function-valued field in the trainable tree is the identity, AND
        # every encoder cell is an `RNNCell` — the second half is the decisive
        # one, because an `LSTMCell` writes its sigmoid gates and its tanh into
        # the forward pass and carries no activation field to be caught by the
        # first.
        @test !isempty(policy_activations(policy))
        @test all(f -> f === identity, policy_activations(policy))
        cells = policy_recurrent_cells(policy)
        @test !isempty(cells)
        @test all(c -> c isa Flux.RNNCell, cells)
        @test all(c -> c.σ === identity, cells)
        # The head is affine end to end: every layer of it is a `Dense` whose
        # activation is the identity, output layer included.
        heads = policy.combiner isa Flux.Chain ? collect(policy.combiner.layers) : [policy.combiner]
        @test all(l -> l isa Flux.Dense && l.σ === identity, heads)
        # The null control: the nonlinear architecture must FAIL this audit, or
        # the audit is not measuring anything.
        @test any(c -> !(c isa Flux.RNNCell), policy_recurrent_cells(mkpolicy(arch = :tsddr_nonlinear)))
        @test any(f -> f !== identity, policy_activations(mkpolicy(arch = :tsddr_nonlinear)))

        # ── The raw map is the one the emitted target is built from ─────────
        # Every gate below is stated about `rollout_raw_targets`. That is only
        # evidence about the POLICY if the emitted target really is the
        # feasibility layer applied to those raw values — otherwise the gates
        # could keep passing on a map the rollout no longer uses. Feeding the
        # raw targets through the shared layer, threading the state exactly as
        # the rollout threads it, must reproduce the rollout to the last bit.
        rawz = Array(rollout_raw_targets(policy, features))
        emitted = Float64.(vec(Array(rollout_targets(policy, features, e0d))))
        rebuilt = Float64[]
        prev = e0d
        for t in 1:T
            ê = feasible_target(policy, _to_like(like, Float64.(rawz[:, t])), prev)
            append!(rebuilt, Float64.(vec(Array(ê))))
            prev = ê
        end
        @test rebuilt == emitted

        # ── Causality: a future atom cannot move an earlier raw target ───────
        noise = _to_like(like, randn(MersenneTwister(2), size(features, 1) * (T - 2)))
        alt = hcat(features[:, 1:2], reshape(noise, size(features, 1), T - 2))
        z0 = Array(rollout_raw_targets(policy, features))
        z1 = Array(rollout_raw_targets(policy, alt))
        @test z0[:, 1:2] == z1[:, 1:2]            # EXACTLY, not approximately
        @test !isapprox(z0[:, 3:T], z1[:, 3:T]; atol = 1e-8)

        # ── History: an EARLIER atom does move a later raw target ────────────
        # Separately, one stage at a time, so a single test cannot pass because
        # some other stage happened to carry the difference.
        for k in 1:(T - 1)
            bumped = copy(features)
            bumped[:, k] .+= one(eltype(features))
            zb = Array(rollout_raw_targets(policy, bumped))
            @test !isapprox(zb[:, T], z0[:, T]; atol = 1e-6)
        end

        # ── Ordering: the rule reads a SEQUENCE, not a bag ──────────────────
        # Swapping two stages' observations, which leaves the multiset of
        # observations untouched, must move a later raw target. A map that
        # summed the history would pass every test above and fail this one.
        swapped = hcat(features[:, 2], features[:, 1], features[:, 3:T])
        @test !isapprox(Array(rollout_raw_targets(policy, swapped))[:, T],
                        z0[:, T]; atol = 1e-6)

        # ── Stage dependence is explicit, through the stage representation ──
        # Perturbing ONLY the deterministic clock rows — leaving every demand
        # row exactly as it was — must move the raw target, or the architecture
        # has quietly stopped reading the stage it is at.
        clocked = copy(features)
        clocked[1:N_CONTEXT, :] .+= one(eltype(features))
        @test !isapprox(Array(rollout_raw_targets(policy, clocked)), z0; atol = 1e-6)

        # ── Affinity of the raw map, in Float64 ──────────────────────────────
        # f(αx + (1−α)y) = αf(x) + (1−α)f(y) on arbitrary inputs, which is the
        # definition of affine and is checked at the precision the claim is made
        # at rather than at the trainer's working precision.
        #
        # This block and the two after it are host-Float64 MATH gates: they
        # difference a map against its own closed form, which needs the
        # precision and not the device. The device path is exercised by the
        # rollout, gradient and training blocks around them.
        p64 = Flux.f64(battery_reachable_policy(case, [6]; n_observation = prob.nBus,
                       n_context = N_CONTEXT, head_layers = [5],
                       architecture = :tsldr_recurrent_linear))
        A = randn(MersenneTwister(3), Float64, size(features))
        B = randn(MersenneTwister(4), Float64, size(features))
        α = 0.37
        fA = Array(rollout_raw_targets(p64, A))
        fB = Array(rollout_raw_targets(p64, B))
        fM = Array(rollout_raw_targets(p64, α .* A .+ (1 - α) .* B))
        @test maximum(abs.(fM .- (α .* fA .+ (1 - α) .* fB))) < 1e-12
        # The null control again: the nonlinear head is NOT affine, so the same
        # identity must fail for it. It reads the incoming energy, so the
        # comparison is made on the emitted target it does produce.
        pnl = Flux.f64(battery_reachable_policy(case, [6]; n_observation = prob.nBus,
                       n_context = N_CONTEXT, head_layers = [5]))
        z64 = _to_like(Float64[0.0], initial_energy(case))
        gA = Float64.(vec(Array(rollout_targets(pnl, A, z64))))
        gB = Float64.(vec(Array(rollout_targets(pnl, B, z64))))
        gM = Float64.(vec(Array(rollout_targets(pnl, α .* A .+ (1 - α) .* B, z64))))
        @test maximum(abs.(gM .- (α .* gA .+ (1 - α) .* gB))) > 1e-6

        # ── Explicit unrolling of the small recurrence ───────────────────────
        # h_t = A h_{t-1} + B ξ_t + b and z_t = C h_t + D ξ_t + d, written out
        # from the weight matrices by hand and compared against the
        # implementation. A single-layer encoder and a bare head, so the formula
        # is the one the documentation states, with nothing composed away.
        Random.seed!(1234)
        plain = Flux.f64(battery_reachable_policy(case, [5];
                    n_observation = prob.nBus, n_context = N_CONTEXT,
                    head_layers = Int[], architecture = :tsldr_recurrent_linear))
        cell = policy_recurrent_cells(plain)[1]
        Bm, Am, bv = Array(cell.Wi), Array(cell.Wh), Array(cell.bias)
        Wc, dv = Array(plain.combiner.weight), Array(plain.combiner.bias)
        width = size(Am, 1)
        Cm, Dm = Wc[:, 1:width], Wc[:, (width + 1):end]
        Ξ = Array(rollout_features(case, prob.net, [2, 1, 3, 1]; like = Float64[0.0]))
        h = zeros(Float64, width)
        manual = similar(Ξ, size(Cm, 1), T)
        for t in 1:T
            h = Am * h + Bm * Ξ[:, t] + bv          # the affine recurrent update
            manual[:, t] = Cm * h + Dm * Ξ[:, t] + dv
        end
        @test Array(rollout_raw_targets(plain, Ξ)) ≈ manual atol = 1e-12
        # And the closed form of the same thing: z_t as a sum over the history.
        closed = similar(manual)
        for t in 1:T
            acc = zeros(Float64, width)
            for k in 1:t
                acc .+= Am^(t - k) * (Bm * Ξ[:, k] + bv)
            end
            closed[:, t] = Cm * acc + Dm * Ξ[:, t] + dv
        end
        @test closed ≈ manual atol = 1e-12

        # ── Targets stay reachable over the full T = 24 horizon ─────────────
        prob24 = build_battery_exa(case, 24; backend = backend)
        p24 = to_device(battery_reachable_policy(case, [8]; n_observation = prob24.nBus,
                        n_context = N_CONTEXT, head_layers = [10],
                        architecture = :tsldr_recurrent_linear))
        like24 = _policy_array(p24)
        f24 = rollout_features(case, prob24.net,
                               [1 + (t % num_atoms(case.demand, t)) for t in 1:24]; like = like24)
        x24 = Float64.(vec(Array(rollout_targets(p24, f24, _to_like(like24, e0)))))
        prev = copy(e0)
        for t in 1:24
            for (k, b) in enumerate(bats)
                l, u = reachable_interval(b, prev[k], Δt)
                @test l - 1e-5 <= x24[(t - 1) * nBat + k] <= u + 1e-5
            end
            prev = x24[((t - 1) * nBat + 1):(t * nBat)]
        end

        # ── The actor gradient against centered finite differences ──────────
        # Directional, in Float64, at T = 4 and T = 24, with NO relaxed
        # tolerance: the surrogate ⟨λ, ê(θ)⟩ is smooth here, and the distance to
        # the nearest kink of the feasibility layer is measured first so that
        # "smooth here" is established rather than assumed.
        for (Th, pr) in ((4, prob), (24, prob24))
            Random.seed!(2026)
            pg = Flux.f64(battery_reachable_policy(case, [6]; n_observation = pr.nBus,
                          n_context = N_CONTEXT, head_layers = [8],
                          architecture = :tsldr_recurrent_linear))
            lk = Float64[0.0]
            e0g = _to_like(lk, e0)
            fg = rollout_features(case, pr.net,
                                  [1 + (t % num_atoms(case.demand, t)) for t in 1:Th]; like = lk)
            λ = _to_like(lk, randn(MersenneTwister(5), Th * nBat))

            # Distance to the nearest branch switch of `max`/`min` in the
            # reachable bounds, along the trajectory actually taken.
            xs = Float64.(vec(Array(rollout_targets(pg, fg, e0g))))
            prev = copy(e0); kink = Inf
            for t in 1:Th
                for (k, b) in enumerate(bats)
                    dec = b.self_discharge * prev[k]
                    kink = min(kink,
                               abs(b.energy_min - (dec - Δt * b.discharge_max / b.discharge_efficiency)),
                               abs(b.energy_max - (dec + b.charge_efficiency * Δt * b.charge_max)))
                end
                prev = xs[((t - 1) * nBat + 1):(t * nBat)]
            end
            @test kink > 1e-5

            _, g = actor_gradient(pg, fg, λ, e0g)
            gv = Float64[]
            Flux.fmap(x -> (x isa AbstractArray && append!(gv, Float64.(vec(Array(x)))); x), g)
            dir = randn(MersenneTwister(11), length(gv)); dir ./= norm(dir)
            ad = dot(gv, dir)
            bump(h) = begin
                q = deepcopy(pg); i = Ref(0)
                shift(x) = x isa AbstractArray ?
                    (a = copy(x); for j in eachindex(a); i[] += 1; a[j] += h * dir[i[]]; end; a) : x
                q.encoder = Flux.fmap(shift, q.encoder)
                q.combiner = Flux.fmap(shift, q.combiner)
                sum(λ .* rollout_targets(q, fg, e0g))
            end
            h = 1e-5
            fd = (bump(h) - bump(-h)) / (2h)
            @test isapprox(fd, ad; rtol = 1e-7)
        end

        # ── Save and reload preserves architecture, parameters, optimizer
        #    state and the emitted trajectory ─────────────────────────────────
        keep = mkpolicy()
        opt = Optimisers.setup(Optimisers.Adam(1e-3), keep)
        # Take one real step so the moment estimates are not their initial value,
        # or "the optimizer state survived" would be true of a fresh setup too.
        _, g1 = actor_gradient(keep, features, _to_like(like, ones(T * nBat)), e0d)
        opt, keep = Optimisers.update!(opt, keep, g1)
        traj = Float64.(vec(Array(rollout_targets(keep, features, e0d))))
        moments = Float64[]
        Flux.fmap(x -> (x isa AbstractArray && append!(moments, Float64.(vec(Array(x)))); x), opt)

        cpath = joinpath(mktempdir(), "tsldr.jld2")
        save_checkpoint(cpath, keep, Dict(
            "case" => case.name,
            "network_sha256" => case.manifest["artifacts"]["network.json"]);
            opt_state = opt, history = [(step = 1, loss = 1.5)])

        back = mkpolicy()
        got = load_checkpoint!(back, cpath; case = case)
        @test got.meta["architecture"] == "tsldr_recurrent_linear"
        @test Float64.(vec(Array(rollout_targets(back, features, e0d)))) == traj
        rmoments = Float64[]
        Flux.fmap(x -> (x isa AbstractArray && append!(rmoments, Float64.(vec(Array(x)))); x), got.opt_state)
        @test rmoments == moments
        @test got.history == [(step = 1, loss = 1.5)]

        # A nonlinear checkpoint may not be loaded into a linear policy, and the
        # refusal is by NAME — before any array is touched, so it does not
        # depend on two encoders happening to have incompatible weight shapes.
        npath = joinpath(mktempdir(), "nl.jld2")
        save_checkpoint(npath, mkpolicy(arch = :tsddr_nonlinear), Dict(
            "case" => case.name,
            "network_sha256" => case.manifest["artifacts"]["network.json"]))
        @test_throws ErrorException load_checkpoint!(back, npath; case = case)
        @test_throws ErrorException load_checkpoint!(mkpolicy(arch = :tsddr_nonlinear),
                                                     cpath; case = case)

        # ── Training runs the linear architecture through the same path ─────
        out = train_strict(; case_dir = CASE_DIR, architecture = :tsldr_recurrent_linear,
                             num_stages = 4, epochs = 1, batches = 2, trajectories = 1,
                             encoder_layers = [8], head_layers = [12],
                             eval_every = 2, eval_columns = [1, 2], device = DEVICE,
                             checkpoint = joinpath(mktempdir(), "tsldr_smoke.jld2"),
                             verbose = false)
        @test out.architecture === :tsldr_recurrent_linear
        @test out.updates == 2
        @test all(isfinite(h.loss) for h in out.history)
        @test isfinite(out.best.cost)
        @test isfile(out.checkpoint)
        @test out.policy.architecture === :tsldr_recurrent_linear
    end

    # ── The four method identifiers ─────────────────────────────────────────
    @testset "method identifiers" begin
        @test sort!(collect(keys(BATTERY_METHODS))) ==
              [:sddp_dc, :sddp_soc, :tsddr_nonlinear, :tsldr_recurrent_linear]
        for id in keys(BATTERY_METHODS)
            m = battery_method(id)
            @test m.id === id
            # The invariants are what make the four comparable, so every row
            # carries them and every row carries the SAME ones.
            @test m.horizon == 24
            @test m.protocol == "screening"
            @test m.stage_semantics == BATTERY_METHOD_INVARIANTS.stage_semantics
            @test m.recourse == BATTERY_METHOD_INVARIANTS.recourse
            @test m.cost_contract == BATTERY_METHOD_INVARIANTS.cost_contract
            @test m.comparison == BATTERY_METHOD_INVARIANTS.comparison
        end
        @test battery_method(:tsddr_nonlinear).engine === :exa
        @test battery_method(:tsldr_recurrent_linear).architecture === :tsldr_recurrent_linear
        @test battery_method(:sddp_soc).backward === :soc
        @test battery_method(:sddp_dc).backward === :dc
        @test_throws ArgumentError battery_method(:tsldr)
        # This engine refuses the two it does not own, by name and with the
        # owning engine in the message, rather than failing deeper down.
        @test_throws ErrorException run_battery_method(:sddp_soc)
        @test_throws ErrorException run_battery_method(:sddp_dc)
        # And it really does run the two it does own.
        out = run_battery_method(:tsldr_recurrent_linear; case_dir = CASE_DIR,
                                 num_stages = 3, epochs = 1, batches = 1, trajectories = 1,
                                 encoder_layers = [6], head_layers = [6], eval_every = 0,
                                 device = DEVICE,
                                 checkpoint = joinpath(mktempdir(), "m.jld2"), verbose = false)
        @test out.architecture === :tsldr_recurrent_linear && out.updates == 1
    end

    # ── Checkpoints ──────────────────────────────────────────────────────────
    @testset "checkpoint round trip" begin
        Random.seed!(3)
        prob = build_battery_exa(case, 3; backend = backend)
        policy = to_device(battery_reachable_policy(case, [6];
                           n_observation = prob.nBus, n_context = N_CONTEXT,
                           head_layers = [8]))
        like = _policy_array(policy)
        e0d = _to_like(like, initial_energy(case))
        features = rollout_features(case, prob.net, [2, 1, 3]; like = like)
        before = Float64.(vec(Array(rollout_targets(policy, features, e0d))))

        path = joinpath(mktempdir(), "ckpt.jld2")
        opt = Optimisers.setup(Optimisers.Adam(1e-3), policy)
        save_checkpoint(path, policy, Dict(
            "case" => case.name,
            "network_sha256" => case.manifest["artifacts"]["network.json"]);
            opt_state = opt, history = [(step = 1, loss = 2.0)])

        fresh = to_device(battery_reachable_policy(case, [6];
                          n_observation = prob.nBus, n_context = N_CONTEXT,
                          head_layers = [8]))
        @test !isapprox(Float64.(vec(Array(rollout_targets(fresh, features, e0d)))), before; atol = 1e-8)
        loaded = load_checkpoint!(fresh, path; case = case)
        @test loaded.meta["case"] == case.name
        @test loaded.meta["schema"] == BATTERY_CHECKPOINT_SCHEMA
        # The architecture travels with the checkpoint, always.
        @test loaded.meta["architecture"] == "tsddr_nonlinear"
        # Exact reproduction, not merely close: the reload must be the policy.
        @test Float64.(vec(Array(rollout_targets(fresh, features, e0d)))) == before
        # The optimizer state and the trajectory came back too.
        @test loaded.opt_state !== nothing
        @test loaded.history == [(step = 1, loss = 2.0)]

        # A checkpoint from a different case is refused rather than mismatched.
        bad = joinpath(mktempdir(), "bad.jld2")
        save_checkpoint(bad, policy, Dict("case" => "other", "network_sha256" => "deadbeef"))
        @test_throws ErrorException load_checkpoint!(fresh, bad; case = case)

        # A checkpoint written before the schema existed is refused rather than
        # resumed with a fresh optimizer while reporting that it resumed.
        stale = joinpath(mktempdir(), "stale.jld2")
        JLD2.jldsave(stale; state = Flux.state(Flux.cpu(policy)),
                     meta = Dict("case" => case.name))
        @test_throws ErrorException load_checkpoint!(fresh, stale; case = case)
    end

    # ── Training performs real updates ───────────────────────────────────────
    @testset "training smoke" begin
        out = train_strict(; case_dir = CASE_DIR, num_stages = 4, epochs = 1, batches = 2,
                             trajectories = 1, encoder_layers = [8], head_layers = [12],
                             eval_every = 2, eval_columns = [1, 2], device = DEVICE,
                             checkpoint = joinpath(mktempdir(), "smoke.jld2"), verbose = false)
        @test out.updates == 2
        @test length(out.history) == 2
        @test all(isfinite(h.loss) for h in out.history)
        # The learning-rate schedule is a declared function of the step index.
        @test out.history[1].lr > out.history[end].lr
        # A complete panel evaluation selected a checkpoint.
        @test isfinite(out.best.cost)
        @test isfile(out.checkpoint)
    end

    # ── The physical cost contract, on THIS engine's own solutions ──────────
    #
    # `physical_stage_cost` arrives with the byte-identical schema file, so the
    # two engines do not merely follow the same recipe — they run the same code.
    # What this checks is that MadNLP's own barrier artifact is removed by it:
    # this engine parks the recourse variables a bound-relaxation BELOW zero,
    # which at the case's recourse price is a visible negative penalty in the
    # raw objective and must not reach a reported cost.
    @testset "physical cost semantics" begin
        prob = build_battery_exa(case, 2)
        pd, qd = realized_demand(case, net, [1, 2], [1, 2])
        set_demand!(prob, pd, qd)
        e0 = [b.energy_initial for b in bats]
        xhat = Float64[]
        e_prev = copy(e0)
        for _ in 1:2
            nxt = similar(e_prev)
            for (k, b) in enumerate(bats)
                lo, hi = reachable_interval(b, e_prev[k], Δt)
                nxt[k] = lo + 0.5 * (hi - lo)
            end
            append!(xhat, nxt); e_prev = nxt
        end
        set_energy_path!(prob, e0, xhat)
        res = solve!(prob)
        sol = battery_solution(prob, res)
        cost = stage_costs(prob, sol)
        bus_of = Dict(i => net.buses[i].id for i in eachindex(net.buses))

        for t in 1:2
            s = (cost_generation = cost.generation[t], cost_throughput = cost.throughput[t],
                 deficit = Dict(bus_of[i] => sol.deficit[i, t] for i in eachindex(net.buses)),
                 surplus = Dict(bus_of[i] => sol.surplus[i, t] for i in eachindex(net.buses)),
                 objective = cost.total[t])
            c = physical_stage_cost(s, case.recourse)
            # This stage uses no recourse, so the corrected cost is exactly the
            # physical generation plus throughput — no barrier residue at all.
            @test c.admissible
            @test c.worst_recourse < PHYSICAL_RECOURSE_TOL
            @test c.deficit == 0.0
            @test c.surplus == 0.0
            @test c.corrected ≈ cost.generation[t] + cost.throughput[t]
            @test isfinite(c.correction)
        end
    end

    # ── The files the two engines must agree on byte for byte ───────────────
    #
    # This package has no PGLib and no PowerModels dependency, so it cannot
    # build a case and cannot construct the portfolio: it CONSUMES both. What it
    # can check, and must, is that the copies it consumes them through have not
    # drifted from the JuMP package's originals — a drifted `battery_case.jl`
    # would let the two engines read one frozen case as two different problems
    # while both reported success.
    @testset "shared files and the portfolio manifest" begin
        peer = get(ENV, "DR_BAT_PEER",
                   normpath(joinpath(EXAMPLE, "..", "..", "..",
                                     "DecisionRules.jl", "examples",
                                     "BatteryStorageOPF")))
        shared = ["battery_case.jl", "battery_solution_schema.jl"]
        manifest = joinpath(EXAMPLE, "battery_portfolio.json")
        if isdir(peer)
            isfile(manifest) && push!(shared, "battery_portfolio.json")
            for f in shared
                @test isfile(joinpath(EXAMPLE, f))
                @test isfile(joinpath(peer, f))
                @test bytes2hex(SHA.sha256(read(joinpath(EXAMPLE, f)))) ==
                      bytes2hex(SHA.sha256(read(joinpath(peer, f))))
            end
        else
            for f in shared
                @test isfile(joinpath(EXAMPLE, f))
            end
        end

        # The portfolio manifest is readable and self-consistent from THIS side
        # too, with nothing but JSON and SHA — which is the whole point of
        # keeping it a small, hash-checkable file rather than a case tree.
        if isfile(manifest)
            m = JSON.parsefile(manifest)
            @test m["schema"] == "battery_storage_opf/portfolio/1"
            @test m["seed"] == 20260814
            @test m["horizon"] == 24
            @test length(m["profile"]) == 24
            @test length(m["cases"]) >= 10
            # Recompute the manifest's self-digest: canonical JSON of everything
            # but the digest field. `canonical_json` came in with
            # `battery_case.jl`, which the byte-identity test above has already
            # pinned to the JuMP package's copy.
            body = Dict{String,Any}(k => v for (k, v) in m if k != "digest")
            @test bytes2hex(SHA.sha256(canonical_json(plain(body)))) == m["digest"]
            for c in m["cases"]
                @test length(c["placement"]["buses"]) == c["counts"]["battery"]
                @test length(c["regions"]["sizes"]) == 6
                @test c["protocols"]["screening"]["excludes"] == "final"
            end
        end
    end
end
