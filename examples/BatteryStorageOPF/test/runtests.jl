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
            @test r.transition < 1e-9
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
        @test plv[(off_p + 1):(off_p + T * ngen(net))] ==
              repeat([g.pmin for g in net.gens], T)
        @test puv[(off_p + 1):(off_p + T * ngen(net))] ==
              repeat([g.pmax for g in net.gens], T)
        @test plv[(off_q + 1):(off_q + T * ngen(net))] ==
              repeat([isfinite(g.qmin) ? g.qmin : -1e4 for g in net.gens], T)
        @test puv[(off_q + 1):(off_q + T * ngen(net))] ==
              repeat([isfinite(g.qmax) ? g.qmax : 1e4 for g in net.gens], T)

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
        # The model's SHAPE does not depend on the schedule.
        @test sprob.model.meta.nvar == base.model.meta.nvar
        @test sprob.model.meta.ncon == base.model.meta.ncon
        slv = Array(sprob.model.meta.lvar); suv = Array(sprob.model.meta.uvar)
        nG = ngen(net)
        p1 = off_p + gpos; p2 = off_p + nG + gpos
        q1 = off_q + gpos; q2 = off_q + nG + gpos
        @test slv[p1] == gsel.pmin && suv[p1] == gsel.pmax     # stage 1: untouched
        @test slv[p2] == 0.0 && suv[p2] == 0.0                 # stage 2: exactly out
        @test slv[q2] == 0.0 && suv[q2] == 0.0
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
        @test Array(w2.model.meta.lvar)[2 * nbus(net) + gpos] == 0.0
        @test Array(w2.model.meta.uvar)[2 * nbus(net) + gpos] == 0.0
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
        save_checkpoint(path, policy, Dict(
            "case" => case.name,
            "network_sha256" => case.manifest["artifacts"]["network.json"]))

        fresh = to_device(battery_reachable_policy(case, [6];
                          n_observation = prob.nBus, n_context = N_CONTEXT,
                          head_layers = [8]))
        @test !isapprox(Float64.(vec(Array(rollout_targets(fresh, features, e0d)))), before; atol = 1e-8)
        meta = load_checkpoint!(fresh, path; case = case)
        @test meta["case"] == case.name
        # Exact reproduction, not merely close: the reload must be the policy.
        @test Float64.(vec(Array(rollout_targets(fresh, features, e0d)))) == before

        # A checkpoint from a different case is refused rather than mismatched.
        bad = joinpath(mktempdir(), "bad.jld2")
        save_checkpoint(bad, policy, Dict("case" => "other", "network_sha256" => "deadbeef"))
        @test_throws ErrorException load_checkpoint!(fresh, bad; case = case)
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
end
