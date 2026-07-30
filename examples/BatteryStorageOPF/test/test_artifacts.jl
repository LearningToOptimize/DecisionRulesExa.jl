# test_artifacts.jl — REPAIR B/C: exact experiment reconstruction, five distinct
# hashes, tampering detection, and a FRESH-PROCESS artifact round trip.
#
# Run (from examples/BatteryStorageOPF):
#   julia --pkgimages=no --project=. test/test_artifacts.jl
#
# The round-trip step launches a SEPARATE Julia process that is given only the
# saved artifacts (manifest, evaluation protocol, checkpoint) and must reproduce
# the scenario indices, statuses, cost decomposition, active recourse, and aggregate
# physical cost of the in-process evaluation.

include(joinpath(@__DIR__, "..", "src", "BatteryStorageOPF.jl"))
using .BatteryStorageOPF
using Test
using MadNLP
using Flux
using Random
using JSON
using SHA
using Serialization
using DecisionRulesExa

const CASE = "case14_ieee"
const REPORT, LOOKAH = 2, 1
const T = REPORT + LOOKAH
const NEVAL = 4

@testset "artifacts: exact reconstruction, hashes, tampering, round trip" begin
    dir = mktempdir()
    case = make_battery_case(CASE; number_of_batteries = 2, seed = 20260722)
    process = make_load_process(case; preset = DEFAULT_DEMAND_PRESET, nregion = 2, period = 4)

    train_mat = scenario_index_matrix(process, T, 4; seed = process.train_seed)
    eval_mat  = scenario_index_matrix(process, T, NEVAL; seed = process.eval_seed)
    train_path = joinpath(dir, "train_protocol.json")
    eval_path  = joinpath(dir, "eval_protocol.json")
    write_scenario_protocol(train_path, process, train_mat; kind = "train", seed = process.train_seed)
    write_scenario_protocol(eval_path,  process, eval_mat;  kind = "eval",  seed = process.eval_seed)
    man_path = joinpath(dir, "manifest.json")

    h_proc  = process_hash(process)
    h_train = index_matrix_hash(train_mat)
    h_eval  = index_matrix_hash(eval_mat)
    f_train = bytes2hex(open(sha256, train_path))
    f_eval  = bytes2hex(open(sha256, eval_path))

    write_stochastic_manifest(man_path, case, process;
        reporting_horizon = REPORT, lookahead = LOOKAH, mode = :soft, stage_hours = 1.0,
        active_recourse_cost_per_mwh = DEFAULT_ACTIVE_RECOURSE_COST_PER_MWH, rho1 = 0.0, rho2 = 0.0,
        activation = string(stretchedsigmoid), safe_upper_margin = 1e-3,
        policy_layers = [8], policy_combiner_layers = Int[], policy_seed = 7,
        train_index_matrix_hash = h_train, eval_index_matrix_hash = h_eval,
        train_protocol_file_sha256 = f_train, eval_protocol_file_sha256 = f_eval,
        train_paths = 4, eval_paths = NEVAL)

    # ── Five DISTINCT hashes, none reused for another role ───────────────────
    @testset "five distinct hashes" begin
        doc = JSON.parsefile(man_path); h = doc["hashes"]
        vals = [h["load_process_content_hash_sha256"], h["train_index_matrix_hash_sha256"],
                h["eval_index_matrix_hash_sha256"], h["train_protocol_file_sha256"],
                h["eval_protocol_file_sha256"]]
        @test all(x -> x isa String && length(x) == 64, vals)
        @test length(unique(vals)) == 5                  # all five differ
        @test h["load_process_content_hash_sha256"] == h_proc
        # The process hash must NOT masquerade as a protocol hash.
        @test h["train_protocol_file_sha256"] != h_proc
        @test h["eval_protocol_file_sha256"] != h_proc
        @test !haskey(doc, "train_protocol_hash") && !haskey(doc, "eval_protocol_hash")
    end

    # ── Exact field-for-field reconstruction ─────────────────────────────────
    @testset "exact reconstruction from manifest" begin
        case2, proc2, meta = reconstruct_stochastic_manifest(man_path)
        @test manifest_hash(case2) == manifest_hash(case)
        @test process_hash(proc2) == h_proc
        @test proc2.base_shape == process.base_shape          # exact vector
        @test proc2.region_of_bus == process.region_of_bus
        @test proc2.anchor_bus_ids == process.anchor_bus_ids
        @test proc2.period == process.period
        @test proc2.preset == process.preset
        @test [a.system_factor for a in proc2.atoms] == [a.system_factor for a in process.atoms]
        @test [a.regional_factors for a in proc2.atoms] == [a.regional_factors for a in process.atoms]
        @test proc2.probs == process.probs
        @test (proc2.train_seed, proc2.eval_seed) == (process.train_seed, process.eval_seed)
        @test meta.reporting_horizon == REPORT && meta.lookahead == LOOKAH
        @test meta.active_recourse_cost_per_mwh == DEFAULT_ACTIVE_RECOURSE_COST_PER_MWH
        @test meta.activation == string(stretchedsigmoid)
        # Materialized scenarios are identical after reconstruction.
        @test materialize_all(proc2, eval_mat) == materialize_all(process, eval_mat)
    end

    # ── Tampering detection ──────────────────────────────────────────────────
    @testset "tampering: base_shape, atoms, indices, protocol files" begin
        # (a) base_shape tampered in the manifest → process-hash mismatch.
        doc = JSON.parsefile(man_path)
        doc["load_process"]["base_shape"][1] += 0.01
        bad = joinpath(dir, "bad_shape.json"); open(io -> JSON.print(io, doc, 2), bad, "w")
        @test_throws ErrorException reconstruct_stochastic_manifest(bad)

        # (b) atom definition tampered → process-hash mismatch.
        doc = JSON.parsefile(man_path)
        doc["load_process"]["atoms"][1]["system_factor"] += 0.05
        bad = joinpath(dir, "bad_atom.json"); open(io -> JSON.print(io, doc, 2), bad, "w")
        @test_throws ErrorException reconstruct_stochastic_manifest(bad)

        # (c) ORDER of scenario indices tampered → index-matrix hash mismatch.
        pdoc = JSON.parsefile(eval_path)
        pdoc["index_matrix"][1], pdoc["index_matrix"][2] =
            pdoc["index_matrix"][2], pdoc["index_matrix"][1]
        badp = joinpath(dir, "bad_order.json"); open(io -> JSON.print(io, pdoc, 2), badp, "w")
        @test_throws ErrorException reconstruct_scenario_protocol(badp)

        # (d) protocol FILE bytes tampered → file-hash mismatch (independent of
        #     the content hash: reformatting alone changes the bytes).
        tampered = joinpath(dir, "eval_reformatted.json")
        open(io -> JSON.print(io, JSON.parsefile(eval_path)), tampered, "w")   # no indent
        @test_throws ErrorException verify_protocol_file(tampered, f_eval)
        @test verify_protocol_file(eval_path, f_eval) == f_eval                # untampered OK
    end

    # ── FRESH-PROCESS round trip ─────────────────────────────────────────────
    @testset "round trip in a separate Julia process" begin
        # Build + save a tiny checkpoint, evaluate in-process, then re-evaluate in
        # a NEW Julia process from the artifacts alone and compare.
        de = build_battery_tsddr_de(case, process; reporting_horizon = REPORT,
                                    lookahead = LOOKAH, mode = :soft,
                                    rho1 = 0.0, rho2 = 0.0, stage_hours = 1.0)
        stage = build_battery_stage_problem(case, process; mode = :soft,
                                            rho1 = 0.0, rho2 = 0.0, stage_hours = 1.0)
        Random.seed!(7)
        policy = battery_reachable_policy(case, process; dt = 1.0, layers = [8])
        ckpt = joinpath(dir, "ckpt.jls")
        save_checkpoint(ckpt, policy, de; case = case, process = process)

        ev = evaluate_paired(policy, stage, process, eval_mat;
                             reporting_horizon = REPORT, keep_trajectories = true)
        @test ev.n_ok == NEVAL

        script = joinpath(dir, "roundtrip.jl")
        open(script, "w") do io
            println(io, """
            include(raw"$(joinpath(@__DIR__, "..", "src", "BatteryStorageOPF.jl"))")
            using .BatteryStorageOPF, MadNLP, Flux, JSON, DecisionRulesExa
            case, process, meta = reconstruct_stochastic_manifest(raw"$man_path")
            verify_protocol_file(raw"$eval_path", meta.eval_protocol_file_sha256)
            _, emat, _ = reconstruct_scenario_protocol(raw"$eval_path")
            policy, ck = load_checkpoint(raw"$ckpt", case, process)
            stage = build_battery_stage_problem(case, process; mode = meta.mode,
                        rho1 = meta.rho1, rho2 = meta.rho2,
                        active_recourse_cost_per_mwh = meta.active_recourse_cost_per_mwh,
                        stage_hours = meta.stage_hours)
            ev = evaluate_paired(policy, stage, process, emat;
                                 reporting_horizon = meta.reporting_horizon,
                                 keep_trajectories = true)
            statuses = String[]
            for tr in ev.trajectories, st in tr.trajectory
                push!(statuses, st["status"])
            end
            open(raw"$(joinpath(dir, "roundtrip_out.json"))", "w") do f
                JSON.print(f, Dict(
                    "process_hash" => process_hash(process),
                    "index_hash" => index_matrix_hash(emat),
                    "indices" => [Int.(emat[t, :]) for t in 1:size(emat, 1)],
                    "mean_cost" => ev.mean_reporting_physical_cost,
                    "costs" => ev.reporting_physical_costs,
                    "deficit_mwh" => ev.total_active_deficit_energy_mwh,
                    "surplus_mwh" => ev.total_active_surplus_energy_mwh,
                    "n_ok" => ev.n_ok, "statuses" => statuses,
                    "gen" => [tr.generator_cost for tr in ev.trajectories],
                    "thr" => [tr.battery_throughput_cost for tr in ev.trajectories],
                    "recourse_cost" => [tr.active_recourse_cost for tr in ev.trajectories],
                    "raw_recourse_cost" => [tr.raw_active_recourse_cost for tr in ev.trajectories],
                    "proj_corr" => [tr.active_recourse_projection_correction for tr in ev.trajectories],
                    "lb_viol" => ev.maximum_active_recourse_lower_bound_violation_pu,
                ), 2)
            end
            """)
        end
        outfile = joinpath(dir, "roundtrip_out.json")
        jl = joinpath(Sys.BINDIR, "julia")
        proj = dirname(Base.active_project())
        run(`$jl --pkgimages=no --project=$proj $script`)
        @test isfile(outfile)
        got = JSON.parsefile(outfile)

        # Identical scenario indices and hashes.
        @test got["process_hash"] == h_proc
        @test got["index_hash"] == h_eval
        @test [Int.(r) for r in got["indices"]] == [Int.(eval_mat[t, :]) for t in 1:T]
        # Identical statuses, recourse, decomposition, and aggregate physical cost.
        @test got["n_ok"] == NEVAL
        @test all(s -> s == "SOLVE_SUCCEEDED" || s == "SOLVED_TO_ACCEPTABLE_LEVEL",
                  got["statuses"])
        @test isapprox(Float64(got["mean_cost"]), ev.mean_reporting_physical_cost; rtol = 1e-8)
        # Both recourse directions round-trip independently and are nonnegative.
        @test 0.0 <= Float64(got["deficit_mwh"])
        @test 0.0 <= Float64(got["surplus_mwh"])
        @test isapprox(Float64(got["deficit_mwh"]), ev.total_active_deficit_energy_mwh;
                       rtol = 1e-6, atol = 1e-9)
        @test isapprox(Float64(got["surplus_mwh"]), ev.total_active_surplus_energy_mwh;
                       rtol = 1e-6, atol = 1e-9)
        # The raw lower-bound violation stayed within the declared tolerance.
        @test Float64(got["lb_viol"]) <= ACTIVE_RECOURSE_LB_TOL_PU
        for (p, c) in enumerate(got["costs"])
            @test isapprox(Float64(c), ev.reporting_physical_costs[p]; rtol = 1e-8)
        end
        for (p, tr) in enumerate(ev.trajectories)
            @test isapprox(Float64(got["gen"][p]), tr.generator_cost; rtol = 1e-8)
            @test isapprox(Float64(got["thr"][p]), tr.battery_throughput_cost; rtol = 1e-8)
            # Public recourse cost is ≥ 0 and round-trips; raw diagnostic round-trips too.
            @test Float64(got["recourse_cost"][p]) >= 0.0
            @test isapprox(Float64(got["recourse_cost"][p]), tr.active_recourse_cost;
                           rtol = 1e-6, atol = 1e-9)
            @test isapprox(Float64(got["raw_recourse_cost"][p]), tr.raw_active_recourse_cost;
                           rtol = 1e-6, atol = 1e-9)
            @test isapprox(Float64(got["proj_corr"][p]), tr.active_recourse_projection_correction;
                           rtol = 1e-6, atol = 1e-9)
        end
    end
end
