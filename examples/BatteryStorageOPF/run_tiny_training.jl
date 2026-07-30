#!/usr/bin/env julia
# run_tiny_training.jl
#
# Beginner-usable CPU TS-DDR training that PRODUCES THE ARTIFACT SET the rest of
# the workflow consumes:
#
#   results/stochastic_manifest_<case>.json  — the experiment definition (exact
#       demand process, horizons, target mode, VOLL/penalties, policy arch, and
#       five distinct hashes)
#   results/train_protocol_<case>.json       — training scenario-index matrix
#   results/eval_protocol_<case>.json        — FIXED held-out evaluation matrix
#   results/checkpoint_<case>.jls            — policy checkpoint
#   results/trajectory_<case>.json           — compact per-stage trajectory
#
# `evaluate_checkpoint.jl` reconstructs the experiment from these artifacts alone
# — it never regenerates the demand process from defaults or env vars.
#
# Run (from examples/BatteryStorageOPF):
#   module load julia
#   julia --pkgimages=no --project=. run_tiny_training.jl
#
# Env overrides: BAT_CASE, BAT_NBAT, BAT_SEED, BAT_NREGION, BAT_PRESET,
#   BAT_REPORT, BAT_LOOKAHEAD, BAT_MODE (soft|strict), BAT_BATCHES, BAT_NTRAIN,
#   BAT_NEVAL, BAT_LR, BAT_POLICY_SEED, BAT_OUTDIR.

include(joinpath(@__DIR__, "src", "BatteryStorageOPF.jl"))
using .BatteryStorageOPF
using MadNLP
using Flux
using Random
using Printf
using SHA

const CASE     = get(ENV, "BAT_CASE", "case14_ieee")
const NBAT     = parse(Int, get(ENV, "BAT_NBAT", "3"))
const SEED     = parse(Int, get(ENV, "BAT_SEED", "20260722"))
const NREGION  = parse(Int, get(ENV, "BAT_NREGION", "3"))
const PRESET   = Symbol(get(ENV, "BAT_PRESET", string(DEFAULT_DEMAND_PRESET)))
const REPORT   = parse(Int, get(ENV, "BAT_REPORT", "4"))
const LOOKAH   = parse(Int, get(ENV, "BAT_LOOKAHEAD", "1"))
const MODE     = Symbol(get(ENV, "BAT_MODE", "strict"))
const BATCHES  = parse(Int, get(ENV, "BAT_BATCHES", "40"))
const NTRAIN   = parse(Int, get(ENV, "BAT_NTRAIN", "8"))
const NEVAL    = parse(Int, get(ENV, "BAT_NEVAL", "8"))
const LR       = parse(Float64, get(ENV, "BAT_LR", "0.01"))
const OUTDIR   = get(ENV, "BAT_OUTDIR", joinpath(@__DIR__, "results"))
const POLSEED  = parse(Int, get(ENV, "BAT_POLICY_SEED", "1234"))
const LAYERS   = [32, 32]
const COMBINER = [32]

function main()
    mkpath(OUTDIR)
    T = REPORT + LOOKAH
    @printf("Case %s: %d batteries, %d regions, preset=%s, mode=%s, T=%d (report=%d + look=%d)\n",
            CASE, NBAT, NREGION, PRESET, MODE, T, REPORT, LOOKAH)

    case = make_battery_case(CASE; number_of_batteries = NBAT, seed = SEED)
    process = make_load_process(case; preset = PRESET, nregion = NREGION, period = max(T, 4))
    mult = demand_multiplier_summary(process)
    @printf("  demand multipliers: max system %.4f, max bus %.4f\n",
            mult.max_system_multiplier, mult.max_bus_multiplier)

    # Paired protocols: distinct declared seeds; eval is the fixed held-out set.
    train_mat = scenario_index_matrix(process, T, NTRAIN; seed = process.train_seed)
    eval_mat  = scenario_index_matrix(process, T, NEVAL;  seed = process.eval_seed)
    train_path = joinpath(OUTDIR, "train_protocol_$(CASE).json")
    eval_path  = joinpath(OUTDIR, "eval_protocol_$(CASE).json")
    write_scenario_protocol(train_path, process, train_mat; kind = "train", seed = process.train_seed)
    write_scenario_protocol(eval_path,  process, eval_mat;  kind = "eval",  seed = process.eval_seed)

    # Five DISTINCT hashes: process content, the two index matrices, and the two
    # protocol FILES (exact bytes).
    man_path = joinpath(OUTDIR, "stochastic_manifest_$(CASE).json")
    write_stochastic_manifest(man_path, case, process;
        reporting_horizon = REPORT, lookahead = LOOKAH, mode = MODE, stage_hours = 1.0,
        active_recourse_cost_per_mwh = DEFAULT_ACTIVE_RECOURSE_COST_PER_MWH, rho1 = 0.0, rho2 = 0.0,
        activation = string(stretchedsigmoid), safe_upper_margin = 1e-3,
        policy_layers = LAYERS, policy_combiner_layers = COMBINER, policy_seed = POLSEED,
        train_index_matrix_hash = index_matrix_hash(train_mat),
        eval_index_matrix_hash = index_matrix_hash(eval_mat),
        train_protocol_file_sha256 = bytes2hex(open(sha256, train_path)),
        eval_protocol_file_sha256 = bytes2hex(open(sha256, eval_path)),
        train_paths = NTRAIN, eval_paths = NEVAL)
    @printf("  artifacts → %s\n  process hash = %s\n", OUTDIR, process_hash(process))

    de    = build_battery_tsddr_de(case, process; reporting_horizon = REPORT,
                                   lookahead = LOOKAH, mode = MODE, stage_hours = 1.0)
    stage = build_battery_stage_problem(case, process; mode = MODE, stage_hours = 1.0)

    Random.seed!(POLSEED)                     # reproducible initialization
    policy = battery_reachable_policy(case, process; dt = 1.0,
                                      layers = LAYERS, combiner_layers = COMBINER)

    @printf("\nEvaluating INITIAL policy on %d held-out paired scenarios ...\n", NEVAL)
    ev0 = evaluate_paired(policy, stage, process, eval_mat; reporting_horizon = REPORT)
    @printf("  initial mean held-out physical cost = %.4f  (n_ok=%d, n_failed=%d, deficit=%.3g / surplus=%.3g MWh)\n",
            ev0.mean_reporting_physical_cost, ev0.n_ok, ev0.n_failed,
            ev0.total_active_deficit_energy_mwh, ev0.total_active_surplus_energy_mwh)

    @printf("\nTraining (%d batches × %d scenarios) ...\n", BATCHES, NTRAIN)
    tr = train_battery_tsddr(policy, de, process, train_mat;
                             num_batches = BATCHES, num_train_per_batch = NTRAIN,
                             optimizer = Flux.Adam(Float32(LR)),
                             madnlp_kwargs = (print_level = MadNLP.ERROR, tol = 1e-6))
    @printf("  solves: n_ok=%d / n_total=%d  (failed=%d)\n", tr.n_ok, tr.n_total, tr.n_failed)
    isempty(tr.failure_counts) || @printf("  failure_counts = %s\n", string(tr.failure_counts))

    @printf("\nEvaluating TRAINED policy on the SAME held-out scenarios ...\n")
    ev1 = evaluate_paired(policy, stage, process, eval_mat; reporting_horizon = REPORT,
                          keep_trajectories = true)
    @printf("  trained mean held-out physical cost = %.4f  (n_ok=%d, n_failed=%d, deficit=%.3g / surplus=%.3g MWh)\n",
            ev1.mean_reporting_physical_cost, ev1.n_ok, ev1.n_failed,
            ev1.total_active_deficit_energy_mwh, ev1.total_active_surplus_energy_mwh)

    ckpt = joinpath(OUTDIR, "checkpoint_$(CASE).jls")
    save_checkpoint(ckpt, policy, de; case = case, process = process,
                    extra = Dict("initial_mean_physical" => ev0.mean_reporting_physical_cost,
                                 "final_mean_physical" => ev1.mean_reporting_physical_cost,
                                 "manifest_path" => man_path,
                                 "eval_protocol_path" => eval_path))
    traj = joinpath(OUTDIR, "trajectory_$(CASE).json")
    write_trajectory(traj, ev1.trajectories;
                     meta = Dict("case" => CASE, "target_mode" => String(MODE),
                                 "reporting_horizon" => REPORT, "lookahead" => LOOKAH))

    Δ = ev0.mean_reporting_physical_cost - ev1.mean_reporting_physical_cost
    println("\n── Summary ─────────────────────────────────────────────")
    @printf("initial held-out physical cost : %.4f\n", ev0.mean_reporting_physical_cost)
    @printf("trained held-out physical cost : %.4f\n", ev1.mean_reporting_physical_cost)
    @printf("improvement (initial − trained): %.4f  (%.2f%%)\n",
            Δ, 100Δ / ev0.mean_reporting_physical_cost)
    @printf("manifest   : %s\ncheckpoint : %s\ntrajectory : %s\n", man_path, ckpt, traj)
    println("────────────────────────────────────────────────────────")
    return nothing
end

main()
