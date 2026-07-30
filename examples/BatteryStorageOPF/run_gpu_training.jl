#!/usr/bin/env julia
# run_gpu_training.jl
#
# GPU TS-DDR training smoke for the battery-storage example (Phase-2 prompt §5),
# using the existing DecisionRulesExa + MadNLPGPU path: the ExaModels model is
# built with a CUDA backend and MadNLP solves it with the CUDSS GPU linear solver.
#
# Run on a GPU node (see the GPU sbatch recipe in the README):
#   module load julia
#   julia --pkgimages=no --project=. run_gpu_training.jl
#
# Requires a functional CUDA device; it errors clearly otherwise. Same defaults /
# env-var overrides as run_tiny_training.jl.

include(joinpath(@__DIR__, "src", "BatteryStorageOPF.jl"))
using .BatteryStorageOPF
using DecisionRulesExa
using ExaModels
using CUDA
using MadNLPGPU
using MadNLP
using Flux
using Random
using Printf

CUDA.functional() || error("CUDA is not functional on this machine; run on a GPU node.")
CUDA.allowscalar(false)

const CASE    = get(ENV, "BAT_CASE", "case300_ieee")
const NBAT    = parse(Int, get(ENV, "BAT_NBAT", "20"))
const SEED    = parse(Int, get(ENV, "BAT_SEED", "20260722"))
const NREGION = parse(Int, get(ENV, "BAT_NREGION", "3"))
const REPORT  = parse(Int, get(ENV, "BAT_REPORT", "4"))
const LOOKAH  = parse(Int, get(ENV, "BAT_LOOKAHEAD", "1"))
const MODE    = Symbol(get(ENV, "BAT_MODE", "strict"))
const BATCHES = parse(Int, get(ENV, "BAT_BATCHES", "10"))
const NTRAIN  = parse(Int, get(ENV, "BAT_NTRAIN", "4"))
const POLSEED = parse(Int, get(ENV, "BAT_POLICY_SEED", "1234"))

function main()
    T = REPORT + LOOKAH
    @printf("GPU device: %s\n", CUDA.name(CUDA.device()))
    @printf("Case %s: %d batteries, %d regions, mode=%s, T=%d\n", CASE, NBAT, NREGION, MODE, T)

    case = make_battery_case(CASE; number_of_batteries = NBAT, seed = SEED)
    process = make_load_process(case; nregion = NREGION, period = max(T, 4))
    train_mat = scenario_index_matrix(process, T, NTRAIN; seed = process.train_seed)

    # Build the target-constrained DE on the GPU (CUDABackend); the policy lives
    # on the GPU too so the rollout produces device targets.
    de = build_battery_tsddr_de(case, process; reporting_horizon = REPORT, lookahead = LOOKAH,
                                mode = MODE, stage_hours = 1.0, backend = CUDABackend())
    Random.seed!(POLSEED)
    policy = battery_reachable_policy(case, process; dt = 1.0, layers = [64, 64],
                                      combiner_layers = [64]) |> Flux.gpu
    x0 = policy_initial_state(case) |> Flux.gpu

    sampler, _ = make_replay_sampler(process, train_mat)
    @printf("\nTraining on GPU (%d batches × %d scenarios) ...\n", BATCHES, NTRAIN)
    train_tsddr(policy, x0, de, de.p_x0, de.p_target, de.p_w, sampler;
                num_batches = BATCHES, num_train_per_batch = NTRAIN,
                optimizer = Flux.Adam(1f-3),
                madnlp_kwargs = (print_level = MadNLP.ERROR, tol = 1e-4,
                                 linear_solver = MadNLPGPU.CUDSSSolver))
    println("GPU training smoke completed.")
    return nothing
end

main()
