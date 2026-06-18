#!/usr/bin/env julia
# examples/end_to_end_gpu.jl
#
# End-to-end GPU demo:
#  - ExaModels model instantiated with `CUDABackend()`
#  - MadNLP solves with cuDSS via MadNLPGPU

using Random
using DecisionRulesExa
using ExaModels
using CUDA
using MadNLPGPU
using Flux

if !CUDA.functional()
    error("CUDA is not functional on this machine. Check CUDA.jl installation.")
end

Random.seed!(1)
CUDA.allowscalar(false)

T = 16
nx = 1

# Build deterministic-equivalent NLP directly on the GPU
prob = build_linear_tracking_problem(
    horizon = T,
    nx = nx,
    backend = CUDABackend(),    # <- GPU instantiation
    slack_penalty = 10.0,
    u_bounds = (-2.0, 2.0),
)

# Policy on GPU
input_dim = nx + (T - 1) * nx
output_dim = T * nx
policy = MLPPolicy(input_dim, output_dim; hidden = (64, 64), act = tanh) |> gpu

# Solver cache on GPU (recommended)
cache = init_madnlp_cache(prob; linear_solver = CUDSSSolver)

sampler(k) = begin
    x0 = cu([1.0])
    w  = cu(0.1 .* randn(T - 1))
    return x0, w
end

hist = train_tsddr!(
    policy, prob;
    n_iters = 3,
    sampler = sampler,
    η = 5e-3,
    cache = cache,
    # MadNLP options (looser tolerances often make sense on GPU)
    tol = 1e-4,
    max_iter = 200,
    linear_solver = CUDSSSolver,
)

last = hist[end]
println("\nLast objective: ", last.result.objective)
println("First 5 target multipliers λ: ", Array(last.lambda[1:min(5, length(last.lambda))]))
