#!/usr/bin/env julia
# examples/end_to_end_gpu.jl
#
# End-to-end GPU demo:
#  - ExaModels model instantiated with CUDABackend()
#  - MadNLP solves with CUDSS via MadNLPGPU

using Random
using DecisionRulesExa
using ExaModels
using CUDA
using MadNLPGPU
using MadNLP
using Flux

if !CUDA.functional()
    error("CUDA is not functional on this machine. Check CUDA.jl installation.")
end

Random.seed!(1)
CUDA.allowscalar(false)

T = 16
nx = 1

prob = build_linear_tracking_problem(
    horizon = T,
    nx = nx,
    backend = CUDABackend(),
    slack_penalty = 10.0,
    u_bounds = (-2.0, 2.0),
)

policy = StateConditionedPolicy(nx, nx, nx, [64, 64])

sampler() = Float64.(0.1 .* randn(T * nx))

train_tsddr(
    policy,
    [1.0],
    prob,
    prob.p_x0,
    prob.p_target,
    prob.p_w,
    sampler;
    num_batches = 3,
    num_train_per_batch = 2,
    optimizer = Flux.Adam(1f-3),
    madnlp_kwargs = (
        print_level = MadNLP.ERROR,
        tol = 1e-4,
        linear_solver = CUDSSSolver,
    ),
)
