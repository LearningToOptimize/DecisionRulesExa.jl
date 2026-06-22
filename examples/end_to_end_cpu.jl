#!/usr/bin/env julia
# examples/end_to_end_cpu.jl

using Random
using LinearAlgebra
using DecisionRulesExa
using MadNLP
using Flux

Random.seed!(1)

T = 8
nx = 1

prob = build_linear_tracking_problem(
    horizon = T,
    nx = nx,
    backend = nothing,
    slack_penalty = 10.0,
    u_bounds = (-2.0, 2.0),
)

policy = StateConditionedPolicy(nx, nx, nx, [32, 32])

sampler() = Float64.(0.1 .* randn(T * nx))

train_tsddr(
    policy,
    [1.0],
    prob,
    prob.p_x0,
    prob.p_target,
    prob.p_w,
    sampler;
    num_batches = 5,
    num_train_per_batch = 2,
    optimizer = Flux.Adam(1f-3),
    madnlp_kwargs = (print_level = MadNLP.ERROR, tol = 1e-6),
)
