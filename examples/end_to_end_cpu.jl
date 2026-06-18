#!/usr/bin/env julia
# examples/end_to_end_cpu.jl

using Random
using LinearAlgebra
using DecisionRulesExa

Random.seed!(1)

# Problem dimensions
T = 8          # horizon
nx = 1         # state dimension (demo uses scalar)
# controls u are also dimension nx in build_linear_tracking_problem()

# Build deterministic-equivalent NLP on CPU
prob = build_linear_tracking_problem(
    horizon = T,
    nx = nx,
    backend = nothing,         # CPU
    slack_penalty = 10.0,
    u_bounds = (-2.0, 2.0),
)

# Policy: input = [x0 ; w(1:T-1)], output = xhat(1:T)
input_dim = nx + (T - 1) * nx
output_dim = T * nx
policy = MLPPolicy(input_dim, output_dim; hidden = (32, 32), act = tanh)

# Optional solver cache (recommended if you solve many times)
cache = init_madnlp_cache(prob)

# Scenario sampler
sampler(k) = begin
    x0 = [1.0]
    w  = 0.1 .* randn(T - 1)   # (T-1)*nx
    return x0, w
end

# Run a few TS-DDR iterations
hist = train_tsddr!(
    policy, prob;
    n_iters = 5,
    sampler = sampler,
    η = 1e-2,
    cache = cache,
    tol = 1e-6,
    max_iter = 200,
)

# Print last objective and a few duals
last = hist[end]
println("\nLast objective: ", last.result.objective)
println("First 5 target multipliers λ: ", collect(last.lambda[1:min(5, length(last.lambda))]))
