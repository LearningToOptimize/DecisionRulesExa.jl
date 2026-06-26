# DecisionRulesExa.jl

GPU-accelerated training of parametric decision rules through multi-stage optimization, using [ExaModels.jl](https://github.com/exanauts/ExaModels.jl) and solvers like [MadNLP.jl](https://github.com/MadNLP/MadNLP.jl).

This package replicates the core functionality of [DecisionRules.jl](https://github.com/LearningToOptimize/DecisionRules.jl) but replaces [JuMP](https://github.com/jump-dev/JuMP.jl) with ExaModels for the optimization backend. JuMP relies on MathOptInterface operations that are not differentiable on GPU and cannot exploit GPU-native sparse linear solvers. ExaModels provides a fully SIMD-compatible algebraic modeling layer whose parameters can be updated at runtime via `ExaModels.set_parameter!`, enabling efficient repeated solves on GPU with [MadNLPGPU](https://github.com/MadNLP/MadNLP.jl) (CUDSS-backed interior point).

## Motivation

In the **Two-Stage Deep Decision Rule (TS-DDR)** framework, a neural-network policy predicts target state trajectories that are projected onto the feasible set by solving a parametric NLP (the deterministic equivalent). Training uses the envelope theorem: dual multipliers on the target constraints give the policy gradient without differentiating through the solver.

The inner NLP solve is the bottleneck. By formulating it in ExaModels and solving with MadNLP + CUDSS on GPU, DecisionRulesExa.jl achieves significant speedups over the CPU JuMP-based workflow in DecisionRules.jl, especially for large-scale problems (AC-OPF, multi-stage hydro scheduling).

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/LearningToOptimize/DecisionRulesExa.jl.git")
```

## Quick start

```julia
using DecisionRulesExa
using ExaModels, MadNLP, Flux, Random

Random.seed!(1)

T  = 8   # horizon
nx = 1   # state dimension

# Build a simple linear-tracking NLP on CPU
prob = build_linear_tracking_problem(
    horizon       = T,
    nx            = nx,
    backend       = nothing,       # CPU (use CUDABackend() for GPU)
    slack_penalty = 10.0,
    u_bounds      = (-2.0, 2.0),
)

# LSTM policy: input = [w_t ; x_{t-1}], called once per stage
policy = StateConditionedPolicy(nx, nx, nx, [64, 64])

# Uncertainty sampler (returns flat vector of length T*nw)
sampler() = Float32.(0.1 .* randn(T * nx))

# Train with TS-DDR policy gradient
train_tsddr(
    policy,
    Float32.([1.0]),        # initial state
    prob,
    prob.p_x0,
    prob.p_target,
    prob.p_w,
    sampler;
    num_batches         = 100,
    num_train_per_batch = 4,
    optimizer           = Flux.Adam(1f-3),
    madnlp_kwargs       = (print_level = MadNLP.ERROR, tol = 1e-6),
)
```

For GPU, replace `backend = nothing` with `backend = CUDABackend()` and add `linear_solver = CUDSSSolver` to `madnlp_kwargs`.

## What you need to provide

DecisionRulesExa.jl is **model-first**: you describe your multi-stage NLP in ExaModels, then the package handles simulation and training.

For a custom problem you need:

- **An ExaModels deterministic-equivalent NLP** with parametric initial state, uncertainty trajectory, and target trajectory. Target constraints must be added last so their multipliers form a contiguous slice of `result.multipliers`.
- **An uncertainty sampler** `() -> w_flat` returning a flat `Float32`/`Float64` vector of length `T * nw`.
- **A Flux policy** (LSTM or MLP) mapping `(w_t, x_{t-1})` to target `x_t` at each stage.

The package provides `build_deterministic_equivalent` for generic problems and `build_linear_tracking_problem` as a ready-made demo. For domain-specific models (power systems, robotics), build the ExaModels NLP directly — see `examples/HydroPowerModels/` for a complete AC-OPF example.

## Strict embedded target equality

The usual TS-DDR deterministic equivalent uses slack-penalized target
constraints,

```text
x_t - pi_theta(w_t, x_{t-1}) = delta_t,
objective += rho * penalty(delta_t).
```

This is the right default for open-loop target trajectories and for cases where
the policy can request states that are not reachable from the previous realized
state. The target multipliers are then gradients of the penalized projection
problem, so their quality depends on the penalty calibration.

For embedded closed-loop policies, a stricter formulation is possible when the
policy output is guaranteed to lie in a one-stage reachable state set:

```text
x_t = pi_theta(w_t, x_{t-1}),      pi_theta(w_t, x_{t-1}) in R(w_t, x_{t-1}).
```

In that case the deterministic equivalent does not need target slack variables
or target penalties. The multiplier on the equality is the local envelope
sensitivity of the true stage problem with respect to the policy-imposed next
state, not the sensitivity of a penalized approximation. This is useful when:

- the policy is embedded stage by stage and receives the realized previous
  state;
- users can define a differentiable or piecewise differentiable map into a
  subset of the one-stage reachable set;
- total recourse is guaranteed by the model for every state produced by that
  map.

Do not use strict equality for a generic open-loop target policy unless the
target generator is proven reachable at every stage. For unreachable targets,
the slack-penalty formulation is the robust fallback.

## Parallel GPU solves

When training samples are independent, multiple NLP instances can be solved concurrently on the same GPU. Pass a `problem_pool` of independent ExaModels problem copies to `train_tsddr`:

```julia
pool = [(prob, prob.p_x0, prob.p_target, prob.p_w)]
for _ in 2:num_workers
    p = build_my_problem(...)
    push!(pool, (p, p.p_x0, p.p_target, p.p_w))
end

train_tsddr(policy, x0, prob, ..., sampler;
    problem_pool = pool,
    num_train_per_batch = num_workers,
)
```

Each pool entry gets its own MadNLP solver on a dedicated thread, with CUDA handles properly bound.

## Optional Critic Control Variates

`train_tsddr` can optionally train a scalar critic `C(w, xhat)`. The critic does
not replace the deterministic-equivalent solve in the default
`:control_variate` mode: solved target-constraint multipliers remain the
actor's primary local sensitivity signal. Instead, the critic supplies a learned
rollout-value guide and optional control variate.

For critic fitting, the preferred target is the stage-wise rollout objective via
`RolloutCriticTarget`, with `policy_state = :target` by default. This matches the
differentiable target recurrence used by the actor while evaluating the true
stage-by-stage objective. Set `policy_state = :realized` to train the critic on
closed-loop realized-state rollout labels. For ablations, use
`DeterministicEquivalentCriticTarget()` or `critic_training_target =
:deterministic_equivalent` to fit the deterministic-equivalent objective instead.

The default `control_variate = NoCriticControlVariate()` recovers the original
dual-only behavior. A scalar critic can be attached with:

```julia
input_dim = length(x0) + 2 * T * nx
critic = Chain(
    Dense(input_dim => 128, tanh),
    Dense(128 => 128, tanh),
    Dense(128 => 1),
)

cv = ScalarCriticControlVariate(
    critic;
    featurizer = default_critic_featurizer,
    value_loss_weight = 1.0,
    gradient_loss_weight = 0.0,
)

critic_target = RolloutCriticTarget(
    stage_problem;
    horizon = T,
    n_uncertainty = nx,
    set_stage_parameters! = set_stage_parameters!,
    realized_state = realized_state,
    objective_no_target_penalty = objective_no_target_penalty,
    policy_state = :target,
    objective_value = :objective_no_target_penalty,
)

train_tsddr(
    policy, x0, prob, prob.p_x0, prob.p_target, prob.p_w, sampler;
    control_variate = cv,
    critic_training_target = critic_target,
    critic_rollout_samples_per_batch = 1,
    actor_gradient_mode = :control_variate,
    critic_cv_weight = 1.0,
    num_cheap_critic_samples_per_batch = 32,
    critic_updates_per_batch = 1,
    critic_optimizer = Flux.Adam(1f-3),
)
```

The critic loss combines value matching against the selected target objective and
optional gradient matching against target multipliers:

```julia
value_loss_weight * mse(C(w, xhat), objective) +
gradient_loss_weight * mse(gradient(xhat -> C(w, xhat), xhat), lambda)
```

Set either weight to zero for objective-only or gradient-only critic training.
For rollout targets, objective-only critic training is usually the clean default,
because DE target multipliers are not exact gradients of the realized rollout
objective. If objectives and multipliers have very different magnitudes, prefer a
custom featurizer and tuned loss weights; the Hydro example normalizes volumes
and inflows before critic evaluation.

Actor modes:

- `:control_variate`: subtracts `critic_cv_weight * gradient_xhat(C)` from the
  solved dual signal and adds the critic actor gradient back on solved or cheap
  rollout samples. This is the recommended mode when dual multipliers are
  reliable. `critic_cv_weight = 0.0` recovers dual-only updates.
- `:surrogate`: uses a practical hybrid of solved dual gradients and critic
  actor gradients, controlled by `dual_actor_weight` and `critic_actor_weight`.
  This is useful when raw dual/subgradient signals are empirically noisy or
  unstable, but it is no longer a pure unbiased control-variate estimator.

`num_cheap_critic_samples_per_batch` draws additional uncertainty samples,
rolls out the current policy, and evaluates the critic actor term without any
extra MadNLP or ExaModels solve.

## Rollout evaluation

`RolloutEvaluation` evaluates the policy in deployment semantics (stage-by-stage sequential solves) on held-out scenarios:

```julia
eval = RolloutEvaluation(
    stage_problem, x0, eval_scenarios;
    horizon              = T,
    n_uncertainty        = nw,
    set_stage_parameters! = my_stage_setter!,
    realized_state       = my_realized_state,
    stride               = 25,
    policy_state         = :realized,
)
```

Supports parallel evaluation across a pool of stage problems via `stage_problem_pool`, and dynamic scenario count via `active_scenarios`.

## Relationship to DecisionRules.jl

[DecisionRules.jl](https://github.com/LearningToOptimize/DecisionRules.jl) implements the same TS-DDR algorithm using JuMP + DiffOpt for CPU-based training. It supports stage-wise decomposition, multiple shooting, and integer strategies that are not yet ported here.

DecisionRulesExa.jl focuses on the **deterministic-equivalent** training mode with GPU acceleration. Choose this package when:
- Your NLP is large enough that GPU acceleration matters (e.g., AC-OPF with hundreds of buses)
- You need to run many training samples per gradient step
- You want to leverage CUDA-native sparse solvers (CUDSS)

Choose DecisionRules.jl when:
- You need stage-wise or multiple-shooting decomposition
- Your problem is naturally expressed in JuMP
- You need DiffOpt-based sensitivity computation
- You want CPU-only deployment

## Examples

- [`examples/end_to_end_cpu.jl`](examples/end_to_end_cpu.jl) — minimal CPU demo with a linear tracking problem
- [`examples/end_to_end_gpu.jl`](examples/end_to_end_gpu.jl) — same demo on GPU with CUDSS
- [`examples/HydroPowerModels/`](examples/HydroPowerModels/) — full multi-stage hydrothermal scheduling with DC and AC OPF

## Citation

If you use this package in academic work, please cite:

```bibtex
@article{rosemberg2024efficiently,
  title={Efficiently Training Deep-Learning Parametric Policies using Lagrangian Duality},
  author={Rosemberg, Andrew and Street, Alexandre and Vallad{\~a}o, Davi M and Van Hentenryck, Pascal},
  journal={arXiv preprint arXiv:2405.14973},
  year={2024}
}
```

## License

MIT. See [LICENSE](LICENSE).
