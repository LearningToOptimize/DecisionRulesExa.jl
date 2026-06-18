module DecisionRulesExa

using ExaModels
using MadNLP
using NLPModels
using LinearAlgebra
using Random

using Flux
using Zygote
using ChainRulesCore

include("utils.jl")
include("deterministic_equivalent.jl")
include("policy.jl")
include("training.jl")
include("rollout.jl")

export
    # Deterministic-equivalent problem
    DeterministicEquivalentProblem,
    MadNLPCache,
    build_deterministic_equivalent,
    build_linear_tracking_problem,
    set_x0!,
    set_uncertainty!,
    set_targets!,
    init_madnlp_cache,
    solve!,
    target_multipliers,
    solution_components,

    # Index helpers (needed when writing custom dynamics_eq / stage_cost)
    x_index,
    u_index,
    w_index,

    # Policies
    MLPPolicy,
    StateConditionedPolicy,

    # Training
    solve_succeeded,
    materialize_tangent,
    _all_finite_gradient,
    simulate_tsddr,
    train_tsddr,

    # Stage-wise rollout evaluation
    rollout_tsddr,
    RolloutEvaluation

end # module
