"""
    DecisionRulesExa

GPU-accelerated companion to DecisionRules.jl for Two-Stage Deep Decision
Rules (TS-DDR) training via ExaModels + MadNLP.

Provides the same TS-DDR workflow — policy predicts target states, NLP
projects onto the feasible set, dual multipliers feed the policy gradient —
but formulates subproblems as `ExaModels.ExaModel` instances solved by MadNLP.
This enables GPU-native training (CUDA via `CUDABackend()`) and exploits
MadNLP's warm-start capability for fast sequential solves.

Key types:
- [`DeterministicEquivalentProblem`](@ref): open-loop DE with explicit targets
- [`EmbeddedDeterministicEquivalentProblem`](@ref): DE with policy embedded via `VectorNonlinearOracle`
- [`StateConditionedPolicy`](@ref): stateful LSTM policy for sequential rollout
- [`MLPPolicy`](@ref): stateless MLP policy for full-horizon prediction
"""
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
include("embedded_deterministic_equivalent.jl")
include("policy.jl")
include("critic_control_variate.jl")
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
    ConstantStatePolicy,
    FixedOutputPolicy,
    bounded_state_policy,
    load_stateconditioned_policy!,

    # Embedded-NN deterministic equivalent
    EmbeddedDeterministicEquivalentProblem,
    build_embedded_deterministic_equivalent,
    invalidate_policy_cache!,

    # Training
    solve_succeeded,
    prepare_solve!,
    materialize_tangent,
    _all_finite_gradient,
    AbstractCriticControlVariate,
    AbstractCriticTrainingTarget,
    NoCriticControlVariate,
    DeterministicEquivalentCriticTarget,
    RolloutCriticTarget,
    ScalarCriticControlVariate,
    CriticSample,
    CriticReplayBuffer,
    default_critic_featurizer,
    critic_value,
    critic_xhat_gradient,
    critic_loss,
    update_critic!,
    critic_samples_from_evaluation,
    simulate_tsddr,
    train_tsddr,
    train_tsddr_embedded,

    # Stage-wise rollout evaluation
    rollout_tsddr,
    RolloutEvaluation

end # module
