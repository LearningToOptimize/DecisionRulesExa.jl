"""
    DecisionRulesExa

GPU-accelerated companion to DecisionRules.jl for Two-Stage Deep Decision Rules
(TS-DDR) training with ExaModels and MadNLP.

DecisionRulesExa implements the same target-projection workflow as
DecisionRules.jl:

1. a policy predicts target states,
2. an NLP projects those targets onto the feasible set, and
3. target-constraint multipliers provide the policy-gradient signal.

The package formulates inner optimization problems as `ExaModels.ExaModel`
instances solved by MadNLP, enabling GPU-native solves and warm-started repeated
training solves.

# Main Types
- [`DeterministicEquivalentProblem`](@ref): deterministic equivalent with
  explicit target parameters.
- [`EmbeddedDeterministicEquivalentProblem`](@ref): deterministic equivalent
  whose target policy is embedded with `VectorNonlinearOracle`.
- [`StateConditionedPolicy`](@ref): recurrent policy for sequential target
  rollout.
- [`MLPPolicy`](@ref): stateless policy for full-horizon target prediction.

# Main Training APIs
- [`train_tsddr`](@ref): open-loop target-parameter training.
- [`train_tsddr_embedded`](@ref): embedded-policy training.
- [`rollout_tsddr`](@ref): stage-wise deployment-style evaluation.
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
