# critic_control_variate.jl
#
# Optional scalar critic / control-variate support for TS-DDR training.

abstract type AbstractCriticControlVariate end

abstract type AbstractCriticTrainingTarget end

"""
    DeterministicEquivalentCriticTarget()

Train critic value targets from the full deterministic-equivalent objective.
This is useful for ablations and for pure DE control-variate experiments.
"""
struct DeterministicEquivalentCriticTarget <: AbstractCriticTrainingTarget end

"""
    RolloutCriticTarget(stage_problem; kwargs...)

Train critic value targets from stage-wise rollout evaluation. This is the
preferred target when the critic is meant to guide convergence of the deployed
rollout objective rather than the deterministic-equivalent surrogate.

Required keyword callbacks match `rollout_tsddr`:
- `set_stage_parameters!`
- `realized_state`

By default `policy_state = :target`, matching the differentiable target
recurrence used by the actor. Set `policy_state = :realized` to train on
closed-loop realized-state rollout targets.

By default `objective_value = :objective`, so critic value targets include the
same target-penalty contribution that appears in the dual actor signal. Set
`objective_value = :objective_no_target_penalty` to train on the rollout
objective with target-slack penalties removed.
"""
struct RolloutCriticTarget{S,R,O,M} <: AbstractCriticTrainingTarget
    stage_problem
    horizon::Int
    n_uncertainty::Int
    set_stage_parameters!::S
    realized_state::R
    objective_no_target_penalty::O
    madnlp_kwargs::M
    warmstart::Bool
    policy_state::Symbol
    reuse_solver::Bool
    objective_value::Symbol
end

function RolloutCriticTarget(
    stage_problem;
    horizon::Int,
    n_uncertainty::Int,
    set_stage_parameters!::Function,
    realized_state::Function,
    objective_no_target_penalty::Function = (prob, result) -> result.objective,
    madnlp_kwargs = NamedTuple(),
    warmstart::Bool = true,
    policy_state::Symbol = :target,
    reuse_solver::Bool = false,
    objective_value::Symbol = :objective,
)
    policy_state in (:target, :realized) ||
        error("policy_state must be :target or :realized")
    objective_value in (:objective, :objective_no_target_penalty) ||
        error("objective_value must be :objective or :objective_no_target_penalty")
    return RolloutCriticTarget(
        stage_problem,
        horizon,
        n_uncertainty,
        set_stage_parameters!,
        realized_state,
        objective_no_target_penalty,
        madnlp_kwargs,
        warmstart,
        policy_state,
        reuse_solver,
        objective_value,
    )
end

"""
    NoCriticControlVariate()

Default no-op critic configuration. Passing this to `train_tsddr` recovers the
original dual-multiplier actor update.
"""
struct NoCriticControlVariate <: AbstractCriticControlVariate end

"""
    ScalarCriticControlVariate(critic; featurizer=default_critic_featurizer,
                               value_loss_weight=0.1,
                               gradient_loss_weight=1.0)

Wrap a scalar Flux-compatible critic `C(w, xhat)` for optional TS-DDR
control-variate training. The critic is called as `critic(features)`, where
`features = featurizer(initial_state, uncertainty, xhat)`.

The critic loss is

    value_loss_weight * mse(C, objective)
  + gradient_loss_weight * mse(gradient(xhat -> C, xhat), target_multipliers)

Either loss weight may be zero.
"""
struct ScalarCriticControlVariate{C,F} <: AbstractCriticControlVariate
    critic::C
    featurizer::F
    value_loss_weight::Float64
    gradient_loss_weight::Float64
end

function ScalarCriticControlVariate(
    critic;
    featurizer = default_critic_featurizer,
    value_loss_weight::Real = 0.1,
    gradient_loss_weight::Real = 1.0,
)
    value_loss_weight >= 0 || error("value_loss_weight must be nonnegative")
    gradient_loss_weight >= 0 || error("gradient_loss_weight must be nonnegative")
    return ScalarCriticControlVariate(
        critic,
        featurizer,
        Float64(value_loss_weight),
        Float64(gradient_loss_weight),
    )
end

"""
    CriticSample(initial_state, uncertainty, xhat, objective_value,
                 target_multipliers; metadata=nothing)

Training sample for a scalar critic. Samples are produced from already-solved
TS-DDR scenarios and do not require additional optimization solves.
"""
struct CriticSample{I,W,X,L,M}
    initial_state::I
    uncertainty::W
    xhat::X
    objective_value::Float64
    target_multipliers::L
    metadata::M
end

function CriticSample(
    initial_state,
    uncertainty,
    xhat,
    objective_value::Real,
    target_multipliers;
    metadata = nothing,
)
    return CriticSample(
        initial_state,
        uncertainty,
        xhat,
        Float64(objective_value),
        target_multipliers,
        metadata,
    )
end

mutable struct CriticReplayBuffer{S}
    samples::Vector{S}
    max_size::Int
end

CriticReplayBuffer(max_size::Integer) =
    CriticReplayBuffer{Any}(Any[], max(0, Int(max_size)))

function push_critic_sample!(buffer::CriticReplayBuffer, sample::CriticSample)
    buffer.max_size == 0 && return buffer
    push!(buffer.samples, sample)
    overflow = length(buffer.samples) - buffer.max_size
    overflow > 0 && deleteat!(buffer.samples, 1:overflow)
    return buffer
end

function push_critic_samples!(buffer::CriticReplayBuffer, samples)
    for sample in samples
        push_critic_sample!(buffer, sample)
    end
    return buffer
end

"""
    default_critic_featurizer(initial_state, uncertainty, xhat)

Default critic featurizer: concatenate flattened initial state, uncertainty, and
policy target trajectory.
"""
default_critic_featurizer(initial_state, uncertainty, xhat) =
    vcat(vec(initial_state), vec(uncertainty), vec(xhat))

_scalar_output(y::Number) = y
_scalar_output(y::AbstractArray) = begin
    length(y) == 1 || error("critic must return a scalar or length-1 array, got length $(length(y))")
    return only(vec(y))
end

function _critic_value(critic, featurizer, initial_state, uncertainty, xhat)
    features = featurizer(initial_state, uncertainty, xhat)
    return _scalar_output(critic(features))
end

"""
    critic_value(control_variate, initial_state, uncertainty, xhat)

Evaluate the scalar critic on one scenario.
"""
critic_value(
    cv::ScalarCriticControlVariate,
    initial_state,
    uncertainty,
    xhat,
) = _critic_value(cv.critic, cv.featurizer, initial_state, uncertainty, xhat)

"""
    critic_xhat_gradient(control_variate, initial_state, uncertainty, xhat)

Return `gradient(xhat -> C(initial_state, uncertainty, xhat), xhat)` and check
that it has the same shape as `xhat`.
"""
function critic_xhat_gradient(
    cv::ScalarCriticControlVariate,
    initial_state,
    uncertainty,
    xhat,
)
    gx = Zygote.gradient(x -> critic_value(cv, initial_state, uncertainty, x), xhat)[1]
    gx = gx === nothing ? zero(xhat) : gx
    size(gx) == size(xhat) ||
        error("critic xhat gradient shape $(size(gx)) does not match xhat shape $(size(xhat))")
    return gx
end

function _check_critic_sample_shapes(sample::CriticSample, grad_xhat = nothing)
    size(sample.target_multipliers) == size(sample.xhat) ||
        error("target_multipliers shape $(size(sample.target_multipliers)) does not match xhat shape $(size(sample.xhat))")
    if grad_xhat !== nothing
        size(grad_xhat) == size(sample.xhat) ||
            error("critic xhat gradient shape $(size(grad_xhat)) does not match xhat shape $(size(sample.xhat))")
    end
    return true
end

function _critic_loss_with(
    critic,
    cv::ScalarCriticControlVariate,
    samples;
    value_loss_weight::Real = cv.value_loss_weight,
    gradient_loss_weight::Real = cv.gradient_loss_weight,
)
    isempty(samples) && return 0.0
    value_w = Float64(value_loss_weight)
    grad_w = Float64(gradient_loss_weight)
    value_w >= 0 || error("value_loss_weight must be nonnegative")
    grad_w >= 0 || error("gradient_loss_weight must be nonnegative")

    total = 0.0
    for sample in samples
        _check_critic_sample_shapes(sample)
        if value_w > 0
            pred = _critic_value(
                critic,
                cv.featurizer,
                sample.initial_state,
                sample.uncertainty,
                sample.xhat,
            )
            target = convert(typeof(pred), sample.objective_value)
            total = total + value_w * abs2(pred - target)
        end
        if grad_w > 0
            gx = Zygote.gradient(sample.xhat) do x
                _critic_value(critic, cv.featurizer, sample.initial_state, sample.uncertainty, x)
            end[1]
            gx = gx === nothing ? zero(sample.xhat) : gx
            _check_critic_sample_shapes(sample, gx)
            total = total + grad_w * sum(abs2, gx .- sample.target_multipliers) / length(sample.xhat)
        end
    end
    return total / length(samples)
end

"""
    critic_loss(control_variate, samples; value_loss_weight, gradient_loss_weight)

Compute the scalar critic loss on a collection of `CriticSample`s.
"""
critic_loss(cv::ScalarCriticControlVariate, samples; kwargs...) =
    _critic_loss_with(cv.critic, cv, samples; kwargs...)

function _critic_minibatch(samples, batch_size)
    n = length(samples)
    n == 0 && return samples
    if batch_size === nothing || batch_size >= n
        return samples
    end
    idx = rand(1:n, Int(batch_size))
    return samples[idx]
end

"""
    update_critic!(opt_state, control_variate, samples; batch_size=nothing)

Run one critic optimizer step and return the numeric loss. Only critic
parameters are updated.
"""
function update_critic!(
    opt_state,
    cv::ScalarCriticControlVariate,
    samples;
    batch_size = nothing,
)
    batch = _critic_minibatch(samples, batch_size)
    isempty(batch) && return NaN
    gs = Zygote.gradient(cv.critic) do critic
        _critic_loss_with(critic, cv, batch)
    end
    grad = materialize_tangent(gs[1])
    if grad !== nothing && _all_finite_gradient(grad)
        Flux.update!(opt_state, cv.critic, grad)
    end
    return Float64(critic_loss(cv, batch))
end
