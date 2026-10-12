# critic_control_variate.jl
#
# Optional scalar critic / control-variate support for TS-DDR training.

abstract type AbstractCriticControlVariate end

abstract type AbstractCriticTrainingTarget end

"""
    DeterministicEquivalentCriticTarget()

Select critic value targets from the full deterministic-equivalent training
objective.

# Returns
- `DeterministicEquivalentCriticTarget`: a target selector for deterministic-
  equivalent critic supervision.

# Notes
Use this target for ablations or for control-variate experiments tied to the
training surrogate rather than to deployed rollout performance.
"""
struct DeterministicEquivalentCriticTarget <: AbstractCriticTrainingTarget end

"""
    RolloutCriticTarget(
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
        state_bounds = nothing,
        project_state = nothing,
        retry_on_failure::Bool = true,
    ) -> RolloutCriticTarget

Select critic value targets from stage-wise rollout evaluation.

# Arguments
- `stage_problem`: single-stage optimization problem template used by
  [`rollout_tsddr`](@ref).

# Keywords
- `horizon::Int`: number of rollout stages.
- `n_uncertainty::Int`: dimension of each per-stage uncertainty slice.
- `set_stage_parameters!::Function`: callback that writes state,
  uncertainty, target, and stage index into `stage_problem`.
- `realized_state::Function`: callback that extracts the realized next state
  from a solved stage.
- `objective_no_target_penalty::Function`: callback returning the stage
  objective with target-tracking penalties removed.
- `madnlp_kwargs`: keyword arguments forwarded to the MadNLP solver wrapper.
- `warmstart::Bool`: whether rollout solves should warm-start from previous
  stage information.
- `policy_state::Symbol`: `:target` trains against the differentiable target
  recurrence; `:realized` trains against closed-loop realized states.
- `reuse_solver::Bool`: whether rollout evaluation may reuse one solver
  object across stages.
- `objective_value::Symbol`: `:objective` uses the full rollout objective;
  `:objective_no_target_penalty` removes target-slack penalties.
- `state_bounds`: optional `(lower, upper)` projection bounds for realized
  states.
- `project_state`: optional custom projection callback for realized states.
- `retry_on_failure::Bool`: whether failed stage solves should be retried with
  a cold-start solver.

# Returns
- `RolloutCriticTarget`: a target selector carrying the rollout callbacks and
  options.

# Throws
- Throws an error if `policy_state` is not `:target` or `:realized`.
- Throws an error if `objective_value` is not `:objective` or
  `:objective_no_target_penalty`.

# Notes
This is the preferred target when the critic is meant to guide convergence of
the deployed rollout objective rather than the deterministic-equivalent
surrogate.
"""
struct RolloutCriticTarget{S,R,O,M,B,P} <: AbstractCriticTrainingTarget
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
    state_bounds::B
    project_state::P
    retry_on_failure::Bool
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
    state_bounds = nothing,
    project_state = nothing,
    retry_on_failure::Bool = true,
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
        state_bounds,
        project_state,
        retry_on_failure,
    )
end

"""
    NoCriticControlVariate()

Construct the no-op critic configuration.

# Returns
- `NoCriticControlVariate`: a sentinel that disables critic/control-variate
  terms.

# Notes
Passing this value to `train_tsddr` recovers the original dual-multiplier actor
update.
"""
struct NoCriticControlVariate <: AbstractCriticControlVariate end

"""
    ScalarCriticControlVariate(
        critic;
        featurizer = default_critic_featurizer,
        value_loss_weight::Real = 0.1,
        gradient_loss_weight::Real = 1.0,
    ) -> ScalarCriticControlVariate

Wrap a scalar Flux-compatible critic for optional TS-DDR control-variate
training.

# Arguments
- `critic`: callable scalar model evaluated as `critic(features)`.

# Keywords
- `featurizer`: callable
  `featurizer(initial_state, uncertainty, xhat) -> features`.
- `value_loss_weight::Real`: nonnegative weight on objective-value matching.
- `gradient_loss_weight::Real`: nonnegative weight on target-gradient
  matching.

# Returns
- `ScalarCriticControlVariate`: critic configuration with loss weights stored
  as `Float64`.

# Throws
- Throws an error if either loss weight is negative.

# Notes
For each [`CriticSample`](@ref), the critic loss is

```math
w_v |C(f) - J|^2
+ w_g \\frac{1}{n}\\|\\nabla_{\\hat{x}} C(f) - \\lambda_{\\hat{x}}\\|_2^2,
```

where `f = featurizer(initial_state, uncertainty, xhat)`, `J` is the scalar
objective target, and ``\\lambda_{\\hat{x}}`` is the target multiplier array.
Either weight may be zero.
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
    CriticSample(
        initial_state,
        uncertainty,
        xhat,
        objective_value::Real,
        target_multipliers;
        metadata = nothing,
    ) -> CriticSample

Store one already-solved TS-DDR scenario as scalar-critic supervision.

# Arguments
- `initial_state`: initial state used for the scenario.
- `uncertainty`: scenario uncertainty trajectory.
- `xhat`: policy target trajectory.
- `objective_value::Real`: scalar value target for the critic.
- `target_multipliers`: multiplier-like target with the same shape as `xhat`.

# Keywords
- `metadata`: optional payload retained with the sample.

# Returns
- `CriticSample`: sample with `objective_value` converted to `Float64`.

# Notes
Creating a `CriticSample` does not run any optimization solve; samples are
intended to be built from existing training or rollout results.
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

"""
    CriticReplayBuffer(max_size::Integer) -> CriticReplayBuffer

Construct a fixed-capacity FIFO replay buffer for [`CriticSample`](@ref)s.

# Arguments
- `max_size::Integer`: maximum number of samples retained; negative values are
  clamped to zero.

# Returns
- `CriticReplayBuffer`: empty replay buffer with capacity `max(0, max_size)`.

# Notes
A capacity of zero disables buffering, so push operations become no-ops.
"""
mutable struct CriticReplayBuffer{S}
    samples::Vector{S}
    max_size::Int
end

CriticReplayBuffer(max_size::Integer) =
    CriticReplayBuffer{Any}(Any[], max(0, Int(max_size)))

"""
    push_critic_sample!(buffer, sample) -> buffer

Append one [`CriticSample`](@ref) to the buffer, evicting the oldest sample if
the buffer is at capacity.

# Arguments
- `buffer::CriticReplayBuffer`: replay buffer to mutate.
- `sample::CriticSample`: sample to append.

# Returns
- `buffer`: the same buffer object, after optional insertion and eviction.

# Notes
If `buffer.max_size == 0`, the function returns without storing `sample`.
"""
function push_critic_sample!(buffer::CriticReplayBuffer, sample::CriticSample)
    buffer.max_size == 0 && return buffer
    push!(buffer.samples, sample)
    overflow = length(buffer.samples) - buffer.max_size
    overflow > 0 && deleteat!(buffer.samples, 1:overflow)
    return buffer
end

"""
    push_critic_samples!(buffer, samples) -> buffer

Append multiple [`CriticSample`](@ref)s to the buffer in order.

# Arguments
- `buffer::CriticReplayBuffer`: replay buffer to mutate.
- `samples`: iterable of [`CriticSample`](@ref) values.

# Returns
- `buffer`: the same buffer after appending the supplied samples.
"""
function push_critic_samples!(buffer::CriticReplayBuffer, samples)
    for sample in samples
        push_critic_sample!(buffer, sample)
    end
    return buffer
end

"""
    default_critic_featurizer(initial_state, uncertainty, xhat) -> AbstractVector

Concatenate flattened critic inputs into one feature vector.

# Arguments
- `initial_state`: initial state for the scenario.
- `uncertainty`: uncertainty trajectory for the scenario.
- `xhat`: policy target trajectory for the scenario.

# Returns
- `AbstractVector`: `vcat(vec(initial_state), vec(uncertainty), vec(xhat))`.
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
    critic_value(control_variate, initial_state, uncertainty, xhat) -> Number

Evaluate the scalar critic on one scenario.

# Arguments
- `control_variate::ScalarCriticControlVariate`: critic configuration.
- `initial_state`: initial state for the scenario.
- `uncertainty`: uncertainty trajectory for the scenario.
- `xhat`: policy target trajectory for the scenario.

# Returns
- `Number`: scalar critic prediction.

# Throws
- Throws an error if the critic returns a non-scalar array.
"""
critic_value(
    cv::ScalarCriticControlVariate,
    initial_state,
    uncertainty,
    xhat,
) = _critic_value(cv.critic, cv.featurizer, initial_state, uncertainty, xhat)

"""
    critic_xhat_gradient(control_variate, initial_state, uncertainty, xhat)

Differentiate the scalar critic with respect to the policy target trajectory.

# Arguments
- `control_variate::ScalarCriticControlVariate`: critic configuration.
- `initial_state`: initial state for the scenario.
- `uncertainty`: uncertainty trajectory for the scenario.
- `xhat`: policy target trajectory for the scenario.

# Returns
- An array with the same shape as `xhat`, equal to
  `gradient(x -> critic_value(control_variate, initial_state, uncertainty, x), xhat)`.

# Throws
- Throws an error if the critic gradient shape differs from `xhat`.

# Notes
If Zygote reports `nothing`, the gradient is replaced by `zero(xhat)`.
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
    critic_loss(
        control_variate,
        samples;
        value_loss_weight = control_variate.value_loss_weight,
        gradient_loss_weight = control_variate.gradient_loss_weight,
    ) -> Real

Compute the scalar critic loss on a collection of `CriticSample`s.

# Arguments
- `control_variate::ScalarCriticControlVariate`: critic configuration.
- `samples`: collection of [`CriticSample`](@ref) values.

# Keywords
- `value_loss_weight`: nonnegative override for objective-value loss weight.
- `gradient_loss_weight`: nonnegative override for target-gradient loss
  weight.

# Returns
- `Real`: average critic loss over `samples`, or `0.0` for an empty
  collection.

# Throws
- Throws an error if either loss weight is negative.
- Throws an error if a sample's target multipliers or critic gradient do not
  match the shape of `xhat`.
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
    update_critic!(
        opt_state,
        control_variate,
        samples;
        batch_size = nothing,
    ) -> Float64

Run one optimizer step for the scalar critic.

# Arguments
- `opt_state`: Flux optimizer state for `control_variate.critic`.
- `control_variate::ScalarCriticControlVariate`: critic configuration.
- `samples`: replay samples available for training.

# Keywords
- `batch_size`: optional minibatch size; `nothing` or a value greater than the
  sample count uses all samples.

# Returns
- `Float64`: critic loss on the selected batch, or `NaN` when the selected
  batch is empty.

# Notes
Only critic parameters are updated. If the materialized gradient is `nothing`
or contains non-finite values, the optimizer update is skipped while the loss
is still reported.
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
