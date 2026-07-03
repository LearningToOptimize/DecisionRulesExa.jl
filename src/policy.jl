# policy.jl
#
# Policy types for TS-DDR training.
#
# Two styles:
#   MLPPolicy             — stateless; called once per scenario with vcat(x0, w_flat)
#   StateConditionedPolicy — stateful LSTM; called once per stage with vcat(w_t, prev)

# ── MLPPolicy ─────────────────────────────────────────────────────────────────

"""
    MLPPolicy(model, output_dim)

Wrap a stateless Flux model as a full-horizon target policy.

The wrapped model is called once per scenario with `vcat(x0, w_flat)` and
returns the full target trajectory as a flat vector of length `output_dim`.

# Arguments
- `model`: Flux-compatible callable.
- `output_dim::Int`: number of target components retained from `vec(model(input))`.

# Returns
- `MLPPolicy`: a Flux layer wrapper around `model`.
"""
struct MLPPolicy{M}
    model::M
    output_dim::Int
end

Flux.@layer MLPPolicy

"""
    (policy::MLPPolicy)(input) -> AbstractVector

Evaluate a stateless full-horizon policy.

# Arguments
- `input`: concatenated initial state and flat uncertainty trajectory.

# Returns
- The first `policy.output_dim` entries of `vec(policy.model(input))`.
"""
function (π::MLPPolicy)(input)
    y = π.model(input)
    return vec(y)[1:π.output_dim]
end

"""
    MLPPolicy(input_dim, output_dim; hidden=(64,64), act=tanh)

Construct a feed-forward [`MLPPolicy`](@ref).

# Arguments
- `input_dim::Int`: length of the concatenated scenario input.
- `output_dim::Int`: length of the flattened target trajectory.

# Keywords
- `hidden`: hidden-layer widths.
- `act`: hidden-layer activation.

# Returns
- `MLPPolicy`: stateless full-horizon policy.
"""
function MLPPolicy(input_dim::Int, output_dim::Int;
    hidden = (64, 64),
    act = tanh,
)
    layers = Any[]
    in_dim = input_dim
    for h in hidden
        push!(layers, Flux.Dense(in_dim, h, act))
        in_dim = h
    end
    push!(layers, Flux.Dense(in_dim, output_dim))
    return MLPPolicy(Flux.Chain(layers...), output_dim)
end

raw"""
    _dense_policy_head(input_dim, output_dim, hidden; activation=tanh)

Build the nonrecurrent target head used by state-conditioned policies.

The head maps the concatenated features `[encoded_uncertainty; previous_state]`
to the normalized or unbounded target vector. When `hidden` is empty, this is the
historical single `Dense(input_dim => output_dim, activation)` head. When
`hidden = [h₁, …, h_L]`, the returned chain is

```math
f(z) =
\sigma\left(W_{L+1}
  \sigma\left(W_L \cdots \sigma(W_1 z + b_1) \cdots + b_L\right)
  + b_{L+1}\right).
```

The output layer intentionally uses the same `activation`. This is different
from a generic MLP helper with a linear output: bounded policies need the final
head output to stay in `[0, 1]` when `activation=sigmoid`, before affine scaling
to physical target bounds.

# Arguments
- `input_dim::Int`: dimension of `[encoded_uncertainty; previous_state]`.
- `output_dim::Int`: number of target components produced by the policy.
- `hidden::AbstractVector{Int}`: hidden widths of the feed-forward target head.
- `activation`: activation applied at every head layer, including the output.

# Returns
- `Flux.Dense` if `hidden` is empty, otherwise `Flux.Chain` of dense layers.

# Examples
```julia
head = DecisionRulesExa._dense_policy_head(135, 11, [128, 128]; activation=sigmoid)
```

See also: [`StateConditionedPolicy`](@ref), [`bounded_state_policy`](@ref)
"""
function _dense_policy_head(
    input_dim::Int,
    output_dim::Int,
    hidden::AbstractVector{Int};
    activation = tanh,
)
    isempty(hidden) && return Flux.Dense(input_dim => output_dim, activation)
    layers = Any[Flux.Dense(input_dim => hidden[1], activation)]
    for i in 1:(length(hidden) - 1)
        push!(layers, Flux.Dense(hidden[i] => hidden[i + 1], activation))
    end
    push!(layers, Flux.Dense(hidden[end] => output_dim, activation))
    return Flux.Chain(layers...)
end

"""
    StateConditionedPolicy{E,C}

Flux-compatible state-conditioned policy for sequential target rollout.

At each stage the policy is called as

```julia
xhat_t = policy(vcat(w_t, x_prev))
```

where the recurrent encoder reads only `w_t` and the combiner reads
`[encoded_uncertainty; x_prev]`.

# Fields
- `encoder`: recurrent uncertainty encoder.
- `combiner`: nonrecurrent target head.
- `n_uncertainty::Int`: number of uncertainty features at each stage.
- `n_state::Int`: number of previous-state features.
- `output_lower`: optional lower bounds for affine output scaling.
- `output_scale`: optional `upper - lower` scale for affine output scaling.

# Notes
Call `Flux.reset!(policy)` before each scenario. Recurrent Flux layers require
two-dimensional input, so the forward pass reshapes the one-dimensional
uncertainty slice to `(n_uncertainty, 1)` before encoding.
"""
struct StateConditionedPolicy{E,C,L,U}
    encoder::E
    combiner::C
    n_uncertainty::Int
    n_state::Int
    output_lower::L
    output_scale::U
end

Flux.@layer StateConditionedPolicy trainable=(encoder, combiner)

"""
    (policy::StateConditionedPolicy)(input) -> AbstractVector

Evaluate one stage of a state-conditioned policy.

# Arguments
- `input`: concatenated vector `[w_t; x_prev]`.

# Returns
- Raw combiner output, or affine-scaled output when `output_bounds` were
  supplied at construction.
"""
function (m::StateConditionedPolicy)(input)
    w = reshape(input[1:m.n_uncertainty], :, 1)   # (n_unc, 1) for LSTM
    s = input[m.n_uncertainty+1:end]
    h = vec(m.encoder(w))                          # (hidden,)
    y = m.combiner(vcat(h, s))
    if m.output_lower === nothing
        return y
    end
    lower = _adapt_policy_bound(m.output_lower, y)
    scale = _adapt_policy_bound(m.output_scale, y)
    return lower .+ scale .* y
end

"""
    Flux.reset!(policy::StateConditionedPolicy)

Reset the recurrent uncertainty encoder.

# Returns
- The result of `Flux.reset!(policy.encoder)`.
"""
Flux.reset!(m::StateConditionedPolicy) = Flux.reset!(m.encoder)

"""
    _adapt_policy_bound(x, ref)

Move a policy bound vector to the same storage family and element type as
`ref`.

# Arguments
- `x::AbstractVector`: bound vector stored on the policy.
- `ref::AbstractVector`: output vector whose element type and device should be
  matched.

# Returns
- `x` itself when the concrete vector type already matches `ref`; otherwise a
  copied vector compatible with `ref`.
"""
function _adapt_policy_bound(x::AbstractVector, ref::AbstractVector)
    typeof(x) === typeof(ref) && return x
    y = similar(ref, length(x))
    copyto!(y, convert.(eltype(ref), x))
    return y
end

"""
    load_stateconditioned_policy!(policy, state)

Load a Flux checkpoint into a [`StateConditionedPolicy`](@ref).

# Arguments
- `policy::StateConditionedPolicy`: policy to update in place.
- `state`: checkpoint object accepted by `Flux.loadmodel!`.

# Returns
- `policy`.

# Notes
Checkpoints saved before output bounds were added contain only the trainable
encoder and combiner state. In that case, this method restores those trainable
components and keeps the current policy's case-defined output bounds.
"""
function load_stateconditioned_policy!(policy::StateConditionedPolicy, state)
    try
        Flux.loadmodel!(policy, state)
        return policy
    catch err
        hasproperty(state, :encoder) && hasproperty(state, :combiner) || rethrow(err)
        @warn "Full StateConditionedPolicy checkpoint load failed; loading encoder/combiner only and keeping current output bounds" exception=(err, catch_backtrace())
        Flux.loadmodel!(policy.encoder, getproperty(state, :encoder))
        Flux.loadmodel!(policy.combiner, getproperty(state, :combiner))
        return policy
    end
end

raw"""
    StateConditionedPolicy(n_uncertainty, n_state, n_out, layers;
                           activation=tanh, encoder_type=Flux.LSTM,
    output_bounds=nothing, combiner_layers=Int[])

Construct a state-conditioned sequential target policy.

The policy separates memory from state conditioning:

```math
h_t = E_\theta(w_t, h_{t-1}), \qquad
\hat{x}_t = H_\theta([h_t;\, x_{t-1}]).
```

The recurrent encoder `Eθ` sees only the stage uncertainty. The previous state
enters only through the feed-forward head `Hθ`, so increasing
`combiner_layers` makes the state-to-target map nonlinear without introducing
recurrence over the state input.

# Arguments
- `n_uncertainty::Int`: number of uncertainty features in each stage input.
- `n_state::Int`: number of previous-state features appended after uncertainty.
- `n_out::Int`: output dimension, usually the state-target dimension.
- `layers::AbstractVector{Int}`: recurrent encoder hidden sizes.

# Keywords
- `activation`: head activation. Use `sigmoid` with `output_bounds` when targets
  must remain inside bounds.
- `encoder_type`: recurrent layer constructor, typically `Flux.LSTM`.
- `output_bounds`: optional `(lower, upper)` vectors. If provided, the raw head
  output `y` is interpreted as normalized and mapped to
  `lower + (upper - lower) .* y`.
- `combiner_layers`: hidden widths for the nonrecurrent target head. `Int[]`
  preserves the original single Dense head.

# Returns
- `StateConditionedPolicy` with trainable `encoder` and `combiner`.

# Notes
The recurrent encoder receives only uncertainty. The previous state enters
through the feed-forward combiner, so `combiner_layers` increases
state-to-target expressiveness without making the state input recurrent.

# Examples
```julia
policy = StateConditionedPolicy(
    11, 11, 11, [128, 128];
    activation = sigmoid,
    output_bounds = (zeros(Float32, 11), ones(Float32, 11)),
    combiner_layers = [128, 128],
)
```

See also: [`bounded_state_policy`](@ref)
"""
function StateConditionedPolicy(
    n_uncertainty::Int,
    n_state::Int,
    n_out::Int,
    layers::AbstractVector{Int};
    activation  = tanh,
    encoder_type = Flux.LSTM,
    output_bounds = nothing,
    combiner_layers = Int[],
)
    enc_sizes  = vcat(n_uncertainty, layers)
    enc_layers = [encoder_type(enc_sizes[i] => enc_sizes[i+1])
                  for i in 1:length(layers)]
    encoder  = Flux.Chain(enc_layers...)
    combiner = _dense_policy_head(
        layers[end] + n_state,
        n_out,
        collect(Int, combiner_layers);
        activation = activation,
    )
    if output_bounds === nothing
        return StateConditionedPolicy(encoder, combiner, n_uncertainty, n_state, nothing, nothing)
    end
    lower, upper = output_bounds
    length(lower) == n_out || throw(ArgumentError("output lower bound length must be n_out=$n_out"))
    length(upper) == n_out || throw(ArgumentError("output upper bound length must be n_out=$n_out"))
    scale = upper .- lower
    any(<(zero(eltype(scale))), scale) &&
        throw(ArgumentError("output upper bounds must be >= lower bounds"))
    return StateConditionedPolicy(
        encoder, combiner, n_uncertainty, n_state,
        collect(lower), collect(scale),
    )
end

# ── Bounded state-target helpers ──────────────────────────────────────────────

"""
    ConstantStatePolicy(output_template, n_uncertainty, n_state)

Represent a policy with no trainable target dimensions.

The policy always returns `output_template`, adapted to the input device.

# Arguments
- `output_template`: fixed full target vector.
- `n_uncertainty::Int`: number of uncertainty features expected in the input.
- `n_state::Int`: number of state features expected in the input.

# Returns
- `ConstantStatePolicy`: Flux layer with no trainable parameters.
"""
struct ConstantStatePolicy{O}
    output_template::O
    n_uncertainty::Int
    n_state::Int
end

Flux.@layer ConstantStatePolicy trainable=()

"""
    (policy::ConstantStatePolicy)(input) -> AbstractVector

Return the fixed target template, adapted to the input storage family.

# Arguments
- `input`: vector used only as an adaptation reference.

# Returns
- The fixed target vector.
"""
(m::ConstantStatePolicy)(input) = _adapt_policy_bound(m.output_template, input)

"""
    Flux.reset!(policy::ConstantStatePolicy) -> Nothing

No-op reset method for constant policies.
"""
Flux.reset!(::ConstantStatePolicy) = nothing

"""
    FixedOutputPolicy(policy, output_template, output_expansion)

Expand active target dimensions into a full target vector.

The wrapped policy predicts only active dimensions. `output_template` stores
constants for inactive dimensions and zeros for active dimensions;
`output_expansion` maps active outputs into the full state vector without
mutation.

# Arguments
- `policy`: Flux-compatible policy for active dimensions.
- `output_template`: full target vector with inactive constants.
- `output_expansion`: matrix mapping active outputs into full target space.

# Returns
- `FixedOutputPolicy`: Flux layer wrapper around `policy`.
"""
struct FixedOutputPolicy{P,O,E}
    policy::P
    output_template::O
    output_expansion::E
end

Flux.@layer FixedOutputPolicy trainable=(policy,)

"""
    (policy::FixedOutputPolicy)(input) -> AbstractVector

Evaluate the active-dimension policy and expand it into the full target vector.

# Arguments
- `input`: stage input forwarded to the wrapped policy.

# Returns
- Full target vector with active outputs inserted and inactive dimensions fixed.
"""
function (m::FixedOutputPolicy)(input)
    y = m.policy(input)
    return m.output_template .+ m.output_expansion * y
end

"""
    Flux.reset!(policy::FixedOutputPolicy)

Reset the wrapped active-dimension policy.

# Returns
- The result of `Flux.reset!(policy.policy)`.
"""
Flux.reset!(m::FixedOutputPolicy) = Flux.reset!(m.policy)

"""
    load_stateconditioned_policy!(policy::FixedOutputPolicy, state)

Load checkpoint state into the wrapped active-dimension policy.

# Arguments
- `policy::FixedOutputPolicy`: wrapper whose inner policy is updated.
- `state`: checkpoint object accepted by the inner policy loader.

# Returns
- The result of `load_stateconditioned_policy!(policy.policy, state)`.
"""
load_stateconditioned_policy!(policy::FixedOutputPolicy, state) =
    load_stateconditioned_policy!(policy.policy, state)

raw"""
    bounded_state_policy(n_uncertainty, lower, upper, layers; kwargs...)

Build a state-conditioned policy whose full output is guaranteed to lie in
`[lower, upper]`, while avoiding trainable outputs for inactive dimensions.

The returned policy has the same rollout semantics as
[`StateConditionedPolicy`](@ref):

```math
\hat{x}_t =
\ell + (u - \ell) \odot H_\theta([E_\theta(w_t);\, x_{t-1}]),
```

where `Hθ` is a sigmoid head by default. Dimensions with `upper == lower` are
treated as constants unless `active_mask` overrides that choice.

By default, a dimension is active when `upper > lower`. Fixed dimensions are
returned as constants, so pure pass-through or no-storage state components do
not create meaningless target parameters. Pass `active_mask` to override this
selection for case-specific target relevance.

Set `combiner_layers` to add hidden layers after `[encoded_uncertainty; state]`
and before the bounded target output. This keeps recurrence confined to the
uncertainty encoder while making the state-to-target map nonlinear.

# Arguments
- `n_uncertainty::Int`: number of uncertainty features in each stage input.
- `lower::AbstractVector`: lower bound for each full target dimension.
- `upper::AbstractVector`: upper bound for each full target dimension.
- `layers::AbstractVector{Int}`: recurrent uncertainty-encoder hidden sizes.

# Keywords
- `activation`: head activation, defaulting to `sigmoid`.
- `encoder_type`: recurrent layer constructor.
- `active_mask`: optional Boolean mask selecting trainable output dimensions.
- `fixed_values`: values used for inactive target dimensions.
- `combiner_layers`: hidden widths for the nonrecurrent target head.

# Returns
- `StateConditionedPolicy` when all dimensions are active.
- `ConstantStatePolicy` when no dimensions are active.
- `FixedOutputPolicy` when only a subset is active.

# Throws
- `ArgumentError` if bound lengths differ, `active_mask` has the wrong length,
  or any upper bound is smaller than its lower bound.

# Examples
```julia
policy = bounded_state_policy(
    11,
    min_volume,
    max_volume,
    [128, 128];
    combiner_layers = [256, 256],
)
```
"""
function bounded_state_policy(
    n_uncertainty::Int,
    lower::AbstractVector,
    upper::AbstractVector,
    layers::AbstractVector{Int};
    activation = sigmoid,
    encoder_type = Flux.LSTM,
    active_mask = nothing,
    fixed_values = lower,
    combiner_layers = Int[],
)
    length(lower) == length(upper) ||
        throw(ArgumentError("lower and upper bound vectors must have the same length"))
    n_state = length(lower)
    length(fixed_values) == n_state ||
        throw(ArgumentError("fixed_values length must match state dimension $n_state"))

    scale = upper .- lower
    any(<(zero(eltype(scale))), scale) &&
        throw(ArgumentError("upper bounds must be >= lower bounds"))

    active = if active_mask === nothing
        collect(scale .> zero(eltype(scale)))
    else
        length(active_mask) == n_state ||
            throw(ArgumentError("active_mask length must match state dimension $n_state"))
        collect(Bool.(active_mask))
    end

    if all(active)
        return StateConditionedPolicy(
            n_uncertainty, n_state, n_state, layers;
            activation = activation,
            encoder_type = encoder_type,
            output_bounds = (lower, upper),
            combiner_layers = combiner_layers,
        )
    elseif !any(active)
        return ConstantStatePolicy(collect(fixed_values), n_uncertainty, n_state)
    end

    idx = findall(active)
    active_policy = StateConditionedPolicy(
        n_uncertainty, n_state, length(idx), layers;
        activation = activation,
        encoder_type = encoder_type,
        output_bounds = (lower[idx], upper[idx]),
        combiner_layers = combiner_layers,
    )
    template = collect(fixed_values)
    template[idx] .= zero(eltype(template))
    expansion = zeros(eltype(template), n_state, length(idx))
    for (j, i) in enumerate(idx)
        expansion[i, j] = one(eltype(template))
    end
    return FixedOutputPolicy(active_policy, template, expansion)
end
