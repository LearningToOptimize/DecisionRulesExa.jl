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

Stateless MLP policy: one call with `vcat(x0, w_flat)` returns the full target
trajectory `x̂` as a flat vector of length `T*nx`.
"""
struct MLPPolicy{M}
    model::M
    output_dim::Int
end

Flux.@layer MLPPolicy

function (π::MLPPolicy)(input)
    y = π.model(input)
    return vec(y)[1:π.output_dim]
end

"""
    MLPPolicy(input_dim, output_dim; hidden=(64,64), act=tanh)
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

Stateful LSTM policy for sequential rollout:

    x̂_t = policy(vcat(w_t, x̂_{t-1}))

- `encoder`: LSTM chain operating on the uncertainty slice `w_t`
- `combiner`: Dense layer combining encoder output with previous state

Call `Flux.reset!(policy)` before each episode.

# Flux 0.16 note
LSTM requires ≥2D input.  The forward pass reshapes the 1D `w_t` slice to
`(n_uncertainty, 1)` before encoding and squeezes back with `vec`.
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

Flux.reset!(m::StateConditionedPolicy) = Flux.reset!(m.encoder)

function _adapt_policy_bound(x::AbstractVector, ref::AbstractVector)
    typeof(x) === typeof(ref) && return x
    y = similar(ref, length(x))
    copyto!(y, convert.(eltype(ref), x))
    return y
end

"""
    load_stateconditioned_policy!(policy, state)

Load a `Flux.state` checkpoint into a `StateConditionedPolicy`.

Checkpoints saved before `output_bounds` existed contain only the trainable
encoder and combiner state.  In that case, restore those trainable components
and keep the current policy's case-defined output bounds.
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

Construct a `StateConditionedPolicy`.

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

Policy for cases where no target dimension is trainable. It always returns the
case-defined target vector and has no trainable parameters.
"""
struct ConstantStatePolicy{O}
    output_template::O
    n_uncertainty::Int
    n_state::Int
end

Flux.@layer ConstantStatePolicy trainable=()

(m::ConstantStatePolicy)(input) = _adapt_policy_bound(m.output_template, input)
Flux.reset!(::ConstantStatePolicy) = nothing

"""
    FixedOutputPolicy(policy, output_template, output_expansion)

Wrap a policy that predicts only active target dimensions and expand its output
to the full state-target vector. `output_template` stores constants for
inactive dimensions and zeros for active dimensions; `output_expansion` maps
active outputs into the full state vector without mutation.
"""
struct FixedOutputPolicy{P,O,E}
    policy::P
    output_template::O
    output_expansion::E
end

Flux.@layer FixedOutputPolicy trainable=(policy,)

function (m::FixedOutputPolicy)(input)
    y = m.policy(input)
    return m.output_template .+ m.output_expansion * y
end

Flux.reset!(m::FixedOutputPolicy) = Flux.reset!(m.policy)

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
- `activation`: head activation, defaulting to `sigmoid`.
- `encoder_type`: recurrent layer constructor.
- `active_mask`: optional Boolean mask selecting trainable output dimensions.
- `fixed_values`: values used for inactive target dimensions.
- `combiner_layers`: hidden widths for the nonrecurrent target head.

# Returns
- `StateConditionedPolicy` when all dimensions are active.
- `ConstantStatePolicy` when no dimensions are active.
- `FixedOutputPolicy` when only a subset is active.

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
