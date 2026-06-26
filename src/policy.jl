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

# ── StateConditionedPolicy ────────────────────────────────────────────────────

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

"""
    StateConditionedPolicy(n_uncertainty, n_state, n_out, layers;
                           activation=tanh, encoder_type=Flux.LSTM,
                           output_bounds=nothing)

Construct a `StateConditionedPolicy`.

- `layers` : hidden sizes for the LSTM encoder, e.g. `[64, 64]`
- `n_out`  : output dimension (= nx = state dimension)
- `output_bounds` : optional `(lower, upper)` vectors.  The combiner output is
  interpreted as a normalized value and mapped as `lower + (upper-lower)*y`.
  With `activation=sigmoid`, this gives bounded targets with useful gradients.
  Fixed dimensions (`lower == upper`) are constant and receive zero gradient.
"""
function StateConditionedPolicy(
    n_uncertainty::Int,
    n_state::Int,
    n_out::Int,
    layers::AbstractVector{Int};
    activation  = tanh,
    encoder_type = Flux.LSTM,
    output_bounds = nothing,
)
    enc_sizes  = vcat(n_uncertainty, layers)
    enc_layers = [encoder_type(enc_sizes[i] => enc_sizes[i+1])
                  for i in 1:length(layers)]
    encoder  = Flux.Chain(enc_layers...)
    combiner = Flux.Dense(layers[end] + n_state => n_out, activation)
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

"""
    bounded_state_policy(n_uncertainty, lower, upper, layers; kwargs...)

Build a state-conditioned policy whose full output is guaranteed to lie in
`[lower, upper]`, while avoiding trainable outputs for inactive dimensions.

By default, a dimension is active when `upper > lower`. Fixed dimensions are
returned as constants, so pure pass-through or no-storage state components do
not create meaningless target parameters. Pass `active_mask` to override this
selection for case-specific target relevance.
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
    )
    template = collect(fixed_values)
    template[idx] .= zero(eltype(template))
    expansion = zeros(eltype(template), n_state, length(idx))
    for (j, i) in enumerate(idx)
        expansion[i, j] = one(eltype(template))
    end
    return FixedOutputPolicy(active_policy, template, expansion)
end
