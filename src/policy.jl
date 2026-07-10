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

# ── ContextualPolicy ──────────────────────────────────────────────────────────

"""
    ContextualPolicy(policy, context)

Wrap a stage policy so each call receives known exogenous context before the
usual policy input.  The wrapped policy is called as
`policy(vcat(context_at(context, t), input))`, with `t` advanced once per call
and reset by `Flux.reset!`.

This keeps training and rollout loops unchanged: they still pass `[w_t; x_prev]`
to the policy, while the wrapper prepends known stage features such as seasonal
phase or forecast covariates.  Only the inner policy is trainable.
"""
mutable struct ContextualPolicy{P,C}
    policy::P
    context::C
    t::Int
end

ContextualPolicy(policy, context) = ContextualPolicy(policy, context, 0)

Flux.@layer ContextualPolicy trainable=(policy,)

"""
    context_at(context, t)

Return the context vector for one-based stage `t`.
"""
function context_at(context::AbstractMatrix, t::Integer)
    1 <= t <= size(context, 2) ||
        throw(BoundsError(context, (:, t)))
    return view(context, :, t)
end

context_at(context::Function, t::Integer) = context(t)

function (m::ContextualPolicy)(input)
    m.t += 1
    return m.policy(vcat(context_at(m.context, m.t), input))
end

function Flux.reset!(m::ContextualPolicy)
    m.t = 0
    Flux.reset!(m.policy)
    return nothing
end

"""
    stage_phase_context(T; period, include_progress=true)

Build a `d x T` context matrix with `sin(2*pi*t/period)`,
`cos(2*pi*t/period)`, and optionally normalized horizon progress `t/T`.
The sine/cosine pair preserves cyclic adjacency between the last and first
seasonal positions while using only two bounded input features.
"""
function stage_phase_context(T::Integer; period::Integer, include_progress::Bool=true)
    T >= 1 || throw(ArgumentError("T must be positive"))
    period >= 1 || throw(ArgumentError("period must be positive"))
    nrows = include_progress ? 3 : 2
    ctx = Matrix{Float32}(undef, nrows, T)
    for t in 1:T
        θ = 2f0 * Float32(pi) * Float32(t) / Float32(period)
        ctx[1, t] = sin(θ)
        ctx[2, t] = cos(θ)
        if include_progress
            ctx[3, t] = Float32(t) / Float32(T)
        end
    end
    return ctx
end

"""
    vcat_contexts(a, b, ...)

Vertically concatenate context matrices after checking that they cover the
same number of stages.
"""
function vcat_contexts(contexts::AbstractMatrix...)
    isempty(contexts) && return Matrix{Float32}(undef, 0, 0)
    T = size(first(contexts), 2)
    all(size(c, 2) == T for c in contexts) ||
        throw(ArgumentError("all contexts must have the same number of columns"))
    return vcat(contexts...)
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

# ── Recurrent-state threading helpers (Flux ≥ 0.16 stateless cells) ──────────
#
# In Flux 0.16 recurrent layers are stateless: `(l::LSTM)(x)` restarts from
# `Flux.initialstates(l)` on EVERY call and `Flux.reset!` on Flux layers is a
# deprecated no-op. A policy that needs cross-stage memory must therefore carry
# the recurrent state itself and advance it with one stateful cell call per
# stage. The helpers below mirror DecisionRules.jl's `_as_cell`,
# `_init_recurrent_state`, `_step_encoder`, and `_state_eltype`
# (src/dense_multilayer_nn.jl) so both packages thread recurrent encoders with
# identical semantics. Unlike DecisionRules.jl, which stores BARE cells, EXA
# encoders keep the `Flux.LSTM` wrapper layers (preserving the weight
# structure of existing checkpoints); `_as_cell` unwraps to the underlying
# cell for the stateful `(x, state) -> (output, new_state)` call.

"""
    _as_cell(layer)

Return the underlying recurrent cell of `layer`. `Flux.LSTM`/`GRU`/`RNN` wrap a
cell (`LSTMCell`/`GRUCell`/`RNNCell`) in a `.cell` field; if `layer` has no such
field it is already a cell and is returned unchanged.
"""
# Unwrap the .cell field if present (LSTM → LSTMCell); return unchanged otherwise.
_as_cell(layer) = hasfield(typeof(layer), :cell) ? layer.cell : layer

"""
    _init_recurrent_state(encoder)

Return the initial recurrent state for `encoder`: `Flux.initialstates` of the
underlying cell for a single layer, or a tuple of per-layer initial states for
a `Chain`.

# Notes
`Flux.initialstates` builds zero states with `zeros_like` on the cell weights,
so the returned state inherits the encoder's device and element type — call
[`Flux.reset!`](@ref) after moving a policy between devices to re-derive the
state on the new device.
"""
# Single layer: initial state of the underlying cell (zeros for LSTM h/c).
_init_recurrent_state(layer) = Flux.initialstates(_as_cell(layer))
# Chain: one initial state per layer, returned as a tuple.
_init_recurrent_state(chain::Flux.Chain) = map(_init_recurrent_state, chain.layers)

"""
    _step_encoder(encoder, x, state) -> (output, new_state)

Advance `encoder` by one step on input `x` from recurrent `state`, returning
the output and the updated state. For a `Chain`, each layer's output feeds the
next layer and each layer's state is threaded independently.
"""
# Single layer: one stateful cell call returns (output, new_state).
_step_encoder(layer, x, state) = _as_cell(layer)(x, state)
function _step_encoder(chain::Flux.Chain, x, states::Tuple)
    # Delegate to the recursive tuple-based implementation.
    return _step_encoder_layers(chain.layers, x, states)
end

"""
    _step_encoder_layers(layers, x, states) -> (output, new_states)

Recursively advance a tuple of recurrent layers by one time step.

Each layer receives the output of the previous layer as input and its own
independent recurrent state. The base case (`layers == ()`) returns the input
unchanged with an empty state tuple.

# Arguments
- `layers::Tuple`: remaining recurrent layers to evaluate.
- `x`: current input (or output of the prior layer).
- `states::Tuple`: per-layer recurrent states, same length as `layers`.

# Returns
- `output`: output of the last layer in `layers`.
- `new_states::Tuple`: updated recurrent states, one per layer.
"""
_step_encoder_layers(::Tuple{}, x, ::Tuple{}) = x, ()
function _step_encoder_layers(layers::Tuple, x, states::Tuple)
    # Advance the first layer with its own recurrent state.
    out, new_state = _step_encoder(first(layers), x, first(states))

    # Recurse on remaining layers, feeding this layer's output as input.
    rest_out, rest_states = _step_encoder_layers(Base.tail(layers), out, Base.tail(states))

    # Reassemble the full state tuple: this layer's state followed by the rest.
    return rest_out, (new_state, rest_states...)
end

"""
    _state_eltype(state) -> Type

Return the scalar element type of a recurrent state.

For nested tuple states (e.g. LSTM's `(h, c)` or a `Chain`'s tuple of per-layer
states) this recurses into the first element until it reaches an
`AbstractVector`, then returns `eltype(v)`. The result is used to cast inputs
to the encoder's precision before each step.
"""
_state_eltype(state::Tuple) = _state_eltype(first(state))
_state_eltype(v::AbstractArray) = eltype(v)   # AbstractArray (not just Vector) so batched matrix states work

"""
    StateConditionedPolicy{E,C,S}

Flux-compatible state-conditioned policy for sequential target rollout.

At each stage the policy is called as

```julia
xhat_t = policy(vcat(w_t, x_prev))
```

where the recurrent encoder reads only `w_t` and the combiner reads
`[encoded_uncertainty; x_prev]`.

Flux's recurrent cells are stateless (Flux ≥ 0.16): each call returns
`(output, new_state)` instead of mutating internal state, and calling the
`LSTM` wrapper directly would restart from `initialstates` on every call.
`StateConditionedPolicy` therefore carries the encoder's recurrent state itself
in `state`, threading it through one cell call per stage — the same semantics
as DecisionRules.jl's `StateConditionedPolicy`. Call `Flux.reset!(policy)` to
restore it to `Flux.initialstates` at the start of a scenario.

# Fields
- `encoder`: recurrent uncertainty encoder (`Chain` of `Flux.LSTM`-style layers).
- `combiner`: nonrecurrent target head.
- `state`: current recurrent state ``s_t``, carried across calls (not trainable).
- `n_uncertainty::Int`: number of uncertainty features at each stage.
- `n_state::Int`: number of previous-state features.
- `output_lower`: optional lower bounds for affine output scaling.
- `output_scale`: optional `upper - lower` scale for affine output scaling.

# Notes
Call `Flux.reset!(policy)` before each scenario — the reset is REAL (it
restores the initial recurrent state), unlike the deprecated Flux-layer
`reset!` no-op. Within a differentiated rollout the threaded state is treated
as data: gradients flow into encoder/combiner parameters through each stage's
forward pass, and the state stored between calls is refreshed by mutation
(mirroring DecisionRules.jl's training semantics).
"""
mutable struct StateConditionedPolicy{E,C,S,L,U}
    encoder::E          # Recurrent uncertainty encoder (Chain of LSTM-style layers)
    combiner::C         # Nonrecurrent target head
    state::S            # Encoder recurrent state, carried across calls
    n_uncertainty::Int  # Number of uncertainty features per stage
    n_state::Int        # Number of previous-state features
    output_lower::L     # Optional affine output lower bounds
    output_scale::U     # Optional affine output scale (upper - lower)
end

Flux.@layer StateConditionedPolicy trainable=(encoder, combiner)

"""
    (policy::StateConditionedPolicy)(input) -> AbstractVector

Evaluate one stage of a state-conditioned policy, threading recurrent state.

The input is split into the uncertainty portion ``w_t`` and the previous state
``x_{t-1}``, and the forward pass computes

```math
h_t, s_t = \\text{encoder}(w_t, s_{t-1}), \\qquad
\\hat{x}_t = f_{\\text{combine}}([h_t;\\; x_{t-1}]),
```

where ``s_t`` is the updated recurrent state (stored in `policy.state` for the
next call).

# Arguments
- `input`: concatenated vector `[w_t; x_prev]`.

# Returns
- Raw combiner output, or affine-scaled output when `output_bounds` were
  supplied at construction.
"""
function (m::StateConditionedPolicy)(input)
    # Split the concatenated input into uncertainty w_t and previous state x_{t-1}.
    w = input[1:m.n_uncertainty]
    s = input[m.n_uncertainty+1:end]

    # Cast the uncertainty to the encoder precision (taken from the recurrent
    # state), matching DecisionRules.jl's forward pass exactly.
    T = _state_eltype(m.state)

    # Advance the recurrent encoder by one step: h_t, s_t = encoder(w_t, s_{t-1}).
    h, new_state = _step_encoder(m.encoder, T.(w), m.state)

    # Persist the new recurrent state so the next call starts from s_t.
    m.state = new_state

    y = m.combiner(vcat(h, s))
    if m.output_lower === nothing
        return y
    end
    lower = _adapt_policy_bound(m.output_lower, y)
    scale = _adapt_policy_bound(m.output_scale, y)
    return lower .+ scale .* y
end

"""
    Flux.reset!(policy::StateConditionedPolicy) -> Nothing

Reset the encoder's recurrent state to `Flux.initialstates`, e.g. at the start
of a scenario rollout.

# Notes
The state is re-derived from the (possibly device-moved) encoder weights on
every reset, so calling `Flux.reset!` after `gpu(policy)`/`cpu(policy)` places
the state on the correct device with the correct element type.
"""
function Flux.reset!(m::StateConditionedPolicy)
    # Reinitialize s_0 to the cell defaults (zeros for LSTM h/c).
    m.state = _init_recurrent_state(m.encoder)
    return nothing
end

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
    _load_encoder_state!(encoder, enc_state) -> encoder

Load a checkpoint `encoder` state into a recurrent encoder, accepting BOTH
weight structures in use across the two packages:

1. EXA structure — `Chain` of `Flux.LSTM` wrapper layers, so each layer state
   is `(cell = (Wi, Wh, bias),)`. Loaded by the stock `Flux.loadmodel!`.
2. DecisionRules.jl (MAIN) structure — `Chain` of BARE `LSTMCell`s (MAIN's
   `_as_cell` strips the wrapper at construction), so each layer state is
   `(Wi, Wh, bias)` directly. `Flux.loadmodel!` matches children by key, so
   loading a bare-cell layer state into an `LSTM` wrapper (children `(cell,)`)
   throws; this helper falls back to cell-by-cell injection, loading each
   layer state into `_as_cell(layer)` — the mathematically identical weight
   assignment (the wrapper's `cell` has exactly the fields `(Wi, Wh, bias)`
   MAIN saved).

# Arguments
- `encoder`: destination encoder (typically a `Chain` of `Flux.LSTM` layers).
- `enc_state`: the checkpoint's encoder state (from `Flux.state`).

# Returns
- `encoder`, mutated in place.

# Throws
- Rethrows the stock `Flux.loadmodel!` error when the fallback does not apply
  (non-`Chain` encoder, missing `layers`, or depth mismatch).
"""
function _load_encoder_state!(encoder, enc_state)
    try
        # Stock path: same weight structure (EXA wrapper layers).
        Flux.loadmodel!(encoder, enc_state)
        return encoder
    catch err
        # Fallback applies only to Chain encoders with a per-layer state list.
        (encoder isa Flux.Chain && hasproperty(enc_state, :layers)) || rethrow(err)
        layer_states = getproperty(enc_state, :layers)
        length(layer_states) == length(encoder.layers) || rethrow(err)
        for (layer, lstate) in zip(encoder.layers, layer_states)
            # Wrapper-style layer state loads into the wrapper; bare-cell
            # (MAIN) layer state loads into the unwrapped cell.
            dest = hasproperty(lstate, :cell) ? layer : _as_cell(layer)
            Flux.loadmodel!(dest, lstate)
        end
        return encoder
    end
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
- Checkpoints saved before output bounds were added contain only the trainable
  encoder and combiner state. In that case, this method restores those
  trainable components and keeps the current policy's case-defined output
  bounds.
- Checkpoints trained BEFORE recurrent-state threading (memoryless-encoder
  era) have the SAME weight structure and load unchanged — only the runtime
  semantics differ (the encoder now carries memory across stages).
- DecisionRules.jl (MAIN) checkpoints, whose encoders are `Chain`s of bare
  `LSTMCell`s, load through the documented cell-by-cell fallback in
  [`_load_encoder_state!`](@ref).
- The loaded policy's recurrent state is reset afterwards so the next rollout
  starts from `Flux.initialstates` of the loaded weights.
"""
function load_stateconditioned_policy!(policy::StateConditionedPolicy, state)
    try
        Flux.loadmodel!(policy, state)
        Flux.reset!(policy)
        return policy
    catch err
        hasproperty(state, :encoder) && hasproperty(state, :combiner) || rethrow(err)
        @warn "Full StateConditionedPolicy checkpoint load failed; loading encoder/combiner only and keeping current output bounds" exception=(err, catch_backtrace())
        _load_encoder_state!(policy.encoder, getproperty(state, :encoder))
        Flux.loadmodel!(policy.combiner, getproperty(state, :combiner))
        Flux.reset!(policy)
        return policy
    end
end

function load_stateconditioned_policy!(policy::ContextualPolicy, state)
    inner_state = hasproperty(state, :policy) ? getproperty(state, :policy) : state
    load_stateconditioned_policy!(policy.policy, inner_state)
    Flux.reset!(policy)
    return policy
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
        # Initialize the recurrent state to Flux.initialstates for the encoder.
        return StateConditionedPolicy(
            encoder, combiner, _init_recurrent_state(encoder),
            n_uncertainty, n_state, nothing, nothing,
        )
    end
    lower, upper = output_bounds
    length(lower) == n_out || throw(ArgumentError("output lower bound length must be n_out=$n_out"))
    length(upper) == n_out || throw(ArgumentError("output upper bound length must be n_out=$n_out"))
    scale = upper .- lower
    any(<(zero(eltype(scale))), scale) &&
        throw(ArgumentError("output upper bounds must be >= lower bounds"))
    return StateConditionedPolicy(
        encoder, combiner, _init_recurrent_state(encoder),
        n_uncertainty, n_state,
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
