# hydro_reachable_policy.jl
#
# Hydro-specific feasible target policy for strict regular and embedded DEs.
# This file owns the policy architecture and reachability/cascade bounds; the
# ExaModels problem builders live in hydro_power_exa.jl and
# hydro_power_exa_embedded.jl.

using Flux
using Zygote
import DecisionRulesExa: load_stateconditioned_policy!

"""
    hydro_reachable_policy(hydro_data, layers; activation=sigmoid, encoder_type=Flux.LSTM,
                           spill_max=nothing, combiner_layers=Int[])

Build a state-conditioned hydro policy whose outputs are one-stage reachable
reservoir targets.

The recurrent encoder reads inflows. The nonrecurrent combiner reads
`[encoded_inflow; reservoir_state]`, emits normalized targets in `[0, 1]`, and
the wrapper maps them into the reachability interval implied by the current
inflow and previous reservoir state.

# Arguments
- `hydro_data::HydroData`: hydro limits, initial volumes, stage duration, and
  cascade metadata.
- `layers::AbstractVector{Int}`: recurrent encoder widths over inflows.

# Keywords
- `activation`: activation used by the target head. It must be sigmoid-style so
  normalized targets remain in `[0, 1]`.
- `encoder_type`: Flux recurrent layer constructor, usually `Flux.LSTM`.
- `spill_max`: optional finite spill cap used to tighten lower bounds.
- `combiner_layers`: hidden widths in the nonrecurrent state-conditioned head.

# Returns
- `HydroReachablePolicy`: a Flux-compatible policy with trainable encoder and
  combiner parameters plus fixed hydro reachability metadata.

# Notes
For strict regular deterministic equivalents, rolling this policy from the true
initial state and feeding each previous target into the next policy call gives a
feasible target path by induction. Embedded strict DEs use realized reservoir
states inside the NLP. Cascade links are handled by clamping downstream targets
to the upper bound implied by same-stage upstream targets.

The recurrent encoder state is threaded across stages explicitly (Flux ≥ 0.16
cells are stateless, so calling the `LSTM` wrapper directly would restart from
`initialstates` every stage): the forward pass advances `state` by one cell
step per call, mirroring DecisionRules.jl's `HydroReachablePolicy` exactly.
Call `Flux.reset!(policy)` at scenario boundaries to restore the initial state.
"""
mutable struct HydroReachablePolicy{E,C,RS,V,S,I}
    encoder::E
    combiner::C
    state::RS            # Encoder recurrent state, threaded across stages
    n_uncertainty::Int
    n_state::Int
    min_vol::V
    max_vol::V
    min_turn::V
    max_turn::V
    spill_max::S
    upstream_max_inflow::V
    K::Float64
    output_lower::Nothing
    output_scale::Nothing
    cascade::Vector{CascadeLink}
    cascade_upstream::I
    cascade_downstream::I
    cascade_turn_only::V
    cascade_k_max_turn::V
    cascade_reservoir_ids::I
end

Flux.@layer HydroReachablePolicy trainable=(encoder, combiner)

"""
    _hydro_adapt_bound(x, ref)

Return `x` as a vector with the same array family and element type as `ref`.

# Arguments
- `x::AbstractVector`: hydro metadata stored on the policy.
- `ref::AbstractVector`: vector whose device family and element type should be
  matched.

# Returns
- A vector with `length(x)` whose storage is compatible with `ref`.

# Notes
This keeps metadata such as `min_vol` and `max_vol` on the same device as the
policy forward pass: CPU inputs stay on CPU, GPU inputs stay on GPU.
"""
function _hydro_adapt_bound(x::AbstractVector, ref::AbstractVector)
    typeof(x) === typeof(ref) && return x
    y = similar(ref, length(x))
    copyto!(y, convert.(eltype(ref), x))
    return y
end

"""
    _hydro_adapt_index(x, ref)

Return integer indices stored on the same device family as `ref`.

# Arguments
- `x::AbstractVector`: integer indices stored in ordinary Julia memory.
- `ref::AbstractVector`: vector whose device family should be matched.

# Returns
- An integer vector compatible with `ref`.

# Notes
This helper is used before GPU gathers such as `inflow[upstream]`; copying to
`similar(ref, Int, ...)` avoids host-side scalar indexing during policy
evaluation.
"""
function _hydro_adapt_index(x::AbstractVector, ref::AbstractVector)
    y = similar(ref, Int, length(x))
    copyto!(y, x)
    return y
end

"""
    _hydro_reachable_bounds(policy, inflow, x_prev, ref) -> (lower, upper)

Compute one-stage reservoir target bounds for the hydro water balance.

# Arguments
- `policy::HydroReachablePolicy`: policy carrying fixed hydro metadata.
- `inflow`: current-stage inflow vector.
- `x_prev`: previous reservoir-state vector.
- `ref`: vector used to choose element type and device family.

# Returns
- `(lower, upper)`: vectors defining the closed interval into which normalized
  policy outputs are mapped.

# Notes
For each reservoir, the simplified balance is

```text
x_next = x_prev + K * inflow - K * turbine_out - spill + upstream_contrib.
```

The upper bound uses the smallest required turbine outflow (`min_turn`) and zero
spill. The lower bound is the physical minimum volume unless `spill_max` is
finite; with finite spill, the lowest reachable storage uses `max_turn` and
maximum spill.

Proof sketch: every feasible turbine/spill choice satisfies the same balance
equation and variable bounds. Substituting extremal admissible outflow/spill
values gives an interval containing all one-stage reachable reservoir states.
Mapping a sigmoid output into this interval therefore produces a reachable
target in the relaxed one-stage balance.
"""
function _hydro_reachable_bounds(policy::HydroReachablePolicy, inflow, x_prev, ref)
    min_vol  = _hydro_adapt_bound(policy.min_vol, ref)
    max_vol  = _hydro_adapt_bound(policy.max_vol, ref)
    min_turn = _hydro_adapt_bound(policy.min_turn, ref)
    max_turn = _hydro_adapt_bound(policy.max_turn, ref)
    upstream = _hydro_adapt_bound(policy.upstream_max_inflow, ref)
    K = convert(eltype(ref), policy.K)

    upper_raw = x_prev .+ K .* inflow .- K .* min_turn .+ upstream
    upper = min.(max_vol, upper_raw)

    lower = if policy.spill_max === nothing
        min_vol
    else
        spill_max = _hydro_adapt_bound(policy.spill_max, ref)
        lower_raw = x_prev .+ K .* inflow .- K .* max_turn .- spill_max
        max.(min_vol, lower_raw)
    end

    upper = max.(upper, lower)
    return lower, upper
end
Zygote.@nograd _hydro_reachable_bounds

"""
    _cascade_upper_bounds(policy, target, inflow, x_prev) -> upper

Tighten downstream reservoir upper bounds using same-stage upstream targets.

# Arguments
- `policy::HydroReachablePolicy`: policy carrying cascade metadata.
- `target`: raw target vector before cascade clamping.
- `inflow`: current-stage inflow vector.
- `x_prev`: previous reservoir-state vector.

# Returns
- A vector of per-reservoir upper bounds induced by incoming cascade links.

# Notes
For a cascade link `u -> d`, the upstream target determines the upstream
release implied by the balance:

```text
release_u = K * inflow_u + x_prev_u - target_u.
```

Proof sketch: downstream storage is increasing in upstream contribution. The
largest physically consistent contribution from an upstream target is exactly
the positive release implied by that target, possibly turbine-capped. Therefore
clamping the downstream target to this bound cannot remove any feasible target
that respects the upstream target and cascade balance.

# Assumptions
- **Single-level cascades.** The implied release
  `release_u = K * inflow_u + x_prev_u - target_u` omits the upstream unit's
  own incoming cascade contribution: if `u` itself receives water from a unit
  further upstream, its true release can be larger than computed here. The
  omission is conservative — it can only under-estimate the available
  downstream contribution, so the clamp is never infeasible, merely possibly
  over-tight for multi-level chains. Bolivia's three links
  (COR→SIS turbine-only, ZON→CHU turbine+spill, TAQ1→TAQ2 turbine+spill) are
  all single-level, so the bound is exact for that case.
- **No gradient through the clamp.** `Zygote.@nograd` on this function means
  that when the cascade clamp binds, the true dependence of the downstream
  target on the upstream target carries no gradient — a deliberate
  approximation that keeps the policy pullback cheap and well-defined.
- **Physically infeasible edge case.** If the cascade upper bound falls below
  the reachable lower bound of `_hydro_reachable_bounds`, the clamped target
  can fall below `lower`. There is no policy-level remedy: the stage is
  genuinely infeasible, and downstream slack/deficit handling must absorb it.
"""
function _cascade_upper_bounds(policy::HydroReachablePolicy, target, inflow, x_prev)
    cascade = policy.cascade
    T = eltype(target)
    K = T(policy.K)
    n = length(target)
    isempty(cascade) && return _hydro_adapt_bound(fill(T(Inf), n), target)

    upstream = _hydro_adapt_index(policy.cascade_upstream, target)
    downstream = _hydro_adapt_index(policy.cascade_downstream, target)
    reservoir_ids = _hydro_adapt_index(policy.cascade_reservoir_ids, target)
    turn_only = _hydro_adapt_bound(policy.cascade_turn_only, target)
    k_max_turn = _hydro_adapt_bound(policy.cascade_k_max_turn, target)
    min_turn = _hydro_adapt_bound(policy.min_turn, target)
    max_vol = _hydro_adapt_bound(policy.max_vol, target)

    # Release implied by asking the upstream reservoir to end at `target`.
    release = K .* inflow[upstream] .+ x_prev[upstream] .- target[upstream]
    positive_release = max.(zero(T), release)
    max_contrib = ifelse.(turn_only .> zero(T), min.(k_max_turn, positive_release), positive_release)

    # One upper bound per cascade link, expressed for the downstream reservoir.
    link_upper = x_prev[downstream] .+
                 K .* inflow[downstream] .-
                 K .* min_turn[downstream] .+
                 max_contrib
    link_upper = min.(max_vol[downstream], link_upper)

    # Reduce link-wise bounds to one bound per reservoir. Reservoirs without
    # incoming cascade links receive Inf and are left unchanged by `min`.
    link_by_reservoir = ifelse.(
        reshape(downstream, :, 1) .== reshape(reservoir_ids, 1, :),
        reshape(link_upper, :, 1),
        T(Inf),
    )
    return vec(minimum(link_by_reservoir; dims = 1))
end
Zygote.@nograd _cascade_upper_bounds

"""
    (policy::HydroReachablePolicy)(input) -> target

Evaluate the reachable hydro policy.

# Arguments
- `input`: concatenated vector `[inflow_t; x_{t-1}]`.

# Returns
- A reservoir target vector in the one-stage reachable set.

# Notes
The encoder reads only the inflow, threading its recurrent state across calls
(one cell step per stage, stored in `policy.state` — DecisionRules.jl
semantics). The combiner reads both encoded inflow and previous reservoir
state, emits normalized targets, and those targets are mapped into the
reachability interval before cascade clamping.

The cascade clamp inherits the assumptions documented on
[`_cascade_upper_bounds`](@ref):

- Cascades are treated as single-level — the upstream release omits that unit's
  own incoming cascade contribution, which is conservative (never infeasible,
  possibly over-tight) on multi-level chains. Bolivia's three links
  (COR→SIS turn-only, ZON→CHU turn+spill, TAQ1→TAQ2 turn+spill) are all
  single-level.
- Both `_hydro_reachable_bounds` and `_cascade_upper_bounds` are
  `Zygote.@nograd`, so when the cascade clamp binds, the downstream target's
  true dependence on the upstream target carries no gradient (deliberate
  approximation).
- In the physically infeasible edge case where the cascade upper bound is
  below the reachable lower bound, the returned target can fall below `lower`;
  the stage is genuinely infeasible and no policy-level remedy exists.
"""
function (m::HydroReachablePolicy)(input)
    # Split input: first n_uncertainty elements are inflow, rest is previous state.
    inflow = input[1:m.n_uncertainty]
    x_prev = input[m.n_uncertainty+1:end]

    # Encode inflow through the recurrent encoder, threading state across calls
    # (mirrors DecisionRules.jl: encoded, s_t = _step_encoder(enc, T.(w_t), s_{t-1})).
    # Cast to encoder precision for type stability (avoids Zygote codegen bugs).
    T = DecisionRulesExa._state_eltype(m.state)
    h, new_state = DecisionRulesExa._step_encoder(m.encoder, T.(inflow), m.state)
    # Thread the recurrent state to the next call.
    m.state = new_state

    y = m.combiner(vcat(h, x_prev))
    lower, upper = _hydro_reachable_bounds(m, inflow, x_prev, y)
    # Because `y` is sigmoid-bounded, this affine map stays in [lower, upper].
    raw_target = lower .+ (upper .- lower) .* y
    if !isempty(m.cascade)
        cascade_upper = _cascade_upper_bounds(m, raw_target, inflow, x_prev)
        return min.(raw_target, cascade_upper)
    end
    return raw_target
end

"""
    Flux.reset!(policy::HydroReachablePolicy) -> Nothing

Reset the inflow encoder's recurrent state to `Flux.initialstates`, e.g. at
scenario boundaries.

# Notes
The combiner is feed-forward and does not carry recurrent state. The recurrent
state is re-derived from the (possibly device-moved) encoder weights on every
reset, so the state always matches the encoder's device and element type.
"""
function Flux.reset!(m::HydroReachablePolicy)
    # Reinitialize the recurrent state from the encoder's initial states.
    m.state = DecisionRulesExa._init_recurrent_state(m.encoder)
    return nothing
end

"""
    load_stateconditioned_policy!(policy::HydroReachablePolicy, state)

Load Flux parameters into a reachable hydro policy.

# Arguments
- `policy::HydroReachablePolicy`: target policy to update in place.
- `state`: checkpoint object accepted by `Flux.loadmodel!`.

# Returns
- `policy`.

# Notes
- If the checkpoint contains only `encoder` and `combiner` fields, those
  trainable parts are loaded while hydro reachability metadata from the current
  case is preserved.
- Checkpoints trained BEFORE recurrent-state threading (memoryless-encoder era)
  have the SAME weight structure and load unchanged — only the runtime
  semantics differ (the encoder now carries memory across stages).
- DecisionRules.jl (MAIN) checkpoints save the encoder as a `Chain` of BARE
  `LSTMCell`s (layer state `(Wi, Wh, bias)` instead of `(cell = …,)`); those
  load through the documented cell-by-cell fallback in
  `DecisionRulesExa._load_encoder_state!`.
- The recurrent state is reset after loading so the next rollout starts from
  `Flux.initialstates` of the loaded weights.
"""
function load_stateconditioned_policy!(policy::HydroReachablePolicy, state)
    try
        Flux.loadmodel!(policy, state)
        Flux.reset!(policy)
        return policy
    catch err
        hasproperty(state, :encoder) && hasproperty(state, :combiner) || rethrow(err)
        @warn "Full HydroReachablePolicy checkpoint load failed; loading encoder/combiner only and keeping hydro reachability bounds" exception=(err, catch_backtrace())
        DecisionRulesExa._load_encoder_state!(policy.encoder, getproperty(state, :encoder))
        Flux.loadmodel!(policy.combiner, getproperty(state, :combiner))
        Flux.reset!(policy)
        return policy
    end
end

function hydro_reachable_policy(
    hydro_data::HydroData,
    layers::AbstractVector{Int};
    activation = sigmoid,
    encoder_type = Flux.LSTM,
    spill_max = nothing,
    combiner_layers = Int[],
)
    (activation === sigmoid || activation === NNlib.sigmoid || activation === NNlib.sigmoid_fast) ||
        throw(ArgumentError("hydro_reachable_policy requires a sigmoid-style activation so normalized targets stay in [0, 1]"))
    nHyd = hydro_data.nHyd
    enc_sizes  = vcat(nHyd, layers)
    enc_layers = [encoder_type(enc_sizes[i] => enc_sizes[i+1])
                  for i in 1:length(layers)]
    encoder  = Flux.Chain(enc_layers...)
    encoder_width = isempty(layers) ? nHyd : layers[end]
    combiner = DecisionRulesExa._dense_policy_head(
        encoder_width + nHyd,
        nHyd,
        collect(Int, combiner_layers);
        activation = activation,
    )
    spill_vec = spill_max === nothing ? nothing : Float32.(collect(spill_max))
    if spill_vec !== nothing && length(spill_vec) != nHyd
        throw(ArgumentError("spill_max length must be nHyd=$nHyd"))
    end

    K = Float64(hydro_data.K)
    upstream_max = zeros(Float32, nHyd)
    for conn in hydro_data.upstream_turns
        upstream_max[conn.downstream_pos] += Float32(K * hydro_data.units[conn.upstream_pos].max_turn)
    end

    spill_dests = Dict{Int,Set{Int}}()
    for conn in hydro_data.upstream_spills
        push!(get!(spill_dests, conn.upstream_pos, Set{Int}()), conn.downstream_pos)
    end
    cascade = CascadeLink[]
    for conn in hydro_data.upstream_turns
        d, u = conn.downstream_pos, conn.upstream_pos
        has_spill = haskey(spill_dests, u) && d in spill_dests[u]
        push!(cascade, CascadeLink(d, u, !has_spill, Float32(K * hydro_data.units[u].max_turn)))
    end
    for conn in hydro_data.upstream_spills
        d, u = conn.downstream_pos, conn.upstream_pos
        already = any(c -> c.downstream == d && c.upstream == u, cascade)
        already || push!(cascade, CascadeLink(d, u, false, Float32(K * hydro_data.units[u].max_turn)))
    end

    return HydroReachablePolicy(
        encoder, combiner,
        DecisionRulesExa._init_recurrent_state(encoder),   # initial recurrent state
        nHyd, nHyd,
        Float32.([h.min_vol for h in hydro_data.units]),
        Float32.([h.max_vol for h in hydro_data.units]),
        Float32.([h.min_turn for h in hydro_data.units]),
        Float32.([h.max_turn for h in hydro_data.units]),
        spill_vec,
        upstream_max,
        K,
        nothing, nothing,
        cascade,
        # Typed comprehensions guarantee concrete Vector{Int}/Vector{Float32}
        # element types: `getfield.(links, :field)` can infer as Vector{Real}
        # (the field symbol does not always constant-propagate through fused
        # broadcast), which breaks the struct's I/V type-parameter binding.
        Int[c.upstream for c in cascade],
        Int[c.downstream for c in cascade],
        Float32[c.turn_only for c in cascade],
        Float32[c.K_max_turn for c in cascade],
        collect(1:nHyd),
    )
end
