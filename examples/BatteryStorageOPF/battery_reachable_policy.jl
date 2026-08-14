# battery_reachable_policy.jl
#
# The strict TS-DDR policy: a recurrent encoder over the observed demand
# history, a state-conditioned head, and a differentiable affine map from the
# head's bounded output into the battery-dynamic one-stage reachable interval.
#
# Every target this policy emits is reachable BY CONSTRUCTION, which is what
# makes the hard target equality of the stage problem well posed. There is no
# target slack and no target penalty anywhere in this file or in the model it
# feeds.
#
# THE GRADIENT THAT MUST NOT BE TRUNCATED.
# The emitted target is
#
#     x̂_t = l_t(e_{t-1}) + (u_t(e_{t-1}) − l_t(e_{t-1})) ⊙ y_t ,
#
# and BOTH endpoints depend on the incoming energy with slope α wherever the
# energy bound is not the binding term. In strict mode the incoming energy IS
# the previous stage's emitted target, so
#
#     ∂x̂_t/∂e_{t-1} = α (1 − y_t)  on coordinates whose lower bound is the
#                                   power-limited one, plus α y_t on those whose
#                                   upper bound is the power-limited one,
#
# is exactly the term that couples the stages. Declaring the reachable bounds
# nondifferentiable truncates the adjoint recursion at EVERY stage, not only
# where a bound binds, and the error compounds with the horizon. The identical
# defect was measured on the hydro study's policy: it left the applied update at
# cosine 0.67 and 6 % of the correct magnitude against finite differences. The
# bounds here are therefore fully differentiable, and only the CONSTANT metadata
# adapters — which merely move frozen numbers onto the right device — are hidden
# from the AD tape.

using Flux
using Zygote
import DecisionRulesExa: load_stateconditioned_policy!

"""
    stretchedsigmoid(x) -> y ∈ [0, 1 − 1e-3]

Boundary-attaining squashing function: `clamp((σ(x) − 0.03)/0.94, 0, 1 − 1e-3)`.

# Notes
A plain sigmoid reaches 0 and 1 only in the limit, so a policy squashed by it
can never place a target exactly at an endpoint of the reachable interval —
which is where an optimal storage decision very often is (charge as hard as
possible, or discharge as hard as possible). The gentle 6.4 % stretch attains
exactly 0 for ``\\sigma(x) \\le 0.03`` while keeping the interior mapping close
to the sigmoid's shape.

The upper end stops a hair short of 1 on purpose. An exact upper endpoint forces
the charge control to sit exactly at its bound with the discharge control
exactly at zero — a measure-zero face that an interior-point solver cannot
converge into once the strict equality pins the state there. The lower endpoint
has no such problem, because the discharge control retains a strict interior
there, so 0 is attained exactly.
"""
function stretchedsigmoid(x::Real)
    T = float(typeof(x))
    return clamp((NNlib.sigmoid(x) - T(0.03)) / T(0.94), zero(T), one(T) - T(1e-3))
end

"Activations whose range lies inside [0,1], so the affine map into the reachable interval stays feasible."
const BOUNDED_ACTIVATIONS = (NNlib.sigmoid, NNlib.sigmoid_fast, NNlib.hardsigmoid, stretchedsigmoid)

"""
    BatteryReachablePolicy

A strict, state-conditioned battery policy.

# Fields
- `encoder`: recurrent chain over `[context_t; observation_t]`.
- `combiner`: feed-forward head over `[encoded_t; e_{t-1}]`, bounded output.
- `state`: the encoder's recurrent state, threaded across stages explicitly.
- `n_context::Int`: context rows prepended before the observation.
- `n_observation::Int`: observation rows (the observed demand).
- `n_battery::Int`: number of batteries, i.e. the output width.
- `energy_min`, `energy_max`, `charge_gain`, `discharge_drop`, `alpha`:
  per-battery reachability metadata (see [`reachable_bounds`](@ref)).

# Notes
The incoming energy enters TWICE: as an input to the head, and inside the
reachable bounds. Only the second of those couples the stages, and it is the one
an earlier generation of this code got wrong.

The encoder is recurrent over the observed UNCERTAINTY only; the state does not
enter the recurrent chain. Flux ≥ 0.16 recurrent cells are stateless, so the
state is threaded by hand here — calling the `LSTM` wrapper directly would
silently restart from `initialstates` every stage and produce a MEMORYLESS
encoder that trains, reduces loss, and answers a different question.
"""
mutable struct BatteryReachablePolicy{E,C,RS,V}
    encoder::E
    combiner::C
    state::RS
    n_context::Int
    n_observation::Int
    n_battery::Int
    energy_min::V
    energy_max::V
    charge_gain::V       # η^{ch} Δt \overline p^{ch}
    discharge_drop::V    # Δt \overline p^{dis} / η^{dis}
    alpha::V             # α, the per-stage retention factor
end

Flux.@layer BatteryReachablePolicy trainable=(encoder, combiner)

"""
    _adapt_metadata(x, ref) -> AbstractVector

Return the frozen metadata vector `x` with `ref`'s element type and device.

# Notes
The returned VALUES are `x`, frozen policy metadata; `ref` contributes only an
element type and a device family. The map is therefore constant in both
arguments and the `Zygote.@nograd` declaration below is EXACT rather than an
approximation. It is needed because the device adaptation goes through
`similar` + `copyto!` and Zygote refuses to differentiate array mutation —
without it the (fully differentiable) reachable bounds could not be traced at
all. This is the ONLY thing hidden from the tape in this file.
"""
function _adapt_metadata(x::AbstractVector, ref::AbstractArray)
    typeof(x) === typeof(ref) && return x
    y = similar(ref, length(x))
    copyto!(y, convert.(eltype(ref), x))
    return y
end
Zygote.@nograd _adapt_metadata

"""
    reachable_bounds(policy, e_prev, ref) -> (lower, upper)

Vectorized one-stage reachable interval for every battery.

# Arguments
- `policy::BatteryReachablePolicy`: carries the frozen reachability metadata.
- `e_prev`: incoming energy, a vector (one scenario) or a matrix (batched,
  batteries × scenarios).
- `ref`: array supplying the working element type and device.

# Returns
- `(lower, upper)`, broadcast-compatible with `e_prev`.

# Notes
This is the vectorized form of `reachable_interval` in the shared case contract:

```math
\\underline r = \\max\\{\\underline e,\\; \\alpha e_{t-1} - \\Delta t\\,
    \\overline p^{dis}/\\eta^{dis}\\},
\\qquad
\\overline r = \\min\\{\\overline e,\\; \\alpha e_{t-1} + \\eta^{ch}\\Delta t\\,
    \\overline p^{ch}\\}.
```

DIFFERENTIABLE in `e_prev`, and that is load-bearing — see the note at the top
of this file. `max`/`min` are subdifferentiable and Zygote's pullback selects the
active branch, which is the correct one-sided derivative away from the kink; the
gradient gate measures the distance to the nearest kink before it differences.

`upper` is finally clamped from below by `lower`. The two can only cross when
the battery's own bounds are inconsistent with its power ratings, which the case
verifier rejects; the clamp keeps the affine map well defined rather than
papering over data that got that far.
"""
function reachable_bounds(policy::BatteryReachablePolicy, e_prev, ref)
    e_min = _adapt_metadata(policy.energy_min, ref)
    e_max = _adapt_metadata(policy.energy_max, ref)
    gain = _adapt_metadata(policy.charge_gain, ref)
    drop = _adapt_metadata(policy.discharge_drop, ref)
    α = _adapt_metadata(policy.alpha, ref)

    decayed = α .* e_prev
    lower = max.(e_min, decayed .- drop)
    upper = min.(e_max, decayed .+ gain)
    return lower, max.(upper, lower)
end

# Row slice that behaves for a vector input (one scenario) and for a matrix
# input (features × scenarios). Plain `input[r]` on a matrix does LINEAR
# indexing and would silently corrupt a batched call.
_row_slice(input::AbstractVector, r) = input[r]
_row_slice(input::AbstractMatrix, r) = input[r, :]

"""
    policy_step(policy, state, input) -> (target, new_state)

Advance the policy by one stage, PURELY: nothing is mutated.

# Arguments
- `policy::BatteryReachablePolicy`: the trainable policy.
- `state`: the encoder's recurrent state entering this stage.
- `input`: the concatenation `[context_t; observation_t; e_{t-1}]`, either as a
  vector (one scenario) or as a matrix whose columns are scenarios.

# Returns
- `target`: the outgoing-energy target ``\\hat e_t``, guaranteed to lie in the
  one-stage reachable interval.
- `new_state`: the encoder state to carry into stage `t+1`.

# Notes
This is the form the TRAINER uses. Threading the recurrent state through the
call signature — rather than through a mutable field — keeps the multistage
rollout a pure function of the parameters, which is what lets automatic
differentiation traverse the whole recurrent chain without meeting a mutation it
must either refuse or silently drop.

The encoder consumes `[context_t; observation_t]` and advances by exactly one
step. The head consumes the encoding together with the incoming energy, and its
bounded output is mapped affinely into the interval returned by
[`reachable_bounds`](@ref).
"""
function policy_step(m::BatteryReachablePolicy, state, input)
    o_end = m.n_context + m.n_observation
    encoder_input = _row_slice(input, 1:o_end)
    e_prev = _row_slice(input, (o_end + 1):size(input, 1))

    # Cast to the encoder's precision for type stability (mixed precision here
    # has previously provoked Zygote codegen failures rather than a slowdown).
    T = DecisionRulesExa._state_eltype(state)
    h, new_state = DecisionRulesExa._step_encoder(m.encoder, T.(encoder_input), state)

    y = m.combiner(vcat(h, e_prev))
    lower, upper = reachable_bounds(m, e_prev, y)
    # `y` is bounded in [0, 1−1e-3], so the image of this affine map is inside
    # the reachable interval and the strict equality is always attainable.
    return lower .+ (upper .- lower) .* y, new_state
end

"""
    (policy::BatteryReachablePolicy)(input) -> target

Evaluate the policy, advancing its own recurrent state in place.

# Notes
The stateful convenience form of [`policy_step`](@ref), for evaluation and
interactive use. Call `Flux.reset!(policy)` at scenario boundaries: failing to
do so leaks one scenario's demand history into the next, and a `reset!` that
silently no-ops leaves the encoder memoryless — a defect that trains, reduces
loss, and answers a different question. The regression tests check that the
state actually changes across a call and actually returns to its initial value
on reset.
"""
function (m::BatteryReachablePolicy)(input)
    target, new_state = policy_step(m, m.state, input)
    m.state = new_state
    return target
end

"""
    Flux.reset!(policy::BatteryReachablePolicy)

Restore the encoder's recurrent state to `Flux.initialstates`.

# Notes
The state is re-derived from the (possibly device-moved) encoder weights on
every reset, so it always matches the encoder's device and element type. The
head is feed-forward and carries no state.
"""
function Flux.reset!(m::BatteryReachablePolicy)
    m.state = DecisionRulesExa._init_recurrent_state(m.encoder)
    return nothing
end

"""
    load_stateconditioned_policy!(policy::BatteryReachablePolicy, state)

Load checkpointed Flux parameters into a policy, keeping the case's frozen
reachability metadata.

# Notes
A checkpoint carries the trainable encoder and head only. Reachability metadata
comes from the case, never from a checkpoint: a checkpoint that could override a
battery's rating would let a stale file silently redefine the problem.

The recurrent state is reset after loading so the next rollout starts from the
loaded weights' own initial state.
"""
function load_stateconditioned_policy!(policy::BatteryReachablePolicy, state)
    Flux.loadmodel!(policy, state)
    Flux.reset!(policy)
    return policy
end

"""
    battery_reachable_policy(case, encoder_layers; n_observation, n_context=0,
                             head_layers=Int[], activation=stretchedsigmoid,
                             encoder_type=Flux.LSTM)
        -> BatteryReachablePolicy

Construct a strict reachable policy for a frozen case.

# Arguments
- `case::BatteryCase`: supplies the batteries and the stage duration.
- `encoder_layers::AbstractVector{Int}`: recurrent encoder widths.

# Keywords
- `n_observation::Int`: width of the per-stage observation.
- `n_context::Int`: deterministic context rows prepended to the observation.
- `head_layers::AbstractVector{Int}`: hidden widths of the state-conditioned
  head.
- `activation`: must be `[0,1]`-bounded; see [`BOUNDED_ACTIVATIONS`](@ref).
- `encoder_type`: recurrent layer constructor.

# Returns
- A [`BatteryReachablePolicy`](@ref) whose reachability metadata is derived from
  the case at CONSTRUCTION, so the policy and the model it feeds cannot disagree
  about the battery.
"""
function battery_reachable_policy(case::BatteryCase, encoder_layers::AbstractVector{Int};
                                  n_observation::Int,
                                  n_context::Int = 0,
                                  head_layers::AbstractVector{Int} = Int[],
                                  activation = stretchedsigmoid,
                                  encoder_type = Flux.LSTM)
    any(a -> activation === a, BOUNDED_ACTIVATIONS) || throw(ArgumentError(
        "the target head needs a [0,1]-bounded activation so targets stay inside the reachable interval"))
    n_context >= 0 || throw(ArgumentError("n_context must be nonnegative"))
    n_observation >= 1 || throw(ArgumentError("n_observation must be positive"))

    batteries = sort(collect(case.batteries); by = b -> b.index)
    nBat = length(batteries)
    Δt = stage_hours(case)

    sizes = vcat(n_context + n_observation, collect(encoder_layers))
    layers = [encoder_type(sizes[i] => sizes[i + 1]) for i in 1:length(encoder_layers)]
    encoder = Flux.Chain(layers...)
    width = isempty(encoder_layers) ? n_context + n_observation : encoder_layers[end]
    combiner = DecisionRulesExa._dense_policy_head(width + nBat, nBat,
                                                   collect(Int, head_layers);
                                                   activation = activation)

    return BatteryReachablePolicy(
        encoder, combiner,
        DecisionRulesExa._init_recurrent_state(encoder),
        n_context, n_observation, nBat,
        Float32[b.energy_min for b in batteries],
        Float32[b.energy_max for b in batteries],
        Float32[b.charge_efficiency * Δt * b.charge_max for b in batteries],
        Float32[Δt * b.discharge_max / b.discharge_efficiency for b in batteries],
        Float32[b.self_discharge for b in batteries],
    )
end
