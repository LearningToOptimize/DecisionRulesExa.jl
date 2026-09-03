# battery_reachable_policy.jl
#
# The strict storage policies of this study: a recurrent encoder over the
# observed demand history, a head, and a differentiable affine map from the
# head's bounded output into the battery-dynamic one-stage reachable interval.
#
# TWO ARCHITECTURES, ONE FEASIBILITY LAYER.
# `:tsddr_nonlinear` is the original: an LSTM encoder and a nonlinear head whose
# output activation is bounded. `:tsldr_recurrent_linear` is a structured
# recurrent parameterization of a TIME-SERIES LINEAR DECISION RULE for the
# storage-state targets: every trainable map in its encoder, its recurrent
# update and its output head is affine, so its RAW target is an affine causal
# function of the observed demand history (see [`raw_target_step`](@ref)). Both
# emit that raw target into the SAME strict feasibility layer,
# [`feasible_target`](@ref), which is what makes every target reachable by
# construction in both. The recourse variables of the stage problem are NOT
# decision rules in either architecture: they are optimized by the stage ACP.
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

raw"""
The two strict storage-target architectures of this study, as stable identifiers.

- `:tsddr_nonlinear` — an LSTM encoder over the observed demand and a nonlinear
  head that also reads the incoming energy. Its raw target is a nonlinear
  function of the history AND of the state.
- `:tsldr_recurrent_linear` — a structured recurrent parameterization of a
  time-series LINEAR decision rule for the storage-state targets:

  ```math
  h_t = A h_{t-1} + B \xi_t + b, \qquad z_t = C h_t + D \xi_t + d,
  ```

  with ``\xi_t = [\text{context}_t; \text{observation}_t]`` the stage's
  deterministic clock features concatenated with the realized per-bus active
  demand. Unrolled from ``h_0 = 0`` this is

  ```math
  z_t = \sum_{k=1}^{t} C A^{t-k} B\, \xi_k + D \xi_t
        + \Big(\sum_{j=0}^{t-1} C A^{j} b\Big) + d,
  ```

  an AFFINE, CAUSAL function of the observed history — which is what makes the
  name accurate. The trainable map does not read the incoming energy at all: the
  state enters only through the shared feasibility layer, whose interval
  endpoints are functions of ``e_{t-1}``. Reading the state in the head would
  make the raw target a nonlinear function of the history, because the incoming
  energy is the PREVIOUS stage's squashed target.

Neither architecture parameterizes the stage's recourse variables. Every
dispatch decision — generation, charge/discharge split, voltages, the two nodal
recourse injections — is optimized by the stage ACP problem; only the outgoing
storage-state target follows a decision rule.
"""
const BATTERY_ARCHITECTURES = (:tsddr_nonlinear, :tsldr_recurrent_linear)

"""
    BatteryReachablePolicy

A strict, state-conditioned battery policy.

# Fields
- `encoder`: recurrent chain over `[context_t; observation_t]`.
- `combiner`: feed-forward head; see [`_head_input`](@ref) for what it reads.
- `state`: the encoder's recurrent state, threaded across stages explicitly.
- `architecture::Symbol`: one of [`BATTERY_ARCHITECTURES`](@ref).
- `n_context::Int`: context rows prepended before the observation.
- `n_observation::Int`: observation rows (the observed demand).
- `n_battery::Int`: number of batteries, i.e. the output width.
- `energy_min`, `energy_max`, `charge_gain`, `discharge_drop`, `alpha`:
  per-battery reachability metadata (see [`reachable_bounds`](@ref)).

# Notes
In `:tsddr_nonlinear` the incoming energy enters TWICE: as an input to the head,
and inside the reachable bounds. Only the second of those couples the stages,
and it is the one an earlier generation of this code got wrong. In
`:tsldr_recurrent_linear` only the second path exists, by design.

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
    architecture::Symbol
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
step. What the head consumes depends on the architecture — see
[`_head_input`](@ref) — and its output goes through [`feasible_target`](@ref),
the feasibility layer both architectures share, which maps it affinely into the
interval returned by [`reachable_bounds`](@ref).

`input` carries the incoming energy in both architectures even though the linear
one's head does not read it: it is what the reachable interval is computed from,
and that path is the one that couples the stages.
"""
function policy_step(m::BatteryReachablePolicy, state, input)
    o_end = m.n_context + m.n_observation
    encoder_input = _row_slice(input, 1:o_end)
    e_prev = _row_slice(input, (o_end + 1):size(input, 1))

    # Cast to the encoder's precision for type stability (mixed precision here
    # has previously provoked Zygote codegen failures rather than a slowdown).
    T = DecisionRulesExa._state_eltype(state)
    ξ = T.(encoder_input)
    h, new_state = DecisionRulesExa._step_encoder(m.encoder, ξ, state)

    z = m.combiner(_head_input(m, h, ξ, e_prev))
    return feasible_target(m, z, e_prev), new_state
end

"""
    _head_input(policy, h, ξ, e_prev) -> AbstractArray

The head's input, which is where the two architectures differ.

# Notes
`:tsddr_nonlinear` reads `[h_t; e_{t-1}]` — the encoding together with the
incoming state, as it always has.

`:tsldr_recurrent_linear` reads `[h_t; ξ_t]`: the encoding together with the
CURRENT observation, giving the direct feedthrough term ``D \\xi_t`` of the
decision rule. It deliberately does NOT read the incoming energy. The incoming
energy is the previous stage's squashed target, so a head that read it would
make the raw target a nonlinear function of the observed history and the word
"linear" in the architecture's name would be false. The state still reaches the
emitted target — through the reachable interval, in the feasibility layer, which
is exactly where a decision rule with feasibility restoration puts it.
"""
_head_input(m::BatteryReachablePolicy, h, ξ, e_prev) =
    m.architecture === :tsldr_recurrent_linear ? vcat(h, ξ) : vcat(h, e_prev)

"""
    _bounded_fraction(policy, z) -> y ∈ [0, 1 − 1e-3]

The squashing step of the feasibility layer.

# Notes
The composite map from the head's PRE-ACTIVATION output to the emitted target is
the same in both architectures:

```math
\\hat e_t = \\underline r_t(e_{t-1})
    + (\\overline r_t(e_{t-1}) - \\underline r_t(e_{t-1}))\\, s(z_t),
```

with ``s`` the bounded, boundary-attaining [`stretchedsigmoid`](@ref). Where the
two differ is only WHICH SIDE of the head boundary ``s`` is applied on, and that
follows from what each architecture has to promise:

- `:tsddr_nonlinear` applies its bounded activation as the head's own output
  activation, because its head is nonlinear anyway and always has. Here `z` has
  already been squashed and this step is the identity.
- `:tsldr_recurrent_linear` may not: an activation on the output head would be a
  trainable nonlinearity in the map that is required to be affine. Its head is
  affine end to end and `s` is applied HERE, in the feasibility layer, which is
  not part of the trainable history map and carries no parameters.

The squashing for the linear architecture is fixed to `stretchedsigmoid` rather
than configurable, because it belongs to the feasibility layer and not to the
head; the constructor rejects any other choice rather than silently ignoring it.
"""
_bounded_fraction(m::BatteryReachablePolicy, z) =
    m.architecture === :tsldr_recurrent_linear ? stretchedsigmoid.(z) : z

"""
    feasible_target(policy, z, e_prev) -> AbstractArray

The strict reachable-target feasibility layer, shared by both architectures.

# Arguments
- `z`: the head's raw output. Affine in the observed demand history for
  `:tsldr_recurrent_linear`; already bounded in `[0, 1−1e-3]` for
  `:tsddr_nonlinear`.
- `e_prev`: the incoming energy, whose reachable interval this maps into.

# Notes
The fraction returned by [`_bounded_fraction`](@ref) lies in `[0, 1−1e-3]`, so
the image of the affine map below is inside the interval
[`reachable_bounds`](@ref) returns and the stage problem's hard target equality
is ALWAYS attainable. There is no target slack, no target penalty and no
projection that could silently move a target: a target this returns is reachable
by construction, not by repair.

Both endpoints depend differentiably on `e_prev`, which is the term that couples
consecutive stages — see the note at the top of this file.
"""
function feasible_target(m::BatteryReachablePolicy, z, e_prev)
    y = _bounded_fraction(m, z)
    lower, upper = reachable_bounds(m, e_prev, y)
    return lower .+ (upper .- lower) .* y
end

"""
    raw_target_step(policy, state, observation) -> (z, new_state)

Advance the LINEAR decision rule by one stage and return its RAW target, before
the feasibility layer.

# Arguments
- `observation`: ``\\xi_t = [\\text{context}_t; \\text{observation}_t]`` alone —
  no incoming energy, because the rule does not read one.

# Returns
- `z`: ``z_t = C h_t + D \\xi_t + d``, affine in ``\\xi_1,\\ldots,\\xi_t``.
- `new_state`: ``h_t``.

# Notes
Defined only for `:tsldr_recurrent_linear`. For `:tsddr_nonlinear` a "raw target
as a function of the observed history" does not exist as a separate object: that
head reads the incoming energy, so its output depends on the whole feedback loop
and calling this would return something whose name would be wrong. It therefore
raises rather than returning a plausible number.

This is the map the causality, history, affinity and explicit-unrolling gates
are stated about, so it is a first-class function rather than something a test
reconstructs from the layers — a test that rebuilt the map would be checking its
own arithmetic.
"""
function raw_target_step(m::BatteryReachablePolicy, state, observation)
    m.architecture === :tsldr_recurrent_linear || throw(ArgumentError(
        "raw_target_step is defined for :tsldr_recurrent_linear; architecture :$(m.architecture) " *
        "reads the incoming energy, so it has no raw target that is a function of the history alone"))
    T = DecisionRulesExa._state_eltype(state)
    ξ = T.(observation)
    h, new_state = DecisionRulesExa._step_encoder(m.encoder, ξ, state)
    return m.combiner(vcat(h, ξ)), new_state
end

"""
    policy_activations(policy) -> Vector{Function}

Every function-valued field reachable inside the policy's trainable layers.

# Notes
The activation audit's first half. A layer stores its activation in an ordinary
field, so collecting every `Function` in the trainable tree finds all of them
without a hard-coded list of layer types — including one introduced by a future
refactor, which a hard-coded list would silently miss.

It is only half the audit, and the weaker half. `Flux.LSTMCell` has no
activation FIELD at all: its sigmoid gates and its `tanh` are written into the
cell's forward pass, so an audit that only read fields would pronounce an LSTM
encoder linear. [`policy_recurrent_cells`](@ref) is the other half, and it is the
one that matters.
"""
policy_activations(m::BatteryReachablePolicy) =
    _collect_activations!(Function[], (m.encoder, m.combiner))

# Recurse into anything that is not a leaf, collecting function-valued fields.
# Numbers, arrays, symbols and strings are leaves and terminate the walk; a
# `bias` of `false` is a `Bool` and lands there rather than being mistaken for
# something structural.
function _collect_activations!(out::Vector{Function}, x)
    x isa Function && (push!(out, x); return out)
    (x isa AbstractArray || x isa Number || x isa Symbol ||
     x isa AbstractString || x === nothing) && return out
    if x isa Tuple || x isa NamedTuple
        for v in x
            _collect_activations!(out, v)
        end
        return out
    end
    for f in fieldnames(typeof(x))
        _collect_activations!(out, getfield(x, f))
    end
    return out
end

"""
    policy_recurrent_cells(policy) -> Vector

The underlying recurrent CELL of every layer of the policy's encoder.

# Notes
The decisive half of the activation audit: what makes an LSTM nonlinear is its
cell type, not a field. `Flux.RNNCell` is the only stock Flux cell whose whole
update is `σ.(Wi*x + Wh*h + b)` with `σ` a settable field, so "every encoder cell
is an `RNNCell` and every one of their `σ` is `identity`" is the statement that
actually establishes an affine recurrence.
"""
policy_recurrent_cells(m::BatteryReachablePolicy) =
    m.encoder isa Flux.Chain ?
        [DecisionRulesExa._as_cell(l) for l in m.encoder.layers] :
        [DecisionRulesExa._as_cell(m.encoder)]

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

An architecture mismatch is caught here as well as by the checkpoint metadata:
`Flux.loadmodel!` compares the two structures, and an `LSTMCell` and an
`RNNCell` of the same declared width carry `Wi` blocks of different shape. That
is a structural accident, not a guarantee — the guarantee is the explicit
architecture check in `load_checkpoint!`.
"""
function load_stateconditioned_policy!(policy::BatteryReachablePolicy, state)
    Flux.loadmodel!(policy, state)
    Flux.reset!(policy)
    return policy
end

"""
    _battery_policy_head(input_dim, output_dim, hidden; hidden_activation,
                         output_activation) -> Flux.Dense or Flux.Chain

The target head, built so that the hidden and the OUTPUT activation can differ.

# Notes
`DecisionRulesExa._dense_policy_head` deliberately applies one activation at
every layer including the output, which is what a bounded nonlinear head wants.
The linear architecture needs `identity` everywhere, and stating that as two
separate arguments — rather than as one activation that happens to be
`identity` — is what makes the audit's claim about the OUTPUT head checkable
independently of the hidden layers.

The layer construction order is identical to the shared helper's, so a head
built here from the same random stream has the same parameters as one built
there.
"""
function _battery_policy_head(input_dim::Int, output_dim::Int, hidden::AbstractVector{Int};
                              hidden_activation, output_activation)
    isempty(hidden) && return Flux.Dense(input_dim => output_dim, output_activation)
    layers = Any[Flux.Dense(input_dim => hidden[1], hidden_activation)]
    for i in 1:(length(hidden) - 1)
        push!(layers, Flux.Dense(hidden[i] => hidden[i + 1], hidden_activation))
    end
    push!(layers, Flux.Dense(hidden[end] => output_dim, output_activation))
    return Flux.Chain(layers...)
end

"""
    battery_reachable_policy(case, encoder_layers; n_observation, n_context=0,
                             head_layers=Int[], activation=stretchedsigmoid,
                             encoder_type=nothing,
                             architecture=:tsddr_nonlinear)
        -> BatteryReachablePolicy

Construct a strict reachable policy for a frozen case.

# Arguments
- `case::BatteryCase`: supplies the batteries and the stage duration.
- `encoder_layers::AbstractVector{Int}`: recurrent encoder widths.

# Keywords
- `n_observation::Int`: width of the per-stage observation.
- `n_context::Int`: deterministic context rows prepended to the observation.
- `head_layers::AbstractVector{Int}`: hidden widths of the head.
- `activation`: must be `[0,1]`-bounded; see [`BOUNDED_ACTIVATIONS`](@ref). It is
  the head's output activation for `:tsddr_nonlinear` and the feasibility
  layer's squashing for `:tsldr_recurrent_linear`, which is why the latter
  accepts only `stretchedsigmoid`.
- `encoder_type`: recurrent layer constructor, or `nothing` for the
  architecture's own — `Flux.LSTM` for `:tsddr_nonlinear`, and for
  `:tsldr_recurrent_linear` a `Flux.RNN` with `identity`, whose cell update is
  exactly ``h_t = A h_{t-1} + B \\xi_t + b``.
- `architecture::Symbol`: one of [`BATTERY_ARCHITECTURES`](@ref).

# Returns
- A [`BatteryReachablePolicy`](@ref) whose reachability metadata is derived from
  the case at CONSTRUCTION, so the policy and the model it feeds cannot disagree
  about the battery.

# Notes
`:tsldr_recurrent_linear` rejects an explicit `encoder_type` rather than
accepting one and hoping it is affine. The whole claim the architecture makes is
about which maps are affine, and a caller who could pass `Flux.GRU` and still
get a policy that called itself linear would make the name meaningless.

Its head layers compose affinely, so `head_layers` widths are a reparameterization
rather than extra capacity — a rank restriction if a width is narrow. They are
accepted so that the two architectures can be given the same shape arguments.
"""
function battery_reachable_policy(case::BatteryCase, encoder_layers::AbstractVector{Int};
                                  n_observation::Int,
                                  n_context::Int = 0,
                                  head_layers::AbstractVector{Int} = Int[],
                                  activation = stretchedsigmoid,
                                  encoder_type = nothing,
                                  architecture::Symbol = :tsddr_nonlinear)
    architecture in BATTERY_ARCHITECTURES || throw(ArgumentError(
        "architecture must be one of $BATTERY_ARCHITECTURES, got :$architecture"))
    any(a -> activation === a, BOUNDED_ACTIVATIONS) || throw(ArgumentError(
        "the target head needs a [0,1]-bounded activation so targets stay inside the reachable interval"))
    n_context >= 0 || throw(ArgumentError("n_context must be nonnegative"))
    n_observation >= 1 || throw(ArgumentError("n_observation must be positive"))

    linear = architecture === :tsldr_recurrent_linear
    if linear
        encoder_type === nothing || throw(ArgumentError(
            ":tsldr_recurrent_linear builds its own affine recurrence; encoder_type is not selectable"))
        activation === stretchedsigmoid || throw(ArgumentError(
            ":tsldr_recurrent_linear squashes in the feasibility layer, which is fixed to stretchedsigmoid"))
    end
    cell = encoder_type === nothing ?
           (linear ? (p -> Flux.RNN(p, identity)) : Flux.LSTM) : encoder_type

    batteries = sort(collect(case.batteries); by = b -> b.index)
    nBat = length(batteries)
    Δt = stage_hours(case)

    sizes = vcat(n_context + n_observation, collect(encoder_layers))
    layers = [cell(sizes[i] => sizes[i + 1]) for i in 1:length(encoder_layers)]
    encoder = Flux.Chain(layers...)
    width = isempty(encoder_layers) ? n_context + n_observation : encoder_layers[end]
    # The head reads the encoding plus either the incoming energy (nonlinear) or
    # the current observation (linear) — see `_head_input`.
    head_in = width + (linear ? n_context + n_observation : nBat)
    combiner = linear ?
        _battery_policy_head(head_in, nBat, collect(Int, head_layers);
                             hidden_activation = identity, output_activation = identity) :
        DecisionRulesExa._dense_policy_head(head_in, nBat, collect(Int, head_layers);
                                            activation = activation)

    return BatteryReachablePolicy(
        encoder, combiner,
        DecisionRulesExa._init_recurrent_state(encoder),
        architecture,
        n_context, n_observation, nBat,
        Float32[b.energy_min for b in batteries],
        Float32[b.energy_max for b in batteries],
        Float32[b.charge_efficiency * Δt * b.charge_max for b in batteries],
        Float32[Δt * b.discharge_max / b.discharge_efficiency for b in batteries],
        Float32[b.self_discharge for b in batteries],
    )
end
