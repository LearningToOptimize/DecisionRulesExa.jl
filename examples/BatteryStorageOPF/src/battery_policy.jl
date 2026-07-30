# battery_policy.jl
#
# Battery-SoC reachable target policy (canonical spec:
# docs/src/casestudies/battery_storage_opf.md, "Reachable target policy").
#
# The policy observes the current battery SoC e_t and the current demand
# features, and outputs the TARGET outgoing energy ê_{t+1}. It never observes
# future atoms; the recurrent encoder carries ξ_{1:t}.
#
# One-stage reachable interval (battery dynamics only — this is NOT a proof of
# ACP network feasibility):
#
#   ℓ = max( e_min , (1 − σΔt)·e_prev − (Δt/η_dis)·p̄_dis )
#   u = min( e_max , (1 − σΔt)·e_prev + η_ch·Δt·p̄_ch )
#
# and the normalized output y is mapped affinely:  ê = ℓ + (u − ℓ)·y.
#
# ── Activation (why the safe upper margin matters) ────────────────────────────
# The canonical default is `stretchedsigmoid`:
#
#   y = clamp((sigmoid(z) − 0.03)/0.94, 0, 1 − 1e-3)
#
# A plain sigmoid attains 0/1 only at ±∞ with vanishing gradient, so a squashed
# policy can never learn to sit at the reachable-interval boundary where optimal
# storage decisions frequently live. The gentle stretch attains exactly 0 at
# finite pre-activation while keeping a small margin BELOW the exact upper edge:
# an exact y = 1 (store-max) target under a STRICT equality drives the stage NLP
# onto a measure-zero set that interior-point solvers cannot converge into
# (observed as MAXIMUM_ITERATIONS / spurious INFEASIBLE). The upper normalized
# target must therefore never equal 1.
#
# `hardsigmoidsafe(z) = clamp(0.5 + 0.5z, 0, 1 − 1e-3)` is retained as a
# documented option with the same safe upper margin.
#
# Reachability bounds are physical projection DATA, not learned functions: the
# gradient stops through ℓ and u (ChainRulesCore `@non_differentiable`), and
# flows only through the normalized policy output. Recurrent state is reset at
# every scenario boundary.

using Flux
using NNlib
using ChainRulesCore
using DecisionRulesExa

"""
    stretchedsigmoid(z) -> y ∈ [0, 1 − 1e-3]

Canonical boundary-attaining activation
`clamp((sigmoid(z) − 0.03)/0.94, 0, 1 − 1e-3)`.

Attains exactly `0` for `sigmoid(z) ≤ 0.03` (z ≈ −3.5) and saturates at the
δ-interior upper value `1 − 1e-3` for `sigmoid(z) ≥ 0.969`, so the policy can
learn boundary targets at finite weights while never requesting an exact
store-max target (which is interior-point degenerate under a strict equality).
"""
function stretchedsigmoid(z::Real)
    T = float(typeof(z))
    return clamp((NNlib.sigmoid(z) - T(0.03)) / T(0.94), zero(T), one(T) - T(1e-3))
end

"""
    hardsigmoidsafe(z) -> y ∈ [0, 1 − 1e-3]

Optional piecewise-linear activation `clamp(0.5 + 0.5z, 0, 1 − 1e-3)`, with the
same safe upper margin as [`stretchedsigmoid`](@ref): it attains exactly `0` at
`z ≤ −1` and `1 − 1e-3` at `z ≥ 1`, with constant interior slope `0.5`.
"""
function hardsigmoidsafe(z::Real)
    T = float(typeof(z))
    return clamp(T(0.5) + T(0.5) * z, zero(T), one(T) - T(1e-3))
end

# Activations admissible for the target head: range inside [0, 1 − 1e-3] so the
# affine map into the reachable interval stays feasible and never hits the exact
# upper edge.
const BOUNDED_TARGET_ACTIVATIONS = (stretchedsigmoid, hardsigmoidsafe)

"""
    battery_reachable_bounds(e_prev, a, discharge_drop, charge_gain, e_min, e_max)
        -> (lower, upper)

One-stage physical reachability interval, element-wise over batteries, from
precomputed coefficients `a = 1 − σΔt`, `discharge_drop = (Δt/η_dis)·p̄_dis`,
`charge_gain = η_ch·Δt·p̄_ch`, and the energy bounds.

`upper` is additionally floored at `lower` so a degenerate interval stays
well-defined. This function is physical projection data: it is marked
non-differentiable, so no gradient propagates through the bounds.
"""
function battery_reachable_bounds(e_prev, a, discharge_drop, charge_gain, e_min, e_max)
    base = a .* e_prev
    lower = max.(e_min, base .- discharge_drop)
    upper = min.(e_max, base .+ charge_gain)
    upper = max.(upper, lower)
    return lower, upper
end

# Gradients must NOT flow through the reachable bounds (canonical spec:
# "Reachability bounds are physical projection data, not learned functions").
ChainRulesCore.@non_differentiable battery_reachable_bounds(::Any, ::Any, ::Any, ::Any, ::Any, ::Any)

"""
    BatteryReachablePolicy{E,C,S,V}

Flux-compatible battery-SoC target policy: a recurrent uncertainty encoder, a
bounded-activation target head, and the affine map into the one-stage reachable
interval.

# Fields
- `encoder`: recurrent uncertainty encoder (`Chain` of `Flux.LSTM`-style layers);
  trainable.
- `combiner`: target head reading `[encoded_uncertainty; e_prev]`, whose output
  activation is a bounded target activation; trainable.
- `state`: encoder recurrent state, threaded across stages (not trainable),
  restored by `Flux.reset!` at every scenario boundary.
- `a`, `discharge_drop`, `charge_gain`, `e_min`, `e_max`: per-battery
  reachability coefficients (physical data; moved across devices with the
  policy).
- `n_uncertainty`: number of leading uncertainty features per input.
- `nbat`: number of batteries (state dimension).
"""
mutable struct BatteryReachablePolicy{E,C,S,V}
    encoder::E
    combiner::C
    state::S
    a::V
    discharge_drop::V
    charge_gain::V
    e_min::V
    e_max::V
    n_uncertainty::Int
    nbat::Int
end

Flux.@layer BatteryReachablePolicy trainable=(encoder, combiner)

"""
    (policy::BatteryReachablePolicy)(input) -> target

One-stage forward pass on `input = vcat(w_t, e_prev)`. Advances the recurrent
encoder by one step, produces the normalized target `y ∈ [0, 1 − 1e-3]`, and
maps it into the one-stage reachable interval derived from `e_prev`.
"""
function (m::BatteryReachablePolicy)(input)
    w = input[1:m.n_uncertainty]
    e_prev = input[m.n_uncertainty + 1:end]          # range index → view, GPU-safe
    F = DecisionRulesExa._state_eltype(m.state)
    h, new_state = DecisionRulesExa._step_encoder(m.encoder, F.(w), m.state)
    m.state = new_state                              # thread recurrent state
    y = m.combiner(vcat(h, e_prev))                  # normalized, in [0, 1−1e-3]
    lower, upper = battery_reachable_bounds(e_prev, m.a, m.discharge_drop,
                                            m.charge_gain, m.e_min, m.e_max)
    return lower .+ (upper .- lower) .* y
end

"""
    Flux.reset!(policy::BatteryReachablePolicy) -> Nothing

Restore the encoder's recurrent state to `Flux.initialstates`, re-derived from
the (possibly device-moved) encoder weights. Called at every scenario boundary.
"""
function Flux.reset!(m::BatteryReachablePolicy)
    m.state = DecisionRulesExa._init_recurrent_state(m.encoder)
    return nothing
end

"""
    load_battery_policy!(policy, state) -> policy

Load a Flux checkpoint state into a [`BatteryReachablePolicy`] and reset the
recurrent state, so the reloaded policy reproduces outputs exactly.
"""
function load_battery_policy!(m::BatteryReachablePolicy, state)
    Flux.loadmodel!(m, state)
    Flux.reset!(m)
    return m
end

# Cast a Flux model's parameters to the requested precision. Flux builds
# Dense/LSTM in Float32 by default; only Float64 needs an explicit cast.
_cast_model(::Type{Float32}, m) = m
_cast_model(::Type{Float64}, m) = Flux.f64(m)
_cast_model(::Type{<:AbstractFloat}, m) = m

"""
    battery_reachable_policy(case, process; dt=1.0, layers=[32,32],
        combiner_layers=Int[], activation=stretchedsigmoid,
        encoder_type=Flux.LSTM, float_type=Float32) -> BatteryReachablePolicy

Construct the reachable target policy for a [`BatteryCase`] and its
[`LoadProcess`].

The recurrent encoder reads the `nw = 1 + nregion` per-stage uncertainty
features; the head reads `[encoded_uncertainty; e_prev]` and emits a normalized
target through `activation`, which must be a bounded target activation (range
inside `[0, 1 − 1e-3]`). `dt` MUST match the `stage_hours` of the problem this
policy drives.
"""
function battery_reachable_policy(case::BatteryCase, process::LoadProcess;
                                  dt::Real = 1.0,
                                  layers::AbstractVector{<:Integer} = [32, 32],
                                  combiner_layers::AbstractVector{<:Integer} = Int[],
                                  activation = stretchedsigmoid,
                                  encoder_type = Flux.LSTM,
                                  float_type::Type{<:AbstractFloat} = Float32)
    nBat = nbattery(case)
    nBat >= 1 || error("battery_reachable_policy needs ≥ 1 battery; got $nBat")
    length(layers) >= 1 || error("layers must have ≥ 1 encoder layer")
    any(a -> activation === a, BOUNDED_TARGET_ACTIVATIONS) ||
        throw(ArgumentError("activation must be a bounded target activation with the safe " *
                            "upper margin (stretchedsigmoid or hardsigmoidsafe)"))
    nw = n_uncertainty(process)
    Δt = Float64(dt)
    validate_stage_hours(case, Δt)

    enc_sizes = vcat(nw, collect(Int, layers))
    enc_layers = [encoder_type(enc_sizes[i] => enc_sizes[i + 1]) for i in 1:length(layers)]
    encoder = _cast_model(float_type, Flux.Chain(enc_layers...))
    # Canonical head: the bounded activation is applied at every head layer,
    # including the output (DecisionRulesExa._dense_policy_head semantics).
    combiner = _cast_model(float_type,
        DecisionRulesExa._dense_policy_head(Int(layers[end]) + nBat, nBat,
                                            collect(Int, combiner_layers);
                                            activation = activation))

    a              = float_type.([1 - b.sigma * Δt for b in case.batteries])
    discharge_drop = float_type.([(Δt / b.eta_dis) * b.p_discharge_max for b in case.batteries])
    charge_gain    = float_type.([b.eta_ch * Δt * b.p_charge_max for b in case.batteries])
    e_min          = float_type.([b.e_min for b in case.batteries])
    e_max          = float_type.([b.e_max for b in case.batteries])

    return BatteryReachablePolicy(encoder, combiner,
                                  DecisionRulesExa._init_recurrent_state(encoder),
                                  a, discharge_drop, charge_gain, e_min, e_max,
                                  nw, nBat)
end

"""
    policy_initial_state(case; float_type=Float32) -> Vector

Initial battery-SoC state vector `e_init` (length `nBat`) in policy precision —
the `initial_state` argument for `train_tsddr`/`rollout_tsddr`.
"""
policy_initial_state(case::BatteryCase; float_type::Type{<:AbstractFloat} = Float32) =
    float_type.([b.e_init for b in case.batteries])
