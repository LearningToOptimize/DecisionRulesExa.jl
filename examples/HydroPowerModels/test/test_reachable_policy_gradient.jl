# Regression tests for the DIFFERENTIABLE reachable-target policy (ExaModels engine).
#
# Every test here fails under the previous implementation, in which
# `_hydro_reachable_bounds` and `_cascade_upper_bounds` carried `Zygote.@nograd`:
#
#   * the reachable interval's dependence on the previous reservoir state was
#     dropped, so the adjoint recursion was truncated at EVERY stage — measured
#     at the production operating point as cosine 0.673 / norm ratio 0.059
#     against the true gradient, worsening with the horizon;
#   * only the constant device/dtype adapters `_hydro_adapt_bound` and
#     `_hydro_adapt_index` remain `@nograd`, and that is EXACT: they return
#     frozen metadata and take their element type and device from a reference
#     array. Those tests below pin that they stay non-differentiable and that
#     removing their annotations is not required for the bounds to be traced.
#
# The mirror of this file in DecisionRules.jl asserts the SAME reference values
# (`hydro_reachable_reference.jl` is byte-identical in both packages), so the two
# engines are checked against a common oracle rather than against each other.
#
# Usage:
#   julia --project=examples/HydroPowerModels examples/HydroPowerModels/test_reachable_policy_gradient.jl

using Test
using Random
using LinearAlgebra
using Flux
using Zygote
using ExaModels
using MadNLP
using DecisionRulesExa

const SCRIPT_DIR = dirname(dirname(@__FILE__))   # examples/HydroPowerModels
# `CascadeLink` is declared in the ExaModels problem builder, and
# `hydro_reachable_policy.jl` refers to it at load time, so the builder must be
# included first even though no problem is built here.
include(joinpath(SCRIPT_DIR, "hydro_power_data.jl"))
include(joinpath(SCRIPT_DIR, "hydro_power_exa.jl"))
include(joinpath(SCRIPT_DIR, "hydro_reachable_policy.jl"))
include(joinpath(SCRIPT_DIR, "hydro_reachable_reference.jl"))
const REF = HydroReachableReference

"""
    build_policy(; spill_max = nothing, seed = 20260802) -> HydroReachablePolicy

A small reachable policy over the synthetic reference cascade.

The struct is built directly rather than through `hydro_reachable_policy`,
which expects a full `HydroData` parsed from a case: these tests exercise the
physics path and must not require case files.
"""
function build_policy(; spill_max = nothing, seed = 20260802)
    Random.seed!(seed)
    nhyd = REF.NHYD
    links = REF.cascade_links()
    cascade = CascadeLink[
        CascadeLink(l.downstream, l.upstream, l.turn_only, Float32(l.K_max_turn))
        for l in links
    ]
    encoder = Flux.Chain(Flux.LSTM(nhyd => 4))
    combiner = DecisionRulesExa._dense_policy_head(
        4 + nhyd, nhyd, Int[]; activation = stretchedsigmoid,
    )
    return HydroReachablePolicy(
        encoder,
        combiner,
        DecisionRulesExa._init_recurrent_state(encoder),
        0, nhyd, nhyd,
        Float32.(REF.MIN_VOL),
        Float32.(REF.MAX_VOL),
        Float32.(REF.MIN_TURN),
        Float32.(REF.MAX_TURN),
        spill_max === nothing ? nothing : Float32.(collect(spill_max)),
        Float32.(REF.upstream_max()),
        REF.K,
        nothing, nothing,
        cascade,
        Int[c.upstream for c in cascade],
        Int[c.downstream for c in cascade],
        Float32[c.turn_only for c in cascade],
        Float32[c.K_max_turn for c in cascade],
        collect(1:nhyd),
    )
end

"""
    target_map(policy, inflow, x_prev, y; freeze_bounds) -> Vector

The policy's target map with the head output `y` held FIXED, so the only path
from `x_prev` to the target is the physics one under test.

# Keywords
- `freeze_bounds::Bool`: when `true`, `x_prev` reaches the bounds and the
  cascade clamp through `Zygote.dropgrad`, reproducing exactly what the removed
  `@nograd` annotations did. The FORWARD value is identical either way; only the
  pullback differs.
"""
function target_map(policy, inflow, x_prev, y; freeze_bounds::Bool)
    xb = freeze_bounds ? Zygote.dropgrad(x_prev) : x_prev
    lower, upper = _hydro_reachable_bounds(policy, inflow, xb, y)
    raw = lower .+ (upper .- lower) .* y
    isempty(policy.cascade) && return raw
    return min.(raw, _cascade_upper_bounds(policy, raw, inflow, xb))
end

"""
    gradient_or_zeros(f, x) -> Vector

`Zygote.gradient(f, x)`, with a `nothing` result materialized as zeros.

Zygote returns `nothing` — not a zero vector — when NO differentiable path
reaches `x`. That is precisely the outcome the removed `@nograd` annotations
produced, so the distinction must be handled rather than crashed on.
"""
function gradient_or_zeros(f, x)
    g = only(Zygote.gradient(f, x))
    return g === nothing ? zeros(eltype(x), length(x)) : collect(g)
end

@testset "reachable policy — differentiable bounds (Exa)" begin

    # ── 1. Forward parity on representative cascade metadata ─────────────────
    @testset "forward parity against the original mutating loop" begin
        policy = build_policy()
        @test length(policy.cascade) == length(REF.cascade_links())
        for point in REF.OPERATING_POINTS, y in REF.Y_POINTS
            x_prev, inflow = point.x_prev, point.inflow

            lower, upper = _hydro_reachable_bounds(policy, inflow, x_prev, y)
            ref_lower, ref_upper = REF.reachable_bounds(x_prev, inflow)
            @test lower ≈ ref_lower rtol = REF.PARITY_RTOL
            @test upper ≈ ref_upper rtol = REF.PARITY_RTOL

            raw = lower .+ (upper .- lower) .* y
            cascade_upper = _cascade_upper_bounds(policy, raw, inflow, x_prev)
            @test cascade_upper ≈ REF.cascade_upper_reference(raw, inflow, x_prev) rtol = REF.PARITY_RTOL
            @test isinf(cascade_upper[1])     # no incoming link => untouched by `min`

            emitted = min.(raw, cascade_upper)
            @test emitted ≈ REF.emitted_target(x_prev, inflow, y) rtol = REF.PARITY_RTOL
            REF.check_invariants(emitted, lower, upper)
        end
    end

    @testset "forward parity with a finite spill cap" begin
        spill_max = [0.4, 0.3, 0.2]
        policy = build_policy(; spill_max = spill_max)
        for point in REF.OPERATING_POINTS
            y = REF.Y_POINTS[1]
            lower, upper = _hydro_reachable_bounds(policy, point.inflow, point.x_prev, y)
            ref_lower, ref_upper =
                REF.reachable_bounds(point.x_prev, point.inflow; spill_max = spill_max)
            @test lower ≈ ref_lower rtol = REF.PARITY_RTOL
            @test upper ≈ ref_upper rtol = REF.PARITY_RTOL
        end
    end

    # ── 2. Finite differences through a state-dependent upper bound ───────────
    @testset "finite differences — upper bound" begin
        policy = build_policy()
        weights = [0.7, -1.3, 0.4]
        for point in REF.OPERATING_POINTS
            inflow, y = point.inflow, REF.Y_POINTS[1]
            f = x -> sum(weights .* _hydro_reachable_bounds(policy, inflow, x, y)[2])
            ad = gradient_or_zeros(f, point.x_prev)
            @test ad ≈ REF.central_gradient(f, point.x_prev) atol = 1e-6
            @test ad ≈ weights atol = 1e-9       # unclipped => identity Jacobian
        end
    end

    @testset "finite differences — clipped upper bound" begin
        policy = build_policy()
        point, y = REF.CLIPPED_POINT, REF.Y_POINTS[1]
        _, upper = _hydro_reachable_bounds(policy, point.inflow, point.x_prev, y)
        @test upper[1] == REF.MAX_VOL[1]
        f = x -> _hydro_reachable_bounds(policy, point.inflow, x, y)[2][1]
        ad = gradient_or_zeros(f, point.x_prev)
        @test ad[1] == 0.0
        @test ad ≈ REF.central_gradient(f, point.x_prev) atol = 1e-6
    end

    # ── 3. Nonzero derivative through the previous reservoir state ────────────
    # Finite differences are compared only at points certified away from every
    # kink of the piecewise-affine map — a central difference straddling a kink
    # measures neither one-sided derivative.
    @testset "derivative through the previous reservoir state" begin
        policy = build_policy()
        multipliers = [1.1, -0.6, 0.9]
        for point in REF.SMOOTH_POINTS
            inflow, x_prev, y = point.inflow, point.x_prev, point.y
            @test REF.kink_margin(x_prev, inflow, y) > REF.MIN_KINK_MARGIN

            live = x -> sum(multipliers .* target_map(policy, inflow, x, y; freeze_bounds = false))
            frozen = x -> sum(multipliers .* target_map(policy, inflow, x, y; freeze_bounds = true))

            @test live(x_prev) == frozen(x_prev)          # forward values unchanged

            g_live = gradient_or_zeros(live, x_prev)
            g_frozen = gradient_or_zeros(frozen, x_prev)

            @test g_live ≈ REF.central_gradient(live, x_prev) atol = 1e-5
            @test norm(g_live) > 1e-3
            @test all(iszero, g_frozen)                   # what the old code gave
            @test !isapprox(g_live, g_frozen; atol = 1e-6)
        end
    end

    # ── 4. Gradient through a BINDING cascade clamp ──────────────────────────
    # At SMOOTH_POINTS[1] the 2->3 clamp binds AND the implied upstream release
    # is strictly positive, so the downstream target inherits
    # d/d(upstream target) = -1 through `release = K w + x - target`. Reservoir 1
    # is not upstream of 3 and must carry no gradient.
    @testset "gradient through a binding cascade clamp" begin
        policy = build_policy()
        point = REF.SMOOTH_POINTS[1]
        inflow, x_prev, y = point.inflow, point.x_prev, point.y

        lower, upper = _hydro_reachable_bounds(policy, inflow, x_prev, y)
        raw = lower .+ (upper .- lower) .* y
        cascade_upper = _cascade_upper_bounds(policy, raw, inflow, x_prev)
        @test cascade_upper[3] < raw[3]              # the 2->3 clamp binds
        @test raw[2] < cascade_upper[2]              # the 1->2 clamp does not

        f = t -> _cascade_upper_bounds(policy, t, inflow, x_prev)[3]
        ad = gradient_or_zeros(f, raw)
        @test ad[2] ≈ -1.0 atol = 1e-9
        @test ad[1] == 0.0
        @test ad ≈ REF.central_gradient(f, raw) atol = 1e-6

        # At SMOOTH_POINTS[2] the 1->2 link sits strictly AT its turbine cap, so
        # the same derivative is correctly zero — the `min` branch, not a
        # dropped term.
        capped = REF.SMOOTH_POINTS[2]
        lower2, upper2 = _hydro_reachable_bounds(policy, capped.inflow, capped.x_prev, capped.y)
        raw2 = lower2 .+ (upper2 .- lower2) .* capped.y
        g = t -> _cascade_upper_bounds(policy, t, capped.inflow, capped.x_prev)[2]
        ad2 = gradient_or_zeros(g, raw2)
        @test ad2[1] == 0.0
        @test ad2 ≈ REF.central_gradient(g, raw2) atol = 1e-6
    end

    # ── 5. The constant metadata adapters stay non-differentiable ────────────
    # `_hydro_adapt_bound` / `_hydro_adapt_index` mutate through `similar` +
    # `copyto!`, which Zygote cannot differentiate. They are `@nograd` and that
    # is EXACT — the returned values are frozen metadata, and the reference
    # argument supplies only an element type and a device. If either annotation
    # were removed the bounds could not be traced at all.
    @testset "constant metadata adapters are exactly non-differentiable" begin
        policy = build_policy()
        reference = [1.0, 2.0, 3.0]
        @test _hydro_adapt_bound(policy.max_vol, reference) ≈ REF.MAX_VOL
        g = Zygote.gradient(r -> sum(_hydro_adapt_bound(policy.max_vol, r)), reference)
        @test only(g) === nothing                     # no gradient path through `ref`
        @test _hydro_adapt_index(policy.cascade_upstream, [1, 2, 3]) ==
              policy.cascade_upstream
    end

    # ── 6. Reachability and box invariants over random draws ─────────────────
    @testset "reachability and box invariants" begin
        policy = build_policy()
        rng = MersenneTwister(4242)
        for _ in 1:200
            x_prev = REF.MIN_VOL .+ rand(rng, REF.NHYD) .* (REF.MAX_VOL .- REF.MIN_VOL)
            inflow = 5 .* rand(rng, REF.NHYD)
            y = rand(rng, REF.NHYD)
            lower, upper = _hydro_reachable_bounds(policy, inflow, x_prev, y)
            @test all(lower .<= upper)
            REF.check_invariants(
                target_map(policy, inflow, x_prev, y; freeze_bounds = false),
                lower, upper,
            )
        end
    end

    # ── 7. The real forward pass still runs, and its gradient is finite ──────
    @testset "end-to-end policy gradient" begin
        policy = build_policy()
        Flux.reset!(policy)
        input = Float32[0.8, 1.2, 0.4, 1.0, 0.5, 0.25]   # [inflow; x_prev]
        multipliers = Float32[1.1, -0.6, 0.9]

        emitted = policy(input)
        @test length(emitted) == REF.NHYD
        @test all(isfinite, emitted)

        Flux.reset!(policy)
        grads = Zygote.gradient(policy) do m
            sum(multipliers .* m(input))
        end
        parameter_grads = collect(Iterators.filter(
            !isnothing,
            (g for g in Flux.trainables(only(grads))),
        ))
        @test !isempty(parameter_grads)
        @test all(g -> all(isfinite, g), parameter_grads)
        @test any(g -> any(!iszero, g), parameter_grads)
    end
end
