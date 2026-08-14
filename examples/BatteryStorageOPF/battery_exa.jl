# battery_exa.jl
#
# The GPU engine's model: a multistage, strict-target battery-storage AC-OPF
# deterministic equivalent written directly in ExaModels.
#
# This is the ONLY manually written network formulation in the project, and it
# exists because there is no equally validated PowerModels-to-ExaModels bridge.
# Its correctness is not asserted — it is MEASURED, by differencing every
# physical variable against the actual `PowerModels.ACPPowerModel` that the JuMP
# engine builds from the same frozen case.
#
# What is here:
#   * a parser from the frozen `network.json` into flat, positionally indexed
#     arrays, with explicit identifier→position maps so nonconsecutive PGLib
#     component identifiers are handled correctly;
#   * the full AC polar formulation — reference angle, Ohm's law at both ends
#     with transformer taps and phase shifts, angle-difference limits,
#     apparent-power limits at BOTH branch ends, shunts, generator boxes and
#     polynomial costs, and a HARD reactive balance;
#   * the battery layer — charge/discharge controls, the state transition with
#     the outgoing energy held as a PARAMETER (strict targets), unity-power-
#     factor injection, throughput cost;
#   * the two-sided uncapped nodal active recourse, priced exactly as the JuMP
#     engine prices it.
#
# What is deliberately NOT here: any SOC-WR model. SDDP does not run through
# this engine, and TS-DDR trains and evaluates on true ACP throughout.
#
# There is exactly ONE formulation: strict. There is no soft-target, no
# penalized-target and no target-deficit variant, here or anywhere else in the
# supported workflow.
#
# STRICT-MODE INVARIANT. The energy trajectory is an ExaModels PARAMETER of
# length `(T+1)·nBat` laid out as `[e_0; \hat e_1; …; \hat e_T]`. Because it is
# a parameter, an explicit initial-condition row `e_0 = x_0` would be a
# parameter-only constraint — an all-zero Jacobian row — so it is omitted and
# the initial condition is maintained by DATA: every writer of the energy
# parameter must keep its first `nBat` entries equal to `x_0`.
# `set_energy_path!` is the only writer and it guarantees this.

using ExaModels
using MadNLP
using JSON
using LinearAlgebra
using Logging
using Printf
# The three verbs below are DecisionRulesExa generics; this file adds the
# battery problem's methods to them rather than shadowing the names, so a script
# that has the package in scope keeps one meaning for each verb.
import DecisionRulesExa: solve!, target_multipliers, solve_succeeded

# ─────────────────────────────────────────────────────────────────────────────
# Flat index helpers
#
# Every array is stage-major: entry (t, i) lives at (t-1)*n + i. The energy
# parameter is the one exception — it is indexed from stage 0 — and has its own
# helper so the off-by-one can never be re-derived by hand at a call site.
# ─────────────────────────────────────────────────────────────────────────────

@inline _bi(nBus, t, i) = (t - 1) * nBus + i          # bus-indexed, stages 1..T
@inline _gi(nGen, t, g) = (t - 1) * nGen + g          # generator-indexed
@inline _bri(nBr, t, l) = (t - 1) * nBr + l           # branch-indexed
@inline _bti(nBat, t, b) = (t - 1) * nBat + b         # battery-indexed, stages 1..T
@inline _ei(nBat, t, b) = t * nBat + b                # energy parameter, stages 0..T

# ─────────────────────────────────────────────────────────────────────────────
# Network data
# ─────────────────────────────────────────────────────────────────────────────

"""
    ExaBusData

One bus, in positional form.

# Fields
- `id::Int`: the network identifier (arbitrary, possibly nonconsecutive).
- `bus_type::Int`: 1 PQ, 2 PV, 3 reference, 4 isolated.
- `gs::Float64`, `bs::Float64`: TOTAL shunt conductance/susceptance at the bus,
  aggregated over the case's shunt table (pu).
- `vmin::Float64`, `vmax::Float64`: voltage-magnitude bounds (pu).
"""
struct ExaBusData
    id::Int
    bus_type::Int
    gs::Float64
    bs::Float64
    vmin::Float64
    vmax::Float64
end

"""
    ExaGenData

One generator, in positional form.

# Fields
- `id::Int`: network identifier.
- `bus_pos::Int`: POSITION of its bus in the bus array.
- `pmin`, `pmax`, `qmin`, `qmax`: capability box (pu), as the case declares it,
  BEFORE any per-stage availability is applied.
- `c2`, `c1`, `c0`: polynomial cost coefficients such that the generator's cost
  is ``c_2 p^2 + c_1 p + c_0`` with `p` in pu.
- `availability::Vector{Float64}`: the case's per-stage availability schedule for
  this unit, read from [`STAGE_AVAILABILITY_KEY`](@ref). EMPTY means the unit
  carries no schedule and is available in every stage — which is what every case
  built before the convention existed says, and why the field is additive.
"""
struct ExaGenData
    id::Int
    bus_pos::Int
    pmin::Float64
    pmax::Float64
    qmin::Float64
    qmax::Float64
    c2::Float64
    c1::Float64
    c0::Float64
    availability::Vector{Float64}
end

"""
    ExaBranchData

One branch, in positional form, with the raw MATPOWER/PowerModels π-model
parameters the AC polar equations need.

# Fields
- `id::Int`: network identifier.
- `f_pos::Int`, `t_pos::Int`: POSITIONS of the from/to buses.
- `br_r`, `br_x`: series resistance and reactance (pu).
- `g_fr`, `b_fr`, `g_to`, `b_to`: line-charging shunts at each end (pu).
- `tap`, `shift`: transformer turns ratio and phase shift (rad).
- `rate_a`: apparent-power limit (pu).
- `angmin`, `angmax`: angle-difference limits (rad).
"""
struct ExaBranchData
    id::Int
    f_pos::Int
    t_pos::Int
    br_r::Float64
    br_x::Float64
    g_fr::Float64
    b_fr::Float64
    g_to::Float64
    b_to::Float64
    tap::Float64
    shift::Float64
    rate_a::Float64
    angmin::Float64
    angmax::Float64
end

"""
    ExaNetwork

The frozen network in the flat, positional form the ExaModels builder needs.

# Fields
- `buses`, `gens`, `branches`: sorted by network identifier.
- `bus_pos::Dict{Int,Int}`: identifier → position. Nothing anywhere assumes an
  identifier equals a position.
- `ref_bus_positions::Vector{Int}`: positions of the reference buses.
- `baseMVA::Float64`.
- `nominal_pd`, `nominal_qd::Vector{Float64}`: nominal per-bus demand (pu),
  aggregated over the case's load table and indexed by bus POSITION.

# Notes
Sorting by identifier is what makes the layout reproducible: dictionary
iteration order is not, and a variable vector whose meaning depends on hash
order cannot be compared with another engine's.

Inactive components are dropped exactly as PowerModels drops them — buses with
`bus_type == 4`, generators with `gen_status == 0`, branches with
`br_status == 0`, loads and shunts with `status == 0` — so both engines see the
same system.
"""
struct ExaNetwork
    buses::Vector{ExaBusData}
    gens::Vector{ExaGenData}
    branches::Vector{ExaBranchData}
    bus_pos::Dict{Int,Int}
    ref_bus_positions::Vector{Int}
    baseMVA::Float64
    nominal_pd::Vector{Float64}
    nominal_qd::Vector{Float64}
end

nbus(net::ExaNetwork) = length(net.buses)
ngen(net::ExaNetwork) = length(net.gens)
nbranch(net::ExaNetwork) = length(net.branches)

# ─────────────────────────────────────────────────────────────────────────────
# Per-stage generator availability
#
# The case may declare, for any generator, a per-stage multiplier on its whole
# capability box (see `STAGE_AVAILABILITY_KEY` in `battery_case.jl`). The JuMP
# engine applies it to the parsed network just before PowerModels instantiates a
# stage; this engine builds every stage of the horizon at once, so it applies it
# to the VARIABLE BOUNDS of each stage's generator block instead. The two
# statements are the same statement — `pmin`, `pmax`, `qmin` and `qmax` scaled by
# the same multiplier — made in the only place each engine has to make it.
#
# The schedule is DATA. It reaches the model through `lvar`/`uvar` and through
# nothing else: no objective term, no constraint coefficient, no parameter. That
# is what keeps it off every automatic-differentiation path in the trainer, whose
# gradient flows through the energy parameter and the multipliers only.
# ─────────────────────────────────────────────────────────────────────────────

"""
    gen_availability(gen) -> Vector{Float64}

The per-stage availability schedule declared by one raw generator row, or an
empty vector when it declares none.

# Notes
Rejected, rather than repaired: a schedule that is not a non-empty vector, a
multiplier that is not finite, and a multiplier outside `[0, 1]`. Availability is
a FRACTION of a declared capability — a value above one would silently give a
unit more capacity than the case says it has, and the case is the only place
capacity may be stated.
"""
function gen_availability(gen::AbstractDict)
    haskey(gen, STAGE_AVAILABILITY_KEY) || return Float64[]
    sched = gen[STAGE_AVAILABILITY_KEY]
    (sched isa AbstractVector && !isempty(sched)) ||
        error("generator $(get(gen, "index", "?")): \"$STAGE_AVAILABILITY_KEY\" must be a non-empty vector of multipliers")
    av = Float64.(collect(sched))
    all(a -> isfinite(a) && 0 <= a <= 1, av) ||
        error("generator $(get(gen, "index", "?")): availability multipliers must be finite and in [0, 1], got $av")
    return av
end

"""
    availability_at(g::ExaGenData, t::Integer) -> Float64

Generator `g`'s availability multiplier at stage `t`.

# Notes
Fails closed on a schedule that does not cover `t`, exactly as the JuMP engine
does: a case declaring a two-stage schedule that is then built over three stages
has been mixed up with a different case, and reusing the last entry would hide
that. A unit carrying no schedule is available in every stage, and the multiplier
returned for it is exactly `1.0` — which the bound builder recognises and skips,
so a case without schedules produces bit-identical bounds to one built before
this convention existed.
"""
@inline function availability_at(g::ExaGenData, t::Integer)
    isempty(g.availability) && return 1.0
    t <= length(g.availability) ||
        error("generator $(g.id): \"$STAGE_AVAILABILITY_KEY\" covers $(length(g.availability)) stages but stage $t was requested")
    return g.availability[t]
end

# Scale one bound, leaving it untouched at full availability. `a == 1` is exact
# for the multiplier written by a case that means "available", and the branch
# also keeps `±Inf` bounds out of a `Inf * 0 == NaN`.
@inline _avail_scale(v::Real, a::Real) = a == 1 ? Float64(v) : Float64(v) * Float64(a)

"""
    exa_network(case::BatteryCase) -> ExaNetwork

Build the flat positional network from a frozen case.

# Notes
No PowerModels call appears here or anywhere else in this engine: the frozen
`network.json` is the interface between the two engines, and it holds the case
exactly as PGLib/PowerModels parsed it.

Generator cost coefficients are read from PowerModels' highest-order-first
polynomial with `ncost` terms and re-expressed as ``(c_2, c_1, c_0)``. A model
with more than three terms is rejected rather than truncated.
"""
function exa_network(case::BatteryCase)
    data = case.network

    bus_rows = sort!([b for (_, b) in data["bus"] if Int(b["bus_type"]) != 4];
                     by = b -> Int(b["index"]))
    bus_pos = Dict{Int,Int}(Int(b["index"]) => i for (i, b) in enumerate(bus_rows))

    # Shunts live in their own table; a bus may carry several.
    gs = zeros(Float64, length(bus_rows))
    bs = zeros(Float64, length(bus_rows))
    for (_, sh) in get(data, "shunt", Dict{String,Any}())
        Int(get(sh, "status", 1)) == 0 && continue
        p = get(bus_pos, Int(sh["shunt_bus"]), 0)
        p == 0 && continue
        gs[p] += Float64(get(sh, "gs", 0.0))
        bs[p] += Float64(get(sh, "bs", 0.0))
    end

    buses = [ExaBusData(Int(b["index"]), Int(b["bus_type"]), gs[i], bs[i],
                        Float64(get(b, "vmin", 0.9)), Float64(get(b, "vmax", 1.1)))
             for (i, b) in enumerate(bus_rows)]
    ref_positions = [i for (i, b) in enumerate(buses) if b.bus_type == 3]
    isempty(ref_positions) && error("exa_network: the case declares no reference bus")

    gen_rows = sort!([g for (_, g) in data["gen"]
                      if Int(get(g, "gen_status", 1)) != 0 &&
                         haskey(bus_pos, Int(g["gen_bus"]))];
                     by = g -> Int(g["index"]))
    gens = ExaGenData[]
    for g in gen_rows
        Int(get(g, "model", 2)) == 2 ||
            error("exa_network: generator $(g["index"]) has a non-polynomial cost model")
        cost = Float64.(g["cost"])
        n = Int(g["ncost"])
        n <= 3 || error("exa_network: generator $(g["index"]) has a degree-$(n-1) cost polynomial")
        tail = cost[(end - n + 1):end]
        c2 = n >= 3 ? tail[end - 2] : 0.0
        c1 = n >= 2 ? tail[end - 1] : 0.0
        c0 = n >= 1 ? tail[end] : 0.0
        av = gen_availability(g)
        pmin, pmax = Float64(get(g, "pmin", 0.0)), Float64(g["pmax"])
        qmin, qmax = Float64(get(g, "qmin", -Inf)), Float64(get(g, "qmax", Inf))
        # Bound consistency is checked for SCHEDULED units only. Scaling by a
        # nonnegative multiplier preserves the order of an interval, so a
        # schedule can never create an inconsistency the case did not already
        # carry; and checking every unit would let this engine REJECT a case that
        # built before the convention existed, which the additivity requirement
        # forbids. A scheduled unit is new data, so it is checked where it is read.
        if !isempty(av)
            pmin <= pmax ||
                error("generator $(g["index"]): scheduled unit has pmin $pmin above pmax $pmax")
            qmin <= qmax ||
                error("generator $(g["index"]): scheduled unit has qmin $qmin above qmax $qmax")
        end
        push!(gens, ExaGenData(Int(g["index"]), bus_pos[Int(g["gen_bus"])],
                               pmin, pmax, qmin, qmax, c2, c1, c0, av))
    end

    br_rows = sort!([b for (_, b) in data["branch"]
                     if Int(get(b, "br_status", 1)) != 0 &&
                        haskey(bus_pos, Int(b["f_bus"])) && haskey(bus_pos, Int(b["t_bus"]))];
                    by = b -> Int(b["index"]))
    branches = ExaBranchData[]
    for b in br_rows
        tap = Float64(get(b, "tap", 1.0))
        tap = tap ≈ 0 ? 1.0 : tap          # PowerModels treats a zero tap as 1
        push!(branches, ExaBranchData(
            Int(b["index"]), bus_pos[Int(b["f_bus"])], bus_pos[Int(b["t_bus"])],
            Float64(get(b, "br_r", 0.0)), Float64(b["br_x"]),
            Float64(get(b, "g_fr", 0.0)), Float64(get(b, "b_fr", 0.0)),
            Float64(get(b, "g_to", 0.0)), Float64(get(b, "b_to", 0.0)),
            tap, Float64(get(b, "shift", 0.0)),
            Float64(get(b, "rate_a", Inf)),
            Float64(get(b, "angmin", -pi)), Float64(get(b, "angmax", pi))))
    end

    pd = zeros(Float64, length(buses))
    qd = zeros(Float64, length(buses))
    for (_, l) in data["load"]
        Int(get(l, "status", 1)) == 0 && continue
        p = get(bus_pos, Int(l["load_bus"]), 0)
        p == 0 && continue
        pd[p] += Float64(l["pd"])
        qd[p] += Float64(l["qd"])
    end

    return ExaNetwork(buses, gens, branches, bus_pos, ref_positions,
                      Float64(data["baseMVA"]), pd, qd)
end

"""
    branch_coefficients(br, T) -> NamedTuple

Precompute the eight AC-polar branch coefficients in element type `T`.

# Notes
With ``t_r = \\tau\\cos\\theta_s``, ``t_i = \\tau\\sin\\theta_s``,
``t_m = t_r^2 + t_i^2``, ``g + jb = 1/(r + jx)``:

```math
\\begin{aligned}
c_1 &= (-g t_r - b t_i)/t_m, & c_2 &= (-b t_r + g t_i)/t_m,\\\\
c_3 &= (-g t_r + b t_i)/t_m, & c_4 &= (-b t_r - g t_i)/t_m,\\\\
c_5 &= (g + g^{fr})/t_m,     & c_6 &= (b + b^{fr})/t_m,\\\\
c_7 &= g + g^{to},           & c_8 &= b + b^{to},
\\end{aligned}
```

so that, writing ``\\theta = \\theta_f - \\theta_t``,

```math
\\begin{aligned}
p^{fr} &= c_5 v_f^2 + c_3 v_f v_t \\cos\\theta + c_4 v_f v_t \\sin\\theta,\\\\
q^{fr} &= -c_6 v_f^2 - c_4 v_f v_t \\cos\\theta + c_3 v_f v_t \\sin\\theta,\\\\
p^{to} &= c_7 v_t^2 + c_1 v_t v_f \\cos(-\\theta) + c_2 v_t v_f \\sin(-\\theta),\\\\
q^{to} &= -c_8 v_t^2 - c_2 v_t v_f \\cos(-\\theta) + c_1 v_t v_f \\sin(-\\theta).
\\end{aligned}
```

This is PowerModels' `constraint_ohms_yt_from`/`_to` for the polar AC form,
transformer taps and phase shifts included. A zero `t_m` (only reachable from a
degenerate tap) is replaced by 1 so the model builds; the case verifier rejects
such data upstream.
"""
function branch_coefficients(br::ExaBranchData, ::Type{T}) where {T}
    r2x2 = br.br_r^2 + br.br_x^2
    g = r2x2 > 0 ? T(br.br_r / r2x2) : zero(T)
    b = r2x2 > 0 ? T(-br.br_x / r2x2) : zero(T)
    tr = T(br.tap) * cos(T(br.shift))
    ti = T(br.tap) * sin(T(br.shift))
    ttm = tr^2 + ti^2
    ttm = ttm > 0 ? ttm : one(T)
    return (c1 = (-g * tr - b * ti) / ttm,
            c2 = (-b * tr + g * ti) / ttm,
            c3 = (-g * tr + b * ti) / ttm,
            c4 = (-b * tr - g * ti) / ttm,
            c5 = (g + T(br.g_fr)) / ttm,
            c6 = (b + T(br.b_fr)) / ttm,
            c7 = g + T(br.g_to),
            c8 = b + T(br.b_to))
end

# ─────────────────────────────────────────────────────────────────────────────
# The deterministic equivalent
# ─────────────────────────────────────────────────────────────────────────────

"""
    BatteryExaProblem

A `T`-stage strict-target battery-storage AC-OPF deterministic equivalent.

# Fields
- `core`, `model`: the `ExaCore` and the built `ExaModel`.
- `net::ExaNetwork`, `case::BatteryCase`: the data the model was built from.
- `batteries::Vector{BatterySpec}`: sorted by identifier; battery `b` occupies
  position `b` in every flat battery array.
- `p_pd`, `p_qd`: per-bus per-stage demand parameters (length `T·nBus`).
- `p_energy`: the energy trajectory parameter, length `(T+1)·nBat`, laid out
  `[e_0; \\hat e_1; …; \\hat e_T]`. See the STRICT-MODE INVARIANT at the top of
  this file.
- `energy_values::Vector{Float64}`: the last trajectory written, kept so the
  solution extractor can report the state without reading a parameter back off
  a device.
- `transition_range::UnitRange{Int}`: rows of `result.multipliers` holding the
  battery state-transition duals.
- `Δt::Float64`: stage duration in hours.
- `stages::Vector{Int}`: which CASE stage each of the model's `T` positions is.
  `1:T` for a model built from the start of the horizon; anything else for a
  window. It matters only for data that is indexed by case stage rather than by
  position — today that is the per-stage generator availability, which is baked
  into the variable bounds at build time and therefore cannot be re-pointed at a
  different window afterwards.
- `horizon::Int`, `nBus`, `nGen`, `nBranch`, `nBat::Int`: sizes.
"""
struct BatteryExaProblem{C,M,P1,P2,P3}
    core::C
    model::M
    net::ExaNetwork
    case::BatteryCase
    batteries::Vector{BatterySpec}
    p_pd::P1
    p_qd::P2
    p_energy::P3
    energy_values::Vector{Float64}
    transition_range::UnitRange{Int}
    Δt::Float64
    stages::Vector{Int}
    horizon::Int
    nBus::Int
    nGen::Int
    nBranch::Int
    nBat::Int
end

"True when any generator of `prob` carries a per-stage availability schedule."
has_stage_schedule(net::ExaNetwork) = any(g -> !isempty(g.availability), net.gens)
has_stage_schedule(prob::BatteryExaProblem) = has_stage_schedule(prob.net)

"""
    assert_stage_window(prob, stages)

Check that `stages` is the window `prob` was BUILT for, and fail if it is not.

# Notes
Demand is written into parameters, so one built model can serve any window of the
horizon by re-imposing the demand — which is how the trainer solves a rolling
window. Availability is not data of that kind: it lives in the variable bounds
and is fixed when the model is built. A case with no schedule is therefore free
to be solved at any offset, exactly as before this convention existed, and a case
WITH one may only be solved on its own window. The alternative — silently solving
stage 5 with stage 1's availability — is the failure this study cannot afford.
"""
function assert_stage_window(prob::BatteryExaProblem, stages::AbstractVector{<:Integer})
    has_stage_schedule(prob) || return nothing
    collect(Int, stages) == prob.stages ||
        error("this problem was built for case stages $(prob.stages) and its generator " *
              "availability is baked into its bounds; it cannot be solved for stages " *
              "$(collect(Int, stages)). Build a problem for that window instead.")
    return nothing
end

"""
    build_battery_exa(case, T; backend=nothing, float_type=Float64, stages=1:T)
        -> BatteryExaProblem

Build the `T`-stage strict deterministic equivalent.

# Arguments
- `case::BatteryCase`: the frozen case.
- `T::Int`: horizon.

# Keywords
- `backend`: `nothing` for CPU, or a KernelAbstractions backend (e.g.
  `CUDABackend()`) for GPU.
- `float_type`: working precision. `Float64` throughout the study; the AC
  equations are ill-conditioned enough that reduced precision changes answers.
- `stages`: which CASE stage each of the `T` model positions is, `1:T` by
  default. Pass a window (`stages = 2:3`, or `[2]` for a single continuation
  problem) to build the model a case's LATER stages describe. It changes exactly
  one thing — which entry of each generator's availability schedule is applied —
  and a case that declares no schedule builds identically for every `stages`.

# Returns
- The [`BatteryExaProblem`](@ref).

# Notes
Constraint ORDER is part of the contract, because the target multipliers are a
slice of `result.multipliers`. The order is: reference angle, the four branch
flow definitions, angle-difference limits, apparent-power limits at each end,
active balance, reactive balance, and LAST the battery state transitions —
whose rows `transition_range` records.

The stage problems of a strict trajectory are COUPLED only through the energy
parameter, which is data; the deterministic equivalent is nonetheless built as
one model so a single solve produces the whole trajectory and all its
multipliers.

A generator the case schedules out of some stage keeps its variables, its cost
row and its position in every flat array — only its bounds close to zero in that
stage. The model's SHAPE is therefore the same whatever the schedule says, which
is what lets one built problem be re-solved across atoms and horizons, and it
matches the JuMP engine, which scales the same four limits rather than flipping
`gen_status` (PowerModels would drop an out-of-service unit from `ref` and change
the variable set). The constant term of a scheduled-out unit's cost polynomial is
still added, in both engines, because a polynomial cost evaluated at a pinned
zero is exactly `c_0`.
"""
function build_battery_exa(case::BatteryCase, T::Int;
                           backend = nothing,
                           float_type::Type{<:AbstractFloat} = Float64,
                           stages::AbstractVector{<:Integer} = 1:T)
    T >= 1 || throw(ArgumentError("horizon must be at least 1"))
    stage_of = collect(Int, stages)
    length(stage_of) == T ||
        throw(ArgumentError("stages must name one case stage per model position, got $(length(stage_of)) for T=$T"))
    all(>=(1), stage_of) || throw(ArgumentError("stages must be 1-based, got $stage_of"))
    net = exa_network(case)
    batteries = sort(collect(case.batteries); by = b -> b.index)
    nB, nG, nBR, nBat = nbus(net), ngen(net), nbranch(net), length(batteries)
    Δt = float_type(stage_hours(case))
    bat_pos = [net.bus_pos[b.bus] for b in batteries]
    for (b, p) in zip(batteries, bat_pos)
        p > 0 || error("battery $(b.index) sits at bus $(b.bus), which is not an active bus")
    end

    # `concrete = Val(false)` keeps the MUTABLE core, which is what
    # `ExaModels.constraint!` needs in order to add terms to an existing
    # constraint row — the nodal balances below are assembled that way. It is
    # passed explicitly because the default is scheduled to flip. ExaModels
    # emits a deprecation warning for the mutable core; it is silenced HERE
    # ONLY, around this single call, because a model build happens once per run
    # while the warning would otherwise print on every one of them.
    core = Logging.with_logger(Logging.NullLogger()) do
        ExaModels.ExaCore(float_type; backend = backend, concrete = Val(false))
    end

    # ── Variables, in the order the extractor unpacks them ───────────────────
    va = ExaModels.variable(core, T * nB)
    vm = ExaModels.variable(core, T * nB;
                            lvar = float_type.(repeat([b.vmin for b in net.buses], T)),
                            uvar = float_type.(repeat([b.vmax for b in net.buses], T)),
                            start = ones(float_type, T * nB))
    # Generator boxes are per STAGE, because the case may take a unit out of
    # service in some stages and not others (see `availability_at`). The
    # comprehension runs `t` outermost so the layout is stage-major, exactly as
    # `_gi` indexes it and exactly as the `repeat` it replaces laid it out; at
    # full availability every entry is the same Float64 as before.
    #
    # The infinite reactive bounds are substituted BEFORE scaling: a case that
    # leaves `qmin` unstated means "unlimited", and `-Inf * 0` is `NaN`, not the
    # zero an unavailable unit must have.
    pg = ExaModels.variable(core, T * nG;
        lvar = float_type[_avail_scale(g.pmin, availability_at(g, s)) for s in stage_of for g in net.gens],
        uvar = float_type[_avail_scale(g.pmax, availability_at(g, s)) for s in stage_of for g in net.gens])
    qg = ExaModels.variable(core, T * nG;
        lvar = float_type[_avail_scale(isfinite(g.qmin) ? g.qmin : -1e4, availability_at(g, s))
                          for s in stage_of for g in net.gens],
        uvar = float_type[_avail_scale(isfinite(g.qmax) ? g.qmax : 1e4, availability_at(g, s))
                          for s in stage_of for g in net.gens])
    # Branch flow boxes are ±rate_a, exactly as PowerModels' bounded branch
    # power variables are; the apparent-power disks below are what actually
    # binds, and a box alone would be a strictly weaker (square) relaxation.
    flow_lb = float_type.(repeat([-b.rate_a for b in net.branches], T))
    flow_ub = float_type.(repeat([b.rate_a for b in net.branches], T))
    p_fr = ExaModels.variable(core, T * nBR; lvar = flow_lb, uvar = flow_ub)
    q_fr = ExaModels.variable(core, T * nBR; lvar = flow_lb, uvar = flow_ub)
    p_to = ExaModels.variable(core, T * nBR; lvar = flow_lb, uvar = flow_ub)
    q_to = ExaModels.variable(core, T * nBR; lvar = flow_lb, uvar = flow_ub)

    # Two-sided physical active recourse: nonnegative, and with NO upper bound
    # of any kind. Capping `deficit` by the realized demand would destroy
    # relatively complete recourse for strict charging targets.
    deficit = ExaModels.variable(core, T * nB; lvar = float_type(0))
    surplus = ExaModels.variable(core, T * nB; lvar = float_type(0))

    p_ch = ExaModels.variable(core, T * nBat; lvar = float_type(0),
                              uvar = float_type.(repeat([b.charge_max for b in batteries], T)))
    p_dis = ExaModels.variable(core, T * nBat; lvar = float_type(0),
                               uvar = float_type.(repeat([b.discharge_max for b in batteries], T)))

    # ── Parameters ───────────────────────────────────────────────────────────
    p_pd = ExaModels.parameter(core, float_type.(repeat(net.nominal_pd, T)))
    p_qd = ExaModels.parameter(core, float_type.(repeat(net.nominal_qd, T)))
    p_energy = ExaModels.parameter(core, zeros(float_type, (T + 1) * nBat))

    coeff = [branch_coefficients(br, float_type) for br in net.branches]

    # ── Objective ────────────────────────────────────────────────────────────
    gen_items = [(t = t, g = i, c2 = float_type(g.c2), c1 = float_type(g.c1), c0 = float_type(g.c0))
                 for t in 1:T for (i, g) in enumerate(net.gens)]
    ExaModels.objective(core,
        item.c2 * pg[_gi(nG, item.t, item.g)]^2
        + item.c1 * pg[_gi(nG, item.t, item.g)]
        + item.c0
        for item in gen_items)

    rec_items = [(idx = _bi(nB, t, i),
                  cd = float_type(case.recourse.deficit),
                  cs = float_type(case.recourse.surplus))
                 for t in 1:T for i in 1:nB]
    ExaModels.objective(core,
        item.cd * deficit[item.idx] + item.cs * surplus[item.idx]
        for item in rec_items)

    if nBat > 0
        thr_items = [(idx = _bti(nBat, t, k), c = float_type(b.throughput_cost) * Δt)
                     for t in 1:T for (k, b) in enumerate(batteries)]
        ExaModels.objective(core,
            item.c * (p_ch[item.idx] + p_dis[item.idx]) for item in thr_items)
    end

    n_con = 0

    # ── 1. Reference angle ───────────────────────────────────────────────────
    ExaModels.constraint(core,
        va[_bi(nB, item.t, item.r)]
        for item in [(t = t, r = r) for t in 1:T for r in net.ref_bus_positions])
    n_con += T * length(net.ref_bus_positions)

    # ── 2-5. Branch flows at both ends ───────────────────────────────────────
    fr_items = [(t = t, l = l, f = br.f_pos, tb = br.t_pos,
                 c3 = coeff[l].c3, c4 = coeff[l].c4, c5 = coeff[l].c5, c6 = coeff[l].c6)
                for t in 1:T for (l, br) in enumerate(net.branches)]
    to_items = [(t = t, l = l, f = br.f_pos, tb = br.t_pos,
                 c1 = coeff[l].c1, c2 = coeff[l].c2, c7 = coeff[l].c7, c8 = coeff[l].c8)
                for t in 1:T for (l, br) in enumerate(net.branches)]

    ExaModels.constraint(core,
        p_fr[_bri(nBR, item.t, item.l)]
        - item.c5 * vm[_bi(nB, item.t, item.f)]^2
        - item.c3 * vm[_bi(nB, item.t, item.f)] * vm[_bi(nB, item.t, item.tb)]
          * cos(va[_bi(nB, item.t, item.f)] - va[_bi(nB, item.t, item.tb)])
        - item.c4 * vm[_bi(nB, item.t, item.f)] * vm[_bi(nB, item.t, item.tb)]
          * sin(va[_bi(nB, item.t, item.f)] - va[_bi(nB, item.t, item.tb)])
        for item in fr_items)
    ExaModels.constraint(core,
        q_fr[_bri(nBR, item.t, item.l)]
        + item.c6 * vm[_bi(nB, item.t, item.f)]^2
        + item.c4 * vm[_bi(nB, item.t, item.f)] * vm[_bi(nB, item.t, item.tb)]
          * cos(va[_bi(nB, item.t, item.f)] - va[_bi(nB, item.t, item.tb)])
        - item.c3 * vm[_bi(nB, item.t, item.f)] * vm[_bi(nB, item.t, item.tb)]
          * sin(va[_bi(nB, item.t, item.f)] - va[_bi(nB, item.t, item.tb)])
        for item in fr_items)
    ExaModels.constraint(core,
        p_to[_bri(nBR, item.t, item.l)]
        - item.c7 * vm[_bi(nB, item.t, item.tb)]^2
        - item.c1 * vm[_bi(nB, item.t, item.tb)] * vm[_bi(nB, item.t, item.f)]
          * cos(va[_bi(nB, item.t, item.tb)] - va[_bi(nB, item.t, item.f)])
        - item.c2 * vm[_bi(nB, item.t, item.tb)] * vm[_bi(nB, item.t, item.f)]
          * sin(va[_bi(nB, item.t, item.tb)] - va[_bi(nB, item.t, item.f)])
        for item in to_items)
    ExaModels.constraint(core,
        q_to[_bri(nBR, item.t, item.l)]
        + item.c8 * vm[_bi(nB, item.t, item.tb)]^2
        + item.c2 * vm[_bi(nB, item.t, item.tb)] * vm[_bi(nB, item.t, item.f)]
          * cos(va[_bi(nB, item.t, item.tb)] - va[_bi(nB, item.t, item.f)])
        - item.c1 * vm[_bi(nB, item.t, item.tb)] * vm[_bi(nB, item.t, item.f)]
          * sin(va[_bi(nB, item.t, item.tb)] - va[_bi(nB, item.t, item.f)])
        for item in to_items)
    n_con += 4 * T * nBR

    # ── 6. Angle-difference limits ───────────────────────────────────────────
    ExaModels.constraint(core,
        va[_bi(nB, item.t, item.f)] - va[_bi(nB, item.t, item.tb)]
        for item in fr_items;
        lcon = float_type.(repeat([br.angmin for br in net.branches], T)),
        ucon = float_type.(repeat([br.angmax for br in net.branches], T)))
    n_con += T * nBR

    # ── 7. Apparent-power limits at BOTH ends ────────────────────────────────
    thermal_ub = float_type.(repeat([br.rate_a^2 for br in net.branches], T))
    ExaModels.constraint(core,
        p_fr[_bri(nBR, item.t, item.l)]^2 + q_fr[_bri(nBR, item.t, item.l)]^2
        for item in fr_items;
        lcon = fill(float_type(-Inf), T * nBR), ucon = thermal_ub)
    ExaModels.constraint(core,
        p_to[_bri(nBR, item.t, item.l)]^2 + q_to[_bri(nBR, item.t, item.l)]^2
        for item in to_items;
        lcon = fill(float_type(-Inf), T * nBR), ucon = thermal_ub)
    n_con += 2 * T * nBR

    # ── 8. Active balance ────────────────────────────────────────────────────
    # Written as  pd + gs·vm² − Σpg + Σp_fr + Σp_to − p^bat − d + s = 0,
    # i.e. the plan's balance moved to one side. The battery injects
    # p^bat = p^dis − p^ch, and the recourse pair enters with opposite signs.
    kcl_p = ExaModels.constraint(core,
        p_pd[_bi(nB, item.t, item.i)] + item.gs * vm[_bi(nB, item.t, item.i)]^2
        for item in [(t = t, i = i, gs = float_type(net.buses[i].gs)) for t in 1:T for i in 1:nB])
    ExaModels.constraint!(core, kcl_p,
        item.row => -pg[item.col]
        for item in [(row = _bi(nB, t, g.bus_pos), col = _gi(nG, t, k))
                     for t in 1:T for (k, g) in enumerate(net.gens)])
    ExaModels.constraint!(core, kcl_p,
        item.row => p_fr[item.col]
        for item in [(row = _bi(nB, t, br.f_pos), col = _bri(nBR, t, l))
                     for t in 1:T for (l, br) in enumerate(net.branches)])
    ExaModels.constraint!(core, kcl_p,
        item.row => p_to[item.col]
        for item in [(row = _bi(nB, t, br.t_pos), col = _bri(nBR, t, l))
                     for t in 1:T for (l, br) in enumerate(net.branches)])
    ExaModels.constraint!(core, kcl_p,
        item.row => -deficit[item.col] + surplus[item.col]
        for item in [(row = _bi(nB, t, i), col = _bi(nB, t, i)) for t in 1:T for i in 1:nB])
    if nBat > 0
        ExaModels.constraint!(core, kcl_p,
            item.row => -p_dis[item.col] + p_ch[item.col]
            for item in [(row = _bi(nB, t, bat_pos[k]), col = _bti(nBat, t, k))
                         for t in 1:T for k in 1:nBat])
    end
    n_con += T * nB

    # ── 9. Reactive balance — HARD, no slack of any kind ─────────────────────
    kcl_q = ExaModels.constraint(core,
        p_qd[_bi(nB, item.t, item.i)] - item.bs * vm[_bi(nB, item.t, item.i)]^2
        for item in [(t = t, i = i, bs = float_type(net.buses[i].bs)) for t in 1:T for i in 1:nB])
    ExaModels.constraint!(core, kcl_q,
        item.row => -qg[item.col]
        for item in [(row = _bi(nB, t, g.bus_pos), col = _gi(nG, t, k))
                     for t in 1:T for (k, g) in enumerate(net.gens)])
    ExaModels.constraint!(core, kcl_q,
        item.row => q_fr[item.col]
        for item in [(row = _bi(nB, t, br.f_pos), col = _bri(nBR, t, l))
                     for t in 1:T for (l, br) in enumerate(net.branches)])
    ExaModels.constraint!(core, kcl_q,
        item.row => q_to[item.col]
        for item in [(row = _bi(nB, t, br.t_pos), col = _bri(nBR, t, l))
                     for t in 1:T for (l, br) in enumerate(net.branches)])
    n_con += T * nB

    # ── 10. Battery state transition — ADDED LAST ────────────────────────────
    # e_t − α e_{t-1} − η^{ch} Δt p^{ch}_t + (Δt/η^{dis}) p^{dis}_t = 0,
    # with e a PARAMETER. The duals of these rows are what
    # `target_multipliers` turns into ∂Q/∂x̂.
    transition_start = n_con + 1
    if nBat > 0
        ExaModels.constraint(core,
            p_energy[_ei(nBat, item.t, item.k)]
            - item.α * p_energy[_ei(nBat, item.t - 1, item.k)]
            - item.ηc * p_ch[_bti(nBat, item.t, item.k)]
            + item.ηd * p_dis[_bti(nBat, item.t, item.k)]
            for item in [(t = t, k = k,
                          α = float_type(b.self_discharge),
                          ηc = float_type(b.charge_efficiency) * Δt,
                          ηd = Δt / float_type(b.discharge_efficiency))
                         for t in 1:T for (k, b) in enumerate(batteries)])
        n_con += T * nBat
    end
    transition_range = transition_start:(transition_start + T * nBat - 1)

    model = ExaModels.ExaModel(core)
    prob = BatteryExaProblem(core, model, net, case, batteries,
                             p_pd, p_qd, p_energy,
                             zeros(Float64, (T + 1) * nBat),
                             transition_range, Float64(Δt), stage_of,
                             T, nB, nG, nBR, nBat)
    # Start from the initial energy so a solve before any explicit write is
    # still a well-posed problem rather than an all-zero trajectory.
    set_energy_path!(prob,
                     [b.energy_initial for b in batteries],
                     repeat([b.energy_initial for b in batteries], T))
    return prob
end

# ─────────────────────────────────────────────────────────────────────────────
# Parameter updates
# ─────────────────────────────────────────────────────────────────────────────

"""
    realized_demand(case, net, stages, atoms) -> (pd, qd)

Per-bus realized demand matrices for one scenario.

# Arguments
- `case::BatteryCase`, `net::ExaNetwork`.
- `stages::AbstractVector{<:Integer}`: the ABSOLUTE stage indices, which is what
  selects the deterministic shape entry.
- `atoms::AbstractVector{<:Integer}`: realized atom index per stage.

# Returns
- `(pd, qd)`: two `T×nBus` matrices in pu, indexed by bus POSITION.

# Notes
The realized demand is read from the FROZEN finite support through the shared
`realized_bus_demand`, which scales each LOAD by its own total multiplier and
then aggregates to the bus. Both matrices carry the same multiplier per load, so
every realization preserves each load's power factor exactly.

Doing it per load rather than per bus matters as soon as the support is anything
but system-wide: a regional or per-load multiplier applied to a bus AGGREGATE
would already have averaged away the structure the study is about, and the two
engines would then be solving two different problems while both reported "the
same demand".

Bus POSITION is this engine's own indexing; the shared function returns bus
IDENTIFIERS, and `net.bus_pos` is the only place the two are related.
"""
function realized_demand(case::BatteryCase, net::ExaNetwork,
                         stages::AbstractVector{<:Integer},
                         atoms::AbstractVector{<:Integer})
    length(stages) == length(atoms) ||
        throw(ArgumentError("stages and atoms must have the same length"))
    T = length(stages)
    pd = zeros(Float64, T, nbus(net))
    qd = zeros(Float64, T, nbus(net))
    for t in 1:T
        bus_pd, bus_qd = realized_bus_demand(case, stages[t], atoms[t])
        for (id, p) in bus_pd
            pos = get(net.bus_pos, id, 0)
            pos == 0 && continue          # a bus this engine dropped (bus_type 4)
            pd[t, pos] = p
            qd[t, pos] = bus_qd[id]
        end
    end
    return pd, qd
end

"""
    set_demand!(prob, pd, qd)

Write the realized per-bus demand into the model's parameters.

# Arguments
- `pd`, `qd`: `T×nBus` matrices in pu, indexed by bus position.
"""
function set_demand!(prob::BatteryExaProblem, pd::AbstractMatrix, qd::AbstractMatrix)
    size(pd) == (prob.horizon, prob.nBus) ||
        error("pd must be $(prob.horizon)×$(prob.nBus), got $(size(pd))")
    size(qd) == size(pd) || error("qd must have the same shape as pd")
    ExaModels.set_parameter!(prob.core, prob.p_pd,
                             [pd[t, i] for t in 1:prob.horizon for i in 1:prob.nBus])
    ExaModels.set_parameter!(prob.core, prob.p_qd,
                             [qd[t, i] for t in 1:prob.horizon for i in 1:prob.nBus])
    return prob
end

"""
    set_energy_path!(prob, x0, xhat)

Write the strict energy trajectory `[x0; xhat]` into the energy parameter.

# Arguments
- `x0::AbstractVector`: initial energy per battery, in battery-position order.
- `xhat::AbstractVector`: the `T·nBat` flat, stage-major target trajectory.

# Notes
This is the ONLY writer of the energy parameter, and it is what maintains the
strict-mode invariant stated at the top of this file: the first `nBat` entries
of the parameter always equal `x0`, because the model has no explicit
initial-condition row to enforce it.
"""
function set_energy_path!(prob::BatteryExaProblem, x0::AbstractVector, xhat::AbstractVector)
    prob.nBat == 0 && return prob
    length(x0) == prob.nBat || error("x0 must have length nBat=$(prob.nBat)")
    length(xhat) == prob.horizon * prob.nBat ||
        error("xhat must have length T*nBat=$(prob.horizon * prob.nBat)")
    vals = vcat(Float64.(vec(Array(x0))), Float64.(vec(Array(xhat))))
    copyto!(prob.energy_values, vals)
    ExaModels.set_parameter!(prob.core, prob.p_energy, vals)
    return prob
end

"""
    target_multipliers(prob, result) -> Vector{Float64}

Turn the battery-transition duals into the actor signal
``\\partial Q/\\partial \\hat e_{b,t}``.

# Notes
Let ``\\mu_{b,t}`` be the multiplier of the transition row

```math
c_{b,t} := \\hat e_{b,t} - \\alpha_b \\hat e_{b,t-1}
           - \\eta^{ch}_b \\Delta t\\, p^{ch}_{b,t}
           + \\tfrac{\\Delta t}{\\eta^{dis}_b} p^{dis}_{b,t} = 0 .
```

The target ``\\hat e_{b,t}`` appears in row ``t`` with coefficient ``+1`` and in
row ``t+1`` with coefficient ``-\\alpha_b``, and in no other row: the recourse
variables and the network appear in the balance, never in the transition. By the
envelope theorem,

```math
\\frac{\\partial Q}{\\partial \\hat e_{b,t}}
  = \\mu_{b,t} - \\alpha_b \\mu_{b,t+1},
\\qquad
\\frac{\\partial Q}{\\partial \\hat e_{b,T}} = \\mu_{b,T}.
```

The ``\\alpha_b`` factor is not decorative: with a self-discharging battery, the
value of energy left at the end of stage `t` reaches stage `t+1` attenuated, and
dropping it would misprice every interstage trade-off by that factor per stage.
"""
function target_multipliers(prob::BatteryExaProblem, result)
    prob.nBat == 0 && return Float64[]
    raw = Float64.(vec(Array(result.multipliers))[prob.transition_range])
    nBat, T = prob.nBat, prob.horizon
    out = copy(raw)
    if T > 1
        for t in 1:(T - 1), k in 1:nBat
            out[_bti(nBat, t, k)] -= prob.batteries[k].self_discharge * raw[_bti(nBat, t + 1, k)]
        end
    end
    return out
end

# ─────────────────────────────────────────────────────────────────────────────
# Solving and extraction
# ─────────────────────────────────────────────────────────────────────────────

"""
    DEFAULT_SOLVER_OPTIONS

MadNLP options every solve in this engine uses unless the caller overrides them.

# Notes
The tolerance sits well below every physical tolerance the study reports at. An
interior-point method parks a nonnegative variable roughly one tolerance below
its zero bound, and a positively priced variable sitting there lowers the
objective by a near-constant amount at every stage — an offset that looks
exactly like a systematic model difference when two engines are compared.
"""
const DEFAULT_SOLVER_OPTIONS = (print_level = MadNLP.ERROR, tol = 1e-10)

"""
    solve!(prob; solver_kwargs...) -> result

Solve the deterministic equivalent with a FRESH MadNLP solver.

# Notes
A fresh solver per solve, deliberately. MadNLP's re-solve path
(`reinitialize!`) is incompatible with this model as written: PGLib cases
contain synchronous condensers whose active-power box is exactly `[0, 0]`, and
MadNLP's default `fixed_variable_treatment = MakeParameter` removes those
variables from its internal primal vector, after which the re-solve path tries
to broadcast the full-length starting point into the reduced one and raises a
`DimensionMismatch`. The only re-solve configuration that works,
`fixed_variable_treatment = RelaxBound`, widens those boxes to about `1e-8` and
therefore makes this engine's feasible set larger than the PowerModels model it
is validated against — a change to the problem, traded for a constant factor of
speed. Correctness wins: on the correctness-phase case a fresh solve of a
24-stage horizon takes about 0.2 s.
"""
function solve!(prob::BatteryExaProblem; solver_kwargs...)
    # `Base.invokelatest` because the GPU linear solver arrives through a
    # package EXTENSION that is loaded at run time: a caller that switched to
    # the GPU inside a function body is executing in a world older than the
    # extension's methods, and MadNLP's option check then reports
    # "no method matching input_type(::CUDSSSolver) … the applicable method may
    # be too new". The dynamic dispatch is free next to an NLP solve.
    return Base.invokelatest(MadNLP.madnlp, prob.model;
                             DEFAULT_SOLVER_OPTIONS..., solver_kwargs...)
end

"""
    battery_solution(prob, result) -> NamedTuple

Unpack the flat solution vector into named, positionally indexed components.

# Notes
The unpacking order MUST match the declaration order in
[`build_battery_exa`](@ref); it is written here as a single sequential walk over
the vector precisely so that the two orders can be read side by side.

The energy trajectory is not part of the solution vector — it is a parameter —
so it is reported from the last value written, which is the strict-mode
invariant's other half.
"""
function battery_solution(prob::BatteryExaProblem, result)
    T, nB, nG, nBR, nBat = prob.horizon, prob.nBus, prob.nGen, prob.nBranch, prob.nBat
    sol = Float64.(vec(Array(result.solution)))
    off = 0
    take(n) = (v = sol[off .+ (1:n)]; off += n; v)

    va = reshape(take(T * nB), nB, T)
    vm = reshape(take(T * nB), nB, T)
    pg = reshape(take(T * nG), nG, T)
    qg = reshape(take(T * nG), nG, T)
    p_fr = reshape(take(T * nBR), nBR, T)
    q_fr = reshape(take(T * nBR), nBR, T)
    p_to = reshape(take(T * nBR), nBR, T)
    q_to = reshape(take(T * nBR), nBR, T)
    deficit = reshape(take(T * nB), nB, T)
    surplus = reshape(take(T * nB), nB, T)
    p_ch = nBat == 0 ? zeros(0, T) : reshape(take(T * nBat), nBat, T)
    p_dis = nBat == 0 ? zeros(0, T) : reshape(take(T * nBat), nBat, T)
    energy = nBat == 0 ? zeros(0, T + 1) : reshape(copy(prob.energy_values), nBat, T + 1)

    return (va = va, vm = vm, pg = pg, qg = qg,
            p_fr = p_fr, q_fr = q_fr, p_to = p_to, q_to = q_to,
            deficit = deficit, surplus = surplus,
            p_ch = p_ch, p_dis = p_dis, p_bat = p_dis .- p_ch,
            energy = energy)
end

"""
    stage_costs(prob, sol) -> NamedTuple

Decompose the objective into its physical components, per stage.

# Returns
`(generation, throughput, deficit, surplus, total)`, each a length-`T` vector.

# Notes
Recomputed from the extracted physical values rather than read off the solver,
so that "the sum of the parts equals the objective" is a real check on both the
extraction order and the objective assembly.
"""
function stage_costs(prob::BatteryExaProblem, sol)
    T = prob.horizon
    gen = zeros(T); thr = zeros(T); def = zeros(T); sur = zeros(T)
    for t in 1:T
        for (k, g) in enumerate(prob.net.gens)
            p = sol.pg[k, t]
            gen[t] += g.c2 * p^2 + g.c1 * p + g.c0
        end
        for (k, b) in enumerate(prob.batteries)
            thr[t] += b.throughput_cost * prob.Δt * (sol.p_ch[k, t] + sol.p_dis[k, t])
        end
        def[t] = prob.case.recourse.deficit * sum(view(sol.deficit, :, t))
        sur[t] = prob.case.recourse.surplus * sum(view(sol.surplus, :, t))
    end
    return (generation = gen, throughput = thr, deficit = def, surplus = sur,
            total = gen .+ thr .+ def .+ sur)
end
