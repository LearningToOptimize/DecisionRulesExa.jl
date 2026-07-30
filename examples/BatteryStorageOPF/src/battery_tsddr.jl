# battery_tsddr.jl
#
# Phase-2 OPERATIONAL / stochastic battery AC-OPF used by TS-DDR training and
# evaluation. Canonical spec: docs/src/casestudies/battery_storage_opf.md.
#
# It EXTENDS the accepted Phase-1 model through the SHARED ACP blocks in
# acp_core.jl (single source of truth) and adds exactly two things:
#
#   1. TWO-SIDED ABSOLUTE ACTIVE RECOURSE — a nonnegative, UNBOUNDED-above pair
#      at EVERY bus,
#        d⁺[t,i] ≥ 0  (active deficit / injection),
#        d⁻[t,i] ≥ 0  (active surplus / absorption),
#      entering ONLY the active balance as `− d⁺ + d⁻`. It is an artificial
#      active-balance recourse, NOT curtailed customer load: d⁺ may exceed local
#      demand and may be positive where p^d = 0. Reactive KCL stays a HARD
#      equality: there is NO reactive slack. The pair is priced at VOLL and is
#      INCLUDED in physical operating cost. It is an operational safety valve —
#      an accepted run leaves BOTH directions at zero within tolerance.
#
#   2. TARGET-STATE PROJECTION on the battery SoC, in one of two modes:
#        strict : ê_{t+1,b} − e_{t+1,b} = 0                (no slack, no penalty)
#        soft   : ê_{t+1,b} − e_{t+1,b} − δ⁺ + δ⁻ = 0,  δ± ≥ 0
#      with a documented TRAINING-ONLY penalty ρ1·Σ(δ⁺+δ⁻) + (ρ2/2)·Σ((δ⁺)²+(δ⁻)²)
#      that NEVER enters physical operating cost.
#
# Active recourse and target slack are different mechanisms with different names
# and different roles. Target constraints are added LAST so their multipliers
# occupy a contiguous slice (`target_con_range`).
#
# Demand realization follows the canonical pattern: the uncertainty parameter
# `p_w` carries the observed atom features (it appears in no constraint), and
# `set_realized_demand!` / `prepare_solve!` write the realized per-bus demand
# p^d = p^{d,0}·h_t·L·R into the SAME `p_pd`/`p_qd` parameters the shared ACP
# balance uses — so the stochastic and deterministic builders solve identical
# equations for identical realized demand.
#
# Generator prices, pmin/pmax, qmin/qmax, branch limits, voltage limits,
# admittances, and topology are the original PGLib values. Nothing here scales
# generator capacity.

using ExaModels
using MadNLP
using NLPModels
import DecisionRulesExa: target_multipliers, prepare_solve!

# w-vector index helpers (stage-major, nw = 1 + nregion): entry 1 of a stage
# block is the realized system multiplier s_t; entry 1+r is region r's multiplier.
@inline _widx_s(nw, t)    = (t - 1) * nw + 1
@inline _widx_r(nw, t, r) = (t - 1) * nw + 1 + r

"""
    DEFAULT_ACTIVE_RECOURSE_COST_PER_MWH

Default price of the two-sided active nodal recourse, `10_000.0` USD/MWh
(canonical spec — the value of lost load). The stage cost is
`c · S^base · Δt · Σ_i (d⁺_{t,i} + d⁻_{t,i})`.
"""
const DEFAULT_ACTIVE_RECOURSE_COST_PER_MWH = 10_000.0

"""
    ACTIVE_RECOURSE_LB_TOL_PU

Declared **per-variable** lower-bound tolerance (pu) for the nominally nonnegative
primal quantities — the two-sided active recourse `d⁺`/`d⁻` and the soft target
slacks `δ⁺`/`δ⁻`. An interior-point solve may return an individual variable a hair
below its zero bound (`~1e-8` pu observed; the loosest solver bound tolerance used
anywhere here is `1e-6`). Reporting VALIDATES that no single value is further below
zero than this tolerance (failing loudly otherwise — that would signal a real
defect, not noise), then PROJECTS the tolerated noise elementwise with `max(·, 0)`
before computing any public physical quantity. This value is `1e-5` pu: 10× the
loosest solver bound tolerance and ≥100× below the smallest material recourse, so
it never trips on noise yet catches a genuinely negative value.
"""
const ACTIVE_RECOURSE_LB_TOL_PU = 1e-5

"""
    BatteryTSDDRProblem

Operational target-constrained ExaModels problem for a [`BatteryCase`] and its
[`LoadProcess`]. Usable by `train_tsddr`, `simulate_tsddr`, and (with
`horizon = 1`) `rollout_tsddr`.

# Key fields
- `p_x0`: initial battery SoC parameter (length `nBat`).
- `p_w`: observed per-stage uncertainty features (length `horizon·nw`); read by
  `prepare_solve!`/`set_realized_demand!`, referenced by no constraint.
- `p_pd`, `p_qd`: realized per-bus active/reactive demand actually used by the
  shared ACP balance (length `horizon·nBus` each).
- `p_target`: target next-SoC parameter (length `horizon·nBat`).
- `mode`: `:strict` or `:soft`.
- `rho1`, `rho2`: soft target-penalty L1/L2 coefficients (training-only).
- `active_recourse_cost_per_mwh`, `baseMVA`: recourse price and power base.
- `reporting_horizon`, `lookahead`, `dt`: terminal treatment and stage duration.
- `target_con_range`: contiguous multiplier slice of the target constraints.
- `realized_pd`, `realized_qd`: `[T × nBus]` buffers holding the demand most
  recently written into `p_pd`/`p_qd` (used by the cost decomposition).
"""
struct BatteryTSDDRProblem
    core
    model
    p_x0
    p_w
    p_pd
    p_qd
    p_target
    nBus::Int
    nGen::Int
    nBranch::Int
    nBat::Int
    nw::Int
    horizon::Int
    reporting_horizon::Int
    lookahead::Int
    dt::Float64
    mode::Symbol
    rho1::Float64
    rho2::Float64
    active_recourse_cost_per_mwh::Float64
    baseMVA::Float64
    gen_c2::Vector{Float64}
    gen_c1::Vector{Float64}
    gen_c0::Vector{Float64}
    cycle_coeffs::Vector{Float64}
    target_con_range::UnitRange{Int}
    realized_pd::Matrix{Float64}
    realized_qd::Matrix{Float64}
    case::BatteryCase
    process::LoadProcess
    float_type::Type
end

nbattery(prob::BatteryTSDDRProblem) = prob.nBat

"""
    build_battery_tsddr_de(case, process; reporting_horizon, lookahead=0,
        mode=:soft, rho1=0.0, rho2=:auto,
        active_recourse_cost_per_mwh=DEFAULT_ACTIVE_RECOURSE_COST_PER_MWH,
        stage_hours=1.0, backend=nothing, float_type=Float64)
        -> BatteryTSDDRProblem

Build the `T = reporting_horizon + lookahead` stage operational problem.

# Keywords
- `reporting_horizon::Int`: stages whose physical cost is the headline metric.
- `lookahead::Int = 0`: look-ahead buffer appended after the reporting window.
- `mode::Symbol = :strict`: `:strict` (hard `ê=e`, no slack/penalty — the
  PRIMARY / default operational and training mode; the two-sided active nodal
  recourse gives it complete recourse so it always solves) or `:soft` (split
  slacks + training-only penalty — a DIAGNOSTIC/fallback only).
- `rho1`, `rho2`: soft target-penalty L1 and L2 coefficients. `rho2 = :auto`
  uses `2·max(c1,c2)` over generators (the project-standard auto scale).
- `active_recourse_cost_per_mwh`: recourse price in USD/MWh (default 10 000).
- `stage_hours::Real = 1.0`: Δt in hours.
- `backend`: `nothing` (CPU) or a CUDA backend (GPU) — same equations either way.

Generator and network data are never modified.
"""
function _build_battery_problem(case::BatteryCase, process::LoadProcess;
                                reporting_horizon::Int,
                                lookahead::Int = 0,
                                mode::Symbol = :soft,
                                rho1::Real = 0.0,
                                rho2::Union{Real,Symbol} = :auto,
                                active_recourse_cost_per_mwh::Real = DEFAULT_ACTIVE_RECOURSE_COST_PER_MWH,
                                allow_active_recourse::Bool = true,
                                stage_hours::Real = 1.0,
                                backend = nothing,
                                float_type::Type{<:AbstractFloat} = Float64)
    reporting_horizon >= 1 || error("reporting_horizon must be ≥ 1; got $reporting_horizon")
    lookahead >= 0 || error("lookahead must be ≥ 0; got $lookahead")
    mode in (:strict, :soft, :none) ||
        error("mode must be :strict, :soft, or :none (internal); got :$mode")
    T = reporting_horizon + lookahead
    dt = Float64(stage_hours)
    validate_stage_hours(case, dt)
    (isfinite(active_recourse_cost_per_mwh) && active_recourse_cost_per_mwh >= 0) ||
        error("active_recourse_cost_per_mwh must be finite and ≥ 0; got $active_recourse_cost_per_mwh")
    nd = case.network
    length(process.region_of_bus) == nbus(nd) ||
        error("process.region_of_bus length $(length(process.region_of_bus)) ≠ nbus=$(nbus(nd)); " *
              "the LoadProcess must be built for THIS case's network")

    nBus = nbus(nd); nGen = ngen(nd); nBranch = nbranch(nd)
    nBat = length(case.batteries)
    nw = n_uncertainty(process)
    ρ1 = Float64(rho1)
    ρ2 = Float64(rho2 === :auto ?
        2.0 * maximum(max(g.cost1, g.cost2) for g in nd.gens; init = 0.0) : rho2)
    (isfinite(ρ1) && ρ1 >= 0) || error("rho1 must be finite and ≥ 0; got $ρ1")
    (isfinite(ρ2) && ρ2 >= 0) || error("rho2 must be finite and ≥ 0; got $ρ2")

    core = ExaModels.ExaCore(float_type; backend = backend, concrete = Val(true))

    # ── Variables: shared ACP + battery, then active recourse, then soft slacks ─
    core, v   = add_acp_variables!(core, nd, T, float_type)
    core, bat = add_battery_variables!(core, case, T, float_type)

    # Two-sided active nodal recourse, both nonnegative and UNBOUNDED above at
    # EVERY bus (active-only). They enter the active balance as `− d⁺ + d⁻`:
    #   active_deficit  = d⁺  (injection, covers a local active shortfall — e.g.
    #                          the power to CHARGE a battery at a congested bus)
    #   active_surplus  = d⁻  (absorption, absorbs a local active excess — e.g.
    #                          the power a battery is forced to DISCHARGE into a
    #                          bus whose outgoing branches are saturated)
    # They are an artificial active-balance recourse, NOT curtailed customer
    # load: d⁺ may exceed local demand and may be positive where p^d = 0. Together
    # they give relatively complete recourse, so the stage subproblem is feasible
    # for every incoming state and every reachable target. Both are priced at
    # VOLL, so an accepted solution leaves both at ~0. `allow_active_recourse =
    # false` fixes BOTH blocks to zero (uvar = 0) — used by the Phase-1 parity
    # test to recover the hard-balance deterministic model exactly.
    rec_ub = allow_active_recourse ? fill(float_type(Inf), T * nBus) : zeros(float_type, T * nBus)
    core, active_deficit = ExaModels.add_var(core, T * nBus; lvar = float_type(0), uvar = rec_ub)
    core, active_surplus = ExaModels.add_var(core, T * nBus; lvar = float_type(0), uvar = rec_ub)

    slack_pos = nothing; slack_neg = nothing
    if mode === :soft && nBat > 0
        core, slack_pos = ExaModels.add_var(core, T * nBat; lvar = float_type(0))
        core, slack_neg = ExaModels.add_var(core, T * nBat; lvar = float_type(0))
    end

    # ── Parameters ────────────────────────────────────────────────────────────
    # Realized demand actually consumed by the shared ACP balance; initialized to
    # the deterministic base shape at unit factors and overwritten per solve.
    init_pd = Float64[]; init_qd = Float64[]
    for t in 1:T
        h = process.base_shape[((t - 1) % process.period) + 1]
        append!(init_pd, h .* nd.bus_pd)
        append!(init_qd, h .* nd.bus_qd)
    end
    core, p_pd = ExaModels.add_par(core, float_type.(init_pd))
    core, p_qd = ExaModels.add_par(core, float_type.(init_qd))
    # Observed uncertainty features (system + regional multipliers). Referenced
    # by NO constraint; consumed by set_realized_demand!/prepare_solve!.
    w0 = Float64[]
    for t in 1:T
        h = process.base_shape[((t - 1) % process.period) + 1]
        push!(w0, h)
        append!(w0, ones(Float64, process.nregion))
    end
    core, p_w = ExaModels.add_par(core, float_type.(w0))
    core, p_x0 = ExaModels.add_par(core, float_type.([b.e_init for b in case.batteries]))
    # Target parameter exists ONLY for the target modes. The targetless
    # diagnostic (`mode === :none`) carries no target data at all.
    p_target = nothing
    if mode !== :none
        core, p_target = ExaModels.add_par(core,
            float_type.([b.e_init for _ in 1:T for b in case.batteries]))
    end

    # ── Objective ─────────────────────────────────────────────────────────────
    core = add_generator_cost!(core, v, nd, T, dt, float_type)
    core = add_cycle_cost!(core, bat, case, nd, T, dt, float_type)

    # Two-sided active-recourse cost: c · S^base · Δt · Σ (d⁺ + d⁻)  (physical
    # operating cost). Both are recourse POWER in pu, so the cost is linear in
    # them directly (no `p^d` factor); pricing both keeps an accepted solution at
    # ~0.
    rec_coeff = float_type(active_recourse_cost_per_mwh * nd.baseMVA * dt)
    if rec_coeff > 0
        rec_items = [(idx = _bidx(nBus, t, b),) for t in 1:T for b in 1:nBus]
        core, _ = ExaModels.add_obj(core,
            rec_coeff * (active_deficit[it.idx] + active_surplus[it.idx]) for it in rec_items)
    end

    # Soft TRAINING-ONLY target penalty ρ1·Σ(δ⁺+δ⁻) + (ρ2/2)·Σ((δ⁺)²+(δ⁻)²).
    if mode === :soft && nBat > 0
        pen_items = [(idx = (t - 1) * nBat + k,) for t in 1:T for k in 1:nBat]
        if ρ1 > 0
            core, _ = ExaModels.add_obj(core,
                float_type(ρ1) * (slack_pos[it.idx] + slack_neg[it.idx]) for it in pen_items)
        end
        if ρ2 > 0
            core, _ = ExaModels.add_obj(core,
                float_type(ρ2 / 2) * (slack_pos[it.idx]^2 + slack_neg[it.idx]^2)
                for it in pen_items)
        end
    end

    # ── Constraints (SHARED blocks; targets appended LAST) ────────────────────
    core = add_acp_network_constraints!(core, v, nd, T, float_type)
    core = add_nodal_balance!(core, v, bat, nd, case, T, float_type, p_pd, p_qd;
                              active_deficit = active_deficit, active_surplus = active_surplus)
    core = add_battery_dynamics!(core, bat, case, T, dt, float_type, p_x0)

    n_before = acp_constraint_count(nd, case, T)
    if mode !== :none && nBat > 0
        tgt_items = [(pt = (t - 1) * nBat + k, en = _eidx(nBat, t + 1, k),
                      sl = (t - 1) * nBat + k) for t in 1:T for k in 1:nBat]
        if mode === :strict
            core, _ = ExaModels.add_con(core, p_target[it.pt] - bat.e[it.en] for it in tgt_items)
        else
            core, _ = ExaModels.add_con(core,
                p_target[it.pt] - bat.e[it.en] - slack_pos[it.sl] + slack_neg[it.sl]
                for it in tgt_items)
        end
    end
    # Targetless diagnostic: EMPTY multiplier slice (no target constraints exist).
    target_range = mode === :none ? ((n_before + 1):n_before) :
                                    ((n_before + 1):(n_before + T * nBat))

    gen_c2 = Float64[g.cost2 * dt for g in nd.gens]
    gen_c1 = Float64[g.cost1 * dt for g in nd.gens]
    gen_c0 = Float64[g.cost0 * dt for g in nd.gens]
    cycle_coeffs = Float64[b.cycle_cost_per_mwh * nd.baseMVA * dt for b in case.batteries]

    model = ExaModels.ExaModel(core)
    prob = BatteryTSDDRProblem(core, model, p_x0, p_w, p_pd, p_qd, p_target,
                               nBus, nGen, nBranch, nBat, nw, T,
                               reporting_horizon, lookahead, dt, mode, ρ1, ρ2,
                               Float64(active_recourse_cost_per_mwh), Float64(nd.baseMVA),
                               gen_c2, gen_c1, gen_c0, cycle_coeffs, target_range,
                               permutedims(reshape(init_pd, nBus, T)),
                               permutedims(reshape(init_qd, nBus, T)),
                               case, process, float_type)
    return prob
end

"""
    build_battery_stage_problem(case, process; mode=:strict, kwargs...) -> BatteryTSDDRProblem

Single-stage (`reporting_horizon = 1`, `lookahead = 0`) operational problem for
stage-wise `rollout_tsddr` evaluation. The realized next SoC is `e[:, 2]`.
Defaults to the primary `:strict` mode.
"""
build_battery_stage_problem(case::BatteryCase, process::LoadProcess; mode::Symbol = :strict, kwargs...) =
    build_battery_tsddr_de(case, process; reporting_horizon = 1, lookahead = 0, mode = mode, kwargs...)

"""
    build_battery_tsddr_de(case, process; mode = :strict, kwargs...) -> BatteryTSDDRProblem

PRODUCTION builder. `mode` is `:strict` (default, primary) or `:soft`
(diagnostic/fallback) ONLY; the targetless diagnostic is not a production target
mode and is built by [`build_targetless_diagnostic_de`](@ref).
"""
function build_battery_tsddr_de(case::BatteryCase, process::LoadProcess;
                                mode::Symbol = :strict, kwargs...)
    mode in (:strict, :soft) ||
        error("mode must be :strict or :soft; got :$mode. The targetless diagnostic " *
              "is built with build_targetless_diagnostic_de and is not a production mode.")
    return _build_battery_problem(case, process; mode = mode, kwargs...)
end

"""
    is_targetless(prob) -> Bool

`true` for a targetless PHYSICAL DIAGNOSTIC problem, which carries no target
constraints, no target slack variables, and no target penalty.
"""
is_targetless(prob::BatteryTSDDRProblem) = prob.mode === :none

"""
    build_targetless_diagnostic_de(case, process; reporting_horizon, lookahead=0,
        allow_active_recourse=true, kwargs...) -> BatteryTSDDRProblem

Build the GENUINELY TARGETLESS physical diagnostic.

The model contains **no target constraints, no target slack variables, and no
target penalty** — not a zero-weight target, but no target data at all
(`p_target === nothing`, `target_con_range` empty). Batteries are freely
dispatchable subject only to their own dynamics and bounds. Everything else —
the shared ACP equations, the realized demand, the two-sided active recourse,
generator and throughput costs, and the battery equations — is identical to the
operational model, and is built from the SAME shared blocks (nothing is copied).

Its objective is exactly `generator + battery throughput + active-recourse cost`.

`allow_active_recourse = false` fixes both recourse blocks (d⁺ and d⁻) to zero by
bounds, giving the recourse-forbidden companion that shares every physical
equation and datum.

Interpretation rules (see the README): an ACCEPTED recourse-forbidden solve
proves a zero-recourse feasible point was FOUND; a FAILED recourse-forbidden
solve is INCONCLUSIVE and never proves infeasibility. Results are local optima of
a nonconvex ACP unless a global certificate exists.

This is a diagnostic only: `train_tsddr`-style training, target-multiplier
extraction, and target rollout all REJECT it.
"""
build_targetless_diagnostic_de(case::BatteryCase, process::LoadProcess; kwargs...) =
    _build_battery_problem(case, process; mode = :none, kwargs...)

# ── Deterministic, target-consistent primal starts (strict-mode start repair) ──
#
# Only the primal STARTING POINT is affected. No tolerance, iteration budget,
# equation, bound, generator, or network value is touched, and among accepted
# starts the FIRST in the fixed order is taken (never the cheapest).

"""
    variable_offsets(prob) -> NamedTuple

Zero-based offsets of each variable block in the flat solution/start vector, in
creation order: `va, vm, pg, qg, p_fr, q_fr, p_to, q_to, p_ch, p_dis, e,
active_deficit, active_surplus, [slack_pos, slack_neg]`.
"""
function variable_offsets(prob::BatteryTSDDRProblem)
    T = prob.horizon; nB = prob.nBus; nG = prob.nGen; nBR = prob.nBranch; nK = prob.nBat
    o = 0
    va = o;      o += T*nB
    vm = o;      o += T*nB
    pg = o;      o += T*nG
    qg = o;      o += T*nG
    p_fr = o;    o += T*nBR
    q_fr = o;    o += T*nBR
    p_to = o;    o += T*nBR
    q_to = o;    o += T*nBR
    p_ch = o;    o += T*nK
    p_dis = o;   o += T*nK
    e = o;       o += (T+1)*nK
    active_deficit = o; o += T*nB
    active_surplus = o; o += T*nB
    slack_pos = o
    slack_neg = prob.mode === :soft ? o + T*nK : o
    return (va = va, vm = vm, pg = pg, qg = qg, p_fr = p_fr, q_fr = q_fr,
            p_to = p_to, q_to = q_to, p_ch = p_ch, p_dis = p_dis, e = e,
            active_deficit = active_deficit, active_surplus = active_surplus,
            slack_pos = slack_pos, slack_neg = slack_neg)
end

"""
    target_consistent_start!(prob, e_prev, target) -> x0

Write a deterministic, TARGET-CONSISTENT battery start into the model's primal
starting point and return it. For each battery, with

```
Δe = target − (1 − σ·Δt)·e_prev,
```

the start is the exact charge/discharge that realizes `Δe`:

```
Δe ≥ 0 :  p_charge = Δe/(η_ch·Δt),        p_discharge = 0
Δe < 0 :  p_charge = 0,                    p_discharge = −Δe·η_dis/Δt
```

clipped to the battery power bounds (the reachability interval guarantees the
clip is inactive for a reachable target). The SoC trajectory is started at the
implied path (`e_1 = e_prev`, `e_2 = target`). Only the START is set.
"""
function target_consistent_start!(prob::BatteryTSDDRProblem, e_prev::AbstractVector,
                                  target::AbstractVector)
    x0 = NLPModels.get_x0(prob.model)
    off = variable_offsets(prob)
    T = prob.horizon; nK = prob.nBat
    nK == 0 && return x0
    length(e_prev) == nK || error("e_prev length must be nBat=$nK")
    length(target) == T * nK || error("target length must be horizon·nBat=$(T*nK)")
    prev = Float64.(collect(e_prev))
    for t in 1:T
        for (k, b) in enumerate(prob.case.batteries)
            tgt = Float64(target[(t-1)*nK + k])
            Δe = tgt - (1 - b.sigma * prob.dt) * prev[k]
            pch, pdis = if Δe >= 0
                (min(Δe / (b.eta_ch * prob.dt), b.p_charge_max), 0.0)
            else
                (0.0, min(-Δe * b.eta_dis / prob.dt, b.p_discharge_max))
            end
            x0[off.p_ch  + (t-1)*nK + k] = pch
            x0[off.p_dis + (t-1)*nK + k] = pdis
            x0[off.e     + t*nK + k]     = tgt      # e index (t+1) is block t (0-based)
            prev[k] = tgt
        end
    end
    for k in 1:nK                                    # e_1 = e_prev
        x0[off.e + k] = Float64(e_prev[k])
    end
    return x0
end

"""
    seed_start_from_solution!(prob, x_src) -> x0

Copy a previously ACCEPTED primal point into this problem's starting point.
Blocks are copied by the shared variable order for the overlapping length, so a
soft/targetless-diagnostic solution can seed a strict solve of the same size.
"""
function seed_start_from_solution!(prob::BatteryTSDDRProblem, x_src::AbstractVector)
    x0 = NLPModels.get_x0(prob.model)
    n = min(length(x0), length(x_src))
    @inbounds for i in 1:n
        x0[i] = Float64(x_src[i])
    end
    return x0
end

"""
    reset_flat_start!(prob) -> x0

Restore the model's default flat start (`vm = 1`, everything else 0, SoC at
`e_init`) — the first entry of the deterministic start sequence.
"""
function reset_flat_start!(prob::BatteryTSDDRProblem)
    x0 = NLPModels.get_x0(prob.model)
    off = variable_offsets(prob)
    T = prob.horizon; nB = prob.nBus; nK = prob.nBat
    fill!(x0, 0.0)
    @inbounds for i in 1:(T*nB)
        x0[off.vm + i] = 1.0
    end
    @inbounds for t in 0:T, (k, b) in enumerate(prob.case.batteries)
        x0[off.e + t*nK + k] = b.e_init
    end
    return x0
end

# ── Demand realization + parameter setters ────────────────────────────────────

"""
    set_realized_demand!(prob, w_flat) -> prob

Compute the realized per-bus demand from the observed uncertainty features and
write it into the `p_pd`/`p_qd` parameters used by the shared ACP balance:

`p^d_{t,i} = p^{d,0}_i · s_t · R_{r(i),t}` and `q^d_{t,i} = q^{d,0}_i · s_t · R_{r(i),t}`,

where `s_t` already contains the deterministic daily shape times the atom's
system factor. The SAME multiplier scales active and reactive demand, preserving
every bus's base power factor. Also refreshes the `realized_pd`/`realized_qd`
buffers used by the cost decomposition.
"""
function set_realized_demand!(prob::BatteryTSDDRProblem, w_flat::AbstractVector)
    T = prob.horizon; nBus = prob.nBus; nw = prob.nw
    w = Float64.(vec(Array(w_flat)))
    length(w) == T * nw ||
        error("w_flat length must be horizon·nw=$(T*nw); got $(length(w))")
    nd = prob.case.network
    region = prob.process.region_of_bus
    @inbounds for t in 1:T
        s_t = w[_widx_s(nw, t)]
        for b in 1:nBus
            f = s_t * w[_widx_r(nw, t, region[b])]
            prob.realized_pd[t, b] = nd.bus_pd[b] * f
            prob.realized_qd[t, b] = nd.bus_qd[b] * f
        end
    end
    ExaModels.set_parameter!(prob.core, prob.p_pd,
        prob.float_type.([prob.realized_pd[t, b] for t in 1:T for b in 1:nBus]))
    ExaModels.set_parameter!(prob.core, prob.p_qd,
        prob.float_type.([prob.realized_qd[t, b] for t in 1:T for b in 1:nBus]))
    return prob
end

"""
    set_tsddr_uncertainty!(prob, w_flat) -> prob

Set the observed uncertainty parameter AND the realized demand it implies.
"""
function set_tsddr_uncertainty!(prob::BatteryTSDDRProblem, w_flat::AbstractVector)
    length(w_flat) == prob.horizon * prob.nw ||
        error("w_flat length must be horizon·nw=$(prob.horizon*prob.nw); got $(length(w_flat))")
    ExaModels.set_parameter!(prob.core, prob.p_w, prob.float_type.(collect(w_flat)))
    set_realized_demand!(prob, w_flat)
    return prob
end

"""
    set_tsddr_initial_soc!(prob, e0) -> prob

Set the initial battery-SoC parameter (length `nBat`, pu·h).
"""
function set_tsddr_initial_soc!(prob::BatteryTSDDRProblem, e0::AbstractVector)
    length(e0) == prob.nBat || error("e0 length must be nBat=$(prob.nBat); got $(length(e0))")
    ExaModels.set_parameter!(prob.core, prob.p_x0, prob.float_type.(collect(e0)))
    return prob
end

"""
    set_tsddr_targets!(prob, xhat) -> prob

Set the target next-SoC parameter (length `horizon·nBat`, stage-major).
"""
function set_tsddr_targets!(prob::BatteryTSDDRProblem, xhat::AbstractVector)
    is_targetless(prob) &&
        error("this is a TARGETLESS diagnostic problem: it has no target parameter, " *
              "no target constraints, and no target slacks; targets cannot be set")
    length(xhat) == prob.horizon * prob.nBat ||
        error("xhat length must be horizon·nBat=$(prob.horizon*prob.nBat); got $(length(xhat))")
    ExaModels.set_parameter!(prob.core, prob.p_target, prob.float_type.(collect(xhat)))
    return prob
end

"""
    prepare_solve!(prob::BatteryTSDDRProblem, init_state, w_flat, xhat_flat) -> Nothing

Pre-solve hook used by the DecisionRulesExa training loop: after the loop writes
`p_x0`, `p_w`, and `p_target`, this realizes the demand implied by `w_flat` into
`p_pd`/`p_qd` (the canonical pattern — the uncertainty parameter itself appears
in no constraint).
"""
function prepare_solve!(prob::BatteryTSDDRProblem, init_state, w_flat, xhat_flat)
    is_targetless(prob) &&
        error("a TARGETLESS diagnostic problem cannot be used for TS-DDR training/rollout")
    set_realized_demand!(prob, w_flat)
    return nothing
end

# ── Envelope multipliers ──────────────────────────────────────────────────────

"""
    target_multipliers(prob::BatteryTSDDRProblem, result) -> λ

Multipliers of the target constraints, `result.multipliers[target_con_range]`.
With the orientation `ê_{t+1} − e_{t+1} (− δ⁺ + δ⁻) = 0`, these are the envelope
derivatives `∂Q/∂ê` used as the policy-gradient signal, up to the solver's
documented dual convention — the sign AND magnitude are finite-difference tested
rather than assumed.
"""
function target_multipliers(prob::BatteryTSDDRProblem, result)
    is_targetless(prob) &&
        error("target_multipliers is undefined for a TARGETLESS diagnostic problem " *
              "(it has no target constraints)")
    return result.multipliers[prob.target_con_range]
end

# ── Solution extraction + cost decomposition ──────────────────────────────────

"""
    tsddr_solution(prob, result) -> NamedTuple

Reshape the flat solution into named matrices (columns are stages): `va`,`vm`
(nBus), `pg`,`qg` (nGen), `p_fr`,`q_fr`,`p_to`,`q_to` (nBranch), `p_ch`,`p_dis`
(nBat), `soc` (nBat×(T+1)), `active_deficit_pu`/`active_surplus_pu` (nBus×T — the
per-bus two-sided active-recourse power in pu), the soft-mode
`target_slack_pos`/`target_slack_neg` (nBat×T; zeros in strict mode), and derived
`p_bat = p_dis − p_ch`.
"""
function tsddr_solution(prob::BatteryTSDDRProblem, result)
    T = prob.horizon; nB = prob.nBus; nG = prob.nGen; nBR = prob.nBranch; nK = prob.nBat
    sol = Array(result.solution)
    off = 0
    take(n, m) = (v = reshape(sol[off .+ (1:n*m)], n, m); off += n*m; v)
    va   = take(nB, T)
    vm   = take(nB, T)
    pg   = take(nG, T)
    qg   = take(nG, T)
    p_fr = take(nBR, T)
    q_fr = take(nBR, T)
    p_to = take(nBR, T)
    q_to = take(nBR, T)
    if nK > 0
        p_ch  = take(nK, T)
        p_dis = take(nK, T)
        soc   = take(nK, T + 1)
    else
        p_ch  = zeros(eltype(sol), 0, T)
        p_dis = zeros(eltype(sol), 0, T)
        soc   = zeros(eltype(sol), 0, T + 1)
    end
    active_deficit_pu = take(nB, T)
    active_surplus_pu = take(nB, T)
    if prob.mode === :soft && nK > 0
        slack_pos = take(nK, T)
        slack_neg = take(nK, T)
    else
        slack_pos = zeros(eltype(sol), nK, T)
        slack_neg = zeros(eltype(sol), nK, T)
    end
    return (va = va, vm = vm, pg = pg, qg = qg,
            p_fr = p_fr, q_fr = q_fr, p_to = p_to, q_to = q_to,
            p_ch = p_ch, p_dis = p_dis, soc = soc,
            active_deficit_pu = active_deficit_pu, active_surplus_pu = active_surplus_pu,
            target_slack_pos = slack_pos, target_slack_neg = slack_neg,
            p_bat = p_dis .- p_ch)
end

"""
    decompose_costs(prob, result; sol=tsddr_solution(prob, result),
                    lb_tol=ACTIVE_RECOURSE_LB_TOL_PU) -> NamedTuple

Recompute every reported cost independently from the primal solution
(canonical spec, "Stage objective and cost accounting").

**Raw solver accounting vs physical reporting are kept separate.** The nominally
nonnegative primal quantities — the two-sided active recourse `d⁺`/`d⁻` and the
soft target slacks `δ⁺`/`δ⁻` — have a hard lower bound of 0, but an interior-point
solve can return an individual value a hair below it. This function therefore:

1. keeps the RAW solver values for objective reproduction (`raw_*`);
2. VALIDATES that no single value is below `−lb_tol` (per variable), failing loudly
   otherwise (a genuinely negative value is a defect, not noise);
3. PROJECTS the tolerated bound noise **elementwise** with `max(·, 0)` — never on
   the aggregate, since negative and positive bus values could otherwise cancel;
4. computes every PUBLIC physical quantity from the projected values.

Public physical fields are consequently always finite and `≥ 0`:
`active_deficit_pu`, `active_surplus_pu`, `active_deficit_energy_mwh`,
`active_surplus_energy_mwh`, `total_active_recourse_energy_mwh`,
`max_active_deficit_pu`, `max_active_surplus_pu`, `active_recourse_cost`,
`target_penalty`, `target_violation`. They satisfy the identities

    total_active_recourse_energy_mwh == active_deficit_energy_mwh + active_surplus_energy_mwh
    active_recourse_cost             == active_recourse_cost_per_mwh * total_active_recourse_energy_mwh
    physical_operating_cost          == generator_cost + battery_throughput_cost + active_recourse_cost
    total_check                      == physical_operating_cost + target_penalty     (projected)

Raw diagnostics reproduce the solver objective exactly (the objective uses the raw
primal): `raw_active_deficit_pu`, `raw_active_surplus_pu`, `raw_active_recourse_cost`,
`raw_total_check ≈ result.objective`, `solver_objective_recompute_residual`, plus
`maximum_active_recourse_lower_bound_violation_pu` and
`active_recourse_projection_correction == active_recourse_cost − raw_active_recourse_cost`.

Recourse is reported by DIRECTION — never merged into one "shed" number and never a
per-load fraction (`d⁺` may exceed local demand or be positive where `p^d = 0`).
"""
function decompose_costs(prob::BatteryTSDDRProblem, result;
                         sol = tsddr_solution(prob, result),
                         lb_tol::Real = ACTIVE_RECOURSE_LB_TOL_PU)
    T = prob.horizon; nG = prob.nGen; nK = prob.nBat; nB = prob.nBus
    Rrep = prob.reporting_horizon

    # ── Validate the per-variable lower-bound noise, then project elementwise ──
    # `worst_viol` is how far the MOST-negative single recourse value sits below 0.
    worst_viol = 0.0
    if nB > 0
        for t in 1:T, b in 1:nB
            worst_viol = max(worst_viol, -Float64(sol.active_deficit_pu[b, t]),
                                          -Float64(sol.active_surplus_pu[b, t]))
        end
    end
    worst_viol <= lb_tol || error(
        "active-recourse lower-bound violation $worst_viol pu exceeds the declared " *
        "per-variable tolerance $lb_tol pu — a value this far below zero is a defect, " *
        "not interior-point noise. Refusing to report a physically impossible " *
        "negative recourse quantity.")

    # Generator + throughput are computed from the raw primal (they contribute to the
    # solver objective unchanged). Recourse is projected elementwise per bus/stage.
    gen_stage = zeros(Float64, T); cyc_stage = zeros(Float64, T); rec_stage = zeros(Float64, T)
    def_pu_stage = zeros(Float64, T); sur_pu_stage = zeros(Float64, T)              # projected
    raw_def_pu_stage = zeros(Float64, T); raw_sur_pu_stage = zeros(Float64, T)      # raw
    coeff = prob.active_recourse_cost_per_mwh * prob.baseMVA * prob.dt
    for t in 1:T
        for g in 1:nG
            pgv = Float64(sol.pg[g, t])
            gen_stage[t] += prob.gen_c2[g] * pgv^2 + prob.gen_c1[g] * pgv + prob.gen_c0[g]
        end
        for k in 1:nK
            cyc_stage[t] += prob.cycle_coeffs[k] *
                            (Float64(sol.p_ch[k, t]) + Float64(sol.p_dis[k, t]))
        end
        for b in 1:nB
            rd = Float64(sol.active_deficit_pu[b, t]); rs = Float64(sol.active_surplus_pu[b, t])
            raw_def_pu_stage[t] += rd;        raw_sur_pu_stage[t] += rs
            def_pu_stage[t] += max(rd, 0.0);  sur_pu_stage[t] += max(rs, 0.0)   # elementwise
        end
        rec_stage[t] = coeff * (def_pu_stage[t] + sur_pu_stage[t])   # from PROJECTED values
    end
    phys_stage = gen_stage .+ cyc_stage .+ rec_stage
    generator_cost = sum(gen_stage)
    throughput_cost = sum(cyc_stage)

    # Projected (public) aggregates.
    deficit_pu = sum(def_pu_stage); surplus_pu = sum(sur_pu_stage)
    active_recourse_cost = coeff * (deficit_pu + surplus_pu)
    physical = generator_cost + throughput_cost + active_recourse_cost
    max_deficit_pu = nB > 0 ? maximum(max(Float64(sol.active_deficit_pu[b, t]), 0.0)
                                      for t in 1:T for b in 1:nB) : 0.0
    max_surplus_pu = nB > 0 ? maximum(max(Float64(sol.active_surplus_pu[b, t]), 0.0)
                                      for t in 1:T for b in 1:nB) : 0.0

    # Raw (diagnostic) aggregates — reproduce the solver objective.
    raw_deficit_pu = sum(raw_def_pu_stage); raw_surplus_pu = sum(raw_sur_pu_stage)
    raw_active_recourse_cost = coeff * (raw_deficit_pu + raw_surplus_pu)
    projection_correction = active_recourse_cost - raw_active_recourse_cost

    # Soft target penalty — raw for objective reproduction, projected for reporting.
    target_penalty = 0.0; target_violation = 0.0
    raw_target_penalty = 0.0
    if prob.mode === :soft && nK > 0
        slack_viol = 0.0
        for x in sol.target_slack_pos
            slack_viol = max(slack_viol, -Float64(x))
        end
        for x in sol.target_slack_neg
            slack_viol = max(slack_viol, -Float64(x))
        end
        slack_viol <= lb_tol || error(
            "target-slack lower-bound violation $slack_viol pu exceeds the declared " *
            "tolerance $lb_tol pu — refusing to report a negative target slack/penalty.")
        raw_sp = sum(x -> Float64(x), sol.target_slack_pos)
        raw_sn = sum(x -> Float64(x), sol.target_slack_neg)
        proj_sp = sum(x -> max(Float64(x), 0.0), sol.target_slack_pos)   # elementwise
        proj_sn = sum(x -> max(Float64(x), 0.0), sol.target_slack_neg)
        raw_target_penalty = prob.rho1 * (raw_sp + raw_sn) +
            (prob.rho2 / 2) * (sum(x -> Float64(x)^2, sol.target_slack_pos) +
                               sum(x -> Float64(x)^2, sol.target_slack_neg))
        target_violation = proj_sp + proj_sn
        target_penalty = prob.rho1 * target_violation +
            (prob.rho2 / 2) * (sum(x -> max(Float64(x), 0.0)^2, sol.target_slack_pos) +
                               sum(x -> max(Float64(x), 0.0)^2, sol.target_slack_neg))
    end

    raw_total_check = generator_cost + throughput_cost + raw_active_recourse_cost + raw_target_penalty
    solver_objective_recompute_residual = abs(raw_total_check - Float64(result.objective))

    return (total_solver_objective = Float64(result.objective),
            generator_cost = generator_cost,
            battery_throughput_cost = throughput_cost,
            # ── public (projected, always ≥ 0) ──
            active_recourse_cost = active_recourse_cost,
            active_deficit_pu = deficit_pu,
            active_surplus_pu = surplus_pu,
            active_deficit_energy_mwh = prob.baseMVA * prob.dt * deficit_pu,
            active_surplus_energy_mwh = prob.baseMVA * prob.dt * surplus_pu,
            total_active_recourse_energy_mwh = prob.baseMVA * prob.dt * (deficit_pu + surplus_pu),
            max_active_deficit_pu = max_deficit_pu,
            max_active_surplus_pu = max_surplus_pu,
            physical_operating_cost = physical,
            target_penalty = target_penalty,
            target_violation = target_violation,
            reporting_physical_cost = sum(@view phys_stage[1:Rrep]),
            lookahead_physical_cost = T > Rrep ? sum(@view phys_stage[Rrep+1:T]) : 0.0,
            total_check = physical + target_penalty,
            # ── raw diagnostics (reproduce the solver objective) ──
            raw_active_deficit_pu = raw_deficit_pu,
            raw_active_surplus_pu = raw_surplus_pu,
            raw_active_recourse_cost = raw_active_recourse_cost,
            raw_target_penalty = raw_target_penalty,
            raw_total_check = raw_total_check,
            solver_objective_recompute_residual = solver_objective_recompute_residual,
            maximum_active_recourse_lower_bound_violation_pu = worst_viol,
            active_recourse_projection_correction = projection_correction,
            gen_stage = gen_stage, cyc_stage = cyc_stage,
            rec_stage = rec_stage, phys_stage = phys_stage)
end

"""
    solve_stage_with_starts(prob, e_prev, w_t, target; madnlp_kwargs,
                            soft_seed = nothing, prev_solution = nothing)
        -> (result, log)

Solve one stage trying a FIXED, deterministic sequence of primal STARTS and
accepting the FIRST solver-accepted result (never the cheapest):

1. flat start (`vm = 1`);
2. target-consistent battery start ([`target_consistent_start!`](@ref));
3. seed from the corresponding solved soft / targetless-diagnostic ACP point (`soft_seed`);
4. previous-stage accepted solution (`prev_solution`), where applicable.

Only the starting point varies. Tolerances, iteration limits, equations, bounds,
and all generator/network data are identical across attempts. Every attempt and
its solver status are returned in `log` (a vector of `(start, status, accepted)`).
"""
function solve_stage_with_starts(prob::BatteryTSDDRProblem, e_prev::AbstractVector,
                                 w_t::AbstractVector, target::AbstractVector;
                                 madnlp_kwargs = (print_level = MadNLP.ERROR, tol = 1e-6),
                                 soft_seed = nothing, prev_solution = nothing)
    is_targetless(prob) &&
        error("solve_stage_with_starts requires a target mode; got a targetless diagnostic")
    set_tsddr_initial_soc!(prob, e_prev)
    set_tsddr_uncertainty!(prob, w_t)
    set_tsddr_targets!(prob, target)

    attempts = Any[(:flat, () -> reset_flat_start!(prob)),
                   (:target_consistent, () -> (reset_flat_start!(prob);
                                               target_consistent_start!(prob, e_prev, target)))]
    soft_seed === nothing ||
        push!(attempts, (:soft_seed, () -> seed_start_from_solution!(prob, soft_seed)))
    prev_solution === nothing ||
        push!(attempts, (:previous_stage, () -> seed_start_from_solution!(prob, prev_solution)))

    log = Tuple{Symbol,Any,Bool}[]
    local result
    for (name, setup) in attempts
        setup()
        result = MadNLP.madnlp(prob.model; madnlp_kwargs...)
        ok = solve_succeeded_result(result) && isfinite(result.objective)
        push!(log, (name, result.status, ok))
        ok && return result, log
    end
    return result, log      # all starts failed; caller reports the log
end

# Accepted-status predicate local to this file (avoids the exported-name clash
# between the Phase-1 status predicate and the DecisionRulesExa result predicate).
solve_succeeded_result(result) =
    result.status == MadNLP.SOLVE_SUCCEEDED ||
    result.status == MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL

"""
    tsddr_balance_residuals(prob, sol) -> Matrix

`[nBat × T]` residuals of the battery energy balance, which should be ≈ 0.
"""
function tsddr_balance_residuals(prob::BatteryTSDDRProblem, sol)
    T = prob.horizon; nK = prob.nBat
    res = zeros(Float64, nK, T)
    for (k, bat) in enumerate(prob.case.batteries), t in 1:T
        res[k, t] = Float64(sol.soc[k, t+1]) - (1 - bat.sigma * prob.dt) * Float64(sol.soc[k, t]) -
                    bat.eta_ch * prob.dt * Float64(sol.p_ch[k, t]) +
                    (prob.dt / bat.eta_dis) * Float64(sol.p_dis[k, t])
    end
    return res
end

"""
    tsddr_max_primal_residual(prob, result) -> Float64

Largest constraint-bound violation of the returned point via the NLPModels
interface (`max(lcon − c, c − ucon, 0)`).
"""
function tsddr_max_primal_residual(prob::BatteryTSDDRProblem, result)
    x = Array(result.solution)
    c = Array(NLPModels.cons(prob.model, x))
    lcon = Array(prob.model.meta.lcon); ucon = Array(prob.model.meta.ucon)
    viol = 0.0
    @inbounds for i in eachindex(c)
        viol = max(viol, lcon[i] - c[i], c[i] - ucon[i], 0.0)
    end
    return viol
end
