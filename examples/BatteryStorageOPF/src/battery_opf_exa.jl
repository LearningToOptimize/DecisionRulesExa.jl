# battery_opf_exa.jl
#
# Full-horizon ExaModels AC-polar deterministic equivalent for the
# battery-storage OPF (BATTERY_STORAGE_OPF_PLAN.md §4). Built directly in
# ExaModels — no JuMP, no MOI.
#
# Model (T stages, stage length Δt hours):
#   min  Σ_t [ Σ_g (c2_g·pg² + c1_g·pg + c0_g)
#              + Σ_b c_cycle·(pᶜʰ_{t,b} + pᵈⁱˢ_{t,b}) ]
#   s.t. AC-polar power flow (ref angle, 4 branch-flow eqs, angle limits,
#          apparent-power thermal limits at both ends);
#        hard active balance incl. battery injection p_bat = pᵈⁱˢ − pᶜʰ;
#        hard reactive balance (no reactive slack, unity-PF batteries);
#        e_{1,b} = e_init,b;
#        e_{t+1,b} = (1−σ_bΔt)·e_{t,b} + η_ch,b·Δt·pᶜʰ − (Δt/η_dis,b)·pᵈⁱˢ;
#        0 ≤ pᶜʰ ≤ p̄ᶜʰ, 0 ≤ pᵈⁱˢ ≤ p̄ᵈⁱˢ, e_min ≤ e ≤ e_max.
#
# There is NO active-recourse variable and NO reactive-slack variable: balance is
# hard on both axes. Batteries support arbitrary (non-consecutive) host-bus ids
# through the network's stable id→position maps.

using ExaModels
using MadNLP
using NLPModels
using LinearAlgebra

# The AC-polar equations, index helpers, and branch coefficients live in the
# SHARED acp_core.jl (single source of truth), which this builder and the
# Phase-2 stochastic builder both call. This file keeps the Phase-1 assembly and
# its accepted behavior: no active recourse, no target constraints — the
# hard-balance base-ACP parity artifact.

"""
    BatteryExaProblem

Holds the ExaModels deterministic equivalent for a [`BatteryCase`].

Fields: the `core`/`model`, the tunable parameters (`p_pd`, `p_qd`, `p_e0`),
problem sizes, horizon `T`, stage length `dt` (hours), the per-battery cycle-cost
coefficients (`cycle_coeffs`), and back-references to the case for extraction.
"""
struct BatteryExaProblem
    core
    model
    p_pd            # active demand parameter   (length T*nBus)
    p_qd            # reactive demand parameter (length T*nBus)
    p_e0            # initial-SoC parameter     (length nBat)
    nBus::Int
    nGen::Int
    nBranch::Int
    nBat::Int
    horizon::Int
    dt::Float64
    # Per-battery cycle-cost objective coefficient (length nBat): entry k is
    # cycle_cost_per_mwh_k · baseMVA · Δt, the $/pu weight on (pᶜʰ_k + pᵈⁱˢ_k).
    cycle_coeffs::Vector{Float64}
    case::BatteryCase
    float_type::Type
end

"""
    build_battery_de(case, T; backend=nothing, float_type=Float64,
                     stage_hours=1.0, demand_profile=ones(T)) -> BatteryExaProblem

Build the `T`-stage ExaModels AC-polar deterministic equivalent for `case`.

Generator costs are PGLib polynomial USD/HOUR values, so each stage contributes
`Δt·(c2·pg² + c1·pg + c0)` to the objective (Δt = `stage_hours`); the battery
cycle cost `cᵦ·(pᶜʰ + pᵈⁱˢ)` already includes Δt through `cᵦ`. The reported
objective is therefore dollars over the whole horizon.

* `stage_hours` — Δt, the physical length of one stage in hours (default 1.0).
* `demand_profile` — length-`T` vector of per-stage load multipliers applied to
  BOTH active and reactive demand (preserving each bus's power factor); default
  all ones (flat base demand).

`backend=nothing` builds on the CPU. Requires `σ·Δt < 1` for every battery so
the self-discharge factor `(1 − σΔt)` stays positive.
"""
function build_battery_de(case::BatteryCase, T::Int;
                          backend = nothing,
                          float_type::Type{<:AbstractFloat} = Float64,
                          stage_hours::Real = 1.0,
                          demand_profile::AbstractVector = ones(T))
    T >= 1 || error("horizon T must be ≥ 1; got $T")
    length(demand_profile) == T ||
        error("demand_profile must have length T=$T; got $(length(demand_profile))")
    dt = Float64(stage_hours)
    (isfinite(dt) && dt > 0) ||
        error("stage_hours (Δt) must be finite and > 0; got $dt")
    for b in case.batteries
        b.sigma * dt < 1 ||
            error("battery $(b.id) has σ·Δt = $(b.sigma*dt) ≥ 1; " *
                  "reduce stage_hours or self_discharge_rate")
    end
    nd = case.network
    nBus = nbus(nd); nGen = ngen(nd); nBranch = nbranch(nd)
    nBat = length(case.batteries)

    # Concrete (immutable) ExaCore + the supported functional builder API
    # (`add_var`/`add_par`/`add_obj`/`add_con`/`add_con!`), each returning
    # `(new_core, handle)`. Handles carry their own offsets and stay valid as the
    # core grows, so we thread `core` through and keep the handles. This is the
    # non-deprecated path for ExaModels 0.11.2 (`ExaCore()` without `concrete`
    # returns the deprecated LegacyExaCore and warns).
    core = ExaModels.ExaCore(float_type; backend = backend, concrete = Val(true))

    # ── Variables (SHARED blocks, canonical order) ────────────────────────────
    core, v   = add_acp_variables!(core, nd, T, float_type)
    core, bat = add_battery_variables!(core, case, T, float_type)

    # ── Parameters (per-stage demand + initial SoC) ───────────────────────────
    # Phase 1 bakes the fixed `demand_profile` into the demand parameters.
    prof = Float64.(collect(demand_profile))
    init_pd = float_type.([nd.bus_pd[b] * prof[t] for t in 1:T for b in 1:nBus])
    init_qd = float_type.([nd.bus_qd[b] * prof[t] for t in 1:T for b in 1:nBus])
    core, p_pd = ExaModels.add_par(core, init_pd)
    core, p_qd = ExaModels.add_par(core, init_qd)
    core, p_e0 = ExaModels.add_par(core, float_type.([b.e_init for b in case.batteries]))

    # ── Objective (SHARED): generator cost + battery throughput cost ──────────
    core = add_generator_cost!(core, v, nd, T, dt, float_type)
    core = add_cycle_cost!(core, bat, case, nd, T, dt, float_type)

    # ── Constraints (SHARED). Phase 1 passes `active_deficit = nothing`: hard
    # active and reactive balance with no recourse — the accepted parity artifact.
    core = add_acp_network_constraints!(core, v, nd, T, float_type)
    core = add_nodal_balance!(core, v, bat, nd, case, T, float_type, p_pd, p_qd;
                              active_deficit = nothing)
    core = add_battery_dynamics!(core, bat, case, T, dt, float_type, p_e0)

    cycle_coeffs = Float64[b.cycle_cost_per_mwh * nd.baseMVA * dt for b in case.batteries]

    model = ExaModels.ExaModel(core)
    return BatteryExaProblem(core, model, p_pd, p_qd, p_e0,
                             nBus, nGen, nBranch, nBat, T, dt,
                             cycle_coeffs, case, float_type)
end

# ── Parameter setters ─────────────────────────────────────────────────────────

"""
    set_demand!(prob, pd::AbstractMatrix, qd::AbstractMatrix)

Overwrite the per-stage active/reactive demand parameters. `pd`, `qd` are
`[T × nBus]` (pu). Both must be supplied so the reactive/active balance stay
consistent.
"""
function set_demand!(prob::BatteryExaProblem, pd::AbstractMatrix, qd::AbstractMatrix)
    T, nB = prob.horizon, prob.nBus
    size(pd) == (T, nB) || error("pd must be [T=$T × nBus=$nB]")
    size(qd) == (T, nB) || error("qd must be [T=$T × nBus=$nB]")
    ExaModels.set_parameter!(prob.core, prob.p_pd,
        prob.float_type.([pd[t, b] for t in 1:T for b in 1:nB]))
    ExaModels.set_parameter!(prob.core, prob.p_qd,
        prob.float_type.([qd[t, b] for t in 1:T for b in 1:nB]))
    return prob
end

"""
    set_initial_soc!(prob, e0::AbstractVector)

Overwrite the initial state-of-charge parameter (length `nBat`, pu·h).
"""
function set_initial_soc!(prob::BatteryExaProblem, e0::AbstractVector)
    length(e0) == prob.nBat || error("e0 must have length nBat=$(prob.nBat)")
    ExaModels.set_parameter!(prob.core, prob.p_e0, prob.float_type.(collect(e0)))
    return prob
end

# ── Solve + structured extraction ─────────────────────────────────────────────

solve_succeeded(status) = status == MadNLP.SOLVE_SUCCEEDED ||
                          status == MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL

"""
    solve_de!(prob; print_level=MadNLP.ERROR, kwargs...) -> result

Solve the deterministic equivalent on the CPU with MadNLP. Returns the raw
MadNLP result (with `.status`, `.objective`, `.solution`, `.multipliers`).

Named `solve_de!` rather than `solve!` to avoid clashing with the `solve!`
that JuMP/MadNLP re-export from CommonSolve.
"""
solve_de!(prob::BatteryExaProblem; print_level = MadNLP.ERROR, kwargs...) =
    MadNLP.madnlp(prob.model; print_level = print_level, kwargs...)

"""
    battery_solution(prob, result) -> NamedTuple

Reshape the flat solution into named `[·×T]` (or `[·×(T+1)]` for SoC) matrices:
`va, vm` (nBus), `pg, qg` (nGen), `p_fr, q_fr, p_to, q_to` (nBranch),
`p_ch, p_dis` (nBat), `soc` (nBat×(T+1)), and derived `p_bat = p_dis − p_ch`.
"""
function battery_solution(prob::BatteryExaProblem, result)
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
    return (va = va, vm = vm, pg = pg, qg = qg,
            p_fr = p_fr, q_fr = q_fr, p_to = p_to, q_to = q_to,
            p_ch = p_ch, p_dis = p_dis, soc = soc, p_bat = p_dis .- p_ch)
end

"""
    battery_balance_residuals(prob, sol) -> Matrix

`[nBat × T]` residuals of the state equation
`e[t+1] − (1−σΔt)·e[t] − η_ch·Δt·p_ch + (Δt/η_dis)·p_dis`, which should be ~0.
"""
function battery_balance_residuals(prob::BatteryExaProblem, sol)
    T = prob.horizon; nK = prob.nBat
    res = zeros(Float64, nK, T)
    for (k, bat) in enumerate(prob.case.batteries), t in 1:T
        res[k, t] = sol.soc[k, t+1] - (1 - bat.sigma * prob.dt) * sol.soc[k, t] -
                    bat.eta_ch * prob.dt * sol.p_ch[k, t] +
                    (prob.dt / bat.eta_dis) * sol.p_dis[k, t]
    end
    return res
end

"""
    simultaneous_charge_discharge_power(sol) -> Matrix

`[nBat × T]` **simultaneous charge/discharge power** in pu: the elementwise
smaller of the charge and discharge powers, `min(p_ch, p_dis)` (NOT a product).
It is the amount of power a battery is charging and discharging at the same time;
material positive entries flag unwanted simultaneous operation, which the
nonnegative cycle cost is designed to prevent.
"""
simultaneous_charge_discharge_power(sol) = min.(sol.p_ch, sol.p_dis)

"""
    max_primal_residual(prob, result) -> Float64

Largest constraint-bound violation of the returned point, evaluated
independently through the NLPModels interface (max over
`max(lcon − c, c − ucon, 0)`).
"""
function max_primal_residual(prob::BatteryExaProblem, result)
    x = Array(result.solution)
    c = NLPModels.cons(prob.model, x)
    lcon = Array(prob.model.meta.lcon); ucon = Array(prob.model.meta.ucon)
    c = Array(c)
    viol = 0.0
    @inbounds for i in eachindex(c)
        viol = max(viol, lcon[i] - c[i], c[i] - ucon[i], 0.0)
    end
    return viol
end
