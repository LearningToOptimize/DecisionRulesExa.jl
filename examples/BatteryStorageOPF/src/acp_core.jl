# acp_core.jl
#
# SHARED AC-polar (ACP) construction — the single source of truth for the battery
# AC-OPF equations (canonical spec: docs/src/casestudies/battery_storage_opf.md,
# "Implementation invariants": *one shared ACP constraint implementation underlies
# deterministic, stochastic, training, and evaluation builders*).
#
# Both builders call these blocks:
#   * `build_battery_de`        (Phase-1 deterministic foundation, no active
#                                recourse, no targets — accepted base-ACP parity
#                                artifact);
#   * `build_battery_tsddr_de`  (Phase-2 operational/stochastic model: adds the
#                                two-sided absolute active nodal recourse (deficit
#                                d⁺ + surplus d⁻) and the target equations).
#
# Nothing here changes generator prices, pmin/pmax, qmin/qmax, branch limits,
# voltage limits, admittances, or topology: every quantity is read from the
# accepted `NetworkData`/`BatteryCase` produced by network_data.jl/battery_data.jl.
#
# ── Variable creation order (shared; extra blocks appended by the caller) ──────
#   va, vm, pg, qg, p_fr, q_fr, p_to, q_to, p_ch, p_dis, e,
#   [active_deficit, active_surplus]  (Phase 2 only)
#   [slack_pos, slack_neg]  (Phase 2 soft target mode only)
#
# ── Constraint creation order (shared) ────────────────────────────────────────
#   1 reference angle              T·nRef
#   2 from-end active flow         T·nBranch
#   3 from-end reactive flow       T·nBranch
#   4 to-end active flow           T·nBranch
#   5 to-end reactive flow         T·nBranch
#   6 angle-difference limits      T·nBranch
#   7 thermal limit (from end)     T·nRated
#   8 thermal limit (to end)       T·nRated
#   9 active nodal balance         T·nBus
#  10 reactive nodal balance       T·nBus   (HARD equality: no reactive slack)
#  11 battery initial SoC          nBat
#  12 battery energy balance       T·nBat
#   ... then the caller appends TARGET constraints LAST (Phase 2).
#
# `acp_constraint_count` reproduces that total so the caller can compute the
# contiguous target-multiplier slice exactly.

using ExaModels

# Flat stage-major index helpers, (t,i) → (t-1)*n + i.
@inline _bidx(nB, t, b)   = (t - 1) * nB + b     # bus        (t = 1..T)
@inline _gidx(nG, t, g)   = (t - 1) * nG + g     # generator  (t = 1..T)
@inline _bridx(nBR, t, r) = (t - 1) * nBR + r    # branch     (t = 1..T)
@inline _kidx(nK, t, k)   = (t - 1) * nK + k     # battery power (t = 1..T)
@inline _eidx(nK, t, k)   = (t - 1) * nK + k     # battery SoC   (t = 1..T+1)

"""
    _ac_branch_coeffs(br, T) -> NamedTuple

ExaModelsPower/PowerModels-compatible AC-polar branch coefficients `c1..c8`,
retaining the transformer tap and phase shift and both line-charging shunts:

```
tr = tap·cos(shift), ti = tap·sin(shift), ttm = tr² + ti²
g  =  br_r/(br_r²+br_x²),  b = −br_x/(br_r²+br_x²)
c1 = (−g·tr − b·ti)/ttm    c2 = (−b·tr + g·ti)/ttm
c3 = (−g·tr + b·ti)/ttm    c4 = (−b·tr − g·ti)/ttm
c5 = (g + g_fr)/ttm        c6 = (b + b_fr)/ttm
c7 =  g + g_to             c8 =  b + b_to
```
"""
function _ac_branch_coeffs(br::BranchData, ::Type{T}) where {T}
    r2x2 = br.br_r^2 + br.br_x^2
    g = r2x2 > 0 ? T(br.br_r / r2x2) : zero(T)
    b = r2x2 > 0 ? T(-br.br_x / r2x2) : zero(T)
    tap = T(br.tap); sh = T(br.shift)
    tr = tap * cos(sh); ti = tap * sin(sh)
    ttm = tr^2 + ti^2
    ttm = ttm > 0 ? ttm : one(T)
    return (
        c1 = (-g * tr - b * ti) / ttm,
        c2 = (-b * tr + g * ti) / ttm,
        c3 = (-g * tr + b * ti) / ttm,
        c4 = (-b * tr - g * ti) / ttm,
        c5 = (g + T(br.g_fr)) / ttm,
        c6 = (b + T(br.b_fr)) / ttm,
        c7 = g + T(br.g_to),
        c8 = b + T(br.b_to),
    )
end

"""
    rated_branch_positions(nd) -> Vector{Int}

Array positions of branches carrying a finite PGLib apparent-power limit. An
absent limit adds no artificial finite bound (canonical spec, Appendix A).
"""
rated_branch_positions(nd::NetworkData) =
    [bp for bp in 1:nbranch(nd) if isfinite(nd.branches[bp].rate_a)]

"""
    acp_constraint_count(nd, case, T) -> Int

Number of constraints the shared ACP + battery blocks add, in creation order
(see the file header). The Phase-2 builder appends its target constraints after
these, so its contiguous target-multiplier slice starts at this count + 1.
"""
function acp_constraint_count(nd::NetworkData, case::BatteryCase, T::Int)
    nBus = nbus(nd); nBranch = nbranch(nd); nBat = length(case.batteries)
    nRef = length(nd.ref_bus_positions)
    nRated = length(rated_branch_positions(nd))
    return T * nRef + 5 * (T * nBranch) + 2 * (T * nRated) +
           2 * (T * nBus) + nBat + (T * nBat)
end

# ── Variables ─────────────────────────────────────────────────────────────────

"""
    add_acp_variables!(core, nd, T, ft) -> (core, vars)

Add the shared AC-polar variables in canonical order and return them as a
NamedTuple `(va, vm, pg, qg, p_fr, q_fr, p_to, q_to)`.

Bounds come straight from the PGLib data: voltage magnitude limits, generator
active/reactive limits (non-finite reactive limits fall back to ±1e4 pu), and
branch-flow box bounds from `rate_a` (±1e4 pu when unlimited).
"""
function add_acp_variables!(core, nd::NetworkData, T::Int, ft::Type{<:AbstractFloat})
    nBus = nbus(nd); nGen = ngen(nd); nBranch = nbranch(nd)
    core, va = ExaModels.add_var(core, T * nBus)
    core, vm = ExaModels.add_var(core, T * nBus;
        lvar = ft.(repeat([b.vmin for b in nd.buses], T)),
        uvar = ft.(repeat([b.vmax for b in nd.buses], T)),
        start = ones(ft, T * nBus))
    core, pg = ExaModels.add_var(core, T * nGen;
        lvar = ft.(repeat([g.pmin for g in nd.gens], T)),
        uvar = ft.(repeat([g.pmax for g in nd.gens], T)))
    core, qg = ExaModels.add_var(core, T * nGen;
        lvar = ft.(repeat([isfinite(g.qmin) ? g.qmin : -1e4 for g in nd.gens], T)),
        uvar = ft.(repeat([isfinite(g.qmax) ? g.qmax :  1e4 for g in nd.gens], T)))
    fr_lb = ft.(repeat([isfinite(b.rate_a) ? -b.rate_a : -1e4 for b in nd.branches], T))
    fr_ub = ft.(repeat([isfinite(b.rate_a) ?  b.rate_a :  1e4 for b in nd.branches], T))
    core, p_fr = ExaModels.add_var(core, T * nBranch; lvar = fr_lb, uvar = fr_ub)
    core, q_fr = ExaModels.add_var(core, T * nBranch; lvar = fr_lb, uvar = fr_ub)
    core, p_to = ExaModels.add_var(core, T * nBranch; lvar = fr_lb, uvar = fr_ub)
    core, q_to = ExaModels.add_var(core, T * nBranch; lvar = fr_lb, uvar = fr_ub)
    return core, (va = va, vm = vm, pg = pg, qg = qg,
                  p_fr = p_fr, q_fr = q_fr, p_to = p_to, q_to = q_to)
end

"""
    add_battery_variables!(core, case, T, ft) -> (core, bat)

Add battery charge/discharge power (`0 ≤ p ≤ p̄`, per battery) and the `(T+1)`
energy states bounded by `[e_min, e_max]` and started at `e_init`. Returns
`(p_ch, p_dis, e)`.
"""
function add_battery_variables!(core, case::BatteryCase, T::Int, ft::Type{<:AbstractFloat})
    nBat = length(case.batteries)
    pch_ub  = ft.(repeat([b.p_charge_max    for b in case.batteries], T))
    pdis_ub = ft.(repeat([b.p_discharge_max for b in case.batteries], T))
    core, p_ch  = ExaModels.add_var(core, T * nBat; lvar = ft(0), uvar = pch_ub)
    core, p_dis = ExaModels.add_var(core, T * nBat; lvar = ft(0), uvar = pdis_ub)
    core, e = ExaModels.add_var(core, (T + 1) * nBat;
        lvar = ft.(repeat([b.e_min for b in case.batteries], T + 1)),
        uvar = ft.(repeat([b.e_max for b in case.batteries], T + 1)),
        start = ft.(repeat([b.e_init for b in case.batteries], T + 1)))
    return core, (p_ch = p_ch, p_dis = p_dis, e = e)
end

# ── Objective blocks ──────────────────────────────────────────────────────────

"""
    add_generator_cost!(core, v, nd, T, dt, ft) -> core

Original PGLib polynomial generator cost, duration-scaled:
`Δt·(c2·pg² + c1·pg + c0)` — every term, including the constant `c0`, is
multiplied by `Δt` (canonical spec, "Stage objective and cost accounting").
"""
function add_generator_cost!(core, v, nd::NetworkData, T::Int, dt::Float64, ft::Type{<:AbstractFloat})
    nGen = ngen(nd)
    items = [(t = t, g = gp, c2 = ft(g.cost2 * dt), c1 = ft(g.cost1 * dt), c0 = ft(g.cost0 * dt))
             for t in 1:T for (gp, g) in enumerate(nd.gens)]
    core, _ = ExaModels.add_obj(core,
        it.c2 * v.pg[_gidx(nGen, it.t, it.g)]^2 + it.c1 * v.pg[_gidx(nGen, it.t, it.g)] + it.c0
        for it in items)
    return core
end

"""
    add_cycle_cost!(core, bat, case, nd, T, dt, ft) -> core

Battery throughput/degradation cost `S^base·Δt·c^cycle_b·(p_ch + p_dis)`,
computed per battery (the fleet need not share one price). Skipped when every
battery has a zero cycle price.
"""
function add_cycle_cost!(core, bat, case::BatteryCase, nd::NetworkData, T::Int,
                         dt::Float64, ft::Type{<:AbstractFloat})
    nBat = length(case.batteries)
    nBat == 0 && return core
    items = [(idx = _kidx(nBat, t, k), c = ft(b.cycle_cost_per_mwh * nd.baseMVA * dt))
             for t in 1:T for (k, b) in enumerate(case.batteries)]
    any(it -> it.c > 0, items) || return core
    core, _ = ExaModels.add_obj(core, it.c * (bat.p_ch[it.idx] + bat.p_dis[it.idx]) for it in items)
    return core
end

# ── Constraint blocks ─────────────────────────────────────────────────────────

"""
    add_acp_network_constraints!(core, v, nd, T, ft) -> core

Blocks 1–8: reference angle; the four AC-polar branch-end flow equalities (with
taps, shifts, and both line-charging shunts); angle-difference limits; and
apparent-power limits at BOTH ends for rate-limited branches only.
"""
function add_acp_network_constraints!(core, v, nd::NetworkData, T::Int, ft::Type{<:AbstractFloat})
    nBus = nbus(nd); nBranch = nbranch(nd)
    br_ac = [_ac_branch_coeffs(br, ft) for br in nd.branches]

    # 1. Reference angle: va[t, ref] = 0
    ref_items = [(t = t, ref = ref) for t in 1:T for ref in nd.ref_bus_positions]
    core, _ = ExaModels.add_con(core, v.va[_bidx(nBus, it.t, it.ref)] for it in ref_items)

    # 2. From-end active flow
    pfr_items = [(t = t, f = br.f_pos, tb = br.t_pos, br = bp,
                  c3 = br_ac[bp].c3, c4 = br_ac[bp].c4, c5 = br_ac[bp].c5)
                 for t in 1:T for (bp, br) in enumerate(nd.branches)]
    core, _ = ExaModels.add_con(core,
        v.p_fr[_bridx(nBranch, it.t, it.br)]
        - it.c5 * v.vm[_bidx(nBus, it.t, it.f)]^2
        - it.c3 * v.vm[_bidx(nBus, it.t, it.f)] * v.vm[_bidx(nBus, it.t, it.tb)]
          * cos(v.va[_bidx(nBus, it.t, it.f)] - v.va[_bidx(nBus, it.t, it.tb)])
        - it.c4 * v.vm[_bidx(nBus, it.t, it.f)] * v.vm[_bidx(nBus, it.t, it.tb)]
          * sin(v.va[_bidx(nBus, it.t, it.f)] - v.va[_bidx(nBus, it.t, it.tb)])
        for it in pfr_items)

    # 3. From-end reactive flow
    qfr_items = [(t = t, f = br.f_pos, tb = br.t_pos, br = bp,
                  c3 = br_ac[bp].c3, c4 = br_ac[bp].c4, c6 = br_ac[bp].c6)
                 for t in 1:T for (bp, br) in enumerate(nd.branches)]
    core, _ = ExaModels.add_con(core,
        v.q_fr[_bridx(nBranch, it.t, it.br)]
        + it.c6 * v.vm[_bidx(nBus, it.t, it.f)]^2
        + it.c4 * v.vm[_bidx(nBus, it.t, it.f)] * v.vm[_bidx(nBus, it.t, it.tb)]
          * cos(v.va[_bidx(nBus, it.t, it.f)] - v.va[_bidx(nBus, it.t, it.tb)])
        - it.c3 * v.vm[_bidx(nBus, it.t, it.f)] * v.vm[_bidx(nBus, it.t, it.tb)]
          * sin(v.va[_bidx(nBus, it.t, it.f)] - v.va[_bidx(nBus, it.t, it.tb)])
        for it in qfr_items)

    # 4. To-end active flow
    pto_items = [(t = t, f = br.f_pos, tb = br.t_pos, br = bp,
                  c1 = br_ac[bp].c1, c2 = br_ac[bp].c2, c7 = br_ac[bp].c7)
                 for t in 1:T for (bp, br) in enumerate(nd.branches)]
    core, _ = ExaModels.add_con(core,
        v.p_to[_bridx(nBranch, it.t, it.br)]
        - it.c7 * v.vm[_bidx(nBus, it.t, it.tb)]^2
        - it.c1 * v.vm[_bidx(nBus, it.t, it.tb)] * v.vm[_bidx(nBus, it.t, it.f)]
          * cos(v.va[_bidx(nBus, it.t, it.tb)] - v.va[_bidx(nBus, it.t, it.f)])
        - it.c2 * v.vm[_bidx(nBus, it.t, it.tb)] * v.vm[_bidx(nBus, it.t, it.f)]
          * sin(v.va[_bidx(nBus, it.t, it.tb)] - v.va[_bidx(nBus, it.t, it.f)])
        for it in pto_items)

    # 5. To-end reactive flow
    qto_items = [(t = t, f = br.f_pos, tb = br.t_pos, br = bp,
                  c1 = br_ac[bp].c1, c2 = br_ac[bp].c2, c8 = br_ac[bp].c8)
                 for t in 1:T for (bp, br) in enumerate(nd.branches)]
    core, _ = ExaModels.add_con(core,
        v.q_to[_bridx(nBranch, it.t, it.br)]
        + it.c8 * v.vm[_bidx(nBus, it.t, it.tb)]^2
        + it.c2 * v.vm[_bidx(nBus, it.t, it.tb)] * v.vm[_bidx(nBus, it.t, it.f)]
          * cos(v.va[_bidx(nBus, it.t, it.tb)] - v.va[_bidx(nBus, it.t, it.f)])
        - it.c1 * v.vm[_bidx(nBus, it.t, it.tb)] * v.vm[_bidx(nBus, it.t, it.f)]
          * sin(v.va[_bidx(nBus, it.t, it.tb)] - v.va[_bidx(nBus, it.t, it.f)])
        for it in qto_items)

    # 6. Angle-difference limits
    ang_lb = ft.(repeat([br.angmin for br in nd.branches], T))
    ang_ub = ft.(repeat([br.angmax for br in nd.branches], T))
    ang_items = [(t = t, f = br.f_pos, tb = br.t_pos) for t in 1:T for br in nd.branches]
    core, _ = ExaModels.add_con(core,
        v.va[_bidx(nBus, it.t, it.f)] - v.va[_bidx(nBus, it.t, it.tb)]
        for it in ang_items; lcon = ang_lb, ucon = ang_ub)

    # 7/8. Apparent-power thermal limits at both ends (rate-limited branches only).
    rated = rated_branch_positions(nd)
    if !isempty(rated)
        th_items = [(t = t, br = bp) for t in 1:T for bp in rated]
        th_ub = ft.([nd.branches[it.br].rate_a^2 for it in th_items])
        th_lb = fill(ft(-Inf), length(th_items))
        core, _ = ExaModels.add_con(core,
            v.p_fr[_bridx(nBranch, it.t, it.br)]^2 + v.q_fr[_bridx(nBranch, it.t, it.br)]^2
            for it in th_items; lcon = th_lb, ucon = th_ub)
        core, _ = ExaModels.add_con(core,
            v.p_to[_bridx(nBranch, it.t, it.br)]^2 + v.q_to[_bridx(nBranch, it.t, it.br)]^2
            for it in th_items; lcon = th_lb, ucon = th_ub)
    end
    return core
end

"""
    add_nodal_balance!(core, v, bat, nd, case, T, ft, p_pd, p_qd; active_deficit=nothing) -> core

Blocks 9–10: active and reactive nodal balance.

Active:   `p^d − d⁺ + d⁻ + gs·vm² − Σpg − Σ(p_dis−p_ch) + Σp_fr + Σp_to = 0`
Reactive: `q^d − bs·vm² − Σqg + Σq_fr + Σq_to = 0`

`active_deficit` (`d⁺ ≥ 0`) and `active_surplus` (`d⁻ ≥ 0`) are the per-bus
**two-sided active nodal recourse** (Phase 2), or both `nothing` (Phase 1, hard
balance). They are UNBOUNDED nonnegative variables at EVERY bus:

* `d⁺` (active deficit / injection) covers an active-power SHORTFALL — e.g. the
  power to CHARGE a battery at a network-constrained bus;
* `d⁻` (active surplus / absorption) absorbs an active-power EXCESS — e.g. the
  power a battery is FORCED to DISCHARGE into a bus whose outgoing branches are
  thermally saturated.

They are an artificial active-balance recourse, NOT curtailed customer load: `d⁺`
may exceed local demand and may be positive at a bus with `p^d = 0`. Together they
give relatively complete recourse: the stage subproblem is feasible for EVERY
incoming state and EVERY dynamically reachable battery target, in both the
charging and discharging directions. Both are priced at VOLL, so an accepted
solution leaves both at ~0. They do NOT touch reactive power: the reactive balance
stays a HARD equality with no slack (canonical spec, "Active load deficit" and
"Implementation invariants").
"""
function add_nodal_balance!(core, v, bat, nd::NetworkData, case::BatteryCase,
                            T::Int, ft::Type{<:AbstractFloat}, p_pd, p_qd;
                            active_deficit = nothing, active_surplus = nothing)
    nBus = nbus(nd); nGen = ngen(nd); nBranch = nbranch(nd)
    nBat = length(case.batteries)

    gen_bus_items = [(brow = _bidx(nBus, t, g.bus_pos), gcol = _gidx(nGen, t, gp))
                     for t in 1:T for (gp, g) in enumerate(nd.gens)]
    fr_bus_items = [(brow = _bidx(nBus, t, br.f_pos), bcol = _bridx(nBranch, t, bp))
                    for t in 1:T for (bp, br) in enumerate(nd.branches)]
    to_bus_items = [(brow = _bidx(nBus, t, br.t_pos), bcol = _bridx(nBranch, t, bp))
                    for t in 1:T for (bp, br) in enumerate(nd.branches)]
    bat_bus_items = [(brow = _bidx(nBus, t, b.bus_pos), kcol = _kidx(nBat, t, k))
                     for t in 1:T for (k, b) in enumerate(case.batteries)]

    # ── Active balance: p^d − d⁺ + d⁻ (both ≥ 0, unbounded above) ──────────────
    kcl_p_init = [(t = t, b = b, gs = ft(nd.buses[b].gs)) for t in 1:T for b in 1:nBus]
    core, c_kcl_p = if active_deficit === nothing
        ExaModels.add_con(core,
            p_pd[_bidx(nBus, it.t, it.b)] + it.gs * v.vm[_bidx(nBus, it.t, it.b)]^2
            for it in kcl_p_init)
    else
        ExaModels.add_con(core,
            p_pd[_bidx(nBus, it.t, it.b)] - active_deficit[_bidx(nBus, it.t, it.b)]
            + active_surplus[_bidx(nBus, it.t, it.b)]
            + it.gs * v.vm[_bidx(nBus, it.t, it.b)]^2
            for it in kcl_p_init)
    end
    core, _ = ExaModels.add_con!(core, c_kcl_p, it.brow => -v.pg[it.gcol] for it in gen_bus_items)
    core, _ = ExaModels.add_con!(core, c_kcl_p, it.brow => v.p_fr[it.bcol] for it in fr_bus_items)
    core, _ = ExaModels.add_con!(core, c_kcl_p, it.brow => v.p_to[it.bcol] for it in to_bus_items)
    if nBat > 0
        core, _ = ExaModels.add_con!(core, c_kcl_p, it.brow => -bat.p_dis[it.kcol] for it in bat_bus_items)
        core, _ = ExaModels.add_con!(core, c_kcl_p, it.brow => bat.p_ch[it.kcol]  for it in bat_bus_items)
    end

    # ── Reactive balance (HARD equality; no reactive slack, recourse is active-only) ─
    kcl_q_init = [(t = t, b = b, bs = ft(nd.buses[b].bs)) for t in 1:T for b in 1:nBus]
    core, c_kcl_q = ExaModels.add_con(core,
        p_qd[_bidx(nBus, it.t, it.b)] - it.bs * v.vm[_bidx(nBus, it.t, it.b)]^2
        for it in kcl_q_init)
    core, _ = ExaModels.add_con!(core, c_kcl_q, it.brow => -v.qg[it.gcol] for it in gen_bus_items)
    core, _ = ExaModels.add_con!(core, c_kcl_q, it.brow => v.q_fr[it.bcol] for it in fr_bus_items)
    core, _ = ExaModels.add_con!(core, c_kcl_q, it.brow => v.q_to[it.bcol] for it in to_bus_items)
    return core
end

"""
    add_battery_dynamics!(core, bat, case, T, dt, ft, p_x0) -> core

Blocks 11–12: the battery initial condition `e[1,b] − e0_b = 0` and the linear
energy balance
`e[t+1,b] − (1−σ_bΔt)e[t,b] − η^ch_bΔt·p_ch + (Δt/η^dis_b)·p_dis = 0`.
"""
function add_battery_dynamics!(core, bat, case::BatteryCase, T::Int, dt::Float64,
                               ft::Type{<:AbstractFloat}, p_x0)
    nBat = length(case.batteries)
    nBat == 0 && return core
    core, _ = ExaModels.add_con(core, bat.e[_eidx(nBat, 1, k)] - p_x0[k] for k in 1:nBat)
    st_items = [(en = _eidx(nBat, t + 1, k), ec = _eidx(nBat, t, k), pc = _kidx(nBat, t, k),
                 a = ft(1 - b.sigma * dt), bch = ft(b.eta_ch * dt), bdis = ft(dt / b.eta_dis))
                for t in 1:T for (k, b) in enumerate(case.batteries)]
    core, _ = ExaModels.add_con(core,
        bat.e[it.en] - it.a * bat.e[it.ec] - it.bch * bat.p_ch[it.pc] + it.bdis * bat.p_dis[it.pc]
        for it in st_items)
    return core
end

"""
    validate_stage_hours(case, dt)

Shared validation: `Δt` finite and positive, and `σ_b·Δt < 1` for every battery
so the self-discharge retention `(1 − σΔt)` stays positive.
"""
function validate_stage_hours(case::BatteryCase, dt::Real)
    (isfinite(dt) && dt > 0) || error("stage_hours (Δt) must be finite and > 0; got $dt")
    for b in case.batteries
        b.sigma * dt < 1 ||
            error("battery $(b.id) has σ·Δt = $(b.sigma*dt) ≥ 1; " *
                  "reduce stage_hours or self_discharge_rate")
    end
    return nothing
end
