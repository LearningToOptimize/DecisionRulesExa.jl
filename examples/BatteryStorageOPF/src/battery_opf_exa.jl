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
# There is NO load-shedding variable and NO reactive-slack variable: balance is
# hard on both axes. Batteries support arbitrary (non-consecutive) host-bus ids
# through the network's stable id→position maps.

using ExaModels
using MadNLP
using NLPModels
using LinearAlgebra

# ── Flat index helpers (stage-major: (t,i) → (t-1)*n + i) ─────────────────────
@inline _bidx(nB, t, b)   = (t - 1) * nB + b     # bus       (t = 1..T)
@inline _gidx(nG, t, g)   = (t - 1) * nG + g     # generator (t = 1..T)
@inline _bridx(nBR, t, r) = (t - 1) * nBR + r    # branch    (t = 1..T)
@inline _kidx(nK, t, k)   = (t - 1) * nK + k     # battery power (t = 1..T)
@inline _eidx(nK, t, k)   = (t - 1) * nK + k     # battery SoC   (t = 1..T+1)

# ExaModelsPower-compatible branch coefficients c1..c8 (AC polar), matching the
# convention used throughout this project. See the docstring block below for the
# resulting power-flow expressions.
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
    baseMVA = float_type(nd.baseMVA)

    # Concrete (immutable) ExaCore + the supported functional builder API
    # (`add_var`/`add_par`/`add_obj`/`add_con`/`add_con!`), each returning
    # `(new_core, handle)`. Handles carry their own offsets and stay valid as the
    # core grows, so we thread `core` through and keep the handles. This is the
    # non-deprecated path for ExaModels 0.11.2 (`ExaCore()` without `concrete`
    # returns the deprecated LegacyExaCore and warns).
    core = ExaModels.ExaCore(float_type; backend = backend, concrete = Val(true))

    # ── Variables ─────────────────────────────────────────────────────────────
    core, va = ExaModels.add_var(core, T * nBus)
    core, vm = ExaModels.add_var(core, T * nBus;
                            lvar = float_type.(repeat([b.vmin for b in nd.buses], T)),
                            uvar = float_type.(repeat([b.vmax for b in nd.buses], T)),
                            start = ones(float_type, T * nBus))
    core, pg = ExaModels.add_var(core, T * nGen;
                            lvar = float_type.(repeat([g.pmin for g in nd.gens], T)),
                            uvar = float_type.(repeat([g.pmax for g in nd.gens], T)))
    core, qg = ExaModels.add_var(core, T * nGen;
                            lvar = float_type.(repeat([isfinite(g.qmin) ? g.qmin : -1e4 for g in nd.gens], T)),
                            uvar = float_type.(repeat([isfinite(g.qmax) ? g.qmax :  1e4 for g in nd.gens], T)))
    fr_lb = float_type.(repeat([isfinite(b.rate_a) ? -b.rate_a : -1e4 for b in nd.branches], T))
    fr_ub = float_type.(repeat([isfinite(b.rate_a) ?  b.rate_a :  1e4 for b in nd.branches], T))
    core, p_fr = ExaModels.add_var(core, T * nBranch; lvar = fr_lb, uvar = fr_ub)
    core, q_fr = ExaModels.add_var(core, T * nBranch; lvar = fr_lb, uvar = fr_ub)
    core, p_to = ExaModels.add_var(core, T * nBranch; lvar = fr_lb, uvar = fr_ub)
    core, q_to = ExaModels.add_var(core, T * nBranch; lvar = fr_lb, uvar = fr_ub)

    # Battery charge / discharge power (pu, non-negative, per-battery capped).
    pch_ub = float_type.(repeat([b.p_charge_max    for b in case.batteries], T))
    pdis_ub = float_type.(repeat([b.p_discharge_max for b in case.batteries], T))
    core, p_ch  = ExaModels.add_var(core, T * nBat; lvar = float_type(0), uvar = pch_ub)
    core, p_dis = ExaModels.add_var(core, T * nBat; lvar = float_type(0), uvar = pdis_ub)
    # Battery SoC (pu·h): (T+1) points, bounded by [e_min, e_max].
    core, e = ExaModels.add_var(core, (T + 1) * nBat;
                           lvar = float_type.(repeat([b.e_min for b in case.batteries], T + 1)),
                           uvar = float_type.(repeat([b.e_max for b in case.batteries], T + 1)),
                           start = float_type.(repeat([b.e_init for b in case.batteries], T + 1)))

    # ── Parameters (per-stage demand + initial SoC) ───────────────────────────
    prof = Float64.(collect(demand_profile))
    init_pd = float_type.([nd.bus_pd[b] * prof[t] for t in 1:T for b in 1:nBus])
    init_qd = float_type.([nd.bus_qd[b] * prof[t] for t in 1:T for b in 1:nBus])
    core, p_pd = ExaModels.add_par(core, init_pd)
    core, p_qd = ExaModels.add_par(core, init_qd)
    core, p_e0 = ExaModels.add_par(core, float_type.([b.e_init for b in case.batteries]))

    br_ac = [_ac_branch_coeffs(br, float_type) for br in nd.branches]

    # ── Objective ─────────────────────────────────────────────────────────────
    # Generator cost. PGLib polynomial costs are USD/HOUR at the stage dispatch,
    # so the physical cost of a Δt-hour stage is Δt·(c2·pg² + c1·pg + c0). We bake
    # Δt into every coefficient (including the constant c0) so the reported
    # objective is dollars over the whole horizon.
    gen_items = [(t = t, g = gp,
                  c2 = float_type(g.cost2 * dt),
                  c1 = float_type(g.cost1 * dt),
                  c0 = float_type(g.cost0 * dt))
                 for t in 1:T for (gp, g) in enumerate(nd.gens)]
    core, _ = ExaModels.add_obj(core,
        it.c2 * pg[_gidx(nGen, it.t, it.g)]^2 + it.c1 * pg[_gidx(nGen, it.t, it.g)] + it.c0
        for it in gen_items)

    # Battery cycle cost, PER BATTERY: cᵦ = cycle_cost_per_mwh_b · baseMVA (pu→MW)
    # · Δt (h), so cᵦ·(pᶜʰ + pᵈⁱˢ) is the $ degradation cost of that battery's
    # throughput energy over the stage. Coefficients are computed per battery — the
    # fleet need not share one cycle price.
    if nBat > 0
        cyc_items = [(idx = _kidx(nBat, t, k),
                      c = float_type(bat.cycle_cost_per_mwh * nd.baseMVA * dt))
                     for t in 1:T for (k, bat) in enumerate(case.batteries)]
        if any(it -> it.c > 0, cyc_items)
            core, _ = ExaModels.add_obj(core,
                it.c * (p_ch[it.idx] + p_dis[it.idx]) for it in cyc_items)
        end
    end

    # ── Constraints ───────────────────────────────────────────────────────────
    # 1. Reference angle: va[t, ref] = 0
    ref_items = [(t = t, ref = ref) for t in 1:T for ref in nd.ref_bus_positions]
    core, _ = ExaModels.add_con(core, va[_bidx(nBus, it.t, it.ref)] for it in ref_items)

    # 2. From-end active flow
    pfr_items = [(t = t, f = br.f_pos, tb = br.t_pos, br = bp,
                  c3 = br_ac[bp].c3, c4 = br_ac[bp].c4, c5 = br_ac[bp].c5)
                 for t in 1:T for (bp, br) in enumerate(nd.branches)]
    core, _ = ExaModels.add_con(core,
        p_fr[_bridx(nBranch, it.t, it.br)]
        - it.c5 * vm[_bidx(nBus, it.t, it.f)]^2
        - it.c3 * vm[_bidx(nBus, it.t, it.f)] * vm[_bidx(nBus, it.t, it.tb)]
          * cos(va[_bidx(nBus, it.t, it.f)] - va[_bidx(nBus, it.t, it.tb)])
        - it.c4 * vm[_bidx(nBus, it.t, it.f)] * vm[_bidx(nBus, it.t, it.tb)]
          * sin(va[_bidx(nBus, it.t, it.f)] - va[_bidx(nBus, it.t, it.tb)])
        for it in pfr_items)

    # 3. From-end reactive flow
    qfr_items = [(t = t, f = br.f_pos, tb = br.t_pos, br = bp,
                  c3 = br_ac[bp].c3, c4 = br_ac[bp].c4, c6 = br_ac[bp].c6)
                 for t in 1:T for (bp, br) in enumerate(nd.branches)]
    core, _ = ExaModels.add_con(core,
        q_fr[_bridx(nBranch, it.t, it.br)]
        + it.c6 * vm[_bidx(nBus, it.t, it.f)]^2
        + it.c4 * vm[_bidx(nBus, it.t, it.f)] * vm[_bidx(nBus, it.t, it.tb)]
          * cos(va[_bidx(nBus, it.t, it.f)] - va[_bidx(nBus, it.t, it.tb)])
        - it.c3 * vm[_bidx(nBus, it.t, it.f)] * vm[_bidx(nBus, it.t, it.tb)]
          * sin(va[_bidx(nBus, it.t, it.f)] - va[_bidx(nBus, it.t, it.tb)])
        for it in qfr_items)

    # 4. To-end active flow
    pto_items = [(t = t, f = br.f_pos, tb = br.t_pos, br = bp,
                  c1 = br_ac[bp].c1, c2 = br_ac[bp].c2, c7 = br_ac[bp].c7)
                 for t in 1:T for (bp, br) in enumerate(nd.branches)]
    core, _ = ExaModels.add_con(core,
        p_to[_bridx(nBranch, it.t, it.br)]
        - it.c7 * vm[_bidx(nBus, it.t, it.tb)]^2
        - it.c1 * vm[_bidx(nBus, it.t, it.tb)] * vm[_bidx(nBus, it.t, it.f)]
          * cos(va[_bidx(nBus, it.t, it.tb)] - va[_bidx(nBus, it.t, it.f)])
        - it.c2 * vm[_bidx(nBus, it.t, it.tb)] * vm[_bidx(nBus, it.t, it.f)]
          * sin(va[_bidx(nBus, it.t, it.tb)] - va[_bidx(nBus, it.t, it.f)])
        for it in pto_items)

    # 5. To-end reactive flow
    qto_items = [(t = t, f = br.f_pos, tb = br.t_pos, br = bp,
                  c1 = br_ac[bp].c1, c2 = br_ac[bp].c2, c8 = br_ac[bp].c8)
                 for t in 1:T for (bp, br) in enumerate(nd.branches)]
    core, _ = ExaModels.add_con(core,
        q_to[_bridx(nBranch, it.t, it.br)]
        + it.c8 * vm[_bidx(nBus, it.t, it.tb)]^2
        + it.c2 * vm[_bidx(nBus, it.t, it.tb)] * vm[_bidx(nBus, it.t, it.f)]
          * cos(va[_bidx(nBus, it.t, it.tb)] - va[_bidx(nBus, it.t, it.f)])
        - it.c1 * vm[_bidx(nBus, it.t, it.tb)] * vm[_bidx(nBus, it.t, it.f)]
          * sin(va[_bidx(nBus, it.t, it.tb)] - va[_bidx(nBus, it.t, it.f)])
        for it in qto_items)

    # 6. Angle-difference limits
    ang_lb = float_type.(repeat([br.angmin for br in nd.branches], T))
    ang_ub = float_type.(repeat([br.angmax for br in nd.branches], T))
    ang_items = [(t = t, f = br.f_pos, tb = br.t_pos) for t in 1:T for br in nd.branches]
    core, _ = ExaModels.add_con(core,
        va[_bidx(nBus, it.t, it.f)] - va[_bidx(nBus, it.t, it.tb)]
        for it in ang_items; lcon = ang_lb, ucon = ang_ub)

    # 7. Apparent-power thermal limits at both ends (p² + q² ≤ rate²), applied
    # only to rate-limited branches (unlimited branches, rate = Inf, add no row —
    # matching PowerModels' ACPPowerModel).
    rated = [bp for bp in 1:nBranch if isfinite(nd.branches[bp].rate_a)]
    if !isempty(rated)
        th_items = [(t = t, br = bp) for t in 1:T for bp in rated]
        th_ub = float_type.([nd.branches[it.br].rate_a^2 for it in th_items])
        th_lb = fill(float_type(-Inf), length(th_items))
        core, _ = ExaModels.add_con(core,
            p_fr[_bridx(nBranch, it.t, it.br)]^2 + q_fr[_bridx(nBranch, it.t, it.br)]^2
            for it in th_items; lcon = th_lb, ucon = th_ub)
        core, _ = ExaModels.add_con(core,
            p_to[_bridx(nBranch, it.t, it.br)]^2 + q_to[_bridx(nBranch, it.t, it.br)]^2
            for it in th_items; lcon = th_lb, ucon = th_ub)
    end

    # Precomputed augmentation items. NOTE: ExaModels' `add_con!` augmentation
    # requires a `Base.Generator` (a SINGLE `for` over a precomputed vector); a
    # double-`for` comprehension yields a `Base.Iterators.Flatten`, which the
    # augmentation method does not accept. Hence every `=> ` term below iterates
    # one flattened item vector.
    gen_bus_items = [(brow = _bidx(nBus, t, g.bus_pos), gcol = _gidx(nGen, t, gp))
                     for t in 1:T for (gp, g) in enumerate(nd.gens)]
    fr_bus_items = [(brow = _bidx(nBus, t, br.f_pos), bcol = _bridx(nBranch, t, bp))
                    for t in 1:T for (bp, br) in enumerate(nd.branches)]
    to_bus_items = [(brow = _bidx(nBus, t, br.t_pos), bcol = _bridx(nBranch, t, bp))
                    for t in 1:T for (bp, br) in enumerate(nd.branches)]
    bat_bus_items = [(brow = _bidx(nBus, t, bat.bus_pos), kcol = _kidx(nBat, t, k))
                     for t in 1:T for (k, bat) in enumerate(case.batteries)]

    # 8. Active balance: pd + gs·vm² − Σpg − Σ(p_dis − p_ch) + Σp_fr + Σp_to = 0
    kcl_p_init = [(t = t, b = b, gs = float_type(nd.buses[b].gs)) for t in 1:T for b in 1:nBus]
    core, c_kcl_p = ExaModels.add_con(core,
        p_pd[_bidx(nBus, it.t, it.b)] + it.gs * vm[_bidx(nBus, it.t, it.b)]^2
        for it in kcl_p_init)
    core, _ = ExaModels.add_con!(core, c_kcl_p, it.brow => -pg[it.gcol] for it in gen_bus_items)
    core, _ = ExaModels.add_con!(core, c_kcl_p, it.brow => p_fr[it.bcol] for it in fr_bus_items)
    core, _ = ExaModels.add_con!(core, c_kcl_p, it.brow => p_to[it.bcol] for it in to_bus_items)
    if nBat > 0
        # Battery active injection p_bat = p_dis − p_ch enters as −p_dis + p_ch.
        core, _ = ExaModels.add_con!(core, c_kcl_p, it.brow => -p_dis[it.kcol] for it in bat_bus_items)
        core, _ = ExaModels.add_con!(core, c_kcl_p, it.brow => p_ch[it.kcol]  for it in bat_bus_items)
    end

    # 9. Reactive balance (hard): qd − bs·vm² − Σqg + Σq_fr + Σq_to = 0
    kcl_q_init = [(t = t, b = b, bs = float_type(nd.buses[b].bs)) for t in 1:T for b in 1:nBus]
    core, c_kcl_q = ExaModels.add_con(core,
        p_qd[_bidx(nBus, it.t, it.b)] - it.bs * vm[_bidx(nBus, it.t, it.b)]^2
        for it in kcl_q_init)
    core, _ = ExaModels.add_con!(core, c_kcl_q, it.brow => -qg[it.gcol] for it in gen_bus_items)
    core, _ = ExaModels.add_con!(core, c_kcl_q, it.brow => q_fr[it.bcol] for it in fr_bus_items)
    core, _ = ExaModels.add_con!(core, c_kcl_q, it.brow => q_to[it.bcol] for it in to_bus_items)

    # 10. Battery initial SoC: e[1,k] − e_init,k = 0
    if nBat > 0
        core, _ = ExaModels.add_con(core,
            e[_eidx(nBat, 1, k)] - p_e0[k] for k in 1:nBat)
    end

    # 11. Battery state equation (last constraint block):
    #     e[t+1,k] − (1−σΔt)·e[t,k] − η_ch·Δt·p_ch + (Δt/η_dis)·p_dis = 0
    if nBat > 0
        st_items = [(en = _eidx(nBat, t + 1, k), ec = _eidx(nBat, t, k),
                     pc = _kidx(nBat, t, k), pd_ = _kidx(nBat, t, k),
                     a = float_type(1 - bat.sigma * dt),
                     bch = float_type(bat.eta_ch * dt),
                     bdis = float_type(dt / bat.eta_dis))
                    for t in 1:T for (k, bat) in enumerate(case.batteries)]
        core, _ = ExaModels.add_con(core,
            e[it.en] - it.a * e[it.ec] - it.bch * p_ch[it.pc] + it.bdis * p_dis[it.pd_]
            for it in st_items)
    end

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
