# hydro_power_exa.jl
#
# Builds a multi-stage deterministic equivalent for the Hydro power system
# directly in ExaModels — no JuMP, no MOI.Parameters.
#
# Supports two OPF formulations (same AC polar math as ExaModelsPower.jl):
#   :dc       — DC linearization (default; fast, single branch flow variable)
#   :ac_polar — Full AC polar-coordinate OPF (vm, va, pg, qg, p_fr/q_fr/p_to/q_to)
#
# Multi-stage structure (T stages):
#   min   Σ_t [ Σ_g gen_cost(g, pg[t,g])
#               + cost_deficit · Σ_b deficit[t,b]
#               + (ρ/2) · Σ_r δ[t,r]² ]
#   s.t.  (DC or AC OPF constraints)
#         reservoir dynamics (initial condition, water balance, turbine coupling)
#         p_target[t,r] − reservoir[t+1,r] − δ[t,r] = 0  ← ADDED LAST
#
# Target constraints are added last so result.multipliers[target_con_range]
# gives ∇_{x̂} Q directly (envelope theorem for policy gradient).

using ExaModels
using MadNLP
using LinearAlgebra

# ── Index helpers ─────────────────────────────────────────────────────────────
# All arrays are flat, stage-major: index (t, i) → (t-1)*n + i

@inline _bi(nB, t, b)    = (t-1)*nB + b   # bus / deficit
@inline _gi(nG, t, g)    = (t-1)*nG + g   # generator
@inline _bri(nBR, t, br) = (t-1)*nBR + br # branch / pf
@inline _ri(nH, t, r)    = (t-1)*nH + r   # hydro: reservoir / outflow / spill / delta

# ── Problem struct ────────────────────────────────────────────────────────────

"""
    HydroExaDEProblem

Holds the ExaModels deterministic equivalent for the hydro power system.

Parameters:
- `p_demand`           : per-bus per-stage active demand   (length T*nBus, stage-major)
- `p_reactive_demand`  : per-bus per-stage reactive demand (length T*nBus; nothing for DC)
- `p_x0`               : initial reservoir levels          (length nHyd)
- `p_inflow`           : inflow trajectory                 (length T*nHyd, stage-major)
- `p_target`           : NN-predicted target levels        (length T*nHyd, stage-major)
"""
struct HydroExaDEProblem
    core
    model
    # parameters (ExaModels parameter objects)
    p_demand
    p_reactive_demand   # nothing for :dc formulation
    p_x0
    p_inflow
    p_target
    p_penalty_half      # ExaModels parameter for (ρ/2)*mult (length T*nHyd)
    base_penalty_half::Float64  # ρ/2 at multiplier=1
    p_penalty_l1        # ExaModels parameter for L1 penalty (length T*nHyd)
    base_penalty_l1::Float64    # L1 coefficient at multiplier=1
    # sizes
    nHyd::Int
    nBus::Int
    nGen::Int
    nBranch::Int
    horizon::Int
    # formulation
    formulation::Symbol   # :dc or :ac_polar
    # range into result.multipliers for target constraints
    target_con_range::UnitRange{Int}
end

# ── AC branch coefficient helper ──────────────────────────────────────────────

"""
    _ac_branch_coeffs(br, T) -> NamedTuple

Compute ExaModelsPower-compatible branch coefficients c1..c8 (in type T).

Formulas match ExaPowerIO.BranchData (same as PowerModels.jl AC polar convention):
  tr = tap·cos(shift),  ti = tap·sin(shift),  ttm = tr² + ti²
  g = br_r / (br_r² + br_x²),  b = −br_x / (br_r² + br_x²)
  c1 = (−g·tr − b·ti) / ttm     (to-end active, vm_f*vm_t cross term)
  c2 = (−b·tr + g·ti) / ttm     (to-end reactive cross term)
  c3 = (−g·tr + b·ti) / ttm     (from-end active cross term)
  c4 = (−b·tr − g·ti) / ttm     (from-end reactive cross term)
  c5 = (g + g_fr) / ttm          (from-end active self term)
  c6 = (b + b_fr) / ttm          (from-end reactive self term)
  c7 = g + g_to                  (to-end active self term)
  c8 = b + b_to                  (to-end reactive self term)

Power flow constraints (= 0 form):
  p_fr − c5·vm_f² − c3·vm_f·vm_t·cos(θ_f−θ_t) − c4·vm_f·vm_t·sin(θ_f−θ_t) = 0
  q_fr + c6·vm_f² + c4·vm_f·vm_t·cos(θ_f−θ_t) − c3·vm_f·vm_t·sin(θ_f−θ_t) = 0
  p_to − c7·vm_t² − c1·vm_t·vm_f·cos(θ_t−θ_f) − c2·vm_t·vm_f·sin(θ_t−θ_f) = 0
  q_to + c8·vm_t² + c2·vm_t·vm_f·cos(θ_t−θ_f) − c1·vm_t·vm_f·sin(θ_t−θ_f) = 0
"""
function _ac_branch_coeffs(br::PowerBranchData, ::Type{T}) where {T}
    r2x2 = br.br_r^2 + br.br_x^2
    g    = r2x2 > 0 ? T(br.br_r / r2x2) : zero(T)
    b    = r2x2 > 0 ? T(-br.br_x / r2x2) : zero(T)
    tap  = T(br.tap)
    sh   = T(br.shift)
    tr   = tap * cos(sh)
    ti   = tap * sin(sh)
    ttm  = tr^2 + ti^2
    ttm  = ttm > 0 ? ttm : one(T)
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

# ── auto target penalty ───────────────────────────────────────────────────────

"""
    auto_target_penalty(power_data, hydro_data) -> Float64

Compute the automatic target-penalty coefficient ρ such that ExaModels'
`(ρ/2)·δ²` uses the same effective multiplier as JuMP's `penalty_l2 = :auto`.

JuMP's `:auto` takes the maximum absolute linear coefficient in the subproblem
objective, which includes generator costs AND hydro penalty costs (spill,
min-outflow violation, min-volume violation).  We mirror that here:

    ρ = 2 × max(max_gen_cost, max_spill_cost, max_min_out_cost, max_min_vol_cost)
"""
function auto_target_penalty(power_data::PowerData, hydro_data::HydroData)
    max_gen = maximum(max(g.cost1, g.cost2) for g in power_data.gens)
    max_hyd = maximum(
        max(h.spill_cost, h.min_out_cost, h.min_vol_cost)
        for h in hydro_data.units
    )
    return 2.0 * max(max_gen, max_hyd)
end

# ── Builder ───────────────────────────────────────────────────────────────────

"""
    build_hydro_de(power_data, hydro_data, T;
                   backend=nothing, float_type=Float64, formulation=:dc,
                   target_penalty=:auto, target_penalty_l1=:auto,
                   demand_matrix=nothing,
                   reactive_demand_matrix=nothing, deficit_cost=nothing)
              -> HydroExaDEProblem

Build the T-stage hydro-power deterministic equivalent.

`formulation` selects the OPF model:
- `:dc`       — DC linearization (default; fast)
- `:ac_polar` — AC polar-coordinate OPF (full nonlinear)

`demand_matrix` is an optional [T × nBus] matrix of initial active demands (pu).
If nothing, uses `power_data.default_bus_demand` for all stages.

`reactive_demand_matrix` is an optional [T × nBus] matrix for reactive demand
(AC only). If nothing, uses `power_data.default_bus_reactive_demand`.

`deficit_cost` overrides `power_data.cost_deficit` (the load-shedding penalty per pu).
Pass a large value (e.g. 1e5, >> max thermal cost) to effectively enforce hard KCL.

`target_penalty` sets the L2 coefficient ρ for the `(ρ/2)·δ²` target slack penalty.
Pass `:auto` (default) to use `2 × max_gen_cost`, matching JuMP's `penalty_l2 = :auto`.

`target_penalty_l1` sets the L1 coefficient for the `λ·|δ|` target slack penalty.
Pass `:auto` (default) to use the same value as L2 ρ.  Pass `nothing` to disable L1.
The L1 term is reformulated as `λ·(δ⁺ + δ⁻)` with `δ = δ⁺ − δ⁻`, `δ⁺,δ⁻ ≥ 0`.
"""
function build_hydro_de(power_data::PowerData,
                         hydro_data::HydroData,
                         T::Int;
                         backend    = nothing,
                         float_type::Type{<:AbstractFloat} = Float64,
                         formulation::Symbol = :dc,
                         target_penalty::Union{Real,Symbol} = :auto,
                         target_penalty_l1::Union{Real,Symbol,Nothing} = :auto,
                         demand_matrix = nothing,
                         reactive_demand_matrix = nothing,
                         deficit_cost::Union{Nothing,Real} = nothing,
                         load_scaler::Real = 1.0)

    formulation in (:dc, :ac_polar) ||
        error("formulation must be :dc or :ac_polar, got :$formulation")

    if formulation === :dc
        return _build_dc_hydro_de(power_data, hydro_data, T;
                                   backend=backend, float_type=float_type,
                                   target_penalty=target_penalty,
                                   target_penalty_l1=target_penalty_l1,
                                   demand_matrix=demand_matrix,
                                   deficit_cost=deficit_cost,
                                   load_scaler=load_scaler)
    else
        return _build_ac_hydro_de(power_data, hydro_data, T;
                                   backend=backend, float_type=float_type,
                                   target_penalty=target_penalty,
                                   target_penalty_l1=target_penalty_l1,
                                   demand_matrix=demand_matrix,
                                   reactive_demand_matrix=reactive_demand_matrix,
                                   deficit_cost=deficit_cost,
                                   load_scaler=load_scaler)
    end
end

# ── DC builder ────────────────────────────────────────────────────────────────

function _build_dc_hydro_de(power_data::PowerData,
                              hydro_data::HydroData,
                              T::Int;
                              backend    = nothing,
                              float_type::Type{<:AbstractFloat} = Float64,
                              target_penalty::Union{Real,Symbol} = :auto,
                              target_penalty_l1::Union{Real,Symbol,Nothing} = :auto,
                              demand_matrix = nothing,
                              deficit_cost::Union{Nothing,Real} = nothing,
                              load_scaler::Real = 1.0)

    nBus    = power_data.nBus
    nGen    = power_data.nGen
    nBranch = power_data.nBranch
    nHyd    = hydro_data.nHyd
    K       = float_type(hydro_data.K)
    ρ       = float_type(target_penalty === :auto ? auto_target_penalty(power_data, hydro_data) : target_penalty)
    ρ_l1    = if target_penalty_l1 === :auto
        ρ
    elseif target_penalty_l1 === nothing
        zero(float_type)
    else
        float_type(target_penalty_l1)
    end
    use_l1  = ρ_l1 > 0
    baseMVA = float_type(power_data.baseMVA)
    cd      = float_type(deficit_cost !== nothing ? deficit_cost : power_data.cost_deficit)

    core = ExaModels.ExaCore(float_type; backend = backend)

    # ── Variables ─────────────────────────────────────────────────────────────

    # Voltage angles: T*nBus  (free)
    va = ExaModels.variable(core, T * nBus)

    # Generator active power: T*nGen
    pg_lb = float_type.(repeat([g.pmin for g in power_data.gens], T))
    pg_ub = float_type.(repeat([g.pmax for g in power_data.gens], T))
    pg = ExaModels.variable(core, T * nGen; lvar = pg_lb, uvar = pg_ub)

    # Branch power flows (DC, one per undirected branch): T*nBranch
    pf_lb = float_type.(repeat([-b.rate_a for b in power_data.branches], T))
    pf_ub = float_type.(repeat([ b.rate_a for b in power_data.branches], T))
    pf = ExaModels.variable(core, T * nBranch; lvar = pf_lb, uvar = pf_ub)

    # Deficit (load shedding): T*nBus  (non-negative)
    deficit = ExaModels.variable(core, T * nBus; lvar = float_type(0))

    # Reservoir levels: (T+1)*nHyd
    res_lb = float_type.(repeat([h.min_vol  for h in hydro_data.units], T+1))
    res_ub = float_type.(repeat([h.max_vol  for h in hydro_data.units], T+1))
    reservoir = ExaModels.variable(core, (T+1) * nHyd; lvar = res_lb, uvar = res_ub)

    # Turbine outflow: T*nHyd
    out_lb = float_type.(repeat([h.min_turn for h in hydro_data.units], T))
    out_ub = float_type.(repeat([h.max_turn for h in hydro_data.units], T))
    outflow = ExaModels.variable(core, T * nHyd; lvar = out_lb, uvar = out_ub)

    # Spill: T*nHyd  (non-negative)
    spill = ExaModels.variable(core, T * nHyd; lvar = float_type(0))

    # Target slack: δ = δ⁺ − δ⁻ with δ⁺,δ⁻ ≥ 0 (L1+L2 Lagrangian penalty)
    delta_pos = ExaModels.variable(core, T * nHyd; lvar = float_type(0))
    delta_neg = ExaModels.variable(core, T * nHyd; lvar = float_type(0))

    # ── Parameters ────────────────────────────────────────────────────────────

    init_demand = if demand_matrix !== nothing
        float_type.(load_scaler .* [demand_matrix[t, b] for t in 1:T for b in 1:nBus])
    else
        float_type.(load_scaler .* repeat(power_data.default_bus_demand, T))
    end

    p_demand       = ExaModels.parameter(core, init_demand)
    p_x0           = ExaModels.parameter(core, zeros(float_type, nHyd))
    p_inflow       = ExaModels.parameter(core, zeros(float_type, T * nHyd))
    p_target       = ExaModels.parameter(core, zeros(float_type, T * nHyd))
    p_penalty_half = ExaModels.parameter(core, fill(float_type(ρ / 2), T * nHyd))
    p_penalty_l1   = ExaModels.parameter(core, fill(ρ_l1, T * nHyd))

    # ── Objective ─────────────────────────────────────────────────────────────

    gen_cost_items = [(t = t, g = g_pos,
                       c1 = float_type(g.cost1), c2 = float_type(g.cost2))
                      for t in 1:T for (g_pos, g) in enumerate(power_data.gens)]
    ExaModels.objective(core,
        item.c2 * pg[_gi(nGen, item.t, item.g)]^2
        + item.c1 * pg[_gi(nGen, item.t, item.g)]
        for item in gen_cost_items
    )

    def_cost_items = [(t = t, b = b, c = cd) for t in 1:T for b in 1:nBus]
    ExaModels.objective(core,
        item.c * deficit[_bi(nBus, item.t, item.b)]
        for item in def_cost_items
    )

    # L2 penalty: (ρ/2)·(δ⁺ − δ⁻)²
    delta_items = [(idx = _ri(nHyd, t, r),) for t in 1:T for r in 1:nHyd]
    ExaModels.objective(core,
        p_penalty_half[item.idx] * (delta_pos[item.idx] - delta_neg[item.idx])^2
        for item in delta_items
    )

    # L1 penalty: λ·(δ⁺ + δ⁻)
    if use_l1
        ExaModels.objective(core,
            p_penalty_l1[item.idx] * (delta_pos[item.idx] + delta_neg[item.idx])
            for item in delta_items
        )
    end

    # ── Constraints ───────────────────────────────────────────────────────────
    n_con = 0

    # 1. Reference angle: va[t, ref] = 0
    nRef = length(power_data.ref_buses)
    ref_items = [(t = t, ref = ref) for t in 1:T for ref in power_data.ref_buses]
    ExaModels.constraint(core,
        va[_bi(nBus, item.t, item.ref)]
        for item in ref_items
    )
    n_con += T * nRef

    # 2. Ohm's law: b·(va_f - va_t) - pf = 0
    ohm_items = [(t  = t,
                  f  = br.f_bus,
                  tb = br.t_bus,
                  br = br_pos,
                  b  = float_type(br.b))
                 for t in 1:T for (br_pos, br) in enumerate(power_data.branches)]
    ExaModels.constraint(core,
        item.b * (va[_bi(nBus, item.t, item.f)] - va[_bi(nBus, item.t, item.tb)])
        - pf[_bri(nBranch, item.t, item.br)]
        for item in ohm_items
    )
    n_con += T * nBranch

    # 3. Phase angle difference limits: angmin ≤ va_f - va_t ≤ angmax
    ang_lb = float_type.(repeat([br.angmin for br in power_data.branches], T))
    ang_ub = float_type.(repeat([br.angmax for br in power_data.branches], T))
    ang_items = [(t = t, f = br.f_bus, tb = br.t_bus)
                 for t in 1:T for br in power_data.branches]
    ExaModels.constraint(core,
        va[_bi(nBus, item.t, item.f)] - va[_bi(nBus, item.t, item.tb)]
        for item in ang_items;
        lcon = ang_lb, ucon = ang_ub,
    )
    n_con += T * nBranch

    # 4. KCL power balance (T * nBus equalities)
    kcl_init_items = [(t = t, b = b) for t in 1:T for b in 1:nBus]
    c_kcl = ExaModels.constraint(core,
        p_demand[_bi(nBus, item.t, item.b)]
        for item in kcl_init_items
    )
    kcl_gen_items = [(t = t, brow = _bi(nBus, t, g.bus), gcol = _gi(nGen, t, g_pos))
                     for t in 1:T for (g_pos, g) in enumerate(power_data.gens)]
    ExaModels.constraint!(core, c_kcl,
        item.brow => -pg[item.gcol]
        for item in kcl_gen_items
    )
    kcl_fr_items = [(t = t, brow = _bi(nBus, t, br.f_bus), bcol = _bri(nBranch, t, br_pos))
                    for t in 1:T for (br_pos, br) in enumerate(power_data.branches)]
    ExaModels.constraint!(core, c_kcl,
        item.brow => pf[item.bcol]
        for item in kcl_fr_items
    )
    kcl_to_items = [(t = t, brow = _bi(nBus, t, br.t_bus), bcol = _bri(nBranch, t, br_pos))
                    for t in 1:T for (br_pos, br) in enumerate(power_data.branches)]
    ExaModels.constraint!(core, c_kcl,
        item.brow => -pf[item.bcol]
        for item in kcl_to_items
    )
    kcl_def_items = [(t = t, brow = _bi(nBus, t, b), dcol = _bi(nBus, t, b))
                     for t in 1:T for b in 1:nBus]
    ExaModels.constraint!(core, c_kcl,
        item.brow => -deficit[item.dcol]
        for item in kcl_def_items
    )
    n_con += T * nBus

    # 5. Initial reservoir condition
    ic_items = [(r = r,) for r in 1:nHyd]
    ExaModels.constraint(core,
        reservoir[_ri(nHyd, 1, item.r)] - p_x0[item.r]
        for item in ic_items
    )
    n_con += nHyd

    # 6. Water balance
    wb_items = [(res_next  = _ri(nHyd, t+1, r),
                 res_curr  = _ri(nHyd, t,   r),
                 out_idx   = _ri(nHyd, t,   r),
                 spill_idx = _ri(nHyd, t,   r),
                 inflow_p  = _ri(nHyd, t,   r),
                 K = K)
                for t in 1:T for r in 1:nHyd]
    c_wb = ExaModels.constraint(core,
        reservoir[item.res_next] - reservoir[item.res_curr]
        + item.K * outflow[item.out_idx]
        + spill[item.spill_idx]
        - item.K * p_inflow[item.inflow_p]
        for item in wb_items
    )
    if !isempty(hydro_data.upstream_turns)
        wb_turn_items = [(row = _ri(nHyd, t, conn.downstream_pos),
                          col = _ri(nHyd, t, conn.upstream_pos), K = K)
                         for t in 1:T for conn in hydro_data.upstream_turns]
        ExaModels.constraint!(core, c_wb,
            item.row => -item.K * outflow[item.col]
            for item in wb_turn_items
        )
    end
    if !isempty(hydro_data.upstream_spills)
        wb_spill_items = [(row = _ri(nHyd, t, conn.downstream_pos),
                           col = _ri(nHyd, t, conn.upstream_pos))
                          for t in 1:T for conn in hydro_data.upstream_spills]
        ExaModels.constraint!(core, c_wb,
            item.row => -spill[item.col]
            for item in wb_spill_items
        )
    end
    n_con += T * nHyd

    # 7. Turbine coupling
    tc_items = [(gen_col = _gi(nGen,  t, h.gen_pos),
                 out_col = _ri(nHyd,  t, h.pos),
                 baseMVA = baseMVA,
                 pf_h    = float_type(h.pf))
                for t in 1:T for h in hydro_data.units]
    ExaModels.constraint(core,
        item.baseMVA * pg[item.gen_col] - item.pf_h * outflow[item.out_col]
        for item in tc_items
    )
    n_con += T * nHyd

    # ── TARGET CONSTRAINTS (ADDED LAST) ───────────────────────────────────────
    # x̂ − x − (δ⁺ − δ⁻) = 0
    target_items = [(param_idx = _ri(nHyd, t, r),
                     res_idx   = _ri(nHyd, t+1, r),
                     delta_idx = _ri(nHyd, t, r))
                    for t in 1:T for r in 1:nHyd]
    ExaModels.constraint(core,
        p_target[item.param_idx] - reservoir[item.res_idx] - delta_pos[item.delta_idx] + delta_neg[item.delta_idx]
        for item in target_items
    )
    target_con_range = (n_con + 1):(n_con + T * nHyd)

    model = ExaModels.ExaModel(core)

    return HydroExaDEProblem(
        core, model,
        p_demand, nothing, p_x0, p_inflow, p_target,
        p_penalty_half, Float64(ρ / 2),
        p_penalty_l1, Float64(ρ_l1),
        nHyd, nBus, nGen, nBranch, T,
        :dc, target_con_range,
    )
end

# ── AC polar builder ──────────────────────────────────────────────────────────

function _build_ac_hydro_de(power_data::PowerData,
                              hydro_data::HydroData,
                              T::Int;
                              backend    = nothing,
                              float_type::Type{<:AbstractFloat} = Float64,
                              target_penalty::Union{Real,Symbol} = :auto,
                              target_penalty_l1::Union{Real,Symbol,Nothing} = :auto,
                              demand_matrix = nothing,
                              reactive_demand_matrix = nothing,
                              deficit_cost::Union{Nothing,Real} = nothing,
                              load_scaler::Real = 1.0)

    nBus    = power_data.nBus
    nGen    = power_data.nGen
    nBranch = power_data.nBranch
    nHyd    = hydro_data.nHyd
    K       = float_type(hydro_data.K)
    ρ       = float_type(target_penalty === :auto ? auto_target_penalty(power_data, hydro_data) : target_penalty)
    ρ_l1    = if target_penalty_l1 === :auto
        ρ
    elseif target_penalty_l1 === nothing
        zero(float_type)
    else
        float_type(target_penalty_l1)
    end
    use_l1  = ρ_l1 > 0
    baseMVA = float_type(power_data.baseMVA)
    cd      = float_type(deficit_cost !== nothing ? deficit_cost : power_data.cost_deficit)

    core = ExaModels.ExaCore(float_type; backend = backend)

    # ── Variables ─────────────────────────────────────────────────────────────

    # Voltage angles: T*nBus  (free)
    va = ExaModels.variable(core, T * nBus)

    # Voltage magnitudes: T*nBus  (bounded; initialized to 1.0)
    vm_lb = float_type.(repeat([b.vmin for b in power_data.buses], T))
    vm_ub = float_type.(repeat([b.vmax for b in power_data.buses], T))
    vm = ExaModels.variable(core, T * nBus;
                             lvar = vm_lb, uvar = vm_ub,
                             start = ones(float_type, T * nBus))

    # Generator active power: T*nGen
    pg_lb = float_type.(repeat([g.pmin for g in power_data.gens], T))
    pg_ub = float_type.(repeat([g.pmax for g in power_data.gens], T))
    pg = ExaModels.variable(core, T * nGen; lvar = pg_lb, uvar = pg_ub)

    # Generator reactive power: T*nGen
    qg_lb = float_type.(repeat([isfinite(g.qmin) ? g.qmin : -1e4 for g in power_data.gens], T))
    qg_ub = float_type.(repeat([isfinite(g.qmax) ? g.qmax :  1e4 for g in power_data.gens], T))
    qg = ExaModels.variable(core, T * nGen; lvar = qg_lb, uvar = qg_ub)

    # Branch from-end active/reactive power flows: T*nBranch each
    p_fr_lb = float_type.(repeat([-b.rate_a for b in power_data.branches], T))
    p_fr_ub = float_type.(repeat([ b.rate_a for b in power_data.branches], T))
    p_fr = ExaModels.variable(core, T * nBranch; lvar = p_fr_lb, uvar = p_fr_ub)
    q_fr = ExaModels.variable(core, T * nBranch; lvar = p_fr_lb, uvar = p_fr_ub)

    # Branch to-end active/reactive power flows: T*nBranch each
    p_to = ExaModels.variable(core, T * nBranch; lvar = p_fr_lb, uvar = p_fr_ub)
    q_to = ExaModels.variable(core, T * nBranch; lvar = p_fr_lb, uvar = p_fr_ub)

    # Active deficit (load shedding): T*nBus  (non-negative)
    deficit = ExaModels.variable(core, T * nBus; lvar = float_type(0))

    # Reactive deficit (free; allows reactive balance at zero cost)
    deficit_q = ExaModels.variable(core, T * nBus)

    # Reservoir levels: (T+1)*nHyd
    res_lb = float_type.(repeat([h.min_vol  for h in hydro_data.units], T+1))
    res_ub = float_type.(repeat([h.max_vol  for h in hydro_data.units], T+1))
    reservoir = ExaModels.variable(core, (T+1) * nHyd; lvar = res_lb, uvar = res_ub)

    # Turbine outflow: T*nHyd
    out_lb = float_type.(repeat([h.min_turn for h in hydro_data.units], T))
    out_ub = float_type.(repeat([h.max_turn for h in hydro_data.units], T))
    outflow = ExaModels.variable(core, T * nHyd; lvar = out_lb, uvar = out_ub)

    # Spill: T*nHyd  (non-negative)
    spill = ExaModels.variable(core, T * nHyd; lvar = float_type(0))

    # Target slack: δ = δ⁺ − δ⁻ with δ⁺,δ⁻ ≥ 0 (L1+L2 Lagrangian penalty)
    delta_pos = ExaModels.variable(core, T * nHyd; lvar = float_type(0))
    delta_neg = ExaModels.variable(core, T * nHyd; lvar = float_type(0))

    # ── Parameters ────────────────────────────────────────────────────────────

    init_demand = if demand_matrix !== nothing
        float_type.(load_scaler .* [demand_matrix[t, b] for t in 1:T for b in 1:nBus])
    else
        float_type.(load_scaler .* repeat(power_data.default_bus_demand, T))
    end

    init_reactive_demand = if reactive_demand_matrix !== nothing
        float_type.(load_scaler .* [reactive_demand_matrix[t, b] for t in 1:T for b in 1:nBus])
    else
        float_type.(load_scaler .* repeat(power_data.default_bus_reactive_demand, T))
    end

    p_demand          = ExaModels.parameter(core, init_demand)
    p_reactive_demand = ExaModels.parameter(core, init_reactive_demand)
    p_x0              = ExaModels.parameter(core, zeros(float_type, nHyd))
    p_inflow          = ExaModels.parameter(core, zeros(float_type, T * nHyd))
    p_target          = ExaModels.parameter(core, zeros(float_type, T * nHyd))
    p_penalty_half    = ExaModels.parameter(core, fill(float_type(ρ / 2), T * nHyd))
    p_penalty_l1      = ExaModels.parameter(core, fill(ρ_l1, T * nHyd))

    # ── Precompute branch AC coefficients ─────────────────────────────────────

    br_ac = [_ac_branch_coeffs(br, float_type) for br in power_data.branches]

    # ── Objective ─────────────────────────────────────────────────────────────

    gen_cost_items = [(t = t, g = g_pos,
                       c1 = float_type(g.cost1), c2 = float_type(g.cost2))
                      for t in 1:T for (g_pos, g) in enumerate(power_data.gens)]
    ExaModels.objective(core,
        item.c2 * pg[_gi(nGen, item.t, item.g)]^2
        + item.c1 * pg[_gi(nGen, item.t, item.g)]
        for item in gen_cost_items
    )

    def_cost_items = [(t = t, b = b, c = cd) for t in 1:T for b in 1:nBus]
    ExaModels.objective(core,
        item.c * deficit[_bi(nBus, item.t, item.b)]
        for item in def_cost_items
    )

    # L2 penalty: (ρ/2)·(δ⁺ − δ⁻)²
    delta_items = [(idx = _ri(nHyd, t, r),) for t in 1:T for r in 1:nHyd]
    ExaModels.objective(core,
        p_penalty_half[item.idx] * (delta_pos[item.idx] - delta_neg[item.idx])^2
        for item in delta_items
    )

    # L1 penalty: λ·(δ⁺ + δ⁻)
    if use_l1
        ExaModels.objective(core,
            p_penalty_l1[item.idx] * (delta_pos[item.idx] + delta_neg[item.idx])
            for item in delta_items
        )
    end

    # ── Constraints ───────────────────────────────────────────────────────────
    n_con = 0

    # 1. Reference angle: va[t, ref] = 0
    nRef = length(power_data.ref_buses)
    ref_items = [(t = t, ref = ref) for t in 1:T for ref in power_data.ref_buses]
    ExaModels.constraint(core,
        va[_bi(nBus, item.t, item.ref)]
        for item in ref_items
    )
    n_con += T * nRef

    # 2. AC from-end active power flow: p_fr − c5·vm_f² − c3·vm_f·vm_t·cos(θ) − c4·vm_f·vm_t·sin(θ) = 0
    pfr_items = [(t = t,
                  f   = br.f_bus, tb = br.t_bus, br = br_pos,
                  c3  = br_ac[br_pos].c3, c4 = br_ac[br_pos].c4, c5 = br_ac[br_pos].c5)
                 for t in 1:T for (br_pos, br) in enumerate(power_data.branches)]
    ExaModels.constraint(core,
        p_fr[_bri(nBranch, item.t, item.br)]
        - item.c5 * vm[_bi(nBus, item.t, item.f)]^2
        - item.c3 * vm[_bi(nBus, item.t, item.f)] * vm[_bi(nBus, item.t, item.tb)]
          * cos(va[_bi(nBus, item.t, item.f)] - va[_bi(nBus, item.t, item.tb)])
        - item.c4 * vm[_bi(nBus, item.t, item.f)] * vm[_bi(nBus, item.t, item.tb)]
          * sin(va[_bi(nBus, item.t, item.f)] - va[_bi(nBus, item.t, item.tb)])
        for item in pfr_items
    )
    n_con += T * nBranch

    # 3. AC from-end reactive power flow: q_fr + c6·vm_f² + c4·vm_f·vm_t·cos(θ) − c3·vm_f·vm_t·sin(θ) = 0
    qfr_items = [(t = t,
                  f  = br.f_bus, tb = br.t_bus, br = br_pos,
                  c3 = br_ac[br_pos].c3, c4 = br_ac[br_pos].c4, c6 = br_ac[br_pos].c6)
                 for t in 1:T for (br_pos, br) in enumerate(power_data.branches)]
    ExaModels.constraint(core,
        q_fr[_bri(nBranch, item.t, item.br)]
        + item.c6 * vm[_bi(nBus, item.t, item.f)]^2
        + item.c4 * vm[_bi(nBus, item.t, item.f)] * vm[_bi(nBus, item.t, item.tb)]
          * cos(va[_bi(nBus, item.t, item.f)] - va[_bi(nBus, item.t, item.tb)])
        - item.c3 * vm[_bi(nBus, item.t, item.f)] * vm[_bi(nBus, item.t, item.tb)]
          * sin(va[_bi(nBus, item.t, item.f)] - va[_bi(nBus, item.t, item.tb)])
        for item in qfr_items
    )
    n_con += T * nBranch

    # 4. AC to-end active power flow: p_to − c7·vm_t² − c1·vm_t·vm_f·cos(θ_t−θ_f) − c2·vm_t·vm_f·sin(θ_t−θ_f) = 0
    pto_items = [(t = t,
                  f  = br.f_bus, tb = br.t_bus, br = br_pos,
                  c1 = br_ac[br_pos].c1, c2 = br_ac[br_pos].c2, c7 = br_ac[br_pos].c7)
                 for t in 1:T for (br_pos, br) in enumerate(power_data.branches)]
    ExaModels.constraint(core,
        p_to[_bri(nBranch, item.t, item.br)]
        - item.c7 * vm[_bi(nBus, item.t, item.tb)]^2
        - item.c1 * vm[_bi(nBus, item.t, item.tb)] * vm[_bi(nBus, item.t, item.f)]
          * cos(va[_bi(nBus, item.t, item.tb)] - va[_bi(nBus, item.t, item.f)])
        - item.c2 * vm[_bi(nBus, item.t, item.tb)] * vm[_bi(nBus, item.t, item.f)]
          * sin(va[_bi(nBus, item.t, item.tb)] - va[_bi(nBus, item.t, item.f)])
        for item in pto_items
    )
    n_con += T * nBranch

    # 5. AC to-end reactive power flow: q_to + c8·vm_t² + c2·vm_t·vm_f·cos(θ_t−θ_f) − c1·vm_t·vm_f·sin(θ_t−θ_f) = 0
    qto_items = [(t = t,
                  f  = br.f_bus, tb = br.t_bus, br = br_pos,
                  c1 = br_ac[br_pos].c1, c2 = br_ac[br_pos].c2, c8 = br_ac[br_pos].c8)
                 for t in 1:T for (br_pos, br) in enumerate(power_data.branches)]
    ExaModels.constraint(core,
        q_to[_bri(nBranch, item.t, item.br)]
        + item.c8 * vm[_bi(nBus, item.t, item.tb)]^2
        + item.c2 * vm[_bi(nBus, item.t, item.tb)] * vm[_bi(nBus, item.t, item.f)]
          * cos(va[_bi(nBus, item.t, item.tb)] - va[_bi(nBus, item.t, item.f)])
        - item.c1 * vm[_bi(nBus, item.t, item.tb)] * vm[_bi(nBus, item.t, item.f)]
          * sin(va[_bi(nBus, item.t, item.tb)] - va[_bi(nBus, item.t, item.f)])
        for item in qto_items
    )
    n_con += T * nBranch

    # 6. Phase angle difference limits: angmin ≤ va_f − va_t ≤ angmax
    ang_lb    = float_type.(repeat([br.angmin for br in power_data.branches], T))
    ang_ub    = float_type.(repeat([br.angmax for br in power_data.branches], T))
    ang_items = [(t = t, f = br.f_bus, tb = br.t_bus)
                 for t in 1:T for br in power_data.branches]
    ExaModels.constraint(core,
        va[_bi(nBus, item.t, item.f)] - va[_bi(nBus, item.t, item.tb)]
        for item in ang_items;
        lcon = ang_lb, ucon = ang_ub,
    )
    n_con += T * nBranch

    # 7. Active KCL: pd + gs·vm² − deficit − Σpg + Σp_fr + Σp_to = 0
    kcl_p_init = [(t = t, b = b, gs = float_type(power_data.buses[b].gs))
                  for t in 1:T for b in 1:nBus]
    c_kcl_p = ExaModels.constraint(core,
        p_demand[_bi(nBus, item.t, item.b)]
        + item.gs * vm[_bi(nBus, item.t, item.b)]^2
        for item in kcl_p_init
    )
    kcl_pg_items = [(t = t, brow = _bi(nBus, t, g.bus), gcol = _gi(nGen, t, g_pos))
                    for t in 1:T for (g_pos, g) in enumerate(power_data.gens)]
    ExaModels.constraint!(core, c_kcl_p,
        item.brow => -pg[item.gcol]
        for item in kcl_pg_items
    )
    kcl_pfr_items = [(t = t, brow = _bi(nBus, t, br.f_bus), bcol = _bri(nBranch, t, br_pos))
                     for t in 1:T for (br_pos, br) in enumerate(power_data.branches)]
    ExaModels.constraint!(core, c_kcl_p,
        item.brow => p_fr[item.bcol]
        for item in kcl_pfr_items
    )
    kcl_pto_items = [(t = t, brow = _bi(nBus, t, br.t_bus), bcol = _bri(nBranch, t, br_pos))
                     for t in 1:T for (br_pos, br) in enumerate(power_data.branches)]
    ExaModels.constraint!(core, c_kcl_p,
        item.brow => p_to[item.bcol]
        for item in kcl_pto_items
    )
    kcl_def_items = [(t = t, brow = _bi(nBus, t, b), dcol = _bi(nBus, t, b))
                     for t in 1:T for b in 1:nBus]
    ExaModels.constraint!(core, c_kcl_p,
        item.brow => -deficit[item.dcol]
        for item in kcl_def_items
    )
    n_con += T * nBus

    # 8. Reactive KCL: qd − bs·vm² − deficit_q − Σqg + Σq_fr + Σq_to = 0
    kcl_q_init = [(t = t, b = b, bs = float_type(power_data.buses[b].bs))
                  for t in 1:T for b in 1:nBus]
    c_kcl_q = ExaModels.constraint(core,
        p_reactive_demand[_bi(nBus, item.t, item.b)]
        - item.bs * vm[_bi(nBus, item.t, item.b)]^2
        for item in kcl_q_init
    )
    kcl_qg_items = [(t = t, brow = _bi(nBus, t, g.bus), gcol = _gi(nGen, t, g_pos))
                    for t in 1:T for (g_pos, g) in enumerate(power_data.gens)]
    ExaModels.constraint!(core, c_kcl_q,
        item.brow => -qg[item.gcol]
        for item in kcl_qg_items
    )
    kcl_qfr_items = [(t = t, brow = _bi(nBus, t, br.f_bus), bcol = _bri(nBranch, t, br_pos))
                     for t in 1:T for (br_pos, br) in enumerate(power_data.branches)]
    ExaModels.constraint!(core, c_kcl_q,
        item.brow => q_fr[item.bcol]
        for item in kcl_qfr_items
    )
    kcl_qto_items = [(t = t, brow = _bi(nBus, t, br.t_bus), bcol = _bri(nBranch, t, br_pos))
                     for t in 1:T for (br_pos, br) in enumerate(power_data.branches)]
    ExaModels.constraint!(core, c_kcl_q,
        item.brow => q_to[item.bcol]
        for item in kcl_qto_items
    )
    ExaModels.constraint!(core, c_kcl_q,
        item.brow => -deficit_q[item.dcol]
        for item in kcl_def_items
    )
    n_con += T * nBus

    # 9. Initial reservoir condition
    ic_items = [(r = r,) for r in 1:nHyd]
    ExaModels.constraint(core,
        reservoir[_ri(nHyd, 1, item.r)] - p_x0[item.r]
        for item in ic_items
    )
    n_con += nHyd

    # 10. Water balance
    wb_items = [(res_next  = _ri(nHyd, t+1, r),
                 res_curr  = _ri(nHyd, t,   r),
                 out_idx   = _ri(nHyd, t,   r),
                 spill_idx = _ri(nHyd, t,   r),
                 inflow_p  = _ri(nHyd, t,   r),
                 K = K)
                for t in 1:T for r in 1:nHyd]
    c_wb = ExaModels.constraint(core,
        reservoir[item.res_next] - reservoir[item.res_curr]
        + item.K * outflow[item.out_idx]
        + spill[item.spill_idx]
        - item.K * p_inflow[item.inflow_p]
        for item in wb_items
    )
    if !isempty(hydro_data.upstream_turns)
        wb_turn_items = [(row = _ri(nHyd, t, conn.downstream_pos),
                          col = _ri(nHyd, t, conn.upstream_pos), K = K)
                         for t in 1:T for conn in hydro_data.upstream_turns]
        ExaModels.constraint!(core, c_wb,
            item.row => -item.K * outflow[item.col]
            for item in wb_turn_items
        )
    end
    if !isempty(hydro_data.upstream_spills)
        wb_spill_items = [(row = _ri(nHyd, t, conn.downstream_pos),
                           col = _ri(nHyd, t, conn.upstream_pos))
                          for t in 1:T for conn in hydro_data.upstream_spills]
        ExaModels.constraint!(core, c_wb,
            item.row => -spill[item.col]
            for item in wb_spill_items
        )
    end
    n_con += T * nHyd

    # 11. Turbine coupling
    tc_items = [(gen_col = _gi(nGen,  t, h.gen_pos),
                 out_col = _ri(nHyd,  t, h.pos),
                 baseMVA = baseMVA,
                 pf_h    = float_type(h.pf))
                for t in 1:T for h in hydro_data.units]
    ExaModels.constraint(core,
        item.baseMVA * pg[item.gen_col] - item.pf_h * outflow[item.out_col]
        for item in tc_items
    )
    n_con += T * nHyd

    # ── TARGET CONSTRAINTS (ADDED LAST) ───────────────────────────────────────
    # x̂ − x − (δ⁺ − δ⁻) = 0
    target_items = [(param_idx = _ri(nHyd, t, r),
                     res_idx   = _ri(nHyd, t+1, r),
                     delta_idx = _ri(nHyd, t, r))
                    for t in 1:T for r in 1:nHyd]
    ExaModels.constraint(core,
        p_target[item.param_idx] - reservoir[item.res_idx] - delta_pos[item.delta_idx] + delta_neg[item.delta_idx]
        for item in target_items
    )
    target_con_range = (n_con + 1):(n_con + T * nHyd)

    model = ExaModels.ExaModel(core)

    return HydroExaDEProblem(
        core, model,
        p_demand, p_reactive_demand, p_x0, p_inflow, p_target,
        p_penalty_half, Float64(ρ / 2),
        p_penalty_l1, Float64(ρ_l1),
        nHyd, nBus, nGen, nBranch, T,
        :ac_polar, target_con_range,
    )
end

# ── Parameter updates ─────────────────────────────────────────────────────────

"""
    set_demand!(prob, demand_matrix)

Update the per-stage bus demands. `demand_matrix` is [T × nBus] (Float64 or Float32).
"""
function set_demand!(prob::HydroExaDEProblem, demand_matrix::AbstractMatrix)
    T, nB = size(demand_matrix)
    T == prob.horizon  || error("demand_matrix must have T=$(prob.horizon) rows")
    nB == prob.nBus    || error("demand_matrix must have nBus=$(prob.nBus) cols")
    flat = [demand_matrix[t, b] for t in 1:T for b in 1:nB]
    ExaModels.set_parameter!(prob.core, prob.p_demand, flat)
    return prob
end

"""
    set_inflows!(prob, w)

Set the inflow trajectory. `w` is a flat vector of length `T*nHyd` (stage-major).
"""
function set_inflows!(prob::HydroExaDEProblem, w::AbstractVector)
    expected = prob.horizon * prob.nHyd
    length(w) == expected || error("w must have length T*nHyd=$expected")
    ExaModels.set_parameter!(prob.core, prob.p_inflow, w)
    return prob
end

# ── Post-processing ───────────────────────────────────────────────────────────

"""
    hydro_solution(prob, result) -> NamedTuple

Reshape the flat solution vector into named components.
`delta` is reconstructed as `delta_pos - delta_neg`.

DC:  (va, pg, pf, deficit, reservoir, outflow, spill, delta)
AC:  (va, vm, pg, qg, p_fr, q_fr, p_to, q_to, deficit, deficit_q,
      reservoir, outflow, spill, delta)
"""
function hydro_solution(prob::HydroExaDEProblem, result)
    T   = prob.horizon
    nH  = prob.nHyd
    nG  = prob.nGen
    nBR = prob.nBranch
    nB  = prob.nBus
    sol = result.solution
    off = 0

    if prob.formulation === :dc
        va_sol    = reshape(sol[off .+ (1:T*nB)],        nB, T);  off += T*nB
        pg_sol    = reshape(sol[off .+ (1:T*nG)],        nG, T);  off += T*nG
        pf_sol    = reshape(sol[off .+ (1:T*nBR)],       nBR, T); off += T*nBR
        def_sol   = reshape(sol[off .+ (1:T*nB)],        nB, T);  off += T*nB
        res_sol   = reshape(sol[off .+ (1:(T+1)*nH)],    nH, T+1); off += (T+1)*nH
        out_sol   = reshape(sol[off .+ (1:T*nH)],        nH, T);  off += T*nH
        spill_sol = reshape(sol[off .+ (1:T*nH)],        nH, T);  off += T*nH
        dp_sol    = reshape(sol[off .+ (1:T*nH)],        nH, T);  off += T*nH
        dn_sol    = reshape(sol[off .+ (1:T*nH)],        nH, T);  off += T*nH
        delta_sol = dp_sol .- dn_sol
        return (va=va_sol, pg=pg_sol, pf=pf_sol, deficit=def_sol,
                reservoir=res_sol, outflow=out_sol, spill=spill_sol, delta=delta_sol)
    else  # :ac_polar
        va_sol     = reshape(sol[off .+ (1:T*nB)],        nB, T);  off += T*nB
        vm_sol     = reshape(sol[off .+ (1:T*nB)],        nB, T);  off += T*nB
        pg_sol     = reshape(sol[off .+ (1:T*nG)],        nG, T);  off += T*nG
        qg_sol     = reshape(sol[off .+ (1:T*nG)],        nG, T);  off += T*nG
        p_fr_sol   = reshape(sol[off .+ (1:T*nBR)],       nBR, T); off += T*nBR
        q_fr_sol   = reshape(sol[off .+ (1:T*nBR)],       nBR, T); off += T*nBR
        p_to_sol   = reshape(sol[off .+ (1:T*nBR)],       nBR, T); off += T*nBR
        q_to_sol   = reshape(sol[off .+ (1:T*nBR)],       nBR, T); off += T*nBR
        def_sol    = reshape(sol[off .+ (1:T*nB)],        nB, T);  off += T*nB
        def_q_sol  = reshape(sol[off .+ (1:T*nB)],        nB, T);  off += T*nB
        res_sol    = reshape(sol[off .+ (1:(T+1)*nH)],    nH, T+1); off += (T+1)*nH
        out_sol    = reshape(sol[off .+ (1:T*nH)],        nH, T);  off += T*nH
        spill_sol  = reshape(sol[off .+ (1:T*nH)],        nH, T);  off += T*nH
        dp_sol     = reshape(sol[off .+ (1:T*nH)],        nH, T);  off += T*nH
        dn_sol     = reshape(sol[off .+ (1:T*nH)],        nH, T);  off += T*nH
        delta_sol  = dp_sol .- dn_sol
        return (va=va_sol, vm=vm_sol, pg=pg_sol, qg=qg_sol,
                p_fr=p_fr_sol, q_fr=q_fr_sol, p_to=p_to_sol, q_to=q_to_sol,
                deficit=def_sol, deficit_q=def_q_sol,
                reservoir=res_sol, outflow=out_sol, spill=spill_sol, delta=delta_sol)
    end
end
