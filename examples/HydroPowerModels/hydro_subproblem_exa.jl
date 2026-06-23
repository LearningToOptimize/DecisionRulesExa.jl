# hydro_subproblem_exa.jl
#
# Builds a SINGLE-STAGE hydro subproblem as a SubproblemDEProblem for
# stage-wise TS-DDR training via MadDiff implicit differentiation.
#
# Reuses the same OPF math as hydro_power_exa.jl but for T=1 with parametric
# incoming state (p_x_in) instead of an initial-condition equality.
#
# Parameter ordering in ExaModels (determines grad_p slicing in the rrule):
#   1. p_demand   (nBus)     — not differentiated
#   2. p_x_in     (nHyd)     — ∂q/∂x_{t-1}
#   3. p_inflow   (nHyd)     — = p_w
#   4. p_xhat     (nHyd)     — ∂q/∂x̂_t
#   5. p_penalty_half (nHyd) — not differentiated
#   6. p_penalty_l1   (nHyd) — not differentiated

using DecisionRulesExa
using ExaModels
using MadNLP

struct HydroSubproblemBundle
    prob::SubproblemDEProblem
    p_demand
    p_penalty_half
    p_penalty_l1
    base_penalty_half::Float64
    base_penalty_l1::Float64
    nBus::Int
    nGen::Int
    nBranch::Int
    nHyd::Int
    formulation::Symbol
end

function build_hydro_single_stage(
    power_data::PowerData,
    hydro_data::HydroData;
    backend = nothing,
    float_type::Type{<:AbstractFloat} = Float64,
    formulation::Symbol = :dc,
    target_penalty::Union{Real,Symbol} = :auto,
    target_penalty_l1::Union{Real,Symbol,Nothing} = :auto,
    deficit_cost::Union{Nothing,Real} = nothing,
    load_scaler::Real = 1.0,
    demand_vector = nothing,
)
    formulation === :dc || error("Only :dc supported for single-stage subproblem (for now)")

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

    # ── Variables (single stage) ─────────────────────────────────────────────
    va       = ExaModels.variable(core, nBus)
    pg_lb    = float_type.([g.pmin for g in power_data.gens])
    pg_ub    = float_type.([g.pmax for g in power_data.gens])
    pg       = ExaModels.variable(core, nGen; lvar = pg_lb, uvar = pg_ub)
    pf_lb    = float_type.([-b.rate_a for b in power_data.branches])
    pf_ub    = float_type.([ b.rate_a for b in power_data.branches])
    pf       = ExaModels.variable(core, nBranch; lvar = pf_lb, uvar = pf_ub)
    deficit  = ExaModels.variable(core, nBus; lvar = float_type(0))
    res_lb   = float_type.([h.min_vol for h in hydro_data.units])
    res_ub   = float_type.([h.max_vol for h in hydro_data.units])
    reservoir = ExaModels.variable(core, nHyd; lvar = res_lb, uvar = res_ub)
    out_lb   = float_type.([h.min_turn for h in hydro_data.units])
    out_ub   = float_type.([h.max_turn for h in hydro_data.units])
    outflow  = ExaModels.variable(core, nHyd; lvar = out_lb, uvar = out_ub)
    spill    = ExaModels.variable(core, nHyd; lvar = float_type(0))
    delta_pos = ExaModels.variable(core, nHyd; lvar = float_type(0))
    delta_neg = ExaModels.variable(core, nHyd; lvar = float_type(0))

    # Track state variable range (reservoir indices in solution vector)
    state_var_start = nBus + nGen + nBranch + nBus + 1
    state_var_range = state_var_start:(state_var_start + nHyd - 1)

    # ── Parameters (ORDER MATTERS for grad_p indexing) ───────────────────────
    # 1. demand (nBus)
    init_demand = if demand_vector !== nothing
        float_type.(load_scaler .* demand_vector)
    else
        float_type.(load_scaler .* power_data.default_bus_demand)
    end
    p_demand = ExaModels.parameter(core, init_demand)
    param_offset = nBus

    # 2. p_x_in (nHyd) — incoming state
    p_x_in = ExaModels.parameter(core, zeros(float_type, nHyd))
    p_x_in_range = (param_offset + 1):(param_offset + nHyd)
    param_offset += nHyd

    # 3. p_inflow = p_w (nHyd) — uncertainty
    p_inflow = ExaModels.parameter(core, zeros(float_type, nHyd))
    param_offset += nHyd

    # 4. p_xhat (nHyd) — target
    p_xhat = ExaModels.parameter(core, zeros(float_type, nHyd))
    p_xhat_range = (param_offset + 1):(param_offset + nHyd)
    param_offset += nHyd

    # 5. p_penalty_half, p_penalty_l1
    p_penalty_half = ExaModels.parameter(core, fill(float_type(ρ / 2), nHyd))
    p_penalty_l1   = ExaModels.parameter(core, fill(ρ_l1, nHyd))

    # ── Objective ────────────────────────────────────────────────────────────
    gen_items = [(g = g_pos, c1 = float_type(g.cost1), c2 = float_type(g.cost2))
                 for (g_pos, g) in enumerate(power_data.gens)]
    ExaModels.objective(core, item.c2 * pg[item.g]^2 + item.c1 * pg[item.g] for item in gen_items)

    def_items = [(b = b, c = cd) for b in 1:nBus]
    ExaModels.objective(core, item.c * deficit[item.b] for item in def_items)

    delta_items = [(r = r,) for r in 1:nHyd]
    ExaModels.objective(core, p_penalty_half[item.r] * (delta_pos[item.r] - delta_neg[item.r])^2 for item in delta_items)
    if use_l1
        ExaModels.objective(core, p_penalty_l1[item.r] * (delta_pos[item.r] + delta_neg[item.r]) for item in delta_items)
    end

    # Hydro operational costs (spill, min outflow violation, min volume violation)
    spill_items = [(r = r, c = float_type(h.spill_cost))
                   for (r, h) in enumerate(hydro_data.units) if h.spill_cost > 0]
    if !isempty(spill_items)
        ExaModels.objective(core, item.c * spill[item.r] for item in spill_items)
    end

    # ── Constraints ──────────────────────────────────────────────────────────
    n_con = 0

    # 1. Reference angle
    nRef = length(power_data.ref_buses)
    ref_items = [(ref = ref,) for ref in power_data.ref_buses]
    ExaModels.constraint(core, va[item.ref] for item in ref_items)
    n_con += nRef

    # 2. Ohm's law
    ohm_items = [(f = br.f_bus, tb = br.t_bus, br_pos = br_pos, b = float_type(br.b))
                 for (br_pos, br) in enumerate(power_data.branches)]
    ExaModels.constraint(core,
        item.b * (va[item.f] - va[item.tb]) - pf[item.br_pos]
        for item in ohm_items)
    n_con += nBranch

    # 3. Phase angle limits
    ang_lb = float_type.([br.angmin for br in power_data.branches])
    ang_ub = float_type.([br.angmax for br in power_data.branches])
    ang_items = [(f = br.f_bus, tb = br.t_bus) for br in power_data.branches]
    ExaModels.constraint(core, va[item.f] - va[item.tb] for item in ang_items;
        lcon = ang_lb, ucon = ang_ub)
    n_con += nBranch

    # 4. KCL power balance
    kcl_items = [(b = b,) for b in 1:nBus]
    c_kcl = ExaModels.constraint(core, p_demand[item.b] for item in kcl_items)

    kcl_gen = [(brow = g.bus, gcol = g_pos)
               for (g_pos, g) in enumerate(power_data.gens)]
    ExaModels.constraint!(core, c_kcl, item.brow => -pg[item.gcol] for item in kcl_gen)

    kcl_fr = [(brow = br.f_bus, bcol = br_pos)
              for (br_pos, br) in enumerate(power_data.branches)]
    ExaModels.constraint!(core, c_kcl, item.brow => pf[item.bcol] for item in kcl_fr)

    kcl_to = [(brow = br.t_bus, bcol = br_pos)
              for (br_pos, br) in enumerate(power_data.branches)]
    ExaModels.constraint!(core, c_kcl, item.brow => -pf[item.bcol] for item in kcl_to)

    kcl_def = [(brow = b, dcol = b) for b in 1:nBus]
    ExaModels.constraint!(core, c_kcl, item.brow => -deficit[item.dcol] for item in kcl_def)
    n_con += nBus

    # 5. Water balance (dynamics): reservoir[r] - p_x_in[r] + K*outflow[r] + spill[r] - K*p_inflow[r] = 0
    wb_items = [(r = r, K = K) for r in 1:nHyd]
    c_wb = ExaModels.constraint(core,
        reservoir[item.r] - p_x_in[item.r] + item.K * outflow[item.r] + spill[item.r] - item.K * p_inflow[item.r]
        for item in wb_items)
    if !isempty(hydro_data.upstream_turns)
        wb_turn = [(row = conn.downstream_pos, col = conn.upstream_pos, K = K)
                   for conn in hydro_data.upstream_turns]
        ExaModels.constraint!(core, c_wb,
            item.row => -item.K * outflow[item.col] for item in wb_turn)
    end
    if !isempty(hydro_data.upstream_spills)
        wb_spill = [(row = conn.downstream_pos, col = conn.upstream_pos)
                    for conn in hydro_data.upstream_spills]
        ExaModels.constraint!(core, c_wb,
            item.row => -spill[item.col] for item in wb_spill)
    end
    dynamics_con_range = (n_con + 1):(n_con + nHyd)
    n_con += nHyd

    # 6. Turbine coupling: baseMVA * pg[gen] - pf * outflow[hyd] = 0
    tc_items = [(gen_col = h.gen_pos, hyd_col = h.pos,
                 baseMVA = baseMVA, pf_h = float_type(h.pf))
                for h in hydro_data.units]
    ExaModels.constraint(core,
        item.baseMVA * pg[item.gen_col] - item.pf_h * outflow[item.hyd_col]
        for item in tc_items)
    n_con += nHyd

    # 7. TARGET CONSTRAINTS (LAST): p_xhat[r] - reservoir[r] - delta_pos[r] + delta_neg[r] = 0
    target_items = [(r = r,) for r in 1:nHyd]
    ExaModels.constraint(core,
        p_xhat[item.r] - reservoir[item.r] - delta_pos[item.r] + delta_neg[item.r]
        for item in target_items)
    target_con_range = (n_con + 1):(n_con + nHyd)

    model = ExaModels.ExaModel(core)

    prob = SubproblemDEProblem(
        core, model,
        p_x_in, p_inflow, p_xhat,
        nHyd, nGen + nHyd + nHyd + nBus, nHyd,  # nx, nu (all non-state vars), nw
        target_con_range, dynamics_con_range, state_var_range,
        p_x_in_range, p_xhat_range,
    )

    return HydroSubproblemBundle(
        prob, p_demand, p_penalty_half, p_penalty_l1,
        Float64(ρ / 2), Float64(ρ_l1),
        nBus, nGen, nBranch, nHyd, :dc,
    )
end

function set_stage_demand!(bundle::HydroSubproblemBundle, demand_vec::AbstractVector)
    ExaModels.set_parameter!(bundle.prob.core, bundle.p_demand, demand_vec)
end

function set_penalty_mult!(bundle::HydroSubproblemBundle, mult::Real)
    nH = bundle.nHyd
    ρ_half = bundle.base_penalty_half * mult
    ρ_l1 = bundle.base_penalty_l1 * mult
    ExaModels.set_parameter!(bundle.prob.core, bundle.p_penalty_half, fill(ρ_half, nH))
    ExaModels.set_parameter!(bundle.prob.core, bundle.p_penalty_l1, fill(ρ_l1, nH))
end
