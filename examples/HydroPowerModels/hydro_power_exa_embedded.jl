# hydro_power_exa_embedded.jl
#
# Embedded-NN deterministic equivalent for the Hydro power system.
# Target constraints from hydro_power_exa.jl are replaced by a
# VectorNonlinearOracle that evaluates the Flux policy inline.
#
# Oracle constraint (matching regular DE sign convention):
#     π_θ(inflow_t, reservoir_t) − reservoir_{t+1,r} − δ⁺_{t,r} + δ⁻_{t,r} = 0
#
# Depends on: hydro_power_data.jl, hydro_power_exa.jl (index helpers, data types)

using Flux
using Zygote
import DecisionRulesExa: set_x0!, set_uncertainty!, set_targets!

# ── Problem struct ──────────────────────────────────────────────────────────────

struct EmbeddedHydroExaDEProblem{P, VT <: AbstractVector{Float64}}
    core
    model
    p_demand
    p_reactive_demand
    p_x0
    p_inflow
    p_penalty_half
    base_penalty_half::Float64
    p_penalty_l1
    base_penalty_l1::Float64
    policy::P
    nHyd::Int
    nx::Int
    nw::Int
    nBus::Int
    nGen::Int
    nBranch::Int
    horizon::Int
    formulation::Symbol
    target_con_range::UnitRange{Int}
    _res_start::Int
    _dp_start::Int
    _dn_start::Int
    _nvar::Int
    _inflow_buf::VT
    _x0_buf::VT
end

# ── Interface (duck-typing for train_tsddr_embedded) ────────────────────────────

function set_x0!(prob::EmbeddedHydroExaDEProblem, x0::AbstractVector)
    length(x0) == prob.nHyd || error("x0 length must be nHyd=$(prob.nHyd)")
    ExaModels.set_parameter!(prob.core, prob.p_x0, x0)
    copyto!(prob._x0_buf, Float64.(x0))
    return prob
end

function set_inflows!(prob::EmbeddedHydroExaDEProblem, w::AbstractVector)
    expected = prob.horizon * prob.nHyd
    length(w) == expected || error("w must have length T*nHyd=$expected")
    ExaModels.set_parameter!(prob.core, prob.p_inflow, w)
    copyto!(prob._inflow_buf, Float64.(w))
    return prob
end

function set_uncertainty!(prob::EmbeddedHydroExaDEProblem, w::AbstractVector)
    set_inflows!(prob, w)
end

function set_targets!(::EmbeddedHydroExaDEProblem, ::AbstractVector)
    return nothing
end

function set_demand!(prob::EmbeddedHydroExaDEProblem, demand_matrix::AbstractMatrix)
    T, nB = size(demand_matrix)
    T == prob.horizon || error("demand_matrix must have T=$(prob.horizon) rows")
    nB == prob.nBus   || error("demand_matrix must have nBus=$(prob.nBus) cols")
    flat = [demand_matrix[t, b] for t in 1:T for b in 1:nB]
    ExaModels.set_parameter!(prob.core, prob.p_demand, flat)
    return prob
end

function embedded_hydro_realized_states(prob::EmbeddedHydroExaDEProblem, result)
    T  = prob.horizon
    nH = prob.nHyd
    sol = result.solution
    return sol[prob._res_start + nH : prob._res_start + (T + 1) * nH - 1]
end

function hydro_solution(prob::EmbeddedHydroExaDEProblem, result)
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
    else
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

# ── Oracle builder helper ───────────────────────────────────────────────────────

function _build_hydro_oracle(policy, T, nHyd, res_start, dp_start, dn_start,
                              nvar_total, inflow_buf, x0_buf)

    _res(s)     = res_start + (s-1)*nHyd : res_start + s*nHyd - 1
    _dp(t)      = dp_start  + (t-1)*nHyd : dp_start  + t*nHyd - 1
    _dn(t)      = dn_start  + (t-1)*nHyd : dn_start  + t*nHyd - 1
    _inflow(t)  = (t-1)*nHyd+1 : t*nHyd
    _crow(t)    = (t-1)*nHyd+1 : t*nHyd

    jac_r = Int[]
    jac_c = Int[]
    for t in 1:T, r in 1:nHyd
        row = (t-1)*nHyd + r
        push!(jac_r, row); push!(jac_c, res_start + t*nHyd + r - 1)
        push!(jac_r, row); push!(jac_c, dp_start + (t-1)*nHyd + r - 1)
        push!(jac_r, row); push!(jac_c, dn_start + (t-1)*nHyd + r - 1)
        if t > 1
            for j in 1:nHyd
                push!(jac_r, row); push!(jac_c, res_start + (t-1)*nHyd + j - 1)
            end
        end
    end
    nnzj = length(jac_r)

    const_jac_cpu = zeros(Float64, nnzj)
    nn_jac_ranges = Dict{Tuple{Int,Int}, UnitRange{Int}}()
    k = 0
    for t in 1:T, r in 1:nHyd
        const_jac_cpu[k+1] = -1.0
        const_jac_cpu[k+2] = -1.0
        const_jac_cpu[k+3] =  1.0
        k += 3
        if t > 1
            nn_jac_ranges[(t,r)] = (k+1):(k+nHyd)
            k += nHyd
        end
    end
    const_jac_dev = similar(inflow_buf, Float64, nnzj)
    copyto!(const_jac_dev, const_jac_cpu)

    eye_cpu = [let e = zeros(Float32, nHyd); e[r] = 1.0f0; e end for r in 1:nHyd]
    eye_basis = [copyto!(similar(x0_buf, Float32, nHyd), e) for e in eye_cpu]

    function oracle_f!(c, xv)
        Flux.reset!(policy)
        for t in 1:T
            x_prev  = t == 1 ? Float32.(x0_buf) : Float32.(xv[_res(t)])
            infl_t  = Float32.(inflow_buf[_inflow(t)])
            nn_out  = policy(vcat(infl_t, x_prev))
            c[_crow(t)] .= Float64.(nn_out) .- xv[_res(t+1)] .- xv[_dp(t)] .+ xv[_dn(t)]
        end
        return nothing
    end

    function oracle_jac!(vals, xv)
        copyto!(vals, const_jac_dev)
        Flux.reset!(policy)

        x_prev_1 = Float32.(x0_buf)
        infl_1   = Float32.(inflow_buf[_inflow(1)])
        policy(vcat(infl_1, x_prev_1))

        for t in 2:T
            x_prev = Float32.(xv[_res(t)])
            infl_t = Float32.(inflow_buf[_inflow(t)])
            _, back = Zygote.pullback(xp -> policy(vcat(infl_t, xp)), x_prev)
            for r in 1:nHyd
                jac_row = back(eye_basis[r])[1]
                if jac_row !== nothing
                    vals[nn_jac_ranges[(t,r)]] .= Float64.(jac_row)
                end
            end
        end
        return nothing
    end

    function oracle_vjp!(Jtv, xv, λ)
        fill!(Jtv, 0.0)
        Flux.reset!(policy)
        for t in 1:T
            x_prev = t == 1 ? Float32.(x0_buf) : Float32.(xv[_res(t)])
            infl_t = Float32.(inflow_buf[_inflow(t)])
            λ_t    = Float32.(λ[_crow(t)])

            Jtv[_res(t+1)] .-= Float64.(λ_t)
            Jtv[_dp(t)]    .-= Float64.(λ_t)
            Jtv[_dn(t)]    .+= Float64.(λ_t)

            if t > 1
                _, back = Zygote.pullback(xp -> policy(vcat(infl_t, xp)), x_prev)
                d_xprev = back(λ_t)[1]
                if d_xprev !== nothing
                    Jtv[_res(t)] .+= Float64.(d_xprev)
                end
            else
                policy(vcat(infl_t, x_prev))
            end
        end
        return nothing
    end

    return ExaModels.VectorNonlinearOracle(
        nvar     = nvar_total,
        ncon     = T * nHyd,
        nnzj     = nnzj,
        jac_rows = jac_r,
        jac_cols = jac_c,
        lcon     = zeros(T * nHyd),
        ucon     = zeros(T * nHyd),
        f!       = oracle_f!,
        jac!     = oracle_jac!,
        vjp!     = oracle_vjp!,
    )
end

# ── DC builder ──────────────────────────────────────────────────────────────────

function _build_embedded_dc_hydro_de(
    policy,
    power_data::PowerData,
    hydro_data::HydroData,
    T::Int;
    backend    = nothing,
    float_type::Type{<:AbstractFloat} = Float64,
    target_penalty::Union{Real,Symbol} = :auto,
    target_penalty_l1::Union{Real,Symbol,Nothing} = :auto,
    demand_matrix = nothing,
    deficit_cost::Union{Nothing,Real} = nothing,
    load_scaler::Real = 1.0,
)
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

    # ── Variables (track offsets) ─────────────────────────────────────────────
    var_offset = 0

    va = ExaModels.variable(core, T * nBus)
    var_offset += T * nBus

    pg_lb = float_type.(repeat([g.pmin for g in power_data.gens], T))
    pg_ub = float_type.(repeat([g.pmax for g in power_data.gens], T))
    pg = ExaModels.variable(core, T * nGen; lvar = pg_lb, uvar = pg_ub)
    var_offset += T * nGen

    pf_lb = float_type.(repeat([-b.rate_a for b in power_data.branches], T))
    pf_ub = float_type.(repeat([ b.rate_a for b in power_data.branches], T))
    pf = ExaModels.variable(core, T * nBranch; lvar = pf_lb, uvar = pf_ub)
    var_offset += T * nBranch

    deficit = ExaModels.variable(core, T * nBus; lvar = float_type(0))
    var_offset += T * nBus

    res_start = var_offset + 1
    res_lb = float_type.(repeat([h.min_vol  for h in hydro_data.units], T+1))
    res_ub = float_type.(repeat([h.max_vol  for h in hydro_data.units], T+1))
    reservoir = ExaModels.variable(core, (T+1) * nHyd; lvar = res_lb, uvar = res_ub)
    var_offset += (T+1) * nHyd

    out_lb = float_type.(repeat([h.min_turn for h in hydro_data.units], T))
    out_ub = float_type.(repeat([h.max_turn for h in hydro_data.units], T))
    outflow = ExaModels.variable(core, T * nHyd; lvar = out_lb, uvar = out_ub)
    var_offset += T * nHyd

    spill = ExaModels.variable(core, T * nHyd; lvar = float_type(0))
    var_offset += T * nHyd

    dp_start = var_offset + 1
    delta_pos = ExaModels.variable(core, T * nHyd; lvar = float_type(0))
    var_offset += T * nHyd

    dn_start = var_offset + 1
    delta_neg = ExaModels.variable(core, T * nHyd; lvar = float_type(0))
    var_offset += T * nHyd

    nvar_total = var_offset

    # ── Parameters ────────────────────────────────────────────────────────────
    init_demand = if demand_matrix !== nothing
        float_type.(load_scaler .* [demand_matrix[t, b] for t in 1:T for b in 1:nBus])
    else
        float_type.(load_scaler .* repeat(power_data.default_bus_demand, T))
    end

    p_demand       = ExaModels.parameter(core, init_demand)
    p_x0           = ExaModels.parameter(core, zeros(float_type, nHyd))
    p_inflow       = ExaModels.parameter(core, zeros(float_type, T * nHyd))
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

    delta_items = [(idx = _ri(nHyd, t, r),) for t in 1:T for r in 1:nHyd]
    ExaModels.objective(core,
        p_penalty_half[item.idx] * (delta_pos[item.idx] - delta_neg[item.idx])^2
        for item in delta_items
    )

    if use_l1
        ExaModels.objective(core,
            p_penalty_l1[item.idx] * (delta_pos[item.idx] + delta_neg[item.idx])
            for item in delta_items
        )
    end

    # ── Constraints 1–7 ──────────────────────────────────────────────────────
    n_con = 0

    nRef = length(power_data.ref_buses)
    ref_items = [(t = t, ref = ref) for t in 1:T for ref in power_data.ref_buses]
    ExaModels.constraint(core,
        va[_bi(nBus, item.t, item.ref)]
        for item in ref_items
    )
    n_con += T * nRef

    ohm_items = [(t = t, f = br.f_bus, tb = br.t_bus, br = br_pos,
                  b = float_type(br.b))
                 for t in 1:T for (br_pos, br) in enumerate(power_data.branches)]
    ExaModels.constraint(core,
        item.b * (va[_bi(nBus, item.t, item.f)] - va[_bi(nBus, item.t, item.tb)])
        - pf[_bri(nBranch, item.t, item.br)]
        for item in ohm_items
    )
    n_con += T * nBranch

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

    ic_items = [(r = r,) for r in 1:nHyd]
    ExaModels.constraint(core,
        reservoir[_ri(nHyd, 1, item.r)] - p_x0[item.r]
        for item in ic_items
    )
    n_con += nHyd

    wb_items = [(res_next = _ri(nHyd, t+1, r), res_curr = _ri(nHyd, t, r),
                 out_idx = _ri(nHyd, t, r), spill_idx = _ri(nHyd, t, r),
                 inflow_p = _ri(nHyd, t, r), K = K)
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

    tc_items = [(gen_col = _gi(nGen, t, h.gen_pos), out_col = _ri(nHyd, t, h.pos),
                 baseMVA = baseMVA, pf_h = float_type(h.pf))
                for t in 1:T for h in hydro_data.units]
    ExaModels.constraint(core,
        item.baseMVA * pg[item.gen_col] - item.pf_h * outflow[item.out_col]
        for item in tc_items
    )
    n_con += T * nHyd

    # ── Oracle (replaces target constraints) ──────────────────────────────────
    inflow_buf = backend === nothing ?
        zeros(Float64, T * nHyd) :
        KernelAbstractions.zeros(backend, Float64, T * nHyd)
    x0_buf = backend === nothing ?
        zeros(Float64, nHyd) :
        KernelAbstractions.zeros(backend, Float64, nHyd)

    oracle = _build_hydro_oracle(policy, T, nHyd, res_start, dp_start, dn_start,
                                  nvar_total, inflow_buf, x0_buf)
    ExaModels.constraint(core, oracle)
    target_con_range = (n_con + 1):(n_con + T * nHyd)

    model = ExaModels.ExaModel(core)

    return EmbeddedHydroExaDEProblem(
        core, model,
        p_demand, nothing, p_x0, p_inflow,
        p_penalty_half, Float64(ρ / 2),
        p_penalty_l1, Float64(ρ_l1),
        policy, nHyd, nHyd, nHyd,
        nBus, nGen, nBranch, T,
        :dc, target_con_range,
        res_start, dp_start, dn_start, nvar_total,
        inflow_buf, x0_buf,
    )
end

# ── AC polar builder ────────────────────────────────────────────────────────────

function _build_embedded_ac_hydro_de(
    policy,
    power_data::PowerData,
    hydro_data::HydroData,
    T::Int;
    backend    = nothing,
    float_type::Type{<:AbstractFloat} = Float64,
    target_penalty::Union{Real,Symbol} = :auto,
    target_penalty_l1::Union{Real,Symbol,Nothing} = :auto,
    demand_matrix = nothing,
    reactive_demand_matrix = nothing,
    deficit_cost::Union{Nothing,Real} = nothing,
    load_scaler::Real = 1.0,
)
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
    var_offset = 0

    va = ExaModels.variable(core, T * nBus)
    var_offset += T * nBus

    vm_lb = float_type.(repeat([b.vmin for b in power_data.buses], T))
    vm_ub = float_type.(repeat([b.vmax for b in power_data.buses], T))
    vm = ExaModels.variable(core, T * nBus; lvar = vm_lb, uvar = vm_ub,
                             start = ones(float_type, T * nBus))
    var_offset += T * nBus

    pg_lb = float_type.(repeat([g.pmin for g in power_data.gens], T))
    pg_ub = float_type.(repeat([g.pmax for g in power_data.gens], T))
    pg = ExaModels.variable(core, T * nGen; lvar = pg_lb, uvar = pg_ub)
    var_offset += T * nGen

    qg_lb = float_type.(repeat([isfinite(g.qmin) ? g.qmin : -1e4 for g in power_data.gens], T))
    qg_ub = float_type.(repeat([isfinite(g.qmax) ? g.qmax :  1e4 for g in power_data.gens], T))
    qg = ExaModels.variable(core, T * nGen; lvar = qg_lb, uvar = qg_ub)
    var_offset += T * nGen

    p_fr_lb = float_type.(repeat([-b.rate_a for b in power_data.branches], T))
    p_fr_ub = float_type.(repeat([ b.rate_a for b in power_data.branches], T))
    p_fr = ExaModels.variable(core, T * nBranch; lvar = p_fr_lb, uvar = p_fr_ub)
    q_fr = ExaModels.variable(core, T * nBranch; lvar = p_fr_lb, uvar = p_fr_ub)
    p_to = ExaModels.variable(core, T * nBranch; lvar = p_fr_lb, uvar = p_fr_ub)
    q_to = ExaModels.variable(core, T * nBranch; lvar = p_fr_lb, uvar = p_fr_ub)
    var_offset += 4 * T * nBranch

    deficit = ExaModels.variable(core, T * nBus; lvar = float_type(0))
    var_offset += T * nBus

    deficit_q = ExaModels.variable(core, T * nBus)
    var_offset += T * nBus

    res_start = var_offset + 1
    res_lb = float_type.(repeat([h.min_vol  for h in hydro_data.units], T+1))
    res_ub = float_type.(repeat([h.max_vol  for h in hydro_data.units], T+1))
    reservoir = ExaModels.variable(core, (T+1) * nHyd; lvar = res_lb, uvar = res_ub)
    var_offset += (T+1) * nHyd

    out_lb = float_type.(repeat([h.min_turn for h in hydro_data.units], T))
    out_ub = float_type.(repeat([h.max_turn for h in hydro_data.units], T))
    outflow = ExaModels.variable(core, T * nHyd; lvar = out_lb, uvar = out_ub)
    var_offset += T * nHyd

    spill = ExaModels.variable(core, T * nHyd; lvar = float_type(0))
    var_offset += T * nHyd

    dp_start = var_offset + 1
    delta_pos = ExaModels.variable(core, T * nHyd; lvar = float_type(0))
    var_offset += T * nHyd

    dn_start = var_offset + 1
    delta_neg = ExaModels.variable(core, T * nHyd; lvar = float_type(0))
    var_offset += T * nHyd

    nvar_total = var_offset

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
    p_penalty_half    = ExaModels.parameter(core, fill(float_type(ρ / 2), T * nHyd))
    p_penalty_l1      = ExaModels.parameter(core, fill(ρ_l1, T * nHyd))

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

    delta_items = [(idx = _ri(nHyd, t, r),) for t in 1:T for r in 1:nHyd]
    ExaModels.objective(core,
        p_penalty_half[item.idx] * (delta_pos[item.idx] - delta_neg[item.idx])^2
        for item in delta_items
    )
    if use_l1
        ExaModels.objective(core,
            p_penalty_l1[item.idx] * (delta_pos[item.idx] + delta_neg[item.idx])
            for item in delta_items
        )
    end

    # ── Constraints ───────────────────────────────────────────────────────────
    n_con = 0

    # 1. Reference angle
    nRef = length(power_data.ref_buses)
    ref_items = [(t = t, ref = ref) for t in 1:T for ref in power_data.ref_buses]
    ExaModels.constraint(core, va[_bi(nBus, item.t, item.ref)] for item in ref_items)
    n_con += T * nRef

    # 2. AC from-end active power flow
    pfr_items = [(t = t, f = br.f_bus, tb = br.t_bus, br = br_pos,
                  c3 = br_ac[br_pos].c3, c4 = br_ac[br_pos].c4, c5 = br_ac[br_pos].c5)
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

    # 3. AC from-end reactive power flow
    qfr_items = [(t = t, f = br.f_bus, tb = br.t_bus, br = br_pos,
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

    # 4. AC to-end active power flow
    pto_items = [(t = t, f = br.f_bus, tb = br.t_bus, br = br_pos,
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

    # 5. AC to-end reactive power flow
    qto_items = [(t = t, f = br.f_bus, tb = br.t_bus, br = br_pos,
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

    # 6. Phase angle difference
    ang_lb = float_type.(repeat([br.angmin for br in power_data.branches], T))
    ang_ub = float_type.(repeat([br.angmax for br in power_data.branches], T))
    ang_items = [(t = t, f = br.f_bus, tb = br.t_bus) for t in 1:T for br in power_data.branches]
    ExaModels.constraint(core,
        va[_bi(nBus, item.t, item.f)] - va[_bi(nBus, item.t, item.tb)]
        for item in ang_items;
        lcon = ang_lb, ucon = ang_ub,
    )
    n_con += T * nBranch

    # 7. Active KCL
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
        item.brow => -pg[item.gcol] for item in kcl_pg_items)
    kcl_pfr_items = [(t = t, brow = _bi(nBus, t, br.f_bus), bcol = _bri(nBranch, t, br_pos))
                     for t in 1:T for (br_pos, br) in enumerate(power_data.branches)]
    ExaModels.constraint!(core, c_kcl_p,
        item.brow => p_fr[item.bcol] for item in kcl_pfr_items)
    kcl_pto_items = [(t = t, brow = _bi(nBus, t, br.t_bus), bcol = _bri(nBranch, t, br_pos))
                     for t in 1:T for (br_pos, br) in enumerate(power_data.branches)]
    ExaModels.constraint!(core, c_kcl_p,
        item.brow => p_to[item.bcol] for item in kcl_pto_items)
    kcl_def_items = [(t = t, brow = _bi(nBus, t, b), dcol = _bi(nBus, t, b))
                     for t in 1:T for b in 1:nBus]
    ExaModels.constraint!(core, c_kcl_p,
        item.brow => -deficit[item.dcol] for item in kcl_def_items)
    n_con += T * nBus

    # 8. Reactive KCL
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
        item.brow => -qg[item.gcol] for item in kcl_qg_items)
    kcl_qfr_items = [(t = t, brow = _bi(nBus, t, br.f_bus), bcol = _bri(nBranch, t, br_pos))
                     for t in 1:T for (br_pos, br) in enumerate(power_data.branches)]
    ExaModels.constraint!(core, c_kcl_q,
        item.brow => q_fr[item.bcol] for item in kcl_qfr_items)
    kcl_qto_items = [(t = t, brow = _bi(nBus, t, br.t_bus), bcol = _bri(nBranch, t, br_pos))
                     for t in 1:T for (br_pos, br) in enumerate(power_data.branches)]
    ExaModels.constraint!(core, c_kcl_q,
        item.brow => q_to[item.bcol] for item in kcl_qto_items)
    ExaModels.constraint!(core, c_kcl_q,
        item.brow => -deficit_q[item.dcol] for item in kcl_def_items)
    n_con += T * nBus

    # 9. Initial reservoir
    ic_items = [(r = r,) for r in 1:nHyd]
    ExaModels.constraint(core,
        reservoir[_ri(nHyd, 1, item.r)] - p_x0[item.r]
        for item in ic_items
    )
    n_con += nHyd

    # 10. Water balance
    wb_items = [(res_next = _ri(nHyd, t+1, r), res_curr = _ri(nHyd, t, r),
                 out_idx = _ri(nHyd, t, r), spill_idx = _ri(nHyd, t, r),
                 inflow_p = _ri(nHyd, t, r), K = K)
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
            item.row => -item.K * outflow[item.col] for item in wb_turn_items)
    end
    if !isempty(hydro_data.upstream_spills)
        wb_spill_items = [(row = _ri(nHyd, t, conn.downstream_pos),
                           col = _ri(nHyd, t, conn.upstream_pos))
                          for t in 1:T for conn in hydro_data.upstream_spills]
        ExaModels.constraint!(core, c_wb,
            item.row => -spill[item.col] for item in wb_spill_items)
    end
    n_con += T * nHyd

    # 11. Turbine coupling
    tc_items = [(gen_col = _gi(nGen, t, h.gen_pos), out_col = _ri(nHyd, t, h.pos),
                 baseMVA = baseMVA, pf_h = float_type(h.pf))
                for t in 1:T for h in hydro_data.units]
    ExaModels.constraint(core,
        item.baseMVA * pg[item.gen_col] - item.pf_h * outflow[item.out_col]
        for item in tc_items
    )
    n_con += T * nHyd

    # ── Oracle ────────────────────────────────────────────────────────────────
    inflow_buf = backend === nothing ?
        zeros(Float64, T * nHyd) :
        KernelAbstractions.zeros(backend, Float64, T * nHyd)
    x0_buf = backend === nothing ?
        zeros(Float64, nHyd) :
        KernelAbstractions.zeros(backend, Float64, nHyd)

    oracle = _build_hydro_oracle(policy, T, nHyd, res_start, dp_start, dn_start,
                                  nvar_total, inflow_buf, x0_buf)
    ExaModels.constraint(core, oracle)
    target_con_range = (n_con + 1):(n_con + T * nHyd)

    model = ExaModels.ExaModel(core)

    return EmbeddedHydroExaDEProblem(
        core, model,
        p_demand, p_reactive_demand, p_x0, p_inflow,
        p_penalty_half, Float64(ρ / 2),
        p_penalty_l1, Float64(ρ_l1),
        policy, nHyd, nHyd, nHyd,
        nBus, nGen, nBranch, T,
        :ac_polar, target_con_range,
        res_start, dp_start, dn_start, nvar_total,
        inflow_buf, x0_buf,
    )
end

# ── Dispatcher ──────────────────────────────────────────────────────────────────

function build_embedded_hydro_de(
    policy,
    power_data::PowerData,
    hydro_data::HydroData,
    T::Int;
    formulation::Symbol = :dc,
    kwargs...,
)
    formulation in (:dc, :ac_polar) ||
        error("formulation must be :dc or :ac_polar, got :$formulation")

    if formulation === :dc
        return _build_embedded_dc_hydro_de(policy, power_data, hydro_data, T; kwargs...)
    else
        return _build_embedded_ac_hydro_de(policy, power_data, hydro_data, T; kwargs...)
    end
end
