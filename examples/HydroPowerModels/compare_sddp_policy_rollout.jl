# compare_sddp_policy_rollout.jl
#
# Diagnostic bridge:
#   1. Load a trained SDDP policy from saved cuts.
#   2. Simulate SDDP on sampled scenarios.
#   3. Reconstruct the exact inflow trajectory used by each SDDP simulation.
#   4. Replay SDDP's realized next reservoir state as the TSDDR rollout target.
#   5. Compare SDDP rollout cost/state trajectory to Exa/TSDDR stagewise rollout.
#
# Run with the Exa project active. The script stacks the sibling SDDP example
# environment via LOAD_PATH so it can load HydroPowerModels/SDDP/CSV, for example:
#   JULIA_DEPOT_PATH=/storage/scratch1/9/arosemberg3/julia_depot_sddp:$JULIA_DEPOT_PATH \
#   julia --project=/storage/home/hcoda1/9/arosemberg3/scratch/DecisionRulesExa.jl \
#     /storage/home/hcoda1/9/arosemberg3/scratch/DecisionRulesExa.jl/examples/HydroPowerModels/compare_sddp_policy_rollout.jl

const SCRIPT_DIR = dirname(@__FILE__)
const DEFAULT_SDDP_ENV =
    "/storage/home/hcoda1/9/arosemberg3/scratch/DecisionRules.jl/examples/HydroPowerModels/sddp"
const SDDP_ENV = get(ENV, "DR_SDDP_ENV", DEFAULT_SDDP_ENV)
const BRIDGE_ENV = joinpath(SCRIPT_DIR, "sddp_bridge_env")

function _prepend_load_path!(env_path::AbstractString)
    filter!(p -> p != env_path, LOAD_PATH)
    pushfirst!(LOAD_PATH, env_path)
    return nothing
end

_prepend_load_path!(SDDP_ENV)
_prepend_load_path!(BRIDGE_ENV)

using Clarabel
using CSV
using DecisionRulesExa
using ExaModels
using HydroPowerModels
using MadNLP
using PowerModels
using Random
using SDDP
using Statistics
import Flux

include(joinpath(SCRIPT_DIR, "hydro_power_data.jl"))
include(joinpath(SCRIPT_DIR, "hydro_power_exa.jl"))

const SEED = parse(Int, get(ENV, "DR_SDDP_BRIDGE_SEED", "1221"))
const NUM_SIMULATIONS = parse(Int, get(ENV, "DR_SDDP_BRIDGE_SIMULATIONS", "4"))
const REPORT_STAGES = parse(Int, get(ENV, "DR_SDDP_BRIDGE_REPORT_STAGES", "96"))
const RM_STAGES = parse(Int, get(ENV, "DR_SDDP_BRIDGE_RM_STAGES", "30"))
const SDDP_STAGES = parse(Int, get(ENV, "DR_SDDP_BRIDGE_SDDP_STAGES", string(REPORT_STAGES + RM_STAGES)))
const TARGET_PENALTY_MULTS = [
    parse(Float64, strip(x))
    for x in split(
        get(
            ENV,
            "DR_SDDP_BRIDGE_TARGET_PENALTY_MULTS",
            get(ENV, "DR_SDDP_BRIDGE_TARGET_PENALTY_MULT", "8.0"),
        ),
        ",",
    )
    if !isempty(strip(x))
]
const TARGET_PENALTY_MULT = first(TARGET_PENALTY_MULTS)
const ACTIVE_TARGET_PENALTY_MULT = Ref(TARGET_PENALTY_MULT)
const TARGET_PENALTY_DISCOUNT_GAMMAS = [
    parse(Float64, strip(x))
    for x in split(get(ENV, "DR_SDDP_BRIDGE_TARGET_DISCOUNT_GAMMAS", "1.0"), ",")
    if !isempty(strip(x))
]
const ACTIVE_TARGET_PENALTY_DISCOUNT_GAMMA = Ref(first(TARGET_PENALTY_DISCOUNT_GAMMAS))
const STAGE1_DETAIL = parse(Bool, get(ENV, "DR_SDDP_BRIDGE_STAGE1_DETAIL", "false"))
const DE_TARGET_SOLVE = parse(Bool, get(ENV, "DR_SDDP_BRIDGE_DE_TARGET_SOLVE", "false"))

const SDDP_ROOT = dirname(SDDP_ENV)
const SDDP_CASE_DIR = joinpath(SDDP_ROOT, "bolivia")
const EXA_CASE_DIR = joinpath(SCRIPT_DIR, "bolivia")
const CUTS_FILE = get(
    ENV,
    "DR_SDDP_BRIDGE_CUTS",
    joinpath(SDDP_CASE_DIR, "ACPPowerModel", "SOCWRConicPowerModel-ACPPowerModel.cuts.json"),
)
const OUT_DIR = joinpath(SCRIPT_DIR, "results", "sddp_bridge")
mkpath(OUT_DIR)

const FORMULATION_BACKWARD = SOCWRConicPowerModel
const FORMULATION_FORWARD = ACPPowerModel
const EXA_FORMULATION = :ac_polar
const DEFICIT_COST = 1e5
const LOAD_SCALER = 0.6
const SOLVER_KWARGS = (print_level = MadNLP.ERROR, tol = 1e-6, max_iter = 9000)

function _cidx(i::Int, n::Int)
    return mod(i, n) == 0 ? n : mod(i, n)
end

function _bridge_target_violation_share(objective::Real, objective_no_target_penalty::Real)
    penalty = objective - objective_no_target_penalty
    (isfinite(objective) && isfinite(penalty) && abs(objective) > 1e-12) || return NaN
    return penalty / objective
end

function _mult_tag(mult::Real)
    return replace(replace(string(mult), "." => "p"), "-" => "m")
end

function _load_sddp_case_data()
    alldata = HydroPowerModels.parse_folder(SDDP_CASE_DIR)
    for load in values(alldata[1]["powersystem"]["load"])
        load["qd"] *= LOAD_SCALER
        load["pd"] *= LOAD_SCALER
    end
    return alldata
end

function _clarabel_optimizer()
    return Clarabel.Optimizer(;
        verbose = false,
        max_iter = parse(Int, get(ENV, "DR_SDDP_CLARABEL_MAX_ITER", "1000")),
        tol_gap_abs = parse(Float64, get(ENV, "DR_SDDP_CLARABEL_TOL", "1e-7")),
        tol_gap_rel = parse(Float64, get(ENV, "DR_SDDP_CLARABEL_TOL", "1e-7")),
        tol_feas = parse(Float64, get(ENV, "DR_SDDP_CLARABEL_TOL", "1e-7")),
    )
end

function _madnlp_optimizer()
    return MadNLP.Optimizer(;
        print_level = parse(Int, get(ENV, "DR_SDDP_MADNLP_PRINT_LEVEL", "0")),
    )
end

function _build_sddp_model()
    isfile(CUTS_FILE) || error("SDDP cuts file not found: $CUTS_FILE")
    alldata = _load_sddp_case_data()
    params = HydroPowerModels.create_param(;
        stages = SDDP_STAGES,
        model_constructor_grid = FORMULATION_BACKWARD,
        model_constructor_grid_forward = FORMULATION_FORWARD,
        post_method = PowerModels.build_opf,
        optimizer = _clarabel_optimizer,
        optimizer_forward = _madnlp_optimizer,
    )
    model = HydroPowerModels.hydro_thermal_operation(alldata, params)
    SDDP.read_cuts_from_file(model.forward_graph, CUTS_FILE)
    return model
end

function _sddp_inflow_flat(results, sim_idx::Int, horizon::Int)
    data = results[:data][1]
    hydro = data["hydro"]
    n_hyd = hydro["nHyd"]
    n_rows = hydro["size_inflow"][1]
    w = Vector{Float64}(undef, horizon * n_hyd)
    for t in 1:horizon
        ω = results[:simulations][sim_idx][t][:noise_term]
        row = _cidx(t, n_rows)
        for r in 1:n_hyd
            w[(t - 1) * n_hyd + r] =
                Float64(hydro["Hydrogenerators"][r]["inflow"][row, ω])
        end
    end
    return w
end

function _sddp_targets(results, sim_idx::Int, horizon::Int)
    data = results[:data][1]
    n_hyd = data["hydro"]["nHyd"]
    targets = Matrix{Float64}(undef, n_hyd, horizon)
    for t in 1:horizon
        stage = results[:simulations][sim_idx][t]
        for r in 1:n_hyd
            targets[r, t] = Float64(stage[:reservoirs][:reservoir][r].out)
        end
    end
    return targets
end

function _sddp_objective(results, sim_idx::Int, horizon::Int)
    return sum(
        Float64(results[:simulations][sim_idx][t][:stage_objective])
        for t in 1:horizon
    )
end

struct ReplayTargetPolicy{M}
    targets::M
    stage::Base.RefValue{Int}
end

ReplayTargetPolicy(targets::AbstractMatrix) = ReplayTargetPolicy(targets, Ref(1))

function Flux.reset!(policy::ReplayTargetPolicy)
    policy.stage[] = 1
    return nothing
end

function (policy::ReplayTargetPolicy)(input)
    t = policy.stage[]
    t <= size(policy.targets, 2) ||
        error("ReplayTargetPolicy called past horizon $(size(policy.targets, 2))")
    policy.stage[] = t + 1
    return copy(view(policy.targets, :, t))
end

function _penalty_weights(prob, target_penalty_mult::Real, discount_gamma::Real)
    return Float64[
        target_penalty_mult * discount_gamma^(t - 1)
        for t in 1:prob.horizon for _ in 1:prob.nHyd
    ]
end

function _set_target_penalty_multiplier!(
    prob,
    target_penalty_mult::Real;
    discount_gamma::Real = 1.0,
)
    weights = _penalty_weights(prob, target_penalty_mult, discount_gamma)
    ExaModels.set_parameter!(
        prob.core,
        prob.p_penalty_half,
        prob.base_penalty_half .* weights,
    )
    ExaModels.set_parameter!(
        prob.core,
        prob.p_penalty_l1,
        prob.base_penalty_l1 .* weights,
    )
    return prob
end

function _build_exa_rollout_problem(
    target_penalty_mult::Real = TARGET_PENALTY_MULT;
    horizon::Int = 1,
    discount_gamma::Real = 1.0,
)
    pm_file = joinpath(EXA_CASE_DIR, "PowerModels.json")
    hydro_file = joinpath(EXA_CASE_DIR, "hydro.json")
    inflow_file = joinpath(EXA_CASE_DIR, "inflows.csv")

    power_data = load_power_data(pm_file)
    hydro_data = load_hydro_data(hydro_file, inflow_file, power_data; num_stages = SDDP_STAGES)
    stage_demand = nothing

    prob = build_hydro_de(power_data, hydro_data, horizon;
        backend = nothing,
        float_type = Float64,
        formulation = EXA_FORMULATION,
        target_penalty = :auto,
        target_penalty_l1 = :auto,
        deficit_cost = DEFICIT_COST,
        demand_matrix = stage_demand,
        load_scaler = LOAD_SCALER,
    )
    _set_target_penalty_multiplier!(
        prob,
        target_penalty_mult;
        discount_gamma = discount_gamma,
    )

    x0 = Float64.([
        clamp(hydro_data.initial_volumes[r], hydro_data.units[r].min_vol, hydro_data.units[r].max_vol)
        for r in 1:hydro_data.nHyd
    ])
    lower = Float64.([h.min_vol for h in hydro_data.units])
    upper = Float64.([h.max_vol for h in hydro_data.units])
    return prob, x0, (lower, upper)
end

function _target_penalty_cost(stage_prob, sol, target_penalty_mult::Real, discount_gamma::Real)
    penalty_l2 = 0.0
    penalty_l1 = 0.0
    for t in 1:stage_prob.horizon
        weight = target_penalty_mult * discount_gamma^(t - 1)
        penalty_l2 += stage_prob.base_penalty_half * weight * sum(abs2, sol.delta[:, t])
        penalty_l1 += stage_prob.base_penalty_l1 * weight * sum(abs, sol.delta[:, t])
    end
    return penalty_l2, penalty_l1
end

function _set_exa_rollout_stage!(stage_prob, state_in, wt, target, stage)
    ExaModels.set_parameter!(stage_prob.core, stage_prob.p_x0, state_in)
    ExaModels.set_parameter!(stage_prob.core, stage_prob.p_inflow, wt)
    ExaModels.set_parameter!(stage_prob.core, stage_prob.p_target, target)
    return stage_prob
end

_exa_realized_state(stage_prob, result) =
    hydro_solution(stage_prob, result).reservoir[:, end]

function _exa_objective_no_target_penalty(stage_prob, result)
    sol = hydro_solution(stage_prob, result)
    target_penalty_mult = ACTIVE_TARGET_PENALTY_MULT[]
    discount_gamma = ACTIVE_TARGET_PENALTY_DISCOUNT_GAMMA[]
    penalty_l2, penalty_l1 = _target_penalty_cost(
        stage_prob,
        sol,
        target_penalty_mult,
        discount_gamma,
    )
    return result.objective - penalty_l2 - penalty_l1
end

function _solve_exa_stage_once!(stage_prob, state_in, wt, target; stage::Int = 1)
    _set_exa_rollout_stage!(stage_prob, state_in, wt, target, stage)
    result = MadNLP.madnlp(stage_prob.model; SOLVER_KWARGS...)
    return result
end

function _field_or_nan(x, name::Symbol)
    name in propertynames(x) || return NaN
    return Float64(getproperty(x, name))
end

function _write_stage1_detail(
    sddp_results,
    sim_idx::Int,
    exa_prob,
    x0,
    state_bounds,
    target_penalty_mult::Real,
)
    stage = sddp_results[:simulations][sim_idx][1]
    wt = _sddp_inflow_flat(sddp_results, sim_idx, 1)
    target = vec(_sddp_targets(sddp_results, sim_idx, 1))
    result = _solve_exa_stage_once!(exa_prob, x0, wt, target; stage = 1)
    sol = hydro_solution(exa_prob, result)

    exa_no_penalty = _exa_objective_no_target_penalty(exa_prob, result)
    penalty_l2, penalty_l1 = _target_penalty_cost(
        exa_prob,
        sol,
        target_penalty_mult,
        ACTIVE_TARGET_PENALTY_DISCOUNT_GAMMA[],
    )
    penalty_total = penalty_l2 + penalty_l1
    violation_share = _bridge_target_violation_share(result.objective, exa_no_penalty)

    lower, upper = state_bounds
    res_stage = stage[:reservoirs][:reservoir]
    n_hyd = exa_prob.nHyd

    detail_rows = NamedTuple[]
    for r in 1:n_hyd
        sddp_res = res_stage[r]
        push!(detail_rows, (
            target_penalty_mult = Float64(target_penalty_mult),
            scenario = sim_idx,
            stage = 1,
            reservoir = r,
            noise_term = Int(stage[:noise_term]),
            sddp_in = _field_or_nan(sddp_res, :in),
            exa_in = Float64(x0[r]),
            inflow = Float64(wt[r]),
            sddp_target_out = Float64(target[r]),
            exa_realized_out = Float64(sol.reservoir[r, end]),
            target_minus_exa = Float64(target[r] - sol.reservoir[r, end]),
            delta = Float64(sol.delta[r, 1]),
            abs_delta = Float64(abs(sol.delta[r, 1])),
            outflow = Float64(sol.outflow[r, 1]),
            spill = Float64(sol.spill[r, 1]),
            lower_bound = Float64(lower[r]),
            upper_bound = Float64(upper[r]),
            at_lower = Bool(isapprox(sol.reservoir[r, end], lower[r]; atol = 1e-6, rtol = 0.0)),
            at_upper = Bool(isapprox(sol.reservoir[r, end], upper[r]; atol = 1e-6, rtol = 0.0)),
        ))
    end

    summary = [(
        target_penalty_mult = Float64(target_penalty_mult),
        scenario = sim_idx,
        stage = 1,
        status = String(string(result.status)),
        solve_succeeded = Bool(DecisionRulesExa.solve_succeeded(result)),
        noise_term = Int(stage[:noise_term]),
        sddp_stage_objective = Float64(stage[:stage_objective]),
        exa_objective = Float64(result.objective),
        exa_objective_no_target_penalty = Float64(exa_no_penalty),
        objective_gap_no_target = Float64(exa_no_penalty - stage[:stage_objective]),
        target_penalty_l2 = Float64(penalty_l2),
        target_penalty_l1 = Float64(penalty_l1),
        target_penalty_total = Float64(penalty_total),
        target_violation_share = Float64(violation_share),
        max_abs_delta = maximum(abs, sol.delta[:, 1]),
        mean_abs_delta = mean(abs.(sol.delta[:, 1])),
        sum_abs_delta = sum(abs, sol.delta[:, 1]),
        total_pg = Float64(sum(sol.pg[:, 1])),
        total_p_deficit = Float64(sum(sol.deficit[:, 1])),
        total_qg = hasproperty(sol, :qg) ? Float64(sum(sol.qg[:, 1])) : NaN,
        total_q_deficit = hasproperty(sol, :deficit_q) ? Float64(sum(sol.deficit_q[:, 1])) : NaN,
    )]

    tag = _mult_tag(target_penalty_mult)
    detail_file = joinpath(OUT_DIR, "sddp_exa_stage1_detail_seed$(SEED)_scenario$(sim_idx)_mult$(tag).csv")
    summary_file = joinpath(OUT_DIR, "sddp_exa_stage1_summary_seed$(SEED)_scenario$(sim_idx)_mult$(tag).csv")
    CSV.write(detail_file, detail_rows)
    CSV.write(summary_file, summary)
    println("Wrote first-stage detail: ", detail_file)
    println("Wrote first-stage summary: ", summary_file)
    println(
        "stage1 scenario=$sim_idx ",
        "mult=$target_penalty_mult ",
        "sddp_obj=$(round(summary[1].sddp_stage_objective; digits=6)) ",
        "exa_no_target=$(round(summary[1].exa_objective_no_target_penalty; digits=6)) ",
        "gap=$(round(summary[1].objective_gap_no_target; digits=6)) ",
        "penalty_share=$(summary[1].target_violation_share) ",
        "max_abs_delta=$(summary[1].max_abs_delta)",
    )
    return summary[1], detail_rows
end

function _write_de_target_solve(
    sddp_results,
    sim_idx::Int,
    horizon::Int,
    target_penalty_mult::Real,
    discount_gamma::Real,
)
    exa_prob, x0, state_bounds = _build_exa_rollout_problem(
        target_penalty_mult;
        horizon = horizon,
        discount_gamma = discount_gamma,
    )
    w_flat = _sddp_inflow_flat(sddp_results, sim_idx, horizon)
    targets = _sddp_targets(sddp_results, sim_idx, horizon)
    target_flat = vec(targets)

    ExaModels.set_parameter!(exa_prob.core, exa_prob.p_x0, x0)
    ExaModels.set_parameter!(exa_prob.core, exa_prob.p_inflow, w_flat)
    ExaModels.set_parameter!(exa_prob.core, exa_prob.p_target, target_flat)

    result = MadNLP.madnlp(exa_prob.model; SOLVER_KWARGS...)
    sol = hydro_solution(exa_prob, result)
    exa_states = sol.reservoir[:, 2:(horizon + 1)]
    diff = exa_states .- targets
    penalty_l2, penalty_l1 = _target_penalty_cost(
        exa_prob,
        sol,
        target_penalty_mult,
        discount_gamma,
    )
    penalty_total = penalty_l2 + penalty_l1
    exa_no_penalty = result.objective - penalty_total
    sddp_obj = _sddp_objective(sddp_results, sim_idx, horizon)

    summary = [(
        target_penalty_mult = Float64(target_penalty_mult),
        target_discount_gamma = Float64(discount_gamma),
        scenario = sim_idx,
        horizon = horizon,
        status = String(string(result.status)),
        solve_succeeded = Bool(DecisionRulesExa.solve_succeeded(result)),
        sddp_objective = Float64(sddp_obj),
        exa_objective = Float64(result.objective),
        exa_objective_no_target_penalty = Float64(exa_no_penalty),
        objective_gap = Float64(result.objective - sddp_obj),
        no_target_gap = Float64(exa_no_penalty - sddp_obj),
        target_penalty_l2 = Float64(penalty_l2),
        target_penalty_l1 = Float64(penalty_l1),
        target_penalty_total = Float64(penalty_total),
        target_violation_share = Float64(_bridge_target_violation_share(result.objective, exa_no_penalty)),
        max_abs_delta = maximum(abs, sol.delta),
        mean_abs_delta = mean(abs.(sol.delta)),
        sum_abs_delta = sum(abs, sol.delta),
        max_state_abs_diff = maximum(abs, diff),
        mean_state_abs_diff = mean(abs.(diff)),
        final_state_abs_diff = maximum(abs.(exa_states[:, end] .- targets[:, end])),
        total_pg = Float64(sum(sol.pg)),
        total_p_deficit = Float64(sum(sol.deficit)),
        total_qg = hasproperty(sol, :qg) ? Float64(sum(sol.qg)) : NaN,
        total_q_deficit = hasproperty(sol, :deficit_q) ? Float64(sum(sol.deficit_q)) : NaN,
    )]

    detail_rows = NamedTuple[]
    lower, upper = state_bounds
    for t in 1:horizon, r in 1:exa_prob.nHyd
        push!(detail_rows, (
            target_penalty_mult = Float64(target_penalty_mult),
            target_discount_gamma = Float64(discount_gamma),
            scenario = sim_idx,
            stage = t,
            reservoir = r,
            sddp_target_out = Float64(targets[r, t]),
            exa_realized_out = Float64(exa_states[r, t]),
            target_minus_exa = Float64(targets[r, t] - exa_states[r, t]),
            delta = Float64(sol.delta[r, t]),
            abs_delta = Float64(abs(sol.delta[r, t])),
            outflow = Float64(sol.outflow[r, t]),
            spill = Float64(sol.spill[r, t]),
            lower_bound = Float64(lower[r]),
            upper_bound = Float64(upper[r]),
            at_lower = Bool(isapprox(exa_states[r, t], lower[r]; atol = 1e-6, rtol = 0.0)),
            at_upper = Bool(isapprox(exa_states[r, t], upper[r]; atol = 1e-6, rtol = 0.0)),
        ))
    end

    mult_tag = _mult_tag(target_penalty_mult)
    gamma_tag = _mult_tag(discount_gamma)
    summary_file = joinpath(
        OUT_DIR,
        "sddp_targets_exa_de_summary_seed$(SEED)_scenario$(sim_idx)_h$(horizon)_mult$(mult_tag)_gamma$(gamma_tag).csv",
    )
    detail_file = joinpath(
        OUT_DIR,
        "sddp_targets_exa_de_detail_seed$(SEED)_scenario$(sim_idx)_h$(horizon)_mult$(mult_tag)_gamma$(gamma_tag).csv",
    )
    CSV.write(summary_file, summary)
    CSV.write(detail_file, detail_rows)
    println("Wrote DE target summary: ", summary_file)
    println("Wrote DE target detail: ", detail_file)
    println(
        "de-target scenario=$sim_idx horizon=$horizon mult=$target_penalty_mult gamma=$discount_gamma ",
        "sddp=$(round(summary[1].sddp_objective; digits=3)) ",
        "exa_no_target=$(round(summary[1].exa_objective_no_target_penalty; digits=3)) ",
        "gap=$(round(summary[1].no_target_gap; digits=3)) ",
        "violation_share=$(summary[1].target_violation_share) ",
        "max_state_diff=$(summary[1].max_state_abs_diff)",
    )
    return summary[1], detail_rows
end

function main()
    println("SDDP cuts: ", CUTS_FILE)
    println("SDDP stages: ", SDDP_STAGES, "  compared stages: ", REPORT_STAGES)
    println("Simulations: ", NUM_SIMULATIONS)
    println("Target penalty multipliers in Exa rollout: ", TARGET_PENALTY_MULTS)
    println("Target penalty discount gammas: ", TARGET_PENALTY_DISCOUNT_GAMMAS)

    Random.seed!(SEED)
    sddp_model = _build_sddp_model()
    sddp_results = HydroPowerModels.simulate(sddp_model, NUM_SIMULATIONS)

    rows = NamedTuple[]
    for target_penalty_mult in TARGET_PENALTY_MULTS
        for discount_gamma in TARGET_PENALTY_DISCOUNT_GAMMAS
            ACTIVE_TARGET_PENALTY_MULT[] = target_penalty_mult
            ACTIVE_TARGET_PENALTY_DISCOUNT_GAMMA[] = discount_gamma

            if DE_TARGET_SOLVE
                for sim_idx in 1:NUM_SIMULATIONS
                    _write_de_target_solve(
                        sddp_results,
                        sim_idx,
                        REPORT_STAGES,
                        target_penalty_mult,
                        discount_gamma,
                    )
                end
            end
        end

        ACTIVE_TARGET_PENALTY_MULT[] = target_penalty_mult
        ACTIVE_TARGET_PENALTY_DISCOUNT_GAMMA[] = 1.0
        exa_prob, x0, state_bounds = _build_exa_rollout_problem(
            target_penalty_mult;
            discount_gamma = 1.0,
        )

        if STAGE1_DETAIL
            for sim_idx in 1:NUM_SIMULATIONS
                _write_stage1_detail(
                    sddp_results,
                    sim_idx,
                    exa_prob,
                    x0,
                    state_bounds,
                    target_penalty_mult,
                )
            end
            REPORT_STAGES == 1 || println("Continuing to aggregate rollout comparison after first-stage detail.")
        end

        for sim_idx in 1:NUM_SIMULATIONS
            w_flat = _sddp_inflow_flat(sddp_results, sim_idx, REPORT_STAGES)
            targets = _sddp_targets(sddp_results, sim_idx, REPORT_STAGES)
            policy = ReplayTargetPolicy(targets)
            sddp_obj = _sddp_objective(sddp_results, sim_idx, REPORT_STAGES)

            exa_result = rollout_tsddr(
                policy,
                x0,
                exa_prob,
                w_flat;
                horizon = REPORT_STAGES,
                n_uncertainty = length(x0),
                set_stage_parameters! = _set_exa_rollout_stage!,
                realized_state = _exa_realized_state,
                objective_no_target_penalty = _exa_objective_no_target_penalty,
                madnlp_kwargs = SOLVER_KWARGS,
                warmstart = false,
                policy_state = :target,
                reuse_solver = false,
                state_bounds = state_bounds,
                retry_on_failure = true,
            )

            if exa_result === nothing
                push!(rows, (
                    target_penalty_mult = Float64(target_penalty_mult),
                    scenario = sim_idx,
                    status = "exa_failed",
                    sddp_objective = sddp_obj,
                    exa_objective = NaN,
                    exa_objective_no_target_penalty = NaN,
                    objective_gap = NaN,
                    no_target_gap = NaN,
                    target_violation_share = NaN,
                    max_state_abs_diff = NaN,
                    mean_state_abs_diff = NaN,
                    final_state_abs_diff = NaN,
                ))
                println("mult=$target_penalty_mult scenario=$sim_idx status=exa_failed sddp_objective=$sddp_obj")
                continue
            end

            exa_states = hcat(exa_result.state_trajectory[2:end]...)
            diff = exa_states .- targets
            row = (
                target_penalty_mult = Float64(target_penalty_mult),
                scenario = sim_idx,
                status = "ok",
                sddp_objective = sddp_obj,
                exa_objective = Float64(exa_result.objective),
                exa_objective_no_target_penalty = Float64(exa_result.objective_no_target_penalty),
                objective_gap = Float64(exa_result.objective - sddp_obj),
                no_target_gap = Float64(exa_result.objective_no_target_penalty - sddp_obj),
                target_violation_share = Float64(exa_result.target_violation_share),
                max_state_abs_diff = maximum(abs, diff),
                mean_state_abs_diff = mean(abs.(diff)),
                final_state_abs_diff = maximum(abs.(exa_result.final_state .- targets[:, end])),
            )
            push!(rows, row)
            println(
                "mult=$(row.target_penalty_mult) scenario=$(row.scenario) status=ok ",
                "sddp=$(round(row.sddp_objective; digits=3)) ",
                "exa_no_target=$(round(row.exa_objective_no_target_penalty; digits=3)) ",
                "gap=$(round(row.no_target_gap; digits=3)) ",
                "violation_share=$(row.target_violation_share) ",
                "max_state_diff=$(row.max_state_abs_diff)",
            )
        end
    end

    out_file = joinpath(
        OUT_DIR,
        "sddp_policy_in_exa_rollout_seed$(SEED)_n$(NUM_SIMULATIONS)_h$(REPORT_STAGES)_sweep.csv",
    )
    CSV.write(out_file, rows)
    println("Wrote: ", out_file)

    ok_rows = filter(r -> r.status == "ok", rows)
    if !isempty(ok_rows)
        println("Mean SDDP objective: ", mean(r.sddp_objective for r in ok_rows))
        println("Mean Exa no-target objective: ", mean(r.exa_objective_no_target_penalty for r in ok_rows))
        println("Mean no-target gap: ", mean(r.no_target_gap for r in ok_rows))
        println("Max state abs diff: ", maximum(r.max_state_abs_diff for r in ok_rows))
    end
end

main()
