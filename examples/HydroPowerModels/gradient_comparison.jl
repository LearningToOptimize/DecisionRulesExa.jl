# gradient_comparison.jl
#
# Compare FD gradient of sequential rollout objective against envelope-theorem
# gradients under different penalty configurations.
#
# Methods:
#   1. FD ground truth: central differences on sequential rollout (no penalty)
#   2. DE uniform: envelope theorem with ρ_t = ρ_auto ∀t
#   3. DE early-low: ρ_t = ρ_auto · (t/T)
#   4. DE early-high: ρ_t = ρ_auto · (1 + α·(T-t)/(T-1))
#   5. DE discounted: ρ_t = ρ_auto · γ^(t-1)
#   6. Embedded TSDDR: envelope theorem from embedded NLP
#
# All ExaModels methods on GPU. Timing data collected throughout.

using DecisionRulesExa
using ExaModels
using Flux
using MadNLP, MadNLPGPU
using CUDA, CUDSS, KernelAbstractions
using Statistics, Random, Printf, LinearAlgebra
using Zygote
using JLD2

@assert CUDA.functional() "CUDA not available!"
@info "GPU: $(CUDA.name(CUDA.device())) — $(round(CUDA.total_memory() / 1e9; digits=1)) GB"

const SCRIPT_DIR = dirname(@__FILE__)
include(joinpath(SCRIPT_DIR, "hydro_power_data.jl"))
include(joinpath(SCRIPT_DIR, "hydro_power_exa.jl"))
include(joinpath(SCRIPT_DIR, "hydro_power_exa_embedded.jl"))

# ── Configuration ────────────────────────────────────────────────────────────

const CASE_NAME   = "bolivia"
const FORMULATION = :ac_polar
const T           = parse(Int, get(ENV, "DR_NUM_STAGES", "12"))
const N_DIRS      = parse(Int, get(ENV, "DR_N_DIRS", "50"))
const N_SCENARIOS = parse(Int, get(ENV, "DR_N_SCENARIOS", "3"))
const SCENARIO_OFFSET = parse(Int, get(ENV, "DR_SCENARIO_OFFSET", "0"))
const FD_EPS      = parse(Float64, get(ENV, "DR_FD_EPS", "1e-4"))
const ENABLE_SURROGATE_FD = parse(Bool, get(ENV, "DR_ENABLE_SURROGATE_FD", "false"))
const POLICY_PATH = get(ENV, "DR_POLICY_PATH", "")
const PHASE_LABEL = get(ENV, "DR_PHASE_LABEL", isempty(POLICY_PATH) ? "cold" : "warm_loaded")
const RESULT_DIR  = get(ENV, "DR_RESULT_DIR", joinpath(SCRIPT_DIR, "results", "gradient_quality"))
const DEFICIT_COST = 1e5
const load_scaler  = 0.6
const F = Float32

const EARLY_HIGH_ALPHAS = [2.0, 5.0, 10.0]
const DISCOUNT_GAMMAS   = [0.95, 0.99]

const TARGET_PEN_ARG = :auto
const SOLVER_KWARGS  = (print_level = MadNLP.ERROR, tol = 1e-6, max_iter = 9000)

@info "Config: phase=$PHASE_LABEL  T=$T  N_DIRS=$N_DIRS  N_SCENARIOS=$N_SCENARIOS  offset=$SCENARIO_OFFSET  ε=$FD_EPS  surrogate_fd=$ENABLE_SURROGATE_FD"
@info "Result dir: $RESULT_DIR"

# ── Load data ────────────────────────────────────────────────────────────────

const CASE_DIR = joinpath(SCRIPT_DIR, CASE_NAME)
power_data = load_power_data(joinpath(CASE_DIR, "PowerModels.json"))
hydro_data = load_hydro_data(joinpath(CASE_DIR, "hydro.json"),
                              joinpath(CASE_DIR, "inflows.csv"),
                              power_data; num_stages=1260)
nHyd = hydro_data.nHyd

demand_file = joinpath(CASE_DIR, "demand.csv")
demand_mat = isfile(demand_file) ? load_demand(demand_file, power_data; T = T) : nothing

backend = CUDA.CUDABackend()

x0_init = F.([clamp(hydro_data.initial_volumes[r],
                     hydro_data.units[r].min_vol,
                     hydro_data.units[r].max_vol)
              for r in 1:nHyd])
const _min_vols = Float64.([h.min_vol for h in hydro_data.units])
const _max_vols = Float64.([h.max_vol for h in hydro_data.units])

# ── Policy ───────────────────────────────────────────────────────────────────

Random.seed!(42)
policy = bounded_state_policy(nHyd, F.(_min_vols), F.(_max_vols), [128, 128];
                              activation=sigmoid, encoder_type=Flux.LSTM,
                              active_mask=trues(nHyd))
if !isempty(POLICY_PATH)
    @info "Loading policy checkpoint from $(POLICY_PATH)"
    load_stateconditioned_policy!(policy, JLD2.load(POLICY_PATH, "model_state"))
end
policy = CUDA.cu(policy)
x0_init = CUDA.cu(x0_init)
@info "Policy on GPU  params=$(sum(length, Flux.trainables(policy)))"

params_vec, re = Flux.destructure(policy)
n_params = length(params_vec)
@info "Flat parameter vector: $n_params elements on $(typeof(params_vec))"

# ── Build GPU DEs ────────────────────────────────────────────────────────────

@info "Building T=$T multi-stage GPU DE..."
t0 = time()
prob_de = build_hydro_de(power_data, hydro_data, T;
    backend = backend, float_type = Float64, formulation = FORMULATION,
    target_penalty = TARGET_PEN_ARG, deficit_cost = DEFICIT_COST,
    demand_matrix = demand_mat, load_scaler = load_scaler)
@info "  DE built in $(round(time()-t0; digits=1))s"

resolved_pen = prob_de.base_penalty_half * 2
resolved_pen_l1 = prob_de.base_penalty_l1
@info "  ρ_auto=$(round(resolved_pen; digits=2))  ρ_l1=$(round(resolved_pen_l1; digits=2))"

@info "Building T=$T embedded GPU DE..."
t0 = time()
prob_emb = build_embedded_hydro_de(policy, power_data, hydro_data, T;
    backend = backend, formulation = FORMULATION,
    target_penalty = TARGET_PEN_ARG, deficit_cost = DEFICIT_COST,
    demand_matrix = demand_mat, load_scaler = load_scaler)
@info "  Embedded DE built in $(round(time()-t0; digits=1))s"

@info "Building 1-stage GPU DE for rollout..."
stage_demand = demand_mat === nothing ? nothing : demand_mat[1:1, :]
t0 = time()
rollout_prob = build_hydro_de(power_data, hydro_data, 1;
    backend = backend, float_type = Float64, formulation = FORMULATION,
    target_penalty = TARGET_PEN_ARG, deficit_cost = DEFICIT_COST,
    demand_matrix = stage_demand, load_scaler = load_scaler)
@info "  Rollout DE built in $(round(time()-t0; digits=1))s"

# ── Warm-up training ────────────────────────────────────────────────────────
#    Train embedded TSDDR for a few iterations so the policy produces
#    targets that lead to feasible single-stage OPF rollouts.

const WARMUP_ITERS = parse(Int, get(ENV, "DR_WARMUP_ITERS", "20"))

function warmup_training!(policy_wu, prob_emb_wu, x0_wu, hydro_data_wu, T_wu, nHyd_wu, n_iters)
    opt = Flux.setup(Flux.Adam(1f-3), policy_wu)
    solver_warmup = MadNLP.MadNLPSolver(prob_emb_wu.model; SOLVER_KWARGS...)
    n_success = 0
    t_warmup = time()
    for iter in 1:n_iters
        w_wu = Float64.(sample_scenario(hydro_data_wu, T_wu))
        set_x0!(prob_emb_wu, x0_wu)
        set_inflows!(prob_emb_wu, w_wu)
        solver_warmup.cnt.k = 0
        solver_warmup.cnt.acceptable_cnt = 0
        solver_warmup.cnt.start_time = time()
        res_wu = MadNLP.solve!(solver_warmup)
        if !DecisionRulesExa.solve_succeeded(res_wu)
            continue
        end
        n_success += 1
        λ_wu = res_wu.multipliers[prob_emb_wu.target_con_range]
        x_sol_wu = embedded_hydro_realized_states(prob_emb_wu, res_wu)
        λf = CUDA.cu(F.(λ_wu))
        xf = CUDA.cu(F.(x_sol_wu))
        w_dev = CUDA.cu(F.(w_wu))
        nx = prob_emb_wu.nx
        gs = Zygote.gradient(policy_wu) do m
            total = zero(F)
            Flux.reset!(m)
            for t in 1:T_wu
                wt = w_dev[(t-1)*nHyd_wu+1:t*nHyd_wu]
                x_prev = (t == 1) ? F.(x0_wu) : xf[(t-2)*nx+1:(t-1)*nx]
                xt = m(vcat(wt, x_prev))
                total = total + sum(λf[(t-1)*nx+1:t*nx] .* xt)
            end
            total
        end
        if gs[1] !== nothing && all(isfinite, Flux.destructure(gs[1])[1])
            Flux.update!(opt, policy_wu, gs[1])
            invalidate_policy_cache!(prob_emb_wu)
        end
    end
    @info "  Warm-up done: $n_success/$n_iters successful solves in $(round(time()-t_warmup; digits=1))s"
    return n_success
end

if WARMUP_ITERS > 0
    @info "Warm-up training: $WARMUP_ITERS embedded iterations..."
    warmup_training!(policy, prob_emb, x0_init, hydro_data, T, nHyd, WARMUP_ITERS)
    params_vec, re = Flux.destructure(policy)
    n_params = length(params_vec)
end

# ── Rollout helpers ──────────────────────────────────────────────────────────

function set_hydro_rollout_stage!(stage_prob, state_in, wt, target, stage)
    ExaModels.set_parameter!(stage_prob.core, stage_prob.p_x0, state_in)
    ExaModels.set_parameter!(stage_prob.core, stage_prob.p_inflow, wt)
    if demand_mat !== nothing
        set_demand!(stage_prob, load_scaler .* demand_mat[stage:stage, :])
    end
    ExaModels.set_parameter!(stage_prob.core, stage_prob.p_target, target)
    return stage_prob
end

const _min_vols_dev = CUDA.cu(_min_vols)
const _max_vols_dev = CUDA.cu(_max_vols)

hydro_realized_state(stage_prob, result) =
    hydro_solution(stage_prob, result).reservoir[:, end]

function hydro_objective_no_target_penalty(stage_prob, result)
    sol = hydro_solution(stage_prob, result)
    delta = sol.delta
    ρ_half = resolved_pen / 2
    penalty_l2_cost = ρ_half * sum(abs2, delta)
    penalty_l1_cost = resolved_pen_l1 * sum(abs, delta)
    return result.objective - penalty_l2_cost - penalty_l1_cost
end

function sequential_rollout_objective(model_local, w_flat, x0)
    result = rollout_tsddr(
        model_local, x0, rollout_prob, w_flat;
        horizon = T, n_uncertainty = nHyd,
        set_stage_parameters! = set_hydro_rollout_stage!,
        realized_state = hydro_realized_state,
        objective_no_target_penalty = hydro_objective_no_target_penalty,
        madnlp_kwargs = SOLVER_KWARGS,
        warmstart = false,
        policy_state = :target,
        state_bounds = (_min_vols_dev, _max_vols_dev),
    )
    return result === nothing ? NaN : result.objective_no_target_penalty
end

# ── Envelope-theorem gradient (flat) ─────────────────────────────────────────

function envelope_gradient_flat(θ_flat, re_fn, w_flat, λ_flat, x0, T_loc, nx;
                                policy_input_states = nothing)
    w_dev = isa(w_flat, CUDA.CuArray) ? w_flat : CUDA.cu(F.(w_flat))
    λf    = F.(λ_flat)
    input_states = policy_input_states === nothing ? nothing : CUDA.cu(F.(policy_input_states))
    gs = Zygote.gradient(θ_flat) do θ
        m = re_fn(θ)
        Flux.reset!(m)
        total = zero(F)
        prev = F.(x0)
        for t in 1:T_loc
            wt = w_dev[(t-1)*nx+1 : t*nx]
            if input_states !== nothing
                prev = input_states[:, t]
            end
            xt = m(vcat(wt, prev))
            total = total + sum(λf[(t-1)*nx+1 : t*nx] .* xt)
            if input_states === nothing
                prev = xt
            end
        end
        total
    end
    return gs[1]
end

# ── DE solve + extract multipliers ───────────────────────────────────────────

function de_solve_and_multipliers(model_local, w_flat, x0, prob)
    Flux.reset!(model_local)
    prev = F.(x0)
    w_dev = isa(w_flat, CUDA.CuArray) ? w_flat : CUDA.cu(F.(w_flat))
    xhat_stages = AbstractVector{F}[]
    for t in 1:T
        wt = w_dev[(t-1)*nHyd+1 : t*nHyd]
        push!(xhat_stages, model_local(vcat(wt, prev)))
        prev = xhat_stages[end]
    end
    xhat_flat = vcat(xhat_stages...)

    ExaModels.set_parameter!(prob.core, prob.p_x0, x0)
    ExaModels.set_parameter!(prob.core, prob.p_inflow, w_flat)
    ExaModels.set_parameter!(prob.core, prob.p_target, Float64.(xhat_flat))
    result = MadNLP.madnlp(prob.model; SOLVER_KWARGS...)
    if !solve_succeeded(result)
        @warn "DE solve did not converge" status = result.status objective = result.objective
        return nothing
    end
    if !isfinite(result.objective)
        @warn "DE solve returned non-finite objective" status = result.status objective = result.objective
        return nothing
    end
    λ = result.multipliers[prob.target_con_range]
    all(isfinite, λ) || return nothing
    sol = hydro_solution(prob, result)
    target_matrix = hcat([Array(Float64.(x)) for x in xhat_stages]...)
    return (lambda = λ, objective = result.objective, delta = sol.delta,
            target_matrix = target_matrix, reservoir = sol.reservoir)
end

# ── Penalty configs ──────────────────────────────────────────────────────────

struct PenaltyConfig
    name::String
    penalty_half::Vector{Float64}
    penalty_l1::Vector{Float64}
end

function make_penalty_configs(ρ_auto, ρ_l1_auto, T_loc, nH)
    configs = PenaltyConfig[]

    ρ_half = ρ_auto / 2
    push!(configs, PenaltyConfig("uniform",
        fill(ρ_half, T_loc * nH),
        fill(ρ_l1_auto, T_loc * nH)))

    early_low_half = [ρ_half * (t / T_loc) for t in 1:T_loc for _ in 1:nH]
    early_low_l1   = [ρ_l1_auto * (t / T_loc) for t in 1:T_loc for _ in 1:nH]
    push!(configs, PenaltyConfig("early_low",
        early_low_half, early_low_l1))

    for α in EARLY_HIGH_ALPHAS
        vals_half = [ρ_half * (1 + α * (T_loc - t) / max(T_loc - 1, 1))
                     for t in 1:T_loc for _ in 1:nH]
        vals_l1   = [ρ_l1_auto * (1 + α * (T_loc - t) / max(T_loc - 1, 1))
                     for t in 1:T_loc for _ in 1:nH]
        push!(configs, PenaltyConfig("early_high_α=$α",
            vals_half, vals_l1))
    end

    for γ in DISCOUNT_GAMMAS
        vals_half = [ρ_half * γ^(t-1) for t in 1:T_loc for _ in 1:nH]
        vals_l1   = [ρ_l1_auto * γ^(t-1) for t in 1:T_loc for _ in 1:nH]
        push!(configs, PenaltyConfig("discounted_γ=$γ",
            vals_half, vals_l1))
    end

    return configs
end

penalty_configs = make_penalty_configs(resolved_pen, resolved_pen_l1, T, nHyd)
@info "Penalty configs: $(length(penalty_configs))"
for pc in penalty_configs
    @info "  $(pc.name): ρ/2 range [$(round(minimum(pc.penalty_half);digits=2)), $(round(maximum(pc.penalty_half);digits=2))]"
end

# ── Embedded solve + multipliers ─────────────────────────────────────────────

function embedded_solve_and_multipliers(w_flat, x0)
    set_x0!(prob_emb, x0)
    set_inflows!(prob_emb, w_flat)
    result = MadNLP.madnlp(prob_emb.model; SOLVER_KWARGS...)
    if !solve_succeeded(result)
        @warn "Embedded solve did not converge" status = result.status objective = result.objective
        return nothing
    end
    if !isfinite(result.objective)
        @warn "Embedded solve returned non-finite objective" status = result.status objective = result.objective
        return nothing
    end
    λ = result.multipliers[prob_emb.target_con_range]
    all(isfinite, λ) || return nothing
    sol = hydro_solution(prob_emb, result)
    target_matrix = embedded_target_matrix(w_flat, x0, sol)
    return (lambda = λ, objective = result.objective, delta = sol.delta,
            target_matrix = target_matrix, reservoir = sol.reservoir)
end

function embedded_target_matrix(w_flat, x0, sol)
    w_dev = isa(w_flat, CUDA.CuArray) ? w_flat : CUDA.cu(F.(w_flat))
    targets = Matrix{Float64}(undef, nHyd, T)
    Flux.reset!(policy)
    for t in 1:T
        wt = w_dev[(t-1)*nHyd+1 : t*nHyd]
        prev = t == 1 ? F.(x0) : F.(sol.reservoir[:, t])
        targets[:, t] .= Array(Float64.(policy(vcat(wt, prev))))
    end
    return targets
end

function target_violation_stats(delta, target_matrix; penalty_half = nothing,
                                penalty_l1 = nothing, state_range = nothing,
                                active_state_mask = nothing,
                                early_frac = 0.25)
    delta_cpu = Array(delta)
    abs_delta = abs.(delta_cpu)
    T_loc = size(abs_delta, 2)
    early_T = max(1, ceil(Int, early_frac * T_loc))
    stage_mean = vec(mean(abs_delta; dims = 1))
    stage_rel = [
        norm(delta_cpu[:, t]) / max(norm(target_matrix[:, t]), 1e-8)
        for t in 1:T_loc
    ]
    if state_range === nothing
        stage_range_rel = fill(NaN, T_loc)
    else
        range_scale = max.(Float64.(state_range), 1e-8)
        if active_state_mask === nothing
            active = trues(length(range_scale))
        else
            active = Bool.(active_state_mask)
        end
        if any(active)
            range_rel = abs_delta[active, :] ./ range_scale[active]
        else
            range_rel = fill(NaN, 1, T_loc)
        end
        stage_range_rel = vec(mean(range_rel; dims = 1))
    end
    if penalty_half === nothing || penalty_l1 === nothing
        stage_penalty = fill(NaN, T_loc)
    else
        ρh = reshape(collect(penalty_half), nHyd, T_loc)
        ρ1 = reshape(collect(penalty_l1), nHyd, T_loc)
        stage_penalty = [
            sum(ρh[:, t] .* abs2.(delta_cpu[:, t]) .+
                ρ1[:, t] .* abs_delta[:, t])
            for t in 1:T_loc
        ]
    end
    return (
        mean_abs = mean(abs_delta),
        max_abs = maximum(abs_delta),
        early_mean_abs = mean(abs_delta[:, 1:early_T]),
        mean_rel = mean(stage_rel),
        early_mean_rel = mean(stage_rel[1:early_T]),
        mean_range_rel = mean(stage_range_rel),
        early_mean_range_rel = mean(stage_range_rel[1:early_T]),
        total_penalty_cost = sum(stage_penalty),
        stage_mean = stage_mean,
        stage_rel = stage_rel,
        stage_range_rel = stage_range_rel,
        stage_penalty = stage_penalty,
    )
end

function de_surrogate_objective(model_local, w_flat, x0, prob)
    sol = de_solve_and_multipliers(model_local, w_flat, x0, prob)
    return sol === nothing ? NaN : sol.objective
end

function de_surrogate_fd_projections(θ_flat, re_fn, directions, w_flat, x0, prob, eps)
    projections = fill(NaN, length(directions))
    obj_base_cache = Ref(NaN)
    for (di, d) in enumerate(directions)
        θ_plus  = θ_flat .+ F(eps) .* d
        θ_minus = θ_flat .- F(eps) .* d
        obj_plus  = de_surrogate_objective(re_fn(θ_plus),  w_flat, x0, prob)
        obj_minus = de_surrogate_objective(re_fn(θ_minus), w_flat, x0, prob)

        if isfinite(obj_plus) && isfinite(obj_minus)
            projections[di] = (obj_plus - obj_minus) / (2 * eps)
        elseif isfinite(obj_plus)
            if isnan(obj_base_cache[])
                obj_base_cache[] = de_surrogate_objective(re_fn(θ_flat), w_flat, x0, prob)
            end
            projections[di] = isfinite(obj_base_cache[]) ? (obj_plus - obj_base_cache[]) / eps : NaN
        elseif isfinite(obj_minus)
            if isnan(obj_base_cache[])
                obj_base_cache[] = de_surrogate_objective(re_fn(θ_flat), w_flat, x0, prob)
            end
            projections[di] = isfinite(obj_base_cache[]) ? (obj_base_cache[] - obj_minus) / eps : NaN
        end
    end
    return projections
end

function projection_metrics(fd_projections, method_projections)
    valid_mask = isfinite.(fd_projections) .& isfinite.(method_projections)
    nv = count(valid_mask)
    nv < 3 && return nothing

    fd_v = fd_projections[valid_mask]
    me_v = method_projections[valid_mask]
    cos_sim = dot(fd_v, me_v) / (norm(fd_v) * norm(me_v) + 1e-30)
    fd_norm = norm(fd_v)
    me_norm = norm(me_v)
    mag_ratio = me_norm / (fd_norm + 1e-30)
    nrmse = sqrt(mean(abs2, me_v .- fd_v)) / (sqrt(mean(abs2, fd_v)) + 1e-30)
    scale_log10_err = abs(log10((me_norm + 1e-30) / (fd_norm + 1e-30)))
    sign_ag = count(sign.(fd_v) .== sign.(me_v)) / nv
    sign_flip = count(sign.(fd_v) .== sign.(-me_v)) / nv
    return (
        n_valid = nv,
        cos = cos_sim,
        cos_flip = -cos_sim,
        mag_ratio = mag_ratio,
        nrmse = nrmse,
        scale_log10_err = scale_log10_err,
        sign = sign_ag,
        sign_flip = sign_flip,
    )
end

finite_values(x) = filter(isfinite, x)
mean_or_nan(x) = isempty(x) ? NaN : mean(x)
median_or_nan(x) = isempty(x) ? NaN : quantile(x, 0.50)
q_or_nan(x, q) = isempty(x) ? NaN : quantile(x, q)
frac_or_nan(x, pred) = isempty(x) ? NaN : count(pred, x) / length(x)

# ── Random directions ────────────────────────────────────────────────────────

Random.seed!(12345)
directions_cpu = [let d = randn(F, n_params); d ./ norm(d) end for _ in 1:N_DIRS]
directions = [CUDA.cu(d) for d in directions_cpu]
@info "Sampled $N_DIRS random unit directions"

# ── Scenarios ────────────────────────────────────────────────────────────────

Random.seed!(9999)
scenario_pool = [sample_scenario(hydro_data, T) for _ in 1:(SCENARIO_OFFSET + N_SCENARIOS)]
scenarios = scenario_pool[(SCENARIO_OFFSET + 1):end]
N_SCENARIOS_ACTUAL = length(scenarios)
@info "Sampled $N_SCENARIOS_ACTUAL scenarios after skipping offset=$SCENARIO_OFFSET"

# ── Main experiment ──────────────────────────────────────────────────────────

method_names = [pc.name for pc in penalty_configs]
push!(method_names, "embedded")
n_methods = length(method_names)

all_cosines     = zeros(N_SCENARIOS_ACTUAL, n_methods)
all_cosines_flip = zeros(N_SCENARIOS_ACTUAL, n_methods)
all_mag_ratios  = zeros(N_SCENARIOS_ACTUAL, n_methods)
all_nrmse       = fill(NaN, N_SCENARIOS_ACTUAL, n_methods)
all_scale_log10_err = fill(NaN, N_SCENARIOS_ACTUAL, n_methods)
all_sign_agree  = zeros(N_SCENARIOS_ACTUAL, n_methods)
all_sign_agree_flip = zeros(N_SCENARIOS_ACTUAL, n_methods)
all_fd_times    = zeros(N_SCENARIOS_ACTUAL)
all_de_times    = zeros(N_SCENARIOS_ACTUAL, n_methods)
all_mean_viols  = fill(NaN, N_SCENARIOS_ACTUAL, n_methods)
all_max_viols   = fill(NaN, N_SCENARIOS_ACTUAL, n_methods)
all_early_viols = fill(NaN, N_SCENARIOS_ACTUAL, n_methods)
all_mean_rel_leaks  = fill(NaN, N_SCENARIOS_ACTUAL, n_methods)
all_early_rel_leaks = fill(NaN, N_SCENARIOS_ACTUAL, n_methods)
all_mean_range_rel_leaks  = fill(NaN, N_SCENARIOS_ACTUAL, n_methods)
all_early_range_rel_leaks = fill(NaN, N_SCENARIOS_ACTUAL, n_methods)
all_penalty_costs   = fill(NaN, N_SCENARIOS_ACTUAL, n_methods)
all_stage_mean_viols = fill(NaN, N_SCENARIOS_ACTUAL, n_methods, T)
all_surrogate_cosines = fill(NaN, N_SCENARIOS_ACTUAL, n_methods)
all_surrogate_cosines_flip = fill(NaN, N_SCENARIOS_ACTUAL, n_methods)
all_surrogate_nrmse = fill(NaN, N_SCENARIOS_ACTUAL, n_methods)
all_surrogate_scale_log10_err = fill(NaN, N_SCENARIOS_ACTUAL, n_methods)
all_surrogate_sign_agree = fill(NaN, N_SCENARIOS_ACTUAL, n_methods)
all_surrogate_sign_agree_flip = fill(NaN, N_SCENARIOS_ACTUAL, n_methods)

state_ranges = max.(_max_vols .- _min_vols, 1e-8)
active_state_mask = (_max_vols .- _min_vols) .> 1e-8

@info "\n" * "="^70
@info "STARTING GRADIENT COMPARISON"
@info "="^70

for (si, w_flat) in enumerate(scenarios)
    @info "\n--- Scenario $si/$N_SCENARIOS_ACTUAL ---"

    # ── Step 1: FD ground truth ──────────────────────────────────────────
    @info "  Computing FD ground truth ($N_DIRS directions, ε=$FD_EPS)..."
    fd_projections = zeros(Float64, N_DIRS)
    t_fd_start = time()

    obj_base_cache = Ref(NaN)

    for (di, d) in enumerate(directions)
        θ_plus  = params_vec .+ F(FD_EPS) .* d
        θ_minus = params_vec .- F(FD_EPS) .* d

        m_plus  = re(θ_plus)
        m_minus = re(θ_minus)

        obj_plus  = sequential_rollout_objective(m_plus,  w_flat, x0_init)
        obj_minus = sequential_rollout_objective(m_minus, w_flat, x0_init)

        if isfinite(obj_plus) && isfinite(obj_minus)
            fd_projections[di] = (obj_plus - obj_minus) / (2 * FD_EPS)
        elseif isfinite(obj_plus)
            if isnan(obj_base_cache[])
                obj_base_cache[] = sequential_rollout_objective(re(params_vec), w_flat, x0_init)
            end
            if isfinite(obj_base_cache[])
                fd_projections[di] = (obj_plus - obj_base_cache[]) / FD_EPS
            else
                fd_projections[di] = NaN
            end
        elseif isfinite(obj_minus)
            if isnan(obj_base_cache[])
                obj_base_cache[] = sequential_rollout_objective(re(params_vec), w_flat, x0_init)
            end
            if isfinite(obj_base_cache[])
                fd_projections[di] = (obj_base_cache[] - obj_minus) / FD_EPS
            else
                fd_projections[di] = NaN
            end
        else
            fd_projections[di] = NaN
            @warn "  FD direction $di: both sides non-finite"
        end

        if di % 10 == 0
            elapsed = time() - t_fd_start
            @info "    FD direction $di/$N_DIRS  elapsed=$(round(elapsed; digits=1))s"
        end
    end

    t_fd = time() - t_fd_start
    all_fd_times[si] = t_fd
    fd_valid = count(isfinite, fd_projections)
    fd_norm  = norm(fd_projections[isfinite.(fd_projections)])
    @info "  FD done: $(round(t_fd; digits=1))s  valid=$fd_valid/$N_DIRS  ‖g_FD‖≈$(round(fd_norm; digits=4))"

    min_valid_fd = min(N_DIRS, max(N_DIRS ÷ 4, 5))
    if fd_valid < min_valid_fd
        @warn "  Too few valid FD directions ($fd_valid) — skipping scenario"
        all_cosines[si, :] .= NaN
        all_cosines_flip[si, :] .= NaN
        all_mag_ratios[si, :] .= NaN
        all_sign_agree[si, :] .= NaN
        all_sign_agree_flip[si, :] .= NaN
        continue
    end

    # ── Step 2: Envelope-theorem gradients for each penalty config ───────

    for (ci, pc) in enumerate(penalty_configs)
        @info "  Method: $(pc.name)..."
        t_method_start = time()

        ExaModels.set_parameter!(prob_de.core, prob_de.p_penalty_half, pc.penalty_half)
        ExaModels.set_parameter!(prob_de.core, prob_de.p_penalty_l1, pc.penalty_l1)

        sol = de_solve_and_multipliers(policy, w_flat, x0_init, prob_de)
        if sol === nothing
            @warn "    DE solve failed"
            all_cosines[si, ci] = NaN
            all_cosines_flip[si, ci] = NaN
            all_mag_ratios[si, ci] = NaN
            all_sign_agree[si, ci] = NaN
            all_sign_agree_flip[si, ci] = NaN
            all_de_times[si, ci] = time() - t_method_start
            continue
        end

        g_flat = envelope_gradient_flat(params_vec, re, w_flat, sol.lambda, x0_init, T, nHyd)
        if g_flat === nothing
            @warn "    Gradient is nothing"
            all_cosines[si, ci] = NaN
            all_cosines_flip[si, ci] = NaN
            all_mag_ratios[si, ci] = NaN
            all_sign_agree[si, ci] = NaN
            all_sign_agree_flip[si, ci] = NaN
            all_de_times[si, ci] = time() - t_method_start
            continue
        end

        method_projections = Float64[dot(g_flat, d) for d in directions]
        viol = target_violation_stats(sol.delta, sol.target_matrix;
            penalty_half = pc.penalty_half, penalty_l1 = pc.penalty_l1,
            state_range = state_ranges, active_state_mask = active_state_mask)
        all_mean_viols[si, ci] = viol.mean_abs
        all_max_viols[si, ci] = viol.max_abs
        all_early_viols[si, ci] = viol.early_mean_abs
        all_mean_rel_leaks[si, ci] = viol.mean_rel
        all_early_rel_leaks[si, ci] = viol.early_mean_rel
        all_mean_range_rel_leaks[si, ci] = viol.mean_range_rel
        all_early_range_rel_leaks[si, ci] = viol.early_mean_range_rel
        all_penalty_costs[si, ci] = viol.total_penalty_cost
        all_stage_mean_viols[si, ci, :] .= viol.stage_mean

        metrics = projection_metrics(fd_projections, method_projections)
        if metrics === nothing
            @warn "    Too few valid projections"
            all_cosines[si, ci] = NaN
            all_cosines_flip[si, ci] = NaN
            all_mag_ratios[si, ci] = NaN
            all_sign_agree[si, ci] = NaN
            all_sign_agree_flip[si, ci] = NaN
        else
            all_cosines[si, ci] = metrics.cos
            all_cosines_flip[si, ci] = metrics.cos_flip
            all_mag_ratios[si, ci] = metrics.mag_ratio
            all_nrmse[si, ci] = metrics.nrmse
            all_scale_log10_err[si, ci] = metrics.scale_log10_err
            all_sign_agree[si, ci] = metrics.sign
            all_sign_agree_flip[si, ci] = metrics.sign_flip
        end

        if ENABLE_SURROGATE_FD
            sur_fd = de_surrogate_fd_projections(params_vec, re, directions, w_flat, x0_init, prob_de, FD_EPS)
            sur_metrics = projection_metrics(sur_fd, method_projections)
            if sur_metrics !== nothing
                all_surrogate_cosines[si, ci] = sur_metrics.cos
                all_surrogate_cosines_flip[si, ci] = sur_metrics.cos_flip
                all_surrogate_nrmse[si, ci] = sur_metrics.nrmse
                all_surrogate_scale_log10_err[si, ci] = sur_metrics.scale_log10_err
                all_surrogate_sign_agree[si, ci] = sur_metrics.sign
                all_surrogate_sign_agree_flip[si, ci] = sur_metrics.sign_flip
            end
        end

        t_method = time() - t_method_start
        all_de_times[si, ci] = t_method
        @info @sprintf("    cos=%.4f  cos_flip=%.4f  nrmse=%.4f  scale_log10=%.4f  mag_ratio=%.4f  sign=%.2f%%  sign_flip=%.2f%%  viol_mean=%.4g  viol_early=%.4g  rel=%.4g  rel_early=%.4g  range_rel=%.4g  range_rel_early=%.4g  penalty=%.4g  viol_max=%.4g  time=%.1fs  obj=%.2f",
            all_cosines[si, ci], all_cosines_flip[si, ci],
            all_nrmse[si, ci], all_scale_log10_err[si, ci],
            all_mag_ratios[si, ci], all_sign_agree[si, ci]*100,
            all_sign_agree_flip[si, ci]*100,
            viol.mean_abs, viol.early_mean_abs, viol.mean_rel, viol.early_mean_rel,
            viol.mean_range_rel, viol.early_mean_range_rel,
            viol.total_penalty_cost, viol.max_abs, t_method, sol.objective)
        if ENABLE_SURROGATE_FD
            @info @sprintf("    surrogate_fd: cos=%.4f  cos_flip=%.4f  nrmse=%.4f  scale_log10=%.4f  sign=%.2f%%  sign_flip=%.2f%%",
                all_surrogate_cosines[si, ci], all_surrogate_cosines_flip[si, ci],
                all_surrogate_nrmse[si, ci], all_surrogate_scale_log10_err[si, ci],
                all_surrogate_sign_agree[si, ci]*100,
                all_surrogate_sign_agree_flip[si, ci]*100)
        end
    end

    # ── Step 3: Embedded TSDDR gradient ──────────────────────────────────

    ci_emb = n_methods
    @info "  Method: embedded..."
    t_emb_start = time()

    sol_emb = embedded_solve_and_multipliers(w_flat, x0_init)
    if sol_emb === nothing
        @warn "    Embedded solve failed"
        all_cosines[si, ci_emb] = NaN
        all_cosines_flip[si, ci_emb] = NaN
        all_mag_ratios[si, ci_emb] = NaN
        all_sign_agree[si, ci_emb] = NaN
        all_sign_agree_flip[si, ci_emb] = NaN
        all_de_times[si, ci_emb] = time() - t_emb_start
    else
        embedded_policy_input_states = Array(sol_emb.reservoir[:, 1:T])
        g_flat_emb = envelope_gradient_flat(
            params_vec, re, w_flat, sol_emb.lambda, x0_init, T, nHyd;
            policy_input_states = embedded_policy_input_states,
        )
        if g_flat_emb === nothing
            @warn "    Embedded gradient is nothing"
            all_cosines[si, ci_emb] = NaN
            all_cosines_flip[si, ci_emb] = NaN
            all_mag_ratios[si, ci_emb] = NaN
            all_sign_agree[si, ci_emb] = NaN
            all_sign_agree_flip[si, ci_emb] = NaN
        else
            emb_projections = Float64[dot(g_flat_emb, d) for d in directions]
            emb_penalty_half = fill(resolved_pen / 2, T * nHyd)
            emb_penalty_l1 = fill(resolved_pen_l1, T * nHyd)
            viol = target_violation_stats(sol_emb.delta, sol_emb.target_matrix;
                penalty_half = emb_penalty_half, penalty_l1 = emb_penalty_l1,
                state_range = state_ranges, active_state_mask = active_state_mask)
            all_mean_viols[si, ci_emb] = viol.mean_abs
            all_max_viols[si, ci_emb] = viol.max_abs
            all_early_viols[si, ci_emb] = viol.early_mean_abs
            all_mean_rel_leaks[si, ci_emb] = viol.mean_rel
            all_early_rel_leaks[si, ci_emb] = viol.early_mean_rel
            all_mean_range_rel_leaks[si, ci_emb] = viol.mean_range_rel
            all_early_range_rel_leaks[si, ci_emb] = viol.early_mean_range_rel
            all_penalty_costs[si, ci_emb] = viol.total_penalty_cost
            all_stage_mean_viols[si, ci_emb, :] .= viol.stage_mean
            metrics = projection_metrics(fd_projections, emb_projections)
            if metrics === nothing
                all_cosines[si, ci_emb] = NaN
                all_cosines_flip[si, ci_emb] = NaN
                all_mag_ratios[si, ci_emb] = NaN
                all_sign_agree[si, ci_emb] = NaN
                all_sign_agree_flip[si, ci_emb] = NaN
            else
                all_cosines[si, ci_emb] = metrics.cos
                all_cosines_flip[si, ci_emb] = metrics.cos_flip
                all_mag_ratios[si, ci_emb] = metrics.mag_ratio
                all_nrmse[si, ci_emb] = metrics.nrmse
                all_scale_log10_err[si, ci_emb] = metrics.scale_log10_err
                all_sign_agree[si, ci_emb] = metrics.sign
                all_sign_agree_flip[si, ci_emb] = metrics.sign_flip
            end
        end
        t_emb = time() - t_emb_start
        all_de_times[si, ci_emb] = t_emb
        @info @sprintf("    cos=%.4f  cos_flip=%.4f  nrmse=%.4f  scale_log10=%.4f  mag_ratio=%.4f  sign=%.2f%%  sign_flip=%.2f%%  viol_mean=%.4g  viol_early=%.4g  rel=%.4g  rel_early=%.4g  range_rel=%.4g  range_rel_early=%.4g  penalty=%.4g  viol_max=%.4g  time=%.1fs  obj=%.2f",
            all_cosines[si, ci_emb], all_cosines_flip[si, ci_emb],
            all_nrmse[si, ci_emb], all_scale_log10_err[si, ci_emb],
            all_mag_ratios[si, ci_emb],
            all_sign_agree[si, ci_emb]*100,
            all_sign_agree_flip[si, ci_emb]*100,
            all_mean_viols[si, ci_emb], all_early_viols[si, ci_emb],
            all_mean_rel_leaks[si, ci_emb], all_early_rel_leaks[si, ci_emb],
            all_mean_range_rel_leaks[si, ci_emb], all_early_range_rel_leaks[si, ci_emb],
            all_penalty_costs[si, ci_emb], all_max_viols[si, ci_emb],
            t_emb, sol_emb.objective)
    end
end

# ── Results summary ──────────────────────────────────────────────────────────

@info "\n" * "="^70
@info "GRADIENT COMPARISON RESULTS"
@info "="^70

@info @sprintf("\n%-25s  %8s  %8s  %8s  %8s  %8s  %10s  %10s  %9s  %9s  %10s  %10s  %8s",
    "Method", "cos_mu", "cos_med", "cos_p10", "bad_%", "sign_%",
    "nrmse_med", "nrmse_p90", "scale_med", "scale_p90", "range_early", "fail_n", "time_s")
@info "-"^70

for (ci, name) in enumerate(method_names)
    cos_vals = finite_values(all_cosines[:, ci])
    nrmse_vals = finite_values(all_nrmse[:, ci])
    scale_vals = finite_values(all_scale_log10_err[:, ci])
    sig_vals = finite_values(all_sign_agree[:, ci])
    t_vals   = finite_values(all_de_times[:, ci])
    rre_vals = finite_values(all_early_range_rel_leaks[:, ci])

    cos_m = mean_or_nan(cos_vals)
    cos_med = median_or_nan(cos_vals)
    cos_p10 = q_or_nan(cos_vals, 0.10)
    bad_frac = frac_or_nan(cos_vals, <(0.0))
    nrmse_med = median_or_nan(nrmse_vals)
    nrmse_p90 = q_or_nan(nrmse_vals, 0.90)
    scale_med = median_or_nan(scale_vals)
    scale_p90 = q_or_nan(scale_vals, 0.90)
    sig_m = mean_or_nan(sig_vals)
    t_m   = mean_or_nan(t_vals)
    rre_m = mean_or_nan(rre_vals)
    fail_n = N_SCENARIOS_ACTUAL - length(cos_vals)

    @info @sprintf("%-25s  %8.4f  %8.4f  %8.4f  %7.1f%%  %7.1f%%  %10.4f  %10.4f  %9.4f  %9.4f  %10.4g  %8d  %7.1fs",
        name, cos_m, cos_med, cos_p10, bad_frac*100, sig_m*100,
        nrmse_med, nrmse_p90, scale_med, scale_p90, rre_m, fail_n, t_m)
end

if ENABLE_SURROGATE_FD
    @info "\n--- Same-surrogate FD check for explicit DE methods ---"
    @info @sprintf("%-25s  %8s  %8s  %10s  %10s  %8s  %8s",
        "Method", "sur_cos", "sur_p10", "sur_nrmse", "sur_scale", "sur_sign", "sur_bad")
    for (ci, name) in enumerate(method_names)
        name == "embedded" && continue
        sc_vals = filter(isfinite, all_surrogate_cosines[:, ci])
        sn_vals = filter(isfinite, all_surrogate_nrmse[:, ci])
        sl_vals = filter(isfinite, all_surrogate_scale_log10_err[:, ci])
        ss_vals = filter(isfinite, all_surrogate_sign_agree[:, ci])
        sc_m = isempty(sc_vals) ? NaN : mean(sc_vals)
        sc_p10 = q_or_nan(sc_vals, 0.10)
        sn_med = median_or_nan(sn_vals)
        sl_med = median_or_nan(sl_vals)
        ss_m = isempty(ss_vals) ? NaN : mean(ss_vals)
        bad_m = frac_or_nan(sc_vals, <(0.0))
        @info @sprintf("%-25s  %8.4f  %8.4f  %10.4f  %10.4f  %7.1f%%  %7.1f%%",
            name, sc_m, sc_p10, sn_med, sl_med, ss_m*100, bad_m*100)
    end
end

@info @sprintf("\nFD ground truth: mean time per scenario = %.1fs", mean(all_fd_times))
@info @sprintf("FD total solves per scenario = %d  (2 × %d dirs × %d stages)",
    2 * N_DIRS * T, N_DIRS, T)

# ── Per-scenario detail ──────────────────────────────────────────────────────

@info "\n--- Per-scenario cosine similarity ---"
@info @sprintf("%-25s  %s", "Method", join([@sprintf("  S%d", s) for s in 1:N_SCENARIOS_ACTUAL]))
for (ci, name) in enumerate(method_names)
    vals = [@sprintf("%.4f", all_cosines[s, ci]) for s in 1:N_SCENARIOS_ACTUAL]
    @info @sprintf("%-25s  %s", name, join(["  " * v for v in vals]))
end

@info "\n--- Per-scenario flipped-sign cosine similarity (-method gradient) ---"
@info @sprintf("%-25s  %s", "Method", join([@sprintf("  S%d", s) for s in 1:N_SCENARIOS_ACTUAL]))
for (ci, name) in enumerate(method_names)
    vals = [@sprintf("%.4f", all_cosines_flip[s, ci]) for s in 1:N_SCENARIOS_ACTUAL]
    @info @sprintf("%-25s  %s", name, join(["  " * v for v in vals]))
end

@info "\n--- Mean stagewise target violation |delta| ---"
stage_cols = join([@sprintf("  t%d", t) for t in 1:T])
@info @sprintf("%-25s  %s", "Method", stage_cols)
for (ci, name) in enumerate(method_names)
    vals = [
        let finite_vals = filter(isfinite, all_stage_mean_viols[:, ci, t])
            isempty(finite_vals) ? NaN : mean(finite_vals)
        end
        for t in 1:T
    ]
    valstr = [@sprintf("%.4g", v) for v in vals]
    @info @sprintf("%-25s  %s", name, join(["  " * v for v in valstr]))
end

mkpath(RESULT_DIR)
slurm_job = get(ENV, "SLURM_JOB_ID", "local")
slurm_task = get(ENV, "SLURM_ARRAY_TASK_ID", "0")
result_file = joinpath(
    RESULT_DIR,
    @sprintf("gradient_quality_%s_T%d_D%d_S%d_offset%d_job%s_task%s.jld2",
        replace(PHASE_LABEL, r"[^A-Za-z0-9_.=-]" => "_"),
        T, N_DIRS, N_SCENARIOS_ACTUAL, SCENARIO_OFFSET, slurm_job, slurm_task),
)
scenario_global_indices = collect((SCENARIO_OFFSET + 1):(SCENARIO_OFFSET + N_SCENARIOS_ACTUAL))
@save result_file PHASE_LABEL POLICY_PATH T N_DIRS N_SCENARIOS SCENARIO_OFFSET N_SCENARIOS_ACTUAL FD_EPS ENABLE_SURROGATE_FD CASE_NAME FORMULATION DEFICIT_COST load_scaler EARLY_HIGH_ALPHAS DISCOUNT_GAMMAS TARGET_PEN_ARG method_names scenario_global_indices all_fd_times all_de_times all_cosines all_cosines_flip all_mag_ratios all_nrmse all_scale_log10_err all_sign_agree all_sign_agree_flip all_mean_viols all_max_viols all_early_viols all_mean_rel_leaks all_early_rel_leaks all_mean_range_rel_leaks all_early_range_rel_leaks all_penalty_costs all_stage_mean_viols all_surrogate_cosines all_surrogate_cosines_flip all_surrogate_nrmse all_surrogate_scale_log10_err all_surrogate_sign_agree all_surrogate_sign_agree_flip
@info "Saved structured results: $result_file"

@info "\n" * "="^70
@info "EXPERIMENT COMPLETE"
@info "="^70
