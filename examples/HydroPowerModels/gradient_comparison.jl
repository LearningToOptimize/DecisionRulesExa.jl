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
const FD_EPS      = parse(Float64, get(ENV, "DR_FD_EPS", "1e-4"))
const DEFICIT_COST = 1e5
const load_scaler  = 0.6
const F = Float32

const EARLY_HIGH_ALPHAS = [2.0, 5.0, 10.0]
const DISCOUNT_GAMMAS   = [0.95, 0.99]

const TARGET_PEN_ARG = :auto
const SOLVER_KWARGS  = (print_level = MadNLP.ERROR, tol = 1e-6, max_iter = 9000)

@info "Config: T=$T  N_DIRS=$N_DIRS  N_SCENARIOS=$N_SCENARIOS  ε=$FD_EPS"

# ── Load data ────────────────────────────────────────────────────────────────

const CASE_DIR = joinpath(SCRIPT_DIR, CASE_NAME)
power_data = load_power_data(joinpath(CASE_DIR, "PowerModels.json"))
hydro_data = load_hydro_data(joinpath(CASE_DIR, "hydro.json"),
                              joinpath(CASE_DIR, "inflows.csv"),
                              power_data; num_stages=1260)
nHyd = hydro_data.nHyd

demand_file = joinpath(CASE_DIR, "demand.csv")
demand_mat = isfile(demand_file) ? load_demand_data(demand_file) : nothing
if demand_mat !== nothing && size(demand_mat, 1) < T
    demand_mat = repeat(demand_mat, cld(T, size(demand_mat, 1)), 1)[1:T, :]
end

backend = CUDA.CUDABackend()

x0_init = F.([clamp(hydro_data.initial_volumes[r],
                     hydro_data.units[r].min_vol,
                     hydro_data.units[r].max_vol)
              for r in 1:nHyd])

# ── Policy ───────────────────────────────────────────────────────────────────

Random.seed!(42)
policy = StateConditionedPolicy(nHyd, nHyd, nHyd, [128, 128];
                                activation=sigmoid, encoder_type=Flux.LSTM)
policy = Flux.gpu(policy)
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
    )
    return result === nothing ? NaN : result.objective_no_target_penalty
end

# ── Envelope-theorem gradient (flat) ─────────────────────────────────────────

function envelope_gradient_flat(θ_flat, re_fn, w_flat, λ_flat, x0, T_loc, nx)
    w_dev = isa(w_flat, CUDA.CuArray) ? w_flat : CUDA.cu(F.(w_flat))
    λf    = F.(λ_flat)
    gs = Zygote.gradient(θ_flat) do θ
        m = re_fn(θ)
        Flux.reset!(m)
        total = zero(F)
        prev = F.(x0)
        for t in 1:T_loc
            wt = w_dev[(t-1)*nx+1 : t*nx]
            xt = m(vcat(wt, prev))
            total = total + sum(λf[(t-1)*nx+1 : t*nx] .* xt)
            prev = xt
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
    solve_succeeded(result) || return nothing
    isfinite(result.objective) || return nothing
    λ = result.multipliers[prob.target_con_range]
    all(isfinite, λ) || return nothing
    return (lambda = λ, objective = result.objective)
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
    solve_succeeded(result) || return nothing
    isfinite(result.objective) || return nothing
    λ = result.multipliers[prob_emb.target_con_range]
    all(isfinite, λ) || return nothing
    return (lambda = λ, objective = result.objective)
end

# ── Random directions ────────────────────────────────────────────────────────

Random.seed!(12345)
directions_cpu = [let d = randn(F, n_params); d ./ norm(d) end for _ in 1:N_DIRS]
directions = [CUDA.cu(d) for d in directions_cpu]
@info "Sampled $N_DIRS random unit directions"

# ── Scenarios ────────────────────────────────────────────────────────────────

Random.seed!(9999)
scenarios = [sample_scenario(hydro_data, T) for _ in 1:N_SCENARIOS]
@info "Sampled $N_SCENARIOS scenarios"

# ── Main experiment ──────────────────────────────────────────────────────────

method_names = [pc.name for pc in penalty_configs]
push!(method_names, "embedded")
n_methods = length(method_names)

all_cosines     = zeros(N_SCENARIOS, n_methods)
all_mag_ratios  = zeros(N_SCENARIOS, n_methods)
all_sign_agree  = zeros(N_SCENARIOS, n_methods)
all_fd_times    = zeros(N_SCENARIOS)
all_de_times    = zeros(N_SCENARIOS, n_methods)

@info "\n" * "="^70
@info "STARTING GRADIENT COMPARISON"
@info "="^70

for (si, w_flat) in enumerate(scenarios)
    @info "\n--- Scenario $si/$N_SCENARIOS ---"

    # ── Step 1: FD ground truth ──────────────────────────────────────────
    @info "  Computing FD ground truth ($N_DIRS directions, ε=$FD_EPS)..."
    fd_projections = zeros(Float64, N_DIRS)
    t_fd_start = time()

    for (di, d) in enumerate(directions)
        θ_plus  = params_vec .+ F(FD_EPS) .* d
        θ_minus = params_vec .- F(FD_EPS) .* d

        m_plus  = re(θ_plus)
        m_minus = re(θ_minus)

        obj_plus  = sequential_rollout_objective(m_plus,  w_flat, x0_init)
        obj_minus = sequential_rollout_objective(m_minus, w_flat, x0_init)

        if isfinite(obj_plus) && isfinite(obj_minus)
            fd_projections[di] = (obj_plus - obj_minus) / (2 * FD_EPS)
        else
            fd_projections[di] = NaN
            @warn "  FD direction $di: non-finite rollout (plus=$(obj_plus), minus=$(obj_minus))"
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

    if fd_valid < N_DIRS ÷ 2
        @warn "  Too few valid FD directions ($fd_valid) — skipping scenario"
        all_cosines[si, :] .= NaN
        all_mag_ratios[si, :] .= NaN
        all_sign_agree[si, :] .= NaN
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
            all_mag_ratios[si, ci] = NaN
            all_sign_agree[si, ci] = NaN
            all_de_times[si, ci] = time() - t_method_start
            continue
        end

        g_flat = envelope_gradient_flat(params_vec, re, w_flat, sol.lambda, x0_init, T, nHyd)
        if g_flat === nothing
            @warn "    Gradient is nothing"
            all_cosines[si, ci] = NaN
            all_mag_ratios[si, ci] = NaN
            all_sign_agree[si, ci] = NaN
            all_de_times[si, ci] = time() - t_method_start
            continue
        end

        method_projections = Float64[dot(g_flat, d) for d in directions]

        valid_mask = isfinite.(fd_projections) .& isfinite.(method_projections)
        nv = count(valid_mask)
        if nv < 3
            @warn "    Too few valid projections ($nv)"
            all_cosines[si, ci] = NaN
            all_mag_ratios[si, ci] = NaN
            all_sign_agree[si, ci] = NaN
        else
            fd_v = fd_projections[valid_mask]
            me_v = method_projections[valid_mask]
            cos_sim = dot(fd_v, me_v) / (norm(fd_v) * norm(me_v) + 1e-30)
            mag_ratio = norm(me_v) / (norm(fd_v) + 1e-30)
            sign_ag = count(sign.(fd_v) .== sign.(me_v)) / nv
            all_cosines[si, ci] = cos_sim
            all_mag_ratios[si, ci] = mag_ratio
            all_sign_agree[si, ci] = sign_ag
        end

        t_method = time() - t_method_start
        all_de_times[si, ci] = t_method
        @info @sprintf("    cos=%.4f  mag_ratio=%.4f  sign=%.2f%%  time=%.1fs  obj=%.2f",
            all_cosines[si, ci], all_mag_ratios[si, ci], all_sign_agree[si, ci]*100,
            t_method, sol.objective)
    end

    # ── Step 3: Embedded TSDDR gradient ──────────────────────────────────

    ci_emb = n_methods
    @info "  Method: embedded..."
    t_emb_start = time()

    sol_emb = embedded_solve_and_multipliers(w_flat, x0_init)
    if sol_emb === nothing
        @warn "    Embedded solve failed"
        all_cosines[si, ci_emb] = NaN
        all_mag_ratios[si, ci_emb] = NaN
        all_sign_agree[si, ci_emb] = NaN
        all_de_times[si, ci_emb] = time() - t_emb_start
    else
        g_flat_emb = envelope_gradient_flat(params_vec, re, w_flat, sol_emb.lambda, x0_init, T, nHyd)
        if g_flat_emb === nothing
            @warn "    Embedded gradient is nothing"
            all_cosines[si, ci_emb] = NaN
            all_mag_ratios[si, ci_emb] = NaN
            all_sign_agree[si, ci_emb] = NaN
        else
            emb_projections = Float64[dot(g_flat_emb, d) for d in directions]
            valid_mask = isfinite.(fd_projections) .& isfinite.(emb_projections)
            nv = count(valid_mask)
            if nv < 3
                all_cosines[si, ci_emb] = NaN
                all_mag_ratios[si, ci_emb] = NaN
                all_sign_agree[si, ci_emb] = NaN
            else
                fd_v = fd_projections[valid_mask]
                me_v = emb_projections[valid_mask]
                cos_sim = dot(fd_v, me_v) / (norm(fd_v) * norm(me_v) + 1e-30)
                mag_ratio = norm(me_v) / (norm(fd_v) + 1e-30)
                sign_ag = count(sign.(fd_v) .== sign.(me_v)) / nv
                all_cosines[si, ci_emb] = cos_sim
                all_mag_ratios[si, ci_emb] = mag_ratio
                all_sign_agree[si, ci_emb] = sign_ag
            end
        end
        t_emb = time() - t_emb_start
        all_de_times[si, ci_emb] = t_emb
        @info @sprintf("    cos=%.4f  mag_ratio=%.4f  sign=%.2f%%  time=%.1fs  obj=%.2f",
            all_cosines[si, ci_emb], all_mag_ratios[si, ci_emb],
            all_sign_agree[si, ci_emb]*100, t_emb, sol_emb.objective)
    end
end

# ── Results summary ──────────────────────────────────────────────────────────

@info "\n" * "="^70
@info "GRADIENT COMPARISON RESULTS"
@info "="^70

@info @sprintf("\n%-25s  %8s  %10s  %8s  %8s",
    "Method", "cos_sim", "mag_ratio", "sign_%", "time_s")
@info "-"^70

for (ci, name) in enumerate(method_names)
    cos_vals = filter(isfinite, all_cosines[:, ci])
    mag_vals = filter(isfinite, all_mag_ratios[:, ci])
    sig_vals = filter(isfinite, all_sign_agree[:, ci])
    t_vals   = filter(isfinite, all_de_times[:, ci])

    cos_m = isempty(cos_vals) ? NaN : mean(cos_vals)
    mag_m = isempty(mag_vals) ? NaN : mean(mag_vals)
    sig_m = isempty(sig_vals) ? NaN : mean(sig_vals)
    t_m   = isempty(t_vals)   ? NaN : mean(t_vals)

    @info @sprintf("%-25s  %8.4f  %10.4f  %7.1f%%  %7.1fs", name, cos_m, mag_m, sig_m*100, t_m)
end

@info @sprintf("\nFD ground truth: mean time per scenario = %.1fs", mean(all_fd_times))
@info @sprintf("FD total solves per scenario = %d  (2 × %d dirs × %d stages)",
    2 * N_DIRS * T, N_DIRS, T)

# ── Per-scenario detail ──────────────────────────────────────────────────────

@info "\n--- Per-scenario cosine similarity ---"
@info @sprintf("%-25s  %s", "Method", join([@sprintf("  S%d", s) for s in 1:N_SCENARIOS]))
for (ci, name) in enumerate(method_names)
    vals = [@sprintf("%.4f", all_cosines[s, ci]) for s in 1:N_SCENARIOS]
    @info @sprintf("%-25s  %s", name, join(["  " * v for v in vals]))
end

@info "\n" * "="^70
@info "EXPERIMENT COMPLETE"
@info "="^70
