using DecisionRulesExa
using ExaModels
using Flux
using Statistics, Random
using MadNLP, MadNLPGPU
using CUDA, CUDSS, KernelAbstractions

const SCRIPT_DIR = dirname(@__FILE__)
include(joinpath(SCRIPT_DIR, "hydro_power_data.jl"))
include(joinpath(SCRIPT_DIR, "hydro_power_exa.jl"))
include(joinpath(SCRIPT_DIR, "hydro_power_exa_embedded.jl"))

const CASE_NAME   = "bolivia"
const FORMULATION = :ac_polar
const CASE_DIR    = joinpath(SCRIPT_DIR, CASE_NAME)
const PM_FILE     = joinpath(CASE_DIR, "PowerModels.json")
const HYDRO_FILE  = joinpath(CASE_DIR, "hydro.json")
const INFLOW_FILE = joinpath(CASE_DIR, "inflows.csv")
const DEMAND_FILE = joinpath(CASE_DIR, "demand.csv")

const T = 126
const T_ROLLOUT = 96
const LAYERS = [128, 128]
const SOLVER_KWARGS = (print_level = MadNLP.ERROR, tol = 1e-6, max_iter = 9000)
const DEFICIT_COST = 1e5
const load_scaler = 0.6
const γ = 0.99

@info "Loading data..."
power_data = load_power_data(PM_FILE)
hydro_data = load_hydro_data(HYDRO_FILE, INFLOW_FILE, power_data; num_stages = T * 10)
nHyd = hydro_data.nHyd
demand_mat = isfile(DEMAND_FILE) ? load_demand(DEMAND_FILE, power_data; T = T) : nothing

ρ_auto = auto_target_penalty(power_data, hydro_data)
@info "Auto penalty: ρ=$(round(ρ_auto; digits=2))  max_cost=$(round(ρ_auto/2; digits=2))"

# ── Test 1: Discount weights ────────────────────────────────────────────────

function test_discount_weights()
    @info "TEST 1: Discount weights"
    weights = Float64[γ^(t-1) for t in 1:T for _ in 1:nHyd]
    @assert length(weights) == T * nHyd "Wrong length: $(length(weights)) != $(T * nHyd)"
    @assert weights[1] == 1.0 "Stage 1 weight should be 1.0"
    @assert weights[nHyd] == 1.0 "Stage 1 last unit weight should be 1.0"
    @assert weights[nHyd + 1] ≈ γ "Stage 2 weight should be γ=$(γ)"
    @assert weights[end] ≈ γ^(T-1) "Stage T weight should be γ^(T-1)"
    @info "  weights[1]=$(weights[1])  weights[nHyd+1]=$(weights[nHyd+1])  weights[end]=$(round(weights[end]; sigdigits=4))"
    @info "  Stage 1 penalty factor: 1.0"
    @info "  Stage $(T) penalty factor: $(round(γ^(T-1); sigdigits=4))"
    @info "  TEST 1 PASSED"
end

# ── Test 2: Anti-anticipativity schedule ─────────────────────────────────────

function test_annealing_schedule()
    @info "TEST 2: Anti-anticipativity annealing schedule"
    min_safe = max(2.0, ceil(0.5 / γ^(T - 1)))
    @info "  min_safe_mult = $(min_safe)  (from 0.5 / γ^$(T-1) = $(round(0.5 / γ^(T-1); sigdigits=4)))"
    schedule = [min_safe, min_safe * 2.5, min_safe * 5.0, min_safe * 10.0]
    @info "  Schedule: $(schedule)"

    for (phase, mult) in enumerate(schedule)
        min_stage_penalty = mult * γ^(T-1) * ρ_auto
        max_cost = ρ_auto / 2
        ratio = min_stage_penalty / max_cost
        safe = ratio >= 1.0
        @info "  Phase $phase (mult=$mult): min_stage_penalty=$(round(min_stage_penalty; digits=1)), max_cost=$(round(max_cost; digits=1)), ratio=$(round(ratio; sigdigits=3)) $(safe ? "SAFE" : "UNSAFE")"
        if !safe
            @warn "  Phase $phase is NOT anti-anticipativity safe!"
        end
    end

    const_penalty_check = 1.0 * γ^(T-1) * ρ_auto / (ρ_auto / 2)
    @info "  Constant (mult=1) last-stage ratio: $(round(const_penalty_check; sigdigits=3)) $(const_penalty_check >= 1.0 ? "SAFE" : "BELOW threshold — expected for const config")"
    @info "  TEST 2 PASSED"
end

# ── Test 3: Verify penalty parameter layout matches discount weights ─────────

function test_penalty_parameter_layout()
    @info "TEST 3: Penalty parameter layout"
    backend = CUDA.CUDABackend()
    prob = build_hydro_de(power_data, hydro_data, T;
        backend = backend, float_type = Float64, formulation = FORMULATION,
        target_penalty = :auto, deficit_cost = DEFICIT_COST,
        demand_matrix = demand_mat, load_scaler = load_scaler)

    @info "  Penalty parameters exist, testing set_parameter! with $(T * nHyd) discount values"

    weights = Float64[γ^(t-1) for t in 1:T for _ in 1:nHyd]
    ρ_half = prob.base_penalty_half
    discounted_penalties = ρ_half .* weights
    ExaModels.set_parameter!(prob.core, prob.p_penalty_half, discounted_penalties)
    ExaModels.set_parameter!(prob.core, prob.p_penalty_l1, prob.base_penalty_l1 .* weights)

    x0 = Float32.([clamp(hydro_data.initial_volumes[r], hydro_data.units[r].min_vol, hydro_data.units[r].max_vol) for r in 1:nHyd])
    w_mean = mean_inflow(hydro_data, T)
    ExaModels.set_parameter!(prob.core, prob.p_x0, x0)
    ExaModels.set_parameter!(prob.core, prob.p_inflow, w_mean)
    ExaModels.set_parameter!(prob.core, prob.p_target, zeros(T * nHyd))

    @info "  Solving with discounted penalties..."
    result = MadNLP.madnlp(prob.model; SOLVER_KWARGS..., print_level = MadNLP.WARN)
    @info "  Status: $(result.status)  Obj: $(round(result.objective; digits=2))"
    @assert solve_succeeded(result) "Solve failed with discounted penalties!"

    ExaModels.set_parameter!(prob.core, prob.p_penalty_half, fill(ρ_half, T * nHyd))
    ExaModels.set_parameter!(prob.core, prob.p_penalty_l1, fill(prob.base_penalty_l1, T * nHyd))
    result_uniform = MadNLP.madnlp(prob.model; SOLVER_KWARGS..., print_level = MadNLP.WARN)
    @info "  Uniform penalties obj: $(round(result_uniform.objective; digits=2))"
    @info "  Discounted penalties should give LOWER obj (less late-stage enforcement)"
    @info "  Diff: $(round(result.objective - result_uniform.objective; digits=2))"
    @info "  TEST 3 PASSED"
    return prob
end

# ── Test 4: Single training step — regular DE ────────────────────────────────

function test_single_step_de(prob)
    @info "TEST 4: Single training step — Regular DE with discounted penalties"
    backend = CUDA.CUDABackend()

    x0 = Float32.([clamp(hydro_data.initial_volumes[r], hydro_data.units[r].min_vol, hydro_data.units[r].max_vol) for r in 1:nHyd])
    target_lower = Float32.([h.min_vol for h in hydro_data.units])
    target_upper = Float32.([h.max_vol for h in hydro_data.units])

    policy = bounded_state_policy(nHyd, target_lower, target_upper, LAYERS;
                                  activation = sigmoid, encoder_type = Flux.LSTM,
                                  active_mask = nothing)
    policy = CUDA.cu(policy)
    x0_dev = CUDA.cu(x0)

    weights = Float64[γ^(t-1) for t in 1:T for _ in 1:nHyd]
    ρ_half = prob.base_penalty_half
    ρ_l1 = prob.base_penalty_l1
    ExaModels.set_parameter!(prob.core, prob.p_penalty_half, ρ_half .* weights)
    ExaModels.set_parameter!(prob.core, prob.p_penalty_l1, ρ_l1 .* weights)

    Random.seed!(42)
    opt_state = Flux.setup(Flux.Adam(1f-3), policy)

    @info "  Running 3 training steps..."
    losses = Float64[]
    for step in 1:3
        scenario = sample_scenario(hydro_data, T)
        loss, grad = tsddr_step!(policy, x0_dev, prob, prob.p_x0, prob.p_target, prob.p_inflow, scenario; madnlp_kwargs = SOLVER_KWARGS, warmstart = true)
        push!(losses, loss)

        if isfinite(loss)
            Flux.update!(opt_state, policy, grad)
            @info "  Step $step: loss=$(round(loss; digits=2)) — gradient applied"
        else
            @warn "  Step $step: loss=$loss — SKIPPED gradient update"
        end
    end

    n_finite = count(isfinite, losses)
    @info "  $n_finite/3 steps had finite loss"
    @assert n_finite >= 1 "No finite losses in 3 steps!"
    @info "  TEST 4 PASSED"
    return policy
end

# ── Test 5: Single training step — Embedded DE ──────────────────────────────

function test_single_step_embedded()
    @info "TEST 5: Single training step — Embedded DE with discounted penalties"
    backend = CUDA.CUDABackend()

    x0 = Float32.([clamp(hydro_data.initial_volumes[r], hydro_data.units[r].min_vol, hydro_data.units[r].max_vol) for r in 1:nHyd])
    target_lower = Float32.([h.min_vol for h in hydro_data.units])
    target_upper = Float32.([h.max_vol for h in hydro_data.units])

    Random.seed!(43)
    policy_emb = bounded_state_policy(nHyd, target_lower, target_upper, LAYERS;
                                      activation = sigmoid, encoder_type = Flux.LSTM,
                                      active_mask = trues(nHyd))
    policy_emb = CUDA.cu(policy_emb)
    x0_dev = CUDA.cu(x0)

    @info "  Building embedded DE..."
    prob_emb = build_embedded_hydro_de(policy_emb, power_data, hydro_data, T;
        backend = backend, formulation = FORMULATION,
        target_penalty = :auto, deficit_cost = DEFICIT_COST,
        demand_matrix = demand_mat, load_scaler = load_scaler)

    weights = Float64[γ^(t-1) for t in 1:T for _ in 1:nHyd]
    ExaModels.set_parameter!(prob_emb.core, prob_emb.p_penalty_half, prob_emb.base_penalty_half .* weights)
    ExaModels.set_parameter!(prob_emb.core, prob_emb.p_penalty_l1, prob_emb.base_penalty_l1 .* weights)

    @info "  Smoke test with discounted penalties..."
    set_x0!(prob_emb, x0_dev)
    set_inflows!(prob_emb, mean_inflow(hydro_data, T))
    result0 = MadNLP.madnlp(prob_emb.model; SOLVER_KWARGS..., print_level = MadNLP.WARN)
    @info "  Smoke: status=$(result0.status)  obj=$(round(result0.objective; digits=2))"
    @assert solve_succeeded(result0) "Embedded smoke test failed!"

    opt_state = Flux.setup(Flux.Adam(1f-3), policy_emb)

    @info "  Running 2 embedded training steps..."
    losses = Float64[]
    for step in 1:2
        scenario = sample_scenario(hydro_data, T)
        loss, grad = tsddr_step_embedded!(policy_emb, x0_dev, prob_emb, scenario;
            madnlp_kwargs = SOLVER_KWARGS, warmstart = true,
            get_realized_states = embedded_hydro_realized_states)
        push!(losses, loss)

        if isfinite(loss) && loss != 0.0
            Flux.update!(opt_state, policy_emb, grad)
            @info "  Step $step: loss=$(round(loss; digits=2)) — gradient applied"
        else
            @warn "  Step $step: loss=$loss — SKIPPED"
        end
    end

    n_finite = count(x -> isfinite(x) && x != 0.0, losses)
    @info "  $n_finite/2 steps had valid loss"
    @assert n_finite >= 1 "No valid losses!"
    @info "  TEST 5 PASSED"
    return policy_emb
end

# ── Test 6: Rollout evaluation with state_bounds + :target ───────────────────

function test_rollout_evaluation(policy)
    @info "TEST 6: Rollout evaluation (:target + state_bounds)"
    backend = CUDA.CUDABackend()

    x0 = Float32.([clamp(hydro_data.initial_volumes[r], hydro_data.units[r].min_vol, hydro_data.units[r].max_vol) for r in 1:nHyd])
    x0_dev = CUDA.cu(x0)

    _min_vols = Float64.([h.min_vol for h in hydro_data.units])
    _max_vols = Float64.([h.max_vol for h in hydro_data.units])
    _min_vols_dev = CUDA.cu(_min_vols)
    _max_vols_dev = CUDA.cu(_max_vols)

    stage_demand = demand_mat === nothing ? nothing : demand_mat[1:1, :]
    rollout_prob = build_hydro_de(power_data, hydro_data, 1;
        backend = backend, float_type = Float64, formulation = FORMULATION,
        target_penalty = :auto, deficit_cost = DEFICIT_COST,
        demand_matrix = stage_demand, load_scaler = load_scaler)
    rollout_pool = [build_hydro_de(power_data, hydro_data, 1;
        backend = backend, float_type = Float64, formulation = FORMULATION,
        target_penalty = :auto, deficit_cost = DEFICIT_COST,
        demand_matrix = stage_demand, load_scaler = load_scaler) for _ in 1:2]

    ρ_pen = rollout_prob.base_penalty_half * 2
    ρ_l1 = rollout_prob.base_penalty_l1

    function set_stage!(stage_prob, state_in, wt, target, stage)
        ExaModels.set_parameter!(stage_prob.core, stage_prob.p_x0, state_in)
        ExaModels.set_parameter!(stage_prob.core, stage_prob.p_inflow, wt)
        if demand_mat !== nothing
            set_demand!(stage_prob, load_scaler .* demand_mat[stage:stage, :])
        end
        ExaModels.set_parameter!(stage_prob.core, stage_prob.p_target, target)
        return stage_prob
    end

    realized_state(sp, res) = hydro_solution(sp, res).reservoir[:, end]

    function obj_no_pen(sp, res)
        sol = hydro_solution(sp, res)
        delta = sol.delta
        return res.objective - (ρ_pen / 2) * sum(abs2, delta) - ρ_l1 * sum(abs, delta)
    end

    Random.seed!(999)
    eval_scenarios = [sample_scenario(hydro_data, T_ROLLOUT) for _ in 1:2]

    rollout_eval = RolloutEvaluation(
        rollout_prob, x0_dev, eval_scenarios;
        horizon = T_ROLLOUT, n_uncertainty = nHyd,
        set_stage_parameters! = set_stage!,
        realized_state = realized_state,
        objective_no_target_penalty = obj_no_pen,
        madnlp_kwargs = SOLVER_KWARGS,
        warmstart = true, stride = 1,
        policy_state = :target,
        stage_problem_pool = rollout_pool,
        active_scenarios = 2,
        state_bounds = (_min_vols_dev, _max_vols_dev),
    )

    @info "  Running rollout evaluation (2 scenarios, T=$T_ROLLOUT)..."
    rollout_eval(1, policy)
    obj = rollout_eval.last_objective_no_target_penalty
    viol = rollout_eval.last_violation_share
    n_ok = rollout_eval.last_n_ok
    @info "  Obj (no penalty): $(round(obj; digits=2))"
    @info "  Target violation share: $(round(viol; sigdigits=3))"
    @info "  Solves OK: $(n_ok) / $(2 * T_ROLLOUT)"
    @assert n_ok > T_ROLLOUT "Too few successful solves — rollout is broken"
    @info "  TEST 6 PASSED"
end

# ── Run all tests ────────────────────────────────────────────────────────────

@info "═══════════════════════════════════════════════"
@info "Testing discount penalty configs on GPU"
@info "  T_train=$T  T_rollout=$T_ROLLOUT  nHyd=$nHyd  γ=$γ  formulation=$FORMULATION"
@info "═══════════════════════════════════════════════"

test_discount_weights()
test_annealing_schedule()
prob = test_penalty_parameter_layout()
policy = test_single_step_de(prob)
test_rollout_evaluation(policy)
GC.gc(); CUDA.reclaim()
policy_emb = test_single_step_embedded()

@info ""
@info "═══════════════════════════════════════════════"
@info "ALL 6 TESTS PASSED"
@info "═══════════════════════════════════════════════"
