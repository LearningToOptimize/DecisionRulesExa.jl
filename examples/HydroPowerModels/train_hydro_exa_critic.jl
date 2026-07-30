# train_hydro_exa_critic.jl
#
# HydroPowerModels training with ExaModels + MadNLP (DC or AC OPF).
# Uses train_tsddr from DecisionRulesExa with a scalar critic control variate.
#
# Usage:
#   julia --project -t auto train_hydro_exa_critic.jl

using DecisionRulesExa
using ExaModels
using Flux
using Statistics, Random, Dates
using Wandb, Logging
using JLD2
using MadNLP
using MadNLPGPU, KernelAbstractions, CUDA
using CUDSS, CUDSS_jll, cuDNN

const SCRIPT_DIR = dirname(@__FILE__)
include(joinpath(SCRIPT_DIR, "hydro_power_data.jl"))
include(joinpath(SCRIPT_DIR, "hydro_power_exa.jl"))

# ── Configuration ─────────────────────────────────────────────────────────────

const CASE_NAME   = "bolivia"
const FORMULATION = :ac_polar        # :dc  or  :ac_polar
const FORM_LABEL  = FORMULATION === :ac_polar ? "ACPPowerModel" : "DCPPowerModel"

const CASE_DIR    = joinpath(SCRIPT_DIR, CASE_NAME)
const PM_FILE     = joinpath(CASE_DIR, "PowerModels.json")
const HYDRO_FILE  = joinpath(CASE_DIR, "hydro.json")
const INFLOW_FILE = joinpath(CASE_DIR, "inflows.csv")
const DEMAND_FILE = joinpath(CASE_DIR, "demand.csv")

const LAYERS      = [128, 128]
const ACTIVATION  = sigmoid
const NUM_STAGES  = parse(Int, get(ENV, "DR_NUM_STAGES", "126"))
const NUM_ROLLOUT_STAGES = parse(Int, get(ENV, "DR_NUM_ROLLOUT_STAGES", "96"))
const NUM_EPOCHS  = 80
const NUM_BATCHES = 100
const MAX_EVAL_SCENARIOS = 32
const EVAL_EVERY = parse(Int, get(ENV, "DR_EVAL_EVERY", "50"))

const EVAL_SCHEDULE = [
    (1,   div(NUM_EPOCHS * NUM_BATCHES, 2), 4),
    (div(NUM_EPOCHS * NUM_BATCHES, 2) + 1, NUM_EPOCHS * NUM_BATCHES, MAX_EVAL_SCENARIOS),
]
const LR          = 1f-3
const GRAD_CLIP   = 10.0f0
const CRITIC_LR   = 5f-4
const CRITIC_HIDDEN = [256, 128]
const CRITIC_VALUE_LOSS_WEIGHT = 1.0
const CRITIC_GRADIENT_LOSS_WEIGHT = 0.0
const CRITIC_CV_WEIGHT = 0.5
const CRITIC_UPDATES_PER_BATCH = 2
const CRITIC_BUFFER_SIZE = 512
const CRITIC_BATCH_SIZE = 32

const TARGET_PEN_ARG = :auto
const HYDRO_TARGET_PENALTY_MULT = parse(Float64, get(ENV, "DR_TARGET_PENALTY_MULT", "8.0"))
const DEFICIT_COST   = 1e5
const USE_GPU        = true
const load_scaler    = 0.6
const NUM_WORKERS    = 4
const CRITIC_ROLLOUT_SAMPLES_PER_BATCH = 0  # eval rollouts feed the critic via external_critic_samples
const CRITIC_POLICY_STATE = :target      # set to :realized for closed-loop critic targets
const ROLLOUT_PARALLEL = parse(Bool, get(ENV, "DR_ROLLOUT_PARALLEL", "false"))
const CRITIC_ROLLOUT_OBJECTIVE = :objective
const NUM_CHEAP_CRITIC_SAMPLES_PER_BATCH = 4 * NUM_WORKERS

const _PENALTY_MODE = get(ENV, "DR_PENALTY_SCHEDULE", "const")
const _N_TOTAL = NUM_EPOCHS * NUM_BATCHES
const _ANNEAL_1_END = max(1, div(_N_TOTAL, 100))
const _ANNEAL_2_END = max(_ANNEAL_1_END + 1, div(_N_TOTAL, 40))
const _ANNEAL_3_END = max(_ANNEAL_2_END + 1, div(_N_TOTAL, 10))
const PENALTY_SCHEDULE = if _PENALTY_MODE == "annealed"
    [
        (1,  _ANNEAL_1_END, 0.1),
        (_ANNEAL_1_END + 1, _ANNEAL_2_END, 1.0),
        (_ANNEAL_2_END + 1, _ANNEAL_3_END, 4.0),
        (_ANNEAL_3_END + 1, _N_TOTAL, HYDRO_TARGET_PENALTY_MULT),
    ]
else
    [(1, _N_TOTAL, HYDRO_TARGET_PENALTY_MULT)]
end

const NUM_TRAIN_SCHEDULE = [
    (1,   div(NUM_EPOCHS * NUM_BATCHES, 5), NUM_WORKERS),
    (div(NUM_EPOCHS * NUM_BATCHES, 5) + 1, div(NUM_EPOCHS * NUM_BATCHES, 5) * 2, 2 * NUM_WORKERS),
    (div(NUM_EPOCHS * NUM_BATCHES, 5) * 2 + 1, div(NUM_EPOCHS * NUM_BATCHES, 5) * 3, 4 * NUM_WORKERS),
    (div(NUM_EPOCHS * NUM_BATCHES, 5) * 3 + 1, div(NUM_EPOCHS * NUM_BATCHES, 5) * 4, 8 * NUM_WORKERS),
    (div(NUM_EPOCHS * NUM_BATCHES, 5) * 4 + 1, NUM_EPOCHS * NUM_BATCHES, 8 * NUM_WORKERS),
]

const MAX_ITER = parse(Int, get(ENV, "DR_MAX_ITER", "9000"))
const SOLVER_KWARGS = (print_level = MadNLP.ERROR, tol = 1e-6, max_iter = MAX_ITER)

const RUN_NAME  = "$(CASE_NAME)-$(FORM_LABEL)-h$(NUM_STAGES)-r$(NUM_ROLLOUT_STAGES)-deteq-gpu-critic-cv-$(Dates.format(now(), "yyyymmdd-HHMMSS"))"
const MODEL_DIR = joinpath(CASE_DIR, FORM_LABEL, "models")
mkpath(MODEL_DIR)
const MODEL_PATH = joinpath(MODEL_DIR, RUN_NAME * ".jld2")

const PRE_TRAINED = nothing

# ── Load data ─────────────────────────────────────────────────────────────────

@info "Loading power system data..."
power_data = load_power_data(PM_FILE)
@info "  nBus=$(power_data.nBus)  nGen=$(power_data.nGen)"

@info "Loading hydro data..."
hydro_data = load_hydro_data(HYDRO_FILE, INFLOW_FILE, power_data;
                              num_stages = max(NUM_STAGES, NUM_ROLLOUT_STAGES) * 10)
nHyd = hydro_data.nHyd
T    = NUM_STAGES
T_ROLLOUT = NUM_ROLLOUT_STAGES
@info "  nHyd=$(nHyd)  nScenarios=$(hydro_data.nScenarios)"

demand_mat = if isfile(DEMAND_FILE)
    @info "Loading demand from $(DEMAND_FILE)..."
    load_demand(DEMAND_FILE, power_data; T = T)
else
    nothing
end

# ── Build ExaModels DE ────────────────────────────────────────────────────────

resolved_pen = TARGET_PEN_ARG === :auto ?
               auto_target_penalty(power_data, hydro_data) :
               Float64(TARGET_PEN_ARG)
@info "Auto target penalty: ρ=$(round(resolved_pen; digits=2))"
@info "Bolivia hydro target-penalty multiplier default: $(HYDRO_TARGET_PENALTY_MULT)"

backend = USE_GPU ? (@info "Using GPU backend"; CUDA.CUDABackend()) :
                    (@info "Using CPU backend"; nothing)

function _build_de()
    build_hydro_de(power_data, hydro_data, T;
        backend        = backend,
        float_type     = Float64,
        formulation    = FORMULATION,
        target_penalty = TARGET_PEN_ARG,
        deficit_cost   = DEFICIT_COST,
        demand_matrix  = demand_mat,
        load_scaler    = load_scaler,
    )
end

@info "Building $(T)-stage ExaModels DE (formulation=$FORMULATION)..."
prob = _build_de()
resolved_pen_l1 = prob.base_penalty_l1

@info "Building $(NUM_WORKERS)-worker problem pool..."
problem_pool = [(prob, prob.p_x0, prob.p_target, prob.p_inflow)]
for i in 2:NUM_WORKERS
    p = _build_de()
    push!(problem_pool, (p, p.p_x0, p.p_target, p.p_inflow))
end
@info "  Pool ready: $(NUM_WORKERS) independent DE instances on GPU"

x0_init = Float32.([clamp(hydro_data.initial_volumes[r],
                           hydro_data.units[r].min_vol,
                           hydro_data.units[r].max_vol)
                    for r in 1:nHyd])
target_lower = Float32.([h.min_vol for h in hydro_data.units])
target_upper = Float32.([h.max_vol for h in hydro_data.units])

# ── Critic/control variate ───────────────────────────────────────────────────

volume_scale = Float32.([max(abs(u.max_vol), abs(u.min_vol), 1.0) for u in hydro_data.units])
inflow_scale = Float32.([
    max(maximum(abs, hydro_data.scenario_inflows[r]), 1.0)
    for r in 1:nHyd
])

const _full_inflow_scale = repeat(inflow_scale, T)
const _full_volume_scale = repeat(volume_scale, T)

function hydro_critic_featurizer(initial_state, w_flat, xhat_flat)
    x0_scaled = Float32.(initial_state) ./ volume_scale
    w_scaled = Float32.(w_flat) ./ _full_inflow_scale
    x_scaled = Float32.(xhat_flat) ./ _full_volume_scale
    return vcat(x0_scaled, w_scaled, x_scaled)
end

critic_input_dim = nHyd + 2 * T * nHyd
critic_layers = Any[]
critic_in = critic_input_dim
for h in CRITIC_HIDDEN
    push!(critic_layers, Flux.Dense(critic_in => h, tanh))
    global critic_in = h
end
push!(critic_layers, Flux.Dense(critic_in => 1))
critic = Flux.Chain(critic_layers...)

control_variate = ScalarCriticControlVariate(
    critic;
    featurizer = hydro_critic_featurizer,
    value_loss_weight = CRITIC_VALUE_LOSS_WEIGHT,
    gradient_loss_weight = CRITIC_GRADIENT_LOSS_WEIGHT,
)

# ── Smoke test ────────────────────────────────────────────────────────────────

w_mean = mean_inflow(hydro_data, T)
ExaModels.set_parameter!(prob.core, prob.p_x0,     x0_init)
ExaModels.set_parameter!(prob.core, prob.p_inflow,  w_mean)
ExaModels.set_parameter!(prob.core, prob.p_target,  zeros(T * nHyd))
@info "Smoke test: solving DE with mean inflows..."
result0 = MadNLP.madnlp(prob.model; SOLVER_KWARGS..., print_level = MadNLP.WARN)
@info "  Status: $(result0.status)   Objective: $(round(result0.objective; digits=4))"
isfinite(result0.objective) || error("Smoke test returned non-finite objective")
solve_succeeded(result0) || @warn "Smoke test did not fully converge; proceeding anyway"

# ── Policy ────────────────────────────────────────────────────────────────────

policy_active_mask = trues(nHyd)
policy = bounded_state_policy(nHyd, target_lower, target_upper, LAYERS;
                              activation   = ACTIVATION,
                              encoder_type = Flux.LSTM,
                              active_mask  = policy_active_mask)

if !isnothing(PRE_TRAINED)
    @info "Loading pre-trained model from $(PRE_TRAINED)..."
    load_stateconditioned_policy!(policy, JLD2.load(PRE_TRAINED, "model_state"))
end

if USE_GPU
    policy  = CUDA.cu(policy)
    x0_init = CUDA.cu(x0_init)
    control_variate = ScalarCriticControlVariate(
        CUDA.cu(control_variate.critic);
        featurizer = control_variate.featurizer,
        value_loss_weight = control_variate.value_loss_weight,
        gradient_loss_weight = control_variate.gradient_loss_weight,
    )
    @info "Policy, critic, and x0 moved to GPU"
end

# ── W&B logging ───────────────────────────────────────────────────────────────

lg = WandbLogger(
    project = "RL",
    name    = RUN_NAME,
    save_code = false,
    config  = Dict(
        "case"            => CASE_NAME,
        "formulation"     => FORM_LABEL,
        "num_stages"      => T,
        "num_rollout_stages" => T_ROLLOUT,
        "layers"          => LAYERS,
        "activation"      => string(ACTIVATION),
        "target_penalty"  => "auto=$(round(resolved_pen; digits=2))",
        "target_penalty_l1" => "auto=$(round(resolved_pen_l1; digits=2))",
        "hydro_target_penalty_mult" => HYDRO_TARGET_PENALTY_MULT,
        "deficit_cost"    => DEFICIT_COST,
        "num_epochs"      => NUM_EPOCHS,
        "num_batches"     => NUM_BATCHES,
        "max_eval_scenarios" => MAX_EVAL_SCENARIOS,
        "eval_schedule"   => string(EVAL_SCHEDULE),
        "eval_every"      => EVAL_EVERY,
        "lr"              => LR,
        "grad_clip"       => GRAD_CLIP,
        "critic_lr"       => CRITIC_LR,
        "critic_hidden"   => CRITIC_HIDDEN,
        "critic_value_loss_weight" => CRITIC_VALUE_LOSS_WEIGHT,
        "critic_gradient_loss_weight" => CRITIC_GRADIENT_LOSS_WEIGHT,
        "critic_cv_weight" => CRITIC_CV_WEIGHT,
        "critic_updates_per_batch" => CRITIC_UPDATES_PER_BATCH,
        "critic_buffer_size" => CRITIC_BUFFER_SIZE,
        "critic_batch_size" => CRITIC_BATCH_SIZE,
        "critic_training_target" => "rollout",
        "critic_rollout_samples_per_batch" => CRITIC_ROLLOUT_SAMPLES_PER_BATCH,
        "critic_policy_state" => string(CRITIC_POLICY_STATE),
        "critic_rollout_objective" => string(CRITIC_ROLLOUT_OBJECTIVE),
        "num_cheap_critic_samples_per_batch" => NUM_CHEAP_CRITIC_SAMPLES_PER_BATCH,
        "backend"         => USE_GPU ? "GPU" : "CPU",
        "load_scaler"     => load_scaler,
        "penalty_schedule" => string(PENALTY_SCHEDULE),
        "num_train_schedule" => string(NUM_TRAIN_SCHEDULE),
        "num_workers" => NUM_WORKERS,
    ),
)

# ── Training ──────────────────────────────────────────────────────────────────

Random.seed!(8788)

best_obj     = Inf
epoch_losses = Float64[]

stage_demand = demand_mat === nothing ? nothing : demand_mat[1:1, :]
function _build_rollout_de()
    build_hydro_de(power_data, hydro_data, 1;
        backend        = backend,
        float_type     = Float64,
        formulation    = FORMULATION,
        target_penalty = TARGET_PEN_ARG,
        deficit_cost   = DEFICIT_COST,
        demand_matrix  = stage_demand,
        load_scaler    = load_scaler,
    )
end
rollout_prob = _build_rollout_de()
rollout_pool = ROLLOUT_PARALLEL ? [_build_rollout_de() for _ in 1:NUM_WORKERS] : []
@info "Rollout evaluation: $(ROLLOUT_PARALLEL ? "parallel ($(NUM_WORKERS) stage-problem copies)" : "sequential")"

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

const _min_vols = Float64.([h.min_vol for h in hydro_data.units])
const _max_vols = Float64.([h.max_vol for h in hydro_data.units])
const _min_vols_dev = USE_GPU ? CUDA.cu(_min_vols) : _min_vols
const _max_vols_dev = USE_GPU ? CUDA.cu(_max_vols) : _max_vols

function hydro_objective_no_target_penalty(stage_prob, result)
    sol = hydro_solution(stage_prob, result)
    penalty_l2_cost = (resolved_pen / 2) * sum(abs2, sol.delta)
    penalty_l1_cost = resolved_pen_l1 * sum(abs, sol.delta)
    return result.objective - penalty_l2_cost - penalty_l1_cost
end

Random.seed!(8789)
eval_scenarios = [sample_scenario(hydro_data, T_ROLLOUT) for _ in 1:MAX_EVAL_SCENARIOS]
rollout_evaluation = RolloutEvaluation(
    rollout_prob,
    x0_init,
    eval_scenarios;
    horizon = T_ROLLOUT,
    n_uncertainty = nHyd,
    set_stage_parameters! = set_hydro_rollout_stage!,
    realized_state = hydro_realized_state,
    objective_no_target_penalty = hydro_objective_no_target_penalty,
    madnlp_kwargs = SOLVER_KWARGS,
    warmstart = false,
    stride = EVAL_EVERY,
    policy_state = :realized,
    stage_problem_pool = rollout_pool,
    retry_on_failure = true,
    active_scenarios = 4,
    state_bounds = (_min_vols_dev, _max_vols_dev),
)

critic_training_target = RolloutCriticTarget(
    rollout_prob;
    horizon = T_ROLLOUT,
    n_uncertainty = nHyd,
    set_stage_parameters! = set_hydro_rollout_stage!,
    realized_state = hydro_realized_state,
    objective_no_target_penalty = hydro_objective_no_target_penalty,
    madnlp_kwargs = SOLVER_KWARGS,
    warmstart = false,
    policy_state = CRITIC_POLICY_STATE,
    objective_value = CRITIC_ROLLOUT_OBJECTIVE,
    state_bounds = (_min_vols_dev, _max_vols_dev),
)

Random.seed!(8788)

function _schedule_value(schedule, iter, default)
    for (lo, hi, val) in schedule
        lo <= iter <= hi && return val
    end
    return default
end

current_penalty_mult = Ref(NaN)
shared_critic_samples = Any[]

train_tsddr(
    policy,
    x0_init,
    prob,
    prob.p_x0,
    prob.p_target,
    prob.p_inflow,
    () -> sample_scenario(hydro_data, T);       # returns flat Float32 vector, length T*nHyd
    num_batches          = NUM_EPOCHS * NUM_BATCHES,
    num_train_per_batch  = NUM_WORKERS,
    optimizer            = Flux.Optimisers.OptimiserChain(
                               Flux.Optimisers.ClipGrad(GRAD_CLIP),
                               Flux.Adam(LR),
                           ),
    madnlp_kwargs        = SOLVER_KWARGS,
    warmstart            = true,
    problem_pool         = problem_pool,
    control_variate      = control_variate,
    actor_gradient_mode  = :control_variate,
    critic_cv_weight     = CRITIC_CV_WEIGHT,
    critic_updates_per_batch = CRITIC_UPDATES_PER_BATCH,
    critic_buffer_size   = CRITIC_BUFFER_SIZE,
    critic_batch_size    = CRITIC_BATCH_SIZE,
    critic_training_target = critic_training_target,
    critic_rollout_samples_per_batch = CRITIC_ROLLOUT_SAMPLES_PER_BATCH,
    num_cheap_critic_samples_per_batch = NUM_CHEAP_CRITIC_SAMPLES_PER_BATCH,
    critic_optimizer     = Flux.Adam(CRITIC_LR),
    external_critic_samples = shared_critic_samples,
    adjust_hyperparameters = (iter, opt_state, n) -> begin
        mult = _schedule_value(PENALTY_SCHEDULE, iter, last(PENALTY_SCHEDULE)[3])
        if mult != current_penalty_mult[]
            current_penalty_mult[] = mult
            ρ_half_scaled = prob.base_penalty_half * mult
            ρ_l1_scaled   = prob.base_penalty_l1 * mult
            penalty_vals = fill(ρ_half_scaled, T * nHyd)
            penalty_l1_vals = fill(ρ_l1_scaled, T * nHyd)
            for (p, _, _, _) in problem_pool
                ExaModels.set_parameter!(p.core, p.p_penalty_half, penalty_vals)
                ExaModels.set_parameter!(p.core, p.p_penalty_l1,   penalty_l1_vals)
            end
            @info "Penalty multiplier → $mult  (ρ/2 = $(round(ρ_half_scaled; digits=2)), λ_l1 = $(round(ρ_l1_scaled; digits=2)))"
        end
        n_eval = _schedule_value(EVAL_SCHEDULE, iter, MAX_EVAL_SCENARIOS)
        rollout_evaluation.active_scenarios = n_eval
        return _schedule_value(NUM_TRAIN_SCHEDULE, iter, n)
    end,
    record_loss          = (iter, m, loss, tag) -> begin
        metrics = Dict{String, Any}(tag => loss, "batch" => iter)
        isfinite(loss) && push!(epoch_losses, loss)

        if iter % EVAL_EVERY == 0
            rollout_evaluation(iter, m)
            append!(shared_critic_samples,
                    critic_samples_from_evaluation(
                        rollout_evaluation;
                        objective_key = CRITIC_ROLLOUT_OBJECTIVE,
                    ))
            metrics["metrics/rollout_objective_no_target_penalty"] =
                rollout_evaluation.last_objective_no_target_penalty
            metrics["metrics/rollout_objective_no_deficit"] =
                rollout_evaluation.last_objective_no_target_penalty
            metrics["metrics/rollout_target_violation_share"] =
                rollout_evaluation.last_violation_share
            metrics["metrics/rollout_n_ok"] =
                rollout_evaluation.last_n_ok
        end

        if !isnan(current_penalty_mult[])
            metrics["metrics/target_penalty_multiplier"] = current_penalty_mult[]
        end
        metrics["metrics/num_train_per_batch"] =
            _schedule_value(NUM_TRAIN_SCHEDULE, iter, 1)
        metrics["metrics/active_eval_scenarios"] =
            _schedule_value(EVAL_SCHEDULE, iter, MAX_EVAL_SCENARIOS)

        batch_in_epoch = (iter - 1) % NUM_BATCHES + 1
        if batch_in_epoch == NUM_BATCHES
            epoch     = (iter - 1) ÷ NUM_BATCHES + 1
            mean_loss = isempty(epoch_losses) ? NaN : mean(epoch_losses)
            n_ok      = length(epoch_losses)
            empty!(epoch_losses)
            Wandb.log(lg, Dict("metrics/epoch_objective" => mean_loss, "epoch" => epoch))
            @info "Epoch $epoch/$NUM_EPOCHS  mean=$(round(mean_loss; digits=2))  ok=$n_ok/$NUM_BATCHES"
            if isfinite(mean_loss) && mean_loss < best_obj
                global best_obj = mean_loss
                jldsave(MODEL_PATH;
                    model_state = Flux.state(cpu(m)),
                    critic_state = Flux.state(cpu(critic)),
                    critic_config = Dict(
                        "mode" => "control_variate",
                        "training_target" => "rollout",
                        "policy_state" => string(CRITIC_POLICY_STATE),
                        "rollout_objective" => string(CRITIC_ROLLOUT_OBJECTIVE),
                        "critic_cv_weight" => CRITIC_CV_WEIGHT,
                        "value_loss_weight" => CRITIC_VALUE_LOSS_WEIGHT,
                        "gradient_loss_weight" => CRITIC_GRADIENT_LOSS_WEIGHT,
                        "critic_rollout_samples_per_batch" => CRITIC_ROLLOUT_SAMPLES_PER_BATCH,
                        "num_cheap_critic_samples_per_batch" => NUM_CHEAP_CRITIC_SAMPLES_PER_BATCH,
                    ),
                )
                @info "  → New best: $(round(mean_loss; digits=4)) — saved $MODEL_PATH"
            end
        end
        Wandb.log(lg, metrics)
        return false
    end,
)

close(lg)
@info "Done. Best model saved to: $(MODEL_PATH)"
