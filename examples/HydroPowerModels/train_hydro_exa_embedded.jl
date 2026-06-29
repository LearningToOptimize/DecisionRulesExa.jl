# train_hydro_exa_embedded.jl
#
# Embedded-NN hydro training with ExaModels + MadNLP (AC or DC OPF).
# Policy is embedded directly in the NLP via VectorNonlinearOracle.
# Gradient: envelope theorem (multiplier-weighted policy Jacobian).
#
# 4 configurations via environment variables:
#   DR_PENALTY_SCHEDULE = "const" | "annealed"
#   DR_TARGET_PENALTY_MULT = 8.0  (Bolivia default multiplier on :auto penalties)
#   DR_PRETRAIN_ITERS   = 0 | 500   (regular TSDDR warmup before embedded training)
#   DR_PRETRAIN_PENALTY_MULT = 0.1  (penalty multiplier during pretrain, default 0.1)

using DecisionRulesExa
using ExaModels
using Flux
using Statistics, Random, Dates
using Wandb, Logging
using JLD2
using MadNLP, MadNLPGPU
using CUDA, CUDSS, KernelAbstractions

const SCRIPT_DIR = dirname(@__FILE__)
include(joinpath(SCRIPT_DIR, "hydro_power_data.jl"))
include(joinpath(SCRIPT_DIR, "hydro_power_exa.jl"))
include(joinpath(SCRIPT_DIR, "hydro_power_exa_embedded.jl"))

# ── Configuration ─────────────────────────────────────────────────────────────

const CASE_NAME   = "bolivia"
const FORMULATION = :ac_polar
const FORM_LABEL  = FORMULATION === :ac_polar ? "ACPPowerModel" : "DCPPowerModel"

const CASE_DIR    = joinpath(SCRIPT_DIR, CASE_NAME)
const PM_FILE     = joinpath(CASE_DIR, "PowerModels.json")
const HYDRO_FILE  = joinpath(CASE_DIR, "hydro.json")
const INFLOW_FILE = joinpath(CASE_DIR, "inflows.csv")
const DEMAND_FILE = joinpath(CASE_DIR, "demand.csv")

const LAYERS      = let s = get(ENV, "DR_LAYERS", "128,128"); [parse(Int, x) for x in split(s, ",")] end
const ACTIVATION  = sigmoid
const NUM_STAGES  = parse(Int, get(ENV, "DR_NUM_STAGES", "126"))
const NUM_ROLLOUT_STAGES = parse(Int, get(ENV, "DR_NUM_ROLLOUT_STAGES", "96"))
const NUM_EPOCHS  = parse(Int, get(ENV, "DR_NUM_EPOCHS", "80"))
const NUM_BATCHES = 100
const NUM_TRAIN_PER_BATCH = 1
const NUM_EVAL_SCENARIOS  = 4
const EVAL_EVERY  = parse(Int, get(ENV, "DR_EVAL_EVERY", "50"))
const LR          = 1f-3

const TARGET_PEN_ARG = :auto
const HYDRO_TARGET_PENALTY_MULT = parse(Float64, get(ENV, "DR_TARGET_PENALTY_MULT", "8.0"))
const DEFICIT_COST   = 1e5
const load_scaler    = 0.6

const PRETRAIN_ITERS       = parse(Int, get(ENV, "DR_PRETRAIN_ITERS", "0"))
const PRETRAIN_PENALTY_MULT = parse(Float64, get(ENV, "DR_PRETRAIN_PENALTY_MULT", "0.1"))
const STRICT_EMBEDDED_TARGETS = parse(Bool, get(ENV, "DR_STRICT_EMBEDDED_TARGETS", "false"))

const DISCOUNT_GAMMA = parse(Float64, get(ENV, "DR_DISCOUNT_GAMMA", "1.0"))
const ROLLOUT_PARALLEL = parse(Bool, get(ENV, "DR_ROLLOUT_PARALLEL", "false"))

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
elseif _PENALTY_MODE == "annealed_discount"
    [
        (1,  _ANNEAL_1_END, 0.1),
        (_ANNEAL_1_END + 1, _ANNEAL_2_END, 1.0),
        (_ANNEAL_2_END + 1, _ANNEAL_3_END, 4.0),
        (_ANNEAL_3_END + 1, _N_TOTAL, HYDRO_TARGET_PENALTY_MULT),
    ]
else
    [(1, _N_TOTAL, HYDRO_TARGET_PENALTY_MULT)]
end

const USE_GPU = parse(Bool, get(ENV, "DR_USE_GPU", "true"))
const MAX_ITER = parse(Int, get(ENV, "DR_MAX_ITER", "9000"))
const SOLVER_KWARGS = (print_level = MadNLP.ERROR, tol = 1e-6, max_iter = MAX_ITER)

const _DISC_TAG    = DISCOUNT_GAMMA < 1.0 ? "-disc$(replace(string(DISCOUNT_GAMMA), "." => ""))" : ""
const _SCHED_TAG   = if _PENALTY_MODE == "annealed"
    "-anneal"
elseif _PENALTY_MODE == "annealed_discount"
    "-anndisc"
else
    "-const"
end
const _PRETRAIN_TAG = PRETRAIN_ITERS > 0 ? "-pre$(PRETRAIN_ITERS)" : ""
const _GPU_TAG     = USE_GPU ? "-gpu" : ""
const _LAYER_TAG   = LAYERS == [128, 128] ? "" : "-L$(join(LAYERS, "_"))"
const _STRICT_TAG = STRICT_EMBEDDED_TARGETS ? "-strict" : ""
const RUN_NAME  = "$(CASE_NAME)-$(FORM_LABEL)-h$(NUM_STAGES)-r$(NUM_ROLLOUT_STAGES)-embedded$(_GPU_TAG)$(_SCHED_TAG)$(_DISC_TAG)$(_LAYER_TAG)$(_PRETRAIN_TAG)$(_STRICT_TAG)-$(Dates.format(now(), "yyyymmdd-HHMMSS"))"
const MODEL_DIR = joinpath(CASE_DIR, FORM_LABEL, "models")
mkpath(MODEL_DIR)
const MODEL_PATH = joinpath(MODEL_DIR, RUN_NAME * ".jld2")

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

resolved_pen = TARGET_PEN_ARG === :auto ?
               auto_target_penalty(power_data, hydro_data) :
               Float64(TARGET_PEN_ARG)
@info "Auto target penalty: ρ=$(round(resolved_pen; digits=2))"
@info "Bolivia hydro target-penalty multiplier default: $(HYDRO_TARGET_PENALTY_MULT)"

backend = USE_GPU ? (@info "Using GPU backend"; CUDA.CUDABackend()) :
                    (@info "Using CPU backend"; nothing)

x0_init = Float32.([clamp(hydro_data.initial_volumes[r],
                           hydro_data.units[r].min_vol,
                           hydro_data.units[r].max_vol)
                    for r in 1:nHyd])
target_lower = Float32.([h.min_vol for h in hydro_data.units])
target_upper = Float32.([h.max_vol for h in hydro_data.units])

const _discount_weights = Float64[DISCOUNT_GAMMA^(t-1) for t in 1:T for _ in 1:nHyd]
if DISCOUNT_GAMMA < 1.0
    @info "Discount γ=$(DISCOUNT_GAMMA): stage 1 weight=1.0, stage $T weight=$(round(DISCOUNT_GAMMA^(T-1); sigdigits=4))"
end

# ── Policy ────────────────────────────────────────────────────────────────────

Random.seed!(42)
policy = if STRICT_EMBEDDED_TARGETS
    @info "Using strict embedded hydro policy with one-stage reachable target bounds"
    hydro_reachable_policy(hydro_data, LAYERS;
                           activation   = ACTIVATION,
                           encoder_type = Flux.LSTM)
else
    bounded_state_policy(nHyd, target_lower, target_upper, LAYERS;
                         activation   = ACTIVATION,
                         encoder_type = Flux.LSTM,
                         active_mask  = trues(nHyd))
end

# ── Optional pretrain with regular TSDDR ──────────────────────────────────────

if PRETRAIN_ITERS > 0
    @info "Building regular DE for pretrain ($(PRETRAIN_ITERS) iters)..."
    prob_reg = build_hydro_de(power_data, hydro_data, T;
        backend        = backend,
        float_type     = Float64,
        formulation    = FORMULATION,
        target_penalty = TARGET_PEN_ARG,
        deficit_cost   = DEFICIT_COST,
        demand_matrix  = demand_mat,
        load_scaler    = load_scaler,
    )

    if PRETRAIN_PENALTY_MULT != 1.0
        ρ_half_pre    = prob_reg.base_penalty_half * PRETRAIN_PENALTY_MULT
        ρ_l1_pre      = prob_reg.base_penalty_l1 * PRETRAIN_PENALTY_MULT
        ExaModels.set_parameter!(prob_reg.core, prob_reg.p_penalty_half,
                                  fill(ρ_half_pre, T * nHyd))
        ExaModels.set_parameter!(prob_reg.core, prob_reg.p_penalty_l1,
                                  fill(ρ_l1_pre, T * nHyd))
        @info "  Pretrain penalty mult=$(PRETRAIN_PENALTY_MULT) (ρ/2=$(round(ρ_half_pre; digits=2)), l1=$(round(ρ_l1_pre; digits=2)))"
    end

    @info "Pretraining policy with regular TSDDR..."
    train_tsddr(
        policy, x0_init, prob_reg,
        prob_reg.p_x0, prob_reg.p_target, prob_reg.p_inflow,
        () -> sample_scenario(hydro_data, T);
        num_batches         = PRETRAIN_ITERS,
        num_train_per_batch = 1,
        optimizer           = Flux.Adam(LR),
        madnlp_kwargs       = SOLVER_KWARGS,
        warmstart           = true,
        record_loss         = (iter, m, loss, tag) -> begin
            if iter % 50 == 0
                @info "  Pretrain iter $iter/$PRETRAIN_ITERS: loss=$(round(loss; digits=2))"
            end
            return false
        end,
    )
    @info "Pretrain done."
end

# ── Move policy + x0 to GPU (BEFORE embedded DE build so oracle captures GPU policy) ──

if USE_GPU
    policy  = CUDA.cu(policy)
    x0_init = CUDA.cu(x0_init)
    @info "Policy and x0 moved to GPU"
end

# ── Build embedded DE ─────────────────────────────────────────────────────────

@info "Building embedded $(T)-stage $(FORMULATION) hydro DE..."
prob_emb = build_embedded_hydro_de(policy, power_data, hydro_data, T;
    backend        = backend,
    formulation    = FORMULATION,
    target_penalty = TARGET_PEN_ARG,
    deficit_cost   = DEFICIT_COST,
    demand_matrix  = demand_mat,
    load_scaler    = load_scaler,
    strict_targets = STRICT_EMBEDDED_TARGETS,
)
@info "  nvar=$(prob_emb._nvar)  oracle_cons=$(length(prob_emb.target_con_range))  strict_targets=$(prob_emb.strict_targets)"

resolved_pen_l1 = prob_emb.base_penalty_l1

# ── Smoke test ────────────────────────────────────────────────────────────────

w_mean = mean_inflow(hydro_data, T)
set_x0!(prob_emb, x0_init)
set_inflows!(prob_emb, w_mean)
@info "Smoke test: solving embedded DE with mean inflows..."
result0 = MadNLP.madnlp(prob_emb.model; SOLVER_KWARGS..., print_level = MadNLP.WARN)
@info "  Status: $(result0.status)  Obj: $(round(result0.objective; digits=4))"
solve_succeeded(result0) || @warn "Smoke test did not fully converge; proceeding anyway"

# ── W&B logging ───────────────────────────────────────────────────────────────

lg = WandbLogger(
    project = "RL",
    name    = RUN_NAME,
    save_code = false,
    config  = Dict(
        "case"            => CASE_NAME,
        "formulation"     => FORM_LABEL,
        "method"          => "embedded_nn",
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
        "num_train_per_batch" => NUM_TRAIN_PER_BATCH,
        "num_eval_scenarios" => NUM_EVAL_SCENARIOS,
        "eval_every"      => EVAL_EVERY,
        "lr"              => LR,
        "penalty_schedule" => string(PENALTY_SCHEDULE),
        "discount_gamma"  => DISCOUNT_GAMMA,
        "pretrain_iters"  => PRETRAIN_ITERS,
        "pretrain_penalty_mult" => PRETRAIN_PENALTY_MULT,
        "strict_embedded_targets" => STRICT_EMBEDDED_TARGETS,
    ),
)

# ── Rollout evaluation ────────────────────────────────────────────────────────

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
        strict_targets = STRICT_EMBEDDED_TARGETS,
    )
end

rollout_prob = _build_rollout_de()
rollout_pool = ROLLOUT_PARALLEL ? [_build_rollout_de() for _ in 1:NUM_EVAL_SCENARIOS] : []
@info "Rollout evaluation: $(ROLLOUT_PARALLEL ? "parallel ($(NUM_EVAL_SCENARIOS) stage-problem copies)" : "sequential")"

function set_hydro_rollout_stage!(stage_prob, state_in, wt, target, stage)
    ExaModels.set_parameter!(stage_prob.core, stage_prob.p_x0, state_in)
    ExaModels.set_parameter!(stage_prob.core, stage_prob.p_inflow, wt)
    if demand_mat !== nothing
        set_demand!(stage_prob, load_scaler .* demand_mat[stage:stage, :])
    end
    ExaModels.set_parameter!(stage_prob.core, stage_prob.p_target, target)
    return stage_prob
end

const _min_vols = Float64.([h.min_vol for h in hydro_data.units])
const _max_vols = Float64.([h.max_vol for h in hydro_data.units])
const _min_vols_dev = USE_GPU ? CUDA.cu(_min_vols) : _min_vols
const _max_vols_dev = USE_GPU ? CUDA.cu(_max_vols) : _max_vols

hydro_realized_state(stage_prob, result) =
    hydro_solution(stage_prob, result).reservoir[:, end]

function hydro_objective_no_target_penalty(stage_prob, result)
    if STRICT_EMBEDDED_TARGETS
        return result.objective
    end
    sol = hydro_solution(stage_prob, result)
    delta = sol.delta
    penalty_l2_cost = (resolved_pen / 2) * sum(abs2, delta)
    penalty_l1_cost = resolved_pen_l1 * sum(abs, delta)
    return result.objective - penalty_l2_cost - penalty_l1_cost
end

Random.seed!(8789)
eval_scenarios = [sample_scenario(hydro_data, T_ROLLOUT) for _ in 1:NUM_EVAL_SCENARIOS]

rollout_evaluation = RolloutEvaluation(
    rollout_prob, x0_init, eval_scenarios;
    horizon = T_ROLLOUT, n_uncertainty = nHyd,
    set_stage_parameters! = set_hydro_rollout_stage!,
    realized_state = hydro_realized_state,
    objective_no_target_penalty = hydro_objective_no_target_penalty,
    madnlp_kwargs = SOLVER_KWARGS,
    warmstart = false,
    stride = EVAL_EVERY,
    policy_state = :realized,
    stage_problem_pool = rollout_pool,
    retry_on_failure = true,
    active_scenarios = NUM_EVAL_SCENARIOS,
    state_bounds = (_min_vols_dev, _max_vols_dev),
)

# ── Training ──────────────────────────────────────────────────────────────────

Random.seed!(8788)

best_obj     = Inf
epoch_losses = Float64[]

function _schedule_value(schedule, iter, default)
    for (lo, hi, val) in schedule
        lo <= iter <= hi && return val
    end
    return default
end

current_penalty_mult = Ref(NaN)
last_batch_stats = Ref(Dict{String, Any}())

function _merge_batch_stats!(metrics, stats)
    isempty(stats) && return metrics
    metrics["metrics/train_n_ok"] = get(stats, "n_ok", 0)
    metrics["metrics/train_n_total"] = get(stats, "n_total", 0)
    metrics["metrics/train_success_share"] =
        get(stats, "n_total", 0) == 0 ? NaN : get(stats, "n_ok", 0) / get(stats, "n_total", 0)
    for (k, v) in get(stats, "status_counts", Dict{String, Int}())
        metrics["metrics/train_status/$k"] = v
    end
    for (k, v) in get(stats, "failure_counts", Dict{String, Int}())
        metrics["metrics/train_failure/$k"] = v
    end
    for (k, v) in get(stats, "retry_counts", Dict{String, Int}())
        metrics["metrics/train_retry/$k"] = v
    end
    return metrics
end

@info "Starting embedded training: $(NUM_EPOCHS) epochs × $(NUM_BATCHES) batches"

train_tsddr_embedded(
    policy, x0_init, prob_emb,
    () -> sample_scenario(hydro_data, T);
    num_batches         = NUM_EPOCHS * NUM_BATCHES,
    num_train_per_batch = NUM_TRAIN_PER_BATCH,
    optimizer           = Flux.Adam(LR),
    madnlp_kwargs       = SOLVER_KWARGS,
    warmstart           = true,
    get_realized_states = embedded_hydro_realized_states,
    batch_diagnostics = (iter, stats) -> begin
        last_batch_stats[] = stats
        n_ok = get(stats, "n_ok", 0)
        n_total = get(stats, "n_total", 0)
        if n_ok < n_total
            @warn "Embedded training solve failures at iter $iter" n_ok n_total status_counts=get(stats, "status_counts", nothing) failure_counts=get(stats, "failure_counts", nothing) retry_counts=get(stats, "retry_counts", nothing)
        elseif iter % 10 == 0
            @info "Embedded training solve status at iter $iter" n_ok n_total status_counts=get(stats, "status_counts", nothing) retry_counts=get(stats, "retry_counts", nothing)
        end
    end,
    adjust_hyperparameters = (iter, opt_state, n) -> begin
        mult = _schedule_value(PENALTY_SCHEDULE, iter, last(PENALTY_SCHEDULE)[3])
        if mult != current_penalty_mult[]
            current_penalty_mult[] = mult
            if STRICT_EMBEDDED_TARGETS
                @info "Strict target equality active; target slack penalties are not used"
            else
                ρ_half_scaled = prob_emb.base_penalty_half * mult
                ρ_l1_scaled   = prob_emb.base_penalty_l1 * mult
                penalty_vals    = ρ_half_scaled .* _discount_weights
                penalty_l1_vals = ρ_l1_scaled   .* _discount_weights
                ExaModels.set_parameter!(prob_emb.core, prob_emb.p_penalty_half, penalty_vals)
                ExaModels.set_parameter!(prob_emb.core, prob_emb.p_penalty_l1,   penalty_l1_vals)
                @info "Penalty multiplier → $mult  (ρ/2 = $(round(ρ_half_scaled; digits=2)), λ_l1 = $(round(ρ_l1_scaled; digits=2)), γ=$DISCOUNT_GAMMA)"
            end
        end
        return n
    end,
    record_loss = (iter, m, loss, tag) -> begin
        metrics = Dict{String, Any}(tag => loss, "batch" => iter)
        _merge_batch_stats!(metrics, last_batch_stats[])
        isfinite(loss) && push!(epoch_losses, loss)

        if iter % EVAL_EVERY == 0
            rollout_evaluation(iter, m)
            metrics["metrics/rollout_objective_no_target_penalty"] =
                rollout_evaluation.last_objective_no_target_penalty
            metrics["metrics/rollout_target_violation_share"] =
                rollout_evaluation.last_violation_share
            metrics["metrics/rollout_n_ok"] =
                rollout_evaluation.last_n_ok
        end

        if !isnan(current_penalty_mult[])
            metrics["metrics/target_penalty_multiplier"] = current_penalty_mult[]
        end

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
                jldsave(MODEL_PATH; model_state = Flux.state(cpu(m)))
                @info "  → New best: $(round(mean_loss; digits=4)) — saved $MODEL_PATH"
            end
        end
        Wandb.log(lg, metrics)
        return false
    end,
)

close(lg)
@info "Done. Best model saved to: $(MODEL_PATH)"
