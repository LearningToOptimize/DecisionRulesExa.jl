# train_hydro_exa_embedded.jl
#
# Embedded-NN hydro training with ExaModels + MadNLP (AC or DC OPF).
# Policy is embedded directly in the NLP via VectorNonlinearOracle.
# Gradient: envelope theorem (multiplier-weighted policy Jacobian).
#
# 4 configurations via environment variables:
#   DR_PENALTY_SCHEDULE = "const" | "annealed"
#   DR_PRETRAIN_ITERS   = 0 | 500   (regular TSDDR warmup before embedded training)
#   DR_PRETRAIN_PENALTY_MULT = 0.1  (penalty multiplier during pretrain, default 0.1)

using DecisionRulesExa
using ExaModels
using Flux
using Statistics, Random, Dates
using Wandb, Logging
using JLD2
using MadNLP

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

const LAYERS      = [128, 128]
const ACTIVATION  = sigmoid
const NUM_STAGES  = parse(Int, get(ENV, "DR_NUM_STAGES", "126"))
const NUM_EPOCHS  = parse(Int, get(ENV, "DR_NUM_EPOCHS", "80"))
const NUM_BATCHES = 100
const NUM_TRAIN_PER_BATCH = 1
const NUM_EVAL_SCENARIOS  = 4
const EVAL_EVERY  = 25
const LR          = 1f-3

const TARGET_PEN_ARG = :auto
const DEFICIT_COST   = 1e5
const load_scaler    = 0.6

const PRETRAIN_ITERS       = parse(Int, get(ENV, "DR_PRETRAIN_ITERS", "0"))
const PRETRAIN_PENALTY_MULT = parse(Float64, get(ENV, "DR_PRETRAIN_PENALTY_MULT", "0.1"))

const _PENALTY_MODE = get(ENV, "DR_PENALTY_SCHEDULE", "annealed")
const PENALTY_SCHEDULE = if _PENALTY_MODE == "annealed"
    [
        (1,   div(NUM_EPOCHS * NUM_BATCHES, 4), 0.1),
        (div(NUM_EPOCHS * NUM_BATCHES, 4) + 1, div(NUM_EPOCHS * NUM_BATCHES, 4) * 2, 1.0),
        (div(NUM_EPOCHS * NUM_BATCHES, 4) * 2 + 1, div(NUM_EPOCHS * NUM_BATCHES, 4) * 3, 10.0),
        (div(NUM_EPOCHS * NUM_BATCHES, 4) * 3 + 1, NUM_EPOCHS * NUM_BATCHES, 30.0),
    ]
else
    [(1, NUM_EPOCHS * NUM_BATCHES, 1.0)]
end

const SOLVER_KWARGS = (print_level = MadNLP.ERROR, tol = 1e-6, max_iter = 9000)

const _SCHED_TAG   = _PENALTY_MODE == "annealed" ? "-anneal" : "-const"
const _PRETRAIN_TAG = PRETRAIN_ITERS > 0 ? "-pre$(PRETRAIN_ITERS)" : ""
const RUN_NAME  = "$(CASE_NAME)-$(FORM_LABEL)-h$(NUM_STAGES)-embedded$(_SCHED_TAG)$(_PRETRAIN_TAG)-$(Dates.format(now(), "yyyymmdd-HHMMSS"))"
const MODEL_DIR = joinpath(CASE_DIR, FORM_LABEL, "models")
mkpath(MODEL_DIR)
const MODEL_PATH = joinpath(MODEL_DIR, RUN_NAME * ".jld2")

# ── Load data ─────────────────────────────────────────────────────────────────

@info "Loading power system data..."
power_data = load_power_data(PM_FILE)
@info "  nBus=$(power_data.nBus)  nGen=$(power_data.nGen)"

@info "Loading hydro data..."
hydro_data = load_hydro_data(HYDRO_FILE, INFLOW_FILE, power_data;
                              num_stages = NUM_STAGES * 10)
nHyd = hydro_data.nHyd
T    = NUM_STAGES
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

x0_init = Float32.([clamp(hydro_data.initial_volumes[r],
                           hydro_data.units[r].min_vol,
                           hydro_data.units[r].max_vol)
                    for r in 1:nHyd])

# ── Policy ────────────────────────────────────────────────────────────────────

Random.seed!(42)
policy = StateConditionedPolicy(nHyd, nHyd, nHyd, LAYERS;
                                activation   = ACTIVATION,
                                encoder_type = Flux.LSTM)

# ── Optional pretrain with regular TSDDR ──────────────────────────────────────

if PRETRAIN_ITERS > 0
    @info "Building regular DE for pretrain ($(PRETRAIN_ITERS) iters)..."
    prob_reg = build_hydro_de(power_data, hydro_data, T;
        backend        = nothing,
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

# ── Build embedded DE ─────────────────────────────────────────────────────────

@info "Building embedded $(T)-stage $(FORMULATION) hydro DE..."
prob_emb = build_embedded_hydro_de(policy, power_data, hydro_data, T;
    formulation    = FORMULATION,
    target_penalty = TARGET_PEN_ARG,
    deficit_cost   = DEFICIT_COST,
    demand_matrix  = demand_mat,
    load_scaler    = load_scaler,
)
@info "  nvar=$(prob_emb._nvar)  oracle_cons=$(length(prob_emb.target_con_range))"

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
        "layers"          => LAYERS,
        "activation"      => string(ACTIVATION),
        "target_penalty"  => "auto=$(round(resolved_pen; digits=2))",
        "target_penalty_l1" => "auto=$(round(resolved_pen_l1; digits=2))",
        "deficit_cost"    => DEFICIT_COST,
        "num_epochs"      => NUM_EPOCHS,
        "num_batches"     => NUM_BATCHES,
        "num_train_per_batch" => NUM_TRAIN_PER_BATCH,
        "num_eval_scenarios" => NUM_EVAL_SCENARIOS,
        "eval_every"      => EVAL_EVERY,
        "lr"              => LR,
        "penalty_schedule" => string(PENALTY_SCHEDULE),
        "pretrain_iters"  => PRETRAIN_ITERS,
        "pretrain_penalty_mult" => PRETRAIN_PENALTY_MULT,
    ),
)

# ── Rollout evaluation ────────────────────────────────────────────────────────

stage_demand = demand_mat === nothing ? nothing : demand_mat[1:1, :]
function _build_rollout_de()
    build_hydro_de(power_data, hydro_data, 1;
        backend        = nothing,
        float_type     = Float64,
        formulation    = FORMULATION,
        target_penalty = TARGET_PEN_ARG,
        deficit_cost   = DEFICIT_COST,
        demand_matrix  = stage_demand,
        load_scaler    = load_scaler,
    )
end

rollout_prob = _build_rollout_de()
rollout_pool = [_build_rollout_de() for _ in 1:NUM_EVAL_SCENARIOS]

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
    Array(hydro_solution(stage_prob, result).reservoir[:, end])

function hydro_objective_no_target_penalty(stage_prob, result)
    sol = hydro_solution(stage_prob, result)
    delta = Array(sol.delta)
    penalty_l2_cost = (resolved_pen / 2) * sum(abs2, delta)
    penalty_l1_cost = resolved_pen_l1 * sum(abs, delta)
    return result.objective - penalty_l2_cost - penalty_l1_cost
end

Random.seed!(8789)
eval_scenarios = [sample_scenario(hydro_data, T) for _ in 1:NUM_EVAL_SCENARIOS]

rollout_evaluation = RolloutEvaluation(
    rollout_prob, x0_init, eval_scenarios;
    horizon = T, n_uncertainty = nHyd,
    set_stage_parameters! = set_hydro_rollout_stage!,
    realized_state = hydro_realized_state,
    objective_no_target_penalty = hydro_objective_no_target_penalty,
    madnlp_kwargs = SOLVER_KWARGS,
    warmstart = true,
    stride = EVAL_EVERY,
    policy_state = :target,
    stage_problem_pool = rollout_pool,
    active_scenarios = NUM_EVAL_SCENARIOS,
)
realized_rollout_evaluation = RolloutEvaluation(
    rollout_prob, x0_init, eval_scenarios;
    horizon = T, n_uncertainty = nHyd,
    set_stage_parameters! = set_hydro_rollout_stage!,
    realized_state = hydro_realized_state,
    objective_no_target_penalty = hydro_objective_no_target_penalty,
    madnlp_kwargs = SOLVER_KWARGS,
    warmstart = true,
    stride = EVAL_EVERY,
    policy_state = :realized,
    stage_problem_pool = rollout_pool,
    active_scenarios = NUM_EVAL_SCENARIOS,
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
    adjust_hyperparameters = (iter, opt_state, n) -> begin
        mult = _schedule_value(PENALTY_SCHEDULE, iter, last(PENALTY_SCHEDULE)[3])
        if mult != current_penalty_mult[]
            current_penalty_mult[] = mult
            ρ_half_scaled = prob_emb.base_penalty_half * mult
            ρ_l1_scaled   = prob_emb.base_penalty_l1 * mult
            penalty_vals    = fill(ρ_half_scaled, T * nHyd)
            penalty_l1_vals = fill(ρ_l1_scaled,   T * nHyd)
            ExaModels.set_parameter!(prob_emb.core, prob_emb.p_penalty_half, penalty_vals)
            ExaModels.set_parameter!(prob_emb.core, prob_emb.p_penalty_l1,   penalty_l1_vals)
            @info "Penalty multiplier → $mult  (ρ/2 = $(round(ρ_half_scaled; digits=2)), λ_l1 = $(round(ρ_l1_scaled; digits=2)))"
        end
        return n
    end,
    record_loss = (iter, m, loss, tag) -> begin
        metrics = Dict{String, Any}(tag => loss, "batch" => iter)
        isfinite(loss) && push!(epoch_losses, loss)

        if iter % EVAL_EVERY == 0
            rollout_evaluation(iter, m)
            realized_rollout_evaluation(iter, m)
            metrics["metrics/rollout_objective_no_target_penalty"] =
                rollout_evaluation.last_objective_no_target_penalty
            metrics["metrics/rollout_target_violation_share"] =
                rollout_evaluation.last_violation_share
            metrics["metrics/rollout_realized_objective_no_target_penalty"] =
                realized_rollout_evaluation.last_objective_no_target_penalty
            metrics["metrics/rollout_realized_target_violation_share"] =
                realized_rollout_evaluation.last_violation_share
            metrics["metrics/rollout_n_ok"] =
                realized_rollout_evaluation.last_n_ok
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
