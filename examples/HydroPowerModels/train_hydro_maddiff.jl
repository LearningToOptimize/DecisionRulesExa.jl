# train_hydro_maddiff.jl
#
# HydroPowerModels training with MadDiff stage-wise implicit differentiation.
# Matches the DecisionRules.jl subproblems+annealing setup (best known config).
#
# Key difference from train_hydro_exa.jl: instead of solving one T-stage DE,
# this solves T independent single-stage NLPs per rollout and differentiates
# through each via MadDiff VJPs. The policy gradient combines envelope-theorem
# duals with solution sensitivities (paper Eq. 2.5).
#
# Usage:
#   julia --project train_hydro_maddiff.jl

using DecisionRulesExa
using DecisionRulesExa: _subproblem_rollout_loss, _make_subproblem_solver,
    SubproblemSolverState
using ExaModels
using Flux
using Statistics, Random, Dates
using Wandb, Logging
using JLD2
using MadNLP

const SCRIPT_DIR = dirname(@__FILE__)
include(joinpath(SCRIPT_DIR, "hydro_power_data.jl"))
include(joinpath(SCRIPT_DIR, "hydro_power_exa.jl"))
include(joinpath(SCRIPT_DIR, "hydro_subproblem_exa.jl"))

# ── Configuration ─────────────────────────────────────────────────────────────

const CASE_NAME   = "bolivia"
const FORMULATION = :dc
const FORM_LABEL  = "DCPPowerModel"

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
const GRAD_CLIP   = parse(Float32, get(ENV, "DR_GRAD_CLIP", "10"))

const TARGET_PEN_ARG = :auto
const DEFICIT_COST   = 1e5
const load_scaler    = 0.6

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

const SOLVER_KWARGS = (print_level = MadNLP.ERROR, tol = 1e-6, max_iter = 3000)

const _CLIP_TAG  = GRAD_CLIP > 0 ? "-clip$(Int(GRAD_CLIP))" : ""
const _SCHED_TAG = _PENALTY_MODE == "annealed" ? "-anneal" : "-const"
const RUN_NAME  = "$(CASE_NAME)-$(FORM_LABEL)-h$(NUM_STAGES)-maddiff-subproblems$(_CLIP_TAG)$(_SCHED_TAG)-$(Dates.format(now(), "yyyymmdd-HHMMSS"))"
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

# ── Build T single-stage subproblems ─────────────────────────────────────────

@info "Building $T single-stage hydro subproblems (formulation=$FORMULATION)..."

x0_init = Float32.([clamp(hydro_data.initial_volumes[r],
                           hydro_data.units[r].min_vol,
                           hydro_data.units[r].max_vol)
                    for r in 1:nHyd])

bundles = HydroSubproblemBundle[]
for t in 1:T
    demand_vec = if demand_mat !== nothing
        load_scaler .* demand_mat[t, :]
    else
        nothing
    end
    b = build_hydro_single_stage(
        power_data, hydro_data;
        formulation = FORMULATION,
        target_penalty = TARGET_PEN_ARG,
        target_penalty_l1 = :auto,
        deficit_cost = DEFICIT_COST,
        load_scaler = load_scaler,
        demand_vector = demand_vec !== nothing ? demand_vec ./ load_scaler : nothing,
    )
    push!(bundles, b)
end
subproblems = SubproblemDEProblem[b.prob for b in bundles]
@info "  $T subproblems built. State dim=$nHyd"

# ── Smoke test ───────────────────────────────────────────────────────────────

@info "Smoke test: solving stage 1 with mean inflows..."
w_mean_stage = mean_inflow(hydro_data, 1)
ExaModels.set_parameter!(bundles[1].prob.core, bundles[1].prob.p_x_in, Float64.(x0_init))
ExaModels.set_parameter!(bundles[1].prob.core, bundles[1].prob.p_w, w_mean_stage)
ExaModels.set_parameter!(bundles[1].prob.core, bundles[1].prob.p_xhat, Float64.(x0_init))
smoke_solver = _make_subproblem_solver(bundles[1].prob.model, SOLVER_KWARGS)
smoke_result = MadNLP.solve!(smoke_solver.solver; SOLVER_KWARGS..., print_level = MadNLP.WARN)
@info "  Status: $(smoke_result.status)  Objective: $(round(smoke_result.objective; digits=4))"
solve_succeeded(smoke_result) || @warn "Smoke test did not converge"

# ── Policy ───────────────────────────────────────────────────────────────────

policy = StateConditionedPolicy(nHyd, nHyd, nHyd, LAYERS;
                                activation   = ACTIVATION,
                                encoder_type = Flux.LSTM)
@info "Policy: StateConditionedPolicy, input=$(2*nHyd), output=$nHyd, layers=$LAYERS"

# ── Rollout evaluation (reuse DE-based rollout from existing code) ───────────

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

resolved_pen_l1 = bundles[1].base_penalty_l1

function set_hydro_rollout_stage!(stage_prob, state_in, wt, target, stage)
    ExaModels.set_parameter!(stage_prob.core, stage_prob.p_x0, state_in)
    ExaModels.set_parameter!(stage_prob.core, stage_prob.p_inflow, wt)
    if demand_mat !== nothing
        set_demand!(stage_prob, load_scaler .* demand_mat[stage:stage, :])
    end
    ExaModels.set_parameter!(stage_prob.core, stage_prob.p_target, target)
    return stage_prob
end

hydro_realized_state_fn(stage_prob, result) =
    Array(hydro_solution(stage_prob, result).reservoir[:, end])

function hydro_objective_no_penalty(stage_prob, result)
    sol = hydro_solution(stage_prob, result)
    delta = Array(sol.delta)
    penalty_l2 = (resolved_pen / 2) * sum(abs2, delta)
    penalty_l1 = resolved_pen_l1 * sum(abs, delta)
    return result.objective - penalty_l2 - penalty_l1
end

rollout_prob = _build_rollout_de()
n_rollout_pool = max(1, NUM_EVAL_SCENARIOS)
rollout_pool = [_build_rollout_de() for _ in 1:n_rollout_pool]

Random.seed!(8789)
eval_scenarios = [sample_scenario(hydro_data, T) for _ in 1:NUM_EVAL_SCENARIOS]

rollout_evaluation = RolloutEvaluation(
    rollout_prob,
    x0_init,
    eval_scenarios;
    horizon = T,
    n_uncertainty = nHyd,
    set_stage_parameters! = set_hydro_rollout_stage!,
    realized_state = hydro_realized_state_fn,
    objective_no_target_penalty = hydro_objective_no_penalty,
    madnlp_kwargs = SOLVER_KWARGS,
    warmstart = true,
    stride = EVAL_EVERY,
    policy_state = :realized,
    stage_problem_pool = rollout_pool,
    active_scenarios = NUM_EVAL_SCENARIOS,
)

# ── W&B logging ──────────────────────────────────────────────────────────────

lg = WandbLogger(
    project = "RL",
    name    = RUN_NAME,
    save_code = false,
    config  = Dict(
        "case"            => CASE_NAME,
        "formulation"     => FORM_LABEL,
        "num_stages"      => T,
        "layers"          => LAYERS,
        "activation"      => string(ACTIVATION),
        "target_penalty"  => "auto=$(round(resolved_pen; digits=2))",
        "target_penalty_l1" => "auto=$(round(resolved_pen_l1; digits=2))",
        "deficit_cost"    => DEFICIT_COST,
        "num_epochs"      => NUM_EPOCHS,
        "num_batches"     => NUM_BATCHES,
        "num_train_per_batch" => NUM_TRAIN_PER_BATCH,
        "lr"              => LR,
        "grad_clip"       => GRAD_CLIP,
        "training_method" => "maddiff-subproblems",
        "penalty_schedule" => string(PENALTY_SCHEDULE),
        "load_scaler"     => load_scaler,
    ),
)

# ── Training ─────────────────────────────────────────────────────────────────

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

train_subproblem_tsddr(
    policy,
    x0_init,
    subproblems,
    () -> Float32.(sample_scenario(hydro_data, T));
    horizon              = T,
    num_batches          = NUM_EPOCHS * NUM_BATCHES,
    num_train_per_batch  = NUM_TRAIN_PER_BATCH,
    optimizer            = GRAD_CLIP > 0 ?
                           Flux.Optimisers.OptimiserChain(
                               Flux.Optimisers.ClipGrad(GRAD_CLIP),
                               Flux.Adam(LR),
                           ) : Flux.Adam(LR),
    madnlp_kwargs        = SOLVER_KWARGS,
    warmstart            = true,
    adjust_hyperparameters = (iter, opt_state, subs, n) -> begin
        mult = _schedule_value(PENALTY_SCHEDULE, iter, last(PENALTY_SCHEDULE)[3])
        if mult != current_penalty_mult[]
            current_penalty_mult[] = mult
            for b in bundles
                set_penalty_mult!(b, mult)
            end
            @info "Penalty multiplier → $mult"
        end
        return n
    end,
    record_loss          = (iter, m, loss, tag) -> begin
        metrics = Dict{String, Any}(tag => loss, "batch" => iter)
        isfinite(loss) && push!(epoch_losses, loss)

        if iter % EVAL_EVERY == 0
            rollout_evaluation(iter, m)
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
