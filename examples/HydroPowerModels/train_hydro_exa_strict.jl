# train_hydro_exa_strict.jl
#
# Strict-mode regular DE training with ExaModels + MadNLP (AC polar OPF).
# Uses HydroReachablePolicy (sigmoid-bounded to one-stage reachable set)
# with strict_targets=true (delta variables fixed to zero, no target penalty).
#
# Key insight: ordinary regular DE target generation is open-loop after x0, so
# strict equality is usually unsafe: the optimizer may discover realized states
# different from the target path, and later targets were not computed from those
# realized states. Here the reachable policy is rolled out from the true x0 and
# uses the previous target as the next policy state. Since every target is
# one-stage reachable from the previous target, the full target trajectory is
# feasible by induction and strict regular DE is valid.
#
# Environment variables:
#   DR_ENCODER_LAYERS       = "128,128"  (LSTM encoder layer sizes)
#   DR_HEAD_LAYERS          = ""         (state-conditioned target-head hidden sizes)
#   DR_LAYERS               = "128,128"  (legacy alias for DR_ENCODER_LAYERS)
#   DR_NUM_STAGES           = "126"
#   DR_NUM_ROLLOUT_STAGES   = "96"
#   DR_NUM_EPOCHS           = "80"
#   DR_NUM_BATCHES          = "100"
#   DR_NUM_TRAIN_PER_BATCH  = "1"        (sampled trajectories per gradient step)
#   DR_NUM_TRAIN_SCHEDULE   = ""         (optional "lo:hi:n,..." schedule)
#   DR_CONTEXT              = ""         (""/"none", "phase", or "phase+progress")
#   DR_NUM_EVAL_SCENARIOS   = "4"        (fixed held-out rollout-selection scenarios)
#   DR_EVAL_SCHEDULE        = ""         (optional "lo:hi:n,..." active-eval schedule)
#   DR_EVAL_EVERY           = "50"
#   DR_SAVE_METRIC          = "training" ("training" or "rollout")
#   DR_LR                   = "0.001"
#   DR_LR_FINAL             = DR_LR      (cosine-decay final learning rate)
#   DR_LR_WARMUP            = "0"        (linear warmup iterations from DR_LR/100)
#   DR_PRETRAINED_MODEL     = ""         (optional diagnostic warmstart checkpoint)
#   DR_REACTIVE_DEFICIT     = "free"     ("free", "hard", or a finite penalty)
#   DR_GRAD_CLIP            = "0"
#   DR_MAX_ITER             = "9000"
#
# Reproducible recipes:
#
#   Historical baseline (all defaults, byte-compatible training semantics):
#     julia --project -t auto train_hydro_exa_strict.jl
#
#   Fresh fast/fair candidate (headline timing starts at this command):
#     DR_REACTIVE_DEFICIT=hard DR_SAVE_METRIC=rollout \
#     DR_NUM_EVAL_SCENARIOS=8 DR_EVAL_EVERY=100 \
#     DR_NUM_TRAIN_PER_BATCH=2 \
#     DR_LR=5e-4 DR_LR_FINAL=5e-5 DR_LR_WARMUP=100 \
#     DR_NUM_EPOCHS=25 DR_NUM_BATCHES=100 \
#     DR_HEAD_LAYERS=128,128 \
#     julia --project -t auto train_hydro_exa_strict.jl
#
#   Progressive-sampling Phase B (spend samples only after coarse movement):
#     DR_REACTIVE_DEFICIT=hard DR_SAVE_METRIC=rollout \
#     DR_NUM_EVAL_SCENARIOS=12 DR_EVAL_SCHEDULE=1:800:4,801:1400:8,1401:1800:12 \
#     DR_NUM_TRAIN_PER_BATCH=1 DR_NUM_TRAIN_SCHEDULE=1:800:1,801:1400:2,1401:1800:4 \
#     DR_LR=7e-4 DR_LR_FINAL=2e-5 DR_LR_WARMUP=100 \
#     DR_NUM_EPOCHS=18 DR_NUM_BATCHES=100 \
#     DR_HEAD_LAYERS=256,256 \
#     julia --project -t auto train_hydro_exa_strict.jl
#
#   Diagnostic warmstart (not a clean headline unless parent time is counted):
#     DR_PRETRAINED_MODEL=<checkpoint.jld2> DR_SAVE_METRIC=rollout \
#     DR_NUM_TRAIN_PER_BATCH=4 DR_LR=1e-4 DR_LR_FINAL=1e-5 DR_LR_WARMUP=50 \
#     DR_NUM_EPOCHS=10 DR_HEAD_LAYERS=<matching head> \
#     julia --project -t auto train_hydro_exa_strict.jl
#
# Usage:
#   julia --project -t auto train_hydro_exa_strict.jl

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
include(joinpath(SCRIPT_DIR, "hydro_training_utils.jl"))
include(joinpath(SCRIPT_DIR, "hydro_power_data.jl"))
include(joinpath(SCRIPT_DIR, "hydro_power_exa.jl"))
include(joinpath(SCRIPT_DIR, "hydro_reachable_policy.jl"))

# ── Configuration ─────────────────────────────────────────────────────────────

const CASE_NAME   = "bolivia"
const FORMULATION = :ac_polar
const FORM_LABEL  = FORMULATION === :ac_polar ? "ACPPowerModel" : "DCPPowerModel"

const CASE_DIR    = joinpath(SCRIPT_DIR, CASE_NAME)
const PM_FILE     = joinpath(CASE_DIR, "PowerModels.json")
const HYDRO_FILE  = joinpath(CASE_DIR, "hydro.json")
const INFLOW_FILE = joinpath(CASE_DIR, "inflows.csv")
const DEMAND_FILE = joinpath(CASE_DIR, "demand.csv")

function parse_reactive_deficit_cost(raw::AbstractString)
    s = lowercase(strip(raw))
    if s == "free"
        return nothing
    elseif s == "hard"
        return Inf
    end
    cost = parse(Float64, s)
    (isnan(cost) || cost < 0) &&
        throw(ArgumentError("DR_REACTIVE_DEFICIT must be free, hard, or a finite nonnegative cost; got $raw"))
    return cost
end

function value_tag(x)
    return replace(replace(string(x), "." => "p"), "-" => "m")
end

function parse_int_schedule(raw::AbstractString, name::AbstractString)
    s = strip(raw)
    if isempty(s) || lowercase(s) in ("fixed", "none", "nothing")
        return nothing
    end

    schedule = Tuple{Int, Int, Int}[]
    for item in split(s, r"[,;]")
        token = strip(item)
        isempty(token) && continue

        parts = split(token, ":")
        local lo::Int
        local hi::Int
        local value::Int
        if length(parts) == 3
            lo = parse(Int, strip(parts[1]))
            hi = parse(Int, strip(parts[2]))
            value = parse(Int, strip(parts[3]))
        elseif length(parts) == 2 && occursin("-", parts[1])
            bounds = split(parts[1], "-")
            length(bounds) == 2 ||
                throw(ArgumentError("$name schedule entry '$token' must be lo:hi:value or lo-hi:value"))
            lo = parse(Int, strip(bounds[1]))
            hi = parse(Int, strip(bounds[2]))
            value = parse(Int, strip(parts[2]))
        else
            throw(ArgumentError("$name schedule entry '$token' must be lo:hi:value or lo-hi:value"))
        end

        lo >= 1 || throw(ArgumentError("$name schedule lower bound must be >= 1 in '$token'"))
        hi >= lo || throw(ArgumentError("$name schedule upper bound must be >= lower bound in '$token'"))
        value >= 1 || throw(ArgumentError("$name schedule value must be >= 1 in '$token'"))
        push!(schedule, (lo, hi, value))
    end

    isempty(schedule) && return nothing
    sort!(schedule; by = first)
    last_hi = 0
    for (lo, hi, _) in schedule
        lo > last_hi || throw(ArgumentError("$name schedule has overlapping entries near iteration $lo"))
        last_hi = hi
    end
    return schedule
end

function schedule_value(schedule, iter::Int, default::Int)
    isnothing(schedule) && return default
    for (lo, hi, value) in schedule
        lo <= iter <= hi && return value
    end
    return default
end

function schedule_tag(schedule, prefix::AbstractString)
    isnothing(schedule) && return ""
    vals = unique(last.(schedule))
    return "-$(prefix)$(join(vals, "_"))"
end

const ENCODER_LAYERS = parse_layers(get(ENV, "DR_ENCODER_LAYERS", get(ENV, "DR_LAYERS", "128,128")))
const HEAD_LAYERS    = parse_layers(get(ENV, "DR_HEAD_LAYERS", ""))
const ACTIVATION  = sigmoid
const NUM_STAGES  = parse(Int, get(ENV, "DR_NUM_STAGES", "126"))
const NUM_ROLLOUT_STAGES = parse(Int, get(ENV, "DR_NUM_ROLLOUT_STAGES", "96"))
const NUM_EPOCHS  = parse(Int, get(ENV, "DR_NUM_EPOCHS", "80"))
const NUM_BATCHES = parse(Int, get(ENV, "DR_NUM_BATCHES", "100"))
const NUM_TRAIN_PER_BATCH = parse(Int, get(ENV, "DR_NUM_TRAIN_PER_BATCH", "1"))
const NUM_TRAIN_SCHEDULE  = parse_int_schedule(get(ENV, "DR_NUM_TRAIN_SCHEDULE", ""), "DR_NUM_TRAIN_SCHEDULE")
const CONTEXT_MODE = canonical_context_mode(get(ENV, "DR_CONTEXT", ""))
const CONTEXT_PERIOD = countlines(INFLOW_FILE)
const CONTEXT_HORIZON = NUM_STAGES
const STAGE_CONTEXT = build_stage_context(CONTEXT_MODE, CONTEXT_HORIZON, CONTEXT_PERIOD)
const N_CONTEXT = isnothing(STAGE_CONTEXT) ? 0 : size(STAGE_CONTEXT, 1)
const NUM_EVAL_SCENARIOS  = parse(Int, get(ENV, "DR_NUM_EVAL_SCENARIOS", "4"))
const EVAL_SCHEDULE       = parse_int_schedule(get(ENV, "DR_EVAL_SCHEDULE", ""), "DR_EVAL_SCHEDULE")
if !isnothing(EVAL_SCHEDULE) && maximum(last.(EVAL_SCHEDULE)) > NUM_EVAL_SCENARIOS
    throw(ArgumentError("DR_EVAL_SCHEDULE cannot exceed DR_NUM_EVAL_SCENARIOS=$(NUM_EVAL_SCENARIOS)"))
end
const EVAL_EVERY  = parse(Int, get(ENV, "DR_EVAL_EVERY", "50"))
const SAVE_METRIC = lowercase(strip(get(ENV, "DR_SAVE_METRIC", "training")))
SAVE_METRIC in ("training", "rollout") ||
    throw(ArgumentError("DR_SAVE_METRIC must be training or rollout; got $SAVE_METRIC"))
const LR          = parse(Float32, get(ENV, "DR_LR", "0.001"))
const LR_FINAL    = parse(Float32, get(ENV, "DR_LR_FINAL", string(LR)))
const LR_WARMUP   = parse(Int, get(ENV, "DR_LR_WARMUP", "0"))
const PRE_TRAINED = strip(get(ENV, "DR_PRETRAINED_MODEL", ""))
const HAS_PRETRAINED = !(isempty(PRE_TRAINED) || lowercase(PRE_TRAINED) == "nothing")
const REACTIVE_DEFICIT_RAW = get(ENV, "DR_REACTIVE_DEFICIT", "free")
const REACTIVE_DEFICIT_COST = parse_reactive_deficit_cost(REACTIVE_DEFICIT_RAW)
const GRAD_CLIP   = parse(Float32, get(ENV, "DR_GRAD_CLIP", "0"))

const TARGET_PEN_ARG = :auto
const HYDRO_TARGET_PENALTY_MULT = 8.0
const DEFICIT_COST   = 1e5
const USE_GPU        = true
const load_scaler    = 0.6
const NUM_WORKERS    = 1

const ROLLOUT_PARALLEL = parse(Bool, get(ENV, "DR_ROLLOUT_PARALLEL", "false"))

const MAX_ITER = parse(Int, get(ENV, "DR_MAX_ITER", "9000"))
const SOLVER_KWARGS = (print_level = MadNLP.ERROR, tol = 1e-6, max_iter = MAX_ITER)

const _CLIP_TAG  = GRAD_CLIP > 0 ? "-clip$(Int(GRAD_CLIP))" : ""
const _ENC_TAG   = ENCODER_LAYERS == [128, 128] ? "" : "-E$(join(ENCODER_LAYERS, "_"))"
const _HEAD_TAG  = isempty(HEAD_LAYERS) ? "-Hlinear" : "-H$(join(HEAD_LAYERS, "_"))"
const _NT_TAG    = isnothing(NUM_TRAIN_SCHEDULE) ?
                   (NUM_TRAIN_PER_BATCH > 1 ? "-nt$(NUM_TRAIN_PER_BATCH)" : "") :
                   schedule_tag(NUM_TRAIN_SCHEDULE, "ntsch")
const _CTX_TAG   = context_run_tag(CONTEXT_MODE)
const _EV_TAG    = schedule_tag(EVAL_SCHEDULE, "evsch")
const _SAVE_TAG  = SAVE_METRIC == "rollout" ? "-rollout" : ""
const _WARM_TAG  = HAS_PRETRAINED ? "-warm" : ""
const _RQ_TAG    = REACTIVE_DEFICIT_COST === nothing ? "" :
                   isinf(Float64(REACTIVE_DEFICIT_COST)) ? "-rqhard" :
                   "-rq$(value_tag(REACTIVE_DEFICIT_COST))"
const RUN_NAME  = "$(CASE_NAME)-$(FORM_LABEL)-h$(NUM_STAGES)-r$(NUM_ROLLOUT_STAGES)-deteq-strict-gpu$(_CLIP_TAG)$(_ENC_TAG)$(_HEAD_TAG)$(_NT_TAG)$(_CTX_TAG)$(_EV_TAG)$(_SAVE_TAG)$(_WARM_TAG)$(_RQ_TAG)-$(Dates.format(now(), "yyyymmdd-HHMMSS"))"
const MODEL_DIR = joinpath(CASE_DIR, FORM_LABEL, "models")
mkpath(MODEL_DIR)
const MODEL_PATH = joinpath(MODEL_DIR, RUN_NAME * ".jld2")
const TOTAL_ITERS = NUM_EPOCHS * NUM_BATCHES

function lr_schedule(iter::Int, total_iters::Int)
    if iter <= LR_WARMUP
        return LR * (0.01f0 + 0.99f0 * Float32(iter) / Float32(max(LR_WARMUP, 1)))
    end
    ρ = clamp((iter - LR_WARMUP) / max(total_iters - LR_WARMUP, 1), 0.0, 1.0)
    return LR_FINAL + 0.5f0 * (LR - LR_FINAL) * (1f0 + cos(Float32(pi * ρ)))
end

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

# ── Build ExaModels DE (strict: delta variables fixed to zero) ────────────────

resolved_pen = TARGET_PEN_ARG === :auto ?
               auto_target_penalty(power_data, hydro_data) :
               Float64(TARGET_PEN_ARG)
@info "Auto target penalty: ρ=$(round(resolved_pen; digits=2)) (not used — strict mode)"

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
        strict_targets = true,
        reactive_deficit_cost = REACTIVE_DEFICIT_COST,
    )
end

@info "Building strict $(T)-stage ExaModels DE (formulation=$FORMULATION)..."
prob = _build_de()

@info "Building $(NUM_WORKERS)-worker problem pool..."
problem_pool = [(prob, prob.p_x0, prob.p_target, prob.p_inflow)]
for i in 2:NUM_WORKERS
    p = _build_de()
    push!(problem_pool, (p, p.p_x0, p.p_target, p.p_inflow))
end
@info "  Pool ready: $(NUM_WORKERS) independent strict DE instances on GPU"

x0_init = Float32.([clamp(hydro_data.initial_volumes[r],
                           hydro_data.units[r].min_vol,
                           hydro_data.units[r].max_vol)
                    for r in 1:nHyd])
target_lower = Float32.([h.min_vol for h in hydro_data.units])
target_upper = Float32.([h.max_vol for h in hydro_data.units])

# ── Policy (HydroReachablePolicy — one-stage reachable sigmoid bounds) ────────

Random.seed!(42)
base_policy = hydro_reachable_policy(hydro_data, ENCODER_LAYERS;
                                     activation       = ACTIVATION,
                                     encoder_type     = Flux.LSTM,
                                     combiner_layers  = HEAD_LAYERS,
                                     n_context        = N_CONTEXT)
policy = isnothing(STAGE_CONTEXT) ? base_policy : ContextualPolicy(base_policy, STAGE_CONTEXT)

if HAS_PRETRAINED
    @info "Loading pre-trained policy checkpoint" PRE_TRAINED
    load_stateconditioned_policy!(policy, JLD2.load(PRE_TRAINED, "model_state"))
    Flux.reset!(policy)
end

"""
    rollout_reachable_targets(policy, x0, w_flat, T, nHyd) -> Vector{Float64}

Roll out a [`HydroReachablePolicy`] target trajectory before solving the strict
regular deterministic equivalent.

The regular DE receives an external target vector, so it cannot query the policy
inside the NLP. This helper constructs that vector in a way that preserves
strict-mode feasibility: it starts from the known feasible initial state `x0`,
feeds `[w_t; previous_target]` to the policy, and stores each reachable target as
the next previous state. By induction, every target in the returned trajectory is
reachable from the prior target under the sampled inflow path.

# Arguments
- `policy`: reachable hydro policy with input `[inflow; previous_state]`.
- `x0`: initial reservoir state.
- `w_flat`: stage-major flat inflow vector of length `T * nHyd`.
- `T::Int`: number of stages.
- `nHyd::Int`: number of hydro reservoir state components.

# Returns
- `Vector{Float64}`: stage-major target trajectory suitable for
  `ExaModels.set_parameter!(prob.core, prob.p_target, targets)`.

# Examples
```julia
targets = rollout_reachable_targets(policy, x0_init, mean_inflow(hydro_data, T), T, nHyd)
```
"""
function rollout_reachable_targets(policy, x0, w_flat, T, nHyd)
    Flux.reset!(policy)
    prev = x0
    targets = Vector{Vector{Float32}}(undef, T)
    for t in 1:T
        wt = Float32.(view(w_flat, ((t - 1) * nHyd + 1):(t * nHyd)))
        target = policy(vcat(wt, prev))
        targets[t] = Float32.(target)
        prev = targets[t]
    end
    return Float64.(vcat(targets...))
end

# ── Smoke test ────────────────────────────────────────────────────────────────

w_mean = mean_inflow(hydro_data, T)
xhat_mean = rollout_reachable_targets(policy, x0_init, w_mean, T, nHyd)
ExaModels.set_parameter!(prob.core, prob.p_x0,     x0_init)
ExaModels.set_parameter!(prob.core, prob.p_inflow,  w_mean)
ExaModels.set_parameter!(prob.core, prob.p_target,  xhat_mean)
prepare_solve!(prob, x0_init, w_mean, xhat_mean)
@info "Smoke test: solving strict DE with mean inflows and reachable policy targets..."
result0 = MadNLP.madnlp(prob.model; SOLVER_KWARGS..., print_level = MadNLP.WARN)
@info "  Status: $(result0.status)   Objective: $(round(result0.objective; digits=4))"
isfinite(result0.objective) || error("Smoke test returned non-finite objective")
solve_succeeded(result0) || @warn "Smoke test did not fully converge; proceeding anyway"

Flux.reset!(policy)

if USE_GPU
    policy  = policy isa ContextualPolicy ?
              ContextualPolicy(CUDA.cu(policy.policy), CUDA.cu(policy.context)) :
              CUDA.cu(policy)
    Flux.reset!(policy)
    x0_init = CUDA.cu(x0_init)
    @info "Policy and x0 moved to GPU"
end

@info "Strict Exa training config" RUN_NAME NUM_STAGES NUM_ROLLOUT_STAGES NUM_EPOCHS NUM_BATCHES NUM_TRAIN_PER_BATCH NUM_TRAIN_SCHEDULE CONTEXT_MODE CONTEXT_PERIOD CONTEXT_HORIZON N_CONTEXT NUM_EVAL_SCENARIOS EVAL_SCHEDULE EVAL_EVERY SAVE_METRIC LR LR_FINAL LR_WARMUP PRE_TRAINED REACTIVE_DEFICIT_RAW REACTIVE_DEFICIT_COST GRAD_CLIP MAX_ITER

function checkpoint_policy_state(m)
    if m isa ContextualPolicy
        return Flux.state(ContextualPolicy(cpu(m.policy), Array(m.context)))
    end
    return Flux.state(cpu(m))
end

# ── W&B logging ───────────────────────────────────────────────────────────────

lg = WandbLogger(
    project = "RL",
    name    = RUN_NAME,
    save_code = false,
    config  = Dict(
        "case"            => CASE_NAME,
        "formulation"     => FORM_LABEL,
        "method"          => "deteq-strict",
        "num_stages"      => T,
        "num_rollout_stages" => T_ROLLOUT,
        "encoder_layers"  => ENCODER_LAYERS,
        "head_layers"     => HEAD_LAYERS,
        "activation"      => string(ACTIVATION),
        "target_penalty"  => "strict (disabled)",
        "deficit_cost"    => DEFICIT_COST,
        "reactive_deficit" => REACTIVE_DEFICIT_RAW,
        "reactive_deficit_cost" => string(REACTIVE_DEFICIT_COST),
        "num_epochs"      => NUM_EPOCHS,
        "num_batches"     => NUM_BATCHES,
        "num_train_per_batch" => NUM_TRAIN_PER_BATCH,
        "num_train_schedule" => string(something(NUM_TRAIN_SCHEDULE, "fixed")),
        "context_mode"    => isempty(CONTEXT_MODE) ? "none" : CONTEXT_MODE,
        "context_period"  => CONTEXT_PERIOD,
        "context_horizon" => CONTEXT_HORIZON,
        "n_context"       => N_CONTEXT,
        "num_eval_scenarios" => NUM_EVAL_SCENARIOS,
        "eval_schedule"   => string(something(EVAL_SCHEDULE, "fixed")),
        "eval_every"      => EVAL_EVERY,
        "save_metric"     => SAVE_METRIC,
        "lr"              => LR,
        "lr_final"        => LR_FINAL,
        "lr_warmup"       => LR_WARMUP,
        "pre_trained_model" => PRE_TRAINED,
        "grad_clip"       => GRAD_CLIP,
        "backend"         => USE_GPU ? "GPU" : "CPU",
        "load_scaler"     => load_scaler,
        "strict_targets"  => true,
        "policy_type"     => N_CONTEXT == 0 ? "HydroReachablePolicy" : "ContextualPolicy{HydroReachablePolicy}",
        "num_workers"     => NUM_WORKERS,
    ),
)

# ── Training ──────────────────────────────────────────────────────────────────

Random.seed!(8788)

best_obj = Inf
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
        strict_targets = true,
        reactive_deficit_cost = REACTIVE_DEFICIT_COST,
    )
end
rollout_prob = _build_rollout_de()
n_rollout_pool = max(NUM_WORKERS, NUM_EVAL_SCENARIOS)
rollout_pool = ROLLOUT_PARALLEL ? [_build_rollout_de() for _ in 1:n_rollout_pool] : []
@info "Rollout evaluation: $(ROLLOUT_PARALLEL ? "parallel ($(n_rollout_pool) stage-problem copies)" : "sequential")"

function set_hydro_rollout_stage!(stage_prob, state_in, wt, target, stage)
    ExaModels.set_parameter!(stage_prob.core, stage_prob.p_x0, state_in)
    ExaModels.set_parameter!(stage_prob.core, stage_prob.p_inflow, wt)
    if demand_mat !== nothing
        set_demand!(stage_prob, load_scaler .* demand_mat[stage:stage, :])
    end
    ExaModels.set_parameter!(stage_prob.core, stage_prob.p_target, target)
    prepare_solve!(stage_prob, state_in, wt, target)
    return stage_prob
end

hydro_realized_state(stage_prob, result) =
    hydro_solution(stage_prob, result).reservoir[:, end]

const _min_vols = Float64.([h.min_vol for h in hydro_data.units])
const _max_vols = Float64.([h.max_vol for h in hydro_data.units])
const _min_vols_dev = USE_GPU ? CUDA.cu(_min_vols) : _min_vols
const _max_vols_dev = USE_GPU ? CUDA.cu(_max_vols) : _max_vols

hydro_objective_no_target_penalty(stage_prob, result) = result.objective

Random.seed!(8789)
eval_scenarios = [sample_scenario(hydro_data, T_ROLLOUT) for _ in 1:NUM_EVAL_SCENARIOS]
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
    active_scenarios = NUM_EVAL_SCENARIOS,
    state_bounds = (_min_vols_dev, _max_vols_dev),
)

current_num_train = Ref(NUM_TRAIN_PER_BATCH)
current_eval_scenarios = Ref(schedule_value(EVAL_SCHEDULE, 1, NUM_EVAL_SCENARIOS))
rollout_evaluation.active_scenarios = current_eval_scenarios[]

if SAVE_METRIC == "rollout"
    rollout_evaluation(EVAL_EVERY, policy)
    best_obj = rollout_evaluation.last_objective_no_target_penalty
    @info "Initial rollout evaluation (checkpoint baseline)" best_obj rollout_evaluation.last_violation_share rollout_evaluation.last_n_ok
end

Random.seed!(8788)

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

train_tsddr(
    policy,
    x0_init,
    prob,
    prob.p_x0,
    prob.p_target,
    prob.p_inflow,
    () -> sample_scenario(hydro_data, T);
    num_batches          = TOTAL_ITERS,
    num_train_per_batch  = NUM_TRAIN_PER_BATCH,
    optimizer            = GRAD_CLIP > 0 ?
                           Flux.Optimisers.OptimiserChain(
                               Flux.Optimisers.ClipGrad(GRAD_CLIP),
                               Flux.Adam(LR),
                           ) : Flux.Adam(LR),
    madnlp_kwargs        = SOLVER_KWARGS,
    warmstart            = true,
    problem_pool         = problem_pool,
    batch_diagnostics    = (iter, stats) -> begin
        last_batch_stats[] = stats
        n_ok = get(stats, "n_ok", 0)
        n_total = get(stats, "n_total", 0)
        if n_ok < n_total
            @warn "Training solve failures at iter $iter" n_ok n_total status_counts=get(stats, "status_counts", nothing) failure_counts=get(stats, "failure_counts", nothing) retry_counts=get(stats, "retry_counts", nothing)
        elseif iter % 10 == 0
            @info "Training solve status at iter $iter" n_ok n_total status_counts=get(stats, "status_counts", nothing) retry_counts=get(stats, "retry_counts", nothing)
        end
    end,
    adjust_hyperparameters = (LR_WARMUP == 0 && LR_FINAL == LR) ?
        ((iter, opt_state, n) -> begin
            n_next = schedule_value(NUM_TRAIN_SCHEDULE, iter, n)
            if n_next != current_num_train[]
                current_num_train[] = n_next
                @info "num_train_per_batch → $n_next"
            end
            n_eval = schedule_value(EVAL_SCHEDULE, iter, NUM_EVAL_SCENARIOS)
            if n_eval != current_eval_scenarios[]
                current_eval_scenarios[] = n_eval
                rollout_evaluation.active_scenarios = n_eval
                @info "rollout active scenarios → $n_eval"
            end
            n_next
        end) :
        ((iter, opt_state, n) -> begin
            Flux.Optimisers.adjust!(opt_state, lr_schedule(iter, TOTAL_ITERS))
            n_next = schedule_value(NUM_TRAIN_SCHEDULE, iter, n)
            if n_next != current_num_train[]
                current_num_train[] = n_next
                @info "num_train_per_batch → $n_next"
            end
            n_eval = schedule_value(EVAL_SCHEDULE, iter, NUM_EVAL_SCENARIOS)
            if n_eval != current_eval_scenarios[]
                current_eval_scenarios[] = n_eval
                rollout_evaluation.active_scenarios = n_eval
                @info "rollout active scenarios → $n_eval"
            end
            n_next
        end),
    record_loss          = (iter, m, loss, tag) -> begin
        metrics = Dict{String, Any}(
            tag => loss,
            "batch" => iter,
            "metrics/lr" => lr_schedule(iter, TOTAL_ITERS),
            "metrics/num_train_per_batch" => current_num_train[],
            "metrics/active_eval_scenarios" => current_eval_scenarios[],
        )
        _merge_batch_stats!(metrics, last_batch_stats[])
        isfinite(loss) && push!(epoch_losses, loss)

        if iter % EVAL_EVERY == 0
            rollout_evaluation(iter, m)
            metrics["metrics/rollout_objective_no_target_penalty"] =
                rollout_evaluation.last_objective_no_target_penalty
            metrics["metrics/rollout_objective_no_deficit"] =
                rollout_evaluation.last_objective_no_target_penalty
            metrics["metrics/rollout_target_violation_share"] =
                rollout_evaluation.last_violation_share
            metrics["metrics/rollout_n_ok"] =
                rollout_evaluation.last_n_ok
            if SAVE_METRIC == "rollout"
                rollout_score = rollout_evaluation.last_objective_no_target_penalty
                if isfinite(rollout_score) && rollout_score < best_obj
                    global best_obj = rollout_score
                    jldsave(MODEL_PATH; model_state = checkpoint_policy_state(m))
                    @info "  -> New best rollout: $(round(rollout_score; digits=4)) -- saved $MODEL_PATH"
                end
            end
        end

        batch_in_epoch = (iter - 1) % NUM_BATCHES + 1
        if batch_in_epoch == NUM_BATCHES
            epoch     = (iter - 1) ÷ NUM_BATCHES + 1
            mean_loss = isempty(epoch_losses) ? NaN : mean(epoch_losses)
            n_ok      = length(epoch_losses)
            empty!(epoch_losses)
            Wandb.log(lg, Dict("metrics/epoch_objective" => mean_loss, "epoch" => epoch))
            @info "Epoch $epoch/$NUM_EPOCHS  mean=$(round(mean_loss; digits=2))  ok=$n_ok/$NUM_BATCHES"
            if SAVE_METRIC == "training" && isfinite(mean_loss) && mean_loss < best_obj
                global best_obj = mean_loss
                jldsave(MODEL_PATH; model_state = checkpoint_policy_state(m))
                @info "  → New best: $(round(mean_loss; digits=4)) — saved $MODEL_PATH"
            end
        end
        Wandb.log(lg, metrics)
        return false
    end,
)

close(lg)
@info "Done. Best model saved to: $(MODEL_PATH)"
