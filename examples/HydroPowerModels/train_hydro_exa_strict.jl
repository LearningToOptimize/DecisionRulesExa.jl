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
using StableRNGs
using ExaModels
using Flux
using Statistics, Random, Dates
using Logging   # Wandb loaded conditionally below (DR_ENABLE_WANDB) — see note at ENABLE_WANDB
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

# Stochastic demand (bolivia/demand_scenarios.csv, single line `s,<value>`):
# i.i.d. per-stage multiplicative factor ξ_t ∈ {1−s, 1, 1+s} (P = 1/3 each) on
# every bus's active demand, independent of the inflow noise — the same model
# the SDDP baselines register via sddp/sddp_demand_noise.jl. When the file is
# ABSENT every code path below is bit-identical to the historical trainer.
# Mechanics: scenarios become augmented stage-major vectors [w_t; ξ_t]
# (sample_scenario 3-arg method / augment_scenario), the DE is built with
# demand_spread (p_inflow sized T·(nHyd+1)), the policy encoder observes ξ_t
# (n_extra_uncertainty = 1), and prepare_solve! applies base_demand·ξ_t via
# set_demand! before every solve (training, rollout, and multi-GPU workers).
const DEMAND_SPREAD = load_demand_spread(joinpath(CASE_DIR, "demand_scenarios.csv"))
const DEMAND_NOISE  = DEMAND_SPREAD !== nothing
DEMAND_NOISE && @info "Stochastic demand ACTIVE" DEMAND_SPREAD

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
# Target-head activation. "sigmoid" is the historical default; it cannot
# exactly attain reachable-interval boundaries (where SDDP places ~24% of its
# realized states), so boundary-attaining alternatives are available:
#   DR_ACTIVATION = "sigmoid" | "hardsigmoid" | "stretched"
const ACTIVATION = let raw = lowercase(strip(get(ENV, "DR_ACTIVATION", "stretched")))
    raw in ("", "sigmoid") ? sigmoid :
    raw == "hardsigmoid"   ? hardsigmoidsafe :
    raw == "stretched"     ? stretchedsigmoid :
    throw(ArgumentError("DR_ACTIVATION must be sigmoid, hardsigmoid, or stretched; got $raw"))
end
const NUM_STAGES  = parse(Int, get(ENV, "DR_NUM_STAGES", "126"))
const NUM_ROLLOUT_STAGES = parse(Int, get(ENV, "DR_NUM_ROLLOUT_STAGES", "96"))
const NUM_EPOCHS  = parse(Int, get(ENV, "DR_NUM_EPOCHS", "80"))
const NUM_BATCHES = parse(Int, get(ENV, "DR_NUM_BATCHES", "100"))
const NUM_TRAIN_PER_BATCH = parse(Int, get(ENV, "DR_NUM_TRAIN_PER_BATCH", "1"))
const NUM_TRAIN_SCHEDULE  = parse_int_schedule(get(ENV, "DR_NUM_TRAIN_SCHEDULE", ""), "DR_NUM_TRAIN_SCHEDULE")
const CONTEXT_MODE = canonical_context_mode(get(ENV, "DR_CONTEXT", ""))
const CONTEXT_PERIOD = countlines(INFLOW_FILE)
const CONTEXT_HORIZON = NUM_STAGES
const _base_context = build_stage_context(CONTEXT_MODE, CONTEXT_HORIZON, CONTEXT_PERIOD)
const STAGE_CONTEXT = _base_context
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
const ENABLE_WANDB = parse(Bool, get(ENV, "DR_ENABLE_WANDB", "true"))
# Load Wandb (PythonCall/CondaPkg) ONLY when enabled. With W&B off this avoids the
# CondaPkg "Downloading artifact: pixi" hang on compute nodes with no/slow internet.
ENABLE_WANDB && @eval using Wandb
const LR          = parse(Float32, get(ENV, "DR_LR", "0.001"))
const LR_FINAL    = parse(Float32, get(ENV, "DR_LR_FINAL", string(LR)))
const LR_WARMUP   = parse(Int, get(ENV, "DR_LR_WARMUP", "0"))
const PRE_TRAINED = strip(get(ENV, "DR_PRETRAINED_MODEL", ""))
const HAS_PRETRAINED = !(isempty(PRE_TRAINED) || lowercase(PRE_TRAINED) == "nothing")
const REACTIVE_DEFICIT_RAW = "hard"
const REACTIVE_DEFICIT_COST = Inf
const GRAD_CLIP   = parse(Float32, get(ENV, "DR_GRAD_CLIP", "0"))

const TARGET_PEN_ARG = :auto
const HYDRO_TARGET_PENALTY_MULT = 8.0
# MAIN: 60 USD/MWh × 100 MVA = 6000 USD/(pu·stage).
const DEFICIT_COST   = 6000.0
const USE_GPU        = true
const load_scaler    = 0.6
const qd_scaler      = 0.6
# Parallel-sample training: solve the `num_train_per_batch` per-gradient DEs
# across worker threads, each with its own MadNLP solver bound to its own CUDA
# stream (see train_tsddr! in src/training.jl). Requires JULIA_NUM_THREADS >=
# DR_NUM_WORKERS. Default 1 = historical sequential path (byte-identical).
const NUM_WORKERS    = let n = parse(Int, get(ENV, "DR_NUM_WORKERS", "1"))
    n >= 1 || throw(ArgumentError("DR_NUM_WORKERS must be >= 1"))
    if n > Threads.nthreads()
        @warn "DR_NUM_WORKERS=$n exceeds JULIA_NUM_THREADS=$(Threads.nthreads()); capping"
        Threads.nthreads()
    else
        n
    end
end

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
const _ACT_TAG   = ACTIVATION === sigmoid ? "" :
                   ACTIVATION === stretchedsigmoid ? "-actstretch" : "-acthardsig"
# Demand-noise tag: runs with stochastic demand are a different experiment
# family (different SP and different policy input width) — mark them.
const _DN_TAG    = DEMAND_NOISE ? "-dnoise$(value_tag(DEMAND_SPREAD))" : ""
const RUN_NAME  = "$(CASE_NAME)-$(FORM_LABEL)-h$(NUM_STAGES)-r$(NUM_ROLLOUT_STAGES)-deteq-strict-gpu$(_CLIP_TAG)$(_ENC_TAG)$(_HEAD_TAG)$(_NT_TAG)$(_CTX_TAG)$(_EV_TAG)$(_SAVE_TAG)$(_WARM_TAG)$(_RQ_TAG)$(_ACT_TAG)$(_DN_TAG)-$(Dates.format(now(), "yyyymmdd-HHMMSS"))"
const MODEL_DIR = joinpath(CASE_DIR, FORM_LABEL, "models")
mkpath(MODEL_DIR)
const MODEL_PATH = joinpath(MODEL_DIR, RUN_NAME * ".jld2")
# Crash-safety: independent of SaveBest (which only writes on improvement), the
# LATEST policy is checkpointed every DR_SAVE_LATEST_EVERY gradient steps to a
# separate "<run>_latest.jld2" file (overwritten each time). A crash then loses
# at most that many steps, regardless of whether the run had improved. Set to 0
# to disable. Default 25 keeps every expensive multi-worker run recoverable.
const SAVE_LATEST_EVERY = parse(Int, get(ENV, "DR_SAVE_LATEST_EVERY", "25"))
const LATEST_PATH = joinpath(MODEL_DIR, RUN_NAME * "_latest.jld2")
const TOTAL_ITERS = NUM_EPOCHS * NUM_BATCHES

function lr_schedule(iter::Int, total_iters::Int)
    if iter <= LR_WARMUP
        return LR * (0.01f0 + 0.99f0 * Float32(iter) / Float32(max(LR_WARMUP, 1)))
    end
    ρ = clamp((iter - LR_WARMUP) / max(total_iters - LR_WARMUP, 1), 0.0, 1.0)
    return LR_FINAL + 0.5f0 * (LR - LR_FINAL) * (1f0 + cos(Float32(pi * ρ)))
end

# ── Advanced scheduler: warm-restart LR per nt-stage + event-based nt stepping + go-back ──
# DR_SCHED_MODE = ""(off, iteration-based default) | "warmrestart" | "event".
#   warmrestart: nt walks DR_NT_STAGES on fixed DR_STAGE_ITERS boundaries; the cosine LR
#                RESTARTS (warmup→peak→floor) at each nt change — replicates the historical
#                warmstart trail (each stage its own LR search) inside one job.
#   event:       nt advances to the next stage only when the smoothed loss PLATEAUS
#                (no improvement for DR_SCHED_PATIENCE iters, min-dwell DR_SCHED_MINSTAGE),
#                LR restarts on each advance, and RETREATS (×0.7, "go back") when the smoothed
#                loss rises >2% above its best — so it keeps descending, never flat/unlearning.
const SCHED_MODE     = lowercase(strip(get(ENV, "DR_SCHED_MODE", "")))
const NT_STAGES      = let s = strip(get(ENV, "DR_NT_STAGES", "")); isempty(s) ? Int[] : parse.(Int, split(s, ',')) end
const STAGE_ITERS    = parse(Int, get(ENV, "DR_STAGE_ITERS", "1200"))
const SCHED_PATIENCE = parse(Int, get(ENV, "DR_SCHED_PATIENCE", "150"))
const SCHED_MINSTAGE = parse(Int, get(ENV, "DR_SCHED_MINSTAGE", "150"))

mutable struct SchedState
    ema::Float64; best::Float64; plateau::Int
    stage::Int; stage_start::Int; lr_retreat::Float64
end
const SCHED = SchedState(NaN, Inf, 0, 1, 1, 1.0)

function sched_observe!(loss::Float64)
    isfinite(loss) || return
    SCHED.ema = isnan(SCHED.ema) ? loss : 0.9 * SCHED.ema + 0.1 * loss
    if SCHED.ema < SCHED.best - 1e-6 * abs(SCHED.best)
        SCHED.best = SCHED.ema; SCHED.plateau = 0
    else
        SCHED.plateau += 1
    end
    if isfinite(SCHED.best) && SCHED.ema > SCHED.best * 1.02     # diverging → retreat LR ("go back")
        SCHED.lr_retreat = max(SCHED.lr_retreat * 0.7, 0.05)
    end
end

function sched_nt!(iter::Int)
    isempty(NT_STAGES) && return NUM_TRAIN_PER_BATCH
    if SCHED_MODE == "event"
        if SCHED.stage < length(NT_STAGES) && SCHED.plateau >= SCHED_PATIENCE &&
           (iter - SCHED.stage_start) >= SCHED_MINSTAGE
            SCHED.stage += 1; SCHED.stage_start = iter; SCHED.plateau = 0; SCHED.lr_retreat = 1.0
            @info "event-sched: nt → $(NT_STAGES[SCHED.stage]) (LR restart) at iter $iter"
        end
    else
        ns = min(length(NT_STAGES), 1 + div(iter - 1, max(STAGE_ITERS, 1)))
        if ns != SCHED.stage
            SCHED.stage = ns; SCHED.stage_start = iter; SCHED.lr_retreat = 1.0
            @info "warmrestart: nt → $(NT_STAGES[SCHED.stage]) (LR restart) at iter $iter"
        end
    end
    return NT_STAGES[SCHED.stage]
end

function sched_lr(iter::Int)
    len = SCHED_MODE == "event" ? max(div(TOTAL_ITERS, max(length(NT_STAGES), 1)), 1) : max(STAGE_ITERS, 1)
    k = iter - SCHED.stage_start
    lr = if k <= LR_WARMUP
        LR * (0.01f0 + 0.99f0 * Float32(k) / Float32(max(LR_WARMUP, 1)))
    else
        ρ = clamp((k - LR_WARMUP) / max(len - LR_WARMUP, 1), 0.0, 1.0)
        LR_FINAL + 0.5f0 * (LR - LR_FINAL) * (1f0 + cos(Float32(pi * ρ)))
    end
    return Float32(lr * SCHED.lr_retreat)
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
# Per-stage uncertainty width fed to the policy/DE/rollout machinery:
# nHyd inflows, plus the demand factor ξ_t when stochastic demand is active.
N_UNC = nHyd + (DEMAND_NOISE ? 1 : 0)
@info "  nHyd=$(nHyd)  nScenarios=$(hydro_data.nScenarios)  n_uncertainty=$(N_UNC)"

demand_mat = nothing

# Reactive demand at qd_scaler × nominal, independent of load_scaler: the
# builder multiplies any given matrix by load_scaler, so divide it back out.
reactive_mat = qd_scaler == load_scaler ? nothing :
    repeat(reshape((qd_scaler / load_scaler) .* power_data.default_bus_reactive_demand,
                   1, :), T, 1)

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
        reactive_demand_matrix = reactive_mat,
        load_scaler    = load_scaler,
        strict_targets = true,
        reactive_deficit_cost = REACTIVE_DEFICIT_COST,
        # Stochastic demand: sizes p_inflow for [w_t; ξ_t] blocks and stores
        # the base demand for prepare_solve!'s ξ_t multiplication (nothing =
        # deterministic, bit-identical legacy model).
        demand_spread  = DEMAND_SPREAD,
    )
end

# Multi-GPU device assignment for the worker pool. With USE_GPU, distribute the
# NUM_WORKERS solver DEs round-robin across the visible CUDA devices (SLURM sets
# CUDA_VISIBLE_DEVICES, so ndevices() = the GPUs this job was granted). Worker wi
# runs on device WORKER_DEVICES[wi], and its DE is BUILT on that device below so
# the DE arrays and the worker's solver live together. Single GPU or CPU → all
# zeros / nothing (unchanged behavior). Set DR_NUM_WORKERS = workers_per_gpu ×
# n_gpus (e.g. 9 on a 3-GPU job packs 3 workers/GPU).
const N_GPU = USE_GPU ? CUDA.ndevices() : 0
const WORKER_DEVICES = (USE_GPU && N_GPU > 1) ?
    [(i - 1) % N_GPU for i in 1:NUM_WORKERS] : nothing
WORKER_DEVICES === nothing || @info "Multi-GPU worker→device map" N_GPU WORKER_DEVICES

# Build a DE on a specific CUDA device (arrays allocate on the active device).
# Diagnostics to stderr (flushed) so a hang in the multi-GPU pool build is
# localizable in the SLURM log.
function _build_de_on(dev, i)
    if dev === nothing
        return _build_de()
    end
    println(stderr, "[pool $i] building DE on CUDA device $dev ..."); flush(stderr)
    CUDA.device!(dev)
    de = _build_de()
    println(stderr, "[pool $i] DE ready on device $dev (current=$(CUDA.device()))"); flush(stderr)
    return de
end

@info "Building strict $(T)-stage ExaModels DE (formulation=$FORMULATION)..."
prob = _build_de()   # metadata DE on the default device (device 0)

# Multi-GPU: workers build their OWN DE in-task (a main-task-built DE deadlocks
# on the first cross-task solve). `worker_de_builder(wi)` runs inside worker
# `wi`'s task AFTER it has bound its device, so the DE lands on the right GPU.
# Single-GPU / CPU: fall back to the pre-built pool (all on device 0).
worker_de_builder = nothing
problem_pool = nothing
if WORKER_DEVICES !== nothing
    worker_de_builder = function (wi)
        de = _build_de()   # current device is set by the worker's CUDA.device!
        (de, de.p_x0, de.p_target, de.p_inflow)
    end
    @info "  Multi-GPU: each of $NUM_WORKERS workers builds its DE in-task" n_gpu=N_GPU
else
    problem_pool = [(prob, prob.p_x0, prob.p_target, prob.p_inflow)]
    for _ in 2:NUM_WORKERS
        p = _build_de()
        push!(problem_pool, (p, p.p_x0, p.p_target, p.p_inflow))
    end
    @info "  Pool ready: $(NUM_WORKERS) DE instances on the default device"
end

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
                                     n_context        = N_CONTEXT,
                                     # Demand noise: the encoder additionally
                                     # observes ξ_t (stage-t revealed demand),
                                     # matching SDDP whose stage subproblem
                                     # sees the realized demand atom.
                                     n_extra_uncertainty = DEMAND_NOISE ? 1 : 0)
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
feeds `[u_t; previous_target]` to the policy, and stores each reachable target as
the next previous state. By induction, every target in the returned trajectory is
reachable from the prior target under the sampled inflow path.

# Arguments
- `policy`: reachable hydro policy with input `[uncertainty; previous_state]`.
- `x0`: initial reservoir state.
- `w_flat`: stage-major flat uncertainty vector of length `T * n_uncertainty`
  (`n_uncertainty = nHyd` historically, `nHyd + 1` with stochastic demand —
  the per-stage stride is derived from `length(w_flat) ÷ T`, so both layouts
  work; the policy slices the physical inflow internally).
- `T::Int`: number of stages.
- `nHyd::Int`: number of hydro reservoir state components (kept for call-site
  compatibility; the stride no longer depends on it).

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
    # Per-stage uncertainty stride derived from the vector itself (nHyd, or
    # nHyd+1 when the demand factor ξ_t is appended to each stage block).
    nu = length(w_flat) ÷ T
    targets = Vector{Vector{Float32}}(undef, T)
    for t in 1:T
        # Full stage-t uncertainty block [w_t] or [w_t; ξ_t].
        wt = Float32.(view(w_flat, ((t - 1) * nu + 1):(t * nu)))
        target = policy(vcat(wt, prev))
        targets[t] = Float32.(target)
        prev = targets[t]
    end
    return Float64.(vcat(targets...))
end

# ── Smoke test ────────────────────────────────────────────────────────────────

# Mean-inflow smoke scenario; with demand noise, append the neutral factor
# ξ_t = 1 to every stage (base demand) so the augmented layout is exercised.
w_mean = DEMAND_NOISE ? augment_scenario(mean_inflow(hydro_data, T), ones(T)) :
                        mean_inflow(hydro_data, T)
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

lg = ENABLE_WANDB ? WandbLogger(
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
        # Demand-noise provenance: "none" = deterministic demand.
        "demand_spread"   => DEMAND_NOISE ? DEMAND_SPREAD : "none",
        "strict_targets"  => true,
        "policy_type"     => N_CONTEXT == 0 ? "HydroReachablePolicy" : "ContextualPolicy{HydroReachablePolicy}",
        "num_workers"     => NUM_WORKERS,
    ),
) : nothing
ENABLE_WANDB || @info "W&B disabled (DR_ENABLE_WANDB=false)"

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
        reactive_demand_matrix = reactive_mat === nothing ? nothing : reactive_mat[1:1, :],
        load_scaler    = load_scaler,
        strict_targets = true,
        reactive_deficit_cost = REACTIVE_DEFICIT_COST,
        # Stochastic demand: the 1-stage problem also carries a base_demand
        # row; set_hydro_rollout_stage! refreshes it per stage and
        # prepare_solve! applies the ξ_t carried in the stage's wt block.
        demand_spread  = DEMAND_SPREAD,
    )
end
rollout_prob = _build_rollout_de()
n_rollout_pool = max(NUM_WORKERS, NUM_EVAL_SCENARIOS)
rollout_pool = ROLLOUT_PARALLEL ? [_build_rollout_de() for _ in 1:n_rollout_pool] : []
@info "Rollout evaluation: $(ROLLOUT_PARALLEL ? "parallel ($(n_rollout_pool) stage-problem copies)" : "sequential")"

function set_hydro_rollout_stage!(stage_prob, state_in, wt, target, stage)
    ExaModels.set_parameter!(stage_prob.core, stage_prob.p_x0, state_in)
    # wt is the full stage uncertainty block ([w_t] or [w_t; ξ_t]); the stage
    # problem's p_inflow is sized to match (n_uncertainty entries).
    ExaModels.set_parameter!(stage_prob.core, stage_prob.p_inflow, wt)
    if demand_mat !== nothing
        if DEMAND_NOISE
            # Stochastic demand: refresh the 1-stage problem's BASE demand row
            # in place; prepare_solve! below multiplies it by the ξ_t carried
            # in wt's last entry and writes the product via set_demand!.
            stage_prob.base_demand[1, :] .= load_scaler .* @view demand_mat[stage, :]
        else
            # Deterministic demand: historical direct write (bit-identical).
            set_demand!(stage_prob, load_scaler .* demand_mat[stage:stage, :])
        end
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

# Held-out evaluation scenarios. Two modes:
#
# 1. DR_EVAL_PROTOCOL_IDS set (comma-separated column ids of the seeded paired
#    protocol): the eval set is those exact columns of the 126×500 protocol
#    matrix (StableRNG seed 20260706 — identical generation to the paired
#    evaluation scripts). With the representative subset found by searching
#    300k candidate subsets against 7 evaluated policies —
#    ids 2,39,81,119,130,156,200,206,378,493 (subset seed 77142) — the
#    10-scenario training-time evaluation tracks the full 500-scenario paired
#    mean within ~75 cost units and preserves paired differences vs SDDP
#    within ~35, so SaveBest selects on (a faithful proxy of) the deployment
#    metric.
# 2. Unset (historical): random draws from the inflow process, seed 8789.
#    NOTE: arbitrary small draws carry offsets of hundreds-to-thousands of
#    cost units vs the paired protocol; never compare their values across
#    runs with different eval sets.
const EVAL_PROTOCOL_IDS = let raw = strip(get(ENV, "DR_EVAL_PROTOCOL_IDS", ""))
    isempty(raw) ? Int[] : [parse(Int, strip(x)) for x in split(raw, ",")]
end

"""
    protocol_eval_scenario(hydro_data, T, protocol_indices, s) -> Vector{Float64}

Flat `T × nHyd` inflow vector for paired-protocol scenario column `s`:
stage `t` realizes joint inflow scenario `protocol_indices[t, s]`, with the
cyclic raw-row mapping shared by every paired evaluation script.
"""
function protocol_eval_scenario(hydro_data::HydroData, T::Int, protocol_indices, s::Int)
    nHyd = hydro_data.nHyd
    w = Vector{Float64}(undef, T * nHyd)
    for t in 1:T
        t_row = mod1(t, hydro_data.nStagesSample)
        j = protocol_indices[t, s]
        for r in 1:nHyd
            w[(t-1)*nHyd + r] = hydro_data.scenario_inflows[r][t_row, j]
        end
    end
    return w
end

# Demand-noise augmentation of a protocol column: pair the column's inflows
# with its SEEDED demand path (StableRNG(DEMAND_NOISE_SEED + column) — the
# identical path eval_paired_exa_strict.jl draws for that column). With noise
# off this is the identity, so the historical eval set is unchanged.
_augment_protocol(w, col, T) = DEMAND_NOISE ?
    augment_scenario(w, protocol_demand_factors(DEMAND_SPREAD, T, col)) : w

eval_scenarios = if isempty(EVAL_PROTOCOL_IDS)
    Random.seed!(8789)
    # Random draws: the 3-arg sampler additionally draws i.i.d. ξ_t from the
    # SAME seeded global stream, so the eval set stays reproducible.
    [DEMAND_NOISE ? sample_scenario(hydro_data, T_ROLLOUT, DEMAND_SPREAD) :
                    sample_scenario(hydro_data, T_ROLLOUT)
     for _ in 1:NUM_EVAL_SCENARIOS]
else
    # Same fixed-shape generation as the paired protocol (126 rows × 500
    # columns; see load_hydropowermodels.jl in the MAIN repo).
    protocol_indices = rand(StableRNG(20260706), 1:hydro_data.nScenarios, 126, 500)
    @assert T_ROLLOUT <= 126 && all(1 .<= EVAL_PROTOCOL_IDS .<= 500)
    @assert length(EVAL_PROTOCOL_IDS) == NUM_EVAL_SCENARIOS "DR_NUM_EVAL_SCENARIOS must match the id count"
    @info "Eval set = paired-protocol columns $(EVAL_PROTOCOL_IDS)"
    [_augment_protocol(protocol_eval_scenario(hydro_data, T_ROLLOUT, protocol_indices, s), s, T_ROLLOUT)
     for s in EVAL_PROTOCOL_IDS]
end
# ── Training sampler ──────────────────────────────────────────────────────────
# Default: fresh random inflow draws (genuine SAA). When DR_TRAIN_PROTOCOL_ALL
# is set, the sampler instead cycles deterministically through the exact 500
# paired-protocol columns (StableRNG(20260706), full T stages). With
# num_train_per_batch=500 and num_batches a multiple of 1, every gradient step
# is a full-batch step over ALL 500 evaluation scenarios — the "cheating" upper
# bound: can gradient descent directly on the test set beat SDDP? If it can't,
# the policy class is the wall.
const TRAIN_PROTOCOL_ALL = lowercase(strip(get(ENV, "DR_TRAIN_PROTOCOL_ALL", ""))) in ("1", "true", "yes")
train_sampler = if TRAIN_PROTOCOL_ALL
    train_protocol_indices = rand(StableRNG(20260706), 1:hydro_data.nScenarios, 126, 500)
    @assert T <= 126
    # With demand noise, each protocol column is paired with its seeded demand
    # path (prefix-consistent with the T_ROLLOUT-length eval draws — see
    # sample_demand_factors' sequential-draw prefix property).
    protocol_cols = [_augment_protocol(protocol_eval_scenario(hydro_data, T, train_protocol_indices, s), s, T)
                     for s in 1:500]
    @info "TRAINING on the exact 500 paired-protocol scenarios (cheating upper-bound test)" NUM_TRAIN_PER_BATCH
    cyc = Ref(0)
    () -> (cyc[] = cyc[] % 500 + 1; protocol_cols[cyc[]])
elseif DEMAND_NOISE
    # Genuine SAA over the PRODUCT distribution: fresh inflow draw + fresh
    # i.i.d. per-stage demand factors, returned as augmented [w_t; ξ_t] blocks.
    () -> sample_scenario(hydro_data, T, DEMAND_SPREAD)
else
    () -> sample_scenario(hydro_data, T)
end

rollout_evaluation = RolloutEvaluation(
    rollout_prob,
    x0_init,
    eval_scenarios;
    horizon = T_ROLLOUT,
    # Per-stage uncertainty width: nHyd, or nHyd+1 with demand noise (the
    # stage callback then receives the full [w_t; ξ_t] block as wt).
    n_uncertainty = N_UNC,
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

# ── Two-stage SaveBest verification (guards against overfitting the fixed eval
# set). The fixed rep-10 eval is cheap but a policy can overfit to those exact
# 10 scenarios, so SaveBest-on-10 may select an overfit iterate. When
# DR_VERIFY_SCENARIOS > 0, a checkpoint that beats the best on the fixed set is
# only ACCEPTED if it also beats the incumbent best on N FRESHLY-sampled random
# scenarios — re-drawn every trigger, so a policy cannot overfit to them. The
# incumbent is re-evaluated on the SAME fresh draw for a paired comparison.
const VERIFY_SCENARIOS = parse(Int, get(ENV, "DR_VERIFY_SCENARIOS", "0"))
verify_evaluation = if VERIFY_SCENARIOS > 0 && SAVE_METRIC == "rollout"
    @info "Two-stage SaveBest: verify accepts on $VERIFY_SCENARIOS fresh random scenarios"
    RolloutEvaluation(
        _build_rollout_de(), x0_init,
        # Fresh verification draws come from the same (augmented, when demand
        # noise is on) sampler family as training.
        [DEMAND_NOISE ? sample_scenario(hydro_data, T_ROLLOUT, DEMAND_SPREAD) :
                        sample_scenario(hydro_data, T_ROLLOUT)
         for _ in 1:VERIFY_SCENARIOS];
        horizon = T_ROLLOUT, n_uncertainty = N_UNC,
        set_stage_parameters! = set_hydro_rollout_stage!,
        realized_state = hydro_realized_state,
        objective_no_target_penalty = hydro_objective_no_target_penalty,
        madnlp_kwargs = SOLVER_KWARGS, warmstart = false, stride = 1,
        policy_state = :realized, stage_problem_pool = [], retry_on_failure = true,
        active_scenarios = VERIFY_SCENARIOS,
        state_bounds = (_min_vols_dev, _max_vols_dev),
    )
else
    nothing
end
best_verify_snapshot = Ref{Any}(nothing)   # deepcopy of the accepted-best policy

# Accept `model` as the new best? With verification off, the rep-10 gate that
# already fired is sufficient (return true). With it on, draw fresh scenarios
# and require `model` to beat the incumbent snapshot on them (all must solve).
function verified_improvement(model)
    verify_evaluation === nothing && return true
    # Fresh draws per trigger; with demand noise the 3-arg sampler pairs each
    # fresh inflow path with fresh i.i.d. demand factors ([w_t; ξ_t] blocks).
    verify_evaluation.scenarios =
        [DEMAND_NOISE ? sample_scenario(hydro_data, T_ROLLOUT, DEMAND_SPREAD) :
                        sample_scenario(hydro_data, T_ROLLOUT)
         for _ in 1:VERIFY_SCENARIOS]
    verify_evaluation(1, model)
    verify_evaluation.last_n_ok == VERIFY_SCENARIOS || return false
    cand = verify_evaluation.last_objective_no_target_penalty
    inc = if best_verify_snapshot[] === nothing
        Inf
    else
        verify_evaluation(1, best_verify_snapshot[])   # SAME fresh scenarios
        verify_evaluation.last_n_ok == VERIFY_SCENARIOS ?
            verify_evaluation.last_objective_no_target_penalty : Inf
    end
    accept = cand < inc
    accept && (best_verify_snapshot[] = deepcopy(model))
    @info "  verify on $VERIFY_SCENARIOS fresh: cand=$(round(cand; digits=1)) inc=$(inc == Inf ? "none" : round(inc; digits=1)) => $(accept ? "ACCEPT" : "reject (overfit to fixed set)")"
    return accept
end

current_num_train = Ref(NUM_TRAIN_PER_BATCH)
current_eval_scenarios = Ref(schedule_value(EVAL_SCHEDULE, 1, NUM_EVAL_SCENARIOS))
rollout_evaluation.active_scenarios = current_eval_scenarios[]

# The initial rollout eval only sets the checkpoint baseline, and it runs
# SEQUENTIALLY on device 0 (a rollout is 96 dependent stages, one scenario at a
# time) — so it blocks the multi-GPU training start with pure single-GPU work.
# Skip it by default: best_obj stays Inf and the first periodic eval sets the
# baseline, letting the 12 workers engage all GPUs immediately.
const SKIP_INITIAL_EVAL = parse(Bool, get(ENV, "DR_SKIP_INITIAL_EVAL", "true"))
if SAVE_METRIC == "rollout" && !SKIP_INITIAL_EVAL
    rollout_evaluation(EVAL_EVERY, policy)
    best_obj = rollout_evaluation.last_objective_no_target_penalty
    @info "Initial rollout evaluation (checkpoint baseline)" best_obj rollout_evaluation.last_violation_share rollout_evaluation.last_n_ok
elseif SAVE_METRIC == "rollout"
    @info "Skipping initial rollout eval (DR_SKIP_INITIAL_EVAL=true) — training starts immediately on all GPUs"
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
    train_sampler;
    num_batches          = TOTAL_ITERS,
    num_train_per_batch  = NUM_TRAIN_PER_BATCH,
    optimizer            = GRAD_CLIP > 0 ?
                           Flux.Optimisers.OptimiserChain(
                               Flux.Optimisers.ClipGrad(GRAD_CLIP),
                               Flux.Adam(LR),
                           ) : Flux.Adam(LR),
    madnlp_kwargs        = SOLVER_KWARGS,
    warmstart            = true,
    problem_pool           = problem_pool,
    worker_devices         = WORKER_DEVICES,
    worker_problem_builder = worker_de_builder,
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
    adjust_hyperparameters = SCHED_MODE != "" ?
        ((iter, opt_state, n) -> begin
            Flux.Optimisers.adjust!(opt_state, sched_lr(iter))
            n_next = sched_nt!(iter)
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
        (LR_WARMUP == 0 && LR_FINAL == LR) ?
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
        SCHED_MODE == "" || sched_observe!(Float64(loss))
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
            metrics["metrics/rollout_target_violation_share"] =
                rollout_evaluation.last_violation_share
            metrics["metrics/rollout_n_ok"] =
                rollout_evaluation.last_n_ok
            if SAVE_METRIC == "rollout"
                rollout_score = rollout_evaluation.last_objective_no_target_penalty
                # Honest selection: the rollout mean is over the scenarios that
                # SOLVED (total / n_ok), so a policy that fails one expensive
                # scenario gets a fake bonus of hundreds of cost units. Only
                # trust evals where every active scenario succeeded.
                if rollout_evaluation.last_n_ok == current_eval_scenarios[] &&
                   isfinite(rollout_score) && rollout_score < best_obj &&
                   verified_improvement(m)   # second-stage fresh-scenario gate
                    global best_obj = rollout_score
                    jldsave(MODEL_PATH; model_state = checkpoint_policy_state(m))
                    @info "  -> New best rollout: $(round(rollout_score; digits=4)) -- saved $MODEL_PATH"
                end
            end
        end

        # Crash-safety: overwrite the "_latest" checkpoint every N steps,
        # independent of improvement, so a failure loses at most N steps.
        if SAVE_LATEST_EVERY > 0 && iter % SAVE_LATEST_EVERY == 0
            jldsave(LATEST_PATH; model_state = checkpoint_policy_state(m))
            @info "  latest checkpoint @ iter $iter -> $LATEST_PATH"
        end

        batch_in_epoch = (iter - 1) % NUM_BATCHES + 1
        if batch_in_epoch == NUM_BATCHES
            epoch     = (iter - 1) ÷ NUM_BATCHES + 1
            mean_loss = isempty(epoch_losses) ? NaN : mean(epoch_losses)
            n_ok      = length(epoch_losses)
            empty!(epoch_losses)
            lg === nothing || Wandb.log(lg, Dict("metrics/epoch_objective" => mean_loss, "epoch" => epoch))
            @info "Epoch $epoch/$NUM_EPOCHS  mean=$(round(mean_loss; digits=2))  ok=$n_ok/$NUM_BATCHES"
            if SAVE_METRIC == "training" && isfinite(mean_loss) && mean_loss < best_obj
                global best_obj = mean_loss
                jldsave(MODEL_PATH; model_state = checkpoint_policy_state(m))
                @info "  → New best: $(round(mean_loss; digits=4)) — saved $MODEL_PATH"
            end
        end
        lg === nothing || Wandb.log(lg, metrics)
        return false
    end,
)

lg === nothing || close(lg)
@info "Done. Best model saved to: $(MODEL_PATH)"
