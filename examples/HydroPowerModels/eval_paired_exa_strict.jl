# eval_paired_exa_strict.jl
#
# Cross-package equivalence check against DecisionRules.jl (the JuMP/Ipopt MAIN
# repo). Evaluates the SAME stage-wise-strict TS-DDR checkpoint on the SAME 100
# paired scenarios that MAIN's eval_paired_tsddr.jl evaluated, using
#
#   (a) the stage-wise strict ExaModels rollout (1-stage strict problems,
#       closed-loop realized-state feedback), and
#   (b) the strict regular full-horizon deterministic equivalent (DE),
#
# and records per-scenario objectives and operative solution values for
# comparison against MAIN's ground truth (paired_strict_rollout.jld2).
#
# DISCREPANCIES ARE THE DELIVERABLE. Nothing here is tuned to force agreement;
# every structural difference is measured and saved. Known structural
# differences handled/documented here (see also compare_paired_evals.jl):
#
#   1. DEFICIT COST. MAIN's ACPPowerModel.mof.json objective is
#          Σ_g c1_g·pg_g + 6000·Σ_b deficit_b        (linear; c2 = 0 for all g)
#      where 6000 = cost_deficit (60 $/MWh) × baseMVA (100): HydroPowerModels
#      scales the deficit cost by baseMVA when exporting the subproblem. The
#      EXA builder uses `deficit_cost` per pu directly, so this script passes
#      deficit_cost = power_data.cost_deficit * power_data.baseMVA = 6000.0.
#      NOTE: train_hydro_exa_strict.jl trains with DEFICIT_COST = 1e5 instead —
#      inert iff deficit never activates (max deficit is recorded per scenario).
#   2. DEMAND. The mof.json bakes in 0.6 × PowerModels.json loads (see MAIN's
#      export_subproblem_mof.jl, which scales pd AND qd by 0.6). The EXA builder
#      reproduces this with load_scaler = 0.6 and demand_matrix = nothing.
#   3. REACTIVE SLACK. By default the EXA AC builder adds a FREE, zero-cost
#      `deficit_q` variable to every reactive KCL; MAIN's mof.json has no
#      reactive slack (hard reactive balance). The builder kwarg
#      `reactive_deficit_cost` now controls this; this script reads the
#      DR_REACTIVE_DEFICIT env var (default "hard" → Inf, i.e. NO reactive
#      slack — MAIN-faithful; "free" → nothing reproduces the historical
#      relaxation; a number → linear |deficit_q| penalty at that cost) and
#      passes it to BOTH the stage problem and the DE leg. max |deficit_q| is
#      still recorded per scenario (exact zeros in hard mode).
#   4. THERMAL LIMITS. mof.json enforces quadratic branch limits
#      p² + q² ≤ rate_a² (62 ScalarQuadraticFunction ≤ constraints); the EXA
#      builder only box-bounds each of p_fr, q_fr, p_to, q_to in
#      [−rate_a, rate_a] — a superset of the disk (relaxation in the corners).
#   5. MIN-VIOLATION SLACKS. mof.json has min_outflow_violation /
#      min_volume_violation slack variables with ZERO objective cost (they only
#      relax the ≥ min bounds); the EXA builder enforces outflow ≥ min_turn
#      hard. For Bolivia min_turn ≡ 0, so both are inert.
#   6. POLICY RECURRENCE (measured by the parity gate below). RESOLVED: the
#      EXA HydroReachablePolicy now threads the LSTM state across stages
#      explicitly (`DecisionRulesExa._step_encoder`, mirroring MAIN's
#      `DecisionRules._step_encoder`), and `Flux.reset!(policy)` is a REAL
#      reset back to `Flux.initialstates`. The as-is open-loop gate below is
#      therefore expected to PASS with max|Δ| = 0.0 against MAIN. The
#      independent manually-threaded diagnostic (gate 4) is retained as a
#      cross-check: if as-is and threaded ever disagree, the policy forward
#      pass has regressed from MAIN's semantics.
#
# Requires: MAIN's paired_policy_reference.jld2 (produced by
# dump_paired_policy_reference.jl in the MAIN repo) — run that job first.
#
# Environment variables:
#   DR_DE_SCENARIOS     = "10"    (number of scenarios for the full-horizon DE leg)
#   DR_MAX_ITER         = "9000"
#   DR_REACTIVE_DEFICIT = "hard"  ("hard" → Inf = no reactive slack (MAIN-faithful);
#                                  "free" → nothing = historical free slack;
#                                  a number → linear |deficit_q| cost)
#   DR_CHECKPOINT       = <abs path .jld2>  (checkpoint to evaluate; default: the
#                                  MAIN reference checkpoint hardcoded below)
#   DR_CHECKPOINT_KIND  = "main"  ("main" → apply ALL MAIN-reference parity gates;
#                                  "exa" → EXA-trained checkpoint: SKIP the
#                                  probe-parity and open-loop-trajectory gates
#                                  (only meaningful for the exact reference
#                                  weights) but KEEP the policy-independent
#                                  inflow-indexing gate; scenario inflow values
#                                  still come from the reference JLD2)
#   DR_ENCODER_LAYERS   = "128,128" (LSTM encoder widths; must match checkpoint;
#                                  DR_LAYERS is the legacy alias, as in
#                                  train_hydro_exa_strict.jl)
#   DR_HEAD_LAYERS      = ""      (state-conditioned head hidden widths; must
#                                  match checkpoint; "" → linear head)
#   DR_CONTEXT          = ""      (""/"none", "phase", or "phase+progress";
#                                  defaults to the reference file metadata
#                                  when present)
#   DR_CONTEXT_HORIZON  = "126"   (denominator/horizon used for progress
#                                  context; defaults to reference metadata)
#   DR_REFERENCE_FILE   = <path>  (paired_policy_reference*.jld2 from MAIN)
#   DR_OUTPUT_TAG       = ""      ("" → save to results/paired_exa_strict.jld2;
#                                  "<tag>" → results/paired_exa_strict_<tag>.jld2
#                                  with checkpoint path + all knobs recorded)
#
# Usage:
#   julia --project -t auto eval_paired_exa_strict.jl

using DecisionRulesExa
using ExaModels
using MadNLP
using Flux
using Statistics, Random
using JLD2

const SCRIPT_DIR = dirname(@__FILE__)
include(joinpath(SCRIPT_DIR, "hydro_training_utils.jl"))   # parse_layers
include(joinpath(SCRIPT_DIR, "hydro_power_data.jl"))
include(joinpath(SCRIPT_DIR, "hydro_power_exa.jl"))
include(joinpath(SCRIPT_DIR, "hydro_reachable_policy.jl"))

# ── Configuration ─────────────────────────────────────────────────────────────

const CASE_NAME   = "bolivia"
const FORMULATION = :ac_polar
const FORM_LABEL  = "ACPPowerModel"

const CASE_DIR    = joinpath(SCRIPT_DIR, CASE_NAME)
const PM_FILE     = joinpath(CASE_DIR, "PowerModels.json")
const HYDRO_FILE  = joinpath(CASE_DIR, "hydro.json")
const INFLOW_FILE = joinpath(CASE_DIR, "inflows.csv")

# Absolute paths into the MAIN (DecisionRules.jl) repository.
const MAIN_HPM_DIR = "/storage/scratch1/9/arosemberg3/DecisionRules.jl/examples/HydroPowerModels"
const DEFAULT_REFERENCE_FILE = joinpath(
    MAIN_HPM_DIR, CASE_NAME, FORM_LABEL, "results", "paired_policy_reference.jld2"
)
const REFERENCE_FILE = get(ENV, "DR_REFERENCE_FILE", DEFAULT_REFERENCE_FILE)
# Default checkpoint: the MAIN reference checkpoint against which the parity
# gates below were designed. DR_CHECKPOINT overrides it with any other
# checkpoint (MAIN- or EXA-trained).
const DEFAULT_MODEL_PATH = joinpath(
    MAIN_HPM_DIR, CASE_NAME, FORM_LABEL, "models",
    "bolivia-ACPPowerModel-h126-r96-subproblems-strict-2026-07-01T09:41:53.026.jld2",
)
const MODEL_PATH = get(ENV, "DR_CHECKPOINT", DEFAULT_MODEL_PATH)
# Checkpoint kind: "main" (DecisionRules.jl-trained; MAIN-reference parity
# gates apply) or "exa" (train_hydro_exa_strict.jl-trained; the probe-parity
# and open-loop-trajectory gates are SKIPPED because the reference probe
# outputs / trajectories were produced by the specific reference checkpoint —
# comparing independently trained weights against them is meaningless. The
# inflow-indexing gate is policy-independent and is KEPT, and scenario inflow
# values are still sourced from the reference JLD2 as authoritative data).
const CHECKPOINT_KIND = lowercase(strip(get(ENV, "DR_CHECKPOINT_KIND", "main")))
CHECKPOINT_KIND in ("main", "exa") ||
    error("DR_CHECKPOINT_KIND must be \"main\" or \"exa\", got \"$CHECKPOINT_KIND\"")
const MAIN_PARITY_GATES = CHECKPOINT_KIND == "main"

# Architecture knobs — parsed exactly as train_hydro_exa_strict.jl parses them
# (including the DR_LAYERS legacy alias); they MUST match the checkpoint.
const ENCODER_LAYERS = parse_layers(get(ENV, "DR_ENCODER_LAYERS", get(ENV, "DR_LAYERS", "128,128")))
const HEAD_LAYERS    = parse_layers(get(ENV, "DR_HEAD_LAYERS", ""))

# Output tag: "" keeps the historical filename results/paired_exa_strict.jld2;
# a non-empty tag saves to results/paired_exa_strict_<tag>.jld2 so evaluating a
# new checkpoint never clobbers the reference results.
const OUTPUT_TAG = String(strip(get(ENV, "DR_OUTPUT_TAG", "")))
const OUT_SUFFIX = isempty(OUTPUT_TAG) ? "" : "_$(OUTPUT_TAG)"
# Record the new provenance knobs in the JLD2 whenever any of them was
# explicitly set; with all of them unset the saved file keeps exactly the
# historical key set (byte-identical default behavior).
const RECORD_KNOBS = any(haskey.(Ref(ENV),
    ("DR_CHECKPOINT", "DR_CHECKPOINT_KIND", "DR_ENCODER_LAYERS", "DR_LAYERS",
     "DR_HEAD_LAYERS", "DR_CONTEXT", "DR_CONTEXT_HORIZON", "DR_REFERENCE_FILE",
     "DR_OUTPUT_TAG", "DR_SNAP_EPS", "DR_ACTIVATION")))
# mof.json demand = 0.6 × PowerModels.json pd/qd (export_subproblem_mof.jl).
const LOAD_SCALER = 0.6
const NUM_DE_SCENARIOS = parse(Int, get(ENV, "DR_DE_SCENARIOS", "10"))
const MAX_ITER = parse(Int, get(ENV, "DR_MAX_ITER", "9000"))
# Reactive-slack control (header item 3): "hard" → Inf (no deficit_q,
# MAIN-faithful), "free" → nothing (historical free slack), number → linear
# |deficit_q| penalty at that cost. Passed to both the stage problem and the
# DE leg builders.
const REACTIVE_DEFICIT_RAW = lowercase(strip(get(ENV, "DR_REACTIVE_DEFICIT", "hard")))
const REACTIVE_DEFICIT_COST = REACTIVE_DEFICIT_RAW == "hard" ? Inf :
                              REACTIVE_DEFICIT_RAW == "free" ? nothing :
                              parse(Float64, REACTIVE_DEFICIT_RAW)

# Stochastic demand (bolivia/demand_scenarios.csv, single line `s,<value>`):
# i.i.d. per-stage multiplicative demand factor ξ_t ∈ {1−s, 1, 1+s} (P = 1/3),
# independent of the inflow noise — the same model the SDDP baselines register
# via sddp/sddp_demand_noise.jl and train_hydro_exa_strict.jl trains under.
# For the PAIRED protocol, scenario column c uses the SEEDED demand path
# StableRNG(DEMAND_NOISE_SEED + c) (protocol_demand_factors), identical to the
# path the trainer's protocol eval and SDDP.Historical paired evaluator draw
# for that column. Thus the comparison is paired over the complete joint
# inflow-and-demand path, not merely distributional over demand.
# When the file is absent every path below is bit-identical to the historical
# evaluation.
const DEMAND_SPREAD = load_demand_spread(joinpath(CASE_DIR, "demand_scenarios.csv"))
const DEMAND_NOISE  = DEMAND_SPREAD !== nothing
# MAIN reference checkpoints have policy input width 2·nHyd (no ξ slot); the
# probe/open-loop parity gates are meaningless and would crash on the wider
# demand-noise policy, so demand noise requires DR_CHECKPOINT_KIND=exa.
DEMAND_NOISE && MAIN_PARITY_GATES &&
    error("demand_scenarios.csv present: demand-noise evaluation requires DR_CHECKPOINT_KIND=exa " *
          "(MAIN reference parity gates only apply to nHyd-input checkpoints)")
DEMAND_NOISE && @info "Stochastic demand ACTIVE (seeded paired demand paths)" DEMAND_SPREAD
# CPU MadNLP: same pattern as train_hydro_exa_strict.jl's SOLVER_KWARGS, but
# this evaluation runs on a CPU node (backend = nothing everywhere).
const SOLVER_KWARGS = (print_level = MadNLP.ERROR, tol = 1e-6, max_iter = MAX_ITER)

# ── Load data ─────────────────────────────────────────────────────────────────

@info "Loading power system data..."
power_data = load_power_data(PM_FILE)
@info "  nBus=$(power_data.nBus)  nGen=$(power_data.nGen)  baseMVA=$(power_data.baseMVA)  cost_deficit=$(power_data.cost_deficit)"

# Deficit cost matching MAIN's mof.json objective coefficient (see header, item 1):
# 6000 = cost_deficit ($/MWh) × baseMVA, applied per pu of shed load.
const DEFICIT_COST = power_data.cost_deficit * power_data.baseMVA
@info "  deficit_cost used for equivalence: $DEFICIT_COST (training used 1e5)"

@info "Loading hydro data (num_stages = full inflow history)..."
# num_stages = nothing keeps the RAW rows of inflows.csv (47 for Bolivia —
# fewer than the 96 eval stages). MAIN's read_inflow tiles those rows
# vertically for longer horizons (load_hydropowermodels.jl:13-21), so stage t
# maps to raw row mod1(t, nrows); the reconstruction gate below applies the
# same cyclic convention.
hydro_data = load_hydro_data(HYDRO_FILE, INFLOW_FILE, power_data; num_stages = nothing)
nHyd = hydro_data.nHyd
@info "  nHyd=$nHyd  nScenarios=$(hydro_data.nScenarios)  K=$(hydro_data.K)"

# ── Load the MAIN policy reference ────────────────────────────────────────────

isfile(REFERENCE_FILE) || error(
    "Missing reference file $REFERENCE_FILE — run dump_paired_policy_reference.jl " *
    "in the MAIN repo first."
)
ref = JLD2.load(REFERENCE_FILE)
const T_EVAL       = Int(ref["num_eval_stages"])
const NUM_SCEN     = Int(ref["num_scenarios"])
inflow_ref  = ref["inflow_values"]          # [T × nHyd × S] authoritative values
xhat_ref    = ref["xhat_trajectories"]      # [T × nHyd × S] MAIN open-loop targets
scen_idx    = Int.(ref["scenario_indices"]) # [T × S]
x0_ref      = Float64.(ref["initial_state"])
probe_in    = Float32.(ref["probe_inputs"]) # [2 nHyd × 3]
probe_out_ref = ref["probe_outputs"]        # [nHyd × 3]
@info "Loaded reference: T=$T_EVAL, S=$NUM_SCEN from $REFERENCE_FILE"
@assert size(inflow_ref) == (T_EVAL, nHyd, NUM_SCEN)
@assert length(x0_ref) == nHyd

# Seeded paired demand paths: column s of the protocol gets the demand factors
# ξ[:, s] = protocol_demand_factors(s) — the identical path the trainer's
# protocol eval uses for that column (column-keyed StableRNG seeding; see
# hydro_power_data.jl). `nothing` when demand is deterministic.
const demand_factors = DEMAND_NOISE ?
    reduce(hcat, [protocol_demand_factors(DEMAND_SPREAD, T_EVAL, s) for s in 1:NUM_SCEN]) :
    nothing   # [T_EVAL × NUM_SCEN] or nothing

"""
    augmented_stage_w(w_t::AbstractVector, t::Int, s::Int) -> Vector{Float64}

Stage-t uncertainty block for paired scenario `s`: the inflow vector `w_t`
alone (deterministic demand), or `[w_t; ξ_t^{(s)}]` with the seeded paired
demand factor appended (stochastic demand).
"""
augmented_stage_w(w_t::AbstractVector, t::Int, s::Int) =
    DEMAND_NOISE ? vcat(Float64.(w_t), demand_factors[t, s]) : Float64.(w_t)

"""
    augmented_flat_w(w_flat::AbstractVector, s::Int) -> Vector{Float64}

Full-horizon stage-major uncertainty vector for paired scenario `s`: the flat
inflow trajectory unchanged (deterministic demand), or with the seeded paired
demand factor ξ_t^{(s)} interleaved into each stage block (stochastic demand).
"""
augmented_flat_w(w_flat::AbstractVector, s::Int) =
    DEMAND_NOISE ? augment_scenario(Float64.(w_flat), demand_factors[:, s]) : Float64.(w_flat)

# Per-stage uncertainty width used by the rollout machinery below.
const N_UNC = nHyd + (DEMAND_NOISE ? 1 : 0)

ref_string(key::AbstractString, default::AbstractString) =
    haskey(ref, key) ? string(ref[key]) : default
ref_int(key::AbstractString, default::Int) =
    haskey(ref, key) ? Int(ref[key]) : default

const CONTEXT_MODE = canonical_context_mode(
    get(ENV, "DR_CONTEXT", ref_string("context_mode", ""))
)
const CONTEXT_PERIOD = ref_int("context_period", countlines(INFLOW_FILE))
const CONTEXT_HORIZON = parse(
    Int,
    get(ENV, "DR_CONTEXT_HORIZON", string(ref_int("context_horizon", 126))),
)
CONTEXT_HORIZON >= T_EVAL ||
    error("DR_CONTEXT_HORIZON=$CONTEXT_HORIZON must cover reference T_EVAL=$T_EVAL")
const STAGE_CONTEXT = build_stage_context(CONTEXT_MODE, CONTEXT_HORIZON, CONTEXT_PERIOD)
const N_CONTEXT = isnothing(STAGE_CONTEXT) ? 0 : size(STAGE_CONTEXT, 1)
@info "Policy context" context_mode=(isempty(CONTEXT_MODE) ? "none" : CONTEXT_MODE) CONTEXT_PERIOD CONTEXT_HORIZON N_CONTEXT

# ── Build the EXA policy and load the requested checkpoint ────────────────────
# Constructor call mirrors train_hydro_exa_strict.jl exactly (sigmoid activation
# and Flux.LSTM encoder are the constructor defaults); combiner_layers must
# match the checkpoint's head architecture.

# Target-head activation must match the checkpoint's training activation
# (DR_ACTIVATION, same values as the trainer: sigmoid|hardsigmoid|stretched).
const ACTIVATION = let raw = lowercase(strip(get(ENV, "DR_ACTIVATION", "sigmoid")))
    raw in ("", "sigmoid") ? Flux.NNlib.sigmoid :
    raw == "hardsigmoid"   ? hardsigmoidsafe :
    raw == "stretched"     ? stretchedsigmoid :
    error("DR_ACTIVATION must be sigmoid, hardsigmoid, or stretched; got $raw")
end
# Diagnostic eval-time snap-to-boundary for sigmoid-trained checkpoints
# (DR_SNAP_EPS > 0): normalized targets within ε of 0/1 become exact boundary
# points — measures the cost of sigmoid's asymptotic boundary gap without any
# retraining. See TARGET_SNAP_EPS in hydro_reachable_policy.jl.
const SNAP_EPS = parse(Float32, get(ENV, "DR_SNAP_EPS", "0"))
0 <= SNAP_EPS < 0.5f0 || error("DR_SNAP_EPS must be in [0, 0.5); got $SNAP_EPS")
TARGET_SNAP_EPS[] = SNAP_EPS
SNAP_EPS > 0 && @info "Eval-time snap-to-boundary ACTIVE" SNAP_EPS

Random.seed!(42)
base_policy = hydro_reachable_policy(
    hydro_data,
    ENCODER_LAYERS;
    activation = ACTIVATION,
    combiner_layers = HEAD_LAYERS,
    n_context = N_CONTEXT,
    # Demand-noise checkpoints were trained with the encoder observing ξ_t;
    # the constructor must match the checkpoint's input width.
    n_extra_uncertainty = DEMAND_NOISE ? 1 : 0,
)
policy = isnothing(STAGE_CONTEXT) ? base_policy : ContextualPolicy(base_policy, STAGE_CONTEXT)
policy_core(policy) = policy isa ContextualPolicy ? policy.policy : policy
stage_encoder_input(policy, t::Int, w) =
    policy isa ContextualPolicy ? vcat(Float32.(context_at(policy.context, t)), w) : w
@info "Policy built" encoder_layers = ENCODER_LAYERS head_layers = HEAD_LAYERS checkpoint_kind = CHECKPOINT_KIND context_mode=(isempty(CONTEXT_MODE) ? "none" : CONTEXT_MODE)
isfile(MODEL_PATH) || error("Checkpoint not found: $MODEL_PATH (set DR_CHECKPOINT)")

"""
    load_main_checkpoint!(policy, model_state) -> String

Load a DecisionRules.jl (MAIN) checkpoint into the EXA `HydroReachablePolicy`,
returning a string describing which loading path succeeded.

# Arguments
- `policy`: EXA `HydroReachablePolicy` (encoder is a `Chain` of `Flux.LSTM`
  wrapper layers, each holding a `.cell`).
- `model_state`: the checkpoint's `Flux.state`, whose `encoder` field was saved
  from MAIN's `Chain` of BARE `LSTMCell`s (MAIN's `_as_cell` strips the `LSTM`
  wrapper), i.e. `state.encoder.layers[i] = (Wi = …, Wh = …, bias = …)`.

# Returns
- `"stock"` if the package loader `load_stateconditioned_policy!` succeeded.
- `"cell-by-cell"` if the stock loader failed and the weights were injected
  directly into each `LSTM.cell` plus the combiner.

# Notes
`load_stateconditioned_policy!` now includes the cell-by-cell MAIN-checkpoint
fallback natively (`DecisionRulesExa._load_encoder_state!`), so the stock path
is expected to SUCCEED and report `"stock"`. EXA-trained checkpoints
(train_hydro_exa_strict.jl saves `Flux.state(cpu(m))` of the same
`HydroReachablePolicy` type, encoder stored as `Flux.LSTM` wrappers) also load
through the stock path. This wrapper's own cell-by-cell
branch is retained as belt-and-braces; reaching it would indicate a loader
regression and is recorded in the output. The cell-by-cell path performs the
mathematically identical weight injection: the wrapper's `cell` has exactly
the fields `(Wi, Wh, bias)` MAIN saved.
"""
function load_main_checkpoint!(policy, model_state)
    try
        load_stateconditioned_policy!(policy, model_state)
        return "stock"
    catch err
        @warn "Stock EXA loader failed on the MAIN checkpoint (expected: MAIN saves bare LSTMCell states, EXA wraps cells in Flux.LSTM). Falling back to cell-by-cell injection." exception = err
        core = policy_core(policy)
        inner_state = hasproperty(model_state, :policy) ? getproperty(model_state, :policy) : model_state
        enc_state = getproperty(inner_state, :encoder)
        layer_states = getproperty(enc_state, :layers)
        length(layer_states) == length(core.encoder.layers) ||
            error("Encoder depth mismatch: checkpoint has $(length(layer_states)) layers, policy has $(length(core.encoder.layers))")
        for (layer, lstate) in zip(core.encoder.layers, layer_states)
            # MAIN layer state keys (Wi, Wh, bias) match LSTMCell's fields.
            Flux.loadmodel!(layer.cell, lstate)
        end
        Flux.loadmodel!(core.combiner, getproperty(inner_state, :combiner))
        Flux.reset!(policy)
        return "cell-by-cell"
    end
end

model_state = JLD2.load(MODEL_PATH, "model_state")
loader_mode = load_main_checkpoint!(policy, model_state)
@info "Checkpoint loaded via: $loader_mode  ($MODEL_PATH)"

# ── Policy-parity gate ────────────────────────────────────────────────────────

"""
    _max_abs_dev(a, b) -> Float64

Maximum absolute element-wise deviation `max_i |a_i − b_i|` between two arrays
of equal size.
"""
_max_abs_dev(a, b) = maximum(abs.(Float64.(a) .- Float64.(b)))

gate = Dict{String, Any}("loader_mode" => loader_mode)
core_policy = policy_core(policy)

# (0) Frozen hydro-metadata parity: the bounds the two policies scale into.
gate["dev_K"]            = abs(core_policy.K - Float64(ref["policy_K"]))
gate["dev_min_vol"]      = _max_abs_dev(core_policy.min_vol,  ref["policy_min_vol"])
gate["dev_max_vol"]      = _max_abs_dev(core_policy.max_vol,  ref["policy_max_vol"])
gate["dev_min_turn"]     = _max_abs_dev(core_policy.min_turn, ref["policy_min_turn"])
gate["dev_max_turn"]     = _max_abs_dev(core_policy.max_turn, ref["policy_max_turn"])
gate["dev_upstream_max"] = _max_abs_dev(core_policy.upstream_max_inflow, ref["policy_upstream_max"])
@info "Metadata deviations" gate["dev_K"] gate["dev_min_vol"] gate["dev_max_vol"] gate["dev_min_turn"] gate["dev_max_turn"] gate["dev_upstream_max"]

# (1) Probe parity: single policy calls from the reset state. Tests weight
# loading independent of any recurrence-threading semantics. ONLY meaningful
# when evaluating the exact MAIN reference checkpoint the probe outputs were
# generated from — SKIPPED for EXA-trained checkpoints.
probe_out_exa = fill(NaN, nHyd, 3)
if MAIN_PARITY_GATES
    for p in 1:3
        Flux.reset!(policy)      # REAL reset: each probe starts from initialstates
        probe_out_exa[:, p] = Float64.(policy(probe_in[:, p]))
    end
    gate["probe_outputs_exa"] = probe_out_exa
    gate["probe_max_dev"] = _max_abs_dev(probe_out_exa, probe_out_ref)
    # Float32 tolerance: relative to the target scale (max_vol up to ~138).
    probe_pass = gate["probe_max_dev"] <= 1e-5 * max(1.0, maximum(abs.(probe_out_ref)))
    gate["probe_pass"] = probe_pass
    println(probe_pass ? "GATE probe parity: PASS" : "GATE probe parity: FAIL",
            "  (max abs dev = $(gate["probe_max_dev"]))")
else
    # The reference probe outputs are the REFERENCE checkpoint's responses;
    # an independently trained EXA checkpoint has different weights, so a
    # weight-parity comparison would fail by construction and prove nothing.
    gate["probe_max_dev"] = NaN
    gate["probe_pass"] = "skipped"
    println("GATE probe parity: SKIPPED (DR_CHECKPOINT_KIND=exa — reference " *
            "probe outputs only characterize the MAIN reference checkpoint)")
end

# (2) Inflow reconstruction: rebuild w[t, r, s] from the EXA loader's
# scenario_inflows and the reference scenario indices; compare against the
# reference values. This validates scenario indexing and units across loaders
# (the historically buggy spot).
inflow_recon_dev = 0.0
first_mismatch = nothing
# MAIN's read_inflow tiles the raw inflow rows vertically when num_stages
# exceeds the file length (load_hydropowermodels.jl:13-21), so stage t maps to
# raw row mod1(t, nrows). The EXA loader keeps the raw (untiled) matrix; apply
# the same cyclic convention here so both sides index identical physical data.
n_inflow_rows = size(hydro_data.scenario_inflows[1], 1)
for s in 1:NUM_SCEN, t in 1:T_EVAL, r in 1:nHyd
    v_exa = hydro_data.scenario_inflows[r][mod1(t, n_inflow_rows), scen_idx[t, s]]
    dev = abs(v_exa - inflow_ref[t, r, s])
    if dev > inflow_recon_dev
        global inflow_recon_dev = dev
        global first_mismatch = (t = t, r = r, s = s, exa = v_exa, ref = inflow_ref[t, r, s])
    end
end
gate["inflow_recon_max_dev"] = inflow_recon_dev
gate["inflow_pass"] = inflow_recon_dev == 0.0
if gate["inflow_pass"]
    println("GATE inflow indexing: PASS (exact reconstruction)")
else
    println("GATE inflow indexing: FAIL  (max abs dev = $inflow_recon_dev at $first_mismatch)")
    # Diagnose common failure patterns at the first mismatching coordinate.
    # All probes use cyclic row indexing so t beyond the raw row count cannot
    # itself throw while diagnosing an indexing mismatch.
    t, r, s = first_mismatch.t, first_mismatch.r, first_mismatch.s
    ω = scen_idx[t, s]
    tr = mod1(t, n_inflow_rows)
    println("  diagnostics at (t=$t → raw row $tr, r=$r, s=$s, ω=$ω):")
    println("    ref value                     = $(inflow_ref[t, r, s])")
    println("    exa [tr, ω]                   = $(hydro_data.scenario_inflows[r][tr, ω])")
    ω <= size(hydro_data.scenario_inflows[r], 1) && tr <= size(hydro_data.scenario_inflows[r], 2) &&
        println("    exa transposed [ω, tr]        = $(hydro_data.scenario_inflows[r][ω, tr])")
    println("    exa row-offset [tr+1, ω]      = $(hydro_data.scenario_inflows[r][mod1(tr + 1, n_inflow_rows), ω])")
    println("    exa unit ratio (exa/ref)      = $(hydro_data.scenario_inflows[r][tr, ω] / inflow_ref[t, r, s])")
    println("  CONTINUING with the REFERENCE inflow values as authoritative scenario data.")
end

# (3) Open-loop trajectory with the EXA policy AS-IS (state-threaded forward
# pass — see header item 6). Expected to MATCH MAIN exactly (max|Δ| = 0.0).
"""
    open_loop_targets_asis(policy, x0, w_mat) -> Matrix{Float64}

Open-loop target recursion `x̂_t = π(w_t, x̂_{t-1})`, `x̂_0 = x0`, using the EXA
policy exactly as its own training/eval pipeline calls it. The policy forward
pass now threads the LSTM recurrent state across stages (DecisionRules.jl
semantics), so this trajectory is expected to reproduce MAIN's exactly.

# Arguments
- `policy`: EXA `HydroReachablePolicy`.
- `x0`: initial reservoir state (length nHyd).
- `w_mat`: `[T × nHyd]` inflow values.

# Returns
- `[T × nHyd]` matrix of open-loop targets.
"""
function open_loop_targets_asis(policy, x0, w_mat)
    Flux.reset!(policy)                       # REAL reset: start from initialstates
    T, nH = size(w_mat)
    prev = Float32.(x0)
    out = zeros(Float64, T, nH)
    for t in 1:T
        target = policy(vcat(Float32.(w_mat[t, :]), prev))
        out[t, :] = Float64.(target)
        prev = Float32.(target)               # open-loop: previous target as next state
    end
    return out
end

# (4) Open-loop trajectory with MANUALLY threaded recurrent state, replicating
# MAIN's `_step_encoder` semantics with the SAME loaded weights. Retained as an
# independent cross-check of (3): both are now expected to pass; if (3) fails
# while (4) passes, the policy forward pass has regressed from MAIN's
# threading semantics (weight parity still holds).
"""
    open_loop_targets_threaded(policy, x0, w_mat) -> Matrix{Float64}

Open-loop target recursion with the LSTM state carried across stages,
mirroring DecisionRules.jl's `HydroReachablePolicy` forward pass:

```math
(h_t^{(l)}, c_t^{(l)}) = \\mathrm{LSTMCell}^{(l)}(h_t^{(l-1)}, (h_{t-1}^{(l)}, c_{t-1}^{(l)})),
\\qquad \\hat{x}_t = \\mathrm{lower} + (\\mathrm{upper} - \\mathrm{lower}) \\cdot \\sigma(\\mathrm{combiner}([h_t; \\hat{x}_{t-1}]))
```

followed by the same cascade clamp as the EXA forward pass. Uses
`layer.cell(x, state)` directly (Flux 0.16 `LSTMCell` returns
`(h, (h, c))`).

# Arguments
- `policy`: EXA `HydroReachablePolicy` with MAIN weights loaded.
- `x0`: initial reservoir state.
- `w_mat`: `[T × nHyd]` inflow values.

# Returns
- `[T × nHyd]` matrix of open-loop targets under MAIN's recurrence semantics.
"""
function open_loop_targets_threaded(policy, x0, w_mat)
    core = policy_core(policy)
    cells = [layer.cell for layer in core.encoder.layers]
    # Zero initial recurrent state per layer, as Flux.initialstates gives.
    states = Any[Flux.initialstates(c) for c in cells]
    T, nH = size(w_mat)
    prev = Float32.(x0)
    out = zeros(Float64, T, nH)
    for t in 1:T
        w = Float32.(w_mat[t, :])
        # Thread the recurrent state layer by layer across stages.
        h = stage_encoder_input(policy, t, w)
        for (i, c) in enumerate(cells)
            h, states[i] = c(h, states[i])
        end
        # Same head + reachable-bounds scaling + cascade clamp as the EXA
        # forward pass (these helpers come from hydro_reachable_policy.jl).
        y = core.combiner(vcat(h, prev))
        lower, upper = _hydro_reachable_bounds(core, w, prev, y)
        raw = lower .+ (upper .- lower) .* y
        target = isempty(core.cascade) ? raw :
                 min.(raw, _cascade_upper_bounds(core, raw, w, prev))
        out[t, :] = Float64.(target)
        prev = Float32.(target)
    end
    return out
end

if MAIN_PARITY_GATES
    w_s1 = inflow_ref[:, :, 1]                               # scenario 1 inflows [T × nHyd]
    xhat_s1_ref = xhat_ref[:, :, 1]                          # MAIN open-loop trajectory
    global xhat_s1_asis = open_loop_targets_asis(policy, x0_ref, w_s1)
    global xhat_s1_threaded = open_loop_targets_threaded(policy, x0_ref, w_s1)
    gate["openloop_asis_max_dev"]     = _max_abs_dev(xhat_s1_asis, xhat_s1_ref)
    gate["openloop_threaded_max_dev"] = _max_abs_dev(xhat_s1_threaded, xhat_s1_ref)
    tol_traj = 1e-5 * max(1.0, maximum(abs.(xhat_s1_ref)))
    gate["openloop_asis_pass"]     = gate["openloop_asis_max_dev"] <= tol_traj
    gate["openloop_threaded_pass"] = gate["openloop_threaded_max_dev"] <= tol_traj
    println("GATE open-loop (EXA policy as-is):   ",
            gate["openloop_asis_pass"] ? "PASS" : "FAIL",
            "  (max abs dev = $(gate["openloop_asis_max_dev"]))")
    println("GATE open-loop (state-threaded diag): ",
            gate["openloop_threaded_pass"] ? "PASS" : "FAIL",
            "  (max abs dev = $(gate["openloop_threaded_max_dev"]))")
else
    # The reference open-loop trajectories (`xhat_trajectories`) were rolled
    # out by the MAIN reference checkpoint; comparing an independently trained
    # EXA checkpoint's trajectory against them is meaningless (its targets
    # SHOULD differ). Its own open-loop targets are still saved via the DE
    # leg's de_target_trajectories.
    global xhat_s1_asis = zeros(Float64, 0, 0)
    global xhat_s1_threaded = zeros(Float64, 0, 0)
    gate["openloop_asis_max_dev"]     = NaN
    gate["openloop_threaded_max_dev"] = NaN
    gate["openloop_asis_pass"]     = "skipped"
    gate["openloop_threaded_pass"] = "skipped"
    println("GATE open-loop trajectories: SKIPPED (DR_CHECKPOINT_KIND=exa — " *
            "reference trajectories only characterize the MAIN reference checkpoint)")
end

# ── Stage problem and callbacks (copied from train_hydro_exa_strict.jl) ───────
# demand_matrix = nothing: the builder bakes in load_scaler × default demand for
# its single stage, matching the mof.json's 0.6-scaled loads at every stage.

function _build_rollout_de()
    build_hydro_de(power_data, hydro_data, 1;
        backend        = nothing,             # CPU node — no GPU backend
        float_type     = Float64,
        formulation    = FORMULATION,
        target_penalty = :auto,               # irrelevant in strict mode
        deficit_cost   = DEFICIT_COST,        # 6000 = MAIN mof.json coefficient
        demand_matrix  = nothing,
        load_scaler    = LOAD_SCALER,
        strict_targets = true,
        reactive_deficit_cost = REACTIVE_DEFICIT_COST,  # header item 3
        # Stochastic demand: prepare_solve! multiplies the (constant, 0.6 ×
        # default) base demand by the ξ_t carried in the stage's wt block.
        demand_spread  = DEMAND_SPREAD,
    )
end
@info "Building 1-stage strict ExaModels stage problem (CPU, formulation=$FORMULATION)..."
rollout_prob = _build_rollout_de()

# Same callback as train_hydro_exa_strict.jl's set_hydro_rollout_stage!, minus
# the demand update (demand_mat === nothing here).
function set_hydro_rollout_stage!(stage_prob, state_in, wt, target, stage)
    ExaModels.set_parameter!(stage_prob.core, stage_prob.p_x0, state_in)
    ExaModels.set_parameter!(stage_prob.core, stage_prob.p_inflow, wt)
    ExaModels.set_parameter!(stage_prob.core, stage_prob.p_target, target)
    prepare_solve!(stage_prob, state_in, wt, target)
    return stage_prob
end

# Realized state: in strict mode hydro_solution reads the (cascade-clamped)
# reservoir parameter trajectory, so the realized state equals the clamped
# target — the strict analogue of MAIN reading value(reservoir_out).
hydro_realized_state(stage_prob, result) =
    hydro_solution(stage_prob, result).reservoir[:, end]

# Thermal generator positions, derived with the same hydro.json/PowerModels.json
# logic as MAIN's eval_paired_tsddr.jl lines 74-80: a generator is thermal iff
# its grid index is not any hydro unit's index_grid.
hydro_grid_idx = Set(power_data.gens[h.gen_pos].idx for h in hydro_data.units)
thermal_pos = [pos for (pos, g) in enumerate(power_data.gens) if !(g.idx in hydro_grid_idx)]
@info "Thermal generators: $(length(thermal_pos)) of $(power_data.nGen)"

"""
    volume_to_mw(volume; k = 0.0036) -> Float64

Convert a reservoir volume (hm³) to the MW-equivalent used by MAIN's
`eval_paired_tsddr.jl`: `volume / k` with `k = 0.0036`.
"""
volume_to_mw(volume; k = 0.0036) = volume / k

"""
    recompute_stage_costs(sol, power_data, deficit_cost) -> Vector{Float64}

Recompute per-stage objective values from a `hydro_solution` NamedTuple:

```math
c_t = \\sum_g (c_{2,g}\\, pg_{g,t}^2 + c_{1,g}\\, pg_{g,t}) + c_d \\sum_b \\mathrm{deficit}_{b,t}
```

which is the complete strict-mode EXA objective (no other terms exist). Used
both to split the full-horizon DE objective into stages and as an internal
consistency check on 1-stage solves.

# Arguments
- `sol`: NamedTuple from [`hydro_solution`](@ref) (`pg` is `[nGen × T]`,
  `deficit` is `[nBus × T]`).
- `power_data::PowerData`: generator cost coefficients.
- `deficit_cost::Real`: load-shedding cost per pu (6000 for this check).

# Returns
- `Vector{Float64}` of length `T` with per-stage objective values.
"""
function recompute_stage_costs(sol, power_data, deficit_cost)
    T = size(sol.pg, 2)
    costs = zeros(Float64, T)
    for t in 1:T
        c = 0.0
        for (gpos, g) in enumerate(power_data.gens)
            # Quadratic + linear generation cost (c2 = 0 for all Bolivia gens).
            c += g.cost2 * sol.pg[gpos, t]^2 + g.cost1 * sol.pg[gpos, t]
        end
        # Per-bus active load shedding at the deficit cost.
        c += deficit_cost * sum(@view sol.deficit[:, t])
        costs[t] = c
    end
    return costs
end

# ── Stage-wise leg: closed-loop strict rollout on all paired scenarios ────────
# Mirrors DecisionRulesExa.rollout_tsddr's fresh-solver path (reuse_solver =
# false, warmstart irrelevant for fresh solvers, retry_on_failure = true) while
# additionally extracting per-stage solution components that rollout_tsddr does
# not expose (pg, deficit, spill, outflow). A rollout_tsddr cross-check on
# scenario 1 verifies the custom loop matches the package machinery.

@info "Stage-wise leg: $NUM_SCEN scenarios × $T_EVAL stages (cold MadNLP solves)..."

costs_stagewise   = fill(NaN, NUM_SCEN)             # per-scenario total objective
stage_costs       = fill(NaN, T_EVAL, NUM_SCEN)     # per-stage objective
vol_trajectories  = fill(NaN, T_EVAL, NUM_SCEN)     # Σ_r volume_to_mw(state_r) after stage t
gen_trajectories  = fill(NaN, T_EVAL, NUM_SCEN)     # Σ_thermal pg × baseMVA at stage t
reservoir_traj    = fill(NaN, nHyd, T_EVAL + 1, NUM_SCEN)  # realized states incl. x0
outflow_traj      = fill(NaN, nHyd, T_EVAL, NUM_SCEN)
spill_traj        = fill(NaN, nHyd, T_EVAL, NUM_SCEN)
target_traj       = fill(NaN, nHyd, T_EVAL, NUM_SCEN)      # raw policy targets
max_deficit       = fill(NaN, NUM_SCEN)             # max_b,t deficit (pu)
sum_deficit       = fill(NaN, NUM_SCEN)             # Σ_b,t deficit (pu)
max_abs_deficit_q = fill(NaN, NUM_SCEN)             # max_b,t |deficit_q| (pu) — EXA-only slack
scenario_ok       = falses(NUM_SCEN)
n_retries_total   = 0
max_objective_recompute_dev = 0.0                    # internal consistency check

baseMVA = power_data.baseMVA

for s in 1:NUM_SCEN
    Flux.reset!(policy)                        # REAL reset at the scenario boundary
    state = copy(x0_ref)                       # closed-loop realized state (Float64)
    reservoir_traj[:, 1, s] = state
    total = 0.0
    failed = false
    for t in 1:T_EVAL
        # Authoritative inflow values from the MAIN reference (see gate item 2),
        # with the seeded paired demand factor ξ_t^{(s)} appended when
        # stochastic demand is active ([w_t; ξ_t] block).
        w_t = augmented_stage_w(inflow_ref[t, :, s], t, s)
        # Policy call with Float32 inputs, exactly as MAIN's closed loop does
        # (the policy slices the physical inflow internally).
        x_hat = Float64.(policy(vcat(Float32.(w_t), Float32.(state))))
        target_traj[:, t, s] = x_hat

        # Write x_{t-1}, [w_t; ξ_t], x̂_t into the strict stage problem
        # (prepare_solve! applies base_demand · ξ_t via set_demand!).
        set_hydro_rollout_stage!(rollout_prob, state, w_t, x_hat, t)

        # Cold one-shot solve (rollout_tsddr's fresh-solver path).
        result = MadNLP.madnlp(rollout_prob.model; SOLVER_KWARGS...)
        if !solve_succeeded(result) || !isfinite(result.objective)
            # Single cold retry, mirroring rollout_tsddr's retry_on_failure.
            global n_retries_total += 1
            result = MadNLP.madnlp(rollout_prob.model; SOLVER_KWARGS...)
        end
        if !solve_succeeded(result) || !isfinite(result.objective)
            @warn "Stage solve failed after retry" scenario = s stage = t status = result.status
            failed = true
            break
        end

        sol = hydro_solution(rollout_prob, result)
        total += result.objective
        stage_costs[t, s] = result.objective
        # Internal consistency: the strict stage objective must equal the
        # recomputed gen+deficit cost exactly (same terms, same data).
        recomputed = recompute_stage_costs(sol, power_data, DEFICIT_COST)[1]
        global max_objective_recompute_dev =
            max(max_objective_recompute_dev, abs(recomputed - result.objective))

        # Advance the closed loop on the realized state (= clamped target).
        state = Float64.(sol.reservoir[:, end])
        reservoir_traj[:, t + 1, s] = state
        outflow_traj[:, t, s] = sol.outflow[:, 1]
        spill_traj[:, t, s]   = sol.spill[:, 1]

        # Operative metrics with the IDENTICAL formulas as MAIN's evaluation.
        vol_trajectories[t, s] = sum(volume_to_mw(state[r]) for r in 1:nHyd)
        gen_trajectories[t, s] = sum(sol.pg[g, 1] * baseMVA for g in thermal_pos)

        # Deficit activity (quantifies whether cost/slack differences are inert).
        def_max = maximum(sol.deficit[:, 1])
        def_sum = sum(sol.deficit[:, 1])
        dq_max  = maximum(abs.(sol.deficit_q[:, 1]))
        max_deficit[s]       = isnan(max_deficit[s]) ? def_max : max(max_deficit[s], def_max)
        sum_deficit[s]       = isnan(sum_deficit[s]) ? def_sum : sum_deficit[s] + def_sum
        max_abs_deficit_q[s] = isnan(max_abs_deficit_q[s]) ? dq_max : max(max_abs_deficit_q[s], dq_max)
    end
    if !failed
        costs_stagewise[s] = total
        scenario_ok[s] = true
    end
    if s % 10 == 0 || s == NUM_SCEN
        ok_costs = costs_stagewise[1:s][scenario_ok[1:s]]
        running_mean = isempty(ok_costs) ? NaN : mean(ok_costs)
        println("  [$s/$NUM_SCEN] cost = $(round(total; digits=1)), running mean = $(round(running_mean; digits=1)), retries = $n_retries_total")
    end
end
@info "Stage-wise leg done" n_ok = count(scenario_ok) n_retries_total max_objective_recompute_dev

# ── rollout_tsddr cross-check on scenario 1 ───────────────────────────────────
# Runs the actual package machinery (same callbacks) to verify the custom loop
# above reproduces it. Float32 initial state so the policy sees Float32 inputs;
# the solver interface buffers are Float64 regardless.
w_flat_s1 = vec(permutedims(inflow_ref[:, :, 1]))   # stage-major [w_1; w_2; …]
rollout_check = rollout_tsddr(
    policy,
    Float32.(x0_ref),
    rollout_prob,
    Float32.(w_flat_s1);
    horizon = T_EVAL,
    n_uncertainty = nHyd,
    set_stage_parameters! = set_hydro_rollout_stage!,
    realized_state = hydro_realized_state,
    madnlp_kwargs = SOLVER_KWARGS,
    warmstart = false,
    policy_state = :realized,
    retry_on_failure = true,
)
rollout_check_obj = rollout_check === nothing ? NaN : rollout_check.objective
@info "rollout_tsddr cross-check (scenario 1)" custom_loop = costs_stagewise[1] rollout_tsddr = rollout_check_obj

# ── DE leg: strict full-horizon deterministic equivalent ──────────────────────
# Strict-mode claim under test: with the target trajectory generated by the
# open-loop recursion (previous target as next policy state — exactly
# train_hydro_exa_strict.jl's rollout_reachable_targets), the T-stage strict DE
# objective equals the stage-wise sum for the same scenario, because the
# reservoir path is pinned to the same targets and stages decouple.

@info "DE leg: building $T_EVAL-stage strict DE and solving $NUM_DE_SCENARIOS scenarios..."
de_prob = build_hydro_de(power_data, hydro_data, T_EVAL;
    backend        = nothing,
    float_type     = Float64,
    formulation    = FORMULATION,
    target_penalty = :auto,
    deficit_cost   = DEFICIT_COST,
    demand_matrix  = nothing,
    load_scaler    = LOAD_SCALER,
    strict_targets = true,
    reactive_deficit_cost = REACTIVE_DEFICIT_COST,  # header item 3
)

"""
    rollout_reachable_targets(policy, x0, w_flat, T, nHyd) -> Vector{Float64}

Roll out the reachable-policy target trajectory for the strict regular DE
(verbatim semantics of train_hydro_exa_strict.jl): start from the feasible
`x0`, feed `[w_t; previous_target]` to the policy, and record each target as
the next previous state, so every target is one-stage reachable from the prior
target by induction.

# Arguments
- `policy`: reachable hydro policy with input `[inflow; previous_state]`.
- `x0`: initial reservoir state.
- `w_flat`: stage-major flat inflow vector of length `T * nHyd`.
- `T::Int`: number of stages.
- `nHyd::Int`: number of hydro reservoirs.

# Returns
- `Vector{Float64}`: stage-major target trajectory for
  `ExaModels.set_parameter!(prob.core, prob.p_target, targets)`.
"""
function rollout_reachable_targets(policy, x0, w_flat, T, nHyd)
    Flux.reset!(policy)
    prev = Float32.(x0)
    targets = Vector{Vector{Float32}}(undef, T)
    for t in 1:T
        wt = Float32.(view(w_flat, ((t - 1) * nHyd + 1):(t * nHyd)))
        target = policy(vcat(wt, prev))
        targets[t] = Float32.(target)
        prev = targets[t]
    end
    return Float64.(vcat(targets...))
end

de_costs        = fill(NaN, NUM_DE_SCENARIOS)
de_stage_costs  = fill(NaN, T_EVAL, NUM_DE_SCENARIOS)
de_vol          = fill(NaN, T_EVAL, NUM_DE_SCENARIOS)
de_gen          = fill(NaN, T_EVAL, NUM_DE_SCENARIOS)
de_reservoir    = fill(NaN, nHyd, T_EVAL + 1, NUM_DE_SCENARIOS)
de_outflow      = fill(NaN, nHyd, T_EVAL, NUM_DE_SCENARIOS)
de_spill        = fill(NaN, nHyd, T_EVAL, NUM_DE_SCENARIOS)
de_targets      = fill(NaN, nHyd, T_EVAL, NUM_DE_SCENARIOS)
de_max_deficit  = fill(NaN, NUM_DE_SCENARIOS)
de_max_abs_deficit_q = fill(NaN, NUM_DE_SCENARIOS)
de_ok           = falses(NUM_DE_SCENARIOS)

for s in 1:NUM_DE_SCENARIOS
    # Stage-major flat inflow vector for scenario s from the reference values.
    w_flat = vec(permutedims(inflow_ref[:, :, s]))
    # Open-loop target trajectory with the EXA policy (previous target as state).
    targets = rollout_reachable_targets(policy, x0_ref, w_flat, T_EVAL, nHyd)
    de_targets[:, :, s] = reshape(targets, nHyd, T_EVAL)

    # Write parameters and pin the strict reservoir path (prepare_solve! also
    # applies the Float64 cascade clamp and enforces p_reservoir[1:nHyd] = x0).
    ExaModels.set_parameter!(de_prob.core, de_prob.p_x0, x0_ref)
    ExaModels.set_parameter!(de_prob.core, de_prob.p_inflow, w_flat)
    ExaModels.set_parameter!(de_prob.core, de_prob.p_target, targets)
    prepare_solve!(de_prob, x0_ref, w_flat, targets)

    result = MadNLP.madnlp(de_prob.model; SOLVER_KWARGS...)
    if !solve_succeeded(result) || !isfinite(result.objective)
        @warn "DE solve failed" scenario = s status = result.status
        continue
    end
    de_ok[s] = true
    de_costs[s] = result.objective

    sol = hydro_solution(de_prob, result)
    de_stage_costs[:, s] = recompute_stage_costs(sol, power_data, DEFICIT_COST)
    de_reservoir[:, :, s] = sol.reservoir
    de_outflow[:, :, s]   = sol.outflow
    de_spill[:, :, s]     = sol.spill
    for t in 1:T_EVAL
        # Operative metrics with the identical MAIN formulas, taken from the
        # post-stage reservoir state and per-stage thermal generation.
        de_vol[t, s] = sum(volume_to_mw(sol.reservoir[r, t + 1]) for r in 1:nHyd)
        de_gen[t, s] = sum(sol.pg[g, t] * baseMVA for g in thermal_pos)
    end
    de_max_deficit[s] = maximum(sol.deficit)
    de_max_abs_deficit_q[s] = maximum(abs.(sol.deficit_q))
    println("  DE [$s/$NUM_DE_SCENARIOS] objective = $(round(result.objective; digits=1)), stagewise total = $(round(costs_stagewise[s]; digits=1))")
end
@info "DE leg done" n_ok = count(de_ok)

# ── Save everything ────────────────────────────────────────────────────────────
out_dir = joinpath(SCRIPT_DIR, CASE_NAME, FORM_LABEL, "results")
mkpath(out_dir)
# DR_OUTPUT_TAG suffixes the filename so non-reference evaluations never
# clobber the untagged reference results file.
out_file = joinpath(out_dir, "paired_exa_strict$(OUT_SUFFIX).jld2")
# Provenance knobs, recorded whenever any of the new env vars was set (the
# all-defaults run keeps exactly the historical key set).
knob_extras = RECORD_KNOBS ? (
    checkpoint_path = MODEL_PATH,
    checkpoint_kind = CHECKPOINT_KIND,
    encoder_layers = ENCODER_LAYERS,
    head_layers = HEAD_LAYERS,
    context_mode = isempty(CONTEXT_MODE) ? "none" : CONTEXT_MODE,
    context_period = CONTEXT_PERIOD,
    context_horizon = CONTEXT_HORIZON,
    n_context = N_CONTEXT,
    output_tag = OUTPUT_TAG,
    main_parity_gates_applied = MAIN_PARITY_GATES,
) : (;)
jldsave(out_file;
    knob_extras...,
    # Gate results (policy parity, inflow indexing, metadata)
    gate_loader_mode = loader_mode,
    gate_probe_max_dev = gate["probe_max_dev"],
    gate_probe_pass = gate["probe_pass"],
    gate_probe_outputs_exa = probe_out_exa,
    gate_inflow_recon_max_dev = gate["inflow_recon_max_dev"],
    gate_inflow_pass = gate["inflow_pass"],
    gate_openloop_asis_max_dev = gate["openloop_asis_max_dev"],
    gate_openloop_asis_pass = gate["openloop_asis_pass"],
    gate_openloop_threaded_max_dev = gate["openloop_threaded_max_dev"],
    gate_openloop_threaded_pass = gate["openloop_threaded_pass"],
    gate_dev_K = gate["dev_K"],
    gate_dev_min_vol = gate["dev_min_vol"],
    gate_dev_max_vol = gate["dev_max_vol"],
    gate_dev_min_turn = gate["dev_min_turn"],
    gate_dev_max_turn = gate["dev_max_turn"],
    gate_dev_upstream_max = gate["dev_upstream_max"],
    xhat_s1_asis = xhat_s1_asis,
    xhat_s1_threaded = xhat_s1_threaded,
    # Stage-wise leg
    costs = costs_stagewise,
    stage_costs = stage_costs,
    vol_trajectories = vol_trajectories,
    gen_trajectories = gen_trajectories,
    reservoir_trajectories = reservoir_traj,
    outflow_trajectories = outflow_traj,
    spill_trajectories = spill_traj,
    target_trajectories = target_traj,
    max_deficit = max_deficit,
    sum_deficit = sum_deficit,
    max_abs_deficit_q = max_abs_deficit_q,
    scenario_ok = collect(scenario_ok),
    n_retries_total = n_retries_total,
    max_objective_recompute_dev = max_objective_recompute_dev,
    rollout_tsddr_check_objective = rollout_check_obj,
    # DE leg
    de_costs = de_costs,
    de_stage_costs = de_stage_costs,
    de_vol_trajectories = de_vol,
    de_gen_trajectories = de_gen,
    de_reservoir_trajectories = de_reservoir,
    de_outflow_trajectories = de_outflow,
    de_spill_trajectories = de_spill,
    de_target_trajectories = de_targets,
    de_max_deficit = de_max_deficit,
    de_max_abs_deficit_q = de_max_abs_deficit_q,
    de_ok = collect(de_ok),
    # Configuration provenance
    deficit_cost_used = DEFICIT_COST,
    load_scaler_used = LOAD_SCALER,
    reactive_deficit_setting = REACTIVE_DEFICIT_RAW,
    reactive_deficit_cost_used =
        REACTIVE_DEFICIT_COST === nothing ? "free" : Float64(REACTIVE_DEFICIT_COST),
    solver_tol = SOLVER_KWARGS.tol,
    solver_max_iter = SOLVER_KWARGS.max_iter,
    model_path = MODEL_PATH,
    reference_file = REFERENCE_FILE,
    num_eval_stages = T_EVAL,
    num_scenarios = NUM_SCEN,
    num_de_scenarios = NUM_DE_SCENARIOS,
)
println("Saved: $out_file")

# ── Final summary ─────────────────────────────────────────────────────────────

"""
    _split_csv_line(line) -> Vector{String}

Split one CSV line into fields with minimal double-quote awareness: commas
inside `"…"` do not delimit (needed for the MAIN header column
`"TS-DDR (strict, paired)"`). No escape handling beyond quote toggling.
"""
function _split_csv_line(line::AbstractString)
    fields = String[]                       # accumulated fields
    buf = IOBuffer()                        # current field characters
    inq = false                             # inside a quoted region?
    for c in line
        if c == '"'
            inq = !inq                      # toggle quoting; quotes are dropped
        elseif c == ',' && !inq
            push!(fields, String(take!(buf)))  # unquoted comma ends the field
        else
            write(buf, c)
        end
    end
    push!(fields, String(take!(buf)))       # trailing field
    return fields
end

"""
    _mean_csv_column(path, needle) -> Float64

Mean of the numeric column whose header contains `needle` in the CSV at
`path`:

```math
\\bar{c} = \\frac{1}{n} \\sum_{i=1}^{n} c_i
```

Returns `NaN` when the file is missing, the column is not found, or no row
parses — the summary then simply reports `NaN` for that baseline.
"""
function _mean_csv_column(path::AbstractString, needle::AbstractString)
    isfile(path) || return NaN                            # ground truth absent
    lines = readlines(path)
    length(lines) >= 2 || return NaN                      # header + ≥1 data row
    header = _split_csv_line(lines[1])                    # quote-aware header parse
    col = findfirst(h -> occursin(needle, h), header)     # column by substring
    col === nothing && return NaN
    vals = Float64[]
    for ln in lines[2:end]
        fs = split(ln, ',')                               # data rows are plain numeric
        length(fs) == length(header) || continue          # skip malformed rows
        v = tryparse(Float64, strip(fs[col]))
        v === nothing || push!(vals, v)
    end
    return isempty(vals) ? NaN : mean(vals)
end

# Successful-scenario cost vectors for the two legs of THIS evaluation.
ok_costs    = costs_stagewise[collect(scenario_ok)]
de_ok_costs = de_costs[collect(de_ok)]

# Untagged ground-truth baselines from the MAIN repo (comparability anchors):
# the .026 reference checkpoint's stage-wise mean (results/paired_strict_rollout.jld2)
# and the SDDP paired mean (SDDP-SOC column of paired_costs.csv). NaN if absent.
main_gt_file = joinpath(MAIN_HPM_DIR, CASE_NAME, FORM_LABEL, "results", "paired_strict_rollout.jld2")
main_gt_mean = isfile(main_gt_file) ? mean(Float64.(JLD2.load(main_gt_file, "costs"))) : NaN
sddp_mean = _mean_csv_column(
    joinpath(MAIN_HPM_DIR, CASE_NAME, FORM_LABEL, "paired_costs.csv"), "SDDP")

println("\n" * "=" ^ 64)
println("SUMMARY — paired EXA strict evaluation")
println("  Checkpoint:          $MODEL_PATH")
println("  Kind:                $CHECKPOINT_KIND  (encoder=$(ENCODER_LAYERS), head=$(HEAD_LAYERS))")
println("  Context:             $(isempty(CONTEXT_MODE) ? "none" : CONTEXT_MODE)  (period=$CONTEXT_PERIOD, horizon=$CONTEXT_HORIZON)")
println("  Reference:           $REFERENCE_FILE")
println("  Output tag:          $(isempty(OUTPUT_TAG) ? "(none)" : OUTPUT_TAG)")
println("  Activation:          $(ACTIVATION)  snap_eps=$(SNAP_EPS)")
println("  Stage-wise mean:     $(round(mean(ok_costs); digits=1))  over $(length(ok_costs))/$NUM_SCEN scenarios")
println("  Stage-wise std:      $(round(std(ok_costs); digits=1))")
println("  DE-leg mean:         $(round(mean(de_ok_costs); digits=1))  over $(length(de_ok_costs))/$NUM_DE_SCENARIOS scenarios")
println("  MAIN .026 mean (GT): $(round(main_gt_mean; digits=1))")
println("  SDDP mean (GT):      $(round(sddp_mean; digits=1))")
println("=" ^ 64)
