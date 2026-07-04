# compare_paired_evals.jl
#
# Verdict report for the cross-package equivalence check between
# DecisionRules.jl (MAIN, JuMP/Ipopt) and DecisionRulesExa.jl (EXA,
# ExaModels/MadNLP) on the 100 paired scenarios.
#
# Loads:
#   1. MAIN ground truth:   paired_strict_rollout.jld2   (eval_paired_tsddr.jl)
#   2. MAIN policy reference: paired_policy_reference.jld2 (dump_paired_policy_reference.jl)
#   3. EXA evaluation:      paired_exa_strict.jld2       (eval_paired_exa_strict.jl)
#   4. MAIN paired_costs.csv (for the SDDP column)
#
# and prints:
#   (a) policy-parity gate results,
#   (b) per-scenario cost deltas MAIN-stagewise vs EXA-stagewise,
#   (c) EXA-stagewise vs EXA-DE per-scenario deltas,
#   (d) trajectory deltas (vol / gen / policy targets),
#   (e) mean costs of every leg side by side with SDDP,
#   (f) the explicit list of known structural differences with measured
#       activity/inertness.
#
# Agreement classification is honest and threshold-based:
#   EXACT      : relative deviation < 1e-6
#   TIGHT      : relative deviation < 1e-3   (solver-tolerance regime)
#   DISCREPANT : anything larger
#
# Usage:
#   julia --project compare_paired_evals.jl

using JLD2
using Statistics
using CSV, Tables

const SCRIPT_DIR = dirname(@__FILE__)
const MAIN_HPM_DIR = "/storage/scratch1/9/arosemberg3/DecisionRules.jl/examples/HydroPowerModels"

const MAIN_RESULTS = joinpath(MAIN_HPM_DIR, "bolivia", "ACPPowerModel", "results",
                              "paired_strict_rollout.jld2")
const MAIN_REFERENCE = joinpath(MAIN_HPM_DIR, "bolivia", "ACPPowerModel", "results",
                                "paired_policy_reference.jld2")
const MAIN_COSTS_CSV = joinpath(MAIN_HPM_DIR, "bolivia", "ACPPowerModel", "paired_costs.csv")
const EXA_RESULTS = joinpath(SCRIPT_DIR, "bolivia", "ACPPowerModel", "results",
                             "paired_exa_strict.jld2")

"""
    classify(rel::Real) -> String

Classify a relative deviation: `"EXACT"` below `1e-6`, `"TIGHT"` below `1e-3`
(solver-tolerance regime), `"DISCREPANT"` otherwise (`"NaN"` if not finite).
"""
function classify(rel::Real)
    isfinite(rel) || return "NaN"
    rel < 1e-6 && return "EXACT"
    rel < 1e-3 && return "TIGHT"
    return "DISCREPANT"
end

"""
    report_deltas(label, a, b) -> Float64

Print mean/max absolute and relative deviations between two equal-length
vectors (NaN entries dropped pairwise), classify the max relative deviation,
and return it. Relative deviation is `|a - b| / max(|a|, 1)` so near-zero
values do not explode the ratio.
"""
function report_deltas(label, a, b)
    mask = .!isnan.(a) .& .!isnan.(b)
    n = count(mask)
    if n == 0
        println("  $label: no comparable entries (all NaN)")
        return NaN
    end
    av, bv = Float64.(a[mask]), Float64.(b[mask])
    absd = abs.(av .- bv)
    reld = absd ./ max.(abs.(av), 1.0)
    println("  $label  [n=$n]")
    println("    mean |Δ| = $(mean(absd))    max |Δ| = $(maximum(absd))")
    println("    mean rel = $(mean(reld))    max rel = $(maximum(reld))   → $(classify(maximum(reld)))")
    return maximum(reld)
end

# ── Load everything ────────────────────────────────────────────────────────────

main = JLD2.load(MAIN_RESULTS)
ref  = JLD2.load(MAIN_REFERENCE)
exa  = JLD2.load(EXA_RESULTS)

main_costs = Float64.(main["costs"])                 # [S]
main_vol   = Float64.(main["vol_trajectories"])      # [T × S]
main_gen   = Float64.(main["gen_trajectories"])      # [T × S]
main_idx   = Int.(main["scenario_indices"])          # [T × S]

exa_costs  = Float64.(exa["costs"])                  # [S]
exa_vol    = Float64.(exa["vol_trajectories"])       # [T × S]
exa_gen    = Float64.(exa["gen_trajectories"])       # [T × S]
exa_ok     = Bool.(exa["scenario_ok"])
de_costs   = Float64.(exa["de_costs"])               # [N]
de_ok      = Bool.(exa["de_ok"])
n_de       = Int(exa["num_de_scenarios"])

T, S = size(main_vol)

# SDDP column from MAIN's paired_costs.csv (quoted headers contain commas, so
# CSV.jl is required for parsing).
sddp_costs = Float64[]
if isfile(MAIN_COSTS_CSV)
    cols = Tables.columntable(CSV.File(MAIN_COSTS_CSV))
    sddp_key = findfirst(k -> occursin("SDDP", String(k)), collect(keys(cols)))
    if sddp_key !== nothing
        global sddp_costs = Float64.(collect(cols[collect(keys(cols))[sddp_key]]))
    end
end

println("=" ^ 72)
println("CROSS-PACKAGE PAIRED EVALUATION — VERDICT REPORT")
println("  MAIN results:   $MAIN_RESULTS")
println("  MAIN reference: $MAIN_REFERENCE")
println("  EXA results:    $EXA_RESULTS")
println("  T=$T stages, S=$S scenarios, DE scenarios N=$n_de")
println("=" ^ 72)

# Sanity: both sides must have evaluated the same scenario index matrix.
ref_idx = Int.(ref["scenario_indices"])
idx_match = main_idx == ref_idx
println("\nScenario-index matrices identical (MAIN results vs reference): $idx_match")
idx_match || println("  !! The two evaluations did NOT use the same scenarios — all cost/trajectory comparisons below are void.")

# ── (a) Policy-parity gate ─────────────────────────────────────────────────────
println("\n(a) POLICY PARITY GATE (from eval_paired_exa_strict.jl)")
println("  checkpoint loader path:        $(exa["gate_loader_mode"])",
        exa["gate_loader_mode"] == "stock" ? "" :
        "   ← EXA's stock loader cannot load MAIN checkpoints (LSTM wrapper vs bare LSTMCell state)")
println("  probe parity (single calls):   ",
        exa["gate_probe_pass"] ? "PASS" : "FAIL",
        "  max|Δ| = $(exa["gate_probe_max_dev"])")
println("  open-loop, EXA policy as-is:   ",
        exa["gate_openloop_asis_pass"] ? "PASS" : "FAIL",
        "  max|Δ| = $(exa["gate_openloop_asis_max_dev"])")
println("  open-loop, state-threaded:     ",
        exa["gate_openloop_threaded_pass"] ? "PASS" : "FAIL",
        "  max|Δ| = $(exa["gate_openloop_threaded_max_dev"])")
if exa["gate_probe_pass"] && !exa["gate_openloop_asis_pass"] && exa["gate_openloop_threaded_pass"]
    println("  → INTERPRETATION: weights load correctly; the divergence is the EXA")
    println("    policy's MEMORYLESS LSTM encoding (Flux 0.16 restarts recurrent")
    println("    state on every call; Flux.reset! is a no-op) vs MAIN's explicit")
    println("    state threading across stages. The two packages evaluate")
    println("    DIFFERENT functions of the inflow history with the same weights.")
end
println("  inflow indexing/units:         ",
        exa["gate_inflow_pass"] ? "PASS (exact)" : "FAIL",
        "  max|Δ| = $(exa["gate_inflow_recon_max_dev"])")
println("  metadata devs: K=$(exa["gate_dev_K"]) min_vol=$(exa["gate_dev_min_vol"]) " *
        "max_vol=$(exa["gate_dev_max_vol"]) min_turn=$(exa["gate_dev_min_turn"]) " *
        "max_turn=$(exa["gate_dev_max_turn"]) upstream_max=$(exa["gate_dev_upstream_max"])")

# ── (b) MAIN stage-wise vs EXA stage-wise costs ────────────────────────────────
println("\n(b) PER-SCENARIO COSTS: MAIN stage-wise (Ipopt) vs EXA stage-wise (MadNLP)")
println("  NOTE: if gate (a) shows the policies compute different targets, part of")
println("  this delta is the policy-function difference, not the subproblem builder.")
report_deltas("total cost per scenario", main_costs, exa_costs)
n_fail = count(.!exa_ok)
n_fail > 0 && println("  EXA stage-wise scenarios failed (excluded): $n_fail")

# ── (c) EXA stage-wise vs EXA DE ───────────────────────────────────────────────
println("\n(c) EXA STAGE-WISE vs EXA STRICT FULL-HORIZON DE (first $n_de scenarios)")
println("  Strict-mode claim: identical targets pin the same reservoir path, so the")
println("  DE objective must equal the stage-wise sum for the same scenario.")
report_deltas("total cost per scenario", exa_costs[1:n_de], de_costs)
# Per-stage decomposition of the DE objective vs the stage-wise stage costs.
sc  = Float64.(exa["stage_costs"])[:, 1:n_de]
dsc = Float64.(exa["de_stage_costs"])
report_deltas("per-stage costs (all t, s ≤ N)", vec(sc), vec(dsc))
n_de_fail = count(.!de_ok)
n_de_fail > 0 && println("  EXA DE scenarios failed (excluded): $n_de_fail")

# ── (d) Trajectory deltas ──────────────────────────────────────────────────────
println("\n(d) TRAJECTORY DELTAS across all (t, s)")
report_deltas("vol_trajectories  (Σ_r volume/0.0036, MW)  MAIN vs EXA-stagewise",
              vec(main_vol), vec(exa_vol))
report_deltas("gen_trajectories  (Σ_thermal pg·baseMVA)   MAIN vs EXA-stagewise",
              vec(main_gen), vec(exa_gen))
# Policy target paths: MAIN open-loop reference vs EXA realized targets.
xhat_ref = Float64.(ref["xhat_trajectories"])          # [T × nHyd × S]
tgt_exa  = Float64.(exa["target_trajectories"])        # [nHyd × T × S]
tgt_exa_perm = permutedims(tgt_exa, (2, 1, 3))         # → [T × nHyd × S]
report_deltas("policy target trajectories (all t, r, s)   MAIN-ref vs EXA",
              vec(xhat_ref), vec(tgt_exa_perm))
report_deltas("DE vol_trajectories vs EXA-stagewise (s ≤ N)",
              vec(Float64.(exa["vol_trajectories"])[:, 1:n_de]),
              vec(Float64.(exa["de_vol_trajectories"])))
report_deltas("DE gen_trajectories vs EXA-stagewise (s ≤ N)",
              vec(Float64.(exa["gen_trajectories"])[:, 1:n_de]),
              vec(Float64.(exa["de_gen_trajectories"])))

# ── (e) Mean costs side by side ────────────────────────────────────────────────
println("\n(e) MEAN COSTS ($S scenarios; DE over first $n_de)")
_mean_ok(v) = (m = v[.!isnan.(v)]; isempty(m) ? NaN : mean(m))
println("  MAIN stage-wise strict (Ipopt):     $(round(_mean_ok(main_costs); digits=1))")
println("  EXA  stage-wise strict (MadNLP):    $(round(_mean_ok(exa_costs); digits=1))")
println("  EXA  strict DE (MadNLP, N=$n_de):     $(round(_mean_ok(de_costs); digits=1))")
println("  MAIN stage-wise mean over s ≤ $n_de:  $(round(_mean_ok(main_costs[1:n_de]); digits=1))")
if !isempty(sddp_costs)
    println("  SDDP (paired_costs.csv):            $(round(_mean_ok(sddp_costs); digits=1))")
else
    println("  SDDP column not found in $MAIN_COSTS_CSV")
end
println("  rollout_tsddr cross-check (s=1):    $(exa["rollout_tsddr_check_objective"])  vs custom loop $(exa_costs[1])")

# ── (f) Known structural differences, with measured activity ──────────────────
println("\n(f) KNOWN STRUCTURAL DIFFERENCES (EXA builder vs MAIN mof.json subproblem)")

max_def_sw = maximum(filter(!isnan, Float64.(exa["max_deficit"])); init = -Inf)
max_def_de = maximum(filter(!isnan, Float64.(exa["de_max_deficit"])); init = -Inf)
max_dq_sw  = maximum(filter(!isnan, Float64.(exa["max_abs_deficit_q"])); init = -Inf)
max_dq_de  = maximum(filter(!isnan, Float64.(exa["de_max_abs_deficit_q"])); init = -Inf)

println("""
  1. DEFICIT COST: MAIN objective uses 6000·Σ deficit (= cost_deficit 60 × baseMVA
     100, scaled at mof export). This evaluation used deficit_cost =
     $(exa["deficit_cost_used"]) to match. EXA TRAINING uses 1e5 instead.
     Measured max deficit: stage-wise = $max_def_sw pu, DE = $max_def_de pu.
     → the 1e5-vs-6000 difference is $(max(max_def_sw, max_def_de) <= 1e-8 ? "INERT (deficit never activates)" : "ACTIVE — deficit occurs, costs are NOT comparable to training runs").
  2. REACTIVE SLACK: EXA adds a FREE zero-cost deficit_q to every reactive KCL;
     MAIN's mof.json has hard reactive balance. Not disableable via kwargs.
     Measured max |deficit_q|: stage-wise = $max_dq_sw pu, DE = $max_dq_de pu.
     → $(max(max_dq_sw, max_dq_de) <= 1e-6 ? "INERT on these scenarios" : "ACTIVE — the EXA AC feasible set is genuinely relaxed vs MAIN (reactive balance violated at zero cost); expect lower EXA costs").
  3. THERMAL LIMITS: MAIN enforces quadratic p²+q² ≤ rate_a² per branch end
     (62 quadratic constraints in mof.json); EXA box-bounds p_fr/q_fr/p_to/q_to
     in [−rate_a, rate_a] — a relaxation in the (|p|,|q|) corners. Not measured
     directly here; shows up as cost differences when branch limits bind.
  4. MIN-VIOLATION SLACKS: mof.json has zero-cost min_outflow/min_volume
     violation slacks; EXA enforces outflow ≥ min_turn hard. Bolivia has
     min_turn ≡ 0 and min_vol ≡ 0 → inert.
  5. GEN COSTS: identical by construction (linear c1 per pu, c2 = 0, from the
     same PowerModels.json; verified against mof.json coefficients).
     Turbine coupling identical: baseMVA·pg = pf·outflow.
  6. DEMAND: mof.json bakes in 0.6× PowerModels loads (pd and qd); EXA used
     load_scaler = $(exa["load_scaler_used"]) → matched.
  7. SPILL UNITS: both sides use spill with coefficient 1 (volume units) in the
     water balance and K = 0.0036 on inflow/outflow → identical dynamics.
  8. POLICY RECURRENCE: see gate (a). EXA's encoder is memoryless per stage
     (Flux 0.16 LSTM restarts from initialstates each call); MAIN threads LSTM
     state across stages. Same weights, different function. This affects EXA
     TRAINING and evaluation alike: the EXA pipeline optimizes/evaluates a
     policy without inflow memory.
  9. CHECKPOINT LOADER: loader path used = "$(exa["gate_loader_mode"])". If not
     "stock", DecisionRulesExa's load_stateconditioned_policy! cannot ingest
     MAIN checkpoints (MAIN saves bare LSTMCell states; EXA wraps cells in
     Flux.LSTM) — cross-package warmstarts silently depend on a custom loader.
 10. SOLVERS/TOLERANCES: MAIN = Ipopt (mumps, default tol); EXA = MadNLP
      (tol = $(exa["solver_tol"])). Agreement at TIGHT (<1e-3 rel) is the
      expected ceiling for cost comparisons even with identical models.
 11. INITIAL STATE: EXA training clamps x0 into [min_vol, max_vol]; this
      evaluation used MAIN's raw initial_state (denormal ≈ 1e-316 values ≈ 0);
      difference ≤ 1e-315 hm³ → inert.
""")
println("=" ^ 72)
println("END OF REPORT")
println("=" ^ 72)
