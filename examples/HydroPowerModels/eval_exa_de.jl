# eval_exa_de.jl
#
# Load the reference results from eval_jump_de.jl (JuMP+MadNLP/DCPPowerModel or ACPPowerModel)
# and reproduce the same problem in ExaModels+MadNLP.
# Prints a side-by-side comparison of objectives and reservoir states.
#
# Key comparison note:
#   JuMP's DCPPowerModel enforces hard KCL (no load shedding variable).
#   ExaModels uses a soft KCL with a deficit slack penalized at DEFICIT_COST.
#   Use DEFICIT_COST >> max thermal generator cost (4244 $/pu) to match JuMP behavior.
#
# Formulation: auto-detected from reference file, or override with FORMULATION below.
#   :dc       — DC linearization (fast, matches DCPPowerModel reference)
#   :ac_polar — Full AC polar OPF (matches ACPPowerModel reference)
#
# Usage (from this directory):
#   julia --project -t auto eval_exa_de.jl

using DecisionRulesExa
using ExaModels
using JLD2
using MadNLP
# GPU packages (only needed when USE_GPU = true below):
using MadNLPGPU, KernelAbstractions, CUDA
using CUDSS_jll, cuDNN

const SCRIPT_DIR = dirname(@__FILE__)
const CASE_DIR   = joinpath(SCRIPT_DIR, "bolivia")
const REF_FILE   = joinpath(CASE_DIR, "jump_de_reference.jld2")

include(joinpath(SCRIPT_DIR, "hydro_power_data.jl"))
include(joinpath(SCRIPT_DIR, "hydro_power_exa.jl"))

# ── Solve-status check (MadNLP 0.8.x) ────────────────────────────────────────
solve_succeeded(r) = r.status == MadNLP.SOLVE_SUCCEEDED ||
                     r.status == MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL

# ── Load reference ────────────────────────────────────────────────────────────

isfile(REF_FILE) || error("Reference file not found: $REF_FILE\n" *
    "Run eval_jump_de.jl first (in DecisionRules.jl).")

ref = load(REF_FILE)
obj_ref      = ref["objective"]
res_ref      = ref["reservoir"]          # nHyd × (T+1)
x0_ref       = ref["initial_state"]      # length nHyd
inflows_flat = ref["inflows_flat"]       # length T*nHyd, stage-major
targets_flat = ref["targets_flat"]       # length T*nHyd, stage-major
max_vol_ref  = ref["max_volume"]
T            = ref["num_stages"]
nHyd         = ref["nHyd"]
formulation  = get(ref, "formulation", "DCPPowerModel")

@info "Reference loaded: obj=$(round(obj_ref; digits=4)),  T=$T,  nHyd=$nHyd,  formulation=$formulation"

# ── Load ExaModels data ───────────────────────────────────────────────────────

const PM_FILE     = joinpath(CASE_DIR, "PowerModels.json")
const HYDRO_FILE  = joinpath(CASE_DIR, "hydro.json")
const INFLOW_FILE = joinpath(CASE_DIR, "inflows.csv")
const DEMAND_FILE = joinpath(CASE_DIR, "_demand.csv")

# Load-shedding cost: must be >> max thermal generator cost (Bolivia: ~4244 $/pu)
# to ensure the solver never prefers deficit over thermal dispatch (matching
# JuMP's DCPPowerModel which has no deficit variable at all).
const DEFICIT_COST = 1e5

# target_penalty = :auto  →  ρ = 2 × max_gen_cost, matching JuMP's penalty_l2 = :auto
# (ExaModels uses (ρ/2)·δ², so ρ/2 = max_gen_cost = the same effective multiplier)
const TARGET_PEN_ARG = :auto

# ── Detect formulation from reference ─────────────────────────────────────────
# Override here if needed: const FORMULATION = :dc  or  :ac_polar
const FORMULATION = if formulation == "ACPPowerModel"
    @info "Detected AC formulation from reference → using :ac_polar"
    :ac_polar
else
    @info "Detected DC formulation from reference → using :dc"
    :dc
end

@info "Loading ExaModels power data..."
power_data = load_power_data(PM_FILE)
hydro_data = load_hydro_data(HYDRO_FILE, INFLOW_FILE, power_data; num_stages = T * 10)

@assert hydro_data.nHyd == nHyd "nHyd mismatch: $(hydro_data.nHyd) vs $nHyd"

demand_mat = isfile(DEMAND_FILE) ? load_demand(DEMAND_FILE, power_data; T = T) : nothing

TARGET_PEN = auto_target_penalty(power_data, hydro_data)
@info "Auto target penalty: ρ=$(round(TARGET_PEN; digits=2))  (= 2 × max_obj_cost = 2 × $(round(TARGET_PEN/2; digits=2)))"

@info "Building $(T)-stage ExaModels DE  (formulation=$FORMULATION, deficit_cost=$DEFICIT_COST, target_penalty=:auto → $TARGET_PEN)..."
prob = build_hydro_de(power_data, hydro_data, T;
    backend        = CUDA.CUDABackend(),
    float_type     = Float64,
    formulation    = FORMULATION,
    target_penalty = TARGET_PEN_ARG,
    deficit_cost   = DEFICIT_COST,
    demand_matrix  = demand_mat,
)

# ── Set problem parameters (same as reference) ────────────────────────────────

ExaModels.set_parameter!(prob.core, prob.p_x0, x0_ref)
set_inflows!(prob, inflows_flat)
ExaModels.set_parameter!(prob.core, prob.p_target, targets_flat)

# ── Solve ─────────────────────────────────────────────────────────────────────

@info "Solving ExaModels DE..."
result = MadNLP.madnlp(prob.model; print_level = MadNLP.ERROR)
@info "  Status: $(result.status)   Objective: $(round(result.objective; digits=4))"

solve_succeeded(result) || @warn "Solve did not fully converge ($(result.status))"

# ── Extract solution ──────────────────────────────────────────────────────────

sol = hydro_solution(prob, result)
res_exa = Array(sol.reservoir)   # nHyd × (T+1), brought to CPU for comparison

# ── Objective decomposition ───────────────────────────────────────────────────
# Decompose into: generator cost + deficit cost + target penalty
# (These should sum to result.objective up to solver tolerance.)

pg_cpu = Array(sol.pg)   # bring to CPU for scalar indexing in generator loop
gen_cost = sum(
    g.cost2 * pg_cpu[g_pos, t]^2 + g.cost1 * pg_cpu[g_pos, t]
    for (g_pos, g) in enumerate(power_data.gens), t in 1:T
)

def_cost_total = DEFICIT_COST * sum(sol.deficit)
tgt_pen_total  = (TARGET_PEN / 2) * sum(sol.delta .^ 2)

total_deficit_pu = sum(sol.deficit)   # total active load shed (pu·stage)

# ── Comparison table ──────────────────────────────────────────────────────────

exa_form_str = FORMULATION === :ac_polar ? "ACPolarOPF" : "DCPPowerModel"

println("\n" * "="^60)
println("  OBJECTIVE COMPARISON")
println("="^60)
println("  JuMP+MadNLP     ($(formulation)):  ", round(obj_ref; digits=4))
println("  ExaModels+MadNLP ($(exa_form_str)):  ", round(result.objective; digits=4))
pct_diff = abs(result.objective - obj_ref) / max(abs(obj_ref), 1.0) * 100
println("  Relative difference:            ", round(pct_diff; digits=2), " %")

println("\n" * "="^60)
println("  ExaModels OBJECTIVE BREAKDOWN")
println("="^60)
println("  Generator cost:   ", round(gen_cost;         digits=4))
println("  Deficit cost:     ", round(def_cost_total;   digits=4),
        "  (total shed = ", round(total_deficit_pu; digits=6), " pu·stage)")
println("  Target penalty:   ", round(tgt_pen_total;    digits=4))
println("  Sum of parts:     ", round(gen_cost + def_cost_total + tgt_pen_total; digits=4))
println("  MadNLP objective: ", round(result.objective; digits=4))

println("\n  NOTE: Both JuMP (penalty_l2=:auto) and ExaModels (target_penalty=:auto)")
println("  use effective L2 coefficient = max_gen_cost = ", round(TARGET_PEN/2; digits=2))
println("  ExaModels: (ρ/2)·δ² with ρ=", round(TARGET_PEN; digits=2), ".  JuMP: penalty_l2·δ².")
println("  ExaModels gen cost only:  ", round(gen_cost; digits=4))

println("\n" * "="^60)
println("  RESERVOIR STATES — end of each stage (per hydro unit)")
println("="^60)
println("  Stage:  ", join(lpad.(0:T, 9)))
for r in 1:nHyd
    println("  Hydro $r (ref): ", join(lpad.(round.(res_ref[r, :]; digits=1), 9)))
    println("  Hydro $r (exa): ", join(lpad.(round.(res_exa[r, :]; digits=1), 9)))
    max_err = maximum(abs.(res_ref[r, :] .- res_exa[r, :]))
    println("  Hydro $r  |err|: ", round(max_err; digits=2), "\n")
end

total_err = maximum(abs.(res_ref .- res_exa))
println("="^60)
println("  Max |reservoir error| across all units/stages: ", round(total_err; digits=2))
println("="^60)
