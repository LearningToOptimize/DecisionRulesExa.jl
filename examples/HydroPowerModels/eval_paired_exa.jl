#!/usr/bin/env julia

# Paired evaluation of a trained strict TS-DDR checkpoint on the frozen Bolivia
# protocol, with the per-stage physical decisions recorded.
#
# This is the evaluator that produced the published TS-DDR distribution. It does
# NOT reimplement the rollout: it calls the production
# `DecisionRulesExa.rollout_tsddr` with the trainer's exact stage configuration
# and attaches a `stage_recorder`, which reads the solution the rollout has
# already computed. The per-column costs therefore average to the same panel mean
# the trainer's own evaluation reports for the same checkpoint.
#
# Four CSVs are always written, in the same schema as the SDDP evaluator's
# `DR_PHYSICAL_AUDIT=1` output so the two can be differenced column by column and
# stage by stage:
#
#   <label>_scenario.csv   one row per protocol column: cost, shedding totals,
#                          solve provenance
#   <label>_stage.csv      one row per (column, stage): stage totals and status
#   <label>_reservoir.csv  one row per (column, stage, reservoir)
#   <label>_deficit.csv    one row per (column, stage, bus) above the shedding
#                          tolerance
#
# With `DR_SOLUTION_DUMP=1` it additionally writes the full primal solution of
# every stage in the shared long format of `hydro_solution_schema.jl`, plus the
# decision trace that reproduces it:
#
#   <label>_solution.csv   scenario,stage,class,index,value — every physical
#                          variable of every stage
#   <label>_trace.csv      scenario,stage,reservoir,state_in,target,inflow
#
# The trace is what lets a different engine replay this policy's decisions
# exactly: `DecisionRules.jl/examples/HydroPowerModels/verify_full_solution_parity.jl`
# reads it, replays it through the serialized JuMP/MathOptFormat stage model, and
# differences the two solutions variable by variable.
#
# ── PHYSICAL load shedding ────────────────────────────────────────────────────
# The only load-shedding quantity is the per-bus active-balance slack
# `deficit[b]` (pu), priced at 6000 USD/pu·stage. Reservoir-target slack is a
# different thing entirely and is structurally absent in strict mode.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#
#   julia --project=. eval_paired_exa.jl
#
# Environment:
#   DR_EVAL_CKPT        checkpoint .jld2 (required)
#   DR_EVAL_LABEL       output filename stem (default "tsddr")
#   DR_EVAL_COLS        comma list of protocol columns
#                       (default: the ten-column screening panel)
#   DR_EVAL_COL_FIRST /
#   DR_EVAL_COL_LAST    scalar shard range; preferred over DR_EVAL_COLS under
#                       `sbatch --export`, which splits values on commas
#   DR_EVAL_T           rollout horizon (default 96, the reported window)
#   DR_EVAL_OUT         output directory (default bolivia/ACPPowerModel/results)
#   DR_ENCODER_LAYERS   LSTM encoder widths, must match the checkpoint
#                       (default "128,128")
#   DR_HEAD_LAYERS      state-conditioned head widths (default "256,256")
#   DR_SOLUTION_DUMP    "1" to also write the full-solution and trace CSVs

using DecisionRulesExa
using StableRNGs
using ExaModels
using Flux
using Statistics, Random, Printf, LinearAlgebra, Dates
using JLD2
using MadNLP
using MadNLPGPU, KernelAbstractions, CUDA
using CUDSS, CUDSS_jll, cuDNN

const EXA_DIR = @__DIR__
include(joinpath(EXA_DIR, "hydro_training_utils.jl"))
include(joinpath(EXA_DIR, "hydro_power_data.jl"))
include(joinpath(EXA_DIR, "hydro_power_exa.jl"))
include(joinpath(EXA_DIR, "hydro_reachable_policy.jl"))
include(joinpath(EXA_DIR, "generate_canonical_case_artifacts.jl"))
include(joinpath(EXA_DIR, "hydro_solution_schema.jl"))
using .HydroCanonicalCase
using .HydroSolutionSchema

"""
    write_csv(path, header, rows) -> Nothing

Write `rows` (a vector of tuples) as a CSV with column names `header`.

DataFrames/CSV are not dependencies of this examples project, and adding them
for an offline dump would change the environment a training run resolves; the
schema here is flat, so it is emitted directly. Floats go through `repr`, the
shortest string that round-trips, so costs can be compared against the SDDP
audit's without a formatting-induced difference.
"""
function write_csv(path::AbstractString, header, rows)
    open(path, "w") do io
        println(io, join(header, ","))
        for r in rows
            println(io, join((x isa AbstractFloat ? repr(x) : string(x) for x in r), ","))
        end
    end
    return nothing
end

"""
    env_layers(name, default) -> Vector{Int}

Parse a comma-separated layer-width list from the environment.
"""
function env_layers(name, default)
    raw = strip(get(ENV, name, default))
    isempty(raw) && return Int[]
    return [parse(Int, strip(x)) for x in split(raw, ',') if !isempty(strip(x))]
end

# ── Configuration ─────────────────────────────────────────────────────────────
# Every value below is a property of the FROZEN case contract, read from the
# contract module rather than restated, so this evaluator cannot drift away from
# what `generate_canonical_case_artifacts.jl --verify` asserts.
const CASE_DIR    = joinpath(EXA_DIR, "bolivia")
const PM_FILE     = joinpath(CASE_DIR, "PowerModels.json")
const HYDRO_FILE  = joinpath(CASE_DIR, "hydro.json")
const INFLOW_FILE = joinpath(CASE_DIR, "inflows.csv")

const FORMULATION   = :ac_polar                    # true ACP
const DEFICIT_COST  = ACTIVE_DEFICIT_COST          # 6000 = 60 USD/MWh x 100 MVA
const LOAD_SCALER   = ACTIVE_LOAD_FACTOR           # 0.6
const QD_SCALER     = REACTIVE_LOAD_FACTOR         # 0.6
const REACTIVE_DEFICIT_COST = Inf                  # hard reactive balance
const ACTIVATION    = stretchedsigmoid
const SOLVER_KWARGS = (print_level = MadNLP.ERROR, tol = 1e-6, max_iter = 9000)
const AUDIT_TOL     = 1e-6                         # pu — the shedding decision gate

const ENCODER_LAYERS = env_layers("DR_ENCODER_LAYERS", "128,128")
const HEAD_LAYERS    = env_layers("DR_HEAD_LAYERS", "256,256")
const T_ROLLOUT      = parse(Int, get(ENV, "DR_EVAL_T", string(REPORTING_STAGES)))
const SOLUTION_DUMP  = get(ENV, "DR_SOLUTION_DUMP", "0") == "1"

const PANEL_COLUMNS = [2, 39, 81, 119, 130, 156, 200, 206, 378, 493]
const COLS = let lo = strip(get(ENV, "DR_EVAL_COL_FIRST", "")),
                 hi = strip(get(ENV, "DR_EVAL_COL_LAST", ""))
    if !isempty(lo) && !isempty(hi)
        collect(parse(Int, lo):parse(Int, hi))
    else
        raw = strip(get(ENV, "DR_EVAL_COLS", join(PANEL_COLUMNS, ",")))
        [parse(Int, strip(c)) for c in split(raw, ",") if !isempty(strip(c))]
    end
end

const CKPT = let c = strip(get(ENV, "DR_EVAL_CKPT", ""))
    isempty(c) && error("DR_EVAL_CKPT must name the checkpoint .jld2 to evaluate")
    isfile(c) || error("checkpoint not found: $c")
    c
end
const LABEL = let l = strip(get(ENV, "DR_EVAL_LABEL", "")); isempty(l) ? "tsddr" : l end
const OUT_DIR = strip(get(ENV, "DR_EVAL_OUT",
    joinpath(CASE_DIR, "ACPPowerModel", "results")))
isdir(OUT_DIR) || mkpath(OUT_DIR)

println("="^78)
println("TS-DDR PAIRED EVALUATION   ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
println("="^78)
@info "config" LABEL CKPT T_ROLLOUT n_columns=length(COLS) OUT_DIR SOLUTION_DUMP

# ── Case data ─────────────────────────────────────────────────────────────────
verify_inputs(CASE_DIR)
verify_index_convention(CASE_DIR, JSON.parsefile)
power_data = load_power_data(PM_FILE)
hydro_data = load_hydro_data(HYDRO_FILE, INFLOW_FILE, power_data;
                             num_stages = PROTOCOL_STAGES * 10)
const nHyd = hydro_data.nHyd
const baseMVA = power_data.baseMVA
abs(hydro_data.K - HYDRO_CONVERSION_K) < 1e-12 ||
    error("K = $(hydro_data.K) != $HYDRO_CONVERSION_K — wrong case data")
@info "case" nBus=power_data.nBus nGen=power_data.nGen nBranch=power_data.nBranch nHyd K=hydro_data.K baseMVA

# Thermal generators = every generator not driven by a hydro unit.
const HYDRO_GEN_IDX = Set(h.gen_pos for h in hydro_data.units)
const THERMAL_GEN_IDX = [g for g in 1:power_data.nGen if !(g in HYDRO_GEN_IDX)]

const reactive_mat = QD_SCALER == LOAD_SCALER ? nothing :
    repeat(reshape((QD_SCALER / LOAD_SCALER) .* power_data.default_bus_reactive_demand, 1, :),
           T_ROLLOUT, 1)

"""
    protocol_eval_scenario(hydro_data, T, indices, s) -> Vector{Float64}

Flat `T x nHyd` inflow vector for paired-protocol column `s`.

Identical to the trainer's own construction, so the uncertainty path used here,
during training, and by the SDDP `Historical` sampling scheme is one path and not
three that happen to agree.
"""
function protocol_eval_scenario(hydro_data::HydroData, T::Int, indices, s::Int)
    nH = hydro_data.nHyd
    w = Vector{Float64}(undef, T * nH)
    for t in 1:T
        t_row = mod1(t, hydro_data.nStagesSample)   # cyclic raw-row mapping
        j = indices[t, s]                           # inflow scenario at stage t
        for r in 1:nH
            w[(t-1)*nH + r] = hydro_data.scenario_inflows[r][t_row, j]
        end
    end
    return w
end

const protocol_indices = rand(
    StableRNG(INFLOW_PROTOCOL_SEED), 1:hydro_data.nScenarios,
    PROTOCOL_STAGES, PROTOCOL_SCENARIOS,
)
T_ROLLOUT <= PROTOCOL_STAGES || error("DR_EVAL_T exceeds the protocol's $PROTOCOL_STAGES stages")
all(1 .<= COLS .<= PROTOCOL_SCENARIOS) || error("column ids must lie in 1:$PROTOCOL_SCENARIOS")

# ── Policy + checkpoint ───────────────────────────────────────────────────────
Random.seed!(42)
policy = hydro_reachable_policy(hydro_data, ENCODER_LAYERS;
                                activation      = ACTIVATION,
                                encoder_type    = Flux.LSTM,
                                combiner_layers = HEAD_LAYERS,
                                n_context       = 0,
                                n_extra_uncertainty = 0)
load_stateconditioned_policy!(policy, JLD2.load(CKPT, "model_state"))
# The rollout's state and uncertainty live on the device, so the weights must
# too; `Flux.reset!` after the move re-derives the recurrent state there.
policy = CUDA.cu(policy)
Flux.reset!(policy)
@info "checkpoint loaded (policy on GPU)" CKPT sha256=sha256_file(CKPT)

# ── Stage problem: the trainer's 1-stage strict deterministic equivalent ──────
const backend = CUDA.CUDABackend()
rollout_prob = build_hydro_de(power_data, hydro_data, 1;
    backend        = backend,
    float_type     = Float64,
    formulation    = FORMULATION,
    target_penalty = :auto,
    deficit_cost   = DEFICIT_COST,
    demand_matrix  = nothing,
    reactive_demand_matrix = reactive_mat === nothing ? nothing : reactive_mat[1:1, :],
    load_scaler    = LOAD_SCALER,
    strict_targets = true,
    reactive_deficit_cost = REACTIVE_DEFICIT_COST,
    demand_spread  = nothing,
)

x0_init = CUDA.cu(Float32.([
    clamp(hydro_data.initial_volumes[r], hydro_data.units[r].min_vol, hydro_data.units[r].max_vol)
    for r in 1:nHyd
]))
const _min_vols_dev = CUDA.cu(Float64.([h.min_vol for h in hydro_data.units]))
const _max_vols_dev = CUDA.cu(Float64.([h.max_vol for h in hydro_data.units]))

function set_hydro_rollout_stage!(stage_prob, state_in, wt, target, stage)
    ExaModels.set_parameter!(stage_prob.core, stage_prob.p_x0, state_in)
    ExaModels.set_parameter!(stage_prob.core, stage_prob.p_inflow, wt)
    ExaModels.set_parameter!(stage_prob.core, stage_prob.p_target, target)
    prepare_solve!(stage_prob, state_in, wt, target)
    return stage_prob
end
hydro_realized_state(stage_prob, result) = hydro_solution(stage_prob, result).reservoir[:, end]

# ── Recording ─────────────────────────────────────────────────────────────────
const DEF_HEADER = ("scenario", "stage", "bus", "deficit_pu", "deficit_MW")
const STAGE_HEADER = (
    "scenario", "stage", "deficit_pu", "deficit_MW",
    "max_bus_deficit_pu", "argmax_bus", "n_buses_shedding",
    "thermal_MW", "hydro_MW", "inflow", "outflow", "spill", "storage",
    "cost_nondeficit", "cost_deficit", "stage_objective", "cum_objective", "status",
)
const RES_HEADER = ("scenario", "stage", "reservoir",
                    "storage", "target", "inflow", "outflow", "spill")
const SCEN_HEADER = (
    "scenario", "cost", "deficit_pu", "deficit_MW",
    "max_bus_stage_deficit_pu", "max_bus_stage_deficit_MW",
    "n_stages_with_deficit", "n_bus_stage_with_deficit",
    "cost_deficit", "cost_nondeficit", "thermal_MW", "spill", "all_stages_solved",
    "attempts", "status",
)
const TRACE_HEADER = ("scenario", "stage", "reservoir", "state_in", "target", "inflow")

def_rows   = Vector{Tuple}()
stage_rows = Vector{Tuple}()
res_rows   = Vector{Tuple}()
trace_rows = Vector{Tuple}()
solution_writer = SOLUTION_DUMP ?
    SolutionWriter(joinpath(OUT_DIR, "$(LABEL)_solution.csv")) : nothing
max_reactive_slack = Ref(0.0)

"""
    make_recorder(scenario, w_path, cum) -> Function

Build the per-stage read-out closure for one column.

`hydro_solution` unpacks the flat solver solution into named blocks, so nothing
here re-derives a physical quantity: every value written is the one the rollout
solved for. The stage objective is `result.objective`; its load-shedding
component is `DEFICIT_COST * sum(deficit)` by construction of the builder's
objective, and `cost_nondeficit` is the remainder (generation plus any spill or
minimum-violation terms), NOT claimed to be generation cost alone.

When `DR_SOLUTION_DUMP=1` the same closure emits the full primal solution and
the decision trace. The reactive slack is checked rather than written: in hard
mode it is structurally absent, and asserting that keeps a silently relaxed
reactive balance from passing as a match against a JuMP model that has no such
variable.
"""
function make_recorder(scenario::Int, w_path::Vector{Float64}, cum::Base.RefValue{Float64})
    return function (stage_prob, result, stage, target, state_in)
        sol = hydro_solution(stage_prob, result)
        d   = Array(vec(sol.deficit))
        pg  = Array(vec(sol.pg))
        res_in  = Array(sol.reservoir)[:, 1]
        res_out = Array(sol.reservoir)[:, end]
        out = Array(vec(sol.outflow))
        spl = Array(vec(sol.spill))
        wt  = w_path[(stage-1)*nHyd+1 : stage*nHyd]
        tgt = Array(target)

        for (b, v) in enumerate(d)
            v > AUDIT_TOL && push!(def_rows, (scenario, stage, b, v, v * baseMVA))
        end
        dmax, dargmax = findmax(d)
        th = sum(pg[g] for g in THERMAL_GEN_IDX; init = 0.0) * baseMVA
        hy = sum(pg[g] for g in 1:power_data.nGen if g in HYDRO_GEN_IDX; init = 0.0) * baseMVA
        cost_def = DEFICIT_COST * sum(d)
        cum[] += result.objective
        push!(stage_rows, (
            scenario, stage,
            sum(d), sum(d) * baseMVA,
            dmax, dargmax, count(>(AUDIT_TOL), d),
            th, hy,
            sum(wt), sum(out), sum(spl), sum(res_out),
            result.objective - cost_def, cost_def, result.objective,
            cum[], string(result.status),
        ))
        for r in 1:nHyd
            push!(res_rows, (scenario, stage, r, res_out[r], tgt[r], wt[r], out[r], spl[r]))
        end

        if solution_writer !== nothing
            max_reactive_slack[] =
                max(max_reactive_slack[], maximum(abs, Array(sol.deficit_q); init = 0.0))
            for r in 1:nHyd
                push!(trace_rows, (scenario, stage, r, res_in[r], tgt[r], wt[r]))
            end
            record_vector!(solution_writer, scenario, stage, "va", Array(vec(sol.va)))
            record_vector!(solution_writer, scenario, stage, "vm", Array(vec(sol.vm)))
            record_vector!(solution_writer, scenario, stage, "pg", pg)
            record_vector!(solution_writer, scenario, stage, "qg", Array(vec(sol.qg)))
            record_vector!(solution_writer, scenario, stage, "p_fr", Array(vec(sol.p_fr)))
            record_vector!(solution_writer, scenario, stage, "p_to", Array(vec(sol.p_to)))
            record_vector!(solution_writer, scenario, stage, "q_fr", Array(vec(sol.q_fr)))
            record_vector!(solution_writer, scenario, stage, "q_to", Array(vec(sol.q_to)))
            record_vector!(solution_writer, scenario, stage, "deficit", d)
            record_vector!(solution_writer, scenario, stage, "reservoir_in", res_in)
            record_vector!(solution_writer, scenario, stage, "reservoir_out", res_out)
            record_vector!(solution_writer, scenario, stage, "target", tgt)
            record_vector!(solution_writer, scenario, stage, "inflow", wt)
            record_vector!(solution_writer, scenario, stage, "outflow", out)
            record_vector!(solution_writer, scenario, stage, "spill", spl)
            record_vector!(
                solution_writer, scenario, stage, "target_multiplier",
                Float64.(vec(Array(target_multipliers(stage_prob, result)))),
            )
            record_scalar!(solution_writer, scenario, stage, "stage_objective", result.objective)
            record_scalar!(solution_writer, scenario, stage, "cum_objective", cum[])
        end
        return nothing
    end
end

# ── Roll out every requested column ───────────────────────────────────────────
# Wrapped in `let`: a bare top-level `for` would give every variable it assigns
# its own scope, silently making loop-carried state loop-local.
scen_rows = Vector{Tuple}()
let
    for s in COLS
        w_path = protocol_eval_scenario(hydro_data, T_ROLLOUT, protocol_indices, s)
        cum = Ref(0.0)
        t0 = time()
        roll() = DecisionRulesExa.rollout_tsddr(
            policy, x0_init, rollout_prob, w_path;
            horizon = T_ROLLOUT,
            n_uncertainty = nHyd,
            set_stage_parameters! = set_hydro_rollout_stage!,
            realized_state = hydro_realized_state,
            objective_no_target_penalty = (prob, res) -> res.objective,
            madnlp_kwargs = SOLVER_KWARGS,
            warmstart = false,
            policy_state = :realized,
            state_bounds = (_min_vols_dev, _max_vols_dev),
            retry_on_failure = true,     # per-STAGE cold-solver recovery
            stage_recorder = make_recorder(s, w_path, cum),
        )
        result = roll()
        attempts = 1
        if result === nothing
            # `rollout_tsddr` already retries each failed STAGE from a fresh cold
            # solver, so a second full pass is an independent attempt from clean
            # solver state. Partial rows from the failed attempt are discarded
            # first so the retry cannot double-count; the column is NEVER dropped
            # or renumbered.
            @warn "rollout FAILED on attempt 1 — retrying from fresh solver state" scenario=s
            for rows in (stage_rows, res_rows, def_rows, trace_rows)
                filter!(r -> r[1] != s, rows)
            end
            cum[] = 0.0
            result = roll()
            attempts = 2
        end
        if result === nothing
            # Recorded as UNSOLVED with its id intact, so the merge step aborts
            # instead of averaging over a partial set.
            @error "rollout UNSOLVED after recovery — recorded, merge must abort" scenario=s
            for rows in (stage_rows, res_rows, def_rows, trace_rows)
                filter!(r -> r[1] != s, rows)
            end
            push!(scen_rows, (s, NaN, NaN, NaN, NaN, NaN, -1, -1,
                              NaN, NaN, NaN, NaN, false, attempts, "UNSOLVED"))
            continue
        end
        g = [r for r in stage_rows if r[1] == s]
        tot_def = sum(r[3] for r in g)
        max_def = maximum(r[5] for r in g)
        n_shed  = count(r -> r[3] > AUDIT_TOL, g)
        push!(scen_rows, (
            s, result.objective,
            tot_def, tot_def * baseMVA,
            max_def, max_def * baseMVA,
            n_shed, sum(r[7] for r in g),
            sum(r[15] for r in g), sum(r[14] for r in g),
            sum(r[8] for r in g), sum(r[12] for r in g),
            all(r -> r[18] in ("SOLVE_SUCCEEDED", "SOLVED_TO_ACCEPTABLE_LEVEL"), g),
            attempts, attempts == 1 ? "SOLVED" : "SOLVED_ON_RETRY",
        ))
        @printf("  col %3d: cost=%.5f  deficit=%.3e pu  stages_shedding=%d/%d  %.1fs\n",
                s, result.objective, tot_def, n_shed, T_ROLLOUT, time() - t0)
        flush(stdout)
    end
end

# ── Save ──────────────────────────────────────────────────────────────────────
write_csv(joinpath(OUT_DIR, "$(LABEL)_deficit.csv"),   DEF_HEADER,   def_rows)
write_csv(joinpath(OUT_DIR, "$(LABEL)_stage.csv"),     STAGE_HEADER, stage_rows)
write_csv(joinpath(OUT_DIR, "$(LABEL)_reservoir.csv"), RES_HEADER,   res_rows)
write_csv(joinpath(OUT_DIR, "$(LABEL)_scenario.csv"),  SCEN_HEADER,  scen_rows)
if solution_writer !== nothing
    close(solution_writer)
    write_csv(joinpath(OUT_DIR, "$(LABEL)_trace.csv"), TRACE_HEADER, trace_rows)
    max_reactive_slack[] == 0.0 || error(
        "reactive slack is not structurally zero (max $(max_reactive_slack[])); " *
        "the reactive balance is not hard and this solution is not comparable " *
        "to the JuMP stage model, which has no reactive slack variable",
    )
end

solved = [r for r in scen_rows if r[13] === true]
println("\n" * "="^78)
println("PER-COLUMN TS-DDR COSTS  ($(LABEL), $(T_ROLLOUT) stages)")
println("="^78)
for r in scen_rows
    @printf("  %3d  %14.5f   deficit %.3e pu   shed_stages %d   %s\n",
            r[1], r[2], r[3], r[7], r[15])
end
if !isempty(solved)
    @printf("  MEAN over %d solved column(s) = %.5f\n",
            length(solved), mean(Float64[r[2] for r in solved]))
end
@printf("  solved %d / %d\n", length(solved), length(scen_rows))
isempty(stage_rows) || @printf(
    "  RAW MAX per-bus deficit over all (col,stage,bus) = %.6e pu\n",
    maximum(Float64[r[5] for r in stage_rows]))
println("  CSVs: $OUT_DIR/$(LABEL)_{scenario,stage,reservoir,deficit}.csv")
solution_writer === nothing ||
    println("  full solution + trace: $OUT_DIR/$(LABEL)_{solution,trace}.csv")
println("="^78)
