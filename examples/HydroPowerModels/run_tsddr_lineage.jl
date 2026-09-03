#!/usr/bin/env julia

# Run a declared multi-stage TS-DDR training lineage, end to end.
#
# A "lineage" is an ordered list of training stages, each a SEPARATE trainer
# process that starts from the previous stage's SELECTED checkpoint. Running
# each stage as its own process is not an implementation detail: it is what
# makes a stage boundary a real restart — the optimizer state, the cosine
# learning-rate phase, the warm-up counter and the evaluation bookkeeping all
# begin again, which is the behaviour the published schedule was tuned around.
#
# This driver is ORCHESTRATION ONLY. It knows nothing about hydro, about
# Bolivia, or about what a good objective value is; every number lives in the
# lineage configuration file. The same driver runs the battery or inventory
# lineage from a different configuration.
#
# ── What it guarantees ────────────────────────────────────────────────────────
#
# * The first stage starts from a deterministic random initialisation (no
#   `DR_PRETRAINED_MODEL`), so the lineage is from-scratch by construction.
# * Every later stage chains from the previous stage's SELECTED checkpoint —
#   the one the trainer saved because a COMPLETE panel evaluation improved on
#   the parent. It refuses a `_latest` file outright: `_latest` is a periodic
#   snapshot, not a selection, and chaining from it would silently publish a
#   policy no evaluation ever accepted.
# * Every parent is hashed before use and the hash is compared with the record
#   written when that parent was produced, so a chain can never be built on a
#   checkpoint that has been overwritten or swapped.
# * The trainer additionally re-evaluates the parent and refuses to train if the
#   value does not reproduce (`DR_SEED_BEST` + `DR_PARENT_REPRO_TOL`).
# * A stage that produces no selectable checkpoint STOPS THE LINEAGE. It does
#   not fall through to `_latest`, and it does not silently continue from the
#   grandparent.
# * A stage whose final evaluation was incomplete, or which shed load beyond the
#   configured tolerance, is rejected for the same reason.
# * Re-running the driver RESUMES: a stage whose record exists and whose
#   checkpoint still hashes correctly is skipped, so an interrupted lineage
#   continues rather than restarting.
#
# ── What it records ───────────────────────────────────────────────────────────
#
# One JSON per stage plus a lineage-level ledger, holding ancestry, the full
# resolved environment, the seed, checkpoint hashes, the update count and both
# the wall time of the stage process and the trainer's own training-loop
# seconds. That is the provenance a published time-to-policy figure needs.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#
#   julia --project run_tsddr_lineage.jl                      # full lineage
#   julia --project run_tsddr_lineage.jl --config=other.json
#   julia --project run_tsddr_lineage.jl --out=/path/to/ledger
#   julia --project run_tsddr_lineage.jl --stages=C1,C3       # a subset, resuming
#   julia --project run_tsddr_lineage.jl --dry-run            # print, run nothing
#
# Neither W&B nor a workload manager is required. `DR_ENABLE_WANDB` defaults to
# whatever the configuration or the ambient environment says; nothing here reads
# a SLURM variable.

using JSON
using SHA
using Dates
using Printf

const HYDRO_DIR = @__DIR__

"""
    option(prefix, default=nothing) -> Union{Nothing,String}

Value of the first `--key=value` argument whose key matches `prefix`.
"""
function option(prefix, default = nothing)
    index = findfirst(value -> startswith(value, prefix), ARGS)
    return isnothing(index) ? default : String(split(ARGS[index], '='; limit = 2)[2])
end

sha256_file(path) = bytes2hex(open(SHA.sha256, path))

"""
    stage_record_path(out_dir, tag) -> String

Where a completed stage's record lives.
"""
stage_record_path(out_dir, tag) = joinpath(out_dir, "stage_$(tag).json")

"""
    completed_stage(out_dir, tag) -> Union{Nothing,Dict}

The record of a previously completed stage, or `nothing`.

A record only counts as completed if its checkpoint still exists AND still
hashes to the value recorded when it was produced. A stale record pointing at a
replaced file is treated as absent, so a resumed lineage re-runs the stage
rather than chaining from a file that is no longer the one that was selected.
"""
function completed_stage(out_dir, tag)
    path = stage_record_path(out_dir, tag)
    isfile(path) || return nothing
    record = JSON.parsefile(path)
    checkpoint = get(record, "checkpoint", "")
    (checkpoint isa String && isfile(checkpoint)) || return nothing
    sha256_file(checkpoint) == get(record, "checkpoint_sha256", "") || return nothing
    return record
end

"""
    validate_selection(summary, tag) -> Nothing

Require that a finished stage actually SELECTED a checkpoint on a complete,
non-shedding evaluation.

Three separate ways a stage can finish without a usable result, each of which
has happened at least once and each of which is fatal here rather than a
warning:

* no valid save — the stage never beat its parent on a complete evaluation, so
  there is nothing to chain from;
* an incomplete evaluation — a mean over fewer scenarios than the schedule asked
  for is not comparable to one over all of them, at ANY tolerance;
* load shedding above the configured tolerance — a policy that shed load is not
  a candidate regardless of its cost.
"""
function validate_selection(summary, tag)
    get(summary, "n_valid_saves", 0) > 0 || error(
        "stage $tag produced no selectable checkpoint (stop reason: " *
        "$(get(summary, "stop_reason", "unknown"))). The lineage stops here; it " *
        "does not fall back to a _latest snapshot.",
    )
    requested = get(summary, "last_eval_n_requested", nothing)
    n_ok = get(summary, "last_eval_n_ok", nothing)
    if requested !== nothing && n_ok !== nothing && n_ok != requested
        error("stage $tag finished on an INCOMPLETE evaluation ($n_ok of $requested " *
              "scenarios); failed ids $(get(summary, "last_eval_failed_scenarios", []))")
    end
    deficit = get(summary, "max_bus_deficit_pu_last_eval", 0.0)
    tolerance = get(summary, "max_deficit_pu_tolerance", 1e-6)
    deficit === nothing || deficit <= tolerance || error(
        "stage $tag shed load at its final evaluation (max per-bus deficit " *
        "$deficit pu > $tolerance pu)",
    )
    checkpoint = get(summary, "checkpoint", "")
    endswith(checkpoint, "_latest.jld2") &&
        error("stage $tag reported a _latest snapshot as its checkpoint; refusing it")
    isfile(checkpoint) || error("stage $tag reported a checkpoint that does not exist: $checkpoint")
    return nothing
end

"""
    run_stage(config, stage, parent_record, out_dir; dry_run) -> Dict

Run one lineage stage in its own process and return its record.

The environment handed to the trainer is, in order: the ambient environment, the
lineage's `common` block, this stage's `env` block, and finally the parent
bindings — so a stage can override a common setting, and the parent bindings can
never be overridden by accident.
"""
function run_stage(config, stage, parent_record, out_dir; dry_run::Bool)
    tag = stage["tag"]
    summary_path = joinpath(out_dir, "trainer_summary_$(tag).json")
    env = Dict{String,String}()
    for (key, value) in get(config, "common", Dict())
        env[key] = string(value)
    end
    for (key, value) in get(stage, "env", Dict())
        env[key] = string(value)
    end
    env["DR_OUTPUT_TAG"] = tag
    env["DR_STAGE_SUMMARY"] = summary_path

    if parent_record === nothing
        # From scratch. Explicitly cleared rather than merely absent, so an
        # ambient DR_PRETRAINED_MODEL cannot make a "from-scratch" stage warm.
        env["DR_PRETRAINED_MODEL"] = ""
        env["DR_SEED_BEST"] = "Inf"
    else
        checkpoint = parent_record["checkpoint"]
        endswith(checkpoint, "_latest.jld2") &&
            error("refusing to chain stage $tag from a _latest snapshot: $checkpoint")
        isfile(checkpoint) || error("parent checkpoint for stage $tag is missing: $checkpoint")
        observed = sha256_file(checkpoint)
        observed == parent_record["checkpoint_sha256"] || error(
            "parent checkpoint for stage $tag changed on disk: recorded " *
            "$(parent_record["checkpoint_sha256"]), found $observed",
        )
        env["DR_PRETRAINED_MODEL"] = checkpoint
        env["DR_SEED_BEST"] = string(parent_record["best_validation"])
    end

    trainer = joinpath(HYDRO_DIR, get(config, "trainer", "train_hydro_exa_strict.jl"))
    isfile(trainer) || error("trainer script not found: $trainer")
    project = Base.active_project()
    command = `$(Base.julia_cmd()) --project=$(dirname(project)) -t auto $trainer`

    println("\n", "="^78)
    println("STAGE $tag   parent = ",
            parent_record === nothing ? "random initialisation" : parent_record["tag"])
    println("="^78)
    for key in sort(collect(keys(env)))
        @printf("  %-28s %s\n", key, env[key])
    end
    dry_run && return Dict{String,Any}("tag" => tag, "dry_run" => true)

    started = now()
    t0 = time()
    process = run(setenv(command, merge(ENV, env)); wait = false)
    wait(process)
    wall = time() - t0
    success(process) ||
        error("stage $tag exited with code $(process.exitcode); the lineage stops here")

    isfile(summary_path) ||
        error("stage $tag wrote no summary at $summary_path; cannot verify its selection")
    summary = JSON.parsefile(summary_path)
    validate_selection(summary, tag)

    record = Dict{String,Any}(
        "tag" => tag,
        "parent" => parent_record === nothing ? nothing : parent_record["tag"],
        "parent_checkpoint_sha256" =>
            parent_record === nothing ? nothing : parent_record["checkpoint_sha256"],
        "from_scratch" => parent_record === nothing,
        "environment" => env,
        "started_utc" => string(started),
        "process_wall_seconds" => wall,
        "train_loop_seconds" => get(summary, "train_seconds", nothing),
        "updates_run" => get(summary, "updates_run", nothing),
        "best_validation" => get(summary, "best_validation", nothing),
        "n_valid_saves" => get(summary, "n_valid_saves", nothing),
        "n_invalid_evals" => get(summary, "n_invalid_evals", nothing),
        "rollout_evaluation" => get(summary, "rollout_evaluation", nothing),
        "eval_protocol_ids" => get(summary, "eval_protocol_ids", nothing),
        "stop_reason" => get(summary, "stop_reason", nothing),
        "checkpoint" => summary["checkpoint"],
        "checkpoint_sha256" => sha256_file(summary["checkpoint"]),
    )
    open(stage_record_path(out_dir, tag), "w") do io
        JSON.print(io, record, 2)
    end
    @printf("  -> selected %s\n     sha256 %s\n     best %s after %s updates in %.1f s\n",
            basename(record["checkpoint"]), record["checkpoint_sha256"],
            record["best_validation"], record["updates_run"], wall)
    return record
end

function main()
    config_path = option("--config=", joinpath(HYDRO_DIR, "lineage_from_scratch.json"))
    isfile(config_path) || error("lineage configuration not found: $config_path")
    config = JSON.parsefile(config_path)
    out_dir = option("--out=", joinpath(HYDRO_DIR, "bolivia", "ACPPowerModel", "lineage"))
    mkpath(out_dir)
    dry_run = "--dry-run" in ARGS
    only_stages = let raw = option("--stages=")
        raw === nothing ? nothing : Set(String.(split(raw, ',')))
    end

    println("lineage: ", get(config, "name", "(unnamed)"), "   config: ", config_path)
    println("records: ", out_dir)

    records = Dict{String,Any}()
    ordered = String[]
    for stage in config["stages"]
        tag = stage["tag"]
        parent_tag = get(stage, "parent", nothing)
        parent_record = parent_tag === nothing ? nothing : get(records, parent_tag) do
            error("stage $tag names parent $parent_tag, which has not run")
        end

        existing = completed_stage(out_dir, tag)
        if existing !== nothing && (only_stages === nothing || !(tag in only_stages))
            println("\nSTAGE $tag  already complete (checkpoint hash verified) — skipping")
            records[tag] = existing
            push!(ordered, tag)
            continue
        end
        if only_stages !== nothing && !(tag in only_stages)
            error("stage $tag was not requested but has no completed record; " *
                  "a lineage cannot skip a stage its successors depend on")
        end

        records[tag] = run_stage(config, stage, parent_record, out_dir; dry_run = dry_run)
        push!(ordered, tag)
    end

    dry_run && return

    ledger = Dict{String,Any}(
        "lineage" => get(config, "name", "(unnamed)"),
        "config" => config_path,
        "generated_utc" => string(now()),
        "ancestry" => ordered,
        "stages" => [records[tag] for tag in ordered],
        "total_updates" => sum(get(records[tag], "updates_run", 0) for tag in ordered),
        "total_process_wall_seconds" =>
            sum(get(records[tag], "process_wall_seconds", 0.0) for tag in ordered),
        "total_train_loop_seconds" =>
            sum(something(get(records[tag], "train_loop_seconds", 0.0), 0.0) for tag in ordered),
        "final_checkpoint" => records[last(ordered)]["checkpoint"],
        "final_checkpoint_sha256" => records[last(ordered)]["checkpoint_sha256"],
        "final_validation" => records[last(ordered)]["best_validation"],
    )
    ledger_path = joinpath(out_dir, "lineage_ledger.json")
    open(ledger_path, "w") do io
        JSON.print(io, ledger, 2)
    end

    println("\n", "="^78)
    println("LINEAGE COMPLETE   ", join(ordered, " -> "))
    println("="^78)
    @printf("  updates          %d\n", ledger["total_updates"])
    @printf("  training loop    %.1f s\n", ledger["total_train_loop_seconds"])
    @printf("  process wall     %.1f s (%.4f h)\n",
            ledger["total_process_wall_seconds"],
            ledger["total_process_wall_seconds"] / 3600)
    @printf("  final validation %s\n", ledger["final_validation"])
    println("  final checkpoint ", ledger["final_checkpoint"])
    println("  sha256           ", ledger["final_checkpoint_sha256"])
    println("  ledger           ", ledger_path)
end

(abspath(PROGRAM_FILE) == @__FILE__) && main()
