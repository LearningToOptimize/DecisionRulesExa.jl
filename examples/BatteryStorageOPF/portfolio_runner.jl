#!/usr/bin/env julia
# portfolio_runner.jl
#
# The production runner for the two methods THIS engine owns:
# `:tsddr_nonlinear` and `:tsldr_recurrent_linear`. The other two study methods,
# `:sddp_soc` and `:sddp_dc`, belong to the JuMP engine
# (`DecisionRules.jl/examples/BatteryStorageOPF/portfolio_runner.jl`) and are
# refused here BY NAME, exactly as `run_battery_method` refuses them.
#
# WHAT THIS FILE IS, AND WHAT IT IS NOT.
# It is a SEGMENT DRIVER: it turns "train this frozen case, with this frozen
# configuration, from global update `a` to global update `b`, and survive being
# killed at any moment" into files on disk. It contains no policy, no stage
# model, no gradient, no cost convention and no evaluation rule of its own —
# every one of those comes from `train_battery_exa_strict.jl` and the two shared
# contract files, unchanged. What it adds is the machinery a long preemptible
# run needs and a single in-process training call does not: identity binding,
# verified checkpoints, resumption, a stop protocol and an honest result record.
#
# ─────────────────────────────────────────────────────────────────────────────
# COMMAND CONTRACT
#
#   julia --project=<this directory> portfolio_runner.jl \
#       --case-manifest <case_manifest.json> \
#       --method        <tsddr_nonlinear|tsldr_recurrent_linear> \
#       --config        <frozen config .toml> \
#       --protocol      <protocol descriptor .toml> \
#       --output        <segment output directory> \
#       --resume-from   <checkpoint path, or the literal string "none">
#
# Those six flags are sufficient on their own; the command above runs with no
# scheduler and no campaign controller of any kind. Five further flags exist
# purely as conveniences for an automated caller and ALL of them default:
#
#   --run-id <string>      (default "standalone")
#   --segment <int>        (default 1)
#   --attempt <int>        (default 1)
#   --stop-file <path>     (default <output>/STOP)
#   --max-seconds <float>  (default 1e9)
#
# ─────────────────────────────────────────────────────────────────────────────
# THE FROZEN CONFIGURATION  (`--config`, TOML)
#
#   target_index      total global updates for the WHOLE run (the run is done
#                     when the global update count reaches this)
#   segment_updates   global updates this invocation may add at most
#   checkpoint_every  updates between checkpoints
#   eval_every        updates between screening-panel evaluations
#   ma_window         window of the reported moving average of the training loss
#   num_stages        horizon T
#   trajectories      scenarios averaged into one gradient step
#   workers           persistent solver workers; 1 (default) is the serial path,
#                     production uses 2. `trajectories` must be at least this
#   lr, lr_final      endpoints of the cosine ramp, indexed by GLOBAL update
#   encoder_layers    e.g. [64, 64]
#   head_layers       e.g. [128, 128]
#   eval_columns      screening-protocol columns forming the fixed panel
#   seed              the run's single seed
#   device            "cpu" or "gpu"
#   max_recourse      physical admissibility tolerance, pu
#   method            OPTIONAL; when present it must equal --method
#
# ─────────────────────────────────────────────────────────────────────────────
# THE PROTOCOL DESCRIPTOR  (`--protocol`, TOML)
#
#   kind          "screening" (or "sole" on a fixture case that declares only
#                 one protocol). "final" is REFUSED.
#   num_stages, num_scenarios, seed, sha256
#
# The descriptor does not CARRY a protocol — the protocol is regenerated from
# the frozen case, as `read_battery_case` regenerates it, and the descriptor is
# what the regenerated one is checked against. A descriptor is therefore an
# assertion about identity, never a second source of truth, and it is what makes
# "this run was selected on the screening panel" a checkable claim rather than a
# promise. Write one for a case with:
#
#   julia --project=. -e 'include("portfolio_runner.jl");
#       write_protocol_descriptor("case/pglib_opf_case118_ieee", "screening.toml")'
#
# ─────────────────────────────────────────────────────────────────────────────
# OUTPUTS, all inside `--output`, none of them ever committed
#
#   checkpoints/ck_XXXXXXXX.jld2        the checkpoint payload
#   checkpoints/ck_XXXXXXXX.jld2.meta.toml   its digest, indices and lineage
#   history.csv       index, train_loss, panel_value, bound, solve_ok,
#                     solve_fail, deficit                    (7 columns, fixed)
#   trajectory.csv    index, train_loss, train_loss_ma<w>, lr, wall_seconds
#   evaluation.csv    index, protocol, columns, mean_cost, worst_recourse,
#                     complete, selected, best_cost
#   result.toml       the segment record
#   identity.toml     every coordinate this segment was bound to
#
# `bound` is `NaN` in every row: this is not a bounding method, and a column
# that exists for the SDDP arms is left empty rather than filled with a number
# that would be read as one.
#
# ─────────────────────────────────────────────────────────────────────────────
# DETERMINISM AND SEGMENTATION
#
# A run cut into pieces must produce what the same run in one piece produces.
# Three things carry that:
#
#   * the learning rate is `cosine_lr(i, target_index, lr, lr_final)` — a pure
#     function of the GLOBAL update index, never of a per-segment step counter;
#   * the scenario sampler's state is checkpointed and restored exactly, so
#     update `i` draws the atoms it would have drawn uninterrupted;
#   * the optimizer's moment estimates are checkpointed and restored, because
#     Adam's position is its moments as much as its weights, and a resumed
#     segment that silently restarts them takes a different first step.
#
# NOTHING here depends on where a segment started or how many times it was
# preempted. `trajectory_checksum`, the running sum of the per-update training
# loss, is the single number that certifies it.

using TOML
using SHA
using Dates
using Printf
using Random
using Statistics
using Flux
using Optimisers
using JLD2
using StableRNGs

# The certified implementation. Everything scientific comes from here; this file
# adds no second copy of any of it. Its `PROGRAM_FILE` guard keeps the include
# from launching a training run.
include(joinpath(@__DIR__, "train_battery_exa_strict.jl"))

# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

"""
Result-record schema. Must match the number the campaign controller verifies;
a runner speaking a different one is rejected rather than half-read.
"""
const RUNNER_RESULT_SCHEMA = 1

"""
Segment-checkpoint schema, distinct from `BATTERY_CHECKPOINT_SCHEMA`.

The policy checkpoint written by [`save_checkpoint`](@ref) restores a POLICY; a
segment checkpoint additionally restores a RUN — its global update count, its
sampler position, its evaluation history and its best admissible selection. The
two tags are checked separately so a policy-only file can never be mistaken for
a resumable segment.
"""
const RUNNER_CHECKPOINT_SCHEMA = "battery_storage_opf/segment/1"

"The two methods this engine owns, and the engine that owns the other two."
const RUNNER_METHODS = (:tsddr_nonlinear, :tsldr_recurrent_linear)

# ─────────────────────────────────────────────────────────────────────────────
# Small self-contained primitives
#
# Deliberately reimplemented here rather than shared with any caller: this file
# must run standalone, from a public checkout, with no orchestration package on
# the load path. `sha256_file` is the one exception — it already exists in
# `battery_case.jl`, which this file includes, and defining a second one would
# leave two digest functions that could drift apart.
# ─────────────────────────────────────────────────────────────────────────────

"UTC timestamp in the one format every record in this campaign uses."
utcnow() = Dates.format(now(UTC), dateformat"yyyy-mm-dd\THH:MM:SS\Z")

"SHA-256 of a byte buffer or a string, as lowercase hex."
sha256_hex(data::Vector{UInt8}) = bytes2hex(sha256(data))
sha256_hex(s::AbstractString) = bytes2hex(sha256(codeunits(String(s))))

"""
    atomic_write(path, data) -> String

Write `data` to a sibling temporary file, flush it, `fsync` it, `rename(2)` it
onto `path`, then read it back and re-hash it. Returns the digest.

# Notes
A file half-written when the node dies is never visible under its final name,
which is the whole reason a controller may trust any file it finds. The read-back
is not paranoia about `rename`: it catches a full filesystem and a silently
truncated write, both of which this project has seen.
"""
function atomic_write(path::AbstractString, data::Vector{UInt8})
    mkpath(dirname(abspath(path)))
    tmp = string(path, ".tmp.", getpid(), ".", time_ns())
    open(tmp, "w") do io
        write(io, data)
        flush(io)
        try
            ccall(:fsync, Cint, (Cint,), fd(io))
        catch
            # fsync is a durability optimisation here, not a correctness one:
            # the rename is what makes the file atomic. A filesystem that
            # refuses it must not take the run down.
        end
    end
    mv(tmp, path; force = true)
    got = sha256_file(path)
    want = sha256_hex(data)
    got == want || error("atomic_write verification failed for $path")
    return got
end
atomic_write(p::AbstractString, s::AbstractString) =
    atomic_write(p, Vector{UInt8}(codeunits(String(s))))

"Serialize a dictionary to TOML and write it atomically. Returns the digest."
function write_toml_atomic(path::AbstractString, d::AbstractDict)
    buf = IOBuffer()
    TOML.print(buf, d; sorted = true)
    return atomic_write(path, take!(buf))
end

"""
    runner_code_digest(dir) -> String

SHA-256 over the sorted `(relative path, file digest)` list of every `.jl` file
beside this runner.

# Notes
Recorded in the result so a number can be tied to the exact code that produced
it even when the checkout was dirty — which, during a study, it usually is. The
git commit is recorded too, and the two answer different questions: the commit
says which revision was checked out, the digest says what was actually run.
"""
function runner_code_digest(dir::AbstractString)
    rows = String[]
    for (root, _, files) in walkdir(dir)
        occursin("/.git", root) && continue
        for f in files
            endswith(f, ".jl") || continue
            full = joinpath(root, f)
            push!(rows, string(relpath(full, dir), " ", sha256_file(full)))
        end
    end
    sort!(rows)
    return sha256_hex(join(rows, "\n"))
end

"The git commit of `dir`, or `\"none\"` outside a repository."
function git_commit(dir::AbstractString)
    try
        return strip(read(`git -C $dir rev-parse HEAD`, String))
    catch
        return "none"
    end
end

"""
    parse_args(args) -> Dict{String,String}

`--key value` / `--flag` parser.

# Notes
Unknown flags are KEPT rather than rejected, so an automated caller may pass
extras; but no flag outside the documented six is ever REQUIRED, which is what
keeps the six-flag command a complete command.
"""
function parse_args(args)
    d = Dict{String,String}()
    i = 1
    while i <= length(args)
        if startswith(args[i], "--")
            k = args[i][3:end]
            if i < length(args) && !startswith(args[i+1], "--")
                d[k] = args[i+1]
                i += 2
            else
                d[k] = "true"
                i += 1
            end
        else
            i += 1
        end
    end
    return d
end

"Read an integer vector from a TOML value that may be a list or a single number."
_int_list(v) = v isa AbstractVector ? [Int(x) for x in v] : [Int(v)]

# ─────────────────────────────────────────────────────────────────────────────
# The protocol descriptor
# ─────────────────────────────────────────────────────────────────────────────

"""
    write_protocol_descriptor(case_dir, out_path; kind=:screening) -> String

Write the protocol descriptor a run is launched against, and return its path.

# Notes
Generated from the frozen case itself: the `kind`, the shape and the digest are
copied out of the case manifest, so a descriptor cannot describe a protocol the
case does not declare. `:final` is refused here as well as at load time — a
descriptor naming the fresh panel should not exist in the first place.
"""
function write_protocol_descriptor(case_dir::AbstractString, out_path::AbstractString;
                                   kind::Symbol = :screening)
    kind === :final && error("refusing to write a descriptor for the FINAL protocol")
    case = read_battery_case(case_dir)
    scr = get(case.manifest, "screening", nothing)
    block, resolved = if scr === nothing
        (case.manifest["protocol"], "sole")
    else
        (scr, "screening")
    end
    kind === :screening || String(kind) == resolved ||
        error("case $(case.name) declares a $resolved protocol, not a $kind one")
    write_toml_atomic(out_path, Dict{String,Any}(
        "kind"          => resolved,
        "case"          => case.name,
        "num_stages"    => Int(block["num_stages"]),
        "num_scenarios" => Int(block["num_scenarios"]),
        "seed"          => Int(block["seed"]),
        "sha256"        => String(block["sha256"]),
        "written_utc"   => utcnow(),
    ))
    return out_path
end

"""
    resolve_protocol(case, descriptor_path) -> (matrix, kind, declared)

Regenerate the evaluation protocol and bind it to the descriptor, fail-closed.

# Returns
`(matrix, kind, declared)` — the `(stages × scenarios)` atom-index matrix, the
kind [`evaluation_protocol`](@ref) actually produced, and the descriptor as read.

# Notes
FOUR refusals, in this order, and every one of them happens before a single
scenario outcome is computed:

 1. a descriptor whose `kind` is `"final"` — training may never be selected on
    the fresh panel, and the refusal must not depend on noticing it later;
 2. a descriptor whose kind disagrees with what the case declares;
 3. a shape that disagrees with the regenerated matrix;
 4. a digest that disagrees with the regenerated protocol's.

The last one is the load-bearing check. `evaluation_protocol` already re-derives
the screening protocol from the frozen support and re-verifies it against the
manifest; the descriptor adds the statement that THIS RUN was launched against
that protocol and not another, which is the part a result file can be audited on
afterwards.
"""
function resolve_protocol(case::BatteryCase, descriptor_path::AbstractString)
    isfile(descriptor_path) || error("no protocol descriptor at $descriptor_path")
    d = TOML.parsefile(descriptor_path)
    declared = String(get(d, "kind", ""))
    declared == "final" && error(
        "protocol descriptor $descriptor_path declares the FINAL protocol; " *
        "training may only be selected on the screening protocol")
    declared in ("screening", "sole") || error(
        "protocol descriptor $descriptor_path declares kind $(repr(declared)); " *
        "expected \"screening\" (or \"sole\" on a fixture case)")

    matrix, kind = evaluation_protocol(case)
    String(kind) == declared || error(
        "protocol descriptor declares $declared but the case regenerates a $kind protocol")

    block = kind === :sole ? case.manifest["protocol"] : case.manifest["screening"]
    Int(get(d, "num_stages", -1)) == Int(block["num_stages"]) ||
        error("protocol descriptor stage count does not match the case")
    Int(get(d, "num_scenarios", -1)) == Int(block["num_scenarios"]) ||
        error("protocol descriptor scenario count does not match the case")
    String(get(d, "sha256", "")) == String(block["sha256"]) ||
        error("protocol descriptor digest does not match the case's $kind protocol")
    size(matrix, 2) == Int(block["num_scenarios"]) ||
        error("regenerated protocol has $(size(matrix, 2)) columns, not $(block["num_scenarios"])")
    return matrix, kind, d
end

# ─────────────────────────────────────────────────────────────────────────────
# Identity
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_identity(; manifest_path, case, method, config_path, conf,
                   protocol_path, protocol_kind, num_stages) -> Dict

Every coordinate that makes two runs scientifically different, plus one digest
over all of them.

# The coordinates
| field | why it is here |
|---|---|
| `case_manifest_sha256` | the manifest FILE, byte for byte |
| `case_content_sha256` | the case CONTENT: the manifest's own artifact digests, so an edited manifest pointing at the same artifacts is still caught, and so is the reverse |
| `method` | which of the study's four this is |
| `config_sha256` | the frozen configuration file |
| `protocol_sha256`, `protocol_kind` | which panel selection may look at |
| `horizon` | the stage count actually trained |
| `seed` | the run's single seed |
| `architecture` | `:tsddr_nonlinear` or `:tsldr_recurrent_linear` |
| `acp_bound_relax_factor` | the common true-ACP setting both engines state explicitly |

# Notes
`identity_sha256` is a digest of the canonical `key=value` rendering of the
others. It is written into every checkpoint and re-derived on resume; a
mismatch on ANY coordinate refuses the resume rather than continuing a run whose
meaning changed underneath it. That is stricter than it needs to be for a
scheduler that always re-launches the same command — deliberately, because the
failure it prevents is silent and the cost of the check is a hash.
"""
function run_identity(; manifest_path, case, method, config_path, conf,
                        protocol_path, protocol_kind, num_stages,
                        conf_workers = 1, conf_trajectories = 1)
    # The case CONTENT digest is taken over the manifest's recorded artifact
    # digests, in sorted order. `read_battery_case` has already verified that
    # each artifact on disk hashes to its entry, so this one string stands for
    # the network, the batteries and the demand together.
    arts = case.manifest["artifacts"]
    content = join([string(k, "=", arts[k]) for k in sort!(collect(keys(arts)))], ";")

    id = Dict{String,Any}(
        "case"                   => case.name,
        "case_manifest"          => abspath(manifest_path),
        "case_manifest_sha256"   => sha256_file(manifest_path),
        "case_content_sha256"    => sha256_hex(content),
        "method"                 => String(method),
        "config_sha256"          => sha256_file(config_path),
        "protocol_sha256"        => sha256_file(protocol_path),
        "protocol_kind"          => String(protocol_kind),
        "horizon"                => Int(num_stages),
        "seed"                   => Int(conf["seed"]),
        "architecture"           => String(method),
        "engine"                 => "exa",
        "workers"                => Int(conf_workers),
        "trajectories"           => Int(conf_trajectories),
        "acp_bound_relax_factor" => ACP_BOUND_RELAX_FACTOR,
        "checkpoint_schema"      => RUNNER_CHECKPOINT_SCHEMA,
    )
    id["identity_sha256"] = sha256_hex(join(
        [string(k, "=", id[k]) for k in sort!(collect(keys(id)))], "\n"))
    return id
end

"""
    assert_identity(want, got, whence)

Refuse a continuation whose identity differs from this segment's, naming the
first field that differs.

# Notes
Reporting the FIELD matters. "identity mismatch" sends a reader to diff two
hashes; "seed 1 vs 2" ends the investigation.
"""
function assert_identity(want::AbstractDict, got::AbstractDict, whence::AbstractString)
    for k in sort!(collect(keys(want)))
        k == "identity_sha256" && continue
        haskey(got, k) || error("$whence is missing the identity field `$k`")
        got[k] == want[k] || error(
            "$whence identity mismatch on `$k`: checkpoint has $(repr(got[k])), " *
            "this segment has $(repr(want[k]))")
    end
    String(get(got, "identity_sha256", "")) == String(want["identity_sha256"]) ||
        error("$whence identity digest mismatch")
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoints
# ─────────────────────────────────────────────────────────────────────────────

"""
    checkpoint_paths(output, index) -> (payload, sidecar)

`checkpoints/ck_XXXXXXXX.jld2` and its `.meta.toml`, zero-padded so a directory
listing sorts in run order.
"""
function checkpoint_paths(output::AbstractString, index::Integer)
    dir = joinpath(output, "checkpoints")
    name = @sprintf("ck_%08d.jld2", index)
    return joinpath(dir, name), joinpath(dir, name * ".meta.toml")
end

"""
    write_segment_checkpoint(output, index, state, identity; kind) -> (path, sha)

Write one self-contained, verified, monotonically numbered checkpoint.

# What it preserves
Policy architecture and parameters, the optimizer state, the global update
count, the sampler's RNG state, the evaluation history, and the best admissible
evaluation with the checkpoint index that produced it. That list is the
definition of "self-contained for continuation": anything missing from it is
something a resumed run would silently restart.

# Notes
ORDER MATTERS, and it is the same order the controller assumes. The payload is
written atomically and hashed FIRST; only then is the sidecar written naming
that digest. A crash between the two leaves a payload with no sidecar, which is
ignored; a crash before either leaves nothing. There is no interleaving that
leaves a sidecar vouching for a file that was never finished.

The payload goes through [`save_checkpoint`](@ref) — the certified writer, which
moves everything to the host so a GPU checkpoint reloads on a CPU — into a
temporary path, which is then renamed. `save_checkpoint` writes with JLD2
directly and is not atomic on its own; the rename is what makes it so.
"""
function write_segment_checkpoint(output::AbstractString, index::Integer,
                                  state::NamedTuple, identity::AbstractDict;
                                  kind::AbstractString = "periodic")
    path, meta_path = checkpoint_paths(output, index)
    mkpath(dirname(path))
    tmp = string(path, ".tmp.", getpid(), ".", time_ns())

    meta = Dict{String,Any}(
        "segment_schema"      => RUNNER_CHECKPOINT_SCHEMA,
        "case"                => identity["case"],
        "network_sha256"      => state.network_sha256,
        "num_stages"          => identity["horizon"],
        "panel_protocol"      => identity["protocol_kind"],
        "panel_columns"       => collect(Int, state.eval_columns),
        "panel_mean_cost"     => state.best_cost,
        "step"                => Int(index),
        "encoder_layers"      => collect(Int, state.encoder_layers),
        "head_layers"         => collect(Int, state.head_layers),
        "n_observation"       => Int(state.n_observation),
        "n_context"           => Int(state.n_context),
        # --- what makes it a SEGMENT checkpoint rather than a policy file ----
        "global_index"        => Int(index),
        "rng_state"           => string(state.rng.state),
        "trajectory_checksum" => state.checksum,
        "best_cost"           => state.best_cost,
        "best_index"          => Int(state.best_index),
        "updates_total"       => Int(index),
        "workers"             => Int(state.workers),
        "trajectories"        => Int(state.trajectories),
        "peak_simultaneous_solves" => Int(state.peak),
        "written_utc"         => utcnow(),
        "identity"            => Dict{String,Any}(identity),
    )
    save_checkpoint(tmp, state.policy, meta;
                    opt_state = state.opt_state, history = state.history)
    mv(tmp, path; force = true)
    sha = sha256_file(path)

    write_toml_atomic(meta_path, Dict{String,Any}(
        "file"          => basename(path),
        "sha256"        => sha,
        "parent_sha256" => state.parent_sha,
        "global_index"  => Int(index),
        "index_from"    => Int(state.index_from),
        "run_id"        => state.run_id,
        "segment"       => Int(state.segment),
        "attempt"       => Int(state.attempt),
        "kind"          => kind,
        "written_utc"   => utcnow(),
    ))
    return path, sha
end

"""
    load_segment_checkpoint(path) -> (data, meta)

Read a parent checkpoint, refusing it unless its sidecar digest matches the
payload on disk and it carries this file's segment schema.

# Notes
The controller verifies checkpoints too. This check is not redundant with it: a
worker resuming from a file nobody re-hashed since the scan would be trusting a
window it cannot see into, and the two ends enforce the rule independently so
neither has to assume the other ran.
"""
function load_segment_checkpoint(path::AbstractString)
    isfile(path) || error("--resume-from does not exist: $path")
    meta_path = path * ".meta.toml"
    if isfile(meta_path)
        side = TOML.parsefile(meta_path)
        got = sha256_file(path)
        got == String(get(side, "sha256", "")) || error(
            "parent checkpoint digest mismatch: $path has $got, its sidecar claims " *
            "$(get(side, "sha256", "missing"))")
    end
    data = JLD2.load(path)
    meta = data["meta"]
    String(get(meta, "schema", "")) == BATTERY_CHECKPOINT_SCHEMA || error(
        "parent checkpoint $path is not a $BATTERY_CHECKPOINT_SCHEMA policy file")
    String(get(meta, "segment_schema", "")) == RUNNER_CHECKPOINT_SCHEMA || error(
        "parent checkpoint $path carries segment schema " *
        "$(repr(get(meta, "segment_schema", missing))) but this runner writes " *
        "$RUNNER_CHECKPOINT_SCHEMA")
    haskey(meta, "identity") || error(
        "parent checkpoint $path carries no identity record; it was not written " *
        "by this runner and cannot be continued")
    return data, meta
end

# ─────────────────────────────────────────────────────────────────────────────
# The segment
# ─────────────────────────────────────────────────────────────────────────────

"""
    moving_average(v, w) -> Vector{Float64}

Trailing moving average of window `w`, defined from the first sample:

``\\mathrm{ma}_i = \\frac{1}{\\min(i,w)} \\sum_{j=\\max(1,i-w+1)}^{i} v_j``

# Notes
The per-update training loss is one sample of a random objective — it is a
different estimand from the fixed-panel evaluation and is never compared with
it. Only its moving average is legible, so both are written and the raw column
is kept beside it.
"""
function moving_average(v::AbstractVector, w::Integer)
    n = length(v)
    out = zeros(Float64, n)
    s = 0.0
    for i in 1:n
        s += v[i]
        i > w && (s -= v[i-w])
        out[i] = s / min(i, w)
    end
    return out
end

"""
    run_segment(a) -> Int

Drive one segment: bind identity, resume or start, train to the segment's stop
index or until asked to stop, and write a verified checkpoint and an honest
result. Returns a process exit code.

# The loop, one update
Draw `trajectories` scenarios from the frozen per-stage supports, roll the
policy forward on each, solve each strict deterministic equivalent, average the
per-trajectory actor gradients, and apply one Adam step at
`cosine_lr(i, target_index, lr, lr_final)`. This is the certified update of
[`train_strict`](@ref) built out of the same functions; what differs is only
that the index `i` is GLOBAL and the loop can be stopped and resumed at any
boundary between updates.

A step in which every trajectory's solve failed applies nothing and does not
advance the global index — an index that moved without an update would make the
learning-rate schedule and the checksum disagree with an uninterrupted run.

# Stopping
The stop file is polled after every COMPLETE update. On a stop request the
current update is already finished, a checkpoint is written and verified, an
honest `preempted` result is written, and the process exits 0. `complete` is
reported only when the configured target index was actually reached.
"""
function run_segment(a::AbstractDict)
    t_start = time()

    # ---- the six scientific flags -----------------------------------------
    manifest_path = a["case-manifest"]
    method        = Symbol(a["method"])
    config_path   = a["config"]
    protocol_path = a["protocol"]
    output        = a["output"]
    resume        = get(a, "resume-from", "none")

    # ---- controller conveniences, every one defaulted ----------------------
    run_id    = get(a, "run-id", "standalone")
    segment   = parse(Int, get(a, "segment", "1"))
    attempt   = parse(Int, get(a, "attempt", "1"))
    stop_file = get(a, "stop-file", joinpath(output, "STOP"))
    max_secs  = parse(Float64, get(a, "max-seconds", "1e9"))

    # ---- method ownership, before anything is loaded -----------------------
    # Refused BY NAME and pointed at the engine that owns it, the same way
    # `run_battery_method` does. A runner that half-implemented the other pair
    # would fail somewhere far less legible than here.
    if !(method in RUNNER_METHODS)
        haskey(BATTERY_METHODS, method) || error(
            "unknown method :$method; the study's methods are " *
            "$(sort!(collect(keys(BATTERY_METHODS))))")
        error("method :$method runs on the $(battery_method(method).engine) engine " *
              "(DecisionRules.jl/examples/BatteryStorageOPF/portfolio_runner.jl), not on " *
              "this one; this package loads neither PowerModels nor SDDP")
    end

    mkpath(joinpath(output, "checkpoints"))
    conf = TOML.parsefile(config_path)
    haskey(conf, "method") && String(conf["method"]) != String(method) && error(
        "the frozen config names method $(conf["method"]) but --method is $method")

    target_index    = Int(get(conf, "target_index", 2000))
    segment_updates = Int(get(conf, "segment_updates", 500))
    ckpt_every      = Int(get(conf, "checkpoint_every", 50))
    eval_every      = Int(get(conf, "eval_every", 100))
    ma_window       = Int(get(conf, "ma_window", 25))
    num_stages      = Int(get(conf, "num_stages", 24))
    trajectories    = Int(get(conf, "trajectories", 2))
    workers         = Int(get(conf, "workers", BATTERY_DEFAULT_WORKERS))
    lr              = Float64(get(conf, "lr", 1e-3))
    lr_final        = Float64(get(conf, "lr_final", 1e-4))
    encoder_layers  = _int_list(get(conf, "encoder_layers", [64, 64]))
    head_layers     = _int_list(get(conf, "head_layers", [128, 128]))
    eval_columns    = _int_list(get(conf, "eval_columns", [1, 2, 3, 4]))
    seed            = Int(get(conf, "seed", 20260804))
    device          = String(get(conf, "device", "cpu"))
    max_recourse    = Float64(get(conf, "max_recourse", 1e-6))

    trajectories >= workers || error(
        "the frozen config sets trajectories=$trajectories and workers=$workers; " *
        "a worker with no scenario to solve would idle for the whole segment")
    workers <= Threads.nthreads() || @warn(
        "workers=$workers but this process has $(Threads.nthreads()) thread(s): a " *
        "blocking solve cannot overlap another on one thread, so the solves will " *
        "serialize and `peak_simultaneous_solves` will not reach `workers`")

    # ---- the frozen case, its protocol, and the identity -------------------
    case_dir = dirname(abspath(manifest_path))
    basename(manifest_path) == "case_manifest.json" || error(
        "--case-manifest must name a case_manifest.json, got $(basename(manifest_path))")
    case = read_battery_case(case_dir)

    # The protocol is resolved BEFORE the first update, so a run launched
    # against the wrong panel dies at its start rather than at its first
    # evaluation — and so no scenario outcome is computed on the way to finding
    # out.
    eval_matrix, protocol_kind, _ = resolve_protocol(case, protocol_path)
    size(eval_matrix, 1) >= num_stages || error(
        "the $protocol_kind protocol covers $(size(eval_matrix, 1)) stages but " *
        "training asks for $num_stages")
    maximum(eval_columns) <= size(eval_matrix, 2) || error(
        "panel column $(maximum(eval_columns)) is outside the $protocol_kind " *
        "protocol's $(size(eval_matrix, 2)) columns")

    ident = run_identity(; manifest_path = manifest_path, case = case, method = method,
                              config_path = config_path, conf = conf,
                              protocol_path = protocol_path, protocol_kind = protocol_kind,
                              num_stages = num_stages, conf_workers = workers,
                              conf_trajectories = trajectories)
    write_toml_atomic(joinpath(output, "identity.toml"), ident)

    @printf("segment %s seg%d att%d · method %s · case %s\n",
            run_id, segment, attempt, method, case.name)
    @printf("  panel: %s protocol, %d of %d columns · identity %s\n",
            protocol_kind, length(eval_columns), size(eval_matrix, 2),
            first(ident["identity_sha256"], 16))

    # ---- build the policy and the problem ----------------------------------
    backend, to_device, solver_kwargs = configure_device(device)
    prob = build_battery_exa(case, num_stages; backend = backend)
    Random.seed!(seed)
    rng = StableRNG(seed)
    policy = to_device(battery_reachable_policy(case, encoder_layers;
                                                n_observation = prob.nBus,
                                                n_context = N_CONTEXT,
                                                head_layers = head_layers,
                                                architecture = method))
    assert_device(policy, device)
    opt_state = Optimisers.setup(Optimisers.Adam(lr), policy)

    history = NamedTuple[]
    evaluations = NamedTuple[]
    best_cost = Inf
    best_index = 0
    checksum = 0.0
    index_from = 0
    parent_sha = "none"

    # ---- resume ------------------------------------------------------------
    if resume != "none" && !isempty(resume)
        data, meta = load_segment_checkpoint(resume)
        assert_identity(ident, Dict{String,Any}(meta["identity"]), "parent checkpoint")
        load_checkpoint!(policy, resume; case = case)
        # `load_checkpoint!` returns the host copy of the optimizer state; it has
        # to travel to the policy's device or the first update mixes memories.
        st = get(data, "opt_state", nothing)
        st === nothing && error("parent checkpoint carries no optimizer state")
        opt_state = to_device(st)
        h = get(data, "history", nothing)
        history = h === nothing ? NamedTuple[] : collect(h)
        index_from = Int(meta["global_index"])
        checksum   = Float64(meta["trajectory_checksum"])
        best_cost  = Float64(meta["best_cost"])
        best_index = Int(meta["best_index"])
        rng.state  = parse(UInt128, String(meta["rng_state"]))
        parent_sha = sha256_file(resume)
        @printf("  resumed from index %d · checksum %.10e · best %.6f\n",
                index_from, checksum, best_cost)
    end

    stop_target = min(target_index, index_from + segment_updates)
    index_from < stop_target || error(
        "nothing to do: resumed at $index_from with stop target $stop_target")

    # ---- local artifacts ---------------------------------------------------
    # `history.csv` is the controller's fixed seven-column schema. The richer
    # per-update record goes to `trajectory.csv` and the panel to
    # `evaluation.csv`, so no consumer has to guess which column means what.
    hist_io = open(joinpath(output, "history.csv"), "w")
    println(hist_io, "index,train_loss,panel_value,bound,solve_ok,solve_fail,deficit")
    flush(hist_io)

    losses_seen = Float64[]
    traj_rows = Tuple{Int,Float64,Float64,Float64}[]
    solve_ok = 0
    solve_fail = 0
    worst_deficit = 0.0
    e0 = initial_energy(case)
    like = _policy_array(policy)
    e0_dev = _to_like(like, e0)

    # Persistent workers for the whole segment: each binds the device and then
    # builds its OWN problem and solver inside its own task. They are shut down
    # in `finally`, including on a stop-file exit or an exception.
    pool = battery_worker_pool(case, num_stages, device, solver_kwargs; workers = workers)
    dispatched_ids = Int[]; accepted_ids = Int[]; rejected_ids = Int[]
    @printf("  workers %d · trajectories %d · peak simultaneous solves reported per batch\n",
            workers, trajectories)
    for r in pool.report
        @printf("    worker %d: device %s stream %s linear_solver %s problem @%s\n",
                r.worker, r.device, r.stream, r.linear_solver, string(r.problem; base = 16))
    end
    # FLUSH. This process is killed by a signal when the wall or a preemption
    # arrives, and anything still in Julia's stdout buffer dies with it — a
    # segment that ran for hours would leave an empty log. Flushing here and
    # after every update keeps the record legible at the moment it matters most.
    flush(stdout)

    # TIME ACCOUNTING. The 12-hour budget is on ACTIVE TRAINING, so the three
    # costs are measured separately rather than lumped into one wall figure:
    # setup (case load, protocol resolution, policy build, worker pool),
    # training (the gradient batches), and evaluation (the screening panel).
    # Queue time and depot construction happen outside this process entirely and
    # are reported by the controller, not here.
    setup_seconds = time() - t_start
    training_seconds = 0.0
    evaluation_seconds = 0.0

    idx = index_from
    last_ck_path = resume == "none" ? "" : resume
    last_ck_sha  = parent_sha
    reason = "segment_updates_reached"

    _state() = (policy = policy, opt_state = opt_state, history = history,
                workers = workers, trajectories = trajectories, peak = pool.peak,
                rng = rng, checksum = checksum, best_cost = best_cost,
                best_index = best_index, parent_sha = parent_sha,
                index_from = index_from, run_id = run_id, segment = segment,
                attempt = attempt, eval_columns = eval_columns,
                encoder_layers = encoder_layers, head_layers = head_layers,
                n_observation = prob.nBus, n_context = N_CONTEXT,
                network_sha256 = case.manifest["artifacts"]["network.json"])

    try
    while idx < stop_target
        i = idx + 1
        this_lr = cosine_lr(i, target_index, lr, lr_final)
        Optimisers.adjust!(opt_state, this_lr)

        # THE SHARED BATCH. Scenarios are drawn in order and numbered, dispatched
        # up to `workers` at a time, collected by scenario NUMBER and reduced in
        # scenario-number order — so the update does not depend on which solve
        # finished first, and the sampler advances by exactly `trajectories`
        # draws whatever the worker count is.
        _t_batch = time()
        batch = trajectory_batch!(pool, policy, prob, case, rng, num_stages, e0_dev, like;
                                  solver_kwargs = solver_kwargs,
                                  trajectories = trajectories, index = i)
        training_seconds += time() - _t_batch
        grads = batch.grads
        losses = batch.losses
        solve_ok += length(batch.accepted)
        solve_fail += length(batch.rejected)
        worst_deficit = max(worst_deficit, batch.worst_recourse)
        append!(dispatched_ids, batch.dispatched)
        append!(accepted_ids, batch.accepted)
        append!(rejected_ids, batch.rejected)

        if isempty(losses)
            # Every trajectory failed: nothing is applied and the global index
            # does not move. Retrying with the next draws is the only choice
            # that keeps index, schedule and checksum consistent.
            @warn "no usable trajectory at index $i; retrying with the next draws"
            if isfile(stop_file) || (time() - t_start) > max_secs
                reason = isfile(stop_file) ? "signal_stop" : "max_seconds"
                break
            end
            continue
        end

        grads = _scale_grads(grads, 1 / length(losses))
        opt_state, policy = Optimisers.update!(opt_state, policy, grads)
        idx = i
        l = mean(losses)
        checksum += l
        push!(history, (step = idx, loss = l, lr = this_lr))
        push!(losses_seen, l)
        push!(traj_rows, (idx, l, this_lr, time() - t_start))
        @printf("update %6d  loss %16.6f  lr %.3e\n", idx, l, this_lr)
        flush(stdout)

        # ---- fixed-panel evaluation and checkpoint selection ---------------
        panel_value = NaN
        if eval_every > 0 && (idx % eval_every == 0 || idx == stop_target)
            _t_eval = time()
            ev = evaluate_panel(policy, prob, case, eval_columns;
                                max_recourse = max_recourse, solver_kwargs = solver_kwargs)
            evaluation_seconds += time() - _t_eval
            panel_value = ev.mean_cost
            selected = ev.complete && ev.mean_cost < best_cost
            selected && (best_cost = ev.mean_cost; best_index = idx)
            push!(evaluations, (index = idx, protocol = String(ev.protocol),
                                mean_cost = ev.mean_cost, worst_recourse = ev.worst_recourse,
                                complete = ev.complete, selected = selected,
                                best_cost = best_cost))
            @printf("  %s panel: mean %16.6f  worst recourse %.3e  complete %s%s\n",
                    ev.protocol, ev.mean_cost, ev.worst_recourse, ev.complete,
                    selected ? "  [selected]" : "")
            flush(stdout)
        end

        @printf(hist_io, "%d,%.10f,%s,NaN,%d,%d,%.10e\n", idx, l,
                isnan(panel_value) ? "NaN" : @sprintf("%.10f", panel_value),
                solve_ok, solve_fail, worst_deficit)
        flush(hist_io)

        # ---- checkpoint ----------------------------------------------------
        if idx % ckpt_every == 0 || idx == stop_target
            last_ck_path, last_ck_sha = write_segment_checkpoint(
                output, idx, _state(), ident;
                kind = idx == stop_target ? "final" : "periodic")
        end

        # ---- graceful stop, between complete updates -----------------------
        if isfile(stop_file) || (time() - t_start) > max_secs
            reason = isfile(stop_file) ? "signal_stop" : "max_seconds"
            @info "stopping early" reason index = idx
            if idx % ckpt_every != 0 && idx != stop_target
                last_ck_path, last_ck_sha = write_segment_checkpoint(
                    output, idx, _state(), ident; kind = "periodic")
            end
            break
        end
    end
    finally
        close_worker_pool!(pool)
    end
    close(hist_io)

    # ---- the richer local artifacts ----------------------------------------
    ma = moving_average(losses_seen, ma_window)
    open(joinpath(output, "trajectory.csv"), "w") do io
        println(io, "index,train_loss,train_loss_ma$(ma_window),lr,wall_seconds")
        for (k, r) in enumerate(traj_rows)
            @printf(io, "%d,%.10f,%.10f,%.6e,%.3f\n", r[1], r[2], ma[k], r[3], r[4])
        end
    end
    open(joinpath(output, "evaluation.csv"), "w") do io
        println(io, "index,protocol,columns,mean_cost,worst_recourse,complete,selected,best_cost")
        for e in evaluations
            @printf(io, "%d,%s,%s,%.10f,%.6e,%s,%s,%.10f\n", e.index, e.protocol,
                    join(eval_columns, " "), e.mean_cost, e.worst_recourse,
                    e.complete, e.selected, e.best_cost)
        end
    end

    # ---- the segment result -------------------------------------------------
    reached = idx >= stop_target
    reached && idx >= target_index && (reason = "target_reached")
    isempty(last_ck_path) && error(
        "the segment produced no checkpoint; refusing to write a result that " *
        "claims progress it cannot evidence")

    here = @__DIR__
    projdir = dirname(something(Base.active_project(), joinpath(here, "Project.toml")))
    result = Dict{String,Any}(
        "schema"               => RUNNER_RESULT_SCHEMA,
        "run_id"               => run_id,
        "segment"              => segment,
        "attempt"              => attempt,
        "method"               => String(method),
        "status"               => reached ? "complete" : "preempted",
        # `status` is the SEGMENT's verdict, and the controller depends on that:
        # a segment that finished its planned updates must be acceptable, or a
        # multi-segment run could never make progress. Whether the RUN is
        # finished is a different question and gets its own field — a reader
        # must never infer "the run reached its target" from "the segment
        # completed". `termination_reason` separates them too:
        # `target_reached` only when the configured target index was reached.
        "run_complete"         => idx >= target_index,
        "termination_reason"   => reason,
        "command"              => join(vcat(["julia", "--project=" * projdir, @__FILE__],
                                            ARGS), " "),
        "julia_version"        => string(VERSION),
        "project_toml_sha256"  => sha256_file(joinpath(projdir, "Project.toml")),
        "manifest_toml_sha256" => isfile(joinpath(projdir, "Manifest.toml")) ?
                                  sha256_file(joinpath(projdir, "Manifest.toml")) : "none",
        "code_commit"          => git_commit(here),
        "code_digest"          => runner_code_digest(here),
        "case_manifest"        => abspath(manifest_path),
        "case_digest"          => sha256_file(manifest_path),
        "config_digest"        => sha256_file(config_path),
        "protocol_digest"      => sha256_file(protocol_path),
        "protocol_kind"        => String(protocol_kind),
        "support_digest"       => String(case.manifest["support"]["sha256"]),
        "identity_digest"      => ident["identity_sha256"],
        "parent_checkpoint"    => resume,
        "parent_sha256"        => parent_sha,
        "child_checkpoint"     => last_ck_path,
        "child_sha256"         => last_ck_sha,
        "index_from"           => index_from,
        "index_to"             => idx,
        "updates_completed"    => idx - index_from,
        "target_index"         => target_index,
        "wall_seconds"         => round(time() - t_start, digits = 3),
        "setup_seconds"        => round(setup_seconds, digits = 3),
        "training_seconds"     => round(training_seconds, digits = 3),
        "evaluation_seconds"   => round(evaluation_seconds, digits = 3),
        "gpu_seconds"          => lowercase(device) == "gpu" ?
                                  round(time() - t_start, digits = 3) : 0.0,
        "device"               => device,
        # The accelerator as MEASURED, not as configured: what each worker bound
        # to, how many solves were ever in flight at once, and which scenario
        # numbers were dispatched, accepted and rejected.
        "workers"              => workers,
        "trajectories"         => trajectories,
        "peak_simultaneous_solves" => pool.peak,
        "worker_devices"       => [String(r.device) for r in pool.report],
        "worker_streams"       => [String(r.stream) for r in pool.report],
        "worker_linear_solver" => [String(r.linear_solver) for r in pool.report],
        "worker_problem_ids"   => [string(r.problem; base = 16) for r in pool.report],
        "distinct_problems"    => length(unique(r.problem for r in pool.report)),
        # TRUE only if a run configured for the GPU ended up with a worker that
        # is not on a CUDA device. Reported for every run, so "no CPU fallback"
        # is a measurement rather than an assumption.
        "cpu_fallback"         => lowercase(device) == "gpu" &&
                                  !all(occursin("CuDevice", String(r.device))
                                       for r in pool.report),
        "scenarios_dispatched" => length(dispatched_ids),
        "scenarios_accepted"   => length(accepted_ids),
        "scenarios_rejected"   => length(rejected_ids),
        "solve_total"          => solve_ok + solve_fail,
        "solve_optimal"        => solve_ok,
        "solve_failed"         => solve_fail,
        "physical_deficit"     => worst_deficit,
        "physical_surplus"     => 0.0,
        "best_panel_cost"      => best_cost == Inf ? NaN : best_cost,
        "best_panel_index"     => best_index,
        "trajectory_checksum"  => checksum,
        "history_rows"         => length(traj_rows),
        "history_sha256"       => sha256_file(joinpath(output, "history.csv")),
        "slurm_job_id"         => get(ENV, "SLURM_JOB_ID", ""),
        "slurm_array_id"       => get(ENV, "SLURM_ARRAY_JOB_ID", ""),
        "node"                 => gethostname(),
        "started_utc"          => Dates.format(unix2datetime(t_start),
                                               dateformat"yyyy-mm-dd\THH:MM:SS\Z"),
        "finished_utc"         => utcnow(),
        "eval_indices"         => [e.index for e in evaluations],
        "eval_values"          => [e.mean_cost for e in evaluations],
    )
    write_toml_atomic(joinpath(output, "result.toml"), result)
    @printf("segment done · %s · index %d→%d · checksum %.10e · best %.6f\n",
            result["status"], index_from, idx, checksum, best_cost)
    return 0
end

"""
    main(args=ARGS) -> Int

Check that the six scientific flags are present, then run one segment.
"""
function main(args = ARGS)
    a = parse_args(args)
    for k in ("case-manifest", "method", "config", "protocol", "output")
        haskey(a, k) || error("--$k is required; see the header of $(@__FILE__)")
    end
    return run_segment(a)
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
