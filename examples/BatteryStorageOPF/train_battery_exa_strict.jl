# train_battery_exa_strict.jl
#
# The single, fully parameterized strict TS-DDR training entry point for the
# battery study, plus the rollout and evaluation machinery it shares with the
# correctness gates.
#
# THE ACTOR GRADIENT, in one paragraph.
# The policy emits a strict, one-stage reachable target trajectory
# ``\hat e(\theta)``; the deterministic equivalent solves the true-ACP stage
# problems with the outgoing energy pinned to it, and returns the multipliers
# ``\lambda`` of the battery state transitions. By the envelope theorem the
# derivative of the solved value with respect to the trajectory is exactly
# ``\lambda``, so
#
#     ∇_θ Q(w; \hat e(θ)) = Σ_t λ_t ∇_θ \hat e_t(θ),
#
# and the update is obtained by differentiating the surrogate ⟨λ, ê(θ)⟩ with λ
# HELD CONSTANT. The whole recurrent chain — encoder memory, head, and the
# reachable bounds' dependence on the previous target — is inside that
# differentiation. Nothing is detached.
#
# Usage
#   julia --project=. -t auto train_battery_exa_strict.jl
#
# Environment (all optional):
#   DR_BAT_CASE_DIR     frozen case directory
#   DR_BAT_ARCH         tsddr_nonlinear | tsldr_recurrent_linear
#                                                              (default tsddr_nonlinear)
#   DR_BAT_STAGES       horizon T                              (default 24)
#   DR_BAT_EPOCHS       number of epochs                       (default 2)
#   DR_BAT_BATCHES      gradient steps per epoch               (default 5)
#   DR_BAT_TRAJ         trajectories per gradient step         (default 2)
#   DR_BAT_LR           initial learning rate                  (default 1e-3)
#   DR_BAT_LR_FINAL     final learning rate of the cosine ramp (default 1e-4)
#   DR_BAT_ENCODER      encoder widths, comma separated        (default 64,64)
#   DR_BAT_HEAD         head widths, comma separated           (default 128,128)
#   DR_BAT_EVAL_EVERY   gradient steps between panel evaluations (default 5)
#   DR_BAT_EVAL_COLS    protocol columns forming the panel     (default 1,2,3,4)
#   DR_BAT_SEED         training seed                          (default 20260804)
#   DR_BAT_DEVICE       "cpu" or "gpu"                         (default cpu)
#   DR_BAT_MAX_RECOURSE physical admissibility tolerance, pu   (default 1e-6)
#   DR_BAT_CHECKPOINT   checkpoint path

using Flux
using Zygote
using Optimisers
using JLD2
using Random
using StableRNGs
using Statistics
using Printf
using LinearAlgebra
using DecisionRulesExa

include(joinpath(@__DIR__, "battery_case.jl"))
include(joinpath(@__DIR__, "battery_solution_schema.jl"))
include(joinpath(@__DIR__, "battery_exa.jl"))
include(joinpath(@__DIR__, "battery_reachable_policy.jl"))

# ─────────────────────────────────────────────────────────────────────────────
# Observation and context
# ─────────────────────────────────────────────────────────────────────────────

"""
    stage_context(case, t) -> Vector{Float64}

Deterministic per-stage context the policy is allowed to see.

# Notes
The pair ``(\\sin 2\\pi t/P, \\cos 2\\pi t/P)`` for the deterministic profile's
period ``P``, which the frozen support records. It encodes the position in the daily cycle without a discontinuity at
midnight, which a raw hour index would introduce. This is DETERMINISTIC
information — knowing the clock is not knowing the future demand — so it does
not violate nonanticipativity.
"""
function stage_context(case::BatteryCase, t::Integer)
    P = profile_period(case.demand)
    θ = 2π * (t - 1) / P
    return [sin(θ), cos(θ)]
end

"Width of the context block."
const N_CONTEXT = 2

"""
    stage_observation(net, case, t, atom) -> Vector{Float64}

The observation revealed at the beginning of stage `t`: the realized per-bus
ACTIVE demand (pu), indexed by bus position.

# Notes
Reactive demand carries no extra information — every load is scaled by the same
multiplier on both sides, because the process preserves each load's power factor
— so feeding it would double the input width for nothing.
"""
function stage_observation(net::ExaNetwork, case::BatteryCase, t::Integer, atom::Integer)
    pd, _ = realized_demand(case, net, [t], [atom])
    return vec(pd)
end

# ─────────────────────────────────────────────────────────────────────────────
# Rollout
# ─────────────────────────────────────────────────────────────────────────────

"""
    rollout_features(case, net, atoms; stage_offset=0, like) -> AbstractMatrix

Build the `(n_context + n_observation) × T` matrix of policy inputs that do NOT
depend on the policy: the deterministic context and the revealed demand.

# Keywords
- `like::AbstractArray`: array supplying the element type and device the matrix
  must land on.

# Notes
These features are constants of the rollout. Building them OUTSIDE the
differentiated region matters for two reasons: it keeps the host-to-device copy
(a mutation, which automatic differentiation refuses to trace) off the tape, and
it makes explicit that nothing in the observation depends on the parameters.
"""
function rollout_features(case::BatteryCase, net::ExaNetwork,
                          atoms::AbstractVector{<:Integer};
                          stage_offset::Integer = 0,
                          like::AbstractArray)
    T = length(atoms)
    cols = [vcat(stage_context(case, stage_offset + i),
                 stage_observation(net, case, stage_offset + i, atoms[i])) for i in 1:T]
    host = reduce(hcat, cols)
    device = similar(like, size(host)...)
    copyto!(device, eltype(like).(host))
    return device
end

"""
    rollout_targets(policy, features, e0) -> AbstractVector

Roll the policy forward over a scenario and return the flat, stage-major target
trajectory ``[\\hat e_1; \\ldots; \\hat e_T]``.

# Arguments
- `policy::BatteryReachablePolicy`.
- `features::AbstractMatrix`: the per-stage inputs from
  [`rollout_features`](@ref); column `i` is stage `i`.
- `e0::AbstractVector`: initial energy, in battery-position order.

# Returns
- A vector of length `T·nBat`, differentiable in the policy parameters.

# Notes
The recurrent state starts from `Flux.initialstates` — the scenario boundary —
and is threaded explicitly through [`policy_step`](@ref), and the emitted target
of stage `t` becomes the incoming energy of stage `t+1`. That feedback is what
makes this a genuinely multistage policy, and it is precisely the path along
which the reachable bounds' dependence on the incoming energy carries gradient.

The trajectory is accumulated in a `Zygote.Buffer` so the loop stays type-stable
and still differentiable; a plain array write would be a mutation Zygote
refuses.
"""
function rollout_targets(policy::BatteryReachablePolicy, features::AbstractMatrix,
                         e0::AbstractVector)
    T = size(features, 2)
    nBat = policy.n_battery
    state = DecisionRulesExa._init_recurrent_state(policy.encoder)
    e_prev = e0
    buf = Zygote.Buffer(similar(e0, T * nBat))
    for i in 1:T
        target, state = policy_step(policy, state, vcat(features[:, i], e_prev))
        buf[((i - 1) * nBat + 1):(i * nBat)] = target
        e_prev = target
    end
    return copy(buf)
end

"""
    rollout_raw_targets(policy, features) -> Matrix

Roll the LINEAR decision rule forward over a scenario and return its RAW
targets, before the feasibility layer: column `t` is ``z_t``.

# Arguments
- `policy::BatteryReachablePolicy`: must be `:tsldr_recurrent_linear`.
- `features::AbstractMatrix`: the per-stage inputs from
  [`rollout_features`](@ref); column `t` is ``\\xi_t``.

# Notes
No incoming energy is threaded through this rollout, and that is the whole
point: the raw target of the linear rule is a function of the observed demand
history ALONE, so the map this returns is the affine causal map
``(\\xi_1,\\ldots,\\xi_T) \\mapsto (z_1,\\ldots,z_T)`` that the causality,
history, affinity and unrolling gates are stated about. The emitted targets —
which do depend on the state, through the reachable interval — come from
[`rollout_targets`](@ref) as they do for either architecture.

Accumulated in a `Zygote.Buffer` for the same reason as the target rollout, so
that the raw map is differentiable too and a gate may difference it directly.
"""
function rollout_raw_targets(policy::BatteryReachablePolicy, features::AbstractMatrix)
    T = size(features, 2)
    nBat = policy.n_battery
    state = DecisionRulesExa._init_recurrent_state(policy.encoder)
    buf = Zygote.Buffer(similar(features, nBat, T))
    for i in 1:T
        z, state = raw_target_step(policy, state, features[:, i])
        buf[:, i] = z
    end
    return copy(buf)
end

"""
    initial_energy(case) -> Vector{Float64}

The frozen initial energy of every battery, in battery-position order.
"""
initial_energy(case::BatteryCase) =
    Float64[b.energy_initial for b in sort(collect(case.batteries); by = b -> b.index)]

# ─────────────────────────────────────────────────────────────────────────────
# One solve of the strict deterministic equivalent
# ─────────────────────────────────────────────────────────────────────────────

"""
    strict_solve!(prob, case, atoms, targets; stage_offset=0, solver_kwargs=NamedTuple())
        -> (result, solution, λ)

Impose one scenario and one target trajectory on the deterministic equivalent,
solve it, and return the solution together with the actor signal.

# Notes
`λ` is [`target_multipliers`](@ref)'s output: ``\\partial Q/\\partial \\hat e``,
already carrying the ``\\alpha`` correction that links consecutive transition
rows.

The scenario and the targets are written into the model's PARAMETERS, so the
model itself — its sparsity pattern, its derivative kernels — is built once and
reused for the whole run; only the solver instance is fresh, for the reason
documented on [`solve!`](@ref).
"""
function strict_solve!(prob::BatteryExaProblem, case::BatteryCase,
                       atoms::AbstractVector{<:Integer}, targets::AbstractVector;
                       stage_offset::Integer = 0,
                       solver_kwargs = NamedTuple())
    stages = collect((stage_offset + 1):(stage_offset + prob.horizon))
    # Demand is a parameter and may be re-imposed for any window; per-stage
    # generator availability is in the bounds and may not. See
    # [`assert_stage_window`](@ref) — a case with no schedule accepts every offset.
    assert_stage_window(prob, stages)
    pd, qd = realized_demand(case, prob.net, stages, collect(atoms))
    set_demand!(prob, pd, qd)
    set_energy_path!(prob, initial_energy(case), Float64.(vec(Array(targets))))
    result = solve!(prob; solver_kwargs...)
    return result, battery_solution(prob, result), target_multipliers(prob, result)
end

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

# `evaluation_protocol(case)` — which panel a policy may be SELECTED on — now
# lives in `battery_case.jl`, the file both engines carry byte-identically. It
# is a property of the frozen CASE and not of this trainer, and the JuMP engine
# needs exactly the same answer; keeping one copy is what makes "both engines
# selected on the same protocol" a fact rather than two implementations that
# agree today.

"""
    evaluate_panel(policy, prob, case, columns; max_recourse, solver_kwargs)
        -> NamedTuple

Evaluate the policy on a fixed panel of paired SCREENING-protocol columns.

# Returns
`(mean_cost, costs, worst_recourse, complete, protocol)`.

# Notes
The panel is FIXED and comes from [`evaluation_protocol`](@ref), so its columns
mean the same demand paths for every checkpoint and for the SDDP baseline.
Checkpoint selection uses this panel and nothing else — never the training loss,
whose sample size changes between phases, and never the final paired protocol,
which is evaluated once after selection.

An evaluation is COMPLETE only if every column solved, every stage of every
column was ADMISSIBLE under the shared cost contract, and the worst physical
recourse on every column is within `max_recourse`. An incomplete evaluation is
invalid: averaging the columns that happened to succeed would report a policy
that does not exist.

**The reported cost is the CORRECTED one.** Each stage goes through
`physical_stage_cost`, the byte-identical contract both engines carry, which
projects every recourse element within `PHYSICAL_RECOURSE_TOL` of zero to exactly
zero and marks the stage inadmissible when an element is outside it. This engine
parks the two nodal recourse injections a bound-relaxation BELOW zero; at a
recourse price of 1e5–1e6 per pu, summed over a couple of thousand buses, the raw
objective carries tens of cost units of pure barrier residue on a stage where no
recourse was used at all. That residue is a property of the solver, not of the
policy, and the contract exists so it never reaches a selection metric — which is
what a screening-panel mean is.
"""
function evaluate_panel(policy::BatteryReachablePolicy, prob::BatteryExaProblem,
                        case::BatteryCase, columns::AbstractVector{<:Integer};
                        max_recourse::Real = 1e-6, solver_kwargs = NamedTuple())
    matrix, kind = evaluation_protocol(case)
    size(matrix, 1) >= prob.horizon ||
        error("the $kind protocol covers $(size(matrix, 1)) stages but the policy has $(prob.horizon)")
    maximum(columns) <= size(matrix, 2) ||
        error("column $(maximum(columns)) is outside the $kind protocol's $(size(matrix, 2)) columns")
    e0 = initial_energy(case)
    costs = Float64[]
    worst = 0.0
    complete = true
    like = _policy_array(policy)
    bus_id = [b.id for b in prob.net.buses]
    for c in columns
        atoms = matrix[1:prob.horizon, c]
        features = rollout_features(case, prob.net, atoms; like = like)
        targets = rollout_targets(policy, features, _to_like(like, e0))
        result, sol, _ = strict_solve!(prob, case, atoms, targets;
                                       solver_kwargs = solver_kwargs)
        if !solve_succeeded(result)
            complete = false
            continue
        end
        stage = stage_costs(prob, sol)
        corrected = 0.0
        admissible = true
        for t in 1:prob.horizon
            c_t = physical_stage_cost(
                (cost_generation = stage.generation[t], cost_throughput = stage.throughput[t],
                 deficit = Dict(bus_id[i] => sol.deficit[i, t] for i in eachindex(bus_id)),
                 surplus = Dict(bus_id[i] => sol.surplus[i, t] for i in eachindex(bus_id)),
                 objective = stage.total[t]), case.recourse)
            corrected += c_t.corrected
            admissible &= c_t.admissible
            worst = max(worst, c_t.worst_recourse)
        end
        admissible || (complete = false)
        push!(costs, corrected)
    end
    complete &= (length(costs) == length(columns)) && (worst <= max_recourse)
    return (mean_cost = isempty(costs) ? NaN : mean(costs), costs = costs,
            worst_recourse = worst, complete = complete, protocol = kind)
end

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

"""
    actor_gradient(policy, features, λ, e0) -> (value, gradient)

Differentiate the surrogate ``\\langle \\lambda, \\hat e(\\theta)\\rangle``
through the complete recurrent reachable policy.

# Returns
- `value`: the surrogate's value.
- `gradient`: the gradient tree with respect to the policy's parameters.

# Notes
`λ` enters as a CONSTANT: it is the envelope-theorem derivative of the solved
stage value, already evaluated at this trajectory, so differentiating it again
would double-count the stage problem's response.

Everything else IS differentiated: the encoder's recurrent chain, the head, the
affine map into the reachable interval, and — critically — the interval's own
dependence on the incoming energy, which is the previous stage's target.
"""
function actor_gradient(policy::BatteryReachablePolicy, features::AbstractMatrix,
                        λ::AbstractVector, e0::AbstractVector)
    out = Flux.withgradient(policy) do m
        sum(λ .* rollout_targets(m, features, e0))
    end
    return out.val, out.grad[1]
end

"""
    sample_atoms(rng, case, T) -> Vector{Int}

Draw one training scenario: `T` independent atom indices, each from its own
stage's frozen support.
"""
function sample_atoms(rng, case::BatteryCase, T::Integer)
    # Per STAGE: the frozen support is stage-dependent in general, so one shared
    # probability vector would sample the wrong distribution on any case whose
    # late stages carry a different support.
    return [begin
                p = cumsum(atom_probabilities(case.demand, t))
                searchsortedfirst(p, rand(rng))
            end for t in 1:T]
end

"""
    cosine_lr(step, total, lr0, lr1) -> Float64

Cosine ramp from `lr0` to `lr1` over `total` steps.

# Notes
Declared as a function of the step index rather than carried as optimizer state,
so a run that is interrupted and resumed follows the same schedule it would have
followed uninterrupted.
"""
function cosine_lr(step::Integer, total::Integer, lr0::Real, lr1::Real)
    total <= 1 && return Float64(lr1)
    x = clamp((step - 1) / (total - 1), 0.0, 1.0)
    return lr1 + 0.5 * (lr0 - lr1) * (1 + cos(π * x))
end

"""
Schema tag every checkpoint this file writes carries, and every checkpoint it
loads must carry.

# Notes
Bumped from the unversioned Phase-A format when the policy gained an
`architecture` and the checkpoint gained the optimizer state and the training
trajectory. A checkpoint written before that has neither, and loading one would
resume a run with a fresh Adam moment estimate while reporting that it had
resumed — so it is refused by tag rather than adapted.
"""
const BATTERY_CHECKPOINT_SCHEMA = "battery_storage_opf/checkpoint/3"

"""
    save_checkpoint(path, policy, meta; opt_state=nothing, history=nothing)

Write the policy's trainable state, its architecture, its optimizer state, its
training trajectory and its metadata to `path`.

# Notes
Reachability metadata is deliberately NOT what a checkpoint restores: it belongs
to the frozen case, and a checkpoint that could override a battery rating would
let a stale file silently redefine the problem it was trained on. The case's
manifest hash and the ARCHITECTURE are recorded instead, so a reload against a
different case, or into a different architecture, FAILS rather than quietly
mismatching.

The optimizer state and the trajectory are written because a checkpoint that
restores only the parameters does not restore the RUN: Adam's moment estimates
are as much of the optimizer's position as the weights are, and a resumed stage
that silently restarts them takes a different first step than an uninterrupted
one would have. Both are moved to the host before writing, so a checkpoint taken
on a GPU reloads on a CPU.
"""
function save_checkpoint(path::AbstractString, policy::BatteryReachablePolicy, meta::AbstractDict;
                         opt_state = nothing, history = nothing)
    mkpath(dirname(abspath(path)))
    record = Dict{String,Any}(meta)
    record["schema"] = BATTERY_CHECKPOINT_SCHEMA
    record["architecture"] = String(policy.architecture)
    JLD2.jldsave(path;
                 state = Flux.state(Flux.cpu(policy)),
                 opt_state = opt_state === nothing ? nothing : Flux.cpu(opt_state),
                 history = history === nothing ? nothing : collect(history),
                 meta = record)
    return path
end

"""
    load_checkpoint!(policy, path; case=nothing) -> NamedTuple

Load a checkpoint into `policy`, verifying it belongs to the case in hand AND to
the architecture in hand.

# Returns
`(meta, opt_state, history)`. `opt_state` and `history` are `nothing` when the
checkpoint carried none; the optimizer state comes back on the HOST and must be
moved with the policy by the caller.

# Notes
Three refusals, each closing a way a run could continue while reporting
something untrue:

- a checkpoint without this file's schema tag predates the architecture field
  and the optimizer state;
- a checkpoint from a different network is a different problem;
- a checkpoint from a different ARCHITECTURE is a different function class. The
  structural mismatch between an `LSTMCell` and an `RNNCell` would usually stop
  it anyway, but "usually" is not a guarantee — two encoders can be built whose
  weight shapes coincide — so the architecture is checked by name first, before
  a single array is touched.
"""
function load_checkpoint!(policy::BatteryReachablePolicy, path::AbstractString;
                          case::Union{Nothing,BatteryCase} = nothing)
    data = JLD2.load(path)
    meta = data["meta"]
    get(meta, "schema", "") == BATTERY_CHECKPOINT_SCHEMA || error(
        "checkpoint $path carries schema $(repr(get(meta, "schema", missing))) " *
        "but this engine writes $BATTERY_CHECKPOINT_SCHEMA")
    want_arch = String(policy.architecture)
    meta["architecture"] == want_arch || error(
        "checkpoint $path was trained with architecture $(meta["architecture"]) " *
        "but the policy in hand is $want_arch")
    if case !== nothing && haskey(meta, "network_sha256")
        want = case.manifest["artifacts"]["network.json"]
        meta["network_sha256"] == want || error(
            "checkpoint $path was trained on network $(meta["network_sha256"]) but the case in hand is $want")
    end
    load_stateconditioned_policy!(policy, data["state"])
    return (meta = meta, opt_state = get(data, "opt_state", nothing),
            history = get(data, "history", nothing))
end


# ─────────────────────────────────────────────────────────────────────────────
# The trajectory batch — the ONE place a batch of scenarios is solved
#
# Both entry points in this study use it: `train_strict` (one process, one
# training call) and the segment driver `portfolio_runner.jl`. Keeping two
# implementations of "solve `trajectories` scenarios and reduce their gradients"
# is how the two silently diverge, and the divergence would be invisible: both
# would still train, just not the same way.
# ─────────────────────────────────────────────────────────────────────────────

"Default worker count. 1 reproduces the serial behaviour exactly."
const BATTERY_DEFAULT_WORKERS = 1

"""
    cuda_device_binder(device) -> (bind!, describe, upload)

Three closures for the accelerator: `bind!()` pins the CALLING TASK to the
device, `describe()` reports what that task actually has, and `upload(x)` moves
a host array onto the CALLING TASK's device.

`upload` is how a worker materializes the targets the main task sent it. The
main task marshals them through the host — anything replied or sent across tasks
must be device-neutral — so the worker has a plain `Vector` and needs it on its
own device. Deriving the destination from `CUDA.cu` inside the worker is what
makes that device the worker's own; taking a reference array out of the problem
does NOT work, because a problem's parameter fields are `ExaModels.Parameter`
handles rather than arrays.

# Notes
`bind!` must run INSIDE a worker task and BEFORE that worker constructs
anything, because the CUDA handles a solver allocates — CUBLAS, CUSPARSE, CUDSS
— bind to the calling task's device and stream at construction time. A problem
built on the main task and then solved from a worker is the documented way to
deadlock this stack.

Everything the `@eval`ed imports bring into scope is newer than this function's
compiled world, so each is reached through `Base.invokelatest`.
"""
function cuda_device_binder(device::AbstractString)
    lowercase(device) == "gpu" ||
        return (() -> nothing, () -> (device = "cpu", stream = "-"), identity)
    cuda = Base.invokelatest(getglobal, Main, :CUDA)
    bind! = function ()
        dev = Base.invokelatest(cuda.device)          # the process's current device
        Base.invokelatest(cuda.device!, dev)
        return nothing
    end
    describe = () -> (device = string(Base.invokelatest(cuda.device)),
                      stream = string(Base.invokelatest(cuda.stream)))
    # Element type is PRESERVED: the serial path handed `strict_solve!` whatever
    # `rollout_targets` produced, and changing precision here would change the
    # solve rather than only where it runs.
    upload = x -> Base.invokelatest(cuda.cu, x)
    return (bind!, describe, upload)
end

"""
    BatteryWorkerPool

Persistent per-segment workers, each owning its own problem and solver.

# Fields
- `workers`: how many.
- `inbox`, `outbox`: one request/reply channel per worker.
- `tasks`: the worker tasks, kept so `close_worker_pool!` can join them.
- `report`: what each worker actually bound to — device, stream, solver type —
  recorded at construction so a claim about the accelerator is evidence rather
  than configuration.
- `peak`: the largest number of solves in flight at once, observed.

# Notes
`workers == 1` builds NO pool and no channels: the batch runs inline on the
calling task, which is bit-for-bit the path this file used before workers
existed. That is the compatible default, and it is also the only way the
single-worker and multi-worker results can be compared at all.
"""
mutable struct BatteryWorkerPool
    workers::Int
    inbox::Vector{Channel{Any}}
    outbox::Vector{Channel{Any}}
    tasks::Vector{Task}
    report::Vector{Any}
    peak::Int
    inflight::Int
end

"""
    battery_worker_pool(case, num_stages, device, solver_kwargs; workers) -> BatteryWorkerPool

Start `workers` persistent tasks, each of which binds the device, then builds
its OWN `BatteryExaProblem` and its own solver state inside that task.

# Notes
Nothing is shared. A pool of workers over one problem would serialize on the
solver's internal state at best and corrupt it at worst; the cost of a problem
per worker is memory, and memory is what an H200 has.

The whole worker body sits under `try`/`catch`: a `Threads.@spawn`ed task that
throws dies SILENTLY — the exception surfaces only at `wait`/`fetch` — so
without this the main task would block forever on `take!`, which is exactly what
a "hang" looks like from the outside. On error the worker reports loudly and
closes its outbox so the main loop fails fast instead of deadlocking.
"""
function battery_worker_pool(case::BatteryCase, num_stages::Integer,
                             device::AbstractString, solver_kwargs;
                             workers::Integer = BATTERY_DEFAULT_WORKERS)
    workers = Int(workers)
    workers >= 1 || throw(ArgumentError("workers must be at least 1, got $workers"))
    pool = BatteryWorkerPool(workers, Channel{Any}[], Channel{Any}[], Task[], Any[], 0, 0)
    if workers == 1
        # The inline path reports too, so "which device did this actually run
        # on" is answerable at every worker count rather than only above one.
        _, describe, _ = cuda_device_binder(device)
        d = describe()
        push!(pool.report, (worker = 1, device = d.device, stream = d.stream,
                            linear_solver = string(get(solver_kwargs, :linear_solver, "default")),
                            problem = UInt(0)))
        return pool
    end

    backend, _, _ = configure_device(device)
    bind!, describe, upload = cuda_device_binder(device)
    ready = Channel{Any}(workers)
    for wi in 1:workers
        inb = Channel{Any}(1); outb = Channel{Any}(1)
        push!(pool.inbox, inb); push!(pool.outbox, outb)
        t = Threads.@spawn try
            bind!()                                   # BEFORE anything is built
            prob_w = build_battery_exa(case, Int(num_stages); backend = backend)
            d = describe()
            put!(ready, (worker = wi, device = d.device, stream = d.stream,
                         linear_solver = string(get(solver_kwargs, :linear_solver, "default")),
                         problem = objectid(prob_w)))
            while true
                msg = take!(inb)
                msg === nothing && break
                (sid, atoms, targets_cpu) = msg
                targets_w = upload(targets_cpu)      # onto THIS worker's device
                result, sol, λ = strict_solve!(prob_w, case, atoms, targets_w;
                                               solver_kwargs = solver_kwargs)
                ok = solve_succeeded(result)
                # Everything sent back is DEVICE-NEUTRAL. A device array replied
                # into the main task's gradient is an illegal access waiting to
                # happen the moment the two tasks are not on the same device.
                put!(outb, (sid = sid, ok = ok, status = string(result.status),
                            loss = ok ? sum(stage_costs(prob_w, sol).total) : NaN,
                            lambda = ok ? Array(λ) : Float64[],
                            worst_recourse = ok ? max(maximum(abs, Array(sol.deficit)),
                                                      maximum(abs, Array(sol.surplus))) : NaN))
            end
        catch e
            @error "battery worker $wi died" exception = (e, catch_backtrace())
            close(outb)
            rethrow()
        end
        push!(pool.tasks, t)
    end
    for _ in 1:workers
        push!(pool.report, take!(ready))
    end
    return pool
end

"""
    close_worker_pool!(pool)

Send every worker its stop message and join it. Safe to call twice, and safe on
a pool whose workers already died.
"""
function close_worker_pool!(pool::BatteryWorkerPool)
    for ch in pool.inbox
        try; isopen(ch) && put!(ch, nothing); catch; end
    end
    for t in pool.tasks
        try; wait(t); catch; end
    end
    empty!(pool.tasks)
    return nothing
end

"""
    trajectory_batch!(pool, policy, prob, case, rng, num_stages, e0_dev, like;
                      solver_kwargs, trajectories, index) -> NamedTuple

Solve one batch of `trajectories` scenarios and return the reduced gradient.

# Returns
`(grads, losses, sids_dispatched, sids_accepted, sids_rejected, statuses,
worst_recourse, peak_inflight)`.

# The determinism contract
1. **All scenarios are drawn FIRST**, in order, from `rng`, and numbered
   `1..trajectories`. The sampler therefore advances by exactly `trajectories`
   draws per batch no matter how many workers run or in what order they finish,
   so a checkpointed RNG position means the same thing at every worker count.
2. Results are collected BY SCENARIO NUMBER, never by completion order.
3. Gradients are reduced in **scenario-number order**, because floating-point
   addition is not associative and a completion-ordered sum would make the run
   depend on which solve happened to finish first.
4. Exactly one optimizer update follows a batch, and only if at least one
   scenario succeeded — the caller applies it.

A failed scenario is retried ONCE, with the SAME scenario identity and the SAME
fixed solver configuration. There is no tolerance ladder and no second solver
setting: a scenario that fails twice under the frozen configuration is reported
as rejected and excluded from the reduction, never rescued by changing the
problem.
"""
function trajectory_batch!(pool::BatteryWorkerPool, policy::BatteryReachablePolicy,
                           prob::BatteryExaProblem, case::BatteryCase, rng,
                           num_stages::Integer, e0_dev, like;
                           solver_kwargs = NamedTuple(), trajectories::Integer = 2,
                           index::Integer = 0)
    n = Int(trajectories)
    n >= pool.workers || throw(ArgumentError(
        "trajectories ($n) must be at least workers ($(pool.workers)): a worker " *
        "with no scenario to solve would idle for the whole segment"))

    # (1) draw every scenario up front, in order
    atoms_by_sid = [sample_atoms(rng, case, Int(num_stages)) for _ in 1:n]
    features_by_sid = [rollout_features(case, prob.net, a; like = like) for a in atoms_by_sid]
    targets_by_sid = [rollout_targets(policy, f, e0_dev) for f in features_by_sid]

    results = Vector{Any}(undef, n)
    peak = 0

    if pool.workers == 1
        for sid in 1:n
            results[sid] = _solve_one_inline(prob, case, atoms_by_sid[sid],
                                             targets_by_sid[sid], sid, solver_kwargs)
        end
        peak = 1
    else
        # (2) dispatch up to `workers` at a time; collect by scenario number
        next = 1
        busy = Dict{Int,Int}()                        # worker -> scenario id
        free = collect(1:pool.workers)
        inflight = 0
        while next <= n || !isempty(busy)
            while next <= n && !isempty(free)
                wi = pop!(free)
                put!(pool.inbox[wi], (next, atoms_by_sid[next], Array(targets_by_sid[next])))
                busy[wi] = next
                next += 1
                inflight += 1
                peak = max(peak, inflight)
            end
            for (wi, sid) in collect(busy)
                # A dead worker closes its outbox on the way out. Without this
                # check the loop polls a channel that will never be ready again
                # and the segment hangs until the wall clock kills it — the
                # failure looks like "slow" and is actually "dead".
                if !isopen(pool.outbox[wi]) && !isready(pool.outbox[wi])
                    error("worker $wi died while solving scenario $sid; its outbox is closed")
                end
                if isready(pool.outbox[wi])
                    r = take!(pool.outbox[wi])
                    results[r.sid] = r
                    delete!(busy, wi); push!(free, wi); inflight -= 1
                end
            end
            isempty(busy) || sleep(0.001)
        end
    end

    # (3) one retry per failed scenario, same identity, same solver configuration
    for sid in 1:n
        results[sid].ok && continue
        @warn "strict solve failed; retrying the SAME scenario under the SAME configuration" index sid status=results[sid].status
        results[sid] = pool.workers == 1 ?
            _solve_one_inline(prob, case, atoms_by_sid[sid], targets_by_sid[sid], sid, solver_kwargs) :
            begin
                put!(pool.inbox[1], (sid, atoms_by_sid[sid], Array(targets_by_sid[sid])))
                take!(pool.outbox[1])
            end
    end

    # (4) reduce in SCENARIO-NUMBER order
    grads = nothing
    losses = Float64[]
    accepted = Int[]; rejected = Int[]; statuses = String[]
    worst = 0.0
    for sid in 1:n
        r = results[sid]
        push!(statuses, r.status)
        if !r.ok
            push!(rejected, sid)
            continue
        end
        push!(accepted, sid)
        push!(losses, Float64(r.loss))
        worst = max(worst, Float64(r.worst_recourse))
        _, g = actor_gradient(policy, features_by_sid[sid], _to_like(like, r.lambda), e0_dev)
        grads = grads === nothing ? g : _add_grads(grads, g)
    end
    pool.peak = max(pool.peak, peak)
    return (grads = grads, losses = losses, dispatched = collect(1:n),
            accepted = accepted, rejected = rejected, statuses = statuses,
            worst_recourse = worst, peak_inflight = peak)
end

"One scenario on the calling task, for the single-worker path."
function _solve_one_inline(prob, case, atoms, targets, sid, solver_kwargs)
    result, sol, λ = strict_solve!(prob, case, atoms, targets; solver_kwargs = solver_kwargs)
    ok = solve_succeeded(result)
    return (sid = sid, ok = ok, status = string(result.status),
            loss = ok ? sum(stage_costs(prob, sol).total) : NaN,
            lambda = ok ? Array(λ) : Float64[],
            worst_recourse = ok ? max(maximum(abs, Array(sol.deficit)),
                                      maximum(abs, Array(sol.surplus))) : NaN)
end

"""
    train_strict(; kwargs...) -> NamedTuple

Run one parameterized strict training stage, for either architecture.

# Keywords
- `architecture::Symbol`: `:tsddr_nonlinear` (default) or
  `:tsldr_recurrent_linear`; see [`BATTERY_ARCHITECTURES`](@ref).
- everything else as documented in this file's header.

# Notes
One gradient step is: draw `trajectories` scenarios, roll the policy forward on
each, solve each strict deterministic equivalent, average the per-trajectory
actor gradients, and apply one Adam step at the scheduled learning rate. A small
sample gives a noisy but cheap gradient, which is what bulk descent wants; a
large one gives a precise gradient, which is what final convergence wants.

Checkpoints are written only when a COMPLETE panel evaluation improves on the
best complete evaluation so far, so a policy that leans on physical recourse can
never become the selected one.

**The architecture changes the policy and NOTHING else.** The stage model, the
strict target equality, the dual-gradient signal, the admissibility rule, the
corrected cost contract, the checkpoint machinery and the screening panel are
the same objects and the same code for both, which is the only way the two can
be compared on the panel afterwards.
"""
function train_strict(; case_dir::AbstractString = get(ENV, "DR_BAT_CASE_DIR",
                                                       joinpath(@__DIR__, "case", "pglib_opf_case14_ieee")),
                        architecture::Symbol = Symbol(get(ENV, "DR_BAT_ARCH", "tsddr_nonlinear")),
                        num_stages::Integer = parse(Int, get(ENV, "DR_BAT_STAGES", "24")),
                        epochs::Integer = parse(Int, get(ENV, "DR_BAT_EPOCHS", "2")),
                        batches::Integer = parse(Int, get(ENV, "DR_BAT_BATCHES", "5")),
                        trajectories::Integer = parse(Int, get(ENV, "DR_BAT_TRAJ", "2")),
                        workers::Integer = parse(Int, get(ENV, "DR_BAT_WORKERS",
                                                          string(BATTERY_DEFAULT_WORKERS))),
                        lr::Real = parse(Float64, get(ENV, "DR_BAT_LR", "1e-3")),
                        lr_final::Real = parse(Float64, get(ENV, "DR_BAT_LR_FINAL", "1e-4")),
                        encoder_layers = _env_ints("DR_BAT_ENCODER", [64, 64]),
                        head_layers = _env_ints("DR_BAT_HEAD", [128, 128]),
                        eval_every::Integer = parse(Int, get(ENV, "DR_BAT_EVAL_EVERY", "5")),
                        eval_columns = _env_ints("DR_BAT_EVAL_COLS", [1, 2, 3, 4]),
                        seed::Integer = parse(Int, get(ENV, "DR_BAT_SEED", "20260804")),
                        device::AbstractString = get(ENV, "DR_BAT_DEVICE", "cpu"),
                        max_recourse::Real = parse(Float64, get(ENV, "DR_BAT_MAX_RECOURSE", "1e-6")),
                        checkpoint::AbstractString = get(ENV, "DR_BAT_CHECKPOINT",
                                                         joinpath(@__DIR__, "battery_policy.jld2")),
                        verbose::Bool = true)

    case = read_battery_case(case_dir)
    backend, to_device, solver_kwargs = configure_device(device)

    prob = build_battery_exa(case, Int(num_stages); backend = backend)

    Random.seed!(seed)
    rng = StableRNG(seed)
    policy = to_device(battery_reachable_policy(case, collect(Int, encoder_layers);
                                                n_observation = prob.nBus,
                                                n_context = N_CONTEXT,
                                                head_layers = collect(Int, head_layers),
                                                architecture = architecture))
    assert_device(policy, device)

    opt_state = Optimisers.setup(Optimisers.Adam(Float64(lr)), policy)
    e0 = initial_energy(case)
    like = _policy_array(policy)
    e0_dev = _to_like(like, e0)

    total_steps = epochs * batches
    # Resolved ONCE, before the first update, so a run that would have selected
    # on the wrong protocol fails at its start rather than at its first
    # evaluation — and so the kind is on the record from the beginning.
    eval_matrix, eval_protocol = evaluation_protocol(case)
    size(eval_matrix, 1) >= num_stages ||
        error("the $eval_protocol protocol has $(size(eval_matrix, 1)) stages but training asks for $num_stages")
    verbose && @printf("selection panel: %s protocol, %d columns of %d\n",
                       eval_protocol, length(eval_columns), size(eval_matrix, 2))

    history = NamedTuple[]
    best = (cost = Inf, step = 0)
    updates = 0
    t_start = time()

    # ONE batch implementation, shared with the segment driver. `workers == 1`
    # runs it inline, which is the path this function always took.
    pool = battery_worker_pool(case, num_stages, device, solver_kwargs; workers = workers)
    try
    for step in 1:total_steps
        Optimisers.adjust!(opt_state, cosine_lr(step, total_steps, lr, lr_final))
        batch = trajectory_batch!(pool, policy, prob, case, rng, num_stages, e0_dev, like;
                                  solver_kwargs = solver_kwargs,
                                  trajectories = trajectories, index = step)
        grads = batch.grads
        losses = batch.losses
        isempty(losses) && continue
        grads = _scale_grads(grads, 1 / length(losses))
        opt_state, policy = Optimisers.update!(opt_state, policy, grads)
        updates += 1
        push!(history, (step = step, loss = mean(losses),
                        lr = cosine_lr(step, total_steps, lr, lr_final)))
        verbose && @printf("step %4d  loss %14.4f  lr %.3e\n", step, mean(losses),
                           cosine_lr(step, total_steps, lr, lr_final))

        if eval_every > 0 && (step % eval_every == 0 || step == total_steps)
            ev = evaluate_panel(policy, prob, case, collect(Int, eval_columns);
                                max_recourse = max_recourse, solver_kwargs = solver_kwargs)
            verbose && @printf("  %s panel: mean %14.4f  worst recourse %.3e  complete %s\n",
                               ev.protocol, ev.mean_cost, ev.worst_recourse, ev.complete)
            if ev.complete && ev.mean_cost < best.cost
                best = (cost = ev.mean_cost, step = step)
                save_checkpoint(checkpoint, policy, Dict(
                    "case" => case.name,
                    "network_sha256" => case.manifest["artifacts"]["network.json"],
                    "num_stages" => Int(num_stages),
                    "panel_protocol" => String(ev.protocol),
                    "panel_columns" => collect(Int, eval_columns),
                    "panel_mean_cost" => ev.mean_cost,
                    "step" => step,
                    "encoder_layers" => collect(Int, encoder_layers),
                    "head_layers" => collect(Int, head_layers),
                    "n_observation" => prob.nBus,
                    "n_context" => N_CONTEXT,
                ); opt_state = opt_state, history = history)
            end
        end
    end

    finally
        # Workers are shut down on EVERY exit — normal, exception, or a caller
        # that stopped early. A leaked worker holds a solver and its device
        # memory for the life of the process.
        close_worker_pool!(pool)
    end

    return (policy = policy, problem = prob, case = case,
            architecture = architecture, opt_state = opt_state,
            protocol = eval_protocol, workers = workers,
            worker_report = pool.report, peak_inflight = pool.peak,
            history = history, best = best, updates = updates,
            elapsed = time() - t_start, checkpoint = checkpoint)
end

# ─────────────────────────────────────────────────────────────────────────────
# Device handling
# ─────────────────────────────────────────────────────────────────────────────

"""
    configure_device(device) -> (backend, to_device, solver_kwargs)

Resolve `"cpu"` or `"gpu"` into an ExaModels backend, a policy mover, and the
MadNLP options that match.

# Notes
The GPU path is loaded LAZILY, so the CPU path — which every correctness gate and
every CI run uses — never depends on a working CUDA installation.

Three traps this function exists to avoid. `MadNLPGPU.CUDSSSolver` is `nothing`
unless CUDSS.jl has been loaded, and passing `nothing` as a linear solver fails
far from its cause. `Flux.gpu` is a silent no-op when cuDNN is absent, so a
"GPU" run can quietly execute the policy on the CPU; `cuDNN` is therefore
imported here and [`assert_device`](@ref) checks the outcome rather than trusting
it. And everything the `@eval`ed imports bring into scope is NEWER than this
function's own compiled world, so each of them is reached through
`Base.invokelatest`; calling them directly raises an "UndefVarError … the
binding may be too new".
"""
function configure_device(device::AbstractString)
    if lowercase(device) == "cpu"
        return nothing, identity, NamedTuple()
    end
    lowercase(device) == "gpu" || throw(ArgumentError("device must be \"cpu\" or \"gpu\""))
    @eval Main using CUDA, CUDSS, MadNLPGPU, cuDNN, KernelAbstractions
    cuda = Base.invokelatest(getglobal, Main, :CUDA)
    Base.invokelatest(cuda.functional) ||
        error("DR_BAT_DEVICE=gpu but CUDA is not functional")
    backend = Base.invokelatest(cuda.CUDABackend)
    madnlpgpu = Base.invokelatest(getglobal, Main, :MadNLPGPU)
    linear_solver = Base.invokelatest(getglobal, madnlpgpu, :CUDSSSolver)
    linear_solver === nothing && error("MadNLPGPU.CUDSSSolver is nothing; CUDSS.jl did not load")
    return backend, x -> Base.invokelatest(Flux.gpu, x), (linear_solver = linear_solver,)
end

"""
    assert_device(policy, device)

Fail loudly unless every trainable array and the recurrent state actually live
on the intended device.

# Notes
A device move that silently did nothing is indistinguishable from a successful
one at the call site, and produces a run that reports GPU timings while
executing on the host. Checking the arrays themselves is the only assertion that
cannot be fooled.
"""
function assert_device(policy::BatteryReachablePolicy, device::AbstractString)
    want_gpu = lowercase(device) == "gpu"
    arrays = Any[]
    Flux.fmap(x -> (x isa AbstractArray && push!(arrays, x); x), policy)
    isempty(arrays) && error("assert_device found no arrays in the policy")
    for a in arrays
        on_gpu = !(a isa Array)
        on_gpu == want_gpu || error(
            "policy array of type $(typeof(a)) is $(on_gpu ? "on the GPU" : "on the CPU") but device=$device")
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    _policy_array(policy) -> AbstractArray

The first trainable array of `policy`, used as the reference for element type
and device when materializing rollout inputs.

# Notes
Deriving the working array type from the policy itself — rather than from a
flag — means a policy that failed to move to the GPU produces CPU inputs and a
consistent (if slow) run, instead of a mixed-device `vcat` that fails deep
inside a kernel launch.
"""
function _policy_array(policy::BatteryReachablePolicy)
    found = Ref{Any}(nothing)
    Flux.fmap(x -> (x isa AbstractArray && found[] === nothing && (found[] = x); x), policy)
    found[] === nothing && error("policy carries no arrays")
    return found[]
end

"Materialize `v` with the element type and device of `like`."
function _to_like(like::AbstractArray, v::AbstractVector)
    out = similar(like, length(v))
    copyto!(out, eltype(like).(vec(Array(v))))
    return out
end

"Parse a comma-separated integer list from the environment, with a default."
function _env_ints(key::AbstractString, default::AbstractVector{Int})
    raw = get(ENV, key, "")
    isempty(strip(raw)) && return default
    return [parse(Int, strip(x)) for x in split(raw, ",") if !isempty(strip(x))]
end

"Recursively add two Zygote gradient trees of identical structure."
_add_grads(a::AbstractArray, b::AbstractArray) = a .+ b
_add_grads(a::Nothing, b) = b
_add_grads(a, b::Nothing) = a
_add_grads(::Nothing, ::Nothing) = nothing
_add_grads(a::NamedTuple{K}, b::NamedTuple{K}) where {K} =
    NamedTuple{K}(map(_add_grads, values(a), values(b)))
_add_grads(a::Tuple, b::Tuple) = map(_add_grads, a, b)
_add_grads(a::Number, b::Number) = a + b

"Recursively scale a Zygote gradient tree."
_scale_grads(a::AbstractArray, s) = a .* s
_scale_grads(::Nothing, s) = nothing
_scale_grads(a::NamedTuple{K}, s) where {K} =
    NamedTuple{K}(map(x -> _scale_grads(x, s), values(a)))
_scale_grads(a::Tuple, s) = map(x -> _scale_grads(x, s), a)
_scale_grads(a::Number, s) = a * s

# ─────────────────────────────────────────────────────────────────────────────
# The study's four method identifiers
#
# One table, carried by BOTH public engines with the same four rows and the same
# invariant fields, because the comparison the study makes is between four
# methods and not between two packages. Each engine can RUN the two methods it
# owns and refuses the other two by name, pointing at the engine that owns them —
# neither package loads the other, and a dispatch layer that pretended otherwise
# would fail somewhere less legible than here.
# ─────────────────────────────────────────────────────────────────────────────

"""
    BATTERY_METHODS

The four policies this study compares, keyed by their stable identifier.

# The rows

| identifier | family | engine | what varies |
|---|---|---|---|
| `:tsddr_nonlinear` | `:tsddr` | `:exa` | LSTM encoder, nonlinear head |
| `:tsldr_recurrent_linear` | `:tsddr` | `:exa` | affine recurrence, affine head |
| `:sddp_soc` | `:sddp` | `:jump` | `SOCWRConicPowerModel` backward cuts |
| `:sddp_dc` | `:sddp` | `:jump` | `DCPPowerModel` backward cuts |

# The invariants

Every row declares the SAME `horizon`, `stage_semantics`, `recourse`,
`cost_contract` and `comparison` fields, and a test in each engine's suite
asserts it. They are recorded rather than assumed because the four methods are
only comparable if they share them: the frozen case and protocol identities,
`T = 24`, strict reachable targets with no target slack, uncapped physical nodal
recourse with the same admissibility rule, `physical_stage_cost` as the only
headline cost, and a true-ACP evaluation on paired protocol columns.

`protocol` is `"screening"` for every row at this phase. The final 500-column
protocol is not opened by anything in this file.
"""
const BATTERY_METHODS = Dict{Symbol,NamedTuple}(
    :tsddr_nonlinear => (
        family = :tsddr, engine = :exa, architecture = :tsddr_nonlinear,
        backward = nothing,
        summary = "strict TS-DDR with an LSTM encoder and a nonlinear bounded head"),
    :tsldr_recurrent_linear => (
        family = :tsddr, engine = :exa, architecture = :tsldr_recurrent_linear,
        backward = nothing,
        summary = "strict recurrent TSLDR: affine recurrence and affine head, " *
                  "raw target affine in the observed demand history"),
    :sddp_soc => (
        family = :sddp, engine = :jump, architecture = nothing,
        backward = :soc,
        summary = "SDDP with SOCWRConicPowerModel backward cuts and ACP forward decisions"),
    :sddp_dc => (
        family = :sddp, engine = :jump, architecture = nothing,
        backward = :dc,
        summary = "SDDP with DCPPowerModel backward cuts and ACP forward decisions"),
)

"""
The properties every one of [`BATTERY_METHODS`](@ref)' four rows shares.
"""
const BATTERY_METHOD_INVARIANTS = (
    horizon = 24,
    protocol = "screening",
    stage_semantics = "strict reachable outgoing-energy target, no target slack",
    recourse = "uncapped two-sided physical nodal active recourse, admissibility at 1e-6 pu",
    cost_contract = "physical_stage_cost from battery_solution_schema.jl",
    comparison = "true-ACP cost on paired frozen protocol columns",
)

"""
    battery_method(id::Symbol) -> NamedTuple

The descriptor of one method identifier, with the shared invariants merged in.

# Notes
An unknown identifier raises and lists the four, rather than returning
`nothing`: a campaign driver that silently skipped a misspelled method would
report three-quarters of a study as a whole one.
"""
function battery_method(id::Symbol)
    haskey(BATTERY_METHODS, id) || throw(ArgumentError(
        "unknown battery method :$id; the study's methods are $(sort!(collect(keys(BATTERY_METHODS))))"))
    return merge(BATTERY_METHODS[id], (id = id,), BATTERY_METHOD_INVARIANTS)
end

"""
    run_battery_method(id::Symbol; kwargs...) -> NamedTuple

Dispatch a method identifier to this engine's implementation.

# Notes
This engine owns the two `:exa` rows and runs them through the ONE
[`train_strict`](@ref) entry point, differing only in `architecture`. The two
`:sddp` rows belong to the JuMP engine: this package has no PowerModels and no
SDDP dependency by design, so they are refused by name here rather than
half-implemented.

This is a dispatch layer, not a campaign runner. It selects an implementation
and forwards keyword arguments; it schedules nothing, resumes nothing and writes
no ledger.
"""
function run_battery_method(id::Symbol; kwargs...)
    m = battery_method(id)
    m.engine === :exa || error(
        "method :$id runs on the $(m.engine) engine (DecisionRules.jl/examples/BatteryStorageOPF), " *
        "not on this one; this package loads neither PowerModels nor SDDP")
    return train_strict(; architecture = m.architecture, kwargs...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    out = train_strict()
    @printf("\narchitecture %s\n", out.architecture)
    @printf("%d updates in %.1f s; best complete panel %.4f at step %d\n",
            out.updates, out.elapsed, out.best.cost, out.best.step)
    @printf("checkpoint: %s\n", out.checkpoint)
end
