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

"""
    evaluate_panel(policy, prob, case, columns; protocol_stages, protocol_scenarios,
                   max_recourse) -> NamedTuple

Evaluate the policy on a fixed panel of paired protocol columns.

# Returns
`(mean_cost, costs, worst_recourse, complete)`.

# Notes
The panel is FIXED and comes from the frozen protocol, so its columns mean the
same demand paths for every checkpoint and for the SDDP baseline. Checkpoint
selection uses this panel and nothing else — never the training loss, whose
sample size changes between phases, and never the final paired protocol, which
is evaluated once after selection.

An evaluation is COMPLETE only if every column solved and the worst physical
recourse on every column is within `max_recourse`. An incomplete evaluation is
invalid: averaging the columns that happened to succeed would report a policy
that does not exist.
"""
function evaluate_panel(policy::BatteryReachablePolicy, prob::BatteryExaProblem,
                        case::BatteryCase, columns::AbstractVector{<:Integer};
                        protocol_stages::Integer, protocol_scenarios::Integer,
                        max_recourse::Real = 1e-6, solver_kwargs = NamedTuple())
    matrix = scenario_index_matrix(case.demand, protocol_stages, protocol_scenarios)
    e0 = initial_energy(case)
    costs = Float64[]
    worst = 0.0
    complete = true
    like = _policy_array(policy)
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
        rec = max(maximum(sol.deficit; init = 0.0), maximum(sol.surplus; init = 0.0))
        worst = max(worst, rec)
        push!(costs, sum(stage_costs(prob, sol).total))
    end
    complete &= (length(costs) == length(columns)) && (worst <= max_recourse)
    return (mean_cost = isempty(costs) ? NaN : mean(costs), costs = costs,
            worst_recourse = worst, complete = complete)
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
    save_checkpoint(path, policy, meta)

Write the policy's trainable state and its metadata to `path`.

# Notes
Only `Flux.state(policy)` is written. Reachability metadata is deliberately NOT
part of the checkpoint: it belongs to the frozen case, and a checkpoint that
could override a battery rating would let a stale file silently redefine the
problem it was trained on. The case's manifest hash is recorded instead, so a
reload against a different case FAILS rather than quietly mismatching.
"""
function save_checkpoint(path::AbstractString, policy::BatteryReachablePolicy, meta::AbstractDict)
    mkpath(dirname(abspath(path)))
    JLD2.jldsave(path; state = Flux.state(Flux.cpu(policy)), meta = Dict(meta))
    return path
end

"""
    load_checkpoint!(policy, path; case=nothing) -> Dict

Load a checkpoint into `policy`, verifying it belongs to the case in hand.
"""
function load_checkpoint!(policy::BatteryReachablePolicy, path::AbstractString;
                          case::Union{Nothing,BatteryCase} = nothing)
    data = JLD2.load(path)
    meta = data["meta"]
    if case !== nothing && haskey(meta, "network_sha256")
        want = case.manifest["artifacts"]["network.json"]
        meta["network_sha256"] == want || error(
            "checkpoint $path was trained on network $(meta["network_sha256"]) but the case in hand is $want")
    end
    load_stateconditioned_policy!(policy, data["state"])
    return meta
end

"""
    train_strict(; kwargs...) -> NamedTuple

Run one parameterized strict TS-DDR training stage.

# Notes
One gradient step is: draw `trajectories` scenarios, roll the policy forward on
each, solve each strict deterministic equivalent, average the per-trajectory
actor gradients, and apply one Adam step at the scheduled learning rate. A small
sample gives a noisy but cheap gradient, which is what bulk descent wants; a
large one gives a precise gradient, which is what final convergence wants.

Checkpoints are written only when a COMPLETE panel evaluation improves on the
best complete evaluation so far, so a policy that leans on physical recourse can
never become the selected one.
"""
function train_strict(; case_dir::AbstractString = get(ENV, "DR_BAT_CASE_DIR",
                                                       joinpath(@__DIR__, "case", "pglib_opf_case14_ieee")),
                        num_stages::Integer = parse(Int, get(ENV, "DR_BAT_STAGES", "24")),
                        epochs::Integer = parse(Int, get(ENV, "DR_BAT_EPOCHS", "2")),
                        batches::Integer = parse(Int, get(ENV, "DR_BAT_BATCHES", "5")),
                        trajectories::Integer = parse(Int, get(ENV, "DR_BAT_TRAJ", "2")),
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
                                                head_layers = collect(Int, head_layers)))
    assert_device(policy, device)

    opt_state = Optimisers.setup(Optimisers.Adam(Float64(lr)), policy)
    e0 = initial_energy(case)
    like = _policy_array(policy)
    e0_dev = _to_like(like, e0)

    total_steps = epochs * batches
    protocol_stages = Int(case.manifest["protocol"]["num_stages"])
    protocol_scenarios = Int(case.manifest["protocol"]["num_scenarios"])
    protocol_stages >= num_stages ||
        error("the frozen protocol has $protocol_stages stages but training asks for $num_stages")

    history = NamedTuple[]
    best = (cost = Inf, step = 0)
    updates = 0
    t_start = time()

    for step in 1:total_steps
        Optimisers.adjust!(opt_state, cosine_lr(step, total_steps, lr, lr_final))
        grads = nothing
        losses = Float64[]
        for _ in 1:trajectories
            atoms = sample_atoms(rng, case, Int(num_stages))
            features = rollout_features(case, prob.net, atoms; like = like)
            targets = rollout_targets(policy, features, e0_dev)
            result, sol, λ = strict_solve!(prob, case, atoms, targets;
                                           solver_kwargs = solver_kwargs)
            if !solve_succeeded(result)
                @warn "strict solve failed; trajectory skipped" status=result.status step=step
                continue
            end
            push!(losses, sum(stage_costs(prob, sol).total))
            _, g = actor_gradient(policy, features, _to_like(like, λ), e0_dev)
            grads = grads === nothing ? g : _add_grads(grads, g)
        end
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
                                protocol_stages = protocol_stages,
                                protocol_scenarios = protocol_scenarios,
                                max_recourse = max_recourse, solver_kwargs = solver_kwargs)
            verbose && @printf("  panel: mean %14.4f  worst recourse %.3e  complete %s\n",
                               ev.mean_cost, ev.worst_recourse, ev.complete)
            if ev.complete && ev.mean_cost < best.cost
                best = (cost = ev.mean_cost, step = step)
                save_checkpoint(checkpoint, policy, Dict(
                    "case" => case.name,
                    "network_sha256" => case.manifest["artifacts"]["network.json"],
                    "num_stages" => Int(num_stages),
                    "panel_columns" => collect(Int, eval_columns),
                    "panel_mean_cost" => ev.mean_cost,
                    "step" => step,
                    "encoder_layers" => collect(Int, encoder_layers),
                    "head_layers" => collect(Int, head_layers),
                    "n_observation" => prob.nBus,
                    "n_context" => N_CONTEXT,
                ))
            end
        end
    end

    return (policy = policy, problem = prob, case = case,
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

if abspath(PROGRAM_FILE) == @__FILE__
    out = train_strict()
    @printf("\n%d updates in %.1f s; best complete panel %.4f at step %d\n",
            out.updates, out.elapsed, out.best.cost, out.best.step)
    @printf("checkpoint: %s\n", out.checkpoint)
end
