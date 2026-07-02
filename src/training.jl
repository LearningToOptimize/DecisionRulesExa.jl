# training.jl
#
# TS-DDR policy gradient training. Mirrors train_multistage from DecisionRules.jl.
#
# Algorithm (one iteration of train_tsddr):
#   1. uncertainty_sampler() → flat w  (length T × nw_per_stage)
#   2. Policy rollout: x̂_t = policy(vcat(w_t, x̂_{t-1}))  for t = 1..T
#   3. ExaModels.set_parameter! for x0, uncertainty, targets → MadNLP.solve!
#   4. λ = result.multipliers[target_con_range]   (∇_{x̂} Q, envelope theorem)
#   5. Zygote: ∇_θ (1/n) Σ_s ⟨λ_s, x̂_s(θ)⟩  →  Flux.update!
#
# The user passes parameter objects (p_x0, p_target, p_uncertainty) exactly as
# train_multistage takes state_params_in / state_params_out in DecisionRules.jl.
# No custom functions or structs are required in examples.
#
# det_equivalent must expose the fields:
#   .core             ExaModels.ExaCore  (for set_parameter!)
#   .model            ExaModels.ExaModel (for MadNLP)
#   .horizon          Int                (T)
#   .target_con_range UnitRange{Int}     (slice of result.multipliers = ∇_{x̂} Q)

# ── Gradient materialization ──────────────────────────────────────────────────

"""
    _mat(x)
    _mat(x::Zygote.OneElement)
    _mat(x::ChainRulesCore.Tangent)
    _mat(x::ChainRulesCore.MutableTangent)

Recursively materialize lazy Zygote / ChainRules tangent wrappers into plain
Julia values (arrays and named tuples).

Zygote may return `OneElement` sparse arrays or `Tangent`/`MutableTangent`
wrappers instead of dense arrays or named tuples. `Flux.update!` expects
concrete data, so every tangent node must be materialized before the optimizer
step.

# Arguments
- `x`: a gradient value — may be a plain array, a `Zygote.OneElement`, a
  `ChainRulesCore.Tangent`, or a `ChainRulesCore.MutableTangent`.

# Returns
- For plain values: returns `x` unchanged.
- For `OneElement`: returns `collect(x)`, a dense array.
- For `Tangent`/`MutableTangent`: returns a `NamedTuple` with recursively
  materialized fields.
"""
_mat(x) = x                                          # plain value — no conversion needed
_mat(x::Zygote.OneElement) = collect(x)               # sparse one-hot → dense array
function _mat(x::ChainRulesCore.Tangent{<:Any})
    nt = ChainRulesCore.backing(x)                    # extract underlying named tuple
    return NamedTuple{keys(nt)}(map(_mat, values(nt)))  # recursively materialize each field
end
function _mat(x::ChainRulesCore.MutableTangent{<:Any})
    nt = ChainRulesCore.backing(x)                    # extract underlying named tuple
    return NamedTuple{keys(nt)}(map(_mat, values(nt)))  # recursively materialize each field
end

"""
    materialize_tangent(g) -> Union{Nothing, Any}

Convert a Zygote gradient `g` into plain Julia arrays and named tuples that
`Flux.update!` can consume. Returns `nothing` when `g` is `nothing` (i.e., no
gradient was produced for that parameter).

This is the public entry point; internally it delegates to [`_mat`](@ref).

# Arguments
- `g`: raw gradient returned by `Zygote.gradient`; may be `nothing`.

# Returns
- `nothing` if `g` is `nothing`.
- A materialized gradient (dense arrays / named tuples) otherwise.

# Examples
```julia
gs = Zygote.gradient(model) do m
    sum(m(x))
end
grad = materialize_tangent(gs[1])   # NamedTuple or nothing
```
"""
materialize_tangent(g) = isnothing(g) ? nothing : _mat(g)  # guard against nothing gradients

"""
    _all_finite_gradient(x) -> Bool

Recursively check that every element of a (possibly nested) gradient structure
is finite (no `NaN` or `±Inf`).

After BPTT through physics-based dynamics, accumulated Jacobian products can
overflow `Float32` range (> 3.4e38), producing `Inf` or `NaN` values that
would corrupt the Adam optimizer state. This guard prevents
`Flux.update!` from being called with non-finite gradients.

# Arguments
- `x`: gradient value — may be an `AbstractArray`, `Number`, `Nothing`,
  `NamedTuple`, `Tuple`, or any other type.

# Returns
- `true` if every numeric leaf is finite (or the value is `nothing` / an
  unrecognized type that carries no numeric data).
- `false` if any leaf contains `NaN` or `±Inf`.

# Examples
```julia
_all_finite_gradient([1.0, 2.0])     # true
_all_finite_gradient([1.0, NaN])     # false
_all_finite_gradient((a=[1.0], b=Inf))  # false
_all_finite_gradient(nothing)        # true
```
"""
_all_finite_gradient(x::AbstractArray) = all(isfinite, x)   # check every element
_all_finite_gradient(x::Number)        = isfinite(x)        # scalar check
_all_finite_gradient(x::Nothing)       = true               # nothing → vacuously finite
_all_finite_gradient(x::NamedTuple)    = all(_all_finite_gradient(v) for v in values(x))  # recurse named tuple fields
_all_finite_gradient(x::Tuple)         = all(_all_finite_gradient(v) for v in x)          # recurse tuple elements
_all_finite_gradient(x)                = true               # fallback: assume finite for unknown types

"""
    _status_key(status) -> String

Convert a MadNLP solver status enum value into a safe string key by replacing
non-alphanumeric characters with underscores. Used to build human-readable
diagnostic dictionaries keyed by solver outcome.

# Arguments
- `status`: a MadNLP status enum (e.g., `MadNLP.SOLVE_SUCCEEDED`).

# Returns
- A sanitized `String` suitable for use as a dictionary key.

# Examples
```julia
_status_key(MadNLP.SOLVE_SUCCEEDED)  # "SOLVE_SUCCEEDED"
```
"""
_status_key(status) = replace(string(status), r"[^A-Za-z0-9_]" => "_")  # sanitize status to dict-safe key

"""
    _inc_status!(counts::Dict{String, Int}, status) -> Dict{String, Int}

Increment the counter for the given MadNLP solver `status` in `counts`.
Converts `status` to a string key via [`_status_key`](@ref) before
incrementing.

# Arguments
- `counts::Dict{String, Int}`: mutable dictionary of status counts.
- `status`: a MadNLP solver status enum.

# Returns
- The mutated `counts` dictionary.
"""
function _inc_status!(counts::Dict{String, Int}, status)
    key = _status_key(status)                  # convert enum to string key
    counts[key] = get(counts, key, 0) + 1      # increment (initialize to 0 if absent)
    return counts
end

"""
    _inc_count!(counts::Dict{String, Int}, key::String) -> Dict{String, Int}

Increment the counter for `key` in the diagnostics dictionary `counts`.
Used to track failure reasons (e.g., `"nonfinite_objective"`) and retry
outcomes during training batches.

# Arguments
- `counts::Dict{String, Int}`: mutable dictionary of event counts.
- `key::String`: the event identifier to increment.

# Returns
- The mutated `counts` dictionary.
"""
function _inc_count!(counts::Dict{String, Int}, key::String)
    counts[key] = get(counts, key, 0) + 1      # increment (initialize to 0 if absent)
    return counts
end

"""
    _adapt_array(x::AbstractVector, ref::AbstractVector) -> AbstractVector

Move `x` onto the same device (CPU or GPU) as `ref`, allocating a new array
only when the concrete types differ.

When training runs on GPU, solver outputs (multipliers, states) may be
`CuVector`s while sampled data may arrive as CPU `Vector`s (or vice versa).
This helper ensures type-homogeneous arithmetic by copying `x` into a
`similar` array derived from `ref`.

# Arguments
- `x::AbstractVector`: the source data to adapt (may be CPU or GPU).
- `ref::AbstractVector`: a reference vector whose concrete type determines the
  target device / storage backend.

# Returns
- `x` itself when `typeof(x) === typeof(ref)` (zero-copy fast path).
- A new array on the same device as `ref`, with the element type of `x` and
  the data of `x` copied in.

# Examples
```julia
using CUDA
cpu_vec = [1.0f0, 2.0f0]
gpu_ref = CUDA.zeros(Float32, 3)
gpu_vec = _adapt_array(cpu_vec, gpu_ref)  # CuVector{Float32}
```
"""
function _adapt_array(x::AbstractVector, ref::AbstractVector)
    typeof(x) === typeof(ref) && return x                       # same type → no copy needed
    copyto!(similar(ref, eltype(x), length(x)), x)              # allocate on ref's device, copy x into it
end

# ── Solve-status check ────────────────────────────────────────────────────────

"""
    solve_succeeded(result) -> Bool

Check whether a MadNLP solve result indicates a usable solution.

MadNLP returns `0.0` objective for failed or infeasible solves, so checking
`isfinite(objective)` alone is not sufficient. This function inspects the
solver status directly.

# Arguments
- `result`: a MadNLP result object with a `.status` field.

# Returns
- `true` if `result.status` is `SOLVE_SUCCEEDED` or `SOLVED_TO_ACCEPTABLE_LEVEL`.
- `false` for all other statuses (e.g., `MAXIMUM_ITERATIONS_EXCEEDED`,
  `INFEASIBLE_PROBLEM_DETECTED`).

# Examples
```julia
result = MadNLP.solve!(solver)
if solve_succeeded(result)
    @show result.objective
end
```
"""
function solve_succeeded(result)
    s = result.status                                                           # extract MadNLP status enum
    return s == MadNLP.SOLVE_SUCCEEDED || s == MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL  # accept both convergence levels
end

# ── Internal: one MadNLP solve with cascade-failure prevention ────────────────
#
# After a failed solve the duals (y, zl, zu) are corrupted.  Instead of cold-
# starting (which resets duals to 0/1 and costs many more iterations) we restore
# the last-good dual snapshot and mark the solver as SOLVE_SUCCEEDED so MadNLP's
# reinitialize!() keeps those duals while resetting only the primal.
#
# Per-solve iteration budget: MadNLP's cnt.k is CUMULATIVE across calls, so we
# reset it before each solve to give each batch a fresh max_iter budget.

"""
    _SolverState

Mutable wrapper around a MadNLP solver that caches the last successful
primal-dual snapshot for warm-start cascade-failure prevention.

After a failed solve, MadNLP's duals (`y`, `zl`, `zu`) are corrupted.
If the next call uses `reinitialize!()` (warm-start path), it keeps those
corrupted duals and the failure cascades. `_SolverState` stores the
last-good dual values so they can be restored after a failure, breaking
the cascade without paying the cost of a full cold start.

# Fields
- `solver`: the `MadNLP.MadNLPSolver` instance.
- `last_good_x`: primal snapshot from the last successful solve (CPU or GPU
  array), or `nothing` before the first success.
- `last_good_y`: equality dual snapshot, or `nothing`.
- `last_good_zl_vals`: lower-bound dual values snapshot, or `nothing`.
- `last_good_zu_vals`: upper-bound dual values snapshot, or `nothing`.
- `has_fixed_vars::Bool`: `true` when the NLP has fixed variables (via
  `MakeParameter`), which requires a fresh solver per solve to avoid stale
  KKT factorization state.
"""
mutable struct _SolverState
    solver                     # MadNLP.MadNLPSolver instance
    last_good_x                # primal snapshot (CPU or GPU array), or nothing
    last_good_y                # dual snapshot, or nothing
    last_good_zl_vals          # lower-bound dual values snapshot, or nothing
    last_good_zu_vals          # upper-bound dual values snapshot, or nothing
    has_fixed_vars::Bool       # true when fixed variables exist in the NLP
end

"""
    _make_solver(nlp, madnlp_kwargs) -> _SolverState

Construct a [`_SolverState`](@ref) wrapping a fresh `MadNLP.MadNLPSolver`.

Detects whether the NLP has fixed variables by comparing the solver's internal
variable count against the NLP's variable count. When fixed variables exist
(via `ExaModels.MakeParameter`), the solver cannot be safely reused across
parameter changes, so `has_fixed_vars` is set to `true`.

# Arguments
- `nlp`: an NLPModels-compatible problem (e.g., `ExaModel`).
- `madnlp_kwargs`: `NamedTuple` of keyword arguments forwarded to
  `MadNLP.MadNLPSolver`.

# Returns
- A fresh [`_SolverState`](@ref) with no cached primal-dual snapshot.

# Examples
```julia
state = _make_solver(det_equivalent.model, (print_level=0, tol=1e-6))
```
"""
function _make_solver(nlp, madnlp_kwargs)
    solver = MadNLP.MadNLPSolver(nlp; madnlp_kwargs...)  # build MadNLP solver with user options
    nvar_solver = length(solver.x.x)                      # internal (reduced) variable count
    nvar_nlp    = length(NLPModels.get_x0(nlp))            # NLP-level variable count
    has_fixed   = nvar_solver != nvar_nlp                  # mismatch → fixed variables present
    return _SolverState(solver, nothing, nothing, nothing, nothing, has_fixed)  # no cached duals yet
end

"""
    _solve!(state::_SolverState, nlp; warmstart::Bool, madnlp_kwargs)

Solve `nlp` using the MadNLP solver cached in `state`, with cascade-failure
prevention via dual snapshot restore.

The warm-start logic implements three paths:

1. **Fixed variables** (`state.has_fixed_vars`): create a fresh solver each
   call because MadNLP's KKT factorization becomes stale when `MakeParameter`
   changes fixed-variable values between solves.

2. **Warm start after success**: copy the last-good primal into `x0` and let
   `reinitialize!()` keep the cached duals.

3. **Warm start after failure**: restore the last-good dual snapshot
   (`y`, `zl.values`, `zu.values`) and mark the solver as `SOLVE_SUCCEEDED`
   so `reinitialize!()` keeps those restored duals rather than the corrupted
   ones. If no good snapshot exists, fall back to `INITIAL` (cold start).

MadNLP's `cnt.k` is cumulative and never reset internally, so we reset it
before each solve to give every call a fresh `max_iter` budget.

# Arguments
- `state::_SolverState`: solver wrapper with optional cached primal-dual
  snapshot.
- `nlp`: the NLPModels-compatible problem to solve.
- `warmstart::Bool`: `true` to reuse duals from prior solves, `false` to
  cold-start.
- `madnlp_kwargs`: `NamedTuple` forwarded to `MadNLP.solve!`.

# Returns
- A MadNLP result object with fields `.status`, `.objective`,
  `.multipliers`, `.solution`.

# Examples
```julia
state = _make_solver(nlp, (print_level=0,))
result = _solve!(state, nlp; warmstart=true, madnlp_kwargs=(print_level=0,))
```
"""
function _solve!(state::_SolverState, nlp; warmstart::Bool, madnlp_kwargs)
    solver = state.solver                                  # cached MadNLP solver instance

    # MadNLP solver reuse with MakeParameter (fixed variables) causes INFEASIBLE
    # on subsequent solves even with INITIAL status — stale KKT factorization state.
    # Fix: create a fresh solver each time when fixed variables exist.
    if state.has_fixed_vars
        return MadNLP.madnlp(nlp; madnlp_kwargs...)        # one-shot fresh solver
    end

    # Normal path (no fixed variables): full warm-start support.
    if warmstart && state.last_good_x !== nothing
        copyto!(NLPModels.get_x0(nlp), state.last_good_x)  # seed primal with last-good solution
    end

    prev_result = solver.status                             # check previous solve outcome
    prev_failed = (prev_result != MadNLP.INITIAL &&         # any non-success, non-initial status
                   prev_result != MadNLP.SOLVE_SUCCEEDED &&   # means the duals may be corrupted
                   prev_result != MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL)

    if !warmstart
        solver.status = MadNLP.INITIAL                      # cold start: reset x, y, zl, zu
    elseif prev_failed && state.last_good_y !== nothing
        solver.y         .= state.last_good_y               # restore last-good equality duals
        solver.zl.values .= state.last_good_zl_vals         # restore last-good lower-bound duals
        solver.zu.values .= state.last_good_zu_vals         # restore last-good upper-bound duals
        solver.status     = MadNLP.SOLVE_SUCCEEDED          # trick reinitialize!() into warm path
    elseif prev_failed
        solver.status = MadNLP.INITIAL                      # no snapshot available → cold start
    end

    # Reset per-solve iteration budget (cnt.k is cumulative in MadNLP).
    solver.cnt.k              = 0                           # reset iteration counter
    solver.cnt.acceptable_cnt = 0                           # reset acceptable-step counter
    solver.cnt.start_time     = time()                      # reset wall-clock timer

    res = MadNLP.solve!(solver; madnlp_kwargs...)           # run the solver

    if solve_succeeded(res)
        state.last_good_x       = copy(solver.x.x)         # snapshot primal (GPU-safe copy)
        state.last_good_y       = copy(solver.y)            # snapshot equality duals
        state.last_good_zl_vals = copy(solver.zl.values)    # snapshot lower-bound duals
        state.last_good_zu_vals = copy(solver.zu.values)    # snapshot upper-bound duals
    end
    return res
end

"""
    _solve_with_retry!(state::_SolverState, nlp;
                       warmstart::Bool, madnlp_kwargs, retry_on_failure::Bool)
        -> (result, retried::Bool)

Solve `nlp` via [`_solve!`](@ref), optionally retrying with a fresh cold-start
solver if the first attempt fails or returns a non-finite objective.

The retry creates a brand-new [`_SolverState`](@ref) (discarding any corrupted
internal state) and solves with `warmstart=false`. This is more expensive than
the dual-restore path in `_solve!` but guarantees a clean factorization.

# Arguments
- `state::_SolverState`: primary solver state (may have cached duals).
- `nlp`: the NLPModels-compatible problem.
- `warmstart::Bool`: whether the first attempt should warm-start.
- `madnlp_kwargs`: `NamedTuple` forwarded to `MadNLP.solve!`.
- `retry_on_failure::Bool`: if `true`, retry with a fresh solver on failure.

# Returns
- `result`: the MadNLP result from whichever attempt succeeded (or the retry
  result if both failed).
- `retried::Bool`: `true` if the retry path was taken.

# Examples
```julia
result, retried = _solve_with_retry!(
    state, nlp;
    warmstart=true, madnlp_kwargs=(print_level=0,), retry_on_failure=true,
)
retried && @warn "solve required retry"
```
"""
function _solve_with_retry!(state::_SolverState, nlp; warmstart::Bool, madnlp_kwargs, retry_on_failure::Bool)
    result = _solve!(state, nlp; warmstart = warmstart, madnlp_kwargs = madnlp_kwargs)  # primary attempt
    retried = false                                                                      # track whether retry was needed
    if retry_on_failure && (!solve_succeeded(result) || !isfinite(result.objective))
        retry_state = _make_solver(nlp, madnlp_kwargs)                                   # fresh solver, clean factorization
        result = _solve!(retry_state, nlp; warmstart = false, madnlp_kwargs = madnlp_kwargs)  # cold-start retry
        retried = true
    end
    return result, retried
end

# ── simulate_tsddr ────────────────────────────────────────────────────────────

"""
    simulate_tsddr(model, initial_state, det_equivalent,
                   p_x0, p_target, p_uncertainty,
                   uncertainty_sampler;
                   madnlp_kwargs, warmstart)
        -> NamedTuple{(:objective, :lambda)} or nothing

Perform a single forward pass of the TS-DDR pipeline without a gradient
update: roll out the policy to produce target states, solve the
deterministic-equivalent NLP, and extract the envelope-theorem multipliers.

The forward pass computes

```math
\\hat{x}_t = \\pi_\\theta(w_t, \\hat{x}_{t-1}), \\quad t = 1, \\ldots, T,
```

then solves ``\\min_z Q(z; \\hat{x}, w, x_0)`` and returns the objective value
and the multipliers ``\\lambda = \\nabla_{\\hat{x}} Q`` from the target
equality constraints.

# Arguments
- `model`: Flux policy network (LSTM or MLP).
- `initial_state::AbstractVector`: initial state vector ``x_0``.
- `det_equivalent`: ExaModels NLP with `.core`, `.model`, `.horizon`,
  `.target_con_range`.
- `p_x0`: ExaModels parameter handle for the initial state.
- `p_target`: ExaModels parameter handle for policy targets.
- `p_uncertainty`: ExaModels parameter handle for per-stage uncertainty.
- `uncertainty_sampler`: `() -> w_flat` returning a flat vector of length
  ``T \\times n_w``.

# Keywords
- `madnlp_kwargs`: `NamedTuple` forwarded to MadNLP (default `NamedTuple()`).
- `warmstart::Bool`: warm-start MadNLP (default `true`).

# Returns
- A `NamedTuple` with fields `objective::Float64` and `lambda::Vector{F}`,
  or `nothing` if the solve failed or the objective is non-finite.

# Examples
```julia
result = simulate_tsddr(model, x0, de, p_x0, p_target, p_unc, sampler)
if result !== nothing
    @show result.objective
end
```
"""
function simulate_tsddr(
    model,
    initial_state::AbstractVector,
    det_equivalent,
    p_x0,
    p_target,
    p_uncertainty,
    uncertainty_sampler;
    madnlp_kwargs   = NamedTuple(),
    warmstart::Bool = true,
)
    T    = det_equivalent.horizon            # number of planning stages
    F    = eltype(initial_state)             # element type (Float32 or Float64)
    nx   = length(initial_state)             # state dimension
    core = det_equivalent.core               # ExaModels core (for set_parameter!)
    nlp  = det_equivalent.model              # ExaModels NLP model (for MadNLP)

    state = _make_solver(nlp, madnlp_kwargs)  # fresh solver — no warm-start cache

    # Sample one uncertainty scenario and move to the correct device.
    w_flat = uncertainty_sampler()            # flat vector of length T * nw_per_stage
    nw     = length(w_flat) ÷ T              # uncertainty dimension per stage
    w_dev  = _adapt_array(F.(w_flat), initial_state)  # move to GPU if initial_state is on GPU

    # Roll out the policy to produce target states (outside AD tape).
    Flux.reset!(model)                       # reset LSTM hidden state
    xhat_stages = Vector{AbstractVector{F}}(undef, T)  # allocate per-stage target storage
    prev = initial_state                     # first policy input is x0
    for t in 1:T
        wt   = view(w_dev, (t-1)*nw+1 : t*nw)  # slice uncertainty for stage t
        xhat_stages[t] = model(vcat(wt, prev))  # policy: [w_t; x̂_{t-1}] → x̂_t
        prev = xhat_stages[t]                    # feed x̂_t to next stage
    end
    xhat_flat = vcat(xhat_stages...)         # flatten targets to a single vector

    # Set NLP parameters: initial state, uncertainty, and policy targets.
    ExaModels.set_parameter!(core, p_x0,          initial_state)
    ExaModels.set_parameter!(core, p_uncertainty,  w_flat)
    ExaModels.set_parameter!(core, p_target,       Float64.(xhat_flat))  # NLP uses Float64

    # Solve the deterministic equivalent (cold start for one-shot simulation).
    result = _solve!(state, nlp; warmstart = false, madnlp_kwargs = madnlp_kwargs)

    # Reject failed or non-finite solves.
    solve_succeeded(result) || return nothing
    isfinite(result.objective) || return nothing

    # Extract envelope-theorem multipliers λ = ∇_{x̂} Q from target constraints.
    λ = result.multipliers[det_equivalent.target_con_range]
    return (objective = result.objective, lambda = F.(λ))  # cast λ to match initial_state eltype
end

"""
    _rollout_xhat_flat(model, initial_state, w_flat, T::Int, F) -> AbstractVector{F}

Roll out the policy network over `T` stages and return the concatenated
target trajectory as a single flat vector.

This is the differentiable inner loop used inside `Zygote.gradient` blocks.
Each stage evaluates

```math
\\hat{x}_t = \\pi_\\theta([w_t; \\hat{x}_{t-1}]), \\quad t = 1, \\ldots, T,
```

and the returned vector is ``[\\hat{x}_1; \\hat{x}_2; \\ldots; \\hat{x}_T]``.

# Arguments
- `model`: Flux policy (LSTM or MLP).
- `initial_state`: state vector ``x_0`` fed to the first policy call.
- `w_flat`: flat uncertainty vector of length ``T \\times n_w``.
- `T::Int`: number of planning stages.
- `F`: element type (e.g., `Float32`).

# Returns
- A flat vector of length ``T \\times n_x`` containing all stage targets.
"""
function _rollout_xhat_flat(model, initial_state, w_flat, T::Int, F)
    nw = length(w_flat) ÷ T                           # uncertainty dimension per stage
    Flux.reset!(model)                                 # reset LSTM hidden state
    prev = F.(initial_state)                           # cast initial state to element type F
    stages = Vector{typeof(prev)}(undef, T)            # pre-allocate per-stage output vector
    for t in 1:T
        wt = view(w_flat, (t-1)*nw+1 : t*nw)          # slice uncertainty for stage t
        stages[t] = model(vcat(wt, prev))              # policy forward pass
        prev = stages[t]                               # feed target to next stage
    end
    return vcat(stages...)                             # flatten to single vector
end

"""
    _has_critic(control_variate::AbstractCriticControlVariate) -> Bool

Return `true` if the control variate wraps an actual critic network, `false`
for the no-op [`NoCriticControlVariate`](@ref).

# Arguments
- `control_variate`: an [`AbstractCriticControlVariate`](@ref) instance.

# Returns
- `false` for `NoCriticControlVariate` (recovers the original dual-only update).
- `true` for any concrete critic (e.g., [`ScalarCriticControlVariate`](@ref)).
"""
_has_critic(::NoCriticControlVariate) = false            # no-op sentinel → no critic
_has_critic(::AbstractCriticControlVariate) = true       # any concrete critic → active

function _validate_critic_training_args(;
    actor_gradient_mode,
    critic_cv_weight,
    dual_actor_weight,
    critic_actor_weight,
    critic_updates_per_batch,
    critic_buffer_size,
    critic_rollout_samples_per_batch,
    num_cheap_critic_samples_per_batch,
)
    actor_gradient_mode in (:control_variate, :surrogate) ||
        error("actor_gradient_mode must be :control_variate or :surrogate")
    critic_cv_weight >= 0 || error("critic_cv_weight must be nonnegative")
    dual_actor_weight >= 0 || error("dual_actor_weight must be nonnegative")
    critic_actor_weight >= 0 || error("critic_actor_weight must be nonnegative")
    critic_updates_per_batch >= 0 || error("critic_updates_per_batch must be nonnegative")
    critic_buffer_size >= 0 || error("critic_buffer_size must be nonnegative")
    if critic_rollout_samples_per_batch !== nothing
        critic_rollout_samples_per_batch >= 0 ||
            error("critic_rollout_samples_per_batch must be nonnegative or nothing")
    end
    num_cheap_critic_samples_per_batch >= 0 ||
        error("num_cheap_critic_samples_per_batch must be nonnegative")
    return true
end

function _resolve_critic_training_target(target, has_critic::Bool)
    has_critic || return DeterministicEquivalentCriticTarget()
    target isa AbstractCriticTrainingTarget && return target
    if target === :deterministic_equivalent || target === :de
        return DeterministicEquivalentCriticTarget()
    elseif target === :rollout
        error("critic_training_target=:rollout requires a RolloutCriticTarget(...) configuration")
    else
        error("critic_training_target must be RolloutCriticTarget(...), DeterministicEquivalentCriticTarget(), :rollout, or :deterministic_equivalent")
    end
end

function _critic_sample_from_rollout(
    model,
    initial_state,
    target::RolloutCriticTarget,
    w_flat,
    lambda,
    F,
    solver_state,
)
    # Keep both rollout objective variants available; target.objective_value
    # below selects which one is used as the critic value target.
    rollout_len = target.horizon * target.n_uncertainty
    length(w_flat) >= rollout_len ||
        error("rollout critic uncertainty has length $(length(w_flat)); expected at least $rollout_len")
    w_rollout = view(w_flat, 1:rollout_len)
    result = rollout_tsddr(
        model,
        initial_state,
        target.stage_problem,
        w_rollout;
        horizon = target.horizon,
        n_uncertainty = target.n_uncertainty,
        set_stage_parameters! = target.set_stage_parameters!,
        realized_state = target.realized_state,
        objective_no_target_penalty = target.objective_no_target_penalty,
        madnlp_kwargs = target.madnlp_kwargs,
        warmstart = target.warmstart,
        policy_state = target.policy_state,
        solver_state = solver_state,
        reuse_solver = target.reuse_solver,
        state_bounds = target.state_bounds,
        project_state = target.project_state,
        retry_on_failure = target.retry_on_failure,
    )
    result === nothing && return nothing

    objective = target.objective_value === :objective ?
        result.objective : result.objective_no_target_penalty
    xhat_flat = F.(vcat(result.target_trajectory...))
    λ_rollout = view(lambda, 1:length(xhat_flat))
    return CriticSample(F.(initial_state), F.(w_rollout), xhat_flat, objective, F.(λ_rollout))
end

function _rollout_critic_samples(
    model,
    initial_state,
    target::RolloutCriticTarget,
    de_samples,
    F,
    max_samples,
    solver_state,
)
    isempty(de_samples) && return CriticSample[]
    n = max_samples === nothing ? length(de_samples) : min(Int(max_samples), length(de_samples))
    n == 0 && return CriticSample[]
    idx = n == length(de_samples) ? eachindex(de_samples) : randperm(length(de_samples))[1:n]
    samples = CriticSample[]
    for i in idx
        s = de_samples[i]
        sample = _critic_sample_from_rollout(
            model,
            initial_state,
            target,
            s.uncertainty,
            s.target_multipliers,
            F,
            solver_state,
        )
        sample === nothing && continue
        push!(samples, sample)
    end
    return samples
end

# ── train_tsddr ───────────────────────────────────────────────────────────────

"""
    train_tsddr(model, initial_state, det_equivalent,
                p_x0, p_target, p_uncertainty,
                uncertainty_sampler;
                num_batches, num_train_per_batch, optimizer,
                adjust_hyperparameters, record_loss,
                madnlp_kwargs, warmstart,
                problem_pool) -> model

TS-DDR policy gradient training. Mirrors `train_multistage` from DecisionRules.jl.

Arguments:
- `model`              : Flux policy (LSTM or MLP)
- `initial_state`      : initial state vector
- `det_equivalent`     : any ExaModels NLP with fields `.core`, `.model`,
                         `.horizon`, `.target_con_range`
- `p_x0`               : ExaModels parameter for the initial state
- `p_target`           : ExaModels parameter for policy targets
- `p_uncertainty`      : ExaModels parameter for per-stage uncertainty
- `uncertainty_sampler`: `() -> w_flat` — flat vector of length `T * nw_per_stage`.
                         For multi-unit problems (e.g., hydro reservoirs) the sampler
                         should draw one joint scenario index per stage to preserve
                         spatial correlation; see `sample_scenario` in examples.

Keyword arguments (mirror `train_multistage`):
- `num_batches`             : total gradient steps (default 100)
- `num_train_per_batch`     : scenarios averaged per step (default 1)
- `optimizer`               : Flux.Optimisers optimizer
- `adjust_hyperparameters`  : `(iter, opt_state, n) -> n`
- `record_loss`             : `(iter, model, loss, tag) -> Bool`; return `true` to stop
- `madnlp_kwargs`           : NamedTuple forwarded to MadNLP
- `warmstart`               : warm-start MadNLP between solves (default `true`)
- `problem_pool`            : vector of `(de, p_x0, p_target, p_uncertainty)` tuples
                              for parallel GPU solves; each entry gets its own MadNLP solver
                              and samples are distributed round-robin across the pool
- `control_variate`         : optional `ScalarCriticControlVariate`; default
                              `NoCriticControlVariate()` recovers the original update
- `critic_training_target`  : `RolloutCriticTarget(...)` for rollout-objective
                              critic fitting, or `DeterministicEquivalentCriticTarget()`
                              / `:deterministic_equivalent` for DE ablations
- `critic_rollout_samples_per_batch`: number of solved batch scenarios to rerun
                              through stage-wise rollout for critic targets;
                              `nothing` uses all successful solved scenarios
- `actor_gradient_mode`     : `:control_variate` or `:surrogate`
- `num_cheap_critic_samples_per_batch`: extra policy rollouts used only for
                              critic actor terms; these do not trigger NLP solves
- `external_critic_samples`  : mutable vector; `record_loss` can push
                              `CriticSample`s (e.g. from `critic_samples_from_evaluation`)
                              to feed the critic replay buffer without extra solves
"""
function train_tsddr(
    model,
    initial_state::AbstractVector,
    det_equivalent,
    p_x0,
    p_target,
    p_uncertainty,
    uncertainty_sampler;
    num_batches::Int         = 100,
    num_train_per_batch::Int = 1,
    optimizer                = Flux.Optimisers.OptimiserChain(
                                   Flux.Optimisers.ClipGrad(1.0f0),
                                   Flux.Adam(1f-3),
                               ),
    adjust_hyperparameters   = (iter, opt_state, n) -> n,
    record_loss              = (iter, model, loss, tag) -> begin
                                   println("$tag  iter=$iter  loss=$(round(loss; digits=4))")
                                   return false
                               end,
    madnlp_kwargs            = NamedTuple(),
    warmstart::Bool          = true,
    retry_on_failure::Bool   = true,
    problem_pool             = nothing,
    control_variate::AbstractCriticControlVariate = NoCriticControlVariate(),
    actor_gradient_mode::Symbol = :control_variate,
    critic_cv_weight::Real   = 1.0,
    dual_actor_weight::Real  = 1.0,
    critic_actor_weight::Real = 1.0,
    critic_updates_per_batch::Int = 1,
    critic_buffer_size::Int  = 0,
    critic_batch_size        = nothing,
    critic_training_target   = :rollout,
    critic_rollout_samples_per_batch = nothing,
    num_cheap_critic_samples_per_batch::Int = 0,
    critic_optimizer         = Flux.Adam(1f-3),
    external_critic_samples  = nothing,
    batch_diagnostics        = (iter, stats) -> nothing,
)
    T    = det_equivalent.horizon
    F    = eltype(initial_state)
    nx   = length(initial_state)

    _validate_critic_training_args(
        actor_gradient_mode = actor_gradient_mode,
        critic_cv_weight = critic_cv_weight,
        dual_actor_weight = dual_actor_weight,
        critic_actor_weight = critic_actor_weight,
        critic_updates_per_batch = critic_updates_per_batch,
        critic_buffer_size = critic_buffer_size,
        critic_rollout_samples_per_batch = critic_rollout_samples_per_batch,
        num_cheap_critic_samples_per_batch = num_cheap_critic_samples_per_batch,
    )
    has_critic = _has_critic(control_variate)
    resolved_critic_training_target = _resolve_critic_training_target(
        critic_training_target,
        has_critic,
    )

    # ── Build worker pool ────────────────────────────────────────────────────
    if problem_pool === nothing
        _pool = [(det_equivalent, p_x0, p_target, p_uncertainty)]
    else
        _pool = problem_pool
    end
    nworkers = length(_pool)

    # Single-worker: create solver on main task (no threading needed)
    single_state = nworkers == 1 ? _make_solver(_pool[1][1].model, madnlp_kwargs) : nothing

    # Multi-worker: persistent worker threads via channels.
    # Each worker creates its own MadNLP solver on its own thread so that
    # CUDA handles (CUBLAS, CUSPARSE, CUDSS) bind to that thread's stream.
    in_channels  = nworkers > 1 ? [Channel{Any}(1) for _ in 1:nworkers] : Channel{Any}[]
    out_channels = nworkers > 1 ? [Channel{Any}(1) for _ in 1:nworkers] : Channel{Any}[]
    worker_tasks = Task[]
    if nworkers > 1
        for wi in 1:nworkers
            (de, px, pt, pu) = _pool[wi]
            in_ch  = in_channels[wi]
            out_ch = out_channels[wi]
            t = Threads.@spawn begin
                st = _make_solver(de.model, madnlp_kwargs)
                while true
                    msg = take!(in_ch)
                    msg === nothing && break
                    (s_idx, init_state, w_flat, xhat_flat) = msg
                    ExaModels.set_parameter!(de.core, px, init_state)
                    ExaModels.set_parameter!(de.core, pu, w_flat)
                    ExaModels.set_parameter!(de.core, pt, Float64.(xhat_flat))
                    result, retried = _solve_with_retry!(
                        st,
                        de.model;
                        warmstart = warmstart,
                        madnlp_kwargs = madnlp_kwargs,
                        retry_on_failure = retry_on_failure,
                    )
                    failure = nothing
                    if !solve_succeeded(result)
                        failure = "status_" * _status_key(result.status)
                    elseif !isfinite(result.objective)
                        failure = "nonfinite_objective"
                    elseif solve_succeeded(result) && isfinite(result.objective)
                        λ = result.multipliers[de.target_con_range]
                        if all(isfinite, λ)
                            put!(out_ch, (s_idx, F.(w_flat), _adapt_array(F.(λ), w_flat),
                                          result.objective, result.status, nothing, retried))
                            continue
                        end
                        failure = "nonfinite_lambda"
                    end
                    put!(out_ch, (s_idx, nothing, nothing, NaN, result.status, failure, retried))
                end
            end
            push!(worker_tasks, t)
        end
    end

    opt_state = Flux.setup(optimizer, model)
    critic_opt_state = has_critic ? Flux.setup(critic_optimizer, control_variate.critic) : nothing
    critic_buffer = CriticReplayBuffer(critic_buffer_size)
    critic_rollout_solver_state = resolved_critic_training_target isa RolloutCriticTarget &&
        resolved_critic_training_target.reuse_solver ?
        _make_solver(resolved_critic_training_target.stage_problem.model,
                     resolved_critic_training_target.madnlp_kwargs) : nothing

    try

    for iter in 1:num_batches
        num_train_per_batch = adjust_hyperparameters(iter, opt_state, num_train_per_batch)

        # ── Forward pass: rollout + solve (outside AD tape) ───────────────────

        # Step 1: Roll out policy for all samples
        sample_data = Vector{Tuple{AbstractVector{F}, AbstractVector{F}}}(undef, num_train_per_batch)
        for s in 1:num_train_per_batch
            w_flat = uncertainty_sampler()
            nw     = length(w_flat) ÷ T
            w_dev  = _adapt_array(F.(w_flat), initial_state)
            Flux.reset!(model)
            xhat_stages = Vector{AbstractVector{F}}(undef, T)
            prev = initial_state
            for t in 1:T
                wt   = view(w_dev, (t-1)*nw+1 : t*nw)
                xhat_stages[t] = model(vcat(wt, prev))
                prev = xhat_stages[t]
            end
            sample_data[s] = (w_dev, vcat(xhat_stages...))
        end

        # Step 2: Solve — parallel across workers if pool provided
        solve_ok  = Vector{Union{Nothing, Tuple{AbstractVector{F}, AbstractVector{F}, Float64}}}(nothing, num_train_per_batch)
        status_counts = Dict{String, Int}()
        failure_counts = Dict{String, Int}()
        retry_counts = Dict{String, Int}()

        if nworkers == 1
            (de, px, pt, pu) = _pool[1]
            st = single_state
            for s in 1:num_train_per_batch
                w_flat, xhat_flat = sample_data[s]
                ExaModels.set_parameter!(de.core, px, initial_state)
                ExaModels.set_parameter!(de.core, pu, w_flat)
                ExaModels.set_parameter!(de.core, pt, Float64.(xhat_flat))
                result, retried = _solve_with_retry!(
                    st,
                    de.model;
                    warmstart = warmstart,
                    madnlp_kwargs = madnlp_kwargs,
                    retry_on_failure = retry_on_failure,
                )
                retried && _inc_count!(retry_counts, solve_succeeded(result) && isfinite(result.objective) ? "retry_success" : "retry_failure")
                _inc_status!(status_counts, result.status)
                if !solve_succeeded(result)
                    _inc_count!(failure_counts, "status_" * _status_key(result.status))
                    continue
                end
                if !isfinite(result.objective)
                    _inc_count!(failure_counts, "nonfinite_objective")
                    continue
                end
                λ = result.multipliers[de.target_con_range]
                if !all(isfinite, λ)
                    _inc_count!(failure_counts, "nonfinite_lambda")
                    continue
                end
                solve_ok[s] = (F.(w_flat), _adapt_array(F.(λ), initial_state), result.objective)
            end
        else
            for round_start in 1:nworkers:num_train_per_batch
                round_end = min(round_start + nworkers - 1, num_train_per_batch)
                round_size = round_end - round_start + 1
                for s in round_start:round_end
                    wi = s - round_start + 1
                    w_flat, xhat_flat = sample_data[s]
                    put!(in_channels[wi], (s, initial_state, w_flat, xhat_flat))
                end
                for wi in 1:round_size
                    msg = take!(out_channels[wi])
                    if length(msg) == 7
                        (s_idx, w_out, λ_out, obj_out, status_out, failure_out, retried_out) = msg
                        _inc_status!(status_counts, status_out)
                        failure_out !== nothing && _inc_count!(failure_counts, failure_out)
                        retried_out && _inc_count!(retry_counts, w_out === nothing ? "retry_failure" : "retry_success")
                    elseif length(msg) == 6
                        (s_idx, w_out, λ_out, obj_out, status_out, failure_out) = msg
                        _inc_status!(status_counts, status_out)
                        failure_out !== nothing && _inc_count!(failure_counts, failure_out)
                    else
                        (s_idx, w_out, λ_out, obj_out) = msg
                    end
                    if w_out !== nothing
                        solve_ok[s_idx] = (w_out, λ_out, obj_out)
                    end
                end
            end
        end

        # Step 3: Collect valid results
        valid   = Tuple{AbstractVector{F}, AbstractVector{F}}[]
        de_samples = CriticSample[]
        obj_sum = 0.0
        for (s, r) in enumerate(solve_ok)
            r === nothing && continue
            push!(valid, (r[1], r[2]))
            if has_critic
                _, xhat_flat = sample_data[s]
                push!(de_samples, CriticSample(F.(initial_state), r[1], xhat_flat, r[3], r[2]))
            end
            obj_sum += r[3]
        end
        n_ok     = length(valid)
        mean_obj = n_ok > 0 ? obj_sum / n_ok : NaN
        batch_diagnostics(iter, Dict{String, Any}(
            "n_ok" => n_ok,
            "n_total" => num_train_per_batch,
            "status_counts" => copy(status_counts),
            "failure_counts" => copy(failure_counts),
            "retry_counts" => copy(retry_counts),
        ))

        if has_critic && n_ok > 0 && critic_updates_per_batch > 0
            valid_samples = if resolved_critic_training_target isa RolloutCriticTarget
                _rollout_critic_samples(
                    model,
                    initial_state,
                    resolved_critic_training_target,
                    de_samples,
                    F,
                    critic_rollout_samples_per_batch,
                    critic_rollout_solver_state,
                )
            else
                de_samples
            end
            if external_critic_samples !== nothing && !isempty(external_critic_samples)
                merged = Any[]
                append!(merged, valid_samples)
                append!(merged, external_critic_samples)
                empty!(external_critic_samples)
                valid_samples = merged
            end
            if critic_buffer_size > 0
                push_critic_samples!(critic_buffer, valid_samples)
                critic_samples = critic_buffer.samples
            else
                critic_samples = valid_samples
            end
            for _ in 1:critic_updates_per_batch
                update_critic!(
                    critic_opt_state,
                    control_variate,
                    critic_samples;
                    batch_size = critic_batch_size,
                )
            end
        end

        # ── Gradient: ∇_θ (1/n) Σ_s ⟨λ_s, rollout_s(θ)⟩ ────────────────────
        if n_ok > 0
            if !has_critic
                gs = Zygote.gradient(model) do m
                    total = zero(F)
                    for (w_flat_s, λf) in valid
                        nw = length(w_flat_s) ÷ T
                        Flux.reset!(m)
                        prev_ad = initial_state
                        for t in 1:T
                            wt      = view(w_flat_s, (t-1)*nw+1 : t*nw)
                            xt      = m(vcat(wt, prev_ad))
                            total   = total + sum(view(λf, (t-1)*nx+1 : t*nx) .* xt)
                            prev_ad = xt
                        end
                    end
                    total / F(n_ok)
                end
            else
                solved_weights = Tuple{AbstractVector{F}, AbstractVector{F}}[]
                for sample in de_samples
                    λf = F.(sample.target_multipliers)
                    if actor_gradient_mode === :control_variate
                        if critic_cv_weight == 0
                            actor_weight = λf
                        else
                            gx = F.(critic_xhat_gradient(
                                control_variate,
                                sample.initial_state,
                                sample.uncertainty,
                                sample.xhat,
                            ))
                            _check_critic_sample_shapes(sample, gx)
                            actor_weight = λf .- F(critic_cv_weight) .* gx
                        end
                    else
                        actor_weight = F(dual_actor_weight) .* λf
                    end
                    push!(solved_weights, (F.(sample.uncertainty), actor_weight))
                end

                critic_uncertainties = if num_cheap_critic_samples_per_batch > 0
                    [F.(uncertainty_sampler()) for _ in 1:num_cheap_critic_samples_per_batch]
                else
                    [F.(sample.uncertainty) for sample in de_samples]
                end

                gs = Zygote.gradient(model) do m
                    residual_total = zero(F)
                    for (w_flat_s, actor_weight) in solved_weights
                        nw = length(w_flat_s) ÷ T
                        Flux.reset!(m)
                        prev_ad = initial_state
                        for t in 1:T
                            wt      = view(w_flat_s, (t-1)*nw+1 : t*nw)
                            xt      = m(vcat(wt, prev_ad))
                            residual_total =
                                residual_total + sum(view(actor_weight, (t-1)*nx+1 : t*nx) .* xt)
                            prev_ad = xt
                        end
                    end
                    actor_loss = residual_total / F(n_ok)

                    critic_coeff = actor_gradient_mode === :control_variate ?
                        F(critic_cv_weight) : F(critic_actor_weight)
                    if critic_coeff != 0 && !isempty(critic_uncertainties)
                        critic_total = zero(actor_loss)
                        for w_flat_s in critic_uncertainties
                            xhat_ad = _rollout_xhat_flat(m, initial_state, w_flat_s, T, F)
                            critic_total = critic_total + critic_value(
                                control_variate,
                                initial_state,
                                w_flat_s,
                                xhat_ad,
                            )
                        end
                        actor_loss = actor_loss +
                            critic_coeff * critic_total / F(length(critic_uncertainties))
                    end
                    actor_loss
                end
            end

            grad = materialize_tangent(gs[1])
            if grad !== nothing && _all_finite_gradient(grad)
                Flux.update!(opt_state, model, grad)
            end
        end

        record_loss(iter, model, mean_obj, "metrics/training_loss") && break
    end

    finally
        # Shut down worker threads
        for ch in in_channels
            put!(ch, nothing)
        end
        for t in worker_tasks
            wait(t)
        end
    end

    return model
end

# ── train_tsddr_embedded ─────────────────────────────────────────────────────

"""
    train_tsddr_embedded(model, initial_state, embedded_de,
                         uncertainty_sampler; kwargs...) -> model

TS-DDR training with the policy embedded inside the NLP via VectorNonlinearOracle.

Unlike `train_tsddr`, this version:
- Does NOT roll out the policy externally to generate targets
- Solves the coupled NLP where oracle constraints evaluate π_θ inline
- Extracts closed-loop duals λ and realized states x* from the solution
- Computes ∇_θ Q = Σ_t λ_t · ∇_θ π_θ(w_t, x*_{t-1}) using realized states

Arguments:
- `model`              : Flux policy (same object captured by the oracle closures)
- `initial_state`      : initial state vector
- `embedded_de`        : `EmbeddedDeterministicEquivalentProblem`
- `uncertainty_sampler` : `() -> w_flat` — flat vector of length `T * nw_per_stage`

Keyword arguments:
- `num_batches`            : total gradient steps (default 100)
- `num_train_per_batch`    : scenarios averaged per step (default 1)
- `optimizer`              : Flux.Optimisers optimizer
- `adjust_hyperparameters` : `(iter, opt_state, n) -> n`
- `record_loss`            : `(iter, model, loss, tag) -> Bool`; return `true` to stop
- `madnlp_kwargs`          : NamedTuple forwarded to MadNLP
- `warmstart`              : warm-start MadNLP between solves (default `true`)
"""
function train_tsddr_embedded(
    model,
    initial_state::AbstractVector,
    embedded_de,
    uncertainty_sampler;
    num_batches::Int         = 100,
    num_train_per_batch::Int = 1,
    optimizer                = Flux.Optimisers.OptimiserChain(
                                   Flux.Optimisers.ClipGrad(1.0f0),
                                   Flux.Adam(1f-3),
                               ),
    adjust_hyperparameters   = (iter, opt_state, n) -> n,
    record_loss              = (iter, model, loss, tag) -> begin
                                   println("$tag  iter=$iter  loss=$(round(loss; digits=4))")
                                   return false
                               end,
    madnlp_kwargs            = NamedTuple(),
    warmstart::Bool          = true,
    retry_on_failure::Bool   = true,
    get_realized_states      = nothing,
    batch_diagnostics        = (iter, stats) -> nothing,
)
    T  = embedded_de.horizon
    F  = eltype(initial_state)
    nx = embedded_de.nx

    _get_states = get_realized_states === nothing ?
        (prob, res) -> res.solution[1 : prob.horizon * prob.nx] :
        get_realized_states

    state = _make_solver(embedded_de.model, madnlp_kwargs)
    opt_state = Flux.setup(optimizer, model)

    for iter in 1:num_batches
        num_train_per_batch = adjust_hyperparameters(iter, opt_state, num_train_per_batch)

        valid   = Tuple{AbstractVector{F}, AbstractVector{F}, AbstractVector{F}}[]
        obj_sum = 0.0
        status_counts = Dict{String, Int}()
        failure_counts = Dict{String, Int}()
        retry_counts = Dict{String, Int}()

        for s in 1:num_train_per_batch
            w_flat = uncertainty_sampler()

            set_x0!(embedded_de, initial_state)
            set_uncertainty!(embedded_de, w_flat)

            result, retried = _solve_with_retry!(
                state,
                embedded_de.model;
                warmstart = warmstart,
                madnlp_kwargs = madnlp_kwargs,
                retry_on_failure = retry_on_failure,
            )
            retried && _inc_count!(retry_counts, solve_succeeded(result) && isfinite(result.objective) ? "retry_success" : "retry_failure")

            _inc_status!(status_counts, result.status)
            if !solve_succeeded(result)
                _inc_count!(failure_counts, "status_" * _status_key(result.status))
                continue
            end
            if !isfinite(result.objective)
                _inc_count!(failure_counts, "nonfinite_objective")
                continue
            end

            λ = result.multipliers[embedded_de.target_con_range]
            if !all(isfinite, λ)
                _inc_count!(failure_counts, "nonfinite_lambda")
                continue
            end

            x_sol = _get_states(embedded_de, result)

            λf    = _adapt_array(F.(λ), initial_state)
            xf    = _adapt_array(F.(x_sol), initial_state)
            w_dev = _adapt_array(F.(w_flat), initial_state)
            push!(valid, (w_dev, λf, xf))
            obj_sum += result.objective
        end

        n_ok     = length(valid)
        mean_obj = n_ok > 0 ? obj_sum / n_ok : NaN
        batch_diagnostics(iter, Dict{String, Any}(
            "n_ok" => n_ok,
            "n_total" => num_train_per_batch,
            "status_counts" => copy(status_counts),
            "failure_counts" => copy(failure_counts),
            "retry_counts" => copy(retry_counts),
        ))

        if n_ok > 0
            gs = Zygote.gradient(model) do m
                total = zero(F)
                for (w_flat_s, λf, x_realized) in valid
                    nw = length(w_flat_s) ÷ T
                    Flux.reset!(m)
                    for t in 1:T
                        wt = view(w_flat_s, (t-1)*nw+1 : t*nw)
                        x_prev = (t == 1) ?
                            initial_state :
                            view(x_realized, (t-2)*nx+1 : (t-1)*nx)
                        xt = m(vcat(wt, x_prev))
                        total = total + sum(view(λf, (t-1)*nx+1 : t*nx) .* xt)
                    end
                end
                total / F(n_ok)
            end

            grad = materialize_tangent(gs[1])
            if grad !== nothing && _all_finite_gradient(grad)
                Flux.update!(opt_state, model, grad)
            end
        end

        record_loss(iter, model, mean_obj, "metrics/training_loss") && break
    end

    return model
end
