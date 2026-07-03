# rollout.jl
#
# Stage-wise, non-cheating policy evaluation.
#
# The deterministic-equivalent training solve sees the full horizon.  This file
# provides a small callback-driven rollout loop that evaluates deployment
# semantics instead: at each stage, solve one stage, extract the realized next
# state, and feed that state into the next policy call.

"""
    _target_violation_share(objective::Real, objective_no_target_penalty::Real) -> Float64

Compute the fraction of a stage objective attributable to the target-tracking
penalty.

# Arguments
- `objective::Real`: total stage objective (operational cost + target penalty).
- `objective_no_target_penalty::Real`: stage objective with target penalty
  stripped out.

# Returns
- `Float64`: fraction in ``[0, 1]``, or `NaN` if the ratio is ill-defined.

# Notes
If the total objective includes both operational cost and a quadratic penalty
``\\lambda \\|x_t - \\hat{x}_t\\|^2``, this function returns

```math
\\frac{\\text{objective} - \\text{objective\\_no\\_target\\_penalty}}{\\text{objective}}.
```

The return value is `NaN` when the objective is non-finite, the penalty is
non-finite, or the objective magnitude is below ``10^{-12}``.
"""
function _target_violation_share(objective::Real, objective_no_target_penalty::Real)
    # The penalty is the difference between the full and penalty-free objectives.
    penalty = objective - objective_no_target_penalty
    # Guard against non-finite or near-zero denominators that would give NaN/Inf.
    (isfinite(objective) && isfinite(penalty) && abs(objective) > 1e-12) || return NaN
    # Return the penalty's share of the total objective.
    return penalty / objective
end

"""
    _to_vec(x) -> AbstractVector

Flatten `x` into a contiguous one-dimensional vector.

# Arguments
- `x`: any array-like object (matrix, vector, or view).

# Returns
- `AbstractVector`: a one-dimensional vector with the same elements as `x`.

# Notes
`SubArray` inputs are materialized with `collect` because downstream solvers
and array operations such as `copyto!` and `vcat` require contiguous storage.
All other array types are reshaped via `vec`.
"""
_to_vec(x) = vec(x)                     # reshape in-place for contiguous arrays
_to_vec(x::SubArray) = collect(x)        # materialize views to contiguous storage

"""
    _state_bound_vector(bound, ref::AbstractVector) -> Union{Nothing, AbstractVector}

Convert a user-supplied state bound into a concrete vector whose element type
and device (CPU or GPU) match `ref`.

# Arguments
- `bound`: the bound specification (`nothing`, a `Real` scalar, or an
  `AbstractVector` / `SubArray`).
- `ref::AbstractVector`: reference vector whose length, element type, and
  device placement determine the output format.

# Returns
- `Nothing` if `bound === nothing`.
- `AbstractVector` matching `ref` in length, element type, and device.

# Throws
- `ArgumentError` if a vector bound has the wrong length or `bound` is an
  unsupported type.

# Notes
Accepted bound forms are `nothing`, a scalar broadcast to all state entries,
or a vector of the same length as `ref`. `SubArray` bounds are materialized via
[`_to_vec`](@ref) before device adaptation so GPU kernels receive contiguous
storage.
"""
function _state_bound_vector(bound, ref::AbstractVector)
    # Nothing means "no bound on this side" — propagate the sentinel.
    bound === nothing && return nothing
    if bound isa AbstractVector || bound isa SubArray
        # Vector bounds must be element-wise, so lengths must agree.
        length(bound) == length(ref) ||
            throw(ArgumentError("state bound length must match state length=$(length(ref)), got $(length(bound))"))
        # Materialize, cast to ref's element type, and adapt to ref's device.
        return _adapt_array(eltype(ref).(_to_vec(bound)), ref)
    end
    # Scalar bounds are broadcast into a constant vector matching ref's shape.
    bound isa Real ||
        throw(ArgumentError("state bounds must be vectors, scalars, or nothing"))
    # Allocate on the same device as ref using `similar`.
    out = similar(ref, length(ref))
    # Fill every element with the scalar bound cast to ref's element type.
    fill!(out, eltype(ref)(bound))
    return out
end

"""
    _project_state_to_bounds(state::AbstractVector, state_bounds) -> AbstractVector

Clamp every element of `state` to lie within `[lower, upper]`.

# Arguments
- `state::AbstractVector`: realized state vector to project.
- `state_bounds`: `nothing` (no projection), or a 2-element tuple/pair
  `(lower, upper)` where each side is `nothing`, a scalar, or a vector.

# Returns
- `AbstractVector`: projected state with the same element type as `state`.

# Throws
- `ArgumentError` if `state_bounds` is not `nothing` and does not have
  exactly two elements.

# Notes
Given bounds ``l`` and ``u``, the projection is the element-wise clamp

```math
\\operatorname{proj}(x)_i = \\min\\bigl(\\max(x_i,\\, l_i),\\, u_i\\bigr).
```

This repair is intended for solver-tolerance drift at stage interfaces, not as
a substitute for a recourse-feasible model.
"""
function _project_state_to_bounds(state::AbstractVector, state_bounds)
    # No bounds means the state passes through unchanged.
    state_bounds === nothing && return state
    # Bounds must be a (lower, upper) pair.
    length(state_bounds) == 2 ||
        throw(ArgumentError("state_bounds must be a pair/tuple (lower, upper)"))
    # Convert each side into a concrete vector (or nothing) matching state.
    lower = _state_bound_vector(state_bounds[1], state)
    upper = _state_bound_vector(state_bounds[2], state)
    # Apply element-wise clamping: first lower, then upper.
    projected = state
    lower !== nothing && (projected = max.(projected, lower))  # enforce lower bound
    upper !== nothing && (projected = min.(projected, upper))  # enforce upper bound
    return projected
end

"""
    _project_realized_state(state::AbstractVector, state_bounds, project_state)
        -> AbstractVector

Repair a realized state before it is fed into the next rollout stage.

# Arguments
- `state::AbstractVector`: raw realized state from the stage solver.
- `state_bounds`: `nothing` or `(lower, upper)` passed to
  [`_project_state_to_bounds`](@ref).
- `project_state`: `nothing`, or a callable `f(state) -> projected_state`
  that enforces non-box feasibility constraints.

# Returns
- `AbstractVector`: the doubly-projected state, cast to the element type
  of the input `state`.

# Notes
The function first applies box projection via
[`_project_state_to_bounds`](@ref), then applies `project_state` when a custom
projector is supplied. If `project_state === nothing`, only the box projection
is applied.
"""
function _project_realized_state(state::AbstractVector, state_bounds, project_state)
    # First pass: box-clamp the raw realized state.
    projected = _project_state_to_bounds(state, state_bounds)
    # If no custom projector is provided, the box projection is sufficient.
    project_state === nothing && return projected
    # Second pass: apply the user's non-box projection and cast back to state's eltype.
    return eltype(state).(_to_vec(project_state(projected)))
end

"""
    rollout_tsddr(
        model,
        initial_state::AbstractVector,
        stage_problem,
        w_flat::AbstractVector;
        horizon::Int,
        n_uncertainty::Int,
        set_stage_parameters!::Function,
        realized_state::Function,
        objective_no_target_penalty::Function = (prob, result) -> result.objective,
        madnlp_kwargs = NamedTuple(),
        warmstart::Bool = true,
        policy_state::Symbol = :realized,
        solver_state = nothing,
        reuse_solver::Bool = false,
        state_bounds = nothing,
        project_state = nothing,
        retry_on_failure::Bool = true,
    ) -> Union{Nothing, NamedTuple}

Evaluate a target-setting decision rule `model` by solving `stage_problem`
sequentially over a materialized uncertainty scenario `w_flat`.

Unlike the deterministic-equivalent training solve (which sees the full
horizon simultaneously), this rollout mirrors deployment semantics: at each
stage the solver receives only one uncertainty slice, solves, extracts the
realized next state, and feeds that state into the next policy call.

The stage-wise recursion is

```math
x_0 = x_{\\text{init}}, \\quad
\\hat{x}_t = \\pi_\\theta(w_t,\\, x_{t-1}), \\quad
x_t = \\operatorname{solve}_t(x_{t-1},\\, w_t,\\, \\hat{x}_t),
```

where ``\\pi_\\theta`` is the learned policy (`model`),
``\\operatorname{solve}_t`` solves the single-stage optimization problem, and
``x_t`` is the realized state forwarded to stage ``t+1``.

# Arguments
- `model`: the target-setting policy ``\\pi_\\theta``. Called as
  `model(vcat(w_t, x_{t-1}))` to produce ``\\hat{x}_t``.
- `initial_state::AbstractVector`: initial state ``x_0``.
- `stage_problem`: stage optimization problem (must expose `.model`).
- `w_flat::AbstractVector`: flat uncertainty vector of length
  `horizon * n_uncertainty`, sliced into per-stage windows.

# Keywords
- `horizon::Int`: number of stages ``T``.
- `n_uncertainty::Int`: dimension of each per-stage uncertainty slice.
- `set_stage_parameters!::Function`: callback
  `(stage_problem, state, w_t, target, stage) -> nothing` that writes the
  current state, uncertainty, and target into the stage problem before each
  solve.
- `realized_state::Function`: callback `(stage_problem, result) -> x_t`
  that reads the realized next state from the solver result.
- `objective_no_target_penalty::Function`: callback
  `(stage_problem, result) -> Float64` returning the stage objective with
  the target-tracking penalty removed.  Defaults to `result.objective`.
- `madnlp_kwargs`: keyword arguments forwarded to the MadNLP solver
  constructor.
- `warmstart::Bool`: whether to warm-start the solver from the previous
  stage's dual solution.
- `policy_state::Symbol`: `:realized` feeds the closed-loop realized state
  ``x_t`` back to the policy; `:target` feeds the policy's own previous
  target ``\\hat{x}_t`` instead, matching deterministic-equivalent training
  semantics.
- `solver_state`: optional pre-built solver object to reuse across stages.
- `reuse_solver::Bool`: if `true`, a single solver object is reused (and
  warm-started) across all stages.
- `state_bounds`: `nothing` or `(lower, upper)` pair to clamp realized
  states via [`_project_state_to_bounds`](@ref).
- `project_state`: `nothing` or a callable `f(state) -> state` for non-box
  feasibility repairs via [`_project_realized_state`](@ref).
- `retry_on_failure::Bool`: if `true`, failed or non-finite solves are
  retried once with a cold-start solver.

# Returns
- `nothing` if any stage solve fails after retry.
- A `NamedTuple` with fields:
  - `objective::Float64`: cumulative objective over the horizon.
  - `objective_no_target_penalty::Float64`: cumulative objective without
    target penalties.
  - `target_violation_share::Float64`: fraction of objective from target
    penalties (see [`_target_violation_share`](@ref)).
  - `final_state::AbstractVector`: realized state after the last stage.
  - `state_trajectory::Vector`: realized states ``[x_0, x_1, \\ldots, x_T]``.
  - `target_trajectory::Vector`: policy targets
    ``[\\hat{x}_1, \\ldots, \\hat{x}_T]``.

# Throws
- `ArgumentError` if `horizon < 1`, `n_uncertainty < 1`, `w_flat` has the
  wrong length, or `policy_state` is not `:realized` or `:target`.

# Examples
```julia
result = rollout_tsddr(
    model, x0, stage_problem, w_flat;
    horizon = 96,
    n_uncertainty = 5,
    set_stage_parameters! = my_set_params!,
    realized_state = my_realized_state,
)
result !== nothing && @show result.objective
```
"""
function rollout_tsddr(
    model,
    initial_state::AbstractVector,
    stage_problem,
    w_flat::AbstractVector;
    horizon::Int,
    n_uncertainty::Int,
    set_stage_parameters!::Function,
    realized_state::Function,
    objective_no_target_penalty::Function = (prob, result) -> result.objective,
    madnlp_kwargs = NamedTuple(),
    warmstart::Bool = true,
    policy_state::Symbol = :realized,
    solver_state = nothing,
    reuse_solver::Bool = false,
    state_bounds = nothing,
    project_state = nothing,
    retry_on_failure::Bool = true,
)
    # --- Input validation ---------------------------------------------------
    horizon >= 1 || throw(ArgumentError("horizon must be >= 1"))
    n_uncertainty >= 1 || throw(ArgumentError("n_uncertainty must be >= 1"))
    # w_flat must contain exactly horizon slices of n_uncertainty each.
    length(w_flat) == horizon * n_uncertainty ||
        throw(ArgumentError("w_flat length must be horizon*n_uncertainty=$(horizon * n_uncertainty), got $(length(w_flat))"))
    # Only two feedback modes are supported.
    policy_state in (:realized, :target) ||
        throw(ArgumentError("policy_state must be :realized or :target, got :$policy_state"))

    # --- Infer numeric types and allocate trajectory storage ---------------
    F = eltype(initial_state)                                # element type (e.g. Float32)
    nx = length(initial_state)                               # state dimension
    state = solver_state                                     # optional pre-built solver
    w_flat = _adapt_array(F.(w_flat), initial_state)         # move w_flat to same device as initial_state

    # Reset any recurrent state in the policy network (e.g. LSTM hidden state).
    Flux.reset!(model)
    # x_0: the realized state entering stage 1.
    realized_prev = F.(_to_vec(initial_state))
    # Target recurrence state (used when policy_state == :target).
    target_prev = copy(realized_prev)
    # Pre-allocate trajectory arrays: states have T+1 entries, targets have T.
    state_trajectory = Vector{AbstractVector{F}}(undef, horizon + 1)
    target_trajectory = Vector{AbstractVector{F}}(undef, horizon)
    state_trajectory[1] = copy(realized_prev)                # store x_0

    # Pre-allocate Float64 buffers for the solver interface (solvers use Float64).
    state_f64  = similar(initial_state, Float64, nx)         # x_{t-1} in Float64
    w_f64      = similar(initial_state, Float64, n_uncertainty)  # w_t in Float64
    target_f64 = similar(initial_state, Float64, nx)         # xhat_t in Float64

    # Running sums of cumulative cost over the horizon.
    objective = 0.0             # total objective (operational + target penalty)
    objective_no_penalty = 0.0  # total objective without target penalty

    # --- Stage-wise forward pass -------------------------------------------
    for stage in 1:horizon
        # Slice out this stage's uncertainty window from the flat vector.
        wt = view(w_flat, (stage-1)*n_uncertainty+1 : stage*n_uncertainty)

        # Choose the state fed to the policy: realized (closed-loop) or target.
        policy_input_state = policy_state === :realized ? realized_prev : target_prev
        # Evaluate the policy: pi_theta(w_t, x_{t-1}) -> xhat_t.
        target = model(vcat(wt, policy_input_state))
        # Flatten the target to a 1-D vector for downstream use.
        target_vec = _to_vec(target)
        # Store the target in the trajectory (cast to the state element type).
        target_trajectory[stage] = F.(target_vec)

        # Copy inputs into the Float64 solver buffers.
        copyto!(state_f64, realized_prev)   # x_{t-1}
        copyto!(w_f64, wt)                  # w_t
        copyto!(target_f64, target_vec)     # xhat_t
        # Write the current state, uncertainty, and target into the stage problem.
        set_stage_parameters!(
            stage_problem,
            state_f64,
            w_f64,
            target_f64,
            stage,
        )

        # --- Solve the single-stage optimization problem -------------------
        if reuse_solver || state !== nothing
            # Reuse an existing solver; create one lazily on first call.
            state === nothing && (state = _make_solver(stage_problem.model, madnlp_kwargs))
            result = _solve!(
                state,
                stage_problem.model;
                warmstart = warmstart,
                madnlp_kwargs = madnlp_kwargs,
            )
        else
            # Create a fresh solver for this stage (no cross-stage warm-start).
            stage_state = _make_solver(stage_problem.model, madnlp_kwargs)
            result = _solve!(
                stage_state,
                stage_problem.model;
                warmstart = false,
                madnlp_kwargs = madnlp_kwargs,
            )
        end

        # --- Retry logic: cold-start fallback on solver failure ------------
        if retry_on_failure && (!solve_succeeded(result) || !isfinite(result.objective))
            # Build a fresh solver and retry without warm-start.
            retry_state = _make_solver(stage_problem.model, madnlp_kwargs)
            result = _solve!(
                retry_state,
                stage_problem.model;
                warmstart = false,
                madnlp_kwargs = madnlp_kwargs,
            )
        end

        # Abort the rollout if the solve still failed after retry.
        solve_succeeded(result) || return nothing
        # Abort on non-finite objective (e.g. MadNLP returning 0.0 for infeasible).
        isfinite(result.objective) || return nothing

        # --- Extract penalty-free objective, with retry --------------------
        no_penalty = objective_no_target_penalty(stage_problem, result)
        if retry_on_failure && !isfinite(no_penalty)
            # Non-finite penalty-free cost can indicate a solver glitch; retry.
            retry_state = _make_solver(stage_problem.model, madnlp_kwargs)
            result = _solve!(
                retry_state,
                stage_problem.model;
                warmstart = false,
                madnlp_kwargs = madnlp_kwargs,
            )
            solve_succeeded(result) || return nothing
            isfinite(result.objective) || return nothing
            no_penalty = objective_no_target_penalty(stage_problem, result)
        end
        # Final guard: abort if the penalty-free cost is still non-finite.
        isfinite(no_penalty) || return nothing

        # --- Accumulate costs and advance the state ------------------------
        objective += result.objective            # add stage cost to cumulative total
        objective_no_penalty += no_penalty       # add penalty-free stage cost
        # Read the realized next state x_t from the solver solution.
        raw_realized = F.(_to_vec(realized_state(stage_problem, result)))
        # Project x_t to feasibility (box bounds + custom projector).
        realized_prev = _project_realized_state(raw_realized, state_bounds, project_state)
        # Update the target recurrence for :target mode.
        target_prev = target_trajectory[stage]
        # Record x_t in the state trajectory.
        state_trajectory[stage + 1] = copy(realized_prev)
    end

    # --- Assemble and return the rollout summary ---------------------------
    return (
        objective = objective,                                                      # sum of stage objectives
        objective_no_target_penalty = objective_no_penalty,                          # sum minus target penalties
        target_violation_share = _target_violation_share(objective, objective_no_penalty),  # penalty fraction
        final_state = realized_prev,                                                # x_T
        state_trajectory = state_trajectory,                                        # [x_0, ..., x_T]
        target_trajectory = target_trajectory,                                      # [xhat_1, ..., xhat_T]
    )
end

"""
    RolloutEvaluation

Store configuration and mutable summaries for periodic rollout evaluation.

# Fields
- `stage_problem`: the single-stage optimization problem template.
- `initial_state`: initial state ``x_0`` for every rollout.
- `scenarios::Vector`: pre-sampled uncertainty vectors, each of length
  `horizon * n_uncertainty`.
- `horizon::Int`: number of stages ``T``.
- `n_uncertainty::Int`: per-stage uncertainty dimension.
- `set_stage_parameters!::Function`: callback to write stage data into the
  problem (see [`rollout_tsddr`](@ref)).
- `realized_state::Function`: callback to extract ``x_t`` from a solve
  result.
- `objective_no_target_penalty::Function`: callback to extract the
  penalty-free stage cost.
- `madnlp_kwargs`: keyword arguments forwarded to MadNLP.
- `warmstart::Bool`: whether to warm-start across stages.
- `stride::Int`: evaluate every `stride` training iterations.
- `policy_state::Symbol`: `:realized` or `:target` (see
  [`rollout_tsddr`](@ref)).
- `solver_state`: pre-built solver object (or `nothing`).
- `reuse_solver::Bool`: whether to reuse a single solver across stages.
- `state_bounds`: optional `(lower, upper)` feasibility bounds.
- `project_state`: optional non-box projection callback.
- `retry_on_failure::Bool`: retry failed solves with a cold start.
- `stage_problem_pool::Vector`: pool of stage problems for parallel
  evaluation.
- `active_scenarios::Int`: number of scenarios to evaluate (at most
  `length(scenarios)`).
- `last_objective::Float64`: mean objective across successful scenarios.
- `last_objective_no_target_penalty::Float64`: mean penalty-free objective.
- `last_violation_share::Float64`: mean target-violation share.
- `last_n_ok::Int`: number of scenarios that solved successfully.
- `last_scenario_data::Vector{Any}`: per-scenario `(index, result)` pairs
  from the most recent evaluation.

# Notes
The struct is callable as `evaluation(iter, model)`. At every `stride`-th
iteration, it runs [`rollout_tsddr`](@ref) on up to `active_scenarios`
scenarios and updates the mutable summary fields. When `stage_problem_pool`
contains more than one entry, scenarios are distributed across the pool with
`Threads.@spawn`.

Thread-safety: the single `set_stage_parameters!`, `realized_state`, and
`objective_no_target_penalty` callbacks are shared across all spawned tasks
while each task receives its own stage problem from the pool. When
`stage_problem_pool` has more than one entry, these callbacks must therefore be
thread-safe and must write only into the stage problem they are handed — any
shared mutable buffer (e.g. a captured scratch array reused across calls)
races across tasks and silently corrupts results.
"""
mutable struct RolloutEvaluation <: Function
    stage_problem                            # single-stage optimization problem template
    initial_state                            # initial state x_0 for all rollouts
    scenarios::Vector                        # pre-sampled uncertainty vectors
    horizon::Int                             # number of decision stages T
    n_uncertainty::Int                       # per-stage uncertainty dimension
    set_stage_parameters!::Function          # callback: write stage data into the problem
    realized_state::Function                 # callback: extract realized x_t from result
    objective_no_target_penalty::Function    # callback: penalty-free stage cost
    madnlp_kwargs                            # solver keyword arguments
    warmstart::Bool                          # warm-start across stages
    stride::Int                              # evaluate every stride-th iteration
    policy_state::Symbol                     # :realized or :target feedback mode
    solver_state                             # pre-built solver (or nothing)
    reuse_solver::Bool                       # reuse single solver across stages
    state_bounds                             # (lower, upper) feasibility bounds or nothing
    project_state                            # non-box projection callback or nothing
    retry_on_failure::Bool                   # retry failed solves with cold start
    stage_problem_pool::Vector               # pool of stage problems for parallel evaluation
    active_scenarios::Int                    # how many scenarios to evaluate (leq length(scenarios))
    last_objective::Float64                  # mean objective from last evaluation
    last_objective_no_target_penalty::Float64 # mean penalty-free objective from last evaluation
    last_violation_share::Float64            # mean target-violation share from last evaluation
    last_n_ok::Int                           # number of successful scenarios in last evaluation
    last_scenario_data::Vector{Any}          # per-scenario (index, result) pairs from last eval
end

"""
    RolloutEvaluation(
        stage_problem,
        initial_state,
        scenarios;
        horizon::Int,
        n_uncertainty::Int,
        set_stage_parameters!::Function,
        realized_state::Function,
        objective_no_target_penalty::Function = (prob, result) -> result.objective,
        madnlp_kwargs = NamedTuple(),
        warmstart::Bool = true,
        stride::Int = 1,
        policy_state::Symbol = :realized,
        reuse_solver::Bool = false,
        state_bounds = nothing,
        project_state = nothing,
        retry_on_failure::Bool = true,
        stage_problem_pool::Vector = [],
        active_scenarios::Int = length(scenarios),
    ) -> RolloutEvaluation

Construct a [`RolloutEvaluation`](@ref) callback for periodic out-of-sample
policy evaluation during training.

All mutable result fields (`last_objective`, `last_n_ok`, etc.) are
initialized to `NaN` / `0` / empty and are populated on the first call.

# Arguments
- `stage_problem`: the single-stage optimization problem (must expose
  `.model`).
- `initial_state`: initial state ``x_0``.
- `scenarios`: iterable of pre-sampled uncertainty vectors, each of length
  `horizon * n_uncertainty`.

# Keywords
- `horizon::Int`: number of stages ``T``.
- `n_uncertainty::Int`: per-stage uncertainty dimension.
- `set_stage_parameters!::Function`: stage-parameter callback.
- `realized_state::Function`: realized-state extraction callback.
- `objective_no_target_penalty::Function`: penalty-free cost callback.
- `madnlp_kwargs`: solver keyword arguments.
- `warmstart::Bool`: warm-start across stages within each rollout.
- `stride::Int`: evaluate every `stride`-th training iteration.
- `policy_state::Symbol`: `:realized` (closed-loop) or `:target`.
- `reuse_solver::Bool`: reuse a single solver across stages.
- `state_bounds`: `nothing` or `(lower, upper)` for box projection.
- `project_state`: `nothing` or custom non-box projector.
- `retry_on_failure::Bool`: retry failed solves with cold start.
- `stage_problem_pool::Vector`: pool of independent stage problems for
  multi-threaded evaluation.  An empty pool uses sequential evaluation.
  With more than one pool entry, the shared `set_stage_parameters!`,
  `realized_state`, and `objective_no_target_penalty` callbacks run
  concurrently on different tasks: they must be thread-safe and write only
  into the stage problem passed to them (no shared mutable buffers),
  otherwise results race.
- `active_scenarios::Int`: cap on how many scenarios to evaluate (defaults
  to all).

# Returns
- `RolloutEvaluation`: callable training callback with empty result summaries.

# Throws
- `ArgumentError` if `scenarios` is empty, `stride < 1`, or `policy_state` is
  not `:realized` or `:target`.

# Examples
```julia
eval_cb = RolloutEvaluation(
    stage_problem, x0, test_scenarios;
    horizon = 96,
    n_uncertainty = 5,
    set_stage_parameters! = my_set_params!,
    realized_state = my_realized_state,
    stride = 10,
)
# Use as a training callback:
eval_cb(iter, model)
```
"""
function RolloutEvaluation(
    stage_problem,
    initial_state,
    scenarios;
    horizon::Int,
    n_uncertainty::Int,
    set_stage_parameters!::Function,
    realized_state::Function,
    objective_no_target_penalty::Function = (prob, result) -> result.objective,
    madnlp_kwargs = NamedTuple(),
    warmstart::Bool = true,
    stride::Int = 1,
    policy_state::Symbol = :realized,
    reuse_solver::Bool = false,
    state_bounds = nothing,
    project_state = nothing,
    retry_on_failure::Bool = true,
    stage_problem_pool::Vector = [],
    active_scenarios::Int = length(scenarios),
)
    # At least one scenario is required for meaningful evaluation.
    isempty(scenarios) && throw(ArgumentError("scenarios must be nonempty"))
    # Stride must be positive; stride=1 evaluates every iteration.
    stride >= 1 || throw(ArgumentError("stride must be >= 1"))
    # Validate the feedback mode.
    policy_state in (:realized, :target) ||
        throw(ArgumentError("policy_state must be :realized or :target, got :$policy_state"))

    return RolloutEvaluation(
        stage_problem,
        initial_state,
        collect(scenarios),                # materialize to a concrete Vector
        horizon,
        n_uncertainty,
        set_stage_parameters!,
        realized_state,
        objective_no_target_penalty,
        madnlp_kwargs,
        warmstart,
        stride,
        policy_state,
        # Pre-build the solver if reuse is requested; otherwise leave as nothing.
        reuse_solver ? _make_solver(stage_problem.model, madnlp_kwargs) : nothing,
        reuse_solver,
        state_bounds,
        project_state,
        retry_on_failure,
        collect(stage_problem_pool),       # materialize the pool to a concrete Vector
        active_scenarios,
        NaN,                               # last_objective: not yet evaluated
        NaN,                               # last_objective_no_target_penalty
        NaN,                               # last_violation_share
        0,                                 # last_n_ok: no successful scenarios yet
        Any[],                             # last_scenario_data: empty until first eval
    )
end

"""
    (evaluation::RolloutEvaluation)(iter, model) -> Nothing

Evaluate `model` on rollout scenarios when `iter` is aligned with
`evaluation.stride`.

# Arguments
- `evaluation::RolloutEvaluation`: callback state and rollout configuration.
- `iter`: current training iteration.
- `model`: policy model passed to [`rollout_tsddr`](@ref).

# Returns
- `nothing`: summary fields on `evaluation` are updated in place.

# Notes
If `iter % evaluation.stride != 0`, the method only clears stale per-scenario
data and returns. Otherwise it records mean objective, mean penalty-free
objective, mean target-violation share, the number of successful scenarios, and
per-scenario results.
"""
function (evaluation::RolloutEvaluation)(iter, model)
    empty!(evaluation.last_scenario_data)
    iter % evaluation.stride == 0 || return nothing

    n_eval = min(evaluation.active_scenarios, length(evaluation.scenarios))
    pool   = evaluation.stage_problem_pool
    nw     = length(pool)

    total = 0.0
    total_no_penalty = 0.0
    n_ok = 0

    if nw <= 1
        # Sequential path (original behavior)
        for i in 1:n_eval
            result = rollout_tsddr(
                model,
                evaluation.initial_state,
                evaluation.stage_problem,
                evaluation.scenarios[i];
                horizon = evaluation.horizon,
                n_uncertainty = evaluation.n_uncertainty,
                set_stage_parameters! = evaluation.set_stage_parameters!,
                realized_state = evaluation.realized_state,
                objective_no_target_penalty = evaluation.objective_no_target_penalty,
                madnlp_kwargs = evaluation.madnlp_kwargs,
                warmstart = evaluation.warmstart,
                policy_state = evaluation.policy_state,
                solver_state = evaluation.solver_state,
                reuse_solver = evaluation.reuse_solver,
                state_bounds = evaluation.state_bounds,
                project_state = evaluation.project_state,
                retry_on_failure = evaluation.retry_on_failure,
            )
            result === nothing && continue
            total += result.objective
            total_no_penalty += result.objective_no_target_penalty
            n_ok += 1
            push!(evaluation.last_scenario_data, (i, result))
        end
    else
        # Parallel path: distribute scenarios across pool
        results = Vector{Union{Nothing, NamedTuple}}(nothing, n_eval)
        for round_start in 1:nw:n_eval
            round_end = min(round_start + nw - 1, n_eval)
            tasks = Task[]
            for i in round_start:round_end
                wi = i - round_start + 1
                sp = pool[wi]
                scenario = evaluation.scenarios[i]
                m_copy = deepcopy(model)
                t = Threads.@spawn rollout_tsddr(
                    m_copy,
                    evaluation.initial_state,
                    sp,
                    scenario;
                    horizon = evaluation.horizon,
                    n_uncertainty = evaluation.n_uncertainty,
                    set_stage_parameters! = evaluation.set_stage_parameters!,
                    realized_state = evaluation.realized_state,
                    objective_no_target_penalty = evaluation.objective_no_target_penalty,
                    madnlp_kwargs = evaluation.madnlp_kwargs,
                    warmstart = evaluation.warmstart,
                    policy_state = evaluation.policy_state,
                    reuse_solver = false,
                    state_bounds = evaluation.state_bounds,
                    project_state = evaluation.project_state,
                    retry_on_failure = evaluation.retry_on_failure,
                )
                push!(tasks, t)
            end
            for (j, t) in enumerate(tasks)
                results[round_start + j - 1] = fetch(t)
            end
        end
        for (idx, r) in enumerate(results)
            r === nothing && continue
            total += r.objective
            total_no_penalty += r.objective_no_target_penalty
            n_ok += 1
            push!(evaluation.last_scenario_data, (idx, r))
        end
    end

    evaluation.last_n_ok = n_ok
    if n_ok == 0
        evaluation.last_objective = NaN
        evaluation.last_objective_no_target_penalty = NaN
        evaluation.last_violation_share = NaN
        return nothing
    end

    evaluation.last_objective = total / n_ok
    evaluation.last_objective_no_target_penalty = total_no_penalty / n_ok
    evaluation.last_violation_share = _target_violation_share(
        evaluation.last_objective,
        evaluation.last_objective_no_target_penalty,
    )
    return nothing
end

"""
    critic_samples_from_evaluation(
        eval_obj::RolloutEvaluation;
        objective_key::Symbol = :objective,
    ) -> Vector{CriticSample}

Convert the last rollout evaluation results into `CriticSample`s for critic
training.

# Arguments
- `eval_obj::RolloutEvaluation`: evaluation callback containing
  `last_scenario_data` from a previous call.

# Keywords
- `objective_key::Symbol`: field of each rollout result used as the scalar
  critic target; commonly `:objective` or `:objective_no_target_penalty`.

# Returns
- `Vector{CriticSample}`: one sample per successful rollout scenario from the
  last evaluation.

# Notes
Rollout evaluation does not produce dual multipliers, so generated samples use
zero target multipliers and contribute only to the value-loss term unless
combined with other samples.
"""
function critic_samples_from_evaluation(
    eval_obj::RolloutEvaluation;
    objective_key::Symbol = :objective,
)
    isempty(eval_obj.last_scenario_data) && return CriticSample[]
    F = eltype(eval_obj.initial_state)
    samples = CriticSample[]
    for (i, result) in eval_obj.last_scenario_data
        w_flat = eval_obj.scenarios[i]
        xhat_flat = F.(vcat(result.target_trajectory...))
        obj = Float64(getfield(result, objective_key))
        push!(samples, CriticSample(
            F.(eval_obj.initial_state),
            F.(w_flat),
            xhat_flat,
            obj,
            zeros(F, length(xhat_flat)),
        ))
    end
    return samples
end
