# rollout.jl
#
# Stage-wise, non-cheating policy evaluation.
#
# The deterministic-equivalent training solve sees the full horizon.  This file
# provides a small callback-driven rollout loop that evaluates deployment
# semantics instead: at each stage, solve one stage, extract the realized next
# state, and feed that state into the next policy call.

function _target_violation_share(objective::Real, objective_no_target_penalty::Real)
    penalty = objective - objective_no_target_penalty
    (isfinite(objective) && isfinite(penalty) && abs(objective) > 1e-12) || return NaN
    return penalty / objective
end

_cpu_vec(x) = vec(collect(Array(x)))

"""
    rollout_tsddr(model, initial_state, stage_problem, w_flat; kwargs...)

Evaluate `model` by solving `stage_problem` sequentially over a materialized
scenario `w_flat`.  Unlike deterministic-equivalent evaluation, the optimizer
only receives one uncertainty slice at a time.

Required callbacks:
- `set_stage_parameters!(stage_problem, state_in, w_t, target, stage)` updates the
  one-stage problem before each solve.
- `realized_state(stage_problem, result)` returns the next state realized by the
  solved stage problem.

Optional callbacks:
- `objective_no_target_penalty(stage_problem, result)` returns the stage objective
  with target-slack penalty removed.  The default is `result.objective`.

`policy_state = :realized` is the closed-loop deployment semantics.  `:target`
keeps the policy recurrence on its own previous target, matching the target
generation used by deterministic-equivalent training while still solving stages
one by one.
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
)
    horizon >= 1 || throw(ArgumentError("horizon must be >= 1"))
    n_uncertainty >= 1 || throw(ArgumentError("n_uncertainty must be >= 1"))
    length(w_flat) == horizon * n_uncertainty ||
        throw(ArgumentError("w_flat length must be horizon*n_uncertainty=$(horizon * n_uncertainty), got $(length(w_flat))"))
    policy_state in (:realized, :target) ||
        throw(ArgumentError("policy_state must be :realized or :target, got :$policy_state"))

    F = eltype(initial_state)
    state = solver_state

    Flux.reset!(model)
    realized_prev = F.(_cpu_vec(initial_state))
    target_prev = copy(realized_prev)
    state_trajectory = Vector{Vector{F}}(undef, horizon + 1)
    target_trajectory = Vector{Vector{F}}(undef, horizon)
    state_trajectory[1] = copy(realized_prev)

    objective = 0.0
    objective_no_penalty = 0.0

    for stage in 1:horizon
        lo = (stage - 1) * n_uncertainty + 1
        hi = stage * n_uncertainty
        wt = F.(_cpu_vec(view(w_flat, lo:hi)))

        policy_input_state = policy_state === :realized ? realized_prev : target_prev
        target = model(vcat(wt, policy_input_state))
        target_trajectory[stage] = F.(_cpu_vec(target))

        set_stage_parameters!(
            stage_problem,
            Float64.(realized_prev),
            Float64.(wt),
            Float64.(_cpu_vec(target)),
            stage,
        )

        if reuse_solver || state !== nothing
            state === nothing && (state = _make_solver(stage_problem.model, madnlp_kwargs))
            result = _solve!(
                state,
                stage_problem.model;
                warmstart = warmstart,
                madnlp_kwargs = madnlp_kwargs,
            )
        else
            stage_state = _make_solver(stage_problem.model, madnlp_kwargs)
            result = _solve!(
                stage_state,
                stage_problem.model;
                warmstart = false,
                madnlp_kwargs = madnlp_kwargs,
            )
        end

        solve_succeeded(result) || return nothing
        isfinite(result.objective) || return nothing

        no_penalty = objective_no_target_penalty(stage_problem, result)
        isfinite(no_penalty) || return nothing

        objective += result.objective
        objective_no_penalty += no_penalty
        realized_prev = F.(_cpu_vec(realized_state(stage_problem, result)))
        target_prev = F.(_cpu_vec(target))
        state_trajectory[stage + 1] = copy(realized_prev)
    end

    return (
        objective = objective,
        objective_no_target_penalty = objective_no_penalty,
        target_violation_share = _target_violation_share(objective, objective_no_penalty),
        final_state = realized_prev,
        state_trajectory = state_trajectory,
        target_trajectory = target_trajectory,
    )
end

mutable struct RolloutEvaluation <: Function
    stage_problem
    initial_state
    scenarios::Vector
    horizon::Int
    n_uncertainty::Int
    set_stage_parameters!::Function
    realized_state::Function
    objective_no_target_penalty::Function
    madnlp_kwargs
    warmstart::Bool
    stride::Int
    policy_state::Symbol
    solver_state
    reuse_solver::Bool
    stage_problem_pool::Vector   # pool of stage problems for parallel evaluation
    active_scenarios::Int        # how many scenarios to evaluate (≤ length(scenarios))
    last_objective::Float64
    last_objective_no_target_penalty::Float64
    last_violation_share::Float64
    last_n_ok::Int
end

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
    stage_problem_pool::Vector = [],
    active_scenarios::Int = length(scenarios),
)
    isempty(scenarios) && throw(ArgumentError("scenarios must be nonempty"))
    stride >= 1 || throw(ArgumentError("stride must be >= 1"))
    policy_state in (:realized, :target) ||
        throw(ArgumentError("policy_state must be :realized or :target, got :$policy_state"))

    return RolloutEvaluation(
        stage_problem,
        initial_state,
        collect(scenarios),
        horizon,
        n_uncertainty,
        set_stage_parameters!,
        realized_state,
        objective_no_target_penalty,
        madnlp_kwargs,
        warmstart,
        stride,
        policy_state,
        reuse_solver ? _make_solver(stage_problem.model, madnlp_kwargs) : nothing,
        reuse_solver,
        collect(stage_problem_pool),
        active_scenarios,
        NaN,
        NaN,
        NaN,
        0,
    )
end

function (evaluation::RolloutEvaluation)(iter, model)
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
            )
            result === nothing && continue
            total += result.objective
            total_no_penalty += result.objective_no_target_penalty
            n_ok += 1
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
                )
                push!(tasks, t)
            end
            for (j, t) in enumerate(tasks)
                results[round_start + j - 1] = fetch(t)
            end
        end
        for r in results
            r === nothing && continue
            total += r.objective
            total_no_penalty += r.objective_no_target_penalty
            n_ok += 1
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
