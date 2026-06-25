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

_mat(x) = x
_mat(x::Zygote.OneElement) = collect(x)
function _mat(x::ChainRulesCore.Tangent{<:Any})
    nt = ChainRulesCore.backing(x)
    return NamedTuple{keys(nt)}(map(_mat, values(nt)))
end
function _mat(x::ChainRulesCore.MutableTangent{<:Any})
    nt = ChainRulesCore.backing(x)
    return NamedTuple{keys(nt)}(map(_mat, values(nt)))
end
materialize_tangent(g) = isnothing(g) ? nothing : _mat(g)

_all_finite_gradient(x::AbstractArray) = all(isfinite, x)
_all_finite_gradient(x::Number)        = isfinite(x)
_all_finite_gradient(x::Nothing)       = true
_all_finite_gradient(x::NamedTuple)    = all(_all_finite_gradient(v) for v in values(x))
_all_finite_gradient(x::Tuple)         = all(_all_finite_gradient(v) for v in x)
_all_finite_gradient(x)                = true

function _adapt_array(x::AbstractVector, ref::AbstractVector)
    typeof(x) === typeof(ref) && return x
    copyto!(similar(ref, eltype(x), length(x)), x)
end

# ── Solve-status check ────────────────────────────────────────────────────────

"""
    solve_succeeded(result) -> Bool
"""
function solve_succeeded(result)
    s = result.status
    return s == MadNLP.SOLVE_SUCCEEDED || s == MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL
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

mutable struct _SolverState
    solver
    last_good_x        # primal snapshot (CPU or GPU array), or nothing
    last_good_y        # dual snapshot, or nothing
    last_good_zl_vals
    last_good_zu_vals
    has_fixed_vars::Bool
end

function _make_solver(nlp, madnlp_kwargs)
    solver = MadNLP.MadNLPSolver(nlp; madnlp_kwargs...)
    nvar_solver = length(solver.x.x)
    nvar_nlp    = length(NLPModels.get_x0(nlp))
    has_fixed   = nvar_solver != nvar_nlp
    return _SolverState(solver, nothing, nothing, nothing, nothing, has_fixed)
end

function _solve!(state::_SolverState, nlp; warmstart::Bool, madnlp_kwargs)
    solver = state.solver

    # MadNLP solver reuse with MakeParameter (fixed variables) causes INFEASIBLE
    # on subsequent solves even with INITIAL status — stale KKT factorization state.
    # Fix: create a fresh solver each time when fixed variables exist.
    if state.has_fixed_vars
        return MadNLP.madnlp(nlp; madnlp_kwargs...)
    end

    # Normal path (no fixed variables): full warm-start support.
    if warmstart && state.last_good_x !== nothing
        copyto!(NLPModels.get_x0(nlp), state.last_good_x)
    end

    prev_result = solver.status
    prev_failed = (prev_result != MadNLP.INITIAL &&
                   prev_result != MadNLP.SOLVE_SUCCEEDED &&
                   prev_result != MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL)

    if !warmstart
        solver.status = MadNLP.INITIAL
    elseif prev_failed && state.last_good_y !== nothing
        solver.y         .= state.last_good_y
        solver.zl.values .= state.last_good_zl_vals
        solver.zu.values .= state.last_good_zu_vals
        solver.status     = MadNLP.SOLVE_SUCCEEDED
    elseif prev_failed
        solver.status = MadNLP.INITIAL
    end

    # Reset per-solve iteration budget.
    solver.cnt.k              = 0
    solver.cnt.acceptable_cnt = 0
    solver.cnt.start_time     = time()

    res = MadNLP.solve!(solver; madnlp_kwargs...)

    if solve_succeeded(res)
        state.last_good_x       = copy(solver.x.x)
        state.last_good_y       = copy(solver.y)
        state.last_good_zl_vals = copy(solver.zl.values)
        state.last_good_zu_vals = copy(solver.zu.values)
    end
    return res
end

# ── simulate_tsddr ────────────────────────────────────────────────────────────

"""
    simulate_tsddr(model, initial_state, det_equivalent,
                   p_x0, p_target, p_uncertainty,
                   uncertainty_sampler; madnlp_kwargs, warmstart)
        -> (objective, lambda) or nothing

Single forward pass without a gradient update.
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
    T    = det_equivalent.horizon
    F    = eltype(initial_state)
    nx   = length(initial_state)
    core = det_equivalent.core
    nlp  = det_equivalent.model

    state = _make_solver(nlp, madnlp_kwargs)

    w_flat = uncertainty_sampler()
    nw     = length(w_flat) ÷ T
    w_dev  = _adapt_array(F.(w_flat), initial_state)

    Flux.reset!(model)
    xhat_stages = AbstractVector{F}[]
    prev = F.(initial_state)
    for t in 1:T
        wt   = w_dev[(t-1)*nw+1 : t*nw]
        push!(xhat_stages, model(vcat(wt, prev)))
        prev = xhat_stages[end]
    end
    xhat_flat = vcat(xhat_stages...)

    ExaModels.set_parameter!(core, p_x0,          initial_state)
    ExaModels.set_parameter!(core, p_uncertainty,  w_flat)
    ExaModels.set_parameter!(core, p_target,       Float64.(xhat_flat))

    result = _solve!(state, nlp; warmstart = false, madnlp_kwargs = madnlp_kwargs)

    solve_succeeded(result) || return nothing
    isfinite(result.objective) || return nothing

    λ = result.multipliers[det_equivalent.target_con_range]
    return (objective = result.objective, lambda = F.(λ))
end

function _rollout_xhat_flat(model, initial_state, w_flat, T::Int, F)
    nw = length(w_flat) ÷ T
    Flux.reset!(model)
    prev = F.(initial_state)
    xhat = model(vcat(w_flat[1:nw], prev))
    prev = xhat
    for t in 2:T
        wt = w_flat[(t-1)*nw+1 : t*nw]
        xt = model(vcat(wt, prev))
        xhat = vcat(xhat, xt)
        prev = xt
    end
    return xhat
end

_has_critic(::NoCriticControlVariate) = false
_has_critic(::AbstractCriticControlVariate) = true

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
    result = rollout_tsddr(
        model,
        initial_state,
        target.stage_problem,
        w_flat;
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
    )
    result === nothing && return nothing

    objective = target.objective_value === :objective ?
        result.objective : result.objective_no_target_penalty
    xhat_flat = F.(vcat(result.target_trajectory...))
    return CriticSample(F.(initial_state), F.(w_flat), xhat_flat, objective, F.(lambda))
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
                    result = _solve!(st, de.model; warmstart=warmstart, madnlp_kwargs=madnlp_kwargs)
                    if solve_succeeded(result) && isfinite(result.objective)
                        λ = result.multipliers[de.target_con_range]
                        if all(isfinite, λ)
                            put!(out_ch, (s_idx, F.(w_flat), _adapt_array(F.(λ), w_flat), result.objective))
                            continue
                        end
                    end
                    put!(out_ch, (s_idx, nothing, nothing, NaN))
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
            xhat_stages = AbstractVector{F}[]
            prev = F.(initial_state)
            for t in 1:T
                wt   = w_dev[(t-1)*nw+1 : t*nw]
                push!(xhat_stages, model(vcat(wt, prev)))
                prev = xhat_stages[end]
            end
            sample_data[s] = (w_dev, vcat(xhat_stages...))
        end

        # Step 2: Solve — parallel across workers if pool provided
        solve_ok  = Vector{Union{Nothing, Tuple{AbstractVector{F}, AbstractVector{F}, Float64}}}(nothing, num_train_per_batch)

        if nworkers == 1
            (de, px, pt, pu) = _pool[1]
            st = single_state
            for s in 1:num_train_per_batch
                w_flat, xhat_flat = sample_data[s]
                ExaModels.set_parameter!(de.core, px, initial_state)
                ExaModels.set_parameter!(de.core, pu, w_flat)
                ExaModels.set_parameter!(de.core, pt, Float64.(xhat_flat))
                result = _solve!(st, de.model; warmstart=warmstart, madnlp_kwargs=madnlp_kwargs)
                solve_succeeded(result) || continue
                isfinite(result.objective) || continue
                λ = result.multipliers[de.target_con_range]
                all(isfinite, λ) || continue
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
                    (s_idx, w_out, λ_out, obj_out) = take!(out_channels[wi])
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
                        prev_ad = F.(initial_state)
                        for t in 1:T
                            wt      = F.(w_flat_s[(t-1)*nw+1 : t*nw])
                            xt      = m(vcat(wt, prev_ad))
                            total   = total + sum(λf[(t-1)*nx+1 : t*nx] .* xt)
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
                        prev_ad = F.(initial_state)
                        for t in 1:T
                            wt      = F.(w_flat_s[(t-1)*nw+1 : t*nw])
                            xt      = m(vcat(wt, prev_ad))
                            residual_total =
                                residual_total + sum(actor_weight[(t-1)*nx+1 : t*nx] .* xt)
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
                                F.(initial_state),
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
                invalidate_policy_cache!(embedded_de)
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
    get_realized_states      = nothing,
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

        for s in 1:num_train_per_batch
            w_flat = uncertainty_sampler()

            set_x0!(embedded_de, initial_state)
            set_uncertainty!(embedded_de, w_flat)

            result = _solve!(state, embedded_de.model;
                warmstart = warmstart, madnlp_kwargs = madnlp_kwargs)

            solve_succeeded(result) || continue
            isfinite(result.objective) || continue

            λ = result.multipliers[embedded_de.target_con_range]
            all(isfinite, λ) || continue

            x_sol = _get_states(embedded_de, result)

            λf    = _adapt_array(F.(λ), initial_state)
            xf    = _adapt_array(F.(x_sol), initial_state)
            w_dev = _adapt_array(F.(w_flat), initial_state)
            push!(valid, (w_dev, λf, xf))
            obj_sum += result.objective
        end

        n_ok     = length(valid)
        mean_obj = n_ok > 0 ? obj_sum / n_ok : NaN

        if n_ok > 0
            gs = Zygote.gradient(model) do m
                total = zero(F)
                for (w_flat_s, λf, x_realized) in valid
                    nw = length(w_flat_s) ÷ T
                    Flux.reset!(m)
                    for t in 1:T
                        wt = F.(w_flat_s[(t-1)*nw+1 : t*nw])
                        x_prev = (t == 1) ?
                            F.(initial_state) :
                            F.(x_realized[(t-2)*nx+1 : (t-1)*nx])
                        xt = m(vcat(wt, x_prev))
                        total = total + sum(λf[(t-1)*nx+1 : t*nx] .* xt)
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
