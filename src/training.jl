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
end

function _make_solver(nlp, madnlp_kwargs)
    solver = MadNLP.MadNLPSolver(nlp; madnlp_kwargs...)
    return _SolverState(solver, nothing, nothing, nothing, nothing)
end

function _solve!(state::_SolverState, nlp; warmstart::Bool, madnlp_kwargs)
    solver = state.solver

    # Primal warm-start: seed NLPModel's x0 from last-good primal.
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

    Flux.reset!(model)
    xhat_stages = Vector{Vector{F}}(undef, T)
    prev = F.(initial_state)
    for t in 1:T
        wt             = F.(w_flat[(t-1)*nw+1 : t*nw])
        xhat_stages[t] = model(vcat(wt, prev))
        prev           = xhat_stages[t]
    end
    xhat_flat = vcat(xhat_stages...)

    ExaModels.set_parameter!(core, p_x0,          initial_state)
    ExaModels.set_parameter!(core, p_uncertainty,  w_flat)
    ExaModels.set_parameter!(core, p_target,       Float64.(xhat_flat))

    result = _solve!(state, nlp; warmstart = false, madnlp_kwargs = madnlp_kwargs)

    solve_succeeded(result) || return nothing
    isfinite(result.objective) || return nothing

    λ = result.multipliers[det_equivalent.target_con_range]
    return (objective = result.objective, lambda = F.(Array(λ)))
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
- `uncertainty_sampler`: `() -> w_flat` — flat vector of length `T * nw_per_stage`

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
)
    T    = det_equivalent.horizon
    F    = eltype(initial_state)
    nx   = length(initial_state)

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
                            put!(out_ch, (s_idx, F.(w_flat), F.(Array(λ)), result.objective))
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

    try

    for iter in 1:num_batches
        num_train_per_batch = adjust_hyperparameters(iter, opt_state, num_train_per_batch)

        # ── Forward pass: rollout + solve (outside AD tape) ───────────────────

        # Step 1: Roll out policy for all samples (CPU, sequential)
        sample_data = Vector{Tuple{Vector{F}, Vector{F}}}(undef, num_train_per_batch)
        for s in 1:num_train_per_batch
            w_flat = uncertainty_sampler()
            nw     = length(w_flat) ÷ T
            Flux.reset!(model)
            xhat_stages = Vector{Vector{F}}(undef, T)
            prev = F.(initial_state)
            for t in 1:T
                wt             = F.(w_flat[(t-1)*nw+1 : t*nw])
                xhat_stages[t] = model(vcat(wt, prev))
                prev           = xhat_stages[t]
            end
            sample_data[s] = (F.(w_flat), vcat(xhat_stages...))
        end

        # Step 2: Solve — parallel across workers if pool provided
        solve_ok  = Vector{Union{Nothing, Tuple{Vector{F}, Vector{F}, Float64}}}(nothing, num_train_per_batch)

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
                solve_ok[s] = (F.(w_flat), F.(Array(λ)), result.objective)
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
        valid   = Vector{Tuple{Vector{F}, Vector{F}}}()
        obj_sum = 0.0
        for r in solve_ok
            r === nothing && continue
            push!(valid, (r[1], r[2]))
            obj_sum += r[3]
        end
        n_ok     = length(valid)
        mean_obj = n_ok > 0 ? obj_sum / n_ok : NaN

        # ── Gradient: ∇_θ (1/n) Σ_s ⟨λ_s, rollout_s(θ)⟩ ────────────────────
        if n_ok > 0
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
