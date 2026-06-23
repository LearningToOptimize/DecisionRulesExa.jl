"""
Stage-wise implicit differentiation via MadDiff.

The rrule for `solve_subproblem` computes VJPs through the KKT system
of the single-stage NLP, providing the chain-rule factors needed by
Eq. 2.5 of the TS-DDR paper:

- ``\\partial q_t / \\partial \\hat x_t = \\lambda_t``
- ``\\partial q_t / \\partial x_{t-1} = \\mu_t``
- ``\\partial x_t^* / \\partial \\hat x_t`` and ``\\partial x_t^* / \\partial x_{t-1}``
  (solution sensitivities via implicit KKT differentiation)

Zygote chains these factors across stages automatically.
"""

import MadDiff

mutable struct SubproblemSolverState
    solver
    maddiff_sens
    last_good_y
    last_good_zl_vals
    last_good_zu_vals
end

function _make_subproblem_solver(nlp, madnlp_kwargs)
    solver = MadNLP.MadNLPSolver(nlp; madnlp_kwargs...)
    return SubproblemSolverState(solver, nothing, nothing, nothing, nothing)
end

function _solve_subproblem!(state::SubproblemSolverState; warmstart::Bool, madnlp_kwargs)
    solver = state.solver

    # Always cold-start: MakeParameter wrapper in klamike fork causes
    # DimensionMismatch in reinitialize! when status != INITIAL.
    solver.status = MadNLP.INITIAL
    solver.cnt.k = 0
    solver.cnt.acceptable_cnt = 0
    solver.cnt.start_time = time()

    res = MadNLP.solve!(solver; madnlp_kwargs...)

    if solve_succeeded(res)
        state.maddiff_sens = MadDiff.MadDiffSolver(solver)
    end
    return res
end

"""
    solve_subproblem(prob, x_in, w_t, xhat_t, solver_state; warmstart, madnlp_kwargs)
        -> (objective, x_realized) or nothing

Solve a single-stage subproblem and return the objective and realized state.
Returns `nothing` on solver failure.
"""
function solve_subproblem(
    prob::SubproblemDEProblem,
    x_in::AbstractVector,
    w_t::AbstractVector,
    xhat_t::AbstractVector,
    solver_state::SubproblemSolverState;
    warmstart::Bool = true,
    madnlp_kwargs = NamedTuple(),
)
    ExaModels.set_parameter!(prob.core, prob.p_x_in, Float64.(x_in))
    ExaModels.set_parameter!(prob.core, prob.p_w, Float64.(w_t))
    ExaModels.set_parameter!(prob.core, prob.p_xhat, Float64.(xhat_t))

    result = _solve_subproblem!(solver_state;
        warmstart = warmstart, madnlp_kwargs = madnlp_kwargs)

    solve_succeeded(result) || return nothing
    isfinite(result.objective) || return nothing

    x_real = Array(result.solution[prob.state_var_range])
    return (objective = result.objective, x_realized = x_real)
end

function _solve_subproblem_diffable(
    prob::SubproblemDEProblem,
    x_in::AbstractVector{F},
    w_t::AbstractVector,
    xhat_t::AbstractVector{F},
    solver_state::SubproblemSolverState;
    warmstart::Bool = true,
    madnlp_kwargs = NamedTuple(),
) where {F}
    ExaModels.set_parameter!(prob.core, prob.p_x_in, Float64.(x_in))
    ExaModels.set_parameter!(prob.core, prob.p_w, Float64.(w_t))
    ExaModels.set_parameter!(prob.core, prob.p_xhat, Float64.(xhat_t))

    result = _solve_subproblem!(solver_state;
        warmstart = warmstart, madnlp_kwargs = madnlp_kwargs)

    ok = solve_succeeded(result) && isfinite(result.objective)

    x_real = ok ? F.(Array(result.solution[prob.state_var_range])) : zeros(F, prob.nx)
    obj = ok ? F(result.objective) : F(0)

    return obj, x_real, ok
end

function ChainRulesCore.rrule(
    ::typeof(_solve_subproblem_diffable),
    prob::SubproblemDEProblem,
    x_in::AbstractVector{F},
    w_t::AbstractVector,
    xhat_t::AbstractVector{F},
    solver_state::SubproblemSolverState;
    warmstart::Bool = true,
    madnlp_kwargs = NamedTuple(),
) where {F}

    obj, x_real, ok = _solve_subproblem_diffable(
        prob, x_in, w_t, xhat_t, solver_state;
        warmstart, madnlp_kwargs,
    )

    # Snapshot at forward-pass time — MadDiffSolver holds a reference to the
    # underlying MadNLP solver, so each stage in a rollout MUST use its own
    # solver_state to prevent later solves from invalidating earlier snapshots.
    _maddiff_sens = solver_state.maddiff_sens
    _n_primal = ok ? NLPModels.get_nvar(solver_state.solver.nlp) : 0

    function subproblem_pullback(Δ)
        Δobj, Δx_real, _ = Δ

        if !ok || _maddiff_sens === nothing
            return (
                ChainRulesCore.NoTangent(),
                ChainRulesCore.NoTangent(),
                zeros(F, prob.nx),
                ChainRulesCore.NoTangent(),
                zeros(F, prob.nx),
                ChainRulesCore.NoTangent(),
            )
        end

        sens = _maddiff_sens
        n_primal = _n_primal

        dL_dx = zeros(Float64, n_primal)
        if Δx_real isa AbstractVector
            for (k, i) in enumerate(prob.state_var_range)
                dL_dx[i] = Float64(Δx_real[k])
            end
        end

        dobj_val = 0.0
        if Δobj isa Number
            dobj_val = Float64(Δobj)
        end

        vjp = MadDiff.vector_jacobian_product!(sens; dL_dx = dL_dx, dobj = dobj_val)
        grad_p = Array(vjp.grad_p)

        if !all(isfinite, grad_p)
            return (
                ChainRulesCore.NoTangent(),
                ChainRulesCore.NoTangent(),
                zeros(F, prob.nx),
                ChainRulesCore.NoTangent(),
                zeros(F, prob.nx),
                ChainRulesCore.NoTangent(),
            )
        end

        d_x_in = F.(grad_p[prob.p_x_in_param_range])
        d_xhat = F.(grad_p[prob.p_xhat_param_range])

        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            d_x_in,
            ChainRulesCore.NoTangent(),
            d_xhat,
            ChainRulesCore.NoTangent(),
        )
    end

    return (obj, x_real, ok), subproblem_pullback
end
