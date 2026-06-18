# deterministic_equivalent.jl
#
# Deterministic equivalent workflow (TS-GDR / TS-DDR style):
#   - Policy predicts per-stage target states x̂_t
#   - Control optimizer solves a deterministic multi-period NLP to hit those targets
#   - Dual multipliers on the target constraints provide ∇_{x̂} Q, enabling training
#     without differentiating through the solver.

"""
    DeterministicEquivalentProblem

Holds an ExaModels parametric NLP for the deterministic equivalent subproblem

    Q(w, x̂) =  min_{x,u,δ}  Σ_t stage_cost(t, x_t, u_t, w_t) + (ρ/2)‖δ‖²
                s.t.        x₁ = x₀
                            dynamics(t, x_t, u_t, w_t, x_{t+1}) = 0
                            x̂_t - x_t - δ_t = 0     (target constraints, **added last**)

We add the target constraints last so their multipliers are a contiguous slice of
`result.multipliers`.
"""
struct DeterministicEquivalentProblem
    core
    model
    # decision variables (flat)
    x
    u
    δ
    # parameters
    p_x0
    p_w
    p_target
    # sizes
    nx::Int
    nu::Int
    nw::Int
    horizon::Int
    # indices of target constraints inside result.multipliers
    target_con_range::UnitRange{Int}
end

"""
    MadNLPCache

Optional cache to re-use a MadNLP solver instance across repeated solves
(warm-start + reusing symbolic factorizations).
"""
mutable struct MadNLPCache
    solver
    last_result
end

"""
    build_deterministic_equivalent(; kwargs...) -> DeterministicEquivalentProblem

Generic builder for a deterministic-equivalent dynamic NLP.

Keyword arguments:
- `horizon::Int`  : number of stages T (states are 1..T, controls are 1..T-1)
- `nx::Int`       : state dimension
- `nu::Int`       : control dimension (demo assumes `nu == nx` for default dynamics)
- `nw::Int`       : disturbance dimension (default: `nx`)
- `backend`       : ExaModels backend (e.g., `nothing` for CPU, `CUDABackend()` for GPU)
- `float_type`    : numeric type (Float64 recommended for MadNLP)
- `x_bounds`      : `(lb, ub)` applied to all state variables
- `u_bounds`      : `(lb, ub)` applied to all control variables
- `slack_penalty` : ρ ≥ 0, weight for (ρ/2)‖δ‖²

- `dynamics_eq`   : function `(t, i, x, u, w, nx, nu, nw) -> expr == 0`
                    returns the scalar equality residual for dimension `i` at stage `t`.
- `stage_cost`    : function `(t, i, x, u, w, nx, nu, nw) -> expr` returns a scalar term.

Notes:
- All variables are stored in flat vectors to keep the interface simple and robust.
- Target constraints are added *last* and written as `x̂ - x - δ = 0` so that their
  multipliers are directly the gradient w.r.t. `x̂` (envelope theorem).
"""
function build_deterministic_equivalent(;
    horizon::Int,
    nx::Int,
    nu::Int = nx,
    nw::Int = nx,
    backend = nothing,
    float_type::Type{<:AbstractFloat} = Float64,
    x_bounds::Tuple{<:Real,<:Real} = (-Inf, Inf),
    u_bounds = (-Inf, Inf),   # scalar Tuple or (lb_vec, ub_vec) with length-nu vectors
    slack_penalty::Real = 1.0,
    dynamics_eq::Function = default_dynamics_eq,
    stage_cost::Function = default_stage_cost,
)
    horizon ≥ 2 || error("horizon must be ≥ 2 (got $horizon)")
    nx ≥ 1 || error("nx must be ≥ 1 (got $nx)")
    nu ≥ 1 || error("nu must be ≥ 1 (got $nu)")
    nw ≥ 1 || error("nw must be ≥ 1 (got $nw)")

    # Default dynamics/cost assume one control & one disturbance per state component
    if (dynamics_eq === default_dynamics_eq || stage_cost === default_stage_cost) && (nu != nx || nw != nx)
        error("Default dynamics/cost assume nu == nx and nw == nx. Provide custom dynamics_eq/stage_cost for other dimensions.")
    end

    T = horizon
    n_x = T * nx
    n_u = (T - 1) * nu
    n_w = (T - 1) * nw

    core = ExaModels.ExaCore(float_type; backend = backend)

    # Expand u_bounds: scalar tuple → scalar; (lb_vec, ub_vec) → repeated flat vectors
    function _u_bound(b, side)
        v = b[side]
        v isa AbstractVector || return float_type(v)
        length(v) == nu || error("u_bounds[$side] length must be nu=$nu (got $(length(v)))")
        return float_type.(repeat(v, T - 1))
    end
    lvar_u = _u_bound(u_bounds, 1)
    uvar_u = _u_bound(u_bounds, 2)

    # Decision variables (flat)
    x = ExaModels.variable(core, n_x;
        lvar = float_type(x_bounds[1]),
        uvar = float_type(x_bounds[2]),
    )
    u = ExaModels.variable(core, n_u;
        lvar = lvar_u,
        uvar = uvar_u,
    )
    δ = ExaModels.variable(core, n_x)  # free slack by default

    # Parameters
    p_x0 = ExaModels.parameter(core, zeros(float_type, nx))
    p_w = ExaModels.parameter(core, zeros(float_type, n_w))
    p_target = ExaModels.parameter(core, zeros(float_type, n_x))

    # Objective: sum of per-(t,i) terms for stage cost
    # (we allow stage_cost to embed weights etc.)
    ExaModels.objective(core,
        stage_cost(t, i, x, u, p_w, nx, nu, nw)
        for t in 1:(T - 1), i in 1:nx
    )
    # Slack penalty (ρ/2)||δ||², smooth for MadNLP
    ρ = float_type(slack_penalty)
    ExaModels.objective(core,
        (ρ / 2) * δ[x_index(nx, t, i)]^2
        for t in 1:T, i in 1:nx
    )

    # Constraints (all equalities here)
    # Initial condition: x₁ = x₀
    ExaModels.constraint(core,
        x[x_index(nx, 1, i)] - p_x0[i]
        for i in 1:nx
    )
    # Dynamics: user function provides residual = 0
    ExaModels.constraint(core,
        dynamics_eq(t, i, x, u, p_w, nx, nu, nw)
        for t in 1:(T - 1), i in 1:nx
    )

    # Target constraints LAST:  x̂ - x - δ = 0
    ExaModels.constraint(core,
        p_target[x_index(nx, t, i)] - x[x_index(nx, t, i)] - δ[x_index(nx, t, i)]
        for t in 1:T, i in 1:nx
    )

    model = ExaModels.ExaModel(core)

    # We know exactly how many constraints we added before the target constraints:
    n_con_before_targets = nx + (T - 1) * nx
    target_start = n_con_before_targets + 1
    target_range = target_start:(target_start + n_x - 1)

    return DeterministicEquivalentProblem(
        core, model, x, u, δ,
        p_x0, p_w, p_target,
        nx, nu, nw, T,
        target_range,
    )
end

"""
    build_linear_tracking_problem(; kwargs...)

Convenience wrapper that builds a *simple* deterministic equivalent problem with:
- dynamics: x_{t+1} = x_t + u_t + w_t
- stage cost: (1/2) x_t^2 + (1/2) u_t^2

This is meant as an end-to-end demo and a template for your real model.
"""
function build_linear_tracking_problem(;
    horizon::Int,
    nx::Int = 1,
    backend = nothing,
    float_type::Type{<:AbstractFloat} = Float64,
    x_bounds::Tuple{<:Real,<:Real} = (-Inf, Inf),
    u_bounds::Tuple{<:Real,<:Real} = (-1.0, 1.0),
    slack_penalty::Real = 10.0,
)
    return build_deterministic_equivalent(
        horizon = horizon,
        nx = nx,
        nu = nx,
        nw = nx,
        backend = backend,
        float_type = float_type,
        x_bounds = x_bounds,
        u_bounds = u_bounds,
        slack_penalty = slack_penalty,
        dynamics_eq = default_dynamics_eq,
        stage_cost = default_stage_cost,
    )
end

# --------------------------
# Default dynamics / cost
# --------------------------

"""
Default per-dimension dynamics residual for the demo:

    x_{t+1,i} - x_{t,i} - u_{t,i} - w_{t,i} == 0

Assumes `nu == nx` and `nw == nx`.
"""
function default_dynamics_eq(t, i, x, u, w, nx::Int, nu::Int, nw::Int)
    return x[x_index(nx, t + 1, i)] -
           x[x_index(nx, t, i)] -
           u[u_index(nu, t, i)] -
           w[w_index(nw, t, i)]
end

"""
Default per-dimension stage cost term for the demo:

    (1/2) x_{t,i}^2 + (1/2) u_{t,i}^2
"""
function default_stage_cost(t, i, x, u, w, nx::Int, nu::Int, nw::Int)
    return (x[x_index(nx, t, i)]^2 + u[u_index(nu, t, i)]^2) / 2
end

# --------------------------
# Parameter updates
# --------------------------

"""
    set_x0!(prob, x0)

Update initial state parameter (length nx).
"""
function set_x0!(prob::DeterministicEquivalentProblem, x0::AbstractVector)
    length(x0) == prob.nx || error("x0 length must be nx=$(prob.nx), got $(length(x0))")
    ExaModels.set_parameter!(prob.core, prob.p_x0, x0)
    return prob
end

"""
    set_uncertainty!(prob, w)

Update disturbance trajectory parameter.
`w` must have length `(T-1)*nw` or `T*nw`; in the latter case the first `(T-1)*nw`
elements are used (the last stage's uncertainty only enters the policy rollout, not
the NLP dynamics).
"""
function set_uncertainty!(prob::DeterministicEquivalentProblem, w::AbstractVector)
    expected = (prob.horizon - 1) * prob.nw
    n = length(w)
    if n == expected
        ExaModels.set_parameter!(prob.core, prob.p_w, w)
    elseif n == prob.horizon * prob.nw
        ExaModels.set_parameter!(prob.core, prob.p_w, view(w, 1:expected))
    else
        error("w length must be (T-1)*nw=$expected or T*nw=$(prob.horizon * prob.nw), got $n")
    end
    return prob
end

"""
    set_targets!(prob, xhat)

Update target trajectory parameter (length T*nx).
"""
function set_targets!(prob::DeterministicEquivalentProblem, xhat::AbstractVector)
    expected = prob.horizon * prob.nx
    length(xhat) == expected || error("xhat length must be T*nx=$expected, got $(length(xhat))")
    ExaModels.set_parameter!(prob.core, prob.p_target, xhat)
    return prob
end

# --------------------------
# Solving & warm-start
# --------------------------

"""
    init_madnlp_cache(prob; solver_kwargs...) -> MadNLPCache

Create and store a `MadNLP.MadNLPSolver` for repeated solves.
"""
function init_madnlp_cache(prob::DeterministicEquivalentProblem; solver_kwargs...)
    solver = MadNLP.MadNLPSolver(prob.model; solver_kwargs...)
    return MadNLPCache(solver, nothing)
end

"""
    solve!(prob; solver_kwargs...) -> result

Solve once by instantiating a fresh MadNLP solver (simplest, but allocates).
"""
function solve!(prob::DeterministicEquivalentProblem; solver_kwargs...)
    return MadNLP.madnlp(prob.model; solver_kwargs...)
end

"""
    solve!(prob, cache; warmstart=true, solver_kwargs...) -> result

Solve using a cached solver instance. Optionally warm-start the primal
variables from the previous solution.
"""
function solve!(prob::DeterministicEquivalentProblem, cache::MadNLPCache;
    warmstart::Bool = true,
    solver_kwargs...,
)
    if warmstart && cache.last_result !== nothing
        # Warm-start primal from previous solution
        copyto!(NLPModels.get_x0(prob.model), cache.last_result.solution)
    end
    res = MadNLP.solve!(cache.solver; solver_kwargs...)
    cache.last_result = res
    return res
end

# --------------------------
# Extracting duals / solutions
# --------------------------

"""
    target_multipliers(prob, result) -> λ

Return the dual multipliers associated with the target constraints (∇_{x̂} Q).
"""
target_multipliers(prob::DeterministicEquivalentProblem, result) =
    result.multipliers[prob.target_con_range]

"""
    solution_components(prob, result) -> (x, u, δ)

Split the flat solution vector into state, control, and slack components.
"""
function solution_components(prob::DeterministicEquivalentProblem, result)
    n_x = prob.horizon * prob.nx
    n_u = (prob.horizon - 1) * prob.nu
    sol = result.solution
    x_sol = sol[1:n_x]
    u_sol = sol[(n_x + 1):(n_x + n_u)]
    δ_sol = sol[(n_x + n_u + 1):(n_x + n_u + n_x)]
    return (x_sol, u_sol, δ_sol)
end
