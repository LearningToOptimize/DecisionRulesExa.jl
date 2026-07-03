# deterministic_equivalent.jl
#
# Deterministic equivalent workflow (TS-GDR / TS-DDR style):
#   - Policy predicts per-stage target states x̂_t
#   - Control optimizer solves a deterministic multi-period NLP to hit those targets
#   - Dual multipliers on the target constraints provide ∇_{x̂} Q, enabling training
#     without differentiating through the solver.

"""
    DeterministicEquivalentProblem

Container for an ExaModels parametric NLP representing a deterministic
equivalent subproblem.

# Fields

- `core`: ExaModels core used to build variables, parameters, objectives, and
  constraints.
- `model`: ExaModels model passed to MadNLP.
- `x`, `u`, `δ`: Flat state, control, and target-slack variables.
- `p_x0`, `p_w`, `p_target`: Initial-state, uncertainty, and target parameters.
- `nx`, `nu`, `nw`, `horizon`: State, control, uncertainty, and time dimensions.
- `target_con_range`: Range of target-constraint multipliers in solver results.

# Notes

The subproblem has the form

```math
Q(w, \\hat{x}) =
    \\min_{x,u,\\delta}
        \\sum_t c_t(x_t, u_t, w_t) + \\frac{\\rho}{2}\\|\\delta\\|^2
```

subject to the initial condition, dynamics constraints, and target constraints
``\\hat{x}_t - x_t - \\delta_t = 0``. The target constraints are added last so
their multipliers occupy the contiguous slice `target_con_range`.
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

Cache for reusing a MadNLP solver across repeated solves.

# Fields

- `solver`: Cached `MadNLP.MadNLPSolver`.
- `last_result`: Most recent solver result, used for optional warm starts.

# Notes

Reusing the solver can avoid repeated symbolic setup and can warm-start the
next primal iterate from the previous solution.
"""
mutable struct MadNLPCache
    solver
    last_result
end

"""
    build_deterministic_equivalent(; kwargs...) -> DeterministicEquivalentProblem

Build a deterministic-equivalent dynamic nonlinear program.

# Keywords

- `horizon::Int`: Number of stages. States are indexed over `1:horizon`;
  controls and uncertainties over `1:(horizon - 1)`.
- `nx::Int`: State dimension.
- `nu::Int = nx`: Control dimension.
- `nw::Int = nx`: Uncertainty dimension.
- `backend = nothing`: ExaModels backend, such as `nothing` for CPU or a CUDA
  backend for GPU execution.
- `float_type::Type{<:AbstractFloat} = Float64`: Scalar type used by the model.
- `x_bounds::Tuple{<:Real,<:Real} = (-Inf, Inf)`: Lower and upper bounds applied
  to every state variable.
- `u_bounds = (-Inf, Inf)`: Control bounds, either a scalar `(lb, ub)` tuple or
  a tuple of length-`nu` lower and upper bound vectors.
- `slack_penalty::Real = 1.0`: Nonnegative target-slack penalty weight ``ρ``.
- `dynamics_eq::Function = default_dynamics_eq`: Function
  `(t, i, x, u, w, nx, nu, nw) -> residual` defining one scalar dynamics
  equality.
- `stage_cost::Function = default_stage_cost`: Function
  `(t, i, x, u, w, nx, nu, nw) -> term` defining one scalar objective term.

# Returns

- `DeterministicEquivalentProblem`: A mutable problem container with parameters
  that can be updated by `set_x0!`, `set_uncertainty!`, and `set_targets!`.

# Throws

Throws an error if dimensions are invalid, if vector control bounds have the
wrong length, or if the default dynamics/cost are used with dimensions other
than `nu == nx` and `nw == nx`.

# Notes

All variables are stored in flat vectors. Target constraints are added last and
written as ``\\hat{x} - x - \\delta = 0`` so the envelope theorem identifies their
multipliers with gradients with respect to the target trajectory.
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

Build the default linear-quadratic tracking demonstration problem.

# Keywords

- `horizon::Int`: Number of stages.
- `nx::Int = 1`: State, control, and uncertainty dimension.
- `backend = nothing`: ExaModels backend.
- `float_type::Type{<:AbstractFloat} = Float64`: Scalar type used by the model.
- `x_bounds::Tuple{<:Real,<:Real} = (-Inf, Inf)`: State bounds.
- `u_bounds::Tuple{<:Real,<:Real} = (-1.0, 1.0)`: Control bounds.
- `slack_penalty::Real = 10.0`: Target-slack penalty weight.

# Returns

- `DeterministicEquivalentProblem`: Problem with dynamics
  ``x_{t+1} = x_t + u_t + w_t`` and stage cost
  ``(x_t^2 + u_t^2) / 2``.

# Notes

This helper is intended as a small end-to-end example and as a template for
model-specific deterministic-equivalent builders.
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
    default_dynamics_eq(t, i, x, u, w, nx::Int, nu::Int, nw::Int)

Return the default scalar dynamics residual.

# Arguments

- `t`: Stage index.
- `i`: State component index.
- `x`: Flat state variable vector.
- `u`: Flat control variable vector.
- `w`: Flat uncertainty parameter vector.
- `nx::Int`: State dimension.
- `nu::Int`: Control dimension.
- `nw::Int`: Uncertainty dimension.

# Returns

- Scalar residual ``x_{t+1,i} - x_{t,i} - u_{t,i} - w_{t,i}``.

# Notes

The default residual assumes `nu == nx` and `nw == nx`.
"""
function default_dynamics_eq(t, i, x, u, w, nx::Int, nu::Int, nw::Int)
    return x[x_index(nx, t + 1, i)] -
           x[x_index(nx, t, i)] -
           u[u_index(nu, t, i)] -
           w[w_index(nw, t, i)]
end

"""
    default_stage_cost(t, i, x, u, w, nx::Int, nu::Int, nw::Int)

Return the default scalar stage-cost term.

# Arguments

- `t`: Stage index.
- `i`: State/control component index.
- `x`: Flat state variable vector.
- `u`: Flat control variable vector.
- `w`: Flat uncertainty parameter vector.
- `nx::Int`: State dimension.
- `nu::Int`: Control dimension.
- `nw::Int`: Uncertainty dimension.

# Returns

- Scalar cost ``(x_{t,i}^2 + u_{t,i}^2) / 2``.
"""
function default_stage_cost(t, i, x, u, w, nx::Int, nu::Int, nw::Int)
    return (x[x_index(nx, t, i)]^2 + u[u_index(nu, t, i)]^2) / 2
end

# --------------------------
# Parameter updates
# --------------------------

"""
    set_x0!(prob::DeterministicEquivalentProblem, x0::AbstractVector)

Update the initial-state parameter.

# Arguments

- `prob::DeterministicEquivalentProblem`: Problem to update.
- `x0::AbstractVector`: Initial state with length `prob.nx`.

# Returns

- `prob`: The updated problem.

# Throws

Throws an error if `length(x0) != prob.nx`.
"""
function set_x0!(prob::DeterministicEquivalentProblem, x0::AbstractVector)
    length(x0) == prob.nx || error("x0 length must be nx=$(prob.nx), got $(length(x0))")
    ExaModels.set_parameter!(prob.core, prob.p_x0, x0)
    return prob
end

"""
    set_uncertainty!(prob::DeterministicEquivalentProblem, w::AbstractVector)

Update the disturbance-trajectory parameter.

# Arguments

- `prob::DeterministicEquivalentProblem`: Problem to update.
- `w::AbstractVector`: Disturbance trajectory with length `(T - 1) * nw` or
  `T * nw`.

# Returns

- `prob`: The updated problem.

# Throws

Throws an error if `w` has any length other than `(T - 1) * nw` or `T * nw`.

# Notes

When `w` has length `T * nw`, only the first `(T - 1) * nw` entries are used in
the NLP dynamics. The final-stage uncertainty may be needed by a policy rollout
but does not enter these dynamics constraints.
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
    set_targets!(prob::DeterministicEquivalentProblem, xhat::AbstractVector)

Update the target-trajectory parameter.

# Arguments

- `prob::DeterministicEquivalentProblem`: Problem to update.
- `xhat::AbstractVector`: Target trajectory with length `prob.horizon * prob.nx`.

# Returns

- `prob`: The updated problem.

# Throws

Throws an error if `length(xhat) != prob.horizon * prob.nx`.
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

Create a cached MadNLP solver for repeated solves.

# Arguments

- `prob::DeterministicEquivalentProblem`: Problem whose ExaModel will be solved.

# Keywords

- `solver_kwargs...`: Keyword arguments forwarded to `MadNLP.MadNLPSolver`.

# Returns

- `MadNLPCache`: Cache containing the solver and an initially empty
  `last_result`.
"""
function init_madnlp_cache(prob::DeterministicEquivalentProblem; solver_kwargs...)
    solver = MadNLP.MadNLPSolver(prob.model; solver_kwargs...)
    return MadNLPCache(solver, nothing)
end

"""
    solve!(prob::DeterministicEquivalentProblem; solver_kwargs...) -> result

Solve a deterministic-equivalent problem with a fresh MadNLP solver.

# Arguments

- `prob::DeterministicEquivalentProblem`: Problem to solve.

# Keywords

- `solver_kwargs...`: Keyword arguments forwarded to `MadNLP.madnlp`.

# Returns

- `result`: MadNLP result object.

# Notes

This path is simple but allocates a new solver for every call.
"""
function solve!(prob::DeterministicEquivalentProblem; solver_kwargs...)
    return MadNLP.madnlp(prob.model; solver_kwargs...)
end

"""
    solve!(prob::DeterministicEquivalentProblem, cache::MadNLPCache; warmstart=true, solver_kwargs...) -> result

Solve a deterministic-equivalent problem with a cached MadNLP solver.

# Arguments

- `prob::DeterministicEquivalentProblem`: Problem to solve.
- `cache::MadNLPCache`: Cached solver and previous result.

# Keywords

- `warmstart::Bool = true`: Whether to copy the previous primal solution into
  the model initial point before solving.
- `solver_kwargs...`: Keyword arguments forwarded to `MadNLP.solve!`.

# Returns

- `result`: MadNLP result object, also stored in `cache.last_result`.

# Notes

Warm starts are used only when `warmstart` is true and `cache.last_result` is
available.

MadNLP's iteration counter `cnt.k` is cumulative across `solve!` calls on the
same solver instance and is never reset by MadNLP itself. Without resetting it
(together with `cnt.acceptable_cnt` and `cnt.start_time`), repeated solves on a
cached solver eventually exhaust `max_iter` spuriously — the counters are reset
here before each solve so every call gets its intended per-solve iteration and
wall-clock budget. This mirrors the reset in `_solve!` (training.jl) and does
not change any numerics of an individual solve.
"""
function solve!(prob::DeterministicEquivalentProblem, cache::MadNLPCache;
    warmstart::Bool = true,
    solver_kwargs...,
)
    if warmstart && cache.last_result !== nothing
        # Warm-start primal from previous solution
        copyto!(NLPModels.get_x0(prob.model), cache.last_result.solution)
    end
    # Reset per-solve iteration budget (cnt.k is cumulative in MadNLP).
    cache.solver.cnt.k              = 0                     # reset iteration counter
    cache.solver.cnt.acceptable_cnt = 0                     # reset acceptable-step counter
    cache.solver.cnt.start_time     = time()                # reset wall-clock timer
    res = MadNLP.solve!(cache.solver; solver_kwargs...)
    cache.last_result = res
    return res
end

# --------------------------
# Extracting duals / solutions
# --------------------------

"""
    target_multipliers(prob::DeterministicEquivalentProblem, result) -> λ

Return the dual multipliers associated with target constraints.

# Arguments

- `prob::DeterministicEquivalentProblem`: Problem that defines the multiplier
  slice.
- `result`: MadNLP result containing `multipliers`.

# Returns

- `λ`: Multipliers in `result.multipliers[prob.target_con_range]`.

# Notes

With target constraints written as ``\\hat{x} - x - \\delta = 0``, these
multipliers are the envelope-theorem derivatives with respect to the target
trajectory.
"""
target_multipliers(prob::DeterministicEquivalentProblem, result) =
    result.multipliers[prob.target_con_range]

"""
    solution_components(prob::DeterministicEquivalentProblem, result) -> (x, u, δ)

Split the flat solution vector into state, control, and slack components.

# Arguments

- `prob::DeterministicEquivalentProblem`: Problem that defines component sizes.
- `result`: MadNLP result containing `solution`.

# Returns

- `(x, u, δ)`: Flat slices of the primal solution for states, controls, and
  target slacks.
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
