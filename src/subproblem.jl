"""
    SubproblemDEProblem

Single-stage parametric NLP for stage-wise TS-DDR training via MadDiff.

At stage ``t`` the subproblem is:

```math
q_t(x_{t-1}, w_t;\\, \\hat x_t) = \\min_{x_t, u_t, \\delta_t}
    f_t(x_t, u_t, w_t) + C_\\delta \\|\\delta_t\\|
\\quad\\text{s.t.}\\quad
    g(x_{t-1}, x_t, u_t, w_t) = 0,\\;
    x_t + \\delta_t = \\hat x_t
```

Parameters `p_x_in`, `p_w`, `p_xhat` are updated each stage via
`ExaModels.set_parameter!`.  After solving, the target-constraint duals
``\\lambda_t`` give ``\\partial q_t / \\partial \\hat x_t`` (envelope theorem)
and the dynamics duals ``\\mu_t`` give ``\\partial q_t / \\partial x_{t-1}``.

MadDiff uses these duals together with the factored KKT system for implicit
differentiation, providing the full VJP needed by the Zygote chain rule
across stages (paper Eq. 2.5).
"""
struct SubproblemDEProblem
    core
    model
    p_x_in
    p_w
    p_xhat
    nx::Int
    nu::Int
    nw::Int
    target_con_range::UnitRange{Int}
    dynamics_con_range::UnitRange{Int}
    state_var_range::UnitRange{Int}
    p_x_in_param_range::UnitRange{Int}
    p_xhat_param_range::UnitRange{Int}
end

function set_x0!(prob::SubproblemDEProblem, x_in::AbstractVector)
    ExaModels.set_parameter!(prob.core, prob.p_x_in, x_in)
    return prob
end

function set_uncertainty!(prob::SubproblemDEProblem, w::AbstractVector)
    ExaModels.set_parameter!(prob.core, prob.p_w, w)
    return prob
end

function set_targets!(prob::SubproblemDEProblem, xhat::AbstractVector)
    ExaModels.set_parameter!(prob.core, prob.p_xhat, xhat)
    return prob
end

"""
    target_multipliers(prob::SubproblemDEProblem, result)

Return ``\\lambda_t = \\partial q_t / \\partial \\hat x_t``.
"""
target_multipliers(prob::SubproblemDEProblem, result) =
    result.multipliers[prob.target_con_range]

"""
    dynamics_multipliers(prob::SubproblemDEProblem, result)

Return ``\\mu_t = \\partial q_t / \\partial x_{t-1}``.
"""
dynamics_multipliers(prob::SubproblemDEProblem, result) =
    result.multipliers[prob.dynamics_con_range]

"""
    realized_state(prob::SubproblemDEProblem, result)

Extract the realized state ``x_t^*`` from the solution.
"""
realized_state(prob::SubproblemDEProblem, result) =
    result.solution[prob.state_var_range]

"""
    build_subproblem(; kwargs...) -> SubproblemDEProblem

Build a single-stage parametric NLP for stage-wise training.

The NLP has three parameter groups:
- `p_x_in`: incoming state ``x_{t-1}`` (length `nx`)
- `p_w`: stage uncertainty ``w_t`` (length `nw`)
- `p_xhat`: target state ``\\hat x_t`` (length `nx`)

Constraints are added in order: dynamics first, then target constraints last,
so `dynamics_con_range` and `target_con_range` are contiguous slices of
`result.multipliers`.

# Keywords
- `nx`, `nu`, `nw`: state, control, uncertainty dimensions
- `backend`: ExaModels backend (`nothing` for CPU, `CUDABackend()` for GPU)
- `float_type`: numeric type (default `Float64`)
- `x_bounds`, `u_bounds`: variable bounds
- `slack_penalty`: ``\\rho/2`` coefficient on ``\\|\\delta\\|^2``
- `dynamics_eq`: `(i, x, u, p_w, p_x_in, nx, nu, nw) -> residual`
- `stage_cost`: `(i, x, u, p_w, nx, nu, nw) -> scalar`
"""
function build_subproblem(;
    nx::Int,
    nu::Int = nx,
    nw::Int = nx,
    backend = nothing,
    float_type::Type{<:AbstractFloat} = Float64,
    x_bounds::Tuple{<:Real,<:Real} = (-Inf, Inf),
    u_bounds::Tuple{<:Real,<:Real} = (-1.0, 1.0),
    slack_penalty::Real = 10.0,
    dynamics_eq::Function = _default_subproblem_dynamics,
    stage_cost::Function = _default_subproblem_cost,
)
    core = ExaModels.ExaCore(float_type; backend = backend)

    x = ExaModels.variable(core, nx;
        lvar = float_type(x_bounds[1]),
        uvar = float_type(x_bounds[2]),
    )
    u = ExaModels.variable(core, nu;
        lvar = float_type(u_bounds[1]),
        uvar = float_type(u_bounds[2]),
    )
    δ = ExaModels.variable(core, nx)

    p_x_in = ExaModels.parameter(core, zeros(float_type, nx))
    p_w    = ExaModels.parameter(core, zeros(float_type, nw))
    p_xhat = ExaModels.parameter(core, zeros(float_type, nx))

    ExaModels.objective(core,
        stage_cost(i, x, u, p_w, nx, nu, nw)
        for i in 1:nx
    )
    ρ = float_type(slack_penalty)
    ExaModels.objective(core,
        (ρ / 2) * δ[i]^2
        for i in 1:nx
    )

    ExaModels.constraint(core,
        dynamics_eq(i, x, u, p_w, p_x_in, nx, nu, nw)
        for i in 1:nx
    )

    ExaModels.constraint(core,
        p_xhat[i] - x[i] - δ[i]
        for i in 1:nx
    )

    model = ExaModels.ExaModel(core)

    dynamics_range = 1:nx
    target_range   = (nx + 1):(2 * nx)
    state_range    = 1:nx
    p_x_in_range   = 1:nx
    p_xhat_range   = (nx + nw + 1):(nx + nw + nx)

    return SubproblemDEProblem(
        core, model,
        p_x_in, p_w, p_xhat,
        nx, nu, nw,
        target_range, dynamics_range, state_range,
        p_x_in_range, p_xhat_range,
    )
end

function _default_subproblem_dynamics(i, x, u, p_w, p_x_in, nx, nu, nw)
    return x[i] - p_x_in[i] - u[i] - p_w[i]
end

function _default_subproblem_cost(i, x, u, p_w, nx, nu, nw)
    return (x[i]^2 + u[i]^2) / 2
end

"""
    build_linear_tracking_subproblem(; kwargs...) -> SubproblemDEProblem

Single-stage version of `build_linear_tracking_problem` for testing.
"""
function build_linear_tracking_subproblem(;
    nx::Int = 1,
    backend = nothing,
    float_type::Type{<:AbstractFloat} = Float64,
    x_bounds::Tuple{<:Real,<:Real} = (-Inf, Inf),
    u_bounds::Tuple{<:Real,<:Real} = (-1.0, 1.0),
    slack_penalty::Real = 10.0,
)
    return build_subproblem(;
        nx, nu = nx, nw = nx, backend, float_type,
        x_bounds, u_bounds, slack_penalty,
    )
end
