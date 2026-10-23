# embedded_deterministic_equivalent.jl
#
# Embedded-NN deterministic equivalent: the policy π_θ is inside the NLP via
# VectorNonlinearOracle.  At convergence the duals λ_t are closed-loop (joint
# NLP) and the gradient ∇_θ Q = Σ_t λ_t · ∇_θ π_θ(w_t, x*_{t-1}) follows
# from the envelope theorem — structurally identical to the open-loop formula
# but with realized states from the coupled solve.
#
# The oracle constraint is:
#     π_θ(w_t, x_{t-1}) − x_t − δ_t = 0    ∀t = 1…T
#
# One oracle for all T stages guarantees sequential LSTM evaluation with
# Flux.reset! at the top of each callback invocation. Flux.reset! on the
# threaded policies is a REAL reset (it restores Flux.initialstates), and each
# stage's policy forward advances the recurrent state exactly once, so within
# every callback the policy sees the stage sequence t = 1…T from a fresh
# initial state. Audit of per-stage advancement: oracle_f! calls the policy
# once per stage; oracle_jac!/oracle_vjp! call it inside Zygote.pullback for
# t > 1 (the pullback CONSTRUCTION runs the forward exactly once; calling the
# returned back(·) does not re-run it) and as a bare call for t = 1 — one
# forward, hence one state advance, per stage in all three callbacks.
#
# Jacobian exactness caveat: the oracle callbacks (oracle_jac!, oracle_vjp!)
# compute only the DIRECT partial ∂π_t/∂x_{t-1} via a per-stage Zygote pullback.
# For a recurrent policy whose recurrent layers see x_{t-1}, stage t's output
# also depends on x_{1..t-2} through the hidden state; those cross-stage entries
# are absent from both the sparsity pattern and the pullbacks, so the oracle
# Jacobian is exact for feedforward (stateless-in-x) policies and a structural
# approximation for such recurrent ones. When the recurrent encoder reads only
# the uncertainty w_t (as in StateConditionedPolicy and HydroReachablePolicy,
# where the combiner over [h_t; x_{t-1}] is feedforward in x), the hidden state
# does not depend on x and the direct partial IS the full derivative. This
# exactness claim SURVIVES recurrent-state threading: with the encoder reading
# only w_t, the threaded hidden state depends only on the inflow history
# w_{1..t}, never on x, so threading changes the VALUE of h_t but adds no
# ∂h_t/∂x dependence — ∂π_t/∂x_{t-1} remains the full derivative.

"""
    EmbeddedDeterministicEquivalentProblem

Container for a deterministic-equivalent NLP whose target constraints are
computed by an embedded Flux policy.

# Fields

- `core`: ExaModels core used to build variables, parameters, objectives, and
  constraints.
- `model`: ExaModels model passed to MadNLP.
- `x`, `u`, `δ`: Flat state, control, and target-slack variables.
- `p_x0`, `p_w`: Initial-state and uncertainty parameters.
- `policy`: Flux policy evaluated by the nonlinear oracle.
- `nx`, `nu`, `nw`, `horizon`: State, control, uncertainty, and time dimensions.
- `target_con_range`: Range of oracle-constraint multipliers in solver results.
- `_w_buf`, `_x0_buf`: Mutable host buffers captured by oracle callbacks.

# Notes

This problem is analogous to `DeterministicEquivalentProblem`, but the explicit
target parameter is replaced by a `VectorNonlinearOracle` enforcing
``\\pi_\\theta(w_t, x_{t-1}) - x_t - \\delta_t = 0``. The oracle closures capture
`policy` by reference, so changing Flux parameters between solves changes the
NLP callbacks without rebuilding the ExaModel.
"""
struct EmbeddedDeterministicEquivalentProblem{P}
    core
    model
    x
    u
    δ
    p_x0
    p_w
    policy::P
    nx::Int
    nu::Int
    nw::Int
    horizon::Int
    target_con_range::UnitRange{Int}
    # mutable buffers captured by oracle closures
    _w_buf::Vector{Float64}
    _x0_buf::Vector{Float64}
end

"""
    set_x0!(prob::EmbeddedDeterministicEquivalentProblem, x0::AbstractVector)

Update the initial state used by the embedded problem and oracle callbacks.

# Arguments

- `prob::EmbeddedDeterministicEquivalentProblem`: Embedded problem to update.
- `x0::AbstractVector`: Initial state with length `prob.nx`.

# Returns

- `prob`: The updated problem.

# Throws

Throws an error if `length(x0) != prob.nx`.

# Notes

The value is written both to the ExaModels parameter and to the oracle closure
buffer, ensuring the policy callback sees the same initial state as the NLP.
"""
function set_x0!(prob::EmbeddedDeterministicEquivalentProblem, x0::AbstractVector)
    length(x0) == prob.nx || error("x0 length must be nx=$(prob.nx), got $(length(x0))")
    ExaModels.set_parameter!(prob.core, prob.p_x0, x0)
    copyto!(prob._x0_buf, Float64.(x0))
    return prob
end

"""
    set_uncertainty!(prob::EmbeddedDeterministicEquivalentProblem, w::AbstractVector)

Update the disturbance trajectory used by the embedded problem and oracle
callbacks.

# Arguments

- `prob::EmbeddedDeterministicEquivalentProblem`: Embedded problem to update.
- `w::AbstractVector`: Disturbance trajectory with length `(T - 1) * nw` or
  `T * nw`.

# Returns

- `prob`: The updated problem.

# Throws

Throws an error if `w` has any length other than `(T - 1) * nw` or `T * nw`.

# Notes

The first `(T - 1) * nw` entries are passed to the ExaModels dynamics
parameter. The full length-`T * nw` trajectory is retained in the oracle buffer
when provided, because the policy callback may evaluate a final-stage
disturbance.
"""
function set_uncertainty!(prob::EmbeddedDeterministicEquivalentProblem, w::AbstractVector)
    expected = (prob.horizon - 1) * prob.nw
    full_len = prob.horizon * prob.nw
    n = length(w)
    if n == expected
        ExaModels.set_parameter!(prob.core, prob.p_w, w)
        copyto!(view(prob._w_buf, 1:expected), Float64.(w))
    elseif n == full_len
        ExaModels.set_parameter!(prob.core, prob.p_w, view(w, 1:expected))
        copyto!(prob._w_buf, Float64.(w))
    else
        error("w length must be (T-1)*nw=$expected or T*nw=$full_len, got $n")
    end
    return prob
end

"""
    set_targets!(::EmbeddedDeterministicEquivalentProblem, ::AbstractVector) -> nothing

Ignore explicit targets for embedded deterministic-equivalent problems.

# Arguments

- `::EmbeddedDeterministicEquivalentProblem`: Embedded problem whose targets are
  generated by the policy oracle.
- `::AbstractVector`: Ignored target vector.

# Returns

- `nothing`.

# Notes

Embedded problems compute targets inline through the nonlinear oracle instead of
storing them in an NLP parameter.
"""
function set_targets!(::EmbeddedDeterministicEquivalentProblem, ::AbstractVector)
    return nothing
end

"""
    invalidate_policy_cache!(embedded_de)

Invalidate policy-dependent caches for an embedded deterministic-equivalent
problem.

# Arguments

- `embedded_de`: Embedded deterministic-equivalent problem.

# Returns

- `embedded_de`.

# Notes

The generic embedded problem evaluates the policy directly in each oracle
callback and has no cache to invalidate. Specialized embedded problem types may
extend this hook when their oracle stores policy-dependent intermediates across
solver calls.
"""
function invalidate_policy_cache!(embedded_de)
    return embedded_de
end

"""
    build_embedded_deterministic_equivalent(policy; kwargs...)

Build a deterministic-equivalent NLP with a Flux policy embedded as a nonlinear
oracle.

# Arguments

- `policy`: Flux model mapping each stage input to an `nx`-vector target.

# Keywords

- `horizon::Int`: Number of stages. States are indexed over `1:horizon`;
  controls and uncertainties over `1:(horizon - 1)`.
- `nx::Int`: State dimension and policy output dimension.
- `nu::Int = nx`: Control dimension.
- `nw::Int = nx`: Uncertainty dimension and first part of the policy input.
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

- `EmbeddedDeterministicEquivalentProblem`: Problem supporting `set_x0!`,
  `set_uncertainty!`, `target_multipliers`, and `solution_components`.

# Throws

Throws an error if dimensions are invalid, if vector control bounds have the
wrong length, or if the default dynamics/cost are used with dimensions other
than `nu == nx` and `nw == nx`.

# Notes

The oracle enforces
``\\pi_\\theta(w_t, x_{t-1}) - x_t - \\delta_t = 0`` and is added last so its
multipliers form a contiguous trailing slice of `result.multipliers`. This
matches the open-loop deterministic-equivalent convention while allowing the
policy to depend on realized previous states.

The oracle Jacobian and vector-Jacobian callbacks differentiate each stage
independently, providing only the direct partial ``\\partial \\pi_t /
\\partial x_{t-1}``. If the policy's recurrent layers consume ``x_{t-1}``,
stage ``t``'s output also depends on ``x_{1..t-2}`` through the hidden state,
and those cross-stage Jacobian entries are omitted — the reported Jacobian is
then a structural approximation. The Jacobian is exact whenever the recurrent
part of the policy reads only ``w_t`` and the state enters through a
feedforward head (the [`StateConditionedPolicy`](@ref) architecture), because
then the hidden state carries no dependence on ``x``. This holds unchanged
with recurrent-state threading: the threaded hidden state is a function of
``w_{1..t}`` only, so it changes the value of ``h_t`` but introduces no
``\\partial h_t / \\partial x`` term.

Every callback resets the policy's recurrent state at its top and evaluates
the stages in order, advancing the state exactly once per stage (for
``t > 1`` the forward runs during `Zygote.pullback` construction; the
returned pullback does not re-run it).
"""
function build_embedded_deterministic_equivalent(
    policy;
    horizon::Int,
    nx::Int,
    nu::Int = nx,
    nw::Int = nx,
    backend = nothing,
    float_type::Type{<:AbstractFloat} = Float64,
    x_bounds::Tuple{<:Real,<:Real} = (-Inf, Inf),
    u_bounds = (-Inf, Inf),
    slack_penalty::Real = 1.0,
    dynamics_eq::Function = default_dynamics_eq,
    stage_cost::Function = default_stage_cost,
)
    horizon ≥ 2 || error("horizon must be ≥ 2 (got $horizon)")
    nx ≥ 1 || error("nx must be ≥ 1 (got $nx)")
    nu ≥ 1 || error("nu must be ≥ 1 (got $nu)")
    nw ≥ 1 || error("nw must be ≥ 1 (got $nw)")

    if (dynamics_eq === default_dynamics_eq || stage_cost === default_stage_cost) && (nu != nx || nw != nx)
        error("Default dynamics/cost assume nu == nx and nw == nx.")
    end

    T = horizon
    n_x = T * nx
    n_u = (T - 1) * nu

    core = ExaModels.ExaCore(float_type; backend = backend)

    function _u_bound(b, side)
        v = b[side]
        v isa AbstractVector || return float_type(v)
        length(v) == nu || error("u_bounds[$side] length must be nu=$nu")
        return float_type.(repeat(v, T - 1))
    end
    lvar_u = _u_bound(u_bounds, 1)
    uvar_u = _u_bound(u_bounds, 2)

    x = ExaModels.variable(core, n_x;
        lvar = float_type(x_bounds[1]),
        uvar = float_type(x_bounds[2]),
    )
    u = ExaModels.variable(core, n_u;
        lvar = lvar_u,
        uvar = uvar_u,
    )
    δ = ExaModels.variable(core, n_x)

    p_x0 = ExaModels.parameter(core, zeros(float_type, nx))
    p_w = ExaModels.parameter(core, zeros(float_type, (T - 1) * nw))

    ExaModels.objective(core,
        stage_cost(t, i, x, u, p_w, nx, nu, nw)
        for t in 1:(T - 1), i in 1:nx
    )
    ρ = float_type(slack_penalty)
    ExaModels.objective(core,
        (ρ / 2) * δ[x_index(nx, t, i)]^2
        for t in 1:T, i in 1:nx
    )

    ExaModels.constraint(core,
        x[x_index(nx, 1, i)] - p_x0[i]
        for i in 1:nx
    )
    ExaModels.constraint(core,
        dynamics_eq(t, i, x, u, p_w, nx, nu, nw)
        for t in 1:(T - 1), i in 1:nx
    )

    n_con_before_oracle = nx + (T - 1) * nx

    # ── Oracle buffers (mutated by set_x0! / set_uncertainty!) ───────────
    w_buf  = zeros(Float64, T * nw)
    x0_buf = zeros(Float64, nx)

    nvar_total = n_x + n_u + n_x   # x, u, δ
    x_start = 1
    δ_start = n_x + n_u + 1

    # ── Pre-allocated oracle buffers ────────────────────────────────────
    _x_prev  = zeros(Float32, nx)
    _w_t     = zeros(Float32, nw)
    _input   = zeros(Float32, nw + nx)
    _J       = zeros(Float32, nx, nx)
    _e       = zeros(Float32, nx)
    _λ_t     = zeros(Float32, nx)

    function _fill_x_prev!(t, xv)
        for i in 1:nx
            _x_prev[i] = (t == 1) ?
                Float32(x0_buf[i]) :
                Float32(xv[x_start + (t-2)*nx + i - 1])
        end
        return _x_prev
    end

    function _fill_w_t!(t)
        for j in 1:nw
            _w_t[j] = Float32(w_buf[(t-1)*nw + j])
        end
        return _w_t
    end

    function _fill_input!(t, xv)
        _fill_w_t!(t)
        _fill_x_prev!(t, xv)
        copyto!(view(_input, 1:nw), _w_t)
        copyto!(view(_input, nw+1:nw+nx), _x_prev)
        return _input
    end

    # ── Oracle callbacks ─────────────────────────────────────────────────

    function oracle_f!(c, xv)
        # Real reset: the stage loop below starts from the initial recurrent
        # state and each policy call advances it exactly once.
        Flux.reset!(policy)
        for t in 1:T
            _fill_input!(t, xv)
            nn_out = policy(_input)
            for i in 1:nx
                row = (t - 1) * nx + i
                xi = x_start + (t-1)*nx + i - 1
                di = δ_start + (t-1)*nx + i - 1
                c[row] = Float64(nn_out[i]) - xv[xi] - xv[di]
            end
        end
        return nothing
    end

    # NOTE: this Jacobian holds only the per-stage direct partial ∂π_t/∂x_{t-1}.
    # Cross-stage terms through a recurrent hidden state that depends on x are
    # not represented (see file-top comment); exact when the recurrent encoder
    # reads only w_t, as in StateConditionedPolicy / HydroReachablePolicy.
    function oracle_jac!(vals, xv)
        # Real reset; each stage advances the recurrent state exactly once:
        # for t > 1 the forward runs during Zygote.pullback construction, and
        # calling back(·) repeatedly does NOT re-run it; t = 1 is a bare call.
        Flux.reset!(policy)
        k = 0
        for t in 1:T
            _fill_x_prev!(t, xv)
            _fill_w_t!(t)

            nn_jac_xprev = if t > 1
                _, back = Zygote.pullback(xp -> policy(vcat(_w_t, xp)), _x_prev)
                fill!(_J, 0f0)
                for row in 1:nx
                    fill!(_e, 0f0)
                    _e[row] = 1.0f0
                    col_grad = back(_e)[1]
                    if col_grad !== nothing
                        _J[row, :] .= col_grad
                    end
                end
                _J
            else
                # t = 1: no x-Jacobian block, but the forward must still run
                # once so the recurrent state advances to stage 2.
                policy(vcat(_w_t, _x_prev))
                nothing
            end

            for i in 1:nx
                k += 1; vals[k] = -1.0
                k += 1; vals[k] = -1.0
                if t > 1
                    for j in 1:nx
                        k += 1; vals[k] = Float64(nn_jac_xprev[i, j])
                    end
                end
            end
        end
        return nothing
    end

    function oracle_vjp!(Jtv, xv, λ)
        fill!(Jtv, 0.0)
        # Real reset; same one-advance-per-stage discipline as oracle_jac!.
        Flux.reset!(policy)
        for t in 1:T
            _fill_x_prev!(t, xv)
            _fill_w_t!(t)

            for i in 1:nx
                _λ_t[i] = Float32(λ[(t-1)*nx + i])
                xi = x_start + (t-1)*nx + i - 1
                di = δ_start + (t-1)*nx + i - 1
                Jtv[xi] -= λ[(t-1)*nx + i]
                Jtv[di] -= λ[(t-1)*nx + i]
            end

            if t > 1
                _, back = Zygote.pullback(xp -> policy(vcat(_w_t, xp)), _x_prev)
                dinput = back(_λ_t)[1]
                if dinput !== nothing
                    for j in 1:nx
                        xj = x_start + (t-2)*nx + j - 1
                        Jtv[xj] += Float64(dinput[j])
                    end
                end
            else
                # t = 1: no x_prev contribution, but run the forward once so
                # the recurrent state advances to stage 2.
                policy(vcat(_w_t, _x_prev))
            end
        end
        return nothing
    end

    # ── Sparsity pattern ─────────────────────────────────────────────────
    jac_r = Int[]
    jac_c = Int[]
    for t in 1:T
        for i in 1:nx
            row = (t - 1) * nx + i
            xi = x_start + (t-1)*nx + i - 1
            di = δ_start + (t-1)*nx + i - 1
            push!(jac_r, row); push!(jac_c, xi)    # ∂g/∂x_{t,i}
            push!(jac_r, row); push!(jac_c, di)    # ∂g/∂δ_{t,i}
            if t > 1
                for j in 1:nx
                    xj = x_start + (t-2)*nx + j - 1
                    push!(jac_r, row); push!(jac_c, xj)  # ∂g/∂x_{t-1,j}
                end
            end
        end
    end

    oracle = ExaModels.VectorNonlinearOracle(
        nvar  = nvar_total,
        ncon  = n_x,
        nnzj  = length(jac_r),
        jac_rows = jac_r,
        jac_cols = jac_c,
        lcon  = zeros(n_x),
        ucon  = zeros(n_x),
        f!    = oracle_f!,
        jac!  = oracle_jac!,
        vjp!  = oracle_vjp!,
        adapt = Val(true),
    )
    ExaModels.constraint(core, oracle)

    model = ExaModels.ExaModel(core)

    target_start = n_con_before_oracle + 1
    target_range = target_start:(target_start + n_x - 1)

    return EmbeddedDeterministicEquivalentProblem(
        core, model, x, u, δ,
        p_x0, p_w,
        policy,
        nx, nu, nw, T,
        target_range,
        w_buf, x0_buf,
    )
end

"""
    target_multipliers(prob::EmbeddedDeterministicEquivalentProblem, result) -> λ

Return the dual multipliers associated with the embedded oracle constraints.

# Arguments

- `prob::EmbeddedDeterministicEquivalentProblem`: Problem that defines the
  multiplier slice.
- `result`: MadNLP result containing `multipliers`.

# Returns

- `λ`: Multipliers in `result.multipliers[prob.target_con_range]`.

# Notes

The multipliers correspond to the constraints
``\\pi_\\theta(w_t, x_{t-1}) - x_t - \\delta_t = 0``.
"""
target_multipliers(prob::EmbeddedDeterministicEquivalentProblem, result) =
    result.multipliers[prob.target_con_range]

"""
    solution_components(prob::EmbeddedDeterministicEquivalentProblem, result) -> (x, u, δ)

Split the flat solution vector into state, control, and slack components.

# Arguments

- `prob::EmbeddedDeterministicEquivalentProblem`: Problem that defines component
  sizes.
- `result`: MadNLP result containing `solution`.

# Returns

- `(x, u, δ)`: Flat slices of the primal solution for states, controls, and
  target slacks.
"""
function solution_components(prob::EmbeddedDeterministicEquivalentProblem, result)
    n_x = prob.horizon * prob.nx
    n_u = (prob.horizon - 1) * prob.nu
    sol = result.solution
    x_sol = sol[1:n_x]
    u_sol = sol[(n_x + 1):(n_x + n_u)]
    δ_sol = sol[(n_x + n_u + 1):(n_x + n_u + n_x)]
    return (x_sol, u_sol, δ_sol)
end
