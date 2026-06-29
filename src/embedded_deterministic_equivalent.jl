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
# Flux.reset! at the top of each callback invocation.

"""
    EmbeddedDeterministicEquivalentProblem

Like `DeterministicEquivalentProblem` but the target constraints are replaced
by a `VectorNonlinearOracle` that evaluates the Flux policy inline.

The oracle closures capture the policy **by reference**: updating Flux
parameters between solves automatically changes the NLP — no rebuild needed.
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

# Duck-typing: set_x0! and set_uncertainty! update both ExaModels parameters
# AND the oracle's closure buffers so the callbacks see the new data.

function set_x0!(prob::EmbeddedDeterministicEquivalentProblem, x0::AbstractVector)
    length(x0) == prob.nx || error("x0 length must be nx=$(prob.nx), got $(length(x0))")
    ExaModels.set_parameter!(prob.core, prob.p_x0, x0)
    copyto!(prob._x0_buf, Float64.(x0))
    return prob
end

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

# No-op: targets live inside the oracle, not as parameters.
function set_targets!(::EmbeddedDeterministicEquivalentProblem, ::AbstractVector)
    return nothing
end

"""
    invalidate_policy_cache!(embedded_de)

Hook for embedded problems whose nonlinear oracle caches policy-dependent
intermediates across solver calls.  Generic embedded problems evaluate the
policy directly in each callback and do not need invalidation.
"""
function invalidate_policy_cache!(embedded_de)
    return embedded_de
end

"""
    build_embedded_deterministic_equivalent(policy; kwargs...)

Build a deterministic-equivalent NLP with the Flux `policy` embedded via
`VectorNonlinearOracle`.

Same keyword interface as `build_deterministic_equivalent` except:
- `policy` is a positional argument (the Flux model to embed)
- No `p_target` parameter — targets are computed inline by the oracle
- The oracle is added **last** so its multipliers are a contiguous trailing
  slice of `result.multipliers` (same convention as the open-loop version)

The returned problem supports `set_x0!`, `set_uncertainty!`, `target_multipliers`,
and `solution_components` with the same signatures.
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

    function oracle_jac!(vals, xv)
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

target_multipliers(prob::EmbeddedDeterministicEquivalentProblem, result) =
    result.multipliers[prob.target_con_range]

function solution_components(prob::EmbeddedDeterministicEquivalentProblem, result)
    n_x = prob.horizon * prob.nx
    n_u = (prob.horizon - 1) * prob.nu
    sol = result.solution
    x_sol = sol[1:n_x]
    u_sol = sol[(n_x + 1):(n_x + n_u)]
    δ_sol = sol[(n_x + n_u + 1):(n_x + n_u + n_x)]
    return (x_sol, u_sol, δ_sol)
end
