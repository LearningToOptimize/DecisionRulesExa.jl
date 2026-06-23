"""
Finite-difference validation for MadDiff-backed subproblem gradients.

Run standalone (not included from runtests.jl) because the klamike forks
are required (MadDiff, ParametricNLPModels, etc.) and not available on CI.

Usage (in SLURM job with /tmp/ar_julia_depot):
    julia --project=. test/test_subproblem_maddiff.jl
"""

using Test
using DecisionRulesExa
using DecisionRulesExa: _solve_subproblem_diffable, _make_subproblem_solver,
    SubproblemSolverState
using MadNLP
import MadDiff
using Flux
using Random
using Zygote
using ChainRulesCore
using LinearAlgebra

const MADNLP_KWARGS = (print_level = MadNLP.ERROR, max_iter = 500, tol = 1e-8)

# ── Test 1: Single-stage envelope-theorem VJP (dobj=1) vs finite differences ──

@testset "Single-stage envelope VJP vs FD" begin
    nx = 2
    prob = build_linear_tracking_subproblem(
        nx = nx,
        slack_penalty = 10.0,
        u_bounds = (-2.0, 2.0),
    )

    x_in  = [1.5, -0.8]
    w_t   = [0.2, -0.1]
    xhat  = [2.0, -1.5]

    solver_state = _make_subproblem_solver(prob.model, MADNLP_KWARGS)

    obj0, x_real0, ok0 = _solve_subproblem_diffable(
        prob, x_in, w_t, xhat, solver_state;
        warmstart = false, madnlp_kwargs = MADNLP_KWARGS,
    )
    @test ok0

    # Get rrule VJP with dobj=1, dL/dx_real=0
    (_, pullback) = ChainRulesCore.rrule(
        _solve_subproblem_diffable,
        prob, x_in, w_t, xhat, solver_state;
        warmstart = false, madnlp_kwargs = MADNLP_KWARGS,
    )
    tangents = pullback((1.0, zeros(nx), nothing))
    d_x_in_ad  = tangents[3]
    d_xhat_ad  = tangents[5]

    # Finite differences
    ε = 1e-5
    d_x_in_fd = zeros(nx)
    d_xhat_fd = zeros(nx)

    for i in 1:nx
        # ∂obj/∂x_in[i]
        x_in_p = copy(x_in); x_in_p[i] += ε
        ss_p = _make_subproblem_solver(prob.model, MADNLP_KWARGS)
        obj_p, _, ok_p = _solve_subproblem_diffable(
            prob, x_in_p, w_t, xhat, ss_p;
            warmstart = false, madnlp_kwargs = MADNLP_KWARGS,
        )
        @test ok_p
        d_x_in_fd[i] = (obj_p - obj0) / ε
    end

    for i in 1:nx
        # ∂obj/∂xhat[i]
        xhat_p = copy(xhat); xhat_p[i] += ε
        ss_p = _make_subproblem_solver(prob.model, MADNLP_KWARGS)
        obj_p, _, ok_p = _solve_subproblem_diffable(
            prob, x_in, w_t, xhat_p, ss_p;
            warmstart = false, madnlp_kwargs = MADNLP_KWARGS,
        )
        @test ok_p
        d_xhat_fd[i] = (obj_p - obj0) / ε
    end

    @test d_x_in_ad ≈ d_x_in_fd atol = 1e-3
    @test d_xhat_ad ≈ d_xhat_fd atol = 1e-3
    println("  Envelope VJP d_x_in: AD=$d_x_in_ad, FD=$d_x_in_fd")
    println("  Envelope VJP d_xhat: AD=$d_xhat_ad, FD=$d_xhat_fd")
end

# ── Test 2: State sensitivity VJP (dL/dx_real=e_i) vs FD ─────────────────────

@testset "State sensitivity VJP vs FD" begin
    nx = 2
    prob = build_linear_tracking_subproblem(
        nx = nx,
        slack_penalty = 10.0,
        u_bounds = (-2.0, 2.0),
    )

    x_in  = [1.5, -0.8]
    w_t   = [0.2, -0.1]
    xhat  = [2.0, -1.5]

    solver_state = _make_subproblem_solver(prob.model, MADNLP_KWARGS)
    _, x_real0, ok0 = _solve_subproblem_diffable(
        prob, x_in, w_t, xhat, solver_state;
        warmstart = false, madnlp_kwargs = MADNLP_KWARGS,
    )
    @test ok0

    ε = 1e-5

    for k in 1:nx
        # VJP with Δx_real = e_k
        e_k = zeros(nx); e_k[k] = 1.0

        (_, pullback) = ChainRulesCore.rrule(
            _solve_subproblem_diffable,
            prob, x_in, w_t, xhat, solver_state;
            warmstart = false, madnlp_kwargs = MADNLP_KWARGS,
        )
        tangents = pullback((0.0, e_k, nothing))
        d_x_in_ad  = tangents[3]
        d_xhat_ad  = tangents[5]

        # FD: ∂x_real[k]/∂x_in[j]
        d_x_in_fd = zeros(nx)
        for j in 1:nx
            x_in_p = copy(x_in); x_in_p[j] += ε
            ss_p = _make_subproblem_solver(prob.model, MADNLP_KWARGS)
            _, xr_p, ok_p = _solve_subproblem_diffable(
                prob, x_in_p, w_t, xhat, ss_p;
                warmstart = false, madnlp_kwargs = MADNLP_KWARGS,
            )
            @test ok_p
            d_x_in_fd[j] = (xr_p[k] - x_real0[k]) / ε
        end

        # FD: ∂x_real[k]/∂xhat[j]
        d_xhat_fd = zeros(nx)
        for j in 1:nx
            xhat_p = copy(xhat); xhat_p[j] += ε
            ss_p = _make_subproblem_solver(prob.model, MADNLP_KWARGS)
            _, xr_p, ok_p = _solve_subproblem_diffable(
                prob, x_in, w_t, xhat_p, ss_p;
                warmstart = false, madnlp_kwargs = MADNLP_KWARGS,
            )
            @test ok_p
            d_xhat_fd[j] = (xr_p[k] - x_real0[k]) / ε
        end

        @test d_x_in_ad ≈ d_x_in_fd atol = 1e-3
        @test d_xhat_ad ≈ d_xhat_fd atol = 1e-3
        println("  State sens [k=$k] d_x_in: AD=$d_x_in_ad, FD=$d_x_in_fd")
        println("  State sens [k=$k] d_xhat: AD=$d_xhat_ad, FD=$d_xhat_fd")
    end
end

# ── Test 3: Full rollout gradient through Zygote vs FD ───────────────────────

@testset "Rollout gradient vs FD (3-stage, frozen MLP)" begin
    Random.seed!(42)
    nx = 1
    T = 3

    builder = () -> build_linear_tracking_subproblem(
        nx = nx,
        slack_penalty = 10.0,
        u_bounds = (-2.0, 2.0),
    )

    policy = Chain(Dense(2 * nx => 4, tanh), Dense(4 => nx))

    initial_state = Float32[1.0]
    w_flat = Float32[0.1, -0.2, 0.15]  # T * nx

    subproblems = SubproblemDEProblem[builder() for _ in 1:T]
    solver_states = SubproblemSolverState[
        _make_subproblem_solver(sp.model, MADNLP_KWARGS) for sp in subproblems
    ]

    # Zygote gradient
    gs = Zygote.gradient(policy) do m
        DecisionRulesExa._subproblem_rollout_loss(
            m, initial_state, subproblems, solver_states, w_flat, T;
            warmstart = false, madnlp_kwargs = MADNLP_KWARGS,
        )
    end

    grad_zy = materialize_tangent(gs[1])
    @test grad_zy !== nothing
    @test _all_finite_gradient(grad_zy)

    # FD: perturb each policy parameter and compare
    ε = 1e-4
    ps = Flux.params(policy)
    param_arrays = [copy(p) for p in ps]

    loss_fn = function(m)
        sps = SubproblemDEProblem[builder() for _ in 1:T]
        sss = SubproblemSolverState[
            _make_subproblem_solver(sp.model, MADNLP_KWARGS) for sp in sps
        ]
        return DecisionRulesExa._subproblem_rollout_loss(
            m, initial_state, sps, sss, w_flat, T;
            warmstart = false, madnlp_kwargs = MADNLP_KWARGS,
        )
    end

    loss0 = loss_fn(policy)
    @test isfinite(loss0)

    function flatten_params(m)
        vcat([vec(copy(p)) for p in Flux.params(m)]...)
    end
    function set_flat_params!(m, θ)
        offset = 0
        for p in Flux.params(m)
            n = length(p)
            p .= reshape(θ[offset+1:offset+n], size(p))
            offset += n
        end
    end
    function flatten_grad(g_nt)
        out = Float64[]
        for layer in g_nt.layers
            append!(out, vec(Float64.(layer.weight)))
            append!(out, vec(Float64.(layer.bias)))
        end
        return out
    end

    θ0 = flatten_params(policy)
    g_flat_ad = flatten_grad(grad_zy)

    g_flat_fd = zeros(length(θ0))
    for i in 1:length(θ0)
        θ_p = copy(θ0); θ_p[i] += ε
        set_flat_params!(policy, θ_p)
        l_p = loss_fn(policy)
        g_flat_fd[i] = (l_p - loss0) / ε
    end
    set_flat_params!(policy, θ0)  # restore

    # Compare (allow larger tolerance for multi-stage chaining)
    nonzero = findall(abs.(g_flat_fd) .> 1e-8)
    if !isempty(nonzero)
        rel_errs = abs.(g_flat_ad[nonzero] .- g_flat_fd[nonzero]) ./
                   max.(abs.(g_flat_fd[nonzero]), 1e-8)
        max_rel_err = maximum(rel_errs)
        println("  Rollout grad: max relative error = $(round(max_rel_err; digits=6))")
        println("  AD (first 5): $(g_flat_ad[1:min(5,end)])")
        println("  FD (first 5): $(g_flat_fd[1:min(5,end)])")
        @test max_rel_err < 0.05
    else
        # All FD gradients near zero — check AD is also near zero
        @test norm(g_flat_ad) < 1e-3
        println("  All gradients near zero — pass")
    end
end

# ── Test 4: Per-stage solver isolation ───────────────────────────────────────

@testset "Per-stage solver isolation (2-stage)" begin
    nx = 1
    T = 2

    builder = () -> build_linear_tracking_subproblem(
        nx = nx,
        slack_penalty = 10.0,
        u_bounds = (-2.0, 2.0),
    )

    subproblems = SubproblemDEProblem[builder() for _ in 1:T]
    solver_states = SubproblemSolverState[
        _make_subproblem_solver(sp.model, MADNLP_KWARGS) for sp in subproblems
    ]

    x_in = [1.0]
    w1 = [0.1]; w2 = [-0.2]
    xhat1 = [1.5]; xhat2 = [0.8]

    # Solve stage 1
    obj1, xr1, ok1 = _solve_subproblem_diffable(
        subproblems[1], x_in, w1, xhat1, solver_states[1];
        warmstart = false, madnlp_kwargs = MADNLP_KWARGS,
    )
    @test ok1
    sens1 = solver_states[1].maddiff_sens

    # Solve stage 2 (different solver_state)
    obj2, xr2, ok2 = _solve_subproblem_diffable(
        subproblems[2], Float64.(xr1), w2, xhat2, solver_states[2];
        warmstart = false, madnlp_kwargs = MADNLP_KWARGS,
    )
    @test ok2
    sens2 = solver_states[2].maddiff_sens

    # Verify stage 1's MadDiffSolver is still valid (not corrupted by stage 2)
    @test sens1 !== nothing
    @test sens2 !== nothing
    @test sens1 !== sens2
    @test solver_states[1].solver !== solver_states[2].solver

    # VJP on stage 1 should give results consistent with stage 1's parameters
    n_primal = length(solver_states[1].solver.x.x)
    dL_dx = zeros(n_primal)
    dL_dx[1] = 1.0  # ∂L/∂x_real[1]
    vjp1 = MadDiff.vector_jacobian_product!(sens1; dL_dx = dL_dx, dobj = 0.0)
    grad_p1 = Array(vjp1.grad_p)

    # FD check for stage 1 state sensitivity
    ε = 1e-5
    x_in_p = copy(x_in); x_in_p[1] += ε
    ss_fd = _make_subproblem_solver(subproblems[1].model, MADNLP_KWARGS)
    _, xr1_p, _ = _solve_subproblem_diffable(
        subproblems[1], x_in_p, w1, xhat1, ss_fd;
        warmstart = false, madnlp_kwargs = MADNLP_KWARGS,
    )
    fd_dx_in = (xr1_p[1] - xr1[1]) / ε

    @test grad_p1[1] ≈ fd_dx_in atol = 1e-3
    println("  Stage 1 isolation: VJP grad_p[x_in]=$(grad_p1[1]), FD=$fd_dx_in")
end

println("\nAll MadDiff subproblem tests passed!")
