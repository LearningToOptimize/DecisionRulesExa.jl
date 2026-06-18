using Test
using DecisionRulesExa

@testset "DeterministicEquivalentProblem (CPU)" begin
    T = 6
    nx = 1

    prob = build_linear_tracking_problem(
        horizon = T,
        nx = nx,
        backend = nothing,
        slack_penalty = 10.0,
        u_bounds = (-2.0, 2.0),
    )

    x0 = [1.0]
    w  = zeros(T - 1)           # (T-1)*nx
    xhat = zeros(T * nx)

    set_x0!(prob, x0)
    set_uncertainty!(prob, w)
    set_targets!(prob, xhat)

    res = solve!(prob; tol = 1e-6, max_iter = 200)

    n_x = T * nx
    n_u = (T - 1) * nx
    n_var = n_x + n_u + n_x
    n_con = nx + (T - 1) * nx + n_x

    @test length(res.solution) == n_var
    @test length(res.multipliers) == n_con

    λ = target_multipliers(prob, res)
    @test length(λ) == n_x

    x_sol, u_sol, δ_sol = solution_components(prob, res)
    @test length(x_sol) == n_x
    @test length(u_sol) == n_u
    @test length(δ_sol) == n_x
end
