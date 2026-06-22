using Test
using DecisionRulesExa
using Flux
using Random
using Zygote

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

function _state_vector(m)
    out = Float64[]
    function visit(x)
        if x isa AbstractArray && eltype(x) <: Number
            append!(out, vec(Float64.(x)))
        elseif x isa NamedTuple
            foreach(visit, values(x))
        elseif x isa Tuple
            foreach(visit, x)
        end
        return nothing
    end
    visit(Flux.state(m))
    return out
end

@testset "Critic control variate helpers" begin
    initial_state = Float32[1]
    uncertainty = Float32[0.2, -0.1]
    xhat = Float32[1.5, -2.0]

    quadratic = x -> sum(abs2, x) / 2
    cv = ScalarCriticControlVariate(
        quadratic;
        featurizer = (x0, w, x) -> x,
        value_loss_weight = 1.0,
        gradient_loss_weight = 1.0,
    )

    @test critic_value(cv, initial_state, uncertainty, xhat) ≈ sum(abs2, xhat) / 2
    @test critic_xhat_gradient(cv, initial_state, uncertainty, xhat) ≈ xhat

    sample = CriticSample(
        initial_state,
        uncertainty,
        xhat,
        sum(abs2, xhat) / 2,
        copy(xhat),
    )
    @test critic_loss(cv, [sample]) ≈ 0 atol = 1e-6

    bad_sample = CriticSample(initial_state, uncertainty, xhat, 0.0, Float32[1.0])
    @test_throws ErrorException critic_loss(cv, [bad_sample])

    value_only = ScalarCriticControlVariate(
        quadratic;
        featurizer = (x0, w, x) -> x,
        value_loss_weight = 1.0,
        gradient_loss_weight = 0.0,
    )
    grad_only = ScalarCriticControlVariate(
        quadratic;
        featurizer = (x0, w, x) -> x,
        value_loss_weight = 0.0,
        gradient_loss_weight = 1.0,
    )
    hybrid = ScalarCriticControlVariate(
        quadratic;
        featurizer = (x0, w, x) -> x,
        value_loss_weight = 0.1,
        gradient_loss_weight = 1.0,
    )
    @test isfinite(critic_loss(value_only, [sample]))
    @test isfinite(critic_loss(grad_only, [sample]))
    @test isfinite(critic_loss(hybrid, [sample]))
end

@testset "Critic and actor update separation" begin
    Random.seed!(11)
    critic = Chain(Dense(2 => 1, bias = false))
    critic[1].weight .= 0.0f0
    cv = ScalarCriticControlVariate(
        critic;
        featurizer = (x0, w, x) -> x,
        value_loss_weight = 1.0,
        gradient_loss_weight = 0.0,
    )
    sample = CriticSample(Float32[0], Float32[0], Float32[1, -1], 2.0, Float32[0, 0])

    critic_before = _state_vector(critic)
    actor = Chain(Dense(1 => 2, bias = false))
    actor_before = _state_vector(actor)
    opt_state = Flux.setup(Flux.Descent(0.1f0), critic)
    loss = update_critic!(opt_state, cv, [sample])
    @test isfinite(loss)
    @test _state_vector(critic) != critic_before
    @test _state_vector(actor) == actor_before

    critic_before_actor_update = _state_vector(critic)
    actor_opt = Flux.setup(Flux.Descent(0.1f0), actor)
    gs = Zygote.gradient(actor) do m
        x = m(Float32[1])
        critic_value(cv, Float32[0], Float32[0], x)
    end
    Flux.update!(actor_opt, actor, materialize_tangent(gs[1]))
    @test _state_vector(actor) != actor_before
    @test _state_vector(critic) == critic_before_actor_update

    for (value_w, grad_w) in ((0.0, 1.0), (0.1, 1.0))
        c = Chain(Dense(2 => 1, bias = false))
        c[1].weight .= 0.0f0
        cv_step = ScalarCriticControlVariate(
            c;
            featurizer = (x0, w, x) -> x,
            value_loss_weight = value_w,
            gradient_loss_weight = grad_w,
        )
        s = CriticSample(Float32[0], Float32[0], Float32[1, -1], 2.0, Float32[1, -1])
        before = _state_vector(c)
        st = Flux.setup(Flux.Descent(0.1f0), c)
        @test isfinite(update_critic!(st, cv_step, [s]))
        @test _state_vector(c) != before
    end
end

@testset "Control-variate actor gradients" begin
    Random.seed!(12)
    actor_dual = Chain(Dense(1 => 2, bias = false))
    Random.seed!(12)
    actor_cv = Chain(Dense(1 => 2, bias = false))

    x_in = Float32[2]
    lambda = Float32[1, -3]
    zero_cv = ScalarCriticControlVariate(
        x -> zero(sum(x));
        featurizer = (x0, w, x) -> x,
    )

    g_dual = Zygote.gradient(actor_dual) do m
        sum(lambda .* m(x_in))
    end[1]
    g_cv = Zygote.gradient(actor_cv) do m
        xhat = m(x_in)
        gx = critic_xhat_gradient(zero_cv, Float32[0], Float32[0], xhat)
        sum((lambda .- gx) .* xhat) + critic_value(zero_cv, Float32[0], Float32[0], xhat)
    end[1]

    @test materialize_tangent(g_cv).layers[1].weight ≈
          materialize_tangent(g_dual).layers[1].weight
end
