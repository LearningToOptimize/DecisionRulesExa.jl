using Test
using DecisionRulesExa
using ExaModels
using Flux
using MadNLP
using Random
using Zygote

include(joinpath(@__DIR__, "..", "examples", "HydroPowerModels", "hydro_training_utils.jl"))
# Hydro example files (data structs, ExaModels builders, reachable policy) used
# by the reactive-deficit and recurrent-threading testsets below.
include(joinpath(@__DIR__, "..", "examples", "HydroPowerModels", "hydro_power_data.jl"))
include(joinpath(@__DIR__, "..", "examples", "HydroPowerModels", "hydro_power_exa.jl"))
include(joinpath(@__DIR__, "..", "examples", "HydroPowerModels", "hydro_reachable_policy.jl"))

@testset "HydroPowerModels training utilities" begin
    @test parse_layers("128, 64") == [128, 64]
    @test parse_layers("") == Int[]
    @test parse_layers("  ") == Int[]
    @test parse_layers("32,,16") == [32, 16]
end

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

    res = DecisionRulesExa.solve!(prob; tol = 1e-6, max_iter = 200)

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

@testset "Bounded state policy helper" begin
    Random.seed!(21)
    lower = Float32[0, 1, -2]
    upper = Float32[10, 1, 2]
    policy = bounded_state_policy(2, lower, upper, [4]; activation = sigmoid)

    y = policy(Float32[0.2, -0.1, 3, 1, -1])
    @test length(y) == 3
    @test lower[1] <= y[1] <= upper[1]
    @test y[2] == lower[2]
    @test lower[3] <= y[3] <= upper[3]
    @test length(policy.policy.combiner.bias) == 2

    deep_policy = bounded_state_policy(
        2,
        lower,
        upper,
        [4];
        activation = sigmoid,
        combiner_layers = [5, 4],
    )
    @test deep_policy.policy.combiner isa Flux.Chain
    y_deep = deep_policy(Float32[0.2, -0.1, 3, 1, -1])
    @test length(y_deep) == 3
    @test lower[1] <= y_deep[1] <= upper[1]
    @test y_deep[2] == lower[2]
    @test lower[3] <= y_deep[3] <= upper[3]

    constant_policy = bounded_state_policy(1, Float32[3, -4], Float32[3, -4], [4])
    @test constant_policy(Float32[0, 99, 100]) == Float32[3, -4]
    @test isempty(Flux.trainables(constant_policy))
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

@testset "EmbeddedDeterministicEquivalentProblem (CPU)" begin
    T = 5
    nx = 1
    nw = 1

    Random.seed!(42)
    policy = StateConditionedPolicy(nw, nx, nx, [8]; activation = tanh)

    prob = build_embedded_deterministic_equivalent(
        policy;
        horizon = T,
        nx = nx,
        nu = nx,
        nw = nw,
        backend = nothing,
        float_type = Float64,
        slack_penalty = 10.0,
        u_bounds = (-2.0, 2.0),
    )

    @test prob isa EmbeddedDeterministicEquivalentProblem
    @test prob.horizon == T
    @test prob.nx == nx

    x0 = [1.0]
    w  = randn(T * nw)

    set_x0!(prob, x0)
    set_uncertainty!(prob, w)

    n_x = T * nx
    n_u = (T - 1) * nx
    n_var = n_x + n_u + n_x
    n_con_initial = nx
    n_con_dynamics = (T - 1) * nx
    n_con_oracle = n_x
    n_con = n_con_initial + n_con_dynamics + n_con_oracle

    res = MadNLP.madnlp(prob.model; tol = 1e-6, max_iter = 500, print_level = MadNLP.ERROR)
    @test DecisionRulesExa.solve_succeeded(res)
    @test length(res.solution) == n_var
    @test length(res.multipliers) == n_con

    λ = target_multipliers(prob, res)
    @test length(λ) == n_x
    @test all(isfinite, λ)

    x_sol, u_sol, δ_sol = solution_components(prob, res)
    @test length(x_sol) == n_x
    @test length(u_sol) == n_u
    @test length(δ_sol) == n_x

    # Verify oracle constraint satisfaction: x_t + δ_t ≈ π_θ(w_t, x_{t-1})
    Flux.reset!(policy)
    for t in 1:T
        x_prev = (t == 1) ? Float32.(x0) : Float32.([x_sol[(t-2)*nx+1:(t-1)*nx]...])
        w_t = Float32.([w[(t-1)*nw+1:t*nw]...])
        nn_out = policy(vcat(w_t, x_prev))
        for i in 1:nx
            xi = x_sol[(t-1)*nx+i]
            di = δ_sol[(t-1)*nx+i]
            @test xi + di ≈ Float64(nn_out[i]) atol = 1e-4
        end
    end
end

@testset "Embedded NN gradient (envelope theorem)" begin
    T = 4
    nx = 1
    nw = 1

    Random.seed!(7)
    policy = StateConditionedPolicy(nw, nx, nx, [8]; activation = tanh)

    prob = build_embedded_deterministic_equivalent(
        policy;
        horizon = T,
        nx = nx,
        nu = nx,
        nw = nw,
        backend = nothing,
        float_type = Float64,
        slack_penalty = 10.0,
        u_bounds = (-2.0, 2.0),
    )

    x0 = [0.5]
    w  = randn(T * nw)

    set_x0!(prob, x0)
    set_uncertainty!(prob, w)

    res = MadNLP.madnlp(prob.model; tol = 1e-6, max_iter = 500, print_level = MadNLP.ERROR)
    @test DecisionRulesExa.solve_succeeded(res)

    λ = target_multipliers(prob, res)
    x_sol = res.solution[1 : T * nx]

    # Zygote gradient: ∇_θ Σ_t ⟨λ_t, π_θ(w_t, x*_{t-1})⟩
    gs = Zygote.gradient(policy) do m
        total = 0.0f0
        Flux.reset!(m)
        for t in 1:T
            wt = Float32.([w[(t-1)*nw+1:t*nw]...])
            x_prev = (t == 1) ?
                Float32.(x0) :
                Float32.([x_sol[(t-2)*nx+1:(t-1)*nx]...])
            xt = m(vcat(wt, x_prev))
            for i in 1:nx
                total = total + Float32(λ[(t-1)*nx+i]) * xt[i]
            end
        end
        total
    end

    g = materialize_tangent(gs[1])
    @test g !== nothing
    @test DecisionRulesExa._all_finite_gradient(g)
end

@testset "train_tsddr_embedded smoke test" begin
    T = 4
    nx = 1
    nw = 1

    Random.seed!(99)
    policy = StateConditionedPolicy(nw, nx, nx, [8]; activation = tanh)

    prob = build_embedded_deterministic_equivalent(
        policy;
        horizon = T,
        nx = nx,
        backend = nothing,
        slack_penalty = 10.0,
        u_bounds = (-2.0, 2.0),
    )

    x0 = Float32[1.0]
    losses = Float64[]

    train_tsddr_embedded(
        policy, x0, prob,
        () -> randn(T * nw);
        num_batches = 5,
        num_train_per_batch = 2,
        madnlp_kwargs = (tol = 1e-6, max_iter = 300, print_level = MadNLP.ERROR),
        warmstart = true,
        record_loss = (iter, m, loss, tag) -> begin
            push!(losses, loss)
            return false
        end,
    )

    @test length(losses) == 5
    @test all(isfinite, losses)
end

@testset "rollout_tsddr (CPU)" begin
    horizon = 3
    nx = 1

    # Stage problem: a horizon-2 linear tracking deterministic equivalent used
    # as a one-stage projection problem. The realized next state is x_2.
    stage_problem = build_linear_tracking_problem(
        horizon = 2,
        nx = nx,
        backend = nothing,
        slack_penalty = 10.0,
        u_bounds = (-2.0, 2.0),
    )

    # Callback: write (x_{t-1}, w_t, xhat_t) into the stage problem. The stage-1
    # target equals the incoming state (its constraint is slack-absorbed anyway);
    # the stage-2 target is the policy target being projected.
    set_stage_params! = (prob, state, w_t, target, stage) -> begin
        set_x0!(prob, state)
        set_uncertainty!(prob, w_t)
        set_targets!(prob, vcat(state, target))
        return nothing
    end
    # Callback: read the realized next state x_2 from the stage solution.
    realized = (prob, result) -> begin
        x_sol, _, _ = solution_components(prob, result)
        return x_sol[end - prob.nx + 1 : end]
    end

    Random.seed!(31)
    policy = StateConditionedPolicy(1, 1, 1, [4]; activation = tanh)
    x0 = Float32[0.5]
    w_flat = Float32.(0.1 .* randn(horizon * 1))

    result = rollout_tsddr(
        policy, x0, stage_problem, w_flat;
        horizon = horizon,
        n_uncertainty = 1,
        set_stage_parameters! = set_stage_params!,
        realized_state = realized,
        madnlp_kwargs = (tol = 1e-6, max_iter = 300, print_level = MadNLP.ERROR),
    )
    @test result isa NamedTuple
    @test isfinite(result.objective)
    @test length(result.state_trajectory) == horizon + 1
    @test length(result.target_trajectory) == horizon

    # The :target feedback mode (policy sees its own previous target) must also run.
    result_target = rollout_tsddr(
        policy, x0, stage_problem, w_flat;
        horizon = horizon,
        n_uncertainty = 1,
        set_stage_parameters! = set_stage_params!,
        realized_state = realized,
        policy_state = :target,
        madnlp_kwargs = (tol = 1e-6, max_iter = 300, print_level = MadNLP.ERROR),
    )
    @test result_target isa NamedTuple
    @test isfinite(result_target.objective)

    # A wrong-length uncertainty vector must be rejected before any solve.
    @test_throws ArgumentError rollout_tsddr(
        policy, x0, stage_problem, Float32[0.1];
        horizon = horizon,
        n_uncertainty = 1,
        set_stage_parameters! = set_stage_params!,
        realized_state = realized,
    )
end

@testset "train_tsddr open-loop smoke test" begin
    T = 4
    nx = 1

    # train_tsddr writes the full length-T*nw uncertainty sample into
    # p_uncertainty via ExaModels.set_parameter!, which enforces an exact size
    # match. build_linear_tracking_problem's p_w has length (T-1)*nw (dynamics
    # stages only), so this test builds the same linear tracking NLP manually
    # with a full-length uncertainty parameter (mirroring the hydro p_inflow
    # layout, where the final-stage entry does not enter the dynamics).
    core = ExaModels.ExaCore(Float64)
    x = ExaModels.variable(core, T * nx)
    u = ExaModels.variable(core, (T - 1) * nx; lvar = -2.0, uvar = 2.0)
    δ = ExaModels.variable(core, T * nx)
    p_x0 = ExaModels.parameter(core, zeros(nx))
    p_w = ExaModels.parameter(core, zeros(T * nx))       # full length T*nw
    p_target = ExaModels.parameter(core, zeros(T * nx))
    # Stage cost (x_t^2 + u_t^2)/2 plus slack penalty (rho/2)*delta^2, rho = 10.
    ExaModels.objective(core,
        (x[x_index(nx, t, i)]^2 + u[u_index(nx, t, i)]^2) / 2
        for t in 1:(T - 1), i in 1:nx
    )
    ExaModels.objective(core,
        5.0 * δ[x_index(nx, t, i)]^2
        for t in 1:T, i in 1:nx
    )
    # Initial condition, dynamics x_{t+1} = x_t + u_t + w_t, then targets LAST.
    ExaModels.constraint(core,
        x[x_index(nx, 1, i)] - p_x0[i]
        for i in 1:nx
    )
    ExaModels.constraint(core,
        x[x_index(nx, t + 1, i)] - x[x_index(nx, t, i)] -
        u[u_index(nx, t, i)] - p_w[w_index(nx, t, i)]
        for t in 1:(T - 1), i in 1:nx
    )
    ExaModels.constraint(core,
        p_target[x_index(nx, t, i)] - x[x_index(nx, t, i)] - δ[x_index(nx, t, i)]
        for t in 1:T, i in 1:nx
    )
    model = ExaModels.ExaModel(core)
    target_start = nx + (T - 1) * nx + 1
    prob = DeterministicEquivalentProblem(
        core, model, x, u, δ,
        p_x0, p_w, p_target,
        nx, nx, nx, T,
        target_start:(target_start + T * nx - 1),
    )

    Random.seed!(123)
    policy = StateConditionedPolicy(nx, nx, nx, [8]; activation = tanh)
    x0 = Float32[1.0]
    losses = Float64[]
    params_before = _state_vector(policy)

    train_tsddr(
        policy, x0, prob,
        prob.p_x0, prob.p_target, prob.p_w,
        () -> randn(T * nx);
        num_batches = 2,
        num_train_per_batch = 1,
        madnlp_kwargs = (tol = 1e-6, max_iter = 300, print_level = MadNLP.ERROR),
        warmstart = true,
        record_loss = (iter, m, loss, tag) -> begin
            push!(losses, loss)
            return false
        end,
    )

    @test length(losses) == 2
    @test all(isfinite, losses)
    @test _state_vector(policy) != params_before
end

@testset "StateConditionedPolicy recurrent-state threading" begin
    Random.seed!(42)
    policy = StateConditionedPolicy(2, 1, 1, [4, 3]; activation = tanh)
    x = Float32[0.3, -0.2, 0.5]

    # Memory: the same input twice WITHOUT reset must give different outputs
    # (the recurrent state advanced between calls). A memoryless (per-call
    # restart) encoder would reproduce the first output exactly.
    Flux.reset!(policy)
    y1 = policy(x)
    y2 = policy(x)
    @test y1 != y2

    # Reset restores the exact initial recurrent state: identical output.
    Flux.reset!(policy)
    @test policy(x) == y1

    # Manual LSTMCell recursion with the same weights reproduces the policy's
    # stage outputs EXACTLY over a 3-stage open-loop sequence.
    ws = [Float32[0.3, -0.2], Float32[0.1, 0.4], Float32[-0.5, 0.2]]
    Flux.reset!(policy)
    prev = Float32[0.5]
    outs = Vector{Vector{Float32}}()
    for t in 1:3
        y = policy(vcat(ws[t], prev))
        push!(outs, Float32.(y))
        prev = Float32.(y)
    end

    cells = [l.cell for l in policy.encoder.layers]
    states = Any[Flux.initialstates(c) for c in cells]
    prev = Float32[0.5]
    for t in 1:3
        h = ws[t]
        for (i, c) in enumerate(cells)
            h, states[i] = c(h, states[i])
        end
        y = policy.combiner(vcat(h, prev))
        @test Float32.(y) == outs[t]
        prev = Float32.(y)
    end
end

@testset "Zygote gradient through threaded 3-stage rollout" begin
    Random.seed!(43)
    policy = StateConditionedPolicy(2, 1, 1, [4]; activation = tanh)
    ws = [Float32[0.3, -0.2], Float32[0.1, 0.4], Float32[-0.5, 0.2]]
    x0 = Float32[0.5]

    # Same pattern as the training loops: reset inside the gradient block,
    # thread the recurrent state through all stages via the policy forward.
    gs = Zygote.gradient(policy) do m
        Flux.reset!(m)
        total = 0.0f0
        prev = x0
        for t in 1:3
            y = m(vcat(ws[t], prev))
            total = total + sum(y)
            prev = y
        end
        total
    end
    g = materialize_tangent(gs[1])
    @test g !== nothing
    @test DecisionRulesExa._all_finite_gradient(g)

    # Encoder gradients must be finite AND nonzero (the LSTM parameters
    # participate in every stage of the threaded rollout).
    enc_leaves = Float64[]
    function _collect_leaves(x)
        if x isa AbstractArray && eltype(x) <: Number
            append!(enc_leaves, vec(Float64.(x)))
        elseif x isa NamedTuple
            foreach(_collect_leaves, values(x))
        elseif x isa Tuple
            foreach(_collect_leaves, x)
        end
        return nothing
    end
    _collect_leaves(g.encoder)
    @test !isempty(enc_leaves)
    @test any(!=(0.0), enc_leaves)
end

# ── Synthetic 2-bus fixture for the hydro AC builder ──────────────────────────
# One thermal generator at the reference bus, one hydro generator at the load
# bus, a single branch, and one reservoir. Small enough for fast MadNLP solves.
function _tiny_ac_case()
    buses = [
        PowerBusData(1, 3, 0.0, 0.0, 0.9, 1.1),
        PowerBusData(2, 1, 0.0, 0.0, 0.9, 1.1),
    ]
    gens = [
        PowerGenData(1, 1, 0.0, 2.0, -1.0, 1.0, 10.0, 0.0),   # thermal at bus 1
        PowerGenData(2, 2, 0.0, 1.0, -1.0, 1.0, 0.0, 0.0),    # hydro at bus 2
    ]
    b_dc = -0.1 / (0.01^2 + 0.1^2)
    branches = [
        PowerBranchData(1, 1, 2, b_dc, 2.0, -0.5, 0.5,
                        0.01, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    ]
    loads = [PowerLoadData(1, 2)]
    power_data = PowerData(
        2, 2, 1, 1, buses, gens, branches, loads,
        [1], 100.0, 60.0, [0.0, 0.5], [0.0, 0.1],
    )
    units = [HydroUnitData(1, 2, 100.0, 0.0, 10.0, 0.0, 10.0, 0.001, 0.0, 0.0)]
    hydro_data = HydroData(
        1, units, UpstreamTurn[], UpstreamSpill[],
        2.6, [50.0], [ones(2, 1)], 1, 2,
    )
    return power_data, hydro_data
end

@testset "AC reactive-deficit modes (nothing / finite / Inf)" begin
    power_data, hydro_data = _tiny_ac_case()
    T = 2
    nBus = power_data.nBus

    _build(cost; strict = false) = build_hydro_de(power_data, hydro_data, T;
        formulation = :ac_polar,
        deficit_cost = 1e4,
        strict_targets = strict,
        reactive_deficit_cost = cost,
    )

    prob_free = _build(nothing)
    prob_pen  = _build(10.0)
    prob_hard = _build(Inf)

    @test prob_free.reactive_deficit_mode === :free
    @test prob_pen.reactive_deficit_mode  === :penalized
    @test prob_hard.reactive_deficit_mode === :hard

    # Variable counts: penalized splits the slack (+T*nBus vs free); hard
    # removes it (−T*nBus vs free). Constraint counts are identical.
    @test prob_pen.model.meta.nvar  == prob_free.model.meta.nvar + T * nBus
    @test prob_hard.model.meta.nvar == prob_free.model.meta.nvar - T * nBus
    @test prob_pen.model.meta.ncon  == prob_free.model.meta.ncon
    @test prob_hard.model.meta.ncon == prob_free.model.meta.ncon

    # target_con_range bookkeeping is unaffected in both non-strict and strict.
    @test prob_pen.target_con_range  == prob_free.target_con_range
    @test prob_hard.target_con_range == prob_free.target_con_range
    strict_free = _build(nothing; strict = true)
    strict_pen  = _build(10.0;    strict = true)
    strict_hard = _build(Inf;     strict = true)
    @test strict_pen.target_con_range  == strict_free.target_con_range
    @test strict_hard.target_con_range == strict_free.target_con_range

    # DC path rejects the kwarg (and is otherwise unaffected by it).
    @test_throws ErrorException build_hydro_de(power_data, hydro_data, T;
        formulation = :dc, reactive_deficit_cost = 10.0)

    # All three AC modes must solve.
    x0 = [50.0]
    w  = [2.0, 2.0]
    xhat = [50.0, 50.0]
    for prob in (prob_free, prob_pen, prob_hard)
        ExaModels.set_parameter!(prob.core, prob.p_x0, x0)
        ExaModels.set_parameter!(prob.core, prob.p_inflow, w)
        ExaModels.set_parameter!(prob.core, prob.p_target, xhat)
        prepare_solve!(prob, x0, w, xhat)
        res = MadNLP.madnlp(prob.model; tol = 1e-6, max_iter = 500,
                            print_level = MadNLP.ERROR)
        @test solve_succeeded(res)
        @test isfinite(res.objective)
        sol = hydro_solution(prob, res)
        @test size(sol.deficit_q) == (nBus, T)
        @test all(isfinite, sol.deficit_q)
        if prob.reactive_deficit_mode === :hard
            @test all(iszero, sol.deficit_q)
        end
    end
end

@testset "HydroReachablePolicy threading matches manual cell recursion" begin
    power_data, hydro_data = _tiny_ac_case()
    Random.seed!(44)
    policy = hydro_reachable_policy(hydro_data, [4, 3])

    # Memory across stages: identical inputs without reset differ; reset
    # restores the exact first output.
    xin = vcat(Float32[2.0], Float32[50.0])
    Flux.reset!(policy)
    a = policy(xin)
    b = policy(xin)
    @test a != b
    Flux.reset!(policy)
    @test policy(xin) == a

    # As-is open loop vs manual LSTMCell threading (the verified diagnostic
    # pattern from eval_paired_exa_strict.jl) — must agree EXACTLY.
    T = 3
    w_stages = Float32[2.0, 1.5, 2.5]
    Flux.reset!(policy)
    prev = Float32[50.0]
    asis = Vector{Vector{Float32}}()
    for t in 1:T
        y = policy(vcat(Float32[w_stages[t]], prev))
        push!(asis, Float32.(y))
        prev = Float32.(y)
    end

    cells = [l.cell for l in policy.encoder.layers]
    states = Any[Flux.initialstates(c) for c in cells]
    prev = Float32[50.0]
    for t in 1:T
        wt = Float32[w_stages[t]]
        h = wt
        for (i, c) in enumerate(cells)
            h, states[i] = c(h, states[i])
        end
        y = policy.combiner(vcat(h, prev))
        lower, upper = _hydro_reachable_bounds(policy, wt, prev, y)
        raw = lower .+ (upper .- lower) .* y
        target = isempty(policy.cascade) ? raw :
                 min.(raw, _cascade_upper_bounds(policy, raw, wt, prev))
        @test Float32.(target) == asis[t]
        prev = Float32.(target)
    end
end
