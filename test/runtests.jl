using Test
using DecisionRulesExa
using Statistics
using ExaModels
using Flux
using MadNLP
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
    # with a full-length uncertainty parameter, where the final-stage entry
    # does not enter the dynamics.
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

# ─────────────────────────────────────────────────────────────────────────────
# RolloutEvaluation: scenario identity, failure visibility, retry, recorder.
#
# The pooled evaluator was observed to lose an individual scenario that the
# sequential path solved. Losing one silently replaces an N-scenario mean with an
# (N-1)-scenario mean — a different statistic, typically a flattering one, since
# the scenarios that fail tend to be the hard, expensive ones. These tests pin
# the properties that make that impossible to miss: identity and order survive
# parallel execution, failures are named, an optional retry recovers them, and
# an incomplete evaluation can never pass for a complete one.
# ─────────────────────────────────────────────────────────────────────────────

"""
    _rollout_eval_fixture(nx) -> (stage_problem_builder, set_stage_params!, realized)

Build the pieces every `RolloutEvaluation` test below shares: a one-stage
projection problem (a horizon-2 linear tracking DE) plus the parameter-writing
and realized-state callbacks, matching the `rollout_tsddr` testset above.
"""
function _rollout_eval_fixture(nx)
    builder = () -> build_linear_tracking_problem(
        horizon = 2, nx = nx, backend = nothing,
        slack_penalty = 10.0, u_bounds = (-2.0, 2.0),
    )
    set_stage_params! = (prob, state, w_t, target, stage) -> begin
        set_x0!(prob, state)
        set_uncertainty!(prob, w_t)
        set_targets!(prob, vcat(state, target))
        return nothing
    end
    realized = (prob, result) -> begin
        x_sol, _, _ = solution_components(prob, result)
        return x_sol[end - prob.nx + 1 : end]
    end
    return builder, set_stage_params!, realized
end

"""
    FailingPolicy(inner, failing_scenarios)

Wrapper that reproduces a per-scenario solver failure deterministically.

The rollout feeds `[w_t; x_{t-1}]` to the policy, and each test scenario carries
a distinct constant `w`, so the wrapper can recognize a scenario from its input
alone. For a scenario in `failing_scenarios` it emits `NaN`, which propagates
into the stage target and makes the solve fail exactly the way a genuinely
unsolvable scenario does — no solver options are weakened and no other scenario
is affected.

`attempts` counts forward passes, so a test can distinguish "never retried" from
"retried and still failed".
"""
mutable struct FailingPolicy{P}
    inner::P
    failing_scenarios::Vector{Float32}   # sentinel w values that must fail
    attempts::Base.RefValue{Int}
end
FailingPolicy(inner, failing) = FailingPolicy(inner, Float32.(failing), Ref(0))
Flux.reset!(p::FailingPolicy) = Flux.reset!(p.inner)
function (p::FailingPolicy)(x)
    out = p.inner(x)
    p.attempts[] += 1
    any(s -> isapprox(Float32(x[1]), s; atol = 1f-6), p.failing_scenarios) || return out
    return out .* eltype(out)(NaN)
end

"""
    OneShotFailure(inner)

Policy wrapper that fails scenario 2 exactly ONCE and succeeds on every later
call — the transient, non-reproducible failure the pooled evaluator exhibits.
A sequential retry of that scenario therefore succeeds, which is what
`retry_failed_sequentially` exists to exploit.
"""
mutable struct OneShotFailure{P}
    inner::P
    fired::Base.RefValue{Bool}
end
OneShotFailure(inner) = OneShotFailure(inner, Ref(false))
Flux.reset!(p::OneShotFailure) = Flux.reset!(p.inner)
function (p::OneShotFailure)(x)
    out = p.inner(x)
    if !p.fired[] && isapprox(Float32(x[1]), 0.2f0; atol = 1f-6)
        p.fired[] = true
        return out .* eltype(out)(NaN)
    end
    return out
end

@testset "RolloutEvaluation — identity, completeness, retry" begin
    nx = 1
    horizon = 3
    builder, set_stage_params!, realized = _rollout_eval_fixture(nx)
    madnlp_kwargs = (tol = 1e-6, max_iter = 300, print_level = MadNLP.ERROR)

    # One constant-`w` scenario per index, so a scenario is identifiable from
    # its own uncertainty and results can be matched to the scenario that
    # produced them.
    n_scen = 4
    scenarios = [Float32.(fill(0.1f0 * s, horizon * nx)) for s in 1:n_scen]
    x0 = Float32[0.5]

    Random.seed!(97)
    base_policy = StateConditionedPolicy(1, 1, 1, [4]; activation = tanh)

    make_eval(; pool_size, retry = false, recorder = nothing) = RolloutEvaluation(
        builder(), x0, scenarios;
        horizon = horizon,
        n_uncertainty = nx,
        set_stage_parameters! = set_stage_params!,
        realized_state = realized,
        madnlp_kwargs = madnlp_kwargs,
        warmstart = false,
        stride = 1,
        policy_state = :realized,
        stage_problem_pool = [builder() for _ in 1:pool_size],
        retry_failed_sequentially = retry,
        stage_recorder = recorder,
    )

    # ── Defaults are unchanged ───────────────────────────────────────────────
    @testset "default construction preserves existing behaviour" begin
        evaluation = RolloutEvaluation(
            builder(), x0, scenarios;
            horizon = horizon, n_uncertainty = nx,
            set_stage_parameters! = set_stage_params!,
            realized_state = realized,
            madnlp_kwargs = madnlp_kwargs, warmstart = false,
        )
        @test evaluation.retry_failed_sequentially == false   # opt-in only
        @test evaluation.stage_recorder === nothing           # inert by default
        @test evaluation.last_n_requested == 0
        @test is_complete(evaluation) == false                # nothing evaluated yet
        # A non-aligned iteration must not evaluate, and must leave no stale data.
        evaluation.stride = 5
        evaluation(3, base_policy)
        @test isempty(evaluation.last_scenario_data)
        @test isempty(evaluation.last_scenario_status)
    end

    # ── Sequential / parallel parity and deterministic ordering ─────────────
    @testset "sequential and pooled evaluation agree, in scenario order" begin
        sequential = make_eval(pool_size = 0)
        pooled = make_eval(pool_size = 3)     # pool smaller than n_scen: two rounds

        sequential(1, deepcopy(base_policy))
        pooled(1, deepcopy(base_policy))

        @test sequential.last_n_requested == n_scen
        @test pooled.last_n_requested == n_scen
        @test is_complete(sequential)
        @test is_complete(pooled)

        # Ordering: entries ascend in scenario index, cover 1:n_scen exactly.
        indices(e) = [i for (i, _) in e.last_scenario_data]
        @test indices(sequential) == collect(1:n_scen)
        @test indices(pooled) == collect(1:n_scen)
        @test sequential.last_scenario_status == fill(:ok, n_scen)
        @test pooled.last_scenario_status == fill(:ok, n_scen)
        @test isempty(sequential.last_failed_scenarios)
        @test isempty(pooled.last_failed_scenarios)

        # Identity: scenario i's result is the SAME rollout in both paths.
        for i in 1:n_scen
            seq_result = last(sequential.last_scenario_data[i])
            par_result = last(pooled.last_scenario_data[i])
            @test seq_result.objective ≈ par_result.objective rtol = 1e-6
            @test seq_result.objective_no_target_penalty ≈
                  par_result.objective_no_target_penalty rtol = 1e-6
        end
        @test sequential.last_objective ≈ pooled.last_objective rtol = 1e-6
        @test sequential.last_objective_no_target_penalty ≈
              pooled.last_objective_no_target_penalty rtol = 1e-6
    end

    # ── Failures are named, not merely counted ──────────────────────────────
    @testset "a failed scenario is reported with its identity" begin
        failing = FailingPolicy(deepcopy(base_policy), [0.2f0])   # scenario 2
        for pool_size in (0, 3)
            evaluation = make_eval(pool_size = pool_size)
            evaluation(1, deepcopy(failing))

            @test evaluation.last_n_requested == n_scen
            @test evaluation.last_n_ok == n_scen - 1
            @test evaluation.last_failed_scenarios == [2]
            @test evaluation.last_scenario_status ==
                  [:ok, :failed, :ok, :ok]
            # The surviving results keep their ORIGINAL indices: nothing is
            # renumbered to close the gap.
            @test [i for (i, _) in evaluation.last_scenario_data] == [1, 3, 4]
            # And the evaluation is not complete, so it cannot select anything.
            @test is_complete(evaluation) == false
            # The mean is still computed (it is diagnostically useful) but it is
            # a mean over 3, which is exactly why it must not be selected on.
            @test isfinite(evaluation.last_objective)
            @test evaluation.last_objective ≈
                  mean(last(d).objective for d in evaluation.last_scenario_data)
        end
    end

    # ── Retry recovers what the pooled path loses ───────────────────────────
    @testset "optional retry runs only on failures" begin
        # A policy that fails on its FIRST pass over scenario 2 and succeeds
        # afterwards models the transient pooled-evaluator failure.
        transient = FailingPolicy(deepcopy(base_policy), [0.2f0])
        evaluation = make_eval(pool_size = 3, retry = true)
        evaluation(1, transient)
        # The sentinel is not transient here, so the retry must fail too — and
        # the scenario must still be reported rather than dropped.
        @test evaluation.last_failed_scenarios == [2]
        @test evaluation.last_scenario_status[2] == :failed
        @test is_complete(evaluation) == false

        # With no failures, retry changes nothing at all.
        clean = make_eval(pool_size = 3, retry = true)
        clean(1, deepcopy(base_policy))
        @test clean.last_scenario_status == fill(:ok, n_scen)
        @test isempty(clean.last_failed_scenarios)
        @test is_complete(clean)
    end

    @testset "retry marks a recovered scenario as :ok_on_retry" begin
        # `OneShotFailure` fails on the FIRST pass over scenario 2 and succeeds
        # afterwards: exactly the transient behaviour observed in production.
        evaluation = make_eval(pool_size = 0, retry = true)
        evaluation(1, OneShotFailure(deepcopy(base_policy)))
        @test evaluation.last_scenario_status[2] == :ok_on_retry
        @test isempty(evaluation.last_failed_scenarios)
        @test evaluation.last_n_ok == n_scen
        @test is_complete(evaluation)
        # Recovered results keep their index and stay in order.
        @test [i for (i, _) in evaluation.last_scenario_data] == collect(1:n_scen)
    end

    # ── Stage recorder ───────────────────────────────────────────────────────
    @testset "stage recorder: inert when absent, thread-safe when pooled" begin
        # Absent: no behaviour change at all.
        without = make_eval(pool_size = 0)
        without(1, deepcopy(base_policy))
        @test without.stage_recorder === nothing

        # Present: called exactly once per ACCEPTED stage solve, and its
        # aggregation is correct under concurrency. Mirrors how the hydro
        # trainer accumulates per-bus load shedding across a pooled evaluation.
        for pool_size in (0, 3)
            lock_ = ReentrantLock()
            calls = Ref(0)
            total = Ref(0.0)
            peak = Ref(-Inf)
            recorder = (prob, result, stage, target, state_in) -> begin
                value = result.objective
                lock(lock_) do
                    calls[] += 1
                    total[] += value
                    peak[] = max(peak[], value)
                end
                return nothing
            end
            evaluation = make_eval(pool_size = pool_size, recorder = recorder)
            evaluation(1, deepcopy(base_policy))

            @test is_complete(evaluation)
            @test calls[] == horizon * evaluation.last_n_ok
            # The recorded stage costs must sum to the rollout objectives, which
            # is only true if no update was lost to a race.
            @test total[] ≈ sum(last(d).objective for d in evaluation.last_scenario_data) rtol = 1e-8
            @test isfinite(peak[])
        end
    end

    # ── Per-scenario metrics keep their identity ────────────────────────────
    @testset "per-scenario metrics are attributable" begin
        # The hydro trainer logs `rollout_col_<id>` by mapping the evaluation
        # index through its protocol-id list. That mapping is only sound because
        # `last_scenario_data` carries the evaluation index; this pins it.
        protocol_ids = [11, 22, 33, 44]
        evaluation = make_eval(pool_size = 3)
        evaluation(1, deepcopy(base_policy))
        per_column = Dict(
            protocol_ids[i] => result.objective
            for (i, result) in evaluation.last_scenario_data
        )
        @test sort(collect(keys(per_column))) == protocol_ids
        @test evaluation.last_objective ≈ mean(values(per_column)) rtol = 1e-8
    end
end
