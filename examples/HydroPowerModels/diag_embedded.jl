using DecisionRulesExa, ExaModels, Flux, MadNLP, Statistics, Random

include(joinpath(@__DIR__, "hydro_power_data.jl"))
include(joinpath(@__DIR__, "hydro_power_exa.jl"))
include(joinpath(@__DIR__, "hydro_power_exa_embedded.jl"))

power_data = load_power_data(joinpath(@__DIR__, "bolivia/PowerModels.json"))
hydro_data = load_hydro_data(joinpath(@__DIR__, "bolivia/hydro.json"),
                              joinpath(@__DIR__, "bolivia/inflows.csv"),
                              power_data; num_stages=1260)
nHyd = hydro_data.nHyd
T = 126

Random.seed!(42)
policy = StateConditionedPolicy(nHyd, nHyd, nHyd, [128,128];
                                activation=sigmoid, encoder_type=Flux.LSTM)

x0_init = Float32.([clamp(hydro_data.initial_volumes[r],
                           hydro_data.units[r].min_vol,
                           hydro_data.units[r].max_vol)
                    for r in 1:nHyd])

@info "Building embedded AC DE (T=$T)..."
prob = build_embedded_hydro_de(policy, power_data, hydro_data, T;
    formulation=:ac_polar, target_penalty=:auto, deficit_cost=1e5)
@info "  nvar=$(prob._nvar) oracle_cons=$(length(prob.target_con_range))"

solver_kw = (print_level=MadNLP.ERROR, tol=1e-6, max_iter=9000)

for s in 1:5
    w = sample_scenario(hydro_data, T)
    set_x0!(prob, x0_init)
    set_inflows!(prob, w)

    t0 = time()
    result = MadNLP.madnlp(prob.model; solver_kw...)
    dt = time() - t0

    obj = result.objective
    λ = result.multipliers[prob.target_con_range]
    n_finite_λ = count(isfinite, λ)
    n_zero_λ = count(==(0.0), λ)

    @info "Solve $s: status=$(result.status) obj=$(round(obj; digits=2)) time=$(round(dt; digits=1))s finite_λ=$(n_finite_λ)/$(length(λ)) zero_λ=$n_zero_λ"
end

@info "Now testing _solve! with warmstart..."
state = DecisionRulesExa._make_solver(prob.model, solver_kw)
@info "  has_fixed_vars=$(state.has_fixed_vars)"

for s in 1:10
    w = sample_scenario(hydro_data, T)
    set_x0!(prob, x0_init)
    set_inflows!(prob, w)

    t0 = time()
    result = DecisionRulesExa._solve!(state, prob.model; warmstart=true, madnlp_kwargs=solver_kw)
    dt = time() - t0

    obj = result.objective
    succeeded = solve_succeeded(result)
    λ = result.multipliers[prob.target_con_range]
    n_finite_λ = count(isfinite, λ)
    n_zero_λ = count(==(0.0), λ)

    @info "  _solve! $s: ok=$succeeded status=$(result.status) obj=$(round(obj; digits=2)) time=$(round(dt; digits=1))s finite_λ=$(n_finite_λ)/$(length(λ)) zero_λ=$n_zero_λ"
end
