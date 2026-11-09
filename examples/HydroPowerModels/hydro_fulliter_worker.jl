# hydro_fulliter_worker.jl — what one worker process of the full-iterate timing
# (hydro_fulliter.jl, DR_TIMING_ONLY=4) holds: the case data, ONE one-stage strict
# CPU model (demand is the same at every stage, so one model serves every week; the
# week enters through its incoming state, inflow and target), and a copy of the
# policy for its share of the rollout and of the reverse-mode gradient.
using DecisionRulesExa, ExaModels, Flux, MadNLP, LinearAlgebra
using Statistics, Random, JLD2, JSON
BLAS.set_num_threads(1)

const SCRIPT_DIR = @__DIR__
include(joinpath(SCRIPT_DIR, "hydro_training_utils.jl"))
include(joinpath(SCRIPT_DIR, "hydro_power_data.jl"))
include(joinpath(SCRIPT_DIR, "hydro_power_exa.jl"))
include(joinpath(SCRIPT_DIR, "hydro_reachable_policy.jl"))

const W_CASE_DIR = joinpath(SCRIPT_DIR, "bolivia")
const W_T = parse(Int, get(ENV, "DR_NUM_STAGES", "126"))
const W_TR = parse(Int, get(ENV, "DR_NUM_ROLLOUT_STAGES", "96"))
const W_MAX_ITER = parse(Int, get(ENV, "DR_MAX_ITER", "9000"))
# The trainer's solver options and model arguments (train_hydro_exa_strict.jl).
const W_KW = (print_level = MadNLP.ERROR, tol = 1e-6, max_iter = W_MAX_ITER)
const W_POWER = load_power_data(joinpath(W_CASE_DIR, "PowerModels.json"))
const W_HYDRO = load_hydro_data(joinpath(W_CASE_DIR, "hydro.json"), joinpath(W_CASE_DIR, "inflows.csv"), W_POWER;
                                num_stages = max(W_T, W_TR) * 10)
const W_NH = W_HYDRO.nHyd
"The one-stage strict model of this worker (the trainer's builder, T = 1)."
const W_STAGE = build_hydro_de(W_POWER, W_HYDRO, 1; backend = nothing, float_type = Float64,
                               formulation = :ac_polar, target_penalty = :auto, deficit_cost = 6000.0,
                               demand_matrix = nothing, reactive_demand_matrix = nothing,
                               load_scaler = 0.6, strict_targets = true, reactive_deficit_cost = Inf,
                               demand_spread = nothing)
"This worker's copy of the policy and its flat-parameter rebuilder."
const W_RE = Ref{Any}(nothing)

"Receive the policy once; later updates send only its flat parameters."
function fw_set_policy!(p)
    _, re = Flux.destructure(p)
    W_RE[] = re
    return true
end

"The trainer's rollout (`rollout_reachable_targets`) under parameters `θ`."
function fw_rollout(θ, x0, w)
    m = W_RE[](θ); Flux.reset!(m)
    nu = length(w) ÷ W_T; prev = x0
    out = Vector{Vector{Float32}}(undef, W_T)
    for t in 1:W_T
        out[t] = Float32.(m(vcat(Float32.(view(w, ((t - 1) * nu + 1):(t * nu))), prev)))
        prev = out[t]
    end
    return Float64.(vcat(out...))
end

"""
    fw_stage(sid, t, xp, wt, xt) -> NamedTuple

Week `t` of scenario `sid` alone: incoming volume `xp`, inflow `wt`, target `xt`
(the cascade-clamped values the full model would carry). Returns the water-balance
multipliers of the week, which assemble into the full model's target multipliers.
"""
function fw_stage(sid, t, xp, wt, xt)
    p = W_STAGE
    ExaModels.set_parameter!(p.core, p.p_x0, xp)
    ExaModels.set_parameter!(p.core, p.p_inflow, wt)
    ExaModels.set_parameter!(p.core, p.p_target, xt)
    prepare_solve!(p, xp, wt, xt)
    t0 = time()
    r = MadNLP.madnlp(p.model; W_KW...)
    s = time() - t0
    ok = solve_succeeded(r)
    return (sid = sid, t = t, ok = ok, status = string(r.status), iter = r.iter, obj = Float64(r.objective),
            y = Float64.(vec(Array(r.multipliers[p.target_con_range]))), s = s)
end

"""
    fw_grad(θ, x0, w, λ) -> Vector

Flat gradient of ``\\sum_t \\lambda_t^\\top \\hat x_t(\\theta)`` for one scenario, the
trainer's actor closure (training.jl) restricted to that scenario.
"""
function fw_grad(θ, x0, w, λ)
    re = W_RE[]
    g = Flux.gradient(θ) do th
        m = re(th); Flux.reset!(m)
        nu = length(w) ÷ W_T; prev = x0; tot = zero(Float32)
        for t in 1:W_T
            xt = m(vcat(Float32.(view(w, ((t - 1) * nu + 1):(t * nu))), prev))
            tot += sum(Float32.(view(λ, ((t - 1) * W_NH + 1):(t * W_NH))) .* xt)
            prev = xt
        end
        tot
    end
    return g[1]
end
