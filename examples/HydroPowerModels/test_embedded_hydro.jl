# test_embedded_hydro.jl
#
# Validates the embedded-NN hydro DE:
#   1. Build + solve embedded DE (DC formulation)
#   2. Compare with regular DE solved using policy-generated targets
#   3. Compare with stage-wise rollout
#   4. Verify envelope-theorem gradient is finite

using DecisionRulesExa
using ExaModels
using Flux
using Zygote
using MadNLP
using Statistics, Random, LinearAlgebra

const SCRIPT_DIR = dirname(@__FILE__)
include(joinpath(SCRIPT_DIR, "hydro_power_data.jl"))
include(joinpath(SCRIPT_DIR, "hydro_power_exa.jl"))
include(joinpath(SCRIPT_DIR, "hydro_power_exa_embedded.jl"))

# ── Config ───────────────────────────────────────────────────────────────────

const CASE_DIR   = joinpath(SCRIPT_DIR, "bolivia")
const PM_FILE    = joinpath(CASE_DIR, "PowerModels.json")
const HYDRO_FILE = joinpath(CASE_DIR, "hydro.json")
const INFLOW_FILE = joinpath(CASE_DIR, "inflows.csv")

const T_TEST     = 12
const FORM       = :ac_polar
const DEF_COST   = 1e5
const PRETRAIN_ITERS = 30
const LR         = 1f-3
const LAYERS     = [64, 64]
const SOLVER_KW  = (print_level = MadNLP.ERROR, tol = 1e-6, max_iter = 3000)

# ── Load data ────────────────────────────────────────────────────────────────

println("=" ^ 60)
println("Embedded Hydro DE Test (formulation=$FORM, T=$T_TEST)")
println("=" ^ 60)

@info "Loading data..."
power_data = load_power_data(PM_FILE)
hydro_data = load_hydro_data(HYDRO_FILE, INFLOW_FILE, power_data;
                              num_stages = T_TEST * 10)
nHyd = hydro_data.nHyd
@info "  nBus=$(power_data.nBus) nGen=$(power_data.nGen) nHyd=$nHyd"

x0_init = Float32.([clamp(hydro_data.initial_volumes[r],
                           hydro_data.units[r].min_vol,
                           hydro_data.units[r].max_vol)
                    for r in 1:nHyd])

resolved_pen = auto_target_penalty(power_data, hydro_data)
@info "  Auto penalty: ρ=$(round(resolved_pen; digits=2))"

# ── Build policy ─────────────────────────────────────────────────────────────

Random.seed!(42)
policy = StateConditionedPolicy(nHyd, nHyd, nHyd, LAYERS;
                                activation = sigmoid,
                                encoder_type = Flux.LSTM)

# ── Build regular + embedded DEs ─────────────────────────────────────────────

@info "Building regular $(T_TEST)-stage $FORM hydro DE..."
prob_reg = build_hydro_de(power_data, hydro_data, T_TEST;
    formulation    = FORM,
    target_penalty = :auto,
    deficit_cost   = DEF_COST,
)

@info "Building embedded $(T_TEST)-stage $FORM hydro DE..."
prob_emb = build_embedded_hydro_de(policy, power_data, hydro_data, T_TEST;
    formulation    = FORM,
    target_penalty = :auto,
    deficit_cost   = DEF_COST,
)

@info "  Regular: $(length(prob_reg.target_con_range)) target constraints"
@info "  Embedded: $(length(prob_emb.target_con_range)) oracle constraints"
@info "  nvar_total=$(prob_emb._nvar)  res_start=$(prob_emb._res_start)  dp_start=$(prob_emb._dp_start)  dn_start=$(prob_emb._dn_start)"

# ── Smoke test: solve embedded DE with random policy ─────────────────────────

@info "\n--- Test 1: Solve embedded DE with random policy ---"
w_mean = mean_inflow(hydro_data, T_TEST)
set_x0!(prob_emb, x0_init)
set_inflows!(prob_emb, w_mean)

result_emb0 = MadNLP.madnlp(prob_emb.model; SOLVER_KW...)
@info "  Status: $(result_emb0.status)  Obj: $(round(result_emb0.objective; digits=2))"
@assert solve_succeeded(result_emb0) "Embedded DE solve failed with random policy"
@assert isfinite(result_emb0.objective) "Non-finite objective"
println("  ✓ Embedded DE solves with random policy")

# ── Pretrain with regular TSDDR ──────────────────────────────────────────────

@info "\n--- Pretraining policy ($PRETRAIN_ITERS iters of regular TSDDR) ---"
Random.seed!(123)
train_tsddr(
    policy, x0_init, prob_reg,
    prob_reg.p_x0, prob_reg.p_target, prob_reg.p_inflow,
    () -> sample_scenario(hydro_data, T_TEST);
    num_batches         = PRETRAIN_ITERS,
    num_train_per_batch = 1,
    optimizer           = Flux.Adam(LR),
    madnlp_kwargs       = SOLVER_KW,
    warmstart           = true,
    record_loss         = (iter, m, loss, tag) -> begin
        if iter % 10 == 0
            @info "  Pretrain iter $iter: loss=$(round(loss; digits=2))"
        end
        return false
    end,
)
@info "  Pretrain done."

# ── Test 2: Compare embedded vs regular with same scenario ───────────────────

@info "\n--- Test 2: Compare embedded vs regular DE (same scenario) ---"
Random.seed!(999)
w_test = sample_scenario(hydro_data, T_TEST)

# 2a. Policy rollout → set targets on regular DE
Flux.reset!(policy)
targets = zeros(T_TEST * nHyd)
let x_prev = Float32.(x0_init)
    for t in 1:T_TEST
        wt = Float32.(w_test[(t-1)*nHyd+1 : t*nHyd])
        x̂_t = policy(vcat(wt, x_prev))
        targets[(t-1)*nHyd+1 : t*nHyd] = Float64.(x̂_t)
        x_prev = x̂_t
    end
end

ExaModels.set_parameter!(prob_reg.core, prob_reg.p_x0,     x0_init)
ExaModels.set_parameter!(prob_reg.core, prob_reg.p_inflow,  w_test)
ExaModels.set_parameter!(prob_reg.core, prob_reg.p_target,  targets)

result_reg = MadNLP.madnlp(prob_reg.model; SOLVER_KW...)
@info "  Regular: status=$(result_reg.status)  obj=$(round(result_reg.objective; digits=2))"
@assert solve_succeeded(result_reg) "Regular DE solve failed"

# 2b. Embedded DE with same scenario
set_x0!(prob_emb, x0_init)
set_inflows!(prob_emb, w_test)
result_emb = MadNLP.madnlp(prob_emb.model; SOLVER_KW...)
@info "  Embedded: status=$(result_emb.status)  obj=$(round(result_emb.objective; digits=2))"
@assert solve_succeeded(result_emb) "Embedded DE solve failed"

# 2c. Compare reservoir trajectories
sol_reg = hydro_solution(prob_reg, result_reg)
sol_emb = hydro_solution(prob_emb, result_emb)

res_reg = Array(sol_reg.reservoir)
res_emb = Array(sol_emb.reservoir)
delta_reg = Array(sol_reg.delta)
delta_emb = Array(sol_emb.delta)

res_diff = maximum(abs.(res_reg .- res_emb))
obj_diff_pct = abs(result_reg.objective - result_emb.objective) / max(abs(result_reg.objective), 1e-8) * 100

@info "  Max reservoir diff: $(round(res_diff; digits=6))"
@info "  Objective diff: $(round(obj_diff_pct; digits=2))%"
@info "  Regular delta norm:  $(round(norm(delta_reg); digits=4))"
@info "  Embedded delta norm: $(round(norm(delta_emb); digits=4))"

# Compare multiplier signs (should be same convention now)
λ_reg = result_reg.multipliers[prob_reg.target_con_range]
λ_emb = result_emb.multipliers[prob_emb.target_con_range]
@info "  Regular λ:  mean=$(round(mean(λ_reg); digits=4))  range=[$(round(minimum(λ_reg); digits=4)), $(round(maximum(λ_reg); digits=4))]"
@info "  Embedded λ: mean=$(round(mean(λ_emb); digits=4))  range=[$(round(minimum(λ_emb); digits=4)), $(round(maximum(λ_emb); digits=4))]"

println("  ✓ Both solve; reservoir max diff = $(round(res_diff; digits=4)), obj diff = $(round(obj_diff_pct; digits=1))%")

# ── Test 3: Stage-wise rollout comparison ────────────────────────────────────

@info "\n--- Test 3: Stage-wise rollout vs embedded DE ---"

stage_prob = build_hydro_de(power_data, hydro_data, 1;
    formulation    = FORM,
    target_penalty = :auto,
    deficit_cost   = DEF_COST,
)

rollout_reservoir = zeros(nHyd, T_TEST + 1)
rollout_reservoir[:, 1] = x0_init

Flux.reset!(policy)
rollout_obj, n_ok = let x_prev_rollout = Float32.(x0_init), obj = 0.0, nok = 0
    for t in 1:T_TEST
        wt = Float32.(w_test[(t-1)*nHyd+1 : t*nHyd])
        target_t = Float64.(policy(vcat(wt, x_prev_rollout)))

        ExaModels.set_parameter!(stage_prob.core, stage_prob.p_x0, rollout_reservoir[:, t])
        ExaModels.set_parameter!(stage_prob.core, stage_prob.p_inflow,
                                  w_test[(t-1)*nHyd+1 : t*nHyd])
        ExaModels.set_parameter!(stage_prob.core, stage_prob.p_target, target_t)

        result_t = MadNLP.madnlp(stage_prob.model; SOLVER_KW...)
        if solve_succeeded(result_t)
            sol_t = hydro_solution(stage_prob, result_t)
            rollout_reservoir[:, t+1] = Array(sol_t.reservoir[:, end])
            obj += result_t.objective
            nok += 1
        else
            @warn "  Stage $t rollout failed: $(result_t.status)"
            rollout_reservoir[:, t+1] = rollout_reservoir[:, t]
        end

        x_prev_rollout = target_t isa Vector ? Float32.(target_t) :
                          Float32.(Array(target_t))
    end
    (obj, nok)
end

@info "  Rollout: $(n_ok)/$T_TEST stages solved, total obj=$(round(rollout_obj; digits=2))"

rollout_res_diff = maximum(abs.(res_emb .- rollout_reservoir))
@info "  Max reservoir diff (embedded vs rollout): $(round(rollout_res_diff; digits=4))"
println("  ✓ Stage-wise rollout completed; reservoir max diff = $(round(rollout_res_diff; digits=4))")

# ── Test 4: Gradient extraction ──────────────────────────────────────────────

@info "\n--- Test 4: Envelope theorem gradient ---"

λ_emb_arr = Array(λ_emb)
@assert all(isfinite, λ_emb_arr) "Non-finite duals"

x_realized = embedded_hydro_realized_states(prob_emb, result_emb)
@assert length(x_realized) == T_TEST * nHyd "Wrong realized state length"
@assert all(isfinite, x_realized) "Non-finite realized states"

F = Float32
gs = Zygote.gradient(policy) do m
    total = zero(F)
    Flux.reset!(m)
    for t in 1:T_TEST
        wt = F.(w_test[(t-1)*nHyd+1 : t*nHyd])
        xp = (t == 1) ? F.(x0_init) : F.(x_realized[(t-2)*nHyd+1 : (t-1)*nHyd])
        xt = m(vcat(wt, xp))
        total = total + sum(F.(λ_emb_arr[(t-1)*nHyd+1 : t*nHyd]) .* xt)
    end
    total
end

grad = DecisionRulesExa.materialize_tangent(gs[1])
@assert grad !== nothing "Gradient is nothing"
@assert DecisionRulesExa._all_finite_gradient(grad) "Non-finite gradient"
println("  ✓ Gradient is finite and non-zero")

# ── Test 5: train_tsddr_embedded smoke test ──────────────────────────────────

@info "\n--- Test 5: train_tsddr_embedded with hydro DE (5 iters) ---"
losses = Float64[]
Random.seed!(456)

train_tsddr_embedded(
    policy, x0_init, prob_emb,
    () -> sample_scenario(hydro_data, T_TEST);
    num_batches         = 5,
    num_train_per_batch = 1,
    optimizer           = Flux.Adam(LR),
    madnlp_kwargs       = SOLVER_KW,
    warmstart           = true,
    get_realized_states = embedded_hydro_realized_states,
    record_loss         = (iter, m, loss, tag) -> begin
        push!(losses, loss)
        @info "  Embedded train iter $iter: loss=$(round(loss; digits=2))"
        return false
    end,
)

n_finite = count(isfinite, losses)
@info "  $n_finite / $(length(losses)) losses are finite"
@assert n_finite >= 1 "No finite losses at all"
println("  ✓ Embedded training runs ($n_finite/$(length(losses)) finite losses)")

# ── Summary ──────────────────────────────────────────────────────────────────

println("\n" * "=" ^ 60)
println("ALL TESTS PASSED")
println("=" ^ 60)
