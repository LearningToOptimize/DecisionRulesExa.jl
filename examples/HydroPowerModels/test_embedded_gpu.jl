# test_embedded_gpu.jl — GPU validation for embedded-NN training pipeline
#
# Tests (GPU-native oracle, adapt=Val(false)):
# 1. GPU ExaModels build + single solve
# 2. Repeated _solve! (warm-start, no cascade failure)
# 3. Envelope-theorem gradient on GPU (solve → GPU Zygote)
# 4. Full training iteration (solve → grad → Flux.update!)
# 5. GPU vs CPU timing comparison
# 6. Full-horizon (T=126) GPU solve
# 7. Non-embedded GPU DE solve (baseline)

using DecisionRulesExa
using ExaModels
using Flux
using MadNLP, MadNLPGPU
using CUDA, CUDSS, KernelAbstractions
using Statistics, Random, Printf
using Zygote

@assert CUDA.functional() "CUDA not available!"
@info "GPU: $(CUDA.name(CUDA.device())) — $(round(CUDA.total_memory() / 1e9; digits=1)) GB"

const SCRIPT_DIR = dirname(@__FILE__)
include(joinpath(SCRIPT_DIR, "hydro_power_data.jl"))
include(joinpath(SCRIPT_DIR, "hydro_power_exa.jl"))
include(joinpath(SCRIPT_DIR, "hydro_power_exa_embedded.jl"))

DIR = SCRIPT_DIR
power_data = load_power_data(joinpath(DIR, "bolivia/PowerModels.json"))
hydro_data = load_hydro_data(joinpath(DIR, "bolivia/hydro.json"),
                              joinpath(DIR, "bolivia/inflows.csv"),
                              power_data; num_stages=1260)
nHyd = hydro_data.nHyd
T_short = 12
T_full  = 126
F = Float32

Random.seed!(42)
policy = StateConditionedPolicy(nHyd, nHyd, nHyd, [128,128];
                                activation=sigmoid, encoder_type=Flux.LSTM)
x0_init_cpu = F.([clamp(hydro_data.initial_volumes[r],
                         hydro_data.units[r].min_vol,
                         hydro_data.units[r].max_vol)
                  for r in 1:nHyd])

policy  = Flux.gpu(policy)
x0_init = CUDA.cu(x0_init_cpu)
@info "Policy and x0 moved to GPU"

solver_kw = (print_level=MadNLP.ERROR, tol=1e-6, max_iter=9000)
gpu_backend = CUDA.CUDABackend()
n_pass = 0
n_fail = 0

function pass(msg)
    global n_pass += 1
    @info "✓ $msg"
end
function fail(msg)
    global n_fail += 1
    @error "✗ $msg"
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: GPU embedded build + single solve (T=12)
# ═══════════════════════════════════════════════════════════════════════════════
@info "\n=== TEST 1: GPU Embedded Build + Single Solve (T=$T_short) ==="
prob_gpu = build_embedded_hydro_de(policy, power_data, hydro_data, T_short;
    backend = gpu_backend, formulation = :ac_polar,
    target_penalty = :auto, deficit_cost = 1e5)
@info "  nvar=$(prob_gpu._nvar) oracle_cons=$(length(prob_gpu.target_con_range))"

w = sample_scenario(hydro_data, T_short)
set_x0!(prob_gpu, x0_init)
set_inflows!(prob_gpu, w)
t0 = time()
res1 = MadNLP.madnlp(prob_gpu.model; solver_kw...)
dt1 = time() - t0
if solve_succeeded(res1)
    lam = Array(res1.multipliers[prob_gpu.target_con_range])
    nf = count(isfinite, lam)
    nz = count(==(0.0), lam)
    pass("GPU solve OK: obj=$(round(res1.objective;digits=2)) t=$(round(dt1;digits=1))s λ_finite=$nf/$(length(lam)) λ_zero=$nz")
else
    fail("GPU solve FAILED: $(res1.status)")
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: Repeated GPU solves via _solve! (fixed-var fresh madnlp path)
# ═══════════════════════════════════════════════════════════════════════════════
@info "\n=== TEST 2: Repeated _solve! on GPU (T=$T_short, 10 solves) ==="
state = DecisionRulesExa._make_solver(prob_gpu.model, solver_kw)
@info "  has_fixed_vars=$(state.has_fixed_vars)"
ok_count, times = let ok=0, ts=Float64[]
    for s in 1:10
        local w = sample_scenario(hydro_data, T_short)
        set_x0!(prob_gpu, x0_init)
        set_inflows!(prob_gpu, w)
        local t0 = time()
        local result = DecisionRulesExa._solve!(state, prob_gpu.model; warmstart=true, madnlp_kwargs=solver_kw)
        push!(ts, time() - t0)
        if solve_succeeded(result) && isfinite(result.objective)
            local lam = Array(result.multipliers[prob_gpu.target_con_range])
            if all(isfinite, lam) && any(!=(0.0), lam)
                ok += 1
            end
        end
    end
    (ok, ts)
end
if ok_count == 10
    pass("All 10 GPU _solve! succeeded — mean=$(round(mean(times[2:end]);digits=2))s (excl JIT)")
else
    fail("Only $ok_count/10 GPU _solve! succeeded")
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: Envelope theorem gradient on GPU result
# ═══════════════════════════════════════════════════════════════════════════════
@info "\n=== TEST 3: Envelope Theorem Gradient (GPU end-to-end) ==="
w = sample_scenario(hydro_data, T_short)
set_x0!(prob_gpu, x0_init)
set_inflows!(prob_gpu, w)
res_g = MadNLP.madnlp(prob_gpu.model; solver_kw...)
@assert solve_succeeded(res_g) "Solve failed for gradient test"

λ = F.(res_g.multipliers[prob_gpu.target_con_range])
x_sol = F.(res_g.solution[1 : T_short * nHyd])
w_dev = CUDA.cu(F.(w))

t0 = time()
gs = Zygote.gradient(policy) do m
    total = zero(F)
    Flux.reset!(m)
    for t in 1:T_short
        wt = w_dev[(t-1)*nHyd+1 : t*nHyd]
        x_prev = t == 1 ? x0_init : x_sol[(t-2)*nHyd+1 : (t-1)*nHyd]
        xt = m(vcat(wt, x_prev))
        total = total + sum(λ[(t-1)*nHyd+1 : t*nHyd] .* xt)
    end
    total
end
dt_grad = time() - t0

g = gs[1]
if g !== nothing
    grad_norm = sqrt(sum(sum(abs2, p) for p in Flux.trainables(g) if p isa AbstractArray))
    if isfinite(grad_norm) && grad_norm > 0
        pass("Gradient OK: norm=$(round(grad_norm;digits=6)) t=$(round(dt_grad;digits=2))s")
    else
        fail("Gradient non-finite or zero: norm=$grad_norm")
    end
else
    fail("Gradient is nothing")
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: Full training iteration (solve → grad → update)
# ═══════════════════════════════════════════════════════════════════════════════
@info "\n=== TEST 4: Full Training Iteration ==="
opt_state = Flux.setup(Flux.Adam(1f-3), policy)
initial_params = [Array(copy(p)) for p in Flux.trainables(policy)]

w = sample_scenario(hydro_data, T_short)
set_x0!(prob_gpu, x0_init)
set_inflows!(prob_gpu, w)
res_t = MadNLP.madnlp(prob_gpu.model; solver_kw...)
@assert solve_succeeded(res_t)

λ_t = F.(res_t.multipliers[prob_gpu.target_con_range])
x_sol_t = F.(res_t.solution[1 : T_short * nHyd])
w_dev_t = CUDA.cu(F.(w))

gs_t = Zygote.gradient(policy) do m
    total = zero(F)
    Flux.reset!(m)
    for t in 1:T_short
        wt = w_dev_t[(t-1)*nHyd+1 : t*nHyd]
        x_prev = t == 1 ? x0_init : x_sol_t[(t-2)*nHyd+1 : (t-1)*nHyd]
        xt = m(vcat(wt, x_prev))
        total = total + sum(λ_t[(t-1)*nHyd+1 : t*nHyd] .* xt)
    end
    total
end

grad_t = DecisionRulesExa.materialize_tangent(gs_t[1])
if grad_t !== nothing && DecisionRulesExa._all_finite_gradient(grad_t)
    Flux.update!(opt_state, policy, grad_t)
    params_changed = any(Array(p1) != p2 for (p1, p2) in zip(Flux.trainables(policy), initial_params))
    if params_changed
        pass("Training iteration OK: params updated")
    else
        fail("Training iteration: params unchanged after update")
    end
else
    fail("Training iteration: gradient invalid")
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: GPU vs CPU timing comparison (T=12)
# ═══════════════════════════════════════════════════════════════════════════════
@info "\n=== TEST 5: GPU vs CPU Timing (T=$T_short) ==="
policy_cpu = Flux.cpu(policy)
prob_cpu = build_embedded_hydro_de(policy_cpu, power_data, hydro_data, T_short;
    backend = nothing, formulation = :ac_polar,
    target_penalty = :auto, deficit_cost = 1e5)

gpu_times, cpu_times = let gt=Float64[], ct=Float64[]
    for s in 1:5
        local w = sample_scenario(hydro_data, T_short)

        set_x0!(prob_gpu, x0_init); set_inflows!(prob_gpu, w)
        local t0 = time(); MadNLP.madnlp(prob_gpu.model; solver_kw...); push!(gt, time() - t0)

        set_x0!(prob_cpu, x0_init_cpu); set_inflows!(prob_cpu, w)
        t0 = time(); MadNLP.madnlp(prob_cpu.model; solver_kw...); push!(ct, time() - t0)
    end
    (gt, ct)
end
@info @sprintf("  GPU: %.2fs mean (%.2f-%.2f)", mean(gpu_times), minimum(gpu_times), maximum(gpu_times))
@info @sprintf("  CPU: %.2fs mean (%.2f-%.2f)", mean(cpu_times), minimum(cpu_times), maximum(cpu_times))
speedup = mean(cpu_times) / mean(gpu_times)
pass(@sprintf("GPU speedup: %.1fx", speedup))

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: Full-horizon GPU embedded (T=126) — production scale
# ═══════════════════════════════════════════════════════════════════════════════
@info "\n=== TEST 6: Full-Horizon GPU Embedded (T=$T_full) ==="
prob_full = build_embedded_hydro_de(policy, power_data, hydro_data, T_full;
    backend = gpu_backend, formulation = :ac_polar,
    target_penalty = :auto, deficit_cost = 1e5)
@info "  nvar=$(prob_full._nvar)"

w = sample_scenario(hydro_data, T_full)
set_x0!(prob_full, x0_init)
set_inflows!(prob_full, w)
t0 = time()
res_full = MadNLP.madnlp(prob_full.model; solver_kw...)
dt_full = time() - t0
if solve_succeeded(res_full)
    lam = Array(res_full.multipliers[prob_full.target_con_range])
    nf = count(isfinite, lam)
    pass("T=$T_full GPU solve OK: obj=$(round(res_full.objective;digits=2)) t=$(round(dt_full;digits=1))s λ_finite=$nf/$(length(lam))")
else
    fail("T=$T_full GPU solve FAILED: $(res_full.status)")
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 7: Non-embedded GPU DE (baseline comparison)
# ═══════════════════════════════════════════════════════════════════════════════
@info "\n=== TEST 7: Non-Embedded GPU DE (T=$T_short) ==="
import DecisionRulesExa: set_x0!, set_uncertainty!, set_targets!
prob_de_gpu = build_hydro_de(power_data, hydro_data, T_short;
    backend = gpu_backend, formulation = :ac_polar,
    target_penalty = :auto, deficit_cost = 1e5)

w_de = sample_scenario(hydro_data, T_short)
w_de_dev = CUDA.cu(F.(w_de))
Flux.reset!(policy)
xhat_flat = let prev = copy(x0_init), stages = typeof(x0_init)[]
    for t in 1:T_short
        local wt = w_de_dev[(t-1)*nHyd+1 : t*nHyd]
        push!(stages, policy(vcat(wt, prev)))
        prev = stages[end]
    end
    vcat(stages...)
end

ExaModels.set_parameter!(prob_de_gpu.core, prob_de_gpu.p_x0, x0_init)
ExaModels.set_parameter!(prob_de_gpu.core, prob_de_gpu.p_inflow, w_de)
ExaModels.set_parameter!(prob_de_gpu.core, prob_de_gpu.p_target, Float64.(xhat_flat))

t0 = time()
res_de = MadNLP.madnlp(prob_de_gpu.model; solver_kw...)
dt_de = time() - t0
if solve_succeeded(res_de)
    pass("Non-embedded GPU DE: obj=$(round(res_de.objective;digits=2)) t=$(round(dt_de;digits=1))s")
else
    fail("Non-embedded GPU DE FAILED: $(res_de.status)")
end

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
@info "\n" * "="^60
@info "RESULTS: $n_pass passed, $n_fail failed out of $(n_pass + n_fail) tests"
@info "="^60
n_fail > 0 && error("$n_fail test(s) failed!")
@info "ALL TESTS PASSED — GPU embedded pipeline validated"
