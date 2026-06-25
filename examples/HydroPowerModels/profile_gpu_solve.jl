# profile_gpu_solve.jl
#
# Profiles the embedded-NN AC polar GPU solve to identify bottlenecks.
# Tests: scalar indexing, oracle callback timing, cuDSS vs oracle breakdown,
# Float32/Float64 conversion overhead.

using DecisionRulesExa
using ExaModels
using Flux
using Statistics, Random
using MadNLP, MadNLPGPU
using CUDA, CUDSS, KernelAbstractions
using Zygote

const SCRIPT_DIR = dirname(@__FILE__)
include(joinpath(SCRIPT_DIR, "hydro_power_data.jl"))
include(joinpath(SCRIPT_DIR, "hydro_power_exa.jl"))
include(joinpath(SCRIPT_DIR, "hydro_power_exa_embedded.jl"))

const FORMULATION = :ac_polar
const CASE_DIR    = joinpath(SCRIPT_DIR, "bolivia")

power_data = load_power_data(joinpath(CASE_DIR, "PowerModels.json"))
hydro_data = load_hydro_data(
    joinpath(CASE_DIR, "hydro.json"),
    joinpath(CASE_DIR, "inflows.csv"),
    power_data;
    num_stages = 126 * 10,
)
nHyd = hydro_data.nHyd
T    = 126

demand_csv = joinpath(CASE_DIR, "demand.csv")
demand_mat = isfile(demand_csv) ? load_demand(demand_csv, power_data; T = T) : nothing
load_scaler = 0.6

@info "Problem dims: nBus=$(power_data.nBus) nGen=$(power_data.nGen) nBranch=$(power_data.nBranch) nHyd=$nHyd T=$T"

x0_init = Float32.([clamp(hydro_data.initial_volumes[r],
                           hydro_data.units[r].min_vol,
                           hydro_data.units[r].max_vol)
                    for r in 1:nHyd])

Random.seed!(42)
policy = StateConditionedPolicy(nHyd, nHyd, nHyd, [128, 128];
                                activation = sigmoid, encoder_type = Flux.LSTM)

SOLVER_KWARGS = (print_level = MadNLP.ERROR, tol = 1e-6, max_iter = 9000)
DEFICIT_COST = 1e5

# ── Phase 1: Scalar indexing check ────────────────────────────────────────────

@info "Phase 1: Testing with CUDA.allowscalar(false)"

policy_gpu = CUDA.cu(policy)
x0_gpu     = CUDA.cu(x0_init)
backend    = CUDA.CUDABackend()

CUDA.allowscalar(false)

@info "Building AC embedded DE on GPU..."
prob_emb = build_embedded_hydro_de(policy_gpu, power_data, hydro_data, T;
    backend        = backend,
    formulation    = FORMULATION,
    target_penalty = :auto,
    deficit_cost   = DEFICIT_COST,
    demand_matrix  = demand_mat,
    load_scaler    = load_scaler,
)
@info "  nvar=$(prob_emb._nvar)  oracle_cons=$(length(prob_emb.target_con_range))"

w_mean = mean_inflow(hydro_data, T)
set_x0!(prob_emb, x0_gpu)
set_inflows!(prob_emb, w_mean)

@info "Smoke test: solving embedded DE (gpu=true, allowscalar=false)..."
try
    result0 = MadNLP.madnlp(prob_emb.model; SOLVER_KWARGS..., print_level = MadNLP.WARN)
    @info "  Status: $(result0.status)  Obj: $(round(result0.objective; digits=4))"
    @info "  PASS: No scalar indexing violations detected"
catch e
    @error "  FAIL: Scalar indexing detected!" exception=(e, catch_backtrace())
    @info "  Re-running with allowscalar(true) to get timing baseline..."
    CUDA.allowscalar(true)
    result0 = MadNLP.madnlp(prob_emb.model; SOLVER_KWARGS..., print_level = MadNLP.WARN)
    @info "  Status: $(result0.status)  Obj: $(round(result0.objective; digits=4))"
end

# ── Phase 2: Timed solve breakdown ───────────────────────────────────────────

@info "\nPhase 2: Timed solve breakdown (3 warm-start solves)"

CUDA.allowscalar(true)

solver = MadNLP.MadNLPSolver(prob_emb.model; SOLVER_KWARGS..., print_level = MadNLP.ERROR)
w_samples = [Float64.(sample_scenario(hydro_data, T)) for _ in 1:3]

for (i, w) in enumerate(w_samples)
    set_x0!(prob_emb, x0_gpu)
    set_inflows!(prob_emb, w)

    solver.cnt.k = 0
    solver.cnt.acceptable_cnt = 0
    solver.cnt.start_time = time()

    CUDA.synchronize()
    t_solve = @elapsed begin
        result_i = MadNLP.solve!(solver)
        CUDA.synchronize()
    end

    status = result_i.status
    obj    = round(result_i.objective; digits=4)
    iters  = solver.cnt.k
    @info "  Solve $i: $(round(t_solve; digits=2))s  status=$status  obj=$obj  iters=$iters"
end

# ── Phase 3: Oracle callback profiling ───────────────────────────────────────

@info "\nPhase 3: Individual oracle callback timing"

x_vec = prob_emb.model.meta.x0
n_oracle_con = T * nHyd
nnzj_oracle  = prob_emb.model.oracles[1].nnzj
nvar_total   = prob_emb._nvar

c_buf    = CUDA.zeros(Float64, n_oracle_con)
jac_buf  = CUDA.zeros(Float64, nnzj_oracle)
Jtv_buf  = CUDA.zeros(Float64, nvar_total)
lam_buf  = CUDA.ones(Float64, n_oracle_con) .* 0.01

oracle = prob_emb.model.oracles[1]

CUDA.synchronize()
t_f = @elapsed begin
    for _ in 1:10
        oracle.f!(c_buf, x_vec)
    end
    CUDA.synchronize()
end

CUDA.synchronize()
t_jac = @elapsed begin
    for _ in 1:10
        oracle.jac!(jac_buf, x_vec)
    end
    CUDA.synchronize()
end

CUDA.synchronize()
t_vjp = @elapsed begin
    for _ in 1:10
        oracle.vjp!(Jtv_buf, x_vec, lam_buf)
    end
    CUDA.synchronize()
end

@info "  oracle_f!:   $(round(t_f/10*1000; digits=1))ms  per call"
@info "  oracle_jac!: $(round(t_jac/10*1000; digits=1))ms  per call"
@info "  oracle_vjp!: $(round(t_vjp/10*1000; digits=1))ms  per call"

# ── Phase 4: Allocation analysis ─────────────────────────────────────────────

@info "\nPhase 4: Allocation analysis (single call)"

alloc_f = @allocated oracle.f!(c_buf, x_vec)
alloc_j = @allocated oracle.jac!(jac_buf, x_vec)
alloc_v = @allocated oracle.vjp!(Jtv_buf, x_vec, lam_buf)

@info "  oracle_f!   allocations: $(round(alloc_f/1024; digits=1)) KB"
@info "  oracle_jac! allocations: $(round(alloc_j/1024; digits=1)) KB"
@info "  oracle_vjp! allocations: $(round(alloc_v/1024; digits=1)) KB"

# ── Phase 5: ForwardDiff Jacobian alternative ────────────────────────────────

@info "\nPhase 5: ForwardDiff vs Zygote pullback for local NN Jacobian"

using ForwardDiff

x_test = CUDA.rand(Float32, nHyd)
w_test = CUDA.rand(Float32, nHyd)
Flux.reset!(policy_gpu)

x_cpu = Array(x_test)
w_cpu = Array(w_test)
policy_cpu = Flux.cpu(policy_gpu)
Flux.reset!(policy_cpu)

t_fd_cpu = @elapsed begin
    for _ in 1:100
        Flux.reset!(policy_cpu)
        policy_cpu(vcat(w_cpu, x_cpu))
        J_fd = ForwardDiff.jacobian(xp -> Array(policy_cpu(vcat(w_cpu, xp))), x_cpu)
    end
end

t_zy_cpu = @elapsed begin
    for _ in 1:100
        Flux.reset!(policy_cpu)
        policy_cpu(vcat(w_cpu, x_cpu))
        _, back = Zygote.pullback(xp -> policy_cpu(vcat(w_cpu, xp)), x_cpu)
        cols = [back(let e = zeros(Float32, nHyd); e[r] = 1f0; e end)[1] for r in 1:nHyd]
        J_zy = hcat(cols...)
    end
end

@info "  ForwardDiff Jacobian (CPU, 100 calls): $(round(t_fd_cpu*10; digits=1))ms/call"
@info "  Zygote pullback×nHyd (CPU, 100 calls): $(round(t_zy_cpu*10; digits=1))ms/call"
@info "  Speedup: $(round(t_zy_cpu/t_fd_cpu; digits=2))x"

# ── Phase 6: Envelope gradient timing ────────────────────────────────────────

@info "\nPhase 6: Envelope theorem gradient timing (T=$T)"

w_sample = Float32.(sample_scenario(hydro_data, T))

set_x0!(prob_emb, x0_gpu)
set_inflows!(prob_emb, Float64.(w_sample))

solver.cnt.k = 0
solver.cnt.acceptable_cnt = 0
solver.cnt.start_time = time()
res_grad = MadNLP.solve!(solver)

if DecisionRulesExa.solve_succeeded(res_grad)
    F = Float32
    λ = res_grad.multipliers[prob_emb.target_con_range]
    x_sol = embedded_hydro_realized_states(prob_emb, res_grad)
    initial_state = x0_gpu

    λf   = DecisionRulesExa._adapt_array(F.(λ), initial_state)
    xf   = DecisionRulesExa._adapt_array(F.(x_sol), initial_state)
    w_dev = DecisionRulesExa._adapt_array(F.(w_sample), initial_state)
    nx = prob_emb.nx

    CUDA.synchronize()
    t_grad = @elapsed begin
        gs = Zygote.gradient(policy_gpu) do m
            total = zero(F)
            Flux.reset!(m)
            for t in 1:T
                nw = nHyd
                wt = F.(w_dev[(t-1)*nw+1 : t*nw])
                x_prev = (t == 1) ?
                    F.(initial_state) :
                    F.(xf[(t-2)*nx+1 : (t-1)*nx])
                xt = m(vcat(wt, x_prev))
                total = total + sum(λf[(t-1)*nx+1 : t*nx] .* xt)
            end
            total
        end
        CUDA.synchronize()
    end
    @info "  Envelope gradient: $(round(t_grad*1000; digits=1))ms"
else
    @info "  Solve failed, skipping gradient timing"
end

# ── Summary ──────────────────────────────────────────────────────────────────

@info "\n========== PROFILING SUMMARY =========="
@info "Problem: Bolivia AC polar, T=$T, nHyd=$nHyd"
@info "nvar=$(prob_emb._nvar)"
@info "Oracle adapt flag: $(oracle.adapt)"
@info "Oracle callback times (per call):"
@info "  f!:   $(round(t_f/10*1000; digits=1))ms"
@info "  jac!: $(round(t_jac/10*1000; digits=1))ms"
@info "  vjp!: $(round(t_vjp/10*1000; digits=1))ms"
@info "Oracle allocations (per call):"
@info "  f!:   $(round(alloc_f/1024; digits=1))KB"
@info "  jac!: $(round(alloc_j/1024; digits=1))KB"
@info "  vjp!: $(round(alloc_v/1024; digits=1))KB"
@info "ForwardDiff vs Zygote: $(round(t_zy_cpu/t_fd_cpu; digits=2))x slower with Zygote pullback×nHyd"
