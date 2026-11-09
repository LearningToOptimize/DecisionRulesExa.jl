# hydro_fulliter.jl — wall time of one complete TS-DDR update (DR_TIMING_ONLY=4),
# included by train_hydro_exa_strict.jl after its setup and smoke test, so `policy`,
# `hydro_data`, `prob` (the full strict DE, CPU), `x0_init`, `T` and `nHyd` are the
# trainer's own.
#
# A strict update of S scenarios separates into S×T independent one-week problems.
# Here every part of the update runs on Distributed worker PROCESSES, one per core
# (threads of one process do not run solves concurrently):
#   rollout    each scenario's T policy evaluations on a worker,
#   solves     the S×T weekly problems, dynamically scheduled over the workers,
#   gradient   each scenario's reverse-mode pass on a worker, summed on the master.
# The cascade clamp of `prepare_solve!` is applied on the master to every scenario
# (no solve), so each week receives the volumes the full model would carry.
# The first update of each S is checked against the trainer's own path: the full
# T-week DE solved per scenario and `target_multipliers`, and the actor gradient
# of the same scenarios computed in one process.
#   Env: DR_FI_WORKERS (default 24), DR_FI_TRAJ ('+'-separated, default 16+24),
#        DR_FI_BATCHES (default 4), DR_FI_OUT (JSON),
#        DR_FI_NET ("workers": rollout, solves and gradients one after another, one
#                   scenario per worker process; "pipelined": as "workers", but a
#                   scenario's gradient joins the solve queue the moment its T weekly
#                   duals are back, so gradients overlap the remaining solves),
#        DR_FI_CHUNK (weekly problems per worker call, default 4).
using Distributed, LinearAlgebra, Printf

const FI_NW = parse(Int, get(ENV, "DR_FI_WORKERS", "24"))
ENV["OPENBLAS_NUM_THREADS"] = "1"; ENV["OMP_NUM_THREADS"] = "1"
t_boot = time()
addprocs(FI_NW; exeflags = ["--project=$(Base.active_project())", "--startup-file=no", "-t", "1"])
@everywhere workers() include($(joinpath(SCRIPT_DIR, "hydro_fulliter_worker.jl")))
θ0, re0 = Flux.destructure(policy)
let p = policy
    @sync for w in workers()
        @async remotecall_fetch(q -> fw_set_policy!(q), w, p)
    end
end
# Warm every worker (compilation of rollout, stage solve and gradient), untimed.
let w = Float64.(mean_inflow(hydro_data, T)), x0 = Float32.(x0_init)
    xh = rollout_reachable_targets(policy, x0_init, w, T, nHyd)
    @sync for wk in workers()
        @async remotecall_fetch(wk) do
            fw_stage(0, 1, Float64.(x0), w[1:nHyd], xh[1:nHyd])
            fw_grad(θ0, x0, w, zeros(Float32, T * nHyd))
        end
    end
end
@printf("[fi] %d workers up and warm in %.1f s\n", FI_NW, time() - t_boot); flush(stdout)

"The clamped reservoir path (T+1 volumes per reservoir) the full model carries for `xhat`."
function fi_clamped(w, xhat)
    prepare_solve!(prob, x0_init, w, xhat)
    return reshape(copy(prob.strict_reservoir_values), nHyd, T + 1)
end

const FI_NET = get(ENV, "DR_FI_NET", "workers")
FI_NET in ("workers", "pipelined") || error("DR_FI_NET must be workers or pipelined, got $FI_NET")
const FI_CHUNK = parse(Int, get(ENV, "DR_FI_CHUNK", "4"))

"""
    fi_pipelined(θ, x0, ws, jobs) -> (res, gs, λs, ok, t_solve)

Weekly solves and per-scenario gradients on one shared worker pool. Jobs are taken
in order, `FI_CHUNK` at a time; when the last of a scenario's T weeks returns, its
λ_t = y_t − y_{t+1} is formed and its gradient is queued ahead of the remaining
solves, so the reverse passes overlap the solves of later scenarios. A worker
leaves only when every solve is done and no gradient is waiting or still to come.
`t_solve` is the time at which the last weekly solve returned.
"""
function fi_pipelined(θ, x0, ws, jobs)
    S = length(ws); nj = length(jobs); t0 = time()
    res = Vector{Any}(undef, nj); left = fill(T, S)
    bysid = [Int[] for _ in 1:S]
    for (k, j) in enumerate(jobs)
        push!(bysid[j[1]], k)
    end
    gq = Int[]; nxt = Ref(1); t_solve = Ref(0.0)
    gs = Vector{Any}(nothing, S); λs = Vector{Vector{Float64}}(undef, S); ok = trues(S)
    @sync for wk in workers()
        @async while true
            if !isempty(gq)
                sid = popfirst!(gq)
                gs[sid] = remotecall_fetch((a...) -> fw_grad(a...), wk, θ, x0, ws[sid], Float32.(λs[sid]))
            elseif nxt[] <= nj
                k0 = nxt[]; k1 = min(k0 + FI_CHUNK - 1, nj); nxt[] = k1 + 1
                rs = remotecall_fetch(js -> [fw_stage(j...) for j in js], wk, jobs[k0:k1])
                for (k, r) in zip(k0:k1, rs)
                    res[k] = r; left[r.sid] -= 1
                    left[r.sid] == 0 || continue
                    st = sort([res[i] for i in bysid[r.sid]]; by = q -> q.t)
                    ok[r.sid] = all(q.ok for q in st)
                    y = reduce(vcat, [q.y for q in st])
                    λ = copy(y); λ[1:((T - 1) * nHyd)] .-= y[(nHyd + 1):(T * nHyd)]    # λ_t = y_t − y_{t+1}
                    λs[r.sid] = λ
                    ok[r.sid] && push!(gq, r.sid)
                end
                nxt[] > nj && all(==(0), left) && t_solve[] == 0.0 && (t_solve[] = time() - t0)
            elseif all(==(0), left)
                break
            else
                sleep(0.0005)
            end
        end
    end
    return (res = res, gs = gs, λs = λs, ok = ok, t_solve = t_solve[])
end

"""
    fi_update(θ, ws) -> NamedTuple

One update of the scenarios `ws`: rollout, weekly solves and gradient on the workers.
"""
function fi_update(θ, ws)
    S = length(ws); x0 = Float32.(x0_init)
    t0 = time()
    xhats = pmap(w -> fw_rollout(θ, x0, w), ws)
    t_roll = time() - t0
    t1 = time()
    jobs = Any[]
    for (sid, w) in enumerate(ws)
        R = fi_clamped(w, xhats[sid])
        for t in 1:T
            push!(jobs, (sid, t, R[:, t], w[((t - 1) * nHyd + 1):(t * nHyd)], R[:, t + 1]))
        end
    end
    t_clamp = time() - t1
    if FI_NET == "pipelined"
        P = fi_pipelined(θ, x0, ws, jobs)
        good = findall(P.ok)
        g = sum(P.gs[good]) ./ length(good)
        res = P.res; t_solve = t_clamp + P.t_solve
        return (g = g, obj = [sum(r.obj for r in res if r.sid == sid) for sid in 1:S], λs = P.λs, ok = count(P.ok),
                rollout = t_roll, clamp = t_clamp, solve = t_solve, gradient = time() - t1 - t_solve,
                wall = time() - t0, stage_s = [r.s for r in res], iters = [r.iter for r in res],
                statuses = unique([r.status for r in res]))
    end
    res = pmap(j -> fw_stage(j...), jobs; batch_size = FI_CHUNK)
    t_solve = time() - t1
    t2 = time()
    λs = Vector{Vector{Float64}}(undef, S); ok = trues(S); obj = zeros(S)
    for sid in 1:S
        st = sort([r for r in res if r.sid == sid]; by = r -> r.t)
        ok[sid] = all(r.ok for r in st)
        obj[sid] = sum(r.obj for r in st)
        y = reduce(vcat, [r.y for r in st])
        λ = copy(y); λ[1:((T - 1) * nHyd)] .-= y[(nHyd + 1):(T * nHyd)]    # λ_t = y_t − y_{t+1}
        λs[sid] = λ
    end
    good = findall(ok)
    gs = pmap(sid -> fw_grad(θ, x0, ws[sid], Float32.(λs[sid])), good)
    g = sum(gs) ./ length(good)
    t_grad = time() - t2
    return (g = g, obj = obj, λs = λs, ok = count(ok), rollout = t_roll, clamp = t_clamp, solve = t_solve,
            gradient = t_grad, wall = time() - t0, stage_s = [r.s for r in res], iters = [r.iter for r in res],
            statuses = unique([r.status for r in res]))
end

"The actor gradient of one scenario on the master (the reference; same closure as `fw_grad`)."
function fi_grad_master(θ, x0, w, λ)
    Flux.gradient(θ) do th
        m = re0(th); Flux.reset!(m)
        nu = length(w) ÷ T; prev = x0; tot = zero(Float32)
        for t in 1:T
            xt = m(vcat(Float32.(view(w, ((t - 1) * nu + 1):(t * nu))), prev))
            tot += sum(view(λ, ((t - 1) * nHyd + 1):(t * nHyd)) .* xt)
            prev = xt
        end
        tot
    end[1]
end

"The trainer's path for the same scenarios: full DE per scenario, then one-process gradient."
function fi_reference(θ, ws)
    λs = Vector{Vector{Float64}}(); obj = Float64[]; t0 = time()
    for w in ws
        xh = rollout_reachable_targets(policy, x0_init, w, T, nHyd)
        ExaModels.set_parameter!(prob.core, prob.p_x0, x0_init)
        ExaModels.set_parameter!(prob.core, prob.p_inflow, w)
        ExaModels.set_parameter!(prob.core, prob.p_target, xh)
        prepare_solve!(prob, x0_init, w, xh)
        r = MadNLP.madnlp(prob.model; SOLVER_KWARGS...)
        push!(λs, Float64.(vec(Array(target_multipliers(prob, r))))); push!(obj, Float64(r.objective))
    end
    t_solve = time() - t0
    g = sum(fi_grad_master(θ, Float32.(x0_init), ws[i], Float32.(λs[i])) for i in eachindex(ws)) ./ length(ws)
    return (g = g, obj = obj, λs = λs, solve = t_solve)
end

let out = Dict{String,Any}("workers" => FI_NW, "net" => FI_NET, "chunk" => FI_CHUNK, "cpu" => Sys.cpu_info()[1].model, "by_traj" => Dict{String,Any}())
    nb = parse(Int, get(ENV, "DR_FI_BATCHES", "4"))
    # The trainer's per-update scenario sampler.
    for S in parse.(Int, split(get(ENV, "DR_FI_TRAJ", "16+24"), '+'))
        rows = Any[]
        for b in 0:nb
            Random.seed!(9000 + 100S + b)
            ws = [Float64.(sample_scenario(hydro_data, T)) for _ in 1:S]
            r = fi_update(θ0, ws)
            row = Dict{String,Any}("batch" => b, "wall_s" => r.wall, "rollout_s" => r.rollout, "clamp_s" => r.clamp,
                                   "solve_s" => r.solve, "gradient_s" => r.gradient, "ok" => r.ok, "S" => S,
                                   "stage_max_s" => maximum(r.stage_s), "stage_sum_s" => sum(r.stage_s),
                                   "iters_max" => maximum(r.iters), "statuses" => r.statuses)
            if b == 0
                q = fi_reference(θ0, ws)
                row["grad_cos"] = dot(r.g, q.g) / (norm(r.g) * norm(q.g))
                row["grad_relnorm_diff"] = norm(r.g .- q.g) / norm(q.g)
                row["obj_relgap_max"] = maximum(abs.(r.obj .- q.obj) ./ abs.(q.obj))
                row["lambda_relerr_max"] = maximum(norm(r.λs[i] .- q.λs[i]) / norm(q.λs[i]) for i in 1:S)
                row["reference_solve_s"] = q.solve
            end
            push!(rows, row)
            @printf("[fi] S=%2d batch %d  update %6.2f s = rollout %.2f + clamp %.2f + solves %.2f + gradient %.2f | %d/%d ok | stage max %.3f sum %.1f | iters max %d %s\n",
                    S, b, r.wall, r.rollout, r.clamp, r.solve - r.clamp, r.gradient, r.ok, S, maximum(r.stage_s),
                    sum(r.stage_s), maximum(r.iters),
                    haskey(row, "grad_cos") ? @sprintf("| check: grad cos %.8f rel %.1e obj gap %.1e λ relerr %.1e (full-DE reference %.1f s)",
                                                       row["grad_cos"], row["grad_relnorm_diff"], row["obj_relgap_max"],
                                                       row["lambda_relerr_max"], row["reference_solve_s"]) : "")
            flush(stdout)
            out["by_traj"][string(S)] = rows
            open(get(ENV, "DR_FI_OUT", joinpath(SCRIPT_DIR, "logs", "fulliter_cpu.json")), "w") do io
                JSON.print(io, out, 2)
            end
        end
    end
end
