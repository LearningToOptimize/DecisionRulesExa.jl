# analyze_gradient_quality_results.jl
#
# Aggregate structured gradient-comparison shards. The unit of comparison is a
# paired scenario/method record: the method gradient projected on fixed random
# directions versus the rollout finite-difference projections.

using JLD2
using Statistics
using Random
using Printf
using Dates

const SCRIPT_DIR = dirname(@__FILE__)
const RESULT_DIR = get(ENV, "DR_RESULT_DIR", joinpath(SCRIPT_DIR, "results", "gradient_quality"))
const REPORT_PATH = get(
    ENV,
    "DR_ANALYSIS_OUT",
    "/storage/home/hcoda1/9/arosemberg3/scratch/DecisionRules.jl/plan/gradient_quality_result_analysis.md",
)
const BOOTSTRAPS = parse(Int, get(ENV, "DR_BOOTSTRAPS", "2000"))
const REPORT_DATE_FORMAT = dateformat"yyyy-mm-dd HH:MM:SS"

finite_values(xs) = collect(skipmissing([isfinite(x) ? x : missing for x in xs]))
mean_or_nan(xs) = isempty(xs) ? NaN : mean(xs)
median_or_nan(xs) = isempty(xs) ? NaN : median(xs)
quantile_or_nan(xs, p) = isempty(xs) ? NaN : quantile(xs, p)
frac_or_nan(xs, pred) = isempty(xs) ? NaN : count(pred, xs) / length(xs)

function ci(values; statistic = mean, B = BOOTSTRAPS, rng = MersenneTwister(20260625))
    vals = finite_values(values)
    n = length(vals)
    n == 0 && return (NaN, NaN)
    draws = Vector{Float64}(undef, B)
    for b in 1:B
        sample = vals[rand(rng, 1:n, n)]
        draws[b] = statistic(sample)
    end
    return (quantile(draws, 0.025), quantile(draws, 0.975))
end

function records_from_file(path)
    data = JLD2.load(path)
    method_names = Vector{String}(data["method_names"])
    scenarios = Vector{Int}(data["scenario_global_indices"])
    phase = String(data["PHASE_LABEL"])
    policy_path = String(data["POLICY_PATH"])

    rows = NamedTuple[]
    for (si, scenario) in enumerate(scenarios), (mi, method) in enumerate(method_names)
        push!(rows, (
            phase = phase,
            policy_path = policy_path,
            scenario = scenario,
            method = method,
            cos = Float64(data["all_cosines"][si, mi]),
            cos_flip = Float64(data["all_cosines_flip"][si, mi]),
            nrmse = Float64(data["all_nrmse"][si, mi]),
            scale_log10 = Float64(data["all_scale_log10_err"][si, mi]),
            sign = Float64(data["all_sign_agree"][si, mi]),
            sign_flip = Float64(data["all_sign_agree_flip"][si, mi]),
            mean_violation = Float64(data["all_mean_viols"][si, mi]),
            early_violation = Float64(data["all_early_viols"][si, mi]),
            mean_range_leak = Float64(data["all_mean_range_rel_leaks"][si, mi]),
            early_range_leak = Float64(data["all_early_range_rel_leaks"][si, mi]),
            penalty_cost = Float64(data["all_penalty_costs"][si, mi]),
            fd_time = Float64(data["all_fd_times"][si]),
            de_time = Float64(data["all_de_times"][si, mi]),
            source = basename(path),
        ))
    end
    return rows
end

function summarize_group(rows)
    cos = finite_values(getfield.(rows, :cos))
    sign = finite_values(getfield.(rows, :sign))
    nrmse = finite_values(getfield.(rows, :nrmse))
    scale = finite_values(getfield.(rows, :scale_log10))
    leak = finite_values(getfield.(rows, :early_range_leak))
    de_time = finite_values(getfield.(rows, :de_time))
    mean_ci = ci(cos; statistic = mean)
    p10_ci = ci(cos; statistic = x -> quantile(x, 0.10))
    return (
        n_total = length(rows),
        n_valid = length(cos),
        fail_n = length(rows) - length(cos),
        cos_mean = mean_or_nan(cos),
        cos_mean_ci = mean_ci,
        cos_median = median_or_nan(cos),
        cos_p10 = quantile_or_nan(cos, 0.10),
        cos_p10_ci = p10_ci,
        bad_frac = frac_or_nan(cos, <(0.0)),
        weak_frac = frac_or_nan(cos, <(0.10)),
        sign_mean = mean_or_nan(sign),
        sign_p10 = quantile_or_nan(sign, 0.10),
        nrmse_median = median_or_nan(nrmse),
        nrmse_p90 = quantile_or_nan(nrmse, 0.90),
        scale_abs_median = median_or_nan(abs.(scale)),
        scale_abs_p90 = quantile_or_nan(abs.(scale), 0.90),
        early_leak_mean = mean_or_nan(leak),
        de_time_mean = mean_or_nan(de_time),
    )
end

function grouped(rows, keyfn)
    groups = Dict{Any, Vector{NamedTuple}}()
    for row in rows
        push!(get!(groups, keyfn(row), NamedTuple[]), row)
    end
    return groups
end

function paired_deltas(rows; from_phase = "cold", to_phase_prefix = "warm")
    by_key = Dict{Tuple{String, Int, String}, NamedTuple}()
    for row in rows
        by_key[(row.phase, row.scenario, row.method)] = row
    end
    phases = sort(unique(getfield.(rows, :phase)))
    warm_phases = filter(p -> startswith(p, to_phase_prefix), phases)
    deltas = NamedTuple[]
    for warm_phase in warm_phases
        for row in rows
            row.phase == warm_phase || continue
            cold = get(by_key, (from_phase, row.scenario, row.method), nothing)
            cold === nothing && continue
            isfinite(row.cos) && isfinite(cold.cos) || continue
            push!(deltas, (
                phase_pair = "$warm_phase - $from_phase",
                scenario = row.scenario,
                method = row.method,
                delta_cos = row.cos - cold.cos,
                delta_nrmse = row.nrmse - cold.nrmse,
                delta_sign = row.sign - cold.sign,
                delta_early_leak = row.early_range_leak - cold.early_range_leak,
            ))
        end
    end
    return deltas
end

function fmt(x; digits = 4)
    isfinite(x) || return "NaN"
    return @sprintf("%.*f", digits, x)
end

function pct(x; digits = 1)
    isfinite(x) || return "NaN%"
    return @sprintf("%.*f%%", digits, 100x)
end

function write_report(path, rows, files)
    mkpath(dirname(path))
    phase_method = grouped(rows, r -> (r.phase, r.method))
    phases = sort(unique(getfield.(rows, :phase)))
    methods = sort(unique(getfield.(rows, :method)))

    open(path, "w") do io
        println(io, "# Gradient Quality Result Analysis")
        println(io)
        println(io, "Generated: $(Dates.format(now(), REPORT_DATE_FORMAT))")
        println(io)
        println(io, "Result directory: `$RESULT_DIR`")
        println(io, "Files read: $(length(files))")
        println(io, "Records: $(length(rows)) scenario-method rows")
        println(io)
        println(io, "## Interpretation")
        println(io)
        println(io, "Gradient quality here means agreement between each method's optimization-derived gradient and the rollout finite-difference gradient, projected on the same random directions. The main decision metrics are mean cosine for average behavior and p10/bad-fraction for the tail that can dominate convergence.")
        println(io)

        println(io, "## Phase And Method Summary")
        println(io)
        println(io, "| phase | method | valid/total | cos mean [95%] | cos median | cos p10 [95%] | bad cos | weak cos<0.10 | sign mean | sign p10 | nrmse med | nrmse p90 | |scale| p90 | early leak | mean DE sec |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for phase in phases, method in methods
            group = get(phase_method, (phase, method), NamedTuple[])
            isempty(group) && continue
            s = summarize_group(group)
            println(io, "| $phase | $method | $(s.n_valid)/$(s.n_total) | $(fmt(s.cos_mean)) [$(fmt(s.cos_mean_ci[1])), $(fmt(s.cos_mean_ci[2]))] | $(fmt(s.cos_median)) | $(fmt(s.cos_p10)) [$(fmt(s.cos_p10_ci[1])), $(fmt(s.cos_p10_ci[2]))] | $(pct(s.bad_frac)) | $(pct(s.weak_frac)) | $(pct(s.sign_mean)) | $(pct(s.sign_p10)) | $(fmt(s.nrmse_median)) | $(fmt(s.nrmse_p90)) | $(fmt(s.scale_abs_p90)) | $(fmt(s.early_leak_mean)) | $(fmt(s.de_time_mean, digits=1)) |")
        end

        deltas = paired_deltas(rows)
        if !isempty(deltas)
            println(io)
            println(io, "## Paired Warm Minus Cold")
            println(io)
            println(io, "| phase pair | method | n | Δcos mean | Δcos median | Δcos p10 | Δnrmse median | Δsign mean | Δearly leak mean |")
            println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            delta_groups = grouped(deltas, r -> (r.phase_pair, r.method))
            for key in sort(collect(keys(delta_groups)); by = string)
                group = delta_groups[key]
                dc = finite_values(getfield.(group, :delta_cos))
                dn = finite_values(getfield.(group, :delta_nrmse))
                ds = finite_values(getfield.(group, :delta_sign))
                dl = finite_values(getfield.(group, :delta_early_leak))
                println(io, "| $(key[1]) | $(key[2]) | $(length(dc)) | $(fmt(mean_or_nan(dc))) | $(fmt(median_or_nan(dc))) | $(fmt(quantile_or_nan(dc, 0.10))) | $(fmt(median_or_nan(dn))) | $(pct(mean_or_nan(ds))) | $(fmt(mean_or_nan(dl))) |")
            end
        end

        println(io)
        println(io, "## Warm-Phase Ranking")
        println(io)
        warm_rows = filter(r -> startswith(r.phase, "warm"), rows)
        if isempty(warm_rows)
            println(io, "No warm-phase records found yet.")
        else
            warm_groups = grouped(warm_rows, r -> r.method)
            ranked = sort(collect(warm_groups); by = kv -> summarize_group(kv[2]).cos_mean, rev = true)
            println(io, "| rank | method | cos mean | cos p10 | bad cos | nrmse p90 | |scale| p90 |")
            println(io, "|---:|---:|---:|---:|---:|---:|---:|")
            for (rank, (method, group)) in enumerate(ranked)
                s = summarize_group(group)
                println(io, "| $rank | $method | $(fmt(s.cos_mean)) | $(fmt(s.cos_p10)) | $(pct(s.bad_frac)) | $(fmt(s.nrmse_p90)) | $(fmt(s.scale_abs_p90)) |")
            end
        end
    end
end

files = isdir(RESULT_DIR) ? sort(filter(f -> endswith(f, ".jld2"), joinpath.(RESULT_DIR, readdir(RESULT_DIR)))) : String[]
if isempty(files)
    @warn "No result files found" RESULT_DIR
else
    rows = reduce(vcat, records_from_file.(files); init = NamedTuple[])
    write_report(REPORT_PATH, rows, files)
    @info "Wrote report" REPORT_PATH n_files=length(files) n_rows=length(rows)
end
