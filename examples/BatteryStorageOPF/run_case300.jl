#!/usr/bin/env julia
# run_case300.jl
#
# CPU smoke runner for the deterministic battery-storage AC-OPF foundation.
# Builds a reproducible battery case from a PGLib benchmark, writes its manifest
# + human-readable battery file, builds a short-horizon ExaModels AC-polar
# deterministic equivalent, solves it on the CPU with MadNLP, and reports the
# solver status, objective, residuals, chosen battery buses, and manifest hash.
#
# Run (from examples/BatteryStorageOPF):
#   module load julia
#   julia --pkgimages=no --project=. run_case300.jl
#
# Change the case / battery count / seed / horizon via environment variables:
#   BAT_CASE=case14_ieee BAT_NBAT=3 BAT_SEED=1 BAT_HORIZON=4 \
#     julia --pkgimages=no --project=. run_case300.jl

include(joinpath(@__DIR__, "src", "BatteryStorageOPF.jl"))
using .BatteryStorageOPF
using MadNLP
using Printf

const CASE    = get(ENV, "BAT_CASE", "case300_ieee")
const NBAT    = parse(Int, get(ENV, "BAT_NBAT", "20"))
const SEED    = parse(Int, get(ENV, "BAT_SEED", "20260722"))
const HORIZON = parse(Int, get(ENV, "BAT_HORIZON", "4"))
const OUTDIR  = get(ENV, "BAT_OUTDIR", joinpath(@__DIR__, "results"))

function main()
    mkpath(OUTDIR)
    @printf("Building battery case \"%s\" (batteries=%d, seed=%d)\n", CASE, NBAT, SEED)
    case = make_battery_case(CASE; number_of_batteries = NBAT, seed = SEED)
    nd = case.network

    @printf("  network: %d buses, %d gens, %d branches, %d loads, baseMVA=%.1f\n",
            nbus(nd), ngen(nd), nbranch(nd), nload(nd), nd.baseMVA)
    @printf("  total load = %.4f pu ; eligible load buses = %d\n",
            case.total_load_pu, length(case.eligible_bus_ids))
    @printf("  battery buses (stable order) = %s\n", string(case.selected_bus_ids))
    if !isempty(case.batteries)
        b = case.batteries[1]
        @printf("  per-battery: p̄=%.4f pu, e_max=%.4f pu·h, e_init=%.4f pu·h, η=%.2f/%.2f\n",
                b.p_charge_max, b.e_max, b.e_init, b.eta_ch, b.eta_dis)
    end

    hash = manifest_hash(case)
    man_path = joinpath(OUTDIR, "manifest_$(CASE)_seed$(SEED).json")
    bat_path = joinpath(OUTDIR, "batteries_$(CASE)_seed$(SEED).csv")
    write_battery_file(case, bat_path)
    write_manifest(case, man_path; extra_files = [bat_path])
    @printf("  manifest hash = %s\n", hash)
    @printf("  wrote %s\n         %s\n", man_path, bat_path)

    # Short horizon with a gentle time-of-day load shape (±3%) so the batteries
    # cycle (peak/off-peak arbitrage) — exercises the state equation — while
    # staying within the base case's feasible region.
    profile = HORIZON == 4 ? [1.00, 1.03, 1.00, 0.97] :
              [1.0 + 0.03 * sinpi(2 * (t - 1) / HORIZON) for t in 1:HORIZON]
    @printf("\nBuilding ExaModels AC-polar DE (T=%d, Δt=1.0 h) ...\n", HORIZON)
    prob = build_battery_de(case, HORIZON; stage_hours = 1.0, demand_profile = profile)

    @printf("Solving on CPU with MadNLP ...\n")
    t0 = time()
    result = solve_de!(prob; print_level = MadNLP.ERROR, tol = 1e-6, max_iter = 1000)
    dt = time() - t0

    sol = battery_solution(prob, result)
    prim = max_primal_residual(prob, result)
    balres = battery_balance_residuals(prob, sol)
    simult = simultaneous_charge_discharge_power(sol)

    println("\n── Results ─────────────────────────────────────────────")
    @printf("status              : %s\n", string(result.status))
    @printf("accepted            : %s\n", solve_succeeded(result.status))
    @printf("objective (USD)     : %.6f\n", result.objective)
    @printf("solve time (s)      : %.2f\n", dt)
    @printf("max primal residual : %.3e\n", prim)
    if !isempty(case.batteries)
        @printf("max |battery balance|: %.3e\n", maximum(abs, balres))
        @printf("max simultaneous charge/discharge power : %.3e pu\n", maximum(simult))
        @printf("SoC[:,1] (init)     : %s\n", string(round.(sol.soc[:, 1], digits = 4)))
        @printf("SoC[:,end] (final)  : %s\n", string(round.(sol.soc[:, end], digits = 4)))
        @printf("Σ discharge (pu)    : %.4f ; Σ charge (pu) : %.4f\n",
                sum(sol.p_dis), sum(sol.p_ch))
    end
    println("────────────────────────────────────────────────────────")

    solve_succeeded(result.status) || error("smoke solve did not reach an accepted status")
    return nothing
end

main()
