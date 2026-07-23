# runtests.jl — example-local tests for the Phase-1 battery-storage foundation.
#
# Run (from examples/BatteryStorageOPF):
#   module load julia
#   julia --pkgimages=no --project=. test/runtests.jl
#
# Fast checks (construction / manifest / mapping / validation) use a small PGLib
# case; solve-based checks use case14_ieee, with a construction+residual smoke
# and placement/count assertions on case300_ieee.

include(joinpath(@__DIR__, "..", "src", "BatteryStorageOPF.jl"))
using .BatteryStorageOPF
using Test
using MadNLP
using JSON

const SMALL = "case14_ieee"      # small solve case
const BIG   = "case300_ieee"     # headline case

@testset "BatteryStorageOPF Phase 1" begin

    # ── Reproducible placement + manifest hash ───────────────────────────────
    @testset "identical seed ⇒ identical placement and hash" begin
        a = make_battery_case(SMALL; number_of_batteries = 3, seed = 123)
        b = make_battery_case(SMALL; number_of_batteries = 3, seed = 123)
        @test a.selected_bus_ids == b.selected_bus_ids
        @test manifest_hash(a) == manifest_hash(b)
        # Order is stable and consistent with the battery vector.
        @test [bat.bus_id for bat in a.batteries] == a.selected_bus_ids
    end

    @testset "different seed ⇒ different placement (when possible)" begin
        # 3 batteries drawn from >3 eligible buses: distinct seeds should give a
        # different ordered placement (and hence a different hash).
        c1 = make_battery_case(SMALL; number_of_batteries = 3, seed = 1)
        c2 = make_battery_case(SMALL; number_of_batteries = 3, seed = 2)
        @test length(c1.eligible_bus_ids) > 3
        @test c1.selected_bus_ids != c2.selected_bus_ids
        @test manifest_hash(c1) != manifest_hash(c2)
    end

    # ── Input validation ─────────────────────────────────────────────────────
    @testset "invalid inputs fail with clear errors" begin
        @test_throws ErrorException make_battery_case("no_such_case_xyz")
        @test_throws ErrorException make_battery_case(SMALL; number_of_batteries = 2,
                                                      charge_efficiency = 1.5)
        @test_throws ErrorException make_battery_case(SMALL; number_of_batteries = 2,
                                                      discharge_efficiency = 0.0)
        @test_throws ErrorException make_battery_case(SMALL; number_of_batteries = 2,
                                                      duration_hours = -1.0)
        @test_throws ErrorException make_battery_case(SMALL; number_of_batteries = 2,
                                                      initial_soc = 1.5)
        @test_throws ErrorException make_battery_case(SMALL; number_of_batteries = 2,
                                                      e_min_fraction = 0.6, initial_soc = 0.5)
        @test_throws ErrorException make_battery_case(SMALL; number_of_batteries = 2,
                                                      self_discharge_rate = 1.2)
        # Too many batteries for the eligible pool.
        @test_throws ErrorException make_battery_case(SMALL; number_of_batteries = 10_000)
        # Explicit bad bus.
        @test_throws ErrorException make_battery_case(SMALL; number_of_batteries = 1,
                                                      buses = [-999])
        # Explicit duplicate buses.
        elig = eligible_load_bus_ids(make_battery_case(SMALL; number_of_batteries = 0).network)
        @test_throws ErrorException make_battery_case(SMALL; number_of_batteries = 2,
                                                      buses = [elig[1], elig[1]])
    end

    # ── Non-consecutive identifier mapping (synthetic, exact) ────────────────
    @testset "arbitrary / non-consecutive id mapping" begin
        # Hand-built PowerModels-style dict with deliberately non-consecutive,
        # out-of-order bus ids and an isolated (type-4) bus that must be dropped.
        data = Dict{String,Any}(
            "baseMVA" => 100.0, "per_unit" => false,
            "bus" => Dict(
                "1" => Dict("index" => 100, "bus_type" => 3, "vmin" => 0.9, "vmax" => 1.1),
                "2" => Dict("index" => 5,   "bus_type" => 1, "vmin" => 0.9, "vmax" => 1.1),
                "3" => Dict("index" => 42,  "bus_type" => 1, "vmin" => 0.9, "vmax" => 1.1),
                "4" => Dict("index" => 7,   "bus_type" => 4, "vmin" => 0.9, "vmax" => 1.1),
            ),
            "gen" => Dict("1" => Dict("index" => 9, "gen_bus" => 100, "pmax" => 500.0,
                                      "pmin" => 0.0, "model" => 2, "ncost" => 3,
                                      "cost" => [0.1, 20.0, 5.0])),
            "branch" => Dict(
                "1" => Dict("index" => 3, "f_bus" => 100, "t_bus" => 5, "br_x" => 0.1,
                            "br_r" => 0.01, "rate_a" => 3.0),
                "2" => Dict("index" => 1, "f_bus" => 5, "t_bus" => 42, "br_x" => 0.2,
                            "br_r" => 0.02, "rate_a" => 0.0)),  # 0 ⇒ unlimited
            "load" => Dict(
                "1" => Dict("index" => 2, "load_bus" => 5,  "pd" => 100.0, "qd" => 20.0),
                "2" => Dict("index" => 1, "load_bus" => 42, "pd" => 50.0,  "qd" => 10.0)),
        )
        nd = parse_network(data, "synthetic")
        # Buses sorted by original id; type-4 dropped.
        @test [b.id for b in nd.buses] == [5, 42, 100]
        @test nbus(nd) == 3
        @test nd.bus_id_to_pos[100] == 3 && nd.bus_id_to_pos[5] == 1
        # Gen/branch/load references resolved to positions, ids preserved.
        # Components are stored sorted by original id, so look them up by id map
        # (never by insertion order).
        @test nd.gens[nd.gen_id_to_pos[9]].bus_pos == nd.bus_id_to_pos[100]
        br3 = nd.branches[nd.branch_id_to_pos[3]]   # branch id 3: bus 100 → bus 5
        @test br3.f_pos == nd.bus_id_to_pos[100]
        @test br3.t_pos == nd.bus_id_to_pos[5]
        @test isfinite(br3.rate_a)                  # rate_a = 3 ⇒ limited
        br1 = nd.branches[nd.branch_id_to_pos[1]]   # branch id 1: rate_a = 0
        @test isinf(br1.rate_a)                     # 0 rate ⇒ unlimited
        # Per-unit conversion (÷ baseMVA) applied to loads and gen limits.
        @test nd.bus_pd[nd.bus_id_to_pos[5]] ≈ 1.0
        @test nd.gens[1].pmax ≈ 5.0
        # Cost converted to pu: c1_pu = 20 · 100, c2_pu = 0.1 · 100².
        @test nd.gens[1].cost1 ≈ 2000.0
        @test nd.gens[1].cost2 ≈ 1000.0
        @test 100 in nd.ref_bus_positions .|> (p -> nd.buses[p].id)
    end

    # ── case300 structure + placement ────────────────────────────────────────
    @testset "case300 has 300 buses and 20 valid battery buses" begin
        case = make_battery_case(BIG; number_of_batteries = 20, seed = 20260722)
        @test nbus(case.network) == 300
        @test length(case.batteries) == 20
        @test length(unique(case.selected_bus_ids)) == 20
        # Every battery bus is a valid, in-service load bus.
        elig = Set(eligible_load_bus_ids(case.network))
        @test all(b -> b.bus_id in elig, case.batteries)
        @test all(b -> haskey(case.network.bus_id_to_pos, b.bus_id), case.batteries)
        # case300 genuinely has non-consecutive bus ids (position ≠ id somewhere).
        @test any(i -> case.network.buses[i].id != i, 1:nbus(case.network))
        validate_battery_case(case)  # must not throw
    end

    # ── Manifest round-trip / verified reconstruction / provenance ───────────
    @testset "manifest write, verified reconstruction, and tampering" begin
        case = make_battery_case(SMALL; number_of_batteries = 3, seed = 77)
        dir = mktempdir()
        man = joinpath(dir, "manifest.json")
        bat = joinpath(dir, "batteries.csv")
        write_battery_file(case, bat)
        write_manifest(case, man; extra_files = [bat])
        @test isfile(man) && isfile(bat)

        # Provenance fields are present and explicit.
        m = JSON.parsefile(man)
        @test haskey(m["pglib"], "matpower_file")
        @test length(String(m["pglib"]["matpower_sha256"])) == 64
        @test m["pglib"]["license"] == "Creative Commons Attribution 4.0 International"
        @test m["pglib"]["license_url"] == "https://creativecommons.org/licenses/by/4.0/"
        @test haskey(m["pglib"], "upstream_release")
        @test haskey(m["versions"], "PGLib") && haskey(m["versions"], "PowerModels") &&
              haskey(m["versions"], "julia")
        # The MATPOWER source hash participates in the scientific content hash.
        @test occursin(String(m["pglib"]["matpower_sha256"]), canonical_content(case))

        # Normal reconstruction succeeds and verifies source bytes + placement
        # + battery parameters.
        rc = reconstruct_case(man)
        @test rc.selected_bus_ids == case.selected_bus_ids
        @test manifest_hash(rc) == manifest_hash(case)

        # Identical inputs ⇒ identical placement and manifest hash.
        again = make_battery_case(SMALL; number_of_batteries = 3, seed = 77)
        @test again.selected_bus_ids == case.selected_bus_ids
        @test manifest_hash(again) == manifest_hash(case)

        # Tampering with the recorded source-file hash must fail reconstruction.
        m["pglib"]["matpower_sha256"] = repeat("0", 64)
        tampered = joinpath(dir, "tampered.json")
        open(io -> JSON.print(io, m, 2), tampered, "w")
        @test_throws ErrorException reconstruct_case(tampered)
    end

    # ── Solve: status, residuals, no simultaneous charge/discharge ───────────
    @testset "case14 solve: status + residuals + battery operation" begin
        case = make_battery_case(SMALL; number_of_batteries = 3, seed = 5,
                                 fleet_power_fraction = 0.3, duration_hours = 4.0)
        prob = build_battery_de(case, 3; stage_hours = 1.0,
                                demand_profile = [0.9, 1.1, 1.0])
        result = solve_de!(prob; print_level = MadNLP.ERROR, tol = 1e-8)
        @test solve_succeeded(result.status)
        sol = battery_solution(prob, result)
        @test max_primal_residual(prob, result) < 1e-5
        @test maximum(abs, battery_balance_residuals(prob, sol)) < 1e-6
        # Cycle cost ⇒ no material simultaneous charge & discharge.
        @test maximum(simultaneous_charge_discharge_power(sol)) < 1e-5
        # SoC stays within bounds.
        @test all(case.batteries[1].e_min - 1e-6 .<= sol.soc .<= case.batteries[1].e_max + 1e-6)
    end

    # ── Base-ACP parity (zero batteries and zero power) ──────────────────────
    @testset "zero-battery / zero-power base-ACP parity" begin
        # Small case: strict agreement (measured ≈1e-10 relative).
        p0 = check_base_acp_parity(SMALL; rtol = 1e-6)
        @test p0.within
        # Headline case300: same check, conservative tolerance (measured ≈1e-12).
        pbig = check_base_acp_parity(BIG; rtol = 1e-4)
        @test pbig.within
        # Zero battery POWER (fleet_power_fraction = 0) with 3 placed batteries
        # must also reproduce the base objective.
        ref = reference_ac_opf(SMALL).objective
        case0 = make_battery_case(SMALL; number_of_batteries = 3, seed = 9,
                                  fleet_power_fraction = 0.0)
        prob0 = build_battery_de(case0, 1)
        r0 = solve_de!(prob0; print_level = MadNLP.ERROR, tol = 1e-8)
        @test solve_succeeded(r0.status)
        @test abs(r0.objective - ref) / abs(ref) < 1e-5
    end

    # ── case300 construction + residual smoke ────────────────────────────────
    @testset "case300 construction + residual smoke" begin
        case = make_battery_case(BIG; number_of_batteries = 20, seed = 20260722)
        prob = build_battery_de(case, 2; stage_hours = 1.0,
                                demand_profile = [1.0, 1.02])
        result = solve_de!(prob; print_level = MadNLP.ERROR, tol = 1e-6, max_iter = 1000)
        @test solve_succeeded(result.status)
        sol = battery_solution(prob, result)
        @test max_primal_residual(prob, result) < 1e-4
        @test maximum(abs, battery_balance_residuals(prob, sol)) < 1e-6
        @test maximum(simultaneous_charge_discharge_power(sol)) < 1e-4
    end

    # ── Stage-duration (Δt) generator-cost scaling ───────────────────────────
    # PGLib polynomial costs are USD/hour, so a Δt-hour stage costs Δt× the
    # one-hour dispatch. These regressions fail under an implementation that
    # omits Δt from the generator cost.
    @testset "stage-duration (Δt) cost scaling" begin
        ref1h = reference_ac_opf(SMALL).objective        # independent 1-hour ACP
        case0 = make_battery_case(SMALL; number_of_batteries = 0)
        madnlp = (print_level = MadNLP.ERROR, tol = 1e-8)

        # Δt = 1 (retained parity): single stage equals the one-hour reference.
        p1 = build_battery_de(case0, 1; stage_hours = 1.0)
        r1 = solve_de!(p1; madnlp...)
        @test solve_succeeded(r1.status)
        @test abs(r1.objective - ref1h) / abs(ref1h) < 1e-5

        # T=1, Δt=2: objective is ≈ 2× the one-hour reference.
        p2 = build_battery_de(case0, 1; stage_hours = 2.0)
        r2 = solve_de!(p2; madnlp...)
        @test solve_succeeded(r2.status)
        @test abs(r2.objective - 2 * ref1h) / abs(2 * ref1h) < 1e-5

        # T=2, Δt=0.5, flat demand: two half-hour stages sum to the one-hour ref.
        p3 = build_battery_de(case0, 2; stage_hours = 0.5, demand_profile = [1.0, 1.0])
        r3 = solve_de!(p3; madnlp...)
        @test solve_succeeded(r3.status)
        @test abs(r3.objective - ref1h) / abs(ref1h) < 1e-5

        # Nonpositive / nonfinite stage_hours are rejected.
        @test_throws ErrorException build_battery_de(case0, 1; stage_hours = 0.0)
        @test_throws ErrorException build_battery_de(case0, 1; stage_hours = -1.0)
        @test_throws ErrorException build_battery_de(case0, 1; stage_hours = Inf)
        @test_throws ErrorException build_battery_de(case0, 1; stage_hours = NaN)
    end

    # ── Per-battery cycle-cost coefficients (not the first battery's) ─────────
    @testset "per-battery cycle-cost coefficients" begin
        # Two batteries whose cycle price we can distinguish: build with distinct
        # per-battery cycle costs by editing the case's battery vector.
        case = make_battery_case(SMALL; number_of_batteries = 2, seed = 3)
        b1, b2 = case.batteries
        case.batteries[1] = BatteryData(b1.id, b1.bus_id, b1.bus_pos,
            b1.p_charge_max, b1.p_discharge_max, b1.e_min, b1.e_max, b1.e_init,
            b1.eta_ch, b1.eta_dis, b1.sigma, 1.0)   # $1/MWh
        case.batteries[2] = BatteryData(b2.id, b2.bus_id, b2.bus_pos,
            b2.p_charge_max, b2.p_discharge_max, b2.e_min, b2.e_max, b2.e_init,
            b2.eta_ch, b2.eta_dis, b2.sigma, 7.0)   # $7/MWh
        prob = build_battery_de(case, 2; stage_hours = 1.5)
        # Coefficient k = cycle_cost_per_mwh_k · baseMVA · Δt, per battery.
        @test prob.cycle_coeffs[1] ≈ 1.0 * case.network.baseMVA * 1.5
        @test prob.cycle_coeffs[2] ≈ 7.0 * case.network.baseMVA * 1.5
        @test prob.cycle_coeffs[1] != prob.cycle_coeffs[2]
    end
end
