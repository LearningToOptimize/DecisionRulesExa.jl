# test_e2e_training.jl — tiny, fixed-seed, end-to-end battery AC-OPF TS-DDR
# training test in the PRIMARY (strict) target mode.
#
# It must:
#   * train from a fresh fixed initialization in STRICT mode (hard ê=e, no target
#     slack, no target penalty; the strict equality multipliers drive training);
#   * complete with accepted solver statuses (failed solves counted, not hidden) —
#     the two-sided active nodal recourse gives strict complete recourse so every
#     initial, training, and held-out solve succeeds;
#   * carry zero active recourse in BOTH directions on the held-out paths
#     (feasible demand): deficit d⁺ and surplus d⁻ each below tolerance, and their
#     sum below twice that (asserted separately — a single "zero deficit" check is
#     insufficient because it inspects only d⁺);
#   * reduce the fixed held-out mean PHYSICAL OPERATING COST from initialization
#     (physical = generator + battery throughput + two-sided active-recourse cost;
#     the target has no penalty in strict mode);
#   * save and reload a checkpoint reproducing the policy output AND both recourse
#     directions exactly.
#
# Run (from examples/BatteryStorageOPF):
#   julia --pkgimages=no --project=. test/test_e2e_training.jl

include(joinpath(@__DIR__, "..", "src", "BatteryStorageOPF.jl"))
using .BatteryStorageOPF
using Test
using MadNLP
using Flux
using Random
using DecisionRulesExa

# Declared numerical tolerance for "zero active recourse" (interior-point barrier
# tolerance leaves each recourse power near, not exactly, zero). Deficit d⁺ and
# surplus d⁻ are each held below this; their sum below 2×.
const RECOURSE_ENERGY_TOL_MWH = 1e-3

@testset "tiny end-to-end battery TS-DDR training (strict)" begin
    case = make_battery_case("case14_ieee"; number_of_batteries = 2, seed = 20260722,
                             fleet_power_fraction = 0.4, duration_hours = 4.0)
    # Mild, demand-feasible process (peak ≈1.1×) so an accepted run needs no recourse.
    # (The public default preset is selected by the demand calibration gates.)
    mild_atoms = [LoadAtom(1.00, [1.00, 1.00]),
                  LoadAtom(1.03, [1.05, 0.99]),
                  LoadAtom(1.03, [0.99, 1.05])]
    process = make_load_process(case; nregion = 2, period = 4, base_amplitude = 0.04,
                                atoms = mild_atoms, probs = [0.5, 0.25, 0.25],
                                train_seed = 4242, eval_seed = 9999)
    REPORT, LOOKAH = 3, 1
    T = REPORT + LOOKAH
    MODE = :strict                     # PRIMARY mode

    train_mat = scenario_index_matrix(process, T, 8; seed = process.train_seed)
    eval_mat  = scenario_index_matrix(process, T, 8; seed = process.eval_seed)
    @test train_mat != eval_mat

    de    = build_battery_tsddr_de(case, process; reporting_horizon = REPORT,
                                   lookahead = LOOKAH, mode = MODE, stage_hours = 1.0)
    stage = build_battery_stage_problem(case, process; mode = MODE, stage_hours = 1.0)
    @test de.mode === :strict && stage.mode === :strict
    @test !isempty(de.target_con_range)                # strict target rows exist
    # Strict carries NO target slack variables (soft would add 2*T*nBat): it has
    # the SAME variables as the targetless diagnostic, plus T*nBat target rows.
    tldiag = build_targetless_diagnostic_de(case, process; reporting_horizon = REPORT,
                                            lookahead = LOOKAH, stage_hours = 1.0)
    @test de.model.meta.nvar == tldiag.model.meta.nvar
    @test de.model.meta.ncon == tldiag.model.meta.ncon + de.horizon * de.nBat

    Random.seed!(20260722)             # fresh, fixed initialization
    policy = battery_reachable_policy(case, process; dt = 1.0, layers = [32, 32],
                                      combiner_layers = [32])

    fixed_in = vcat(Float32.([0.7, 1.05, 0.95]), Float32.([2.0, 3.0]))

    ev0 = evaluate_paired(policy, stage, process, eval_mat; reporting_horizon = REPORT)
    @test ev0.n_ok == size(eval_mat, 2)          # all held-out solves accepted
    # Zero active recourse at init — deficit d⁺ and surplus d⁻ asserted SEPARATELY,
    # and their sum below 2×. A single d⁺-only check would be insufficient.
    @test 0.0 <= ev0.total_active_deficit_energy_mwh < RECOURSE_ENERGY_TOL_MWH
    @test 0.0 <= ev0.total_active_surplus_energy_mwh < RECOURSE_ENERGY_TOL_MWH
    @test 0.0 <= ev0.total_active_recourse_energy_mwh < 2 * RECOURSE_ENERGY_TOL_MWH
    @test ev0.maximum_active_recourse_lower_bound_violation_pu <= ACTIVE_RECOURSE_LB_TOL_PU
    init_cost = ev0.mean_reporting_physical_cost
    @test isfinite(init_cost)

    tr = train_battery_tsddr(policy, de, process, train_mat;
                             num_batches = 60, num_train_per_batch = 8,
                             optimizer = Flux.Adam(1f-2),
                             madnlp_kwargs = (print_level = MadNLP.ERROR, tol = 1e-6),
                             record_loss = (i, m, l, t) -> false)
    @test tr.n_ok == tr.n_total                  # every training solve accepted
    @test tr.n_failed == 0

    # Strict equality multipliers are what training consumes: on a solved stage
    # they must be finite and their count equals the target block.
    let w1 = materialize_scenario(process, [eval_mat[1, 1]]; horizon = 1),
        e1 = policy_initial_state(case; float_type = Float64)
        Flux.reset!(policy)
        t1 = Float64.(policy(vcat(Float32.(w1), Float32.(e1))))
        set_tsddr_initial_soc!(stage, e1); set_tsddr_uncertainty!(stage, w1); set_tsddr_targets!(stage, t1)
        r1 = MadNLP.madnlp(stage.model; print_level = MadNLP.ERROR, tol = 1e-6)
        @test DecisionRulesExa.solve_succeeded(r1)
        λ = target_multipliers(stage, r1)
        @test length(λ) == stage.nBat && all(isfinite, λ)
        # Strict mode carries NO target slack and NO target penalty. Public recourse
        # is projected (≥ 0); the RAW diagnostics reproduce the solver objective.
        d1 = decompose_costs(stage, r1)
        s1 = tsddr_solution(stage, r1)
        @test d1.target_penalty == 0.0 && d1.target_violation == 0.0
        @test all(iszero, s1.target_slack_pos) && all(iszero, s1.target_slack_neg)
        @test d1.active_recourse_cost >= 0.0 && d1.active_deficit_energy_mwh >= 0.0 &&
              d1.active_surplus_energy_mwh >= 0.0
        @test d1.maximum_active_recourse_lower_bound_violation_pu <= ACTIVE_RECOURSE_LB_TOL_PU
        @test isapprox(d1.raw_total_check, r1.objective; rtol = 1e-6)
        @test d1.solver_objective_recompute_residual < 1e-6
        @test isapprox(d1.total_check, d1.physical_operating_cost + d1.target_penalty; rtol = 1e-12)
    end

    ev1 = evaluate_paired(policy, stage, process, eval_mat; reporting_horizon = REPORT)
    @test ev1.n_ok == size(eval_mat, 2)
    # Still zero recourse in BOTH directions after training (asserted separately).
    # Public recourse is projected (≥ 0): assert nonnegative-and-below-tolerance
    # directly, never abs() on a quantity that must not be negative.
    @test 0.0 <= ev1.total_active_deficit_energy_mwh < RECOURSE_ENERGY_TOL_MWH
    @test 0.0 <= ev1.total_active_surplus_energy_mwh < RECOURSE_ENERGY_TOL_MWH
    @test 0.0 <= ev1.total_active_recourse_energy_mwh < 2 * RECOURSE_ENERGY_TOL_MWH
    @test ev1.maximum_active_recourse_lower_bound_violation_pu <= ACTIVE_RECOURSE_LB_TOL_PU
    final_cost = ev1.mean_reporting_physical_cost
    @info "tiny e2e (strict)" init_cost final_cost improvement = init_cost - final_cost
    @info "tiny e2e recourse (MWh)" deficit = ev1.total_active_deficit_energy_mwh surplus = ev1.total_active_surplus_energy_mwh
    @test final_cost < init_cost                 # reduced held-out PHYSICAL cost

    # Checkpoint save + exact reload reproduction on a fixed CPU input.
    Flux.reset!(policy); y_before = copy(policy(fixed_in))
    dir = mktempdir(); ckpt = joinpath(dir, "e2e.jls")
    save_checkpoint(ckpt, policy, de; case = case, process = process)
    pol2, meta = load_checkpoint(ckpt, case, process)
    Flux.reset!(pol2); y_after = copy(pol2(fixed_in))
    @test y_after == y_before
    @test meta["hashes"]["case_manifest_content_hash"] == manifest_hash(case)
    @test meta["architecture"]["activation"] == string(stretchedsigmoid)
    @test meta["active_recourse_cost_per_mwh"] == DEFAULT_ACTIVE_RECOURSE_COST_PER_MWH

    # The reloaded policy reproduces BOTH recourse directions exactly on the fixed
    # held-out paths (deterministic CPU rollout ⇒ identical deficit and surplus).
    ev1r = evaluate_paired(pol2, stage, process, eval_mat; reporting_horizon = REPORT)
    @test isapprox(ev1r.total_active_deficit_energy_mwh, ev1.total_active_deficit_energy_mwh;
                   rtol = 1e-8, atol = 1e-12)
    @test isapprox(ev1r.total_active_surplus_energy_mwh, ev1.total_active_surplus_energy_mwh;
                   rtol = 1e-8, atol = 1e-12)
    @test isapprox(ev1r.mean_reporting_physical_cost, final_cost; rtol = 1e-8)
end
