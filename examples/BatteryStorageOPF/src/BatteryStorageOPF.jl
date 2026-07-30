# BatteryStorageOPF.jl
#
# Self-contained example module for a stochastic battery-storage AC-OPF problem
# built on any PGLib-OPF case, with a full TS-DDR path (policy → target-constrained
# ExaModels projection → envelope-theorem gradient) trained/evaluated on the true
# AC-polar model.
#
#   Phase 1: deterministic AC-polar foundation + reproducible PGLib battery-case
#            generator + manifest (network_data, battery_data, manifest,
#            battery_opf_exa, reference_powermodels).
#   Phase 2: seeded finite-support demand process + paired protocol, battery-SoC
#            reachable policy, target-constrained DE (strict/soft) with cost
#            decomposition and reporting/look-ahead horizons, CPU/GPU training and
#            rollout entrypoints, checkpointing, and the stochastic manifest
#            (demand_process, battery_tsddr, battery_policy, battery_training,
#            stochastic_manifest).
#
# Load it from the example environment (see README):
#
#   module load julia
#   julia --pkgimages=no --project=. -e 'include("src/BatteryStorageOPF.jl")'
#
# `--pkgimages=no` is required on this cluster (the system Julia cannot build the
# native precompile image for the `Pkg` stdlib, a MadNLP dependency).

module BatteryStorageOPF

using TOML

# Order matters: data layer → case construction → manifest → physical model →
# reference → demand process → target-constrained DE → policy → training →
# stochastic manifest. The Phase-2 files depend on the parent DecisionRulesExa
# and Flux, declared in this example's Project.toml (see [sources]).
include("network_data.jl")
include("battery_data.jl")
include("manifest.jl")
include("acp_core.jl")          # SHARED AC-polar equations (single source of truth)
include("battery_opf_exa.jl")
include("reference_powermodels.jl")
include("demand_process.jl")
include("battery_tsddr.jl")
include("battery_policy.jl")
include("battery_training.jl")
include("stochastic_manifest.jl")

# ── Public API ────────────────────────────────────────────────────────────────
# Network / case construction
export NetworkData, BusData, GenData, BranchData, LoadData
export nbus, ngen, nbranch, nload
export resolve_pglib_case, available_pglib_cases, load_pglib_network, parse_network
export BatteryData, BatteryCase, nbattery
export make_battery_case, validate_battery_case, eligible_load_bus_ids

# Phase-1 manifest
export battery_manifest, manifest_hash, canonical_content
export write_manifest, write_battery_file, reconstruct_case

# Phase-1 ExaModels physical model
export BatteryExaProblem, build_battery_de, solve_de!, solve_succeeded
export set_demand!, set_initial_soc!
export battery_solution, battery_balance_residuals
export simultaneous_charge_discharge_power, max_primal_residual

# Reference / parity
export reference_ac_opf, exa_base_objective, check_base_acp_parity

# Phase-2 demand process + paired protocol
export LoadAtom, LoadProcess, make_load_process, assign_regions
export default_base_shape, default_load_atoms
export n_uncertainty, natom
export scenario_index_matrix, materialize_scenario, materialize_all
export process_canonical_content, process_hash, index_matrix_hash
export DEMAND_PRESETS, DEFAULT_DEMAND_PRESET, load_process_from_fields
export demand_multiplier_summary
export write_scenario_protocol, reconstruct_scenario_protocol

# Shared AC-polar blocks (single source of truth)
export add_acp_variables!, add_battery_variables!, add_acp_network_constraints!
export add_nodal_balance!, add_battery_dynamics!
export add_generator_cost!, add_cycle_cost!
export acp_constraint_count, rated_branch_positions, validate_stage_hours

# Phase-2 operational (target-constrained) problem + cost decomposition
export BatteryTSDDRProblem, build_battery_tsddr_de, build_battery_stage_problem
export set_tsddr_uncertainty!, set_tsddr_initial_soc!, set_tsddr_targets!
export set_realized_demand!, DEFAULT_ACTIVE_RECOURSE_COST_PER_MWH, ACTIVE_RECOURSE_LB_TOL_PU
export build_targetless_diagnostic_de, is_targetless, variable_offsets, target_consistent_start!
export seed_start_from_solution!, reset_flat_start!, solve_stage_with_starts
export target_multipliers, tsddr_solution, decompose_costs
export tsddr_balance_residuals, tsddr_max_primal_residual

# Phase-2 reachable policy
export BatteryReachablePolicy, battery_reachable_policy
export battery_reachable_bounds, stretchedsigmoid, hardsigmoidsafe
export BOUNDED_TARGET_ACTIVATIONS, policy_initial_state, load_battery_policy!

# Phase-2 training / rollout / checkpoint / trajectory
export make_replay_sampler, train_battery_tsddr
export evaluate_battery_policy, evaluate_paired
export battery_checkpoint, save_checkpoint, load_checkpoint, write_trajectory

# Phase-2 stochastic manifest
export stochastic_manifest, write_stochastic_manifest, reconstruct_stochastic_manifest
export verify_protocol_file

end # module
