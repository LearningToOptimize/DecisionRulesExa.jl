# BatteryStorageOPF.jl
#
# Self-contained example module: a deterministic ExaModels AC-OPF foundation for
# battery-storage dispatch, plus a reproducible PGLib battery-case generator.
# (Phase 1 of BATTERY_STORAGE_OPF_PLAN.md; no TS-DDR / SDDP / uncertainty here.)
#
# Load it from the example environment:
#
#   module load julia
#   julia --pkgimages=no --project=. -e 'include("src/BatteryStorageOPF.jl")'
#
# `--pkgimages=no` is required on this cluster: the system Julia cannot build
# the native precompile image for the `Pkg` stdlib (a MadNLP dependency), so the
# example is run with native pkgimages disabled. Everything else is standard.

module BatteryStorageOPF

using TOML

# Order matters: data layer → case construction → manifest → model → reference.
include("network_data.jl")
include("battery_data.jl")
include("manifest.jl")
include("battery_opf_exa.jl")
include("reference_powermodels.jl")

# ── Public API ────────────────────────────────────────────────────────────────
# Network / case construction
export NetworkData, BusData, GenData, BranchData, LoadData
export nbus, ngen, nbranch, nload
export resolve_pglib_case, available_pglib_cases, load_pglib_network, parse_network
export BatteryData, BatteryCase, nbattery
export make_battery_case, validate_battery_case, eligible_load_bus_ids

# Manifest
export battery_manifest, manifest_hash, canonical_content
export write_manifest, write_battery_file, reconstruct_case

# ExaModels model
export BatteryExaProblem, build_battery_de, solve_de!, solve_succeeded
export set_demand!, set_initial_soc!
export battery_solution, battery_balance_residuals
export simultaneous_charge_discharge_power, max_primal_residual

# Reference / parity
export reference_ac_opf, exa_base_objective, check_base_acp_parity

end # module
