# setup_env.jl
#
# One-shot helper that resolves the BatteryStorageOPF example environment.
# It activates THIS directory and adds the pinned dependency set, letting Pkg
# fetch the correct UUIDs and a mutually compatible set of versions from the
# registry. Run once (or after deleting Manifest.toml):
#
#   module load julia
#   julia --project=examples/BatteryStorageOPF examples/BatteryStorageOPF/setup_env.jl
#
# The depot lives in /tmp per project policy; do not redirect it elsewhere.

import Pkg
Pkg.activate(@__DIR__)

# User-facing PGLib benchmark source + power-flow modeling stack, the ExaModels
# builder + its CPU NLP solver (MadNLP), the independent JuMP/Ipopt reference,
# the seeded RNG for reproducible placement, and JSON/SHA for the manifest.
Pkg.add([
    Pkg.PackageSpec(name = "PGLib"),
    Pkg.PackageSpec(name = "PowerModels"),
    Pkg.PackageSpec(name = "StableRNGs"),
    Pkg.PackageSpec(name = "ExaModels"),
    Pkg.PackageSpec(name = "MadNLP"),
    Pkg.PackageSpec(name = "NLPModels"),
    Pkg.PackageSpec(name = "JuMP"),
    Pkg.PackageSpec(name = "Ipopt"),
    Pkg.PackageSpec(name = "JSON"),
    # Standard libraries used directly by the example (must be explicit deps in
    # a project environment).
    Pkg.PackageSpec(name = "TOML"),
    Pkg.PackageSpec(name = "SHA"),
    Pkg.PackageSpec(name = "Random"),
    Pkg.PackageSpec(name = "LinearAlgebra"),
    Pkg.PackageSpec(name = "Test"),
    Pkg.PackageSpec(name = "Statistics"),
    Pkg.PackageSpec(name = "Printf"),
])

Pkg.precompile()

@info "BatteryStorageOPF environment resolved" project = Base.active_project()
