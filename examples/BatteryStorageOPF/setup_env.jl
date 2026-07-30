# setup_env.jl
#
# One-shot helper that resolves the BatteryStorageOPF example environment. It
# activates THIS directory and instantiates the dependency set declared in
# Project.toml — including the parent DecisionRulesExa package, brought in through
# the relative `[sources]` path — letting Pkg fetch a mutually compatible set of
# versions and precompile them. Run once (or after deleting Manifest.toml):
#
#   module load julia            # Julia 1.12.x (matches the parent package)
#   julia --pkgimages=no --project=examples/BatteryStorageOPF \
#         examples/BatteryStorageOPF/setup_env.jl
#
# The depot lives in /tmp per project policy; do not redirect it elsewhere. On a
# fresh node-local depot the General registry is installed first.

import Pkg
Pkg.activate(@__DIR__)

# A fresh /tmp depot has no registry; install General before resolving.
try
    Pkg.Registry.add("General")
catch err
    @info "General registry already present or add skipped" err
end

# Resolve + install + precompile everything declared in Project.toml (deps +
# [sources] DecisionRulesExa). This pulls the TS-DDR stack (Flux, Zygote, CUDA,
# MadNLP/MadNLPGPU) plus the PGLib/PowerModels/ExaModels modeling stack.
Pkg.resolve()
Pkg.instantiate()
Pkg.precompile()

@info "BatteryStorageOPF environment resolved" project = Base.active_project()
