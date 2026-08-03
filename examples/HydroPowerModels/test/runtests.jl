#!/usr/bin/env julia

# Regression suite for the Bolivia hydro example (ExaModels engine).
#
#   case       the frozen case contract — input hashes, water balance, horizon,
#              cost convention, protocol, and the generated stage models. This
#              file is byte-identical to the JuMP engine's copy, so both engines
#              assert the same contract without depending on each other.
#   gradient   the reachable-policy map and its derivative, checked against the
#              same engine-independent oracle the JuMP engine uses. Passing on
#              both sides establishes CPU parity between the two
#              implementations.
#
# The package's own suite (`test/runtests.jl` at the repository root) covers
# `rollout_tsddr` and `RolloutEvaluation`, including scenario identity, retry
# and completeness; it is not duplicated here.
#
# Usage:
#   julia --project=examples/HydroPowerModels examples/HydroPowerModels/test/runtests.jl
#   julia --project=examples/HydroPowerModels examples/HydroPowerModels/test/runtests.jl case

using Test

const TEST_DIR = dirname(@__FILE__)

const GROUPS = [
    "case" => "test_case_manifest.jl",
    "gradient" => "test_reachable_policy_gradient.jl",
]

requested = isempty(ARGS) ? first.(GROUPS) : ARGS
for name in requested
    any(g -> first(g) == name, GROUPS) ||
        error("unknown test group $name; known groups: $(join(first.(GROUPS), ", "))")
end

@testset "Bolivia hydro example (ExaModels)" begin
    for (name, file) in GROUPS
        name in requested || continue
        @testset "$name" begin
            empty!(ARGS)
            include(joinpath(TEST_DIR, file))
        end
    end
end
