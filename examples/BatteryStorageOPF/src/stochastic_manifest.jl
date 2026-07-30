# stochastic_manifest.jl
#
# Phase-2 experiment manifest: the single artifact from which the whole
# experiment is reconstructed EXACTLY, with no defaults and no environment-
# variable guesses.
#
# It builds on the accepted Phase-1 `battery_manifest` (which records and lets
# `reconstruct_case` verify the MATPOWER source hash) and adds every value needed
# to rebuild the demand process field-for-field: the exact `base_shape` vector,
# `region_of_bus`, region anchors, the complete ORDERED atom definitions and
# probabilities, the period, all seeds, and the selected calibration preset —
# plus the horizons, stage duration, target mode, active-recourse / target-penalty
# configuration, and the policy architecture and activation.
#
# FIVE DISTINCT hashes are stored; none of them is reused for another role:
#   1. load_process_content_hash_sha256   — the process definition
#   2. train_index_matrix_hash_sha256     — training scenario-index matrix
#   3. eval_index_matrix_hash_sha256      — evaluation scenario-index matrix
#   4. train_protocol_file_sha256         — exact training protocol JSON bytes
#   5. eval_protocol_file_sha256          — exact evaluation protocol JSON bytes

using JSON
using SHA

"""
    stochastic_manifest(case, process; kwargs...) -> Dict

Assemble the Phase-2 experiment manifest. Required keywords: `reporting_horizon`,
`lookahead`, `mode`. Optional: `stage_hours`, `active_recourse_cost_per_mwh`,
`rho1`, `rho2`, `activation`, `safe_upper_margin`, `policy_layers`,
`policy_combiner_layers`, `policy_seed`, and the four artifact hashes
(`train_index_matrix_hash`, `eval_index_matrix_hash`,
`train_protocol_file_sha256`, `eval_protocol_file_sha256`) with their path counts.
"""
function stochastic_manifest(case::BatteryCase, process::LoadProcess;
                             reporting_horizon::Int, lookahead::Int, mode::Symbol,
                             stage_hours::Real = 1.0,
                             active_recourse_cost_per_mwh::Real = DEFAULT_ACTIVE_RECOURSE_COST_PER_MWH,
                             rho1::Real = 0.0, rho2::Real = 0.0,
                             activation::AbstractString = "stretchedsigmoid",
                             safe_upper_margin::Real = 1e-3,
                             policy_layers = Int[], policy_combiner_layers = Int[],
                             policy_seed = nothing,
                             train_index_matrix_hash = nothing,
                             eval_index_matrix_hash = nothing,
                             train_protocol_file_sha256 = nothing,
                             eval_protocol_file_sha256 = nothing,
                             train_paths = nothing, eval_paths = nothing)
    mult = demand_multiplier_summary(process)
    return Dict{String,Any}(
        "schema" => "battery_storage_opf_stochastic/3",
        "case" => battery_manifest(case),   # Phase-1 manifest (with MATPOWER source hash)

        # ── EXACT demand-process definition (field-for-field reconstruction) ──
        "load_process" => Dict{String,Any}(
            "preset" => String(process.preset),
            "nregion" => process.nregion,
            "region_rule" => process.region_rule,
            "anchor_bus_ids" => process.anchor_bus_ids,
            "region_of_bus" => process.region_of_bus,
            "period" => process.period,
            "base_shape" => process.base_shape,
            "atoms" => [Dict("prob" => process.probs[a],
                             "system_factor" => process.atoms[a].system_factor,
                             "regional_factors" => process.atoms[a].regional_factors)
                        for a in 1:natom(process)],
            "nw_per_stage" => n_uncertainty(process),
            "natom" => natom(process),
            "train_seed" => process.train_seed,
            "eval_seed" => process.eval_seed,
            "max_system_multiplier" => mult.max_system_multiplier,
            "max_bus_multiplier" => mult.max_bus_multiplier,
            "min_bus_multiplier" => mult.min_bus_multiplier,
        ),

        # ── Five distinct hashes (never reused across roles) ─────────────────
        "hashes" => Dict{String,Any}(
            "load_process_content_hash_sha256" => process_hash(process),
            "train_index_matrix_hash_sha256" => train_index_matrix_hash,
            "eval_index_matrix_hash_sha256" => eval_index_matrix_hash,
            "train_protocol_file_sha256" => train_protocol_file_sha256,
            "eval_protocol_file_sha256" => eval_protocol_file_sha256,
        ),
        "paths" => Dict{String,Any}("train" => train_paths, "eval" => eval_paths),

        "horizon" => Dict{String,Any}(
            "reporting_horizon" => reporting_horizon,
            "lookahead" => lookahead,
            "horizon" => reporting_horizon + lookahead,
            "stage_hours" => Float64(stage_hours),
            "terminal_treatment" =>
                "reporting horizon followed by a look-ahead buffer; identical for every method",
        ),
        "target_mode" => String(mode),
        "active_recourse" => Dict{String,Any}(
            "formulation" => "two-sided active nodal slack",
            "deficit_sign" => "d⁺ ≥ 0 (injection, covers a local active shortfall)",
            "surplus_sign" => "d⁻ ≥ 0 (absorption, absorbs a local active excess)",
            "unbounded_above" => true,
            "active_only" => true,
            "active_recourse_cost_per_mwh" => Float64(active_recourse_cost_per_mwh),
            "units" => "USD/MWh; stage coefficient = cost · baseMVA · Δt",
            "included_in_physical_operating_cost" => true,
            "accepted_scientific_requirement" =>
                "both directions (deficit d⁺ and surplus d⁻) zero within declared tolerance",
        ),
        "target_penalty" => Dict{String,Any}(
            "rho1" => Float64(rho1), "rho2" => Float64(rho2),
            "form" => "rho1*sum(slack_pos+slack_neg) + rho2/2*sum(slack_pos^2+slack_neg^2)",
            "included_in_physical_operating_cost" => false,
        ),
        "policy" => Dict{String,Any}(
            "architecture" => "BatteryReachablePolicy",
            "encoder_layers" => collect(Int, policy_layers),
            "combiner_layers" => collect(Int, policy_combiner_layers),
            "activation" => String(activation),
            "safe_upper_margin" => Float64(safe_upper_margin),
            "policy_seed" => policy_seed,
            "note" => "normalized target lies in [0, 1 - safe_upper_margin]; never exactly 1",
        ),
        "cost_definitions" => Dict{String,Any}(
            "physical_operating_cost" =>
                "generator + battery throughput + active-recourse cost (deficit + surplus)",
            "training_only" => "target penalty (soft mode)",
            "headline_metric" => "reporting-window physical operating cost",
        ),
        "units" => Dict{String,Any}(
            "power" => "per-unit on baseMVA", "energy" => "per-unit-hours (pu·h)",
            "time" => "hours",
        ),
    )
end

"""
    write_stochastic_manifest(path, case, process; kwargs...) -> String

Write the [`stochastic_manifest`](@ref) to `path` as JSON (adding a UTC
timestamp). Returns `path`.
"""
function write_stochastic_manifest(path::AbstractString, case::BatteryCase,
                                   process::LoadProcess; kwargs...)
    man = stochastic_manifest(case, process; kwargs...)
    man["generated_at_utc"] = _utc_now_string()
    open(io -> JSON.print(io, man, 2), path, "w")
    return path
end

"""
    reconstruct_stochastic_manifest(path) -> (case, process, meta)

Rebuild the experiment from a Phase-2 manifest and verify it:

1. the Phase-1 case is reconstructed through the accepted path (the MATPOWER
   source hash is re-checked against the resolved artifact and the case content
   hash is reproduced);
2. the demand process is rebuilt FIELD-FOR-FIELD from the stored exact values
   (base shape, regions, anchors, ordered atoms, probabilities, period, seeds,
   preset) with no defaults, and must reproduce
   `hashes.load_process_content_hash_sha256`.

`meta` carries the horizons, stage duration, target mode, VOLL, penalty
coefficients, policy architecture/activation, path counts, and the four artifact
hashes so a caller can verify the protocol files it loads.
"""
function reconstruct_stochastic_manifest(path::AbstractString)
    doc = JSON.parsefile(path)

    # 1. Phase-1 case via the accepted verification path.
    tmp = tempname() * ".json"
    open(io -> JSON.print(io, doc["case"], 2), tmp, "w")
    case = try
        reconstruct_case(tmp)
    finally
        isfile(tmp) && rm(tmp; force = true)
    end

    # 2. Demand process from EXACT stored fields.
    lp = doc["load_process"]
    atoms = LoadAtom[LoadAtom(Float64(a["system_factor"]), Float64.(a["regional_factors"]))
                     for a in lp["atoms"]]
    probs = Float64[Float64(a["prob"]) for a in lp["atoms"]]
    process = load_process_from_fields(;
        nregion = Int(lp["nregion"]),
        region_of_bus = Int.(lp["region_of_bus"]),
        anchor_bus_ids = Int.(lp["anchor_bus_ids"]),
        region_rule = String(lp["region_rule"]),
        base_shape = Float64.(lp["base_shape"]),
        period = Int(lp["period"]),
        atoms = atoms, probs = probs,
        train_seed = Int(lp["train_seed"]), eval_seed = Int(lp["eval_seed"]),
        preset = String(lp["preset"]))

    h = doc["hashes"]
    got = process_hash(process)
    want = String(h["load_process_content_hash_sha256"])
    got == want ||
        error("load-process content-hash mismatch: manifest recorded $want but rebuilt $got")

    hz = doc["horizon"]; pol = doc["policy"]; tp = doc["target_penalty"]
    meta = (reporting_horizon = Int(hz["reporting_horizon"]),
            lookahead = Int(hz["lookahead"]),
            horizon = Int(hz["horizon"]),
            stage_hours = Float64(hz["stage_hours"]),
            mode = Symbol(doc["target_mode"]),
            active_recourse_cost_per_mwh =
                Float64(doc["active_recourse"]["active_recourse_cost_per_mwh"]),
            rho1 = Float64(tp["rho1"]), rho2 = Float64(tp["rho2"]),
            activation = String(pol["activation"]),
            encoder_layers = Int.(pol["encoder_layers"]),
            combiner_layers = Int.(pol["combiner_layers"]),
            train_paths = get(doc["paths"], "train", nothing),
            eval_paths = get(doc["paths"], "eval", nothing),
            train_index_matrix_hash = get(h, "train_index_matrix_hash_sha256", nothing),
            eval_index_matrix_hash = get(h, "eval_index_matrix_hash_sha256", nothing),
            train_protocol_file_sha256 = get(h, "train_protocol_file_sha256", nothing),
            eval_protocol_file_sha256 = get(h, "eval_protocol_file_sha256", nothing))
    return case, process, meta
end

"""
    verify_protocol_file(path, expected_sha256) -> String

Verify that a protocol JSON file's EXACT BYTES hash to `expected_sha256` and
return the hash. Raises on mismatch (tamper detection at the file level, distinct
from the index-matrix content hash).
"""
function verify_protocol_file(path::AbstractString, expected_sha256)
    got = bytes2hex(open(sha256, path))
    expected_sha256 === nothing && return got
    got == String(expected_sha256) ||
        error("protocol file hash mismatch for $path: expected $expected_sha256, got $got")
    return got
end
