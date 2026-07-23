# manifest.jl
#
# Deterministic manifest for a BatteryCase: everything needed to rebuild the
# exact same battery placement and parameters, plus provenance and attribution
# (see BATTERY_STORAGE_OPF_PLAN.md §5).
#
# Two representations:
#   * a canonical CONTENT string → SHA-256 `manifest_hash` that depends only on
#     the scientific content (case name, MATPOWER source-file SHA-256, seed,
#     eligible-bus rule, selected buses, battery params, units, counts) and NOT on
#     timestamps, absolute paths, or package versions, so "same seed ⇒ same hash"
#     holds across runs of the same PGLib artifact;
#   * a JSON manifest file that additionally records provenance, attribution, the
#     license (CC BY 4.0), the content hash, and file hashes.

using SHA
using JSON

# ── PGLib attribution / upstream version (best effort) ────────────────────────

const PGLIB_ATTRIBUTION = string(
    "Network data from the Power Grid Library for Benchmarking AC Optimal ",
    "Power Flow Algorithms (PGLib-OPF), distributed via PGLib.jl. ",
    "Cite: S. Babaeinejadsarookolaee et al., \"The Power Grid Library for ",
    "Benchmarking AC Optimal Power Flow Algorithms\", arXiv:1908.02788.",
)

# PGLib-OPF is released under the Creative Commons Attribution 4.0 license.
const PGLIB_LICENSE_NAME = "Creative Commons Attribution 4.0 International"
const PGLIB_LICENSE_URL  = "https://creativecommons.org/licenses/by/4.0/"

# The PGLib.jl artifact directory is named `pglib-opf-<release>` (e.g.
# `pglib-opf-23.07`), so the upstream benchmark release is read straight from
# the path; fall back to the pinning PGLib.jl package version.
function _pglib_upstream_version()
    m = match(r"pglib-opf-([0-9]+\.[0-9]+)", _pglib_case_dir())
    return m === nothing ? "pinned-by-PGLib.jl-" * _pkg_version("PGLib") : m.captures[1]
end

# ── Canonical content + hash ──────────────────────────────────────────────────

# Deterministic full-precision rendering of a Float64 (round-trippable shortest
# form is stable for equal values).
_fmt(x::Real) = string(Float64(x))
_fmt(x::Integer) = string(x)
_fmt(x::AbstractString) = String(x)

"""
    canonical_content(bc) -> String

Ordered, human-inspectable string capturing exactly the scientific content that
defines the case: the PGLib case name, the SHA-256 of the exact MATPOWER source
bytes, the seed and eligible-bus rule, the ordered battery placement, and every
battery parameter with units. Its SHA-256 is [`manifest_hash`](@ref).
Deliberately excludes timestamps, absolute paths, and package versions so the
hash is reproducible; including the MATPOWER source hash makes the hash sensitive
to the exact network bytes.
"""
function canonical_content(bc::BatteryCase)
    io = IOBuffer()
    nd = bc.network
    println(io, "schema=battery_storage_opf/2")
    println(io, "case_name=", bc.case_name)
    println(io, "matpower_file=", basename(bc.parse_meta.filepath))
    println(io, "matpower_sha256=", bc.parse_meta.matpower_sha256)
    println(io, "baseMVA=", _fmt(nd.baseMVA))
    println(io, "nbus=", nbus(nd), " ngen=", ngen(nd),
                " nbranch=", nbranch(nd), " nload=", nload(nd))
    println(io, "total_load_pu=", _fmt(bc.total_load_pu))
    println(io, "seed=", bc.seed)
    println(io, "eligible_bus_rule=", bc.eligible_bus_rule)
    println(io, "explicit_buses=", bc.explicit_buses)
    println(io, "number_of_batteries=", bc.number_of_batteries)
    println(io, "duration_hours=", _fmt(bc.duration_hours))
    println(io, "initial_soc=", _fmt(bc.initial_soc))
    println(io, "charge_efficiency=", _fmt(bc.charge_efficiency))
    println(io, "discharge_efficiency=", _fmt(bc.discharge_efficiency))
    println(io, "fleet_power_fraction=", _fmt(bc.fleet_power_fraction))
    println(io, "self_discharge_rate=", _fmt(bc.self_discharge_rate))
    println(io, "e_min_fraction=", _fmt(bc.e_min_fraction))
    println(io, "cycle_cost_per_mwh=", _fmt(bc.cycle_cost_per_mwh))
    println(io, "units=power:pu[baseMVA];energy:pu_hours;time:hours")
    println(io, "selected_bus_ids=", join(bc.selected_bus_ids, ","))
    for b in bc.batteries
        println(io, "battery ", b.id,
                    " bus=", b.bus_id,
                    " pch_max=", _fmt(b.p_charge_max),
                    " pdis_max=", _fmt(b.p_discharge_max),
                    " e_min=", _fmt(b.e_min),
                    " e_max=", _fmt(b.e_max),
                    " e_init=", _fmt(b.e_init),
                    " eta_ch=", _fmt(b.eta_ch),
                    " eta_dis=", _fmt(b.eta_dis),
                    " sigma=", _fmt(b.sigma),
                    " cycle_cost_per_mwh=", _fmt(b.cycle_cost_per_mwh))
    end
    return String(take!(io))
end

"""
    manifest_hash(bc) -> String

Lowercase hex SHA-256 of [`canonical_content`](@ref). Identical inputs (same
PGLib case, seed, and sizing parameters) reproduce the same hash.
"""
manifest_hash(bc::BatteryCase) = bytes2hex(sha256(canonical_content(bc)))

_sha256_file(path) = bytes2hex(open(sha256, path))

# ── JSON manifest ─────────────────────────────────────────────────────────────

"""
    battery_manifest(bc) -> Dict

Assemble the full machine-readable manifest: scientific content, provenance
(package + Julia + upstream PGLib versions), attribution, units, the horizon /
terminal-treatment note, and the content hash. File hashes are added by
[`write_manifest`](@ref) once files exist on disk.
"""
function battery_manifest(bc::BatteryCase)
    nd = bc.network
    return Dict(
        "schema" => "battery_storage_opf/2",
        "content_hash_sha256" => manifest_hash(bc),
        "pglib" => Dict(
            "case_name" => bc.case_name,
            "matpower_file" => basename(bc.parse_meta.filepath),
            "matpower_path" => bc.parse_meta.filepath,
            "matpower_sha256" => bc.parse_meta.matpower_sha256,
            "upstream_release" => _pglib_upstream_version(),
            "attribution" => PGLIB_ATTRIBUTION,
            "license" => PGLIB_LICENSE_NAME,
            "license_url" => PGLIB_LICENSE_URL,
        ),
        "versions" => Dict(
            "julia" => string(VERSION),
            "PGLib" => bc.parse_meta.pglib_version,
            "PowerModels" => bc.parse_meta.powermodels_version,
        ),
        "network" => Dict(
            "baseMVA" => nd.baseMVA,
            "nbus" => nbus(nd), "ngen" => ngen(nd),
            "nbranch" => nbranch(nd), "nload" => nload(nd),
            "total_load_pu" => bc.total_load_pu,
            "per_unit_input" => nd.per_unit_input,
        ),
        "placement" => Dict(
            "seed" => bc.seed,
            "eligible_bus_rule" => bc.eligible_bus_rule,
            "explicit_buses" => bc.explicit_buses,
            "n_eligible_buses" => length(bc.eligible_bus_ids),
            "selected_bus_ids" => bc.selected_bus_ids,
        ),
        "battery_parameters" => Dict(
            "number_of_batteries" => bc.number_of_batteries,
            "duration_hours" => bc.duration_hours,
            "initial_soc" => bc.initial_soc,
            "charge_efficiency" => bc.charge_efficiency,
            "discharge_efficiency" => bc.discharge_efficiency,
            "fleet_power_fraction" => bc.fleet_power_fraction,
            "self_discharge_rate" => bc.self_discharge_rate,
            "e_min_fraction" => bc.e_min_fraction,
            "cycle_cost_per_mwh" => bc.cycle_cost_per_mwh,
            "per_battery_p_charge_max_pu" =>
                isempty(bc.batteries) ? 0.0 : bc.batteries[1].p_charge_max,
            "per_battery_p_discharge_max_pu" =>
                isempty(bc.batteries) ? 0.0 : bc.batteries[1].p_discharge_max,
            "per_battery_e_max_pu_h" =>
                isempty(bc.batteries) ? 0.0 : bc.batteries[1].e_max,
            "per_battery_e_init_pu_h" =>
                isempty(bc.batteries) ? 0.0 : bc.batteries[1].e_init,
        ),
        "batteries" => [Dict(
            "id" => b.id, "bus_id" => b.bus_id, "bus_pos" => b.bus_pos,
            "p_charge_max_pu" => b.p_charge_max,
            "p_discharge_max_pu" => b.p_discharge_max,
            "e_min_pu_h" => b.e_min, "e_max_pu_h" => b.e_max,
            "e_init_pu_h" => b.e_init,
            "eta_ch" => b.eta_ch, "eta_dis" => b.eta_dis,
            "sigma_per_h" => b.sigma,
            "cycle_cost_per_mwh" => b.cycle_cost_per_mwh,
        ) for b in bc.batteries],
        "units" => Dict(
            "power" => "per-unit on baseMVA",
            "energy" => "per-unit-hours (pu·h)",
            "time" => "hours",
            "cost" => "USD; generator cost per pu power, battery cost per MWh throughput",
        ),
        "horizon" => Dict(
            "note" => string(
                "Phase 1 deterministic foundation: horizon T and stage length ",
                "Δt (hours) are chosen at model-build time (build_battery_de). ",
                "The stochastic load process, paired protocol, and terminal ",
                "treatment are defined in later phases."),
        ),
    )
end

"""
    write_manifest(bc, path; extra_files = String[]) -> String

Write the JSON manifest to `path`, adding SHA-256 hashes of any `extra_files`
(e.g. a human-readable battery file) and of the manifest's own canonical
content. Returns `path`.
"""
function write_manifest(bc::BatteryCase, path::AbstractString; extra_files = String[])
    man = battery_manifest(bc)
    man["generated_at_utc"] = _utc_now_string()
    fh = Dict{String,String}()
    for f in extra_files
        isfile(f) && (fh[basename(f)] = _sha256_file(f))
    end
    man["file_hashes_sha256"] = fh
    open(path, "w") do io
        JSON.print(io, man, 2)
    end
    return path
end

# ISO-8601 UTC timestamp without pulling in Dates' TimeZones; Libc.strftime is
# stdlib and sufficient for a provenance stamp.
function _utc_now_string()
    t = round(Int, time())
    return Libc.strftime("%Y-%m-%dT%H:%M:%SZ", t)
end

# ── Human-readable battery file ───────────────────────────────────────────────

"""
    write_battery_file(bc, path) -> String

Write a human-readable CSV of the battery fleet (one row per battery, original
bus id preserved). Returns `path`.
"""
function write_battery_file(bc::BatteryCase, path::AbstractString)
    open(path, "w") do io
        println(io, "# BatteryStorageOPF fleet for PGLib case \"", bc.case_name,
                    "\" (seed=", bc.seed, ", baseMVA=", bc.network.baseMVA, ")")
        println(io, "# power/energy are per-unit on baseMVA; energy in pu·h")
        println(io, "battery_id,bus_id,p_charge_max_pu,p_discharge_max_pu,",
                    "e_min_pu_h,e_max_pu_h,e_init_pu_h,eta_ch,eta_dis,sigma_per_h,cycle_cost_per_mwh")
        for b in bc.batteries
            println(io, b.id, ",", b.bus_id, ",", b.p_charge_max, ",",
                        b.p_discharge_max, ",", b.e_min, ",", b.e_max, ",",
                        b.e_init, ",", b.eta_ch, ",", b.eta_dis, ",", b.sigma,
                        ",", b.cycle_cost_per_mwh)
        end
    end
    return path
end

# ── Reconstruction ────────────────────────────────────────────────────────────

"""
    reconstruct_case(manifest_path) -> BatteryCase

Rebuild the case recorded in a JSON manifest and verify it, returning the case
only when ALL of the following match the manifest:

1. **Source network bytes** — the SHA-256 of the currently resolved MATPOWER
   file equals the manifest's recorded `matpower_sha256`. This is checked FIRST,
   directly against the resolved artifact, so a tampered/moved/updated network
   file is rejected before anything else.
2. **Ordered placement** — the reconstructed battery buses match the recorded
   `selected_bus_ids` in the same order.
3. **Battery parameters** — captured by the content hash: re-running
   `make_battery_case` with the recorded case name, seed, and sizing parameters
   reproduces the recorded `content_hash_sha256` (which itself includes the
   source-file hash).

Any mismatch raises an error, so a returned case is verified identical in source
network bytes, ordered placement, and battery parameters to the recorded one.
"""
function reconstruct_case(manifest_path::AbstractString)
    man = JSON.parsefile(manifest_path)
    bp = man["battery_parameters"]
    pl = man["placement"]
    pg = man["pglib"]

    # 1. Verify the source network bytes FIRST, against the currently resolved
    # MATPOWER file, before trusting anything else in the manifest.
    recorded_src = String(pg["matpower_sha256"])
    _, filepath = resolve_pglib_case(String(pg["case_name"]))
    current_src = _sha256_file(filepath)
    current_src == recorded_src || error(
        "source-file hash mismatch for \"$(pg["case_name"])\": manifest recorded " *
        "$recorded_src but the resolved MATPOWER file hashes to $current_src. " *
        "The network data differs from when the manifest was written.")

    explicit = Bool(pl["explicit_buses"])
    kwargs = (
        number_of_batteries = Int(bp["number_of_batteries"]),
        seed = Int(pl["seed"]),
        duration_hours = Float64(bp["duration_hours"]),
        initial_soc = Float64(bp["initial_soc"]),
        charge_efficiency = Float64(bp["charge_efficiency"]),
        discharge_efficiency = Float64(bp["discharge_efficiency"]),
        fleet_power_fraction = Float64(bp["fleet_power_fraction"]),
        self_discharge_rate = Float64(bp["self_discharge_rate"]),
        e_min_fraction = Float64(bp["e_min_fraction"]),
        cycle_cost_per_mwh = Float64(bp["cycle_cost_per_mwh"]),
    )
    bc = if explicit
        make_battery_case(String(pg["case_name"]);
                          buses = Int.(pl["selected_bus_ids"]), kwargs...)
    else
        make_battery_case(String(pg["case_name"]); kwargs...)
    end

    # 2. Ordered placement.
    Int.(pl["selected_bus_ids"]) == bc.selected_bus_ids ||
        error("reconstruction produced a different battery placement order")

    # 3. Battery parameters (via the content hash, which embeds the source hash).
    got = manifest_hash(bc)
    want = String(man["content_hash_sha256"])
    got == want || error(
        "reconstruction content-hash mismatch: manifest recorded $want but " *
        "rebuilt $got. Battery parameters differ from when the manifest was written.")
    return bc
end
