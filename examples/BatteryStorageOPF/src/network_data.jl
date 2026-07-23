# network_data.jl
#
# Typed parsing of a PGLib / PowerModels AC network into the flat, per-unit
# structures the ExaModels AC-polar builder consumes.
#
# Design rules honored here (see BATTERY_STORAGE_OPF_PLAN.md §5):
#   * PGLib component identifiers are NEVER assumed consecutive or equal to
#     their array position. Every component keeps its original PGLib `index`
#     and we build explicit, stable ID → position maps (`*_id_to_pos`).
#   * All unit conversion happens HERE, in one documented data layer, so model
#     code sees only per-unit quantities on the `baseMVA` base with angles in
#     radians. Nothing downstream re-scales.
#
# We parse the raw MATPOWER data exactly as `PowerModels.parse_file` returns it
# (mixed units, `per_unit = false`: powers in MW/MVAr, voltages in pu, angles
# already converted to radians by the parser). The conversion below is gated on
# the observed `per_unit` flag so a pre-converted dict is also handled safely.

using PowerModels
using PGLib
using SHA

# ── Typed, per-unit network components ────────────────────────────────────────
# Every struct stores the original PGLib id AND the resolved 1-based array
# position of any component it references.

"""
    BusData

One AC bus, per unit on `baseMVA`.

* `id`        : original PGLib bus index (may be non-consecutive).
* `bus_type`  : PowerModels bus type (1 = PQ, 2 = PV, 3 = reference/slack).
* `gs`, `bs`  : shunt conductance / susceptance at the bus (pu).
* `vmin`,`vmax`: voltage-magnitude limits (pu).
"""
struct BusData
    id::Int
    bus_type::Int
    gs::Float64
    bs::Float64
    vmin::Float64
    vmax::Float64
end

"""
    GenData

One in-service generator, per unit on `baseMVA`.

* `id`               : original PGLib generator index.
* `bus_id`,`bus_pos` : the bus it injects into (original id and array position).
* `pmin`,`pmax`      : active-power limits (pu).
* `qmin`,`qmax`      : reactive-power limits (pu).
* `cost2`,`cost1`,`cost0`: polynomial cost so that the \$/hour cost of `pg` pu is
  `cost2·pg² + cost1·pg + cost0`. Converted once from the MATPOWER \$/MW model.
"""
struct GenData
    id::Int
    bus_id::Int
    bus_pos::Int
    pmin::Float64
    pmax::Float64
    qmin::Float64
    qmax::Float64
    cost2::Float64
    cost1::Float64
    cost0::Float64
end

"""
    BranchData

One in-service branch, per unit on `baseMVA`, with the raw AC-polar parameters
used by [`_ac_branch_coeffs`](@ref).

* `id`                 : original PGLib branch index.
* `f_id`,`f_pos`       : from-bus original id and array position.
* `t_id`,`t_pos`       : to-bus original id and array position.
* `br_r`,`br_x`        : series resistance / reactance (pu).
* `g_fr`,`b_fr`,`g_to`,`b_to`: line-charging shunts (pu).
* `tap`,`shift`        : transformer turns ratio and phase shift (rad).
* `rate_a`             : apparent-power thermal limit (pu); `Inf` if unlimited.
* `angmin`,`angmax`    : angle-difference limits (rad).
"""
struct BranchData
    id::Int
    f_id::Int
    f_pos::Int
    t_id::Int
    t_pos::Int
    br_r::Float64
    br_x::Float64
    g_fr::Float64
    b_fr::Float64
    g_to::Float64
    b_to::Float64
    tap::Float64
    shift::Float64
    rate_a::Float64
    angmin::Float64
    angmax::Float64
end

"""
    LoadData

One in-service load, per unit on `baseMVA`.

* `id`               : original PGLib load index.
* `bus_id`,`bus_pos` : the bus it draws from (original id and array position).
* `pd`,`qd`          : active / reactive demand (pu).
"""
struct LoadData
    id::Int
    bus_id::Int
    bus_pos::Int
    pd::Float64
    qd::Float64
end

"""
    NetworkData

Fully parsed, per-unit AC network. Components are stored sorted by original id;
`*_id_to_pos` give the stable inverse maps. `bus_pd`/`bus_qd` aggregate the
loads onto buses (length `nbus`, positional).
"""
struct NetworkData
    case_name::String
    baseMVA::Float64
    buses::Vector{BusData}
    gens::Vector{GenData}
    branches::Vector{BranchData}
    loads::Vector{LoadData}
    ref_bus_positions::Vector{Int}
    bus_id_to_pos::Dict{Int,Int}
    gen_id_to_pos::Dict{Int,Int}
    branch_id_to_pos::Dict{Int,Int}
    load_id_to_pos::Dict{Int,Int}
    bus_pd::Vector{Float64}   # aggregated active demand per bus (pu)
    bus_qd::Vector{Float64}   # aggregated reactive demand per bus (pu)
    per_unit_input::Bool      # whether the source dict was already per-unit
end

nbus(nd::NetworkData)    = length(nd.buses)
ngen(nd::NetworkData)    = length(nd.gens)
nbranch(nd::NetworkData) = length(nd.branches)
nload(nd::NetworkData)   = length(nd.loads)

# ── PGLib case-name resolution ────────────────────────────────────────────────

"""
    _pglib_case_dir() -> String

Absolute directory holding the `pglib_opf_*.m` benchmark files that ship with
the pinned `PGLib.jl` artifact.
"""
_pglib_case_dir() = PGLib.PGLib_opf

"""
    available_pglib_cases() -> Vector{String}

Sorted canonical short names (the `pglib_opf_` prefix and `.m` suffix stripped)
of every case in the pinned PGLib artifact, e.g. `"case300_ieee"`.
"""
function available_pglib_cases()
    dir = _pglib_case_dir()
    names = String[]
    for f in readdir(dir)
        (endswith(f, ".m") && startswith(f, "pglib_opf_")) || continue
        push!(names, replace(replace(f, r"\.m$" => ""), r"^pglib_opf_" => ""))
    end
    return sort!(names)
end

"""
    resolve_pglib_case(name) -> (canonical_name, filepath)

Resolve a user-supplied PGLib case name to its canonical short name and the
absolute `.m` file path, unambiguously.

Accepted forms (all mapped to the same case): `"case300_ieee"`,
`"pglib_opf_case300_ieee"`, `"pglib_opf_case300_ieee.m"`.

Errors:
* a name matching no case lists the closest available names;
* a name matching several cases (only possible via a partial/loose query) is
  rejected with the full ambiguous set. Exact canonical matches are never
  ambiguous.
"""
function resolve_pglib_case(name::AbstractString)
    dir = _pglib_case_dir()
    cases = available_pglib_cases()

    # Normalize the query to a canonical short name.
    q = String(name)
    q = replace(q, r"\.m$" => "")
    q = replace(q, r"^pglib_opf_" => "")

    # Exact canonical match wins outright (never ambiguous).
    if q in cases
        return q, joinpath(dir, "pglib_opf_" * q * ".m")
    end

    # Otherwise treat the query as a substring and demand a unique hit.
    hits = filter(c -> occursin(q, c), cases)
    if isempty(hits)
        # Offer nearby suggestions by shared prefix to make the error actionable.
        prefix = first(q, min(length(q), 6))
        near = filter(c -> startswith(c, prefix), cases)
        suffix = isempty(near) ? "" : "\n  Did you mean: " * join(first(near, 8), ", ")
        error("PGLib case \"$name\" not found in $(length(cases)) available cases." *
              suffix *
              "\n  Use available_pglib_cases() to list them all.")
    elseif length(hits) > 1
        error("PGLib case \"$name\" is ambiguous; it matches $(length(hits)) cases:\n  " *
              join(hits, ", ") *
              "\n  Pass an exact canonical name (e.g. one of the above).")
    end
    c = only(hits)
    return c, joinpath(dir, "pglib_opf_" * c * ".m")
end

# ── Parsing + one documented unit-conversion layer ────────────────────────────

# Extract the polynomial generator cost as (cost2, cost1, cost0) with pg in pu.
# MATPOWER model-2 polynomials are in \$ with power in MW; converting to pu
# multiplies the degree-k coefficient by baseMVA^k. Piecewise-linear (model 1)
# cost is not supported by this example and errors loudly.
function _gen_cost_pu(gen, sbase::Float64)
    model = Int(get(gen, "model", 2))
    model == 2 || error("generator $(get(gen,"index","?")) uses cost model $model; " *
                        "only MATPOWER polynomial cost (model 2) is supported")
    cost = Float64.(get(gen, "cost", Float64[]))
    ncost = Int(get(gen, "ncost", length(cost)))
    # Right-align: MATPOWER lists highest-order coefficient first.
    c2 = c1 = c0 = 0.0
    if ncost >= 3
        c2, c1, c0 = cost[end-2], cost[end-1], cost[end]
    elseif ncost == 2
        c1, c0 = cost[end-1], cost[end]
    elseif ncost == 1
        c0 = cost[end]
    end
    return (c2 * sbase^2, c1 * sbase, c0)
end

"""
    parse_network(data::Dict, case_name) -> NetworkData

Parse a PowerModels network dict into per-unit [`NetworkData`], keeping only
in-service components (bus_type ≠ 4, and unit/branch/load status = 1) and
building explicit stable id → position maps.

Power quantities are divided by `baseMVA` iff the dict is not already per-unit
(`data["per_unit"] == false`, the state `PowerModels.parse_file` returns).
"""
function parse_network(data::AbstractDict, case_name::AbstractString)
    sbase = Float64(data["baseMVA"])
    per_unit = Bool(get(data, "per_unit", false))
    # Scale factor applied to MW/MVAr quantities (1.0 when already per-unit).
    s = per_unit ? 1.0 : 1.0 / sbase

    # ── Shunt admittance (PowerModels keeps shunts as a SEPARATE component, not
    # on the bus). Aggregate in-service shunt gs/bs per original bus id so they
    # enter the power balance as gs·vm² (active) and −bs·vm² (reactive). Shunt
    # gs/bs are already per-unit; the same `s` gate applies for consistency.
    shunt_gs = Dict{Int,Float64}(); shunt_bs = Dict{Int,Float64}()
    for sh in values(get(data, "shunt", Dict{String,Any}()))
        Int(get(sh, "status", 1)) == 1 || continue
        bid = Int(sh["shunt_bus"])
        shunt_gs[bid] = get(shunt_gs, bid, 0.0) + Float64(get(sh, "gs", 0.0)) * s
        shunt_bs[bid] = get(shunt_bs, bid, 0.0) + Float64(get(sh, "bs", 0.0)) * s
    end

    # ── Buses (drop isolated bus_type == 4) ───────────────────────────────────
    bus_raw = collect(values(data["bus"]))
    filter!(b -> Int(b["bus_type"]) != 4, bus_raw)
    sort!(bus_raw, by = b -> Int(b["index"]))
    buses = BusData[]
    bus_id_to_pos = Dict{Int,Int}()
    for (pos, b) in enumerate(bus_raw)
        id = Int(b["index"])
        bus_id_to_pos[id] = pos
        # Bus-level gs/bs (rare) plus any separate shunt components at this bus.
        gs = Float64(get(b, "gs", 0.0)) * s + get(shunt_gs, id, 0.0)
        bs = Float64(get(b, "bs", 0.0)) * s + get(shunt_bs, id, 0.0)
        push!(buses, BusData(id, Int(b["bus_type"]), gs, bs,
                             Float64(get(b, "vmin", 0.9)),
                             Float64(get(b, "vmax", 1.1))))
    end
    ref_bus_positions = [bus_id_to_pos[b.id] for b in buses if b.bus_type == 3]
    isempty(ref_bus_positions) &&
        error("network \"$case_name\" has no reference (type-3) bus")

    # ── Generators (in-service only) ──────────────────────────────────────────
    gen_raw = collect(values(data["gen"]))
    filter!(g -> Int(get(g, "gen_status", 1)) == 1, gen_raw)
    sort!(gen_raw, by = g -> Int(g["index"]))
    gens = GenData[]
    gen_id_to_pos = Dict{Int,Int}()
    for (pos, g) in enumerate(gen_raw)
        id = Int(g["index"])
        bus_id = Int(g["gen_bus"])
        haskey(bus_id_to_pos, bus_id) ||
            error("generator $id references out-of-service/unknown bus $bus_id")
        c2, c1, c0 = _gen_cost_pu(g, per_unit ? 1.0 : sbase)
        gen_id_to_pos[id] = pos
        push!(gens, GenData(id, bus_id, bus_id_to_pos[bus_id],
                            Float64(get(g, "pmin", 0.0)) * s,
                            Float64(g["pmax"]) * s,
                            Float64(get(g, "qmin", -Inf)) * s,
                            Float64(get(g, "qmax",  Inf)) * s,
                            c2, c1, c0))
    end

    # ── Branches (in-service only) ────────────────────────────────────────────
    br_raw = collect(values(data["branch"]))
    filter!(br -> Int(get(br, "br_status", 1)) == 1, br_raw)
    sort!(br_raw, by = br -> Int(br["index"]))
    branches = BranchData[]
    branch_id_to_pos = Dict{Int,Int}()
    for (pos, br) in enumerate(br_raw)
        id = Int(br["index"])
        f_id = Int(br["f_bus"]); t_id = Int(br["t_bus"])
        (haskey(bus_id_to_pos, f_id) && haskey(bus_id_to_pos, t_id)) ||
            error("branch $id references an out-of-service/unknown bus " *
                  "($f_id → $t_id)")
        tap = Float64(get(br, "tap", 1.0)); tap = tap ≈ 0 ? 1.0 : tap
        rate = Float64(get(br, "rate_a", 0.0))
        rate_pu = rate ≈ 0 ? Inf : rate * s   # 0 ⇒ unlimited (PowerModels convention)
        branch_id_to_pos[id] = pos
        push!(branches, BranchData(id, f_id, bus_id_to_pos[f_id],
                                   t_id, bus_id_to_pos[t_id],
                                   Float64(get(br, "br_r", 0.0)),
                                   Float64(br["br_x"]),
                                   Float64(get(br, "g_fr", 0.0)),
                                   Float64(get(br, "b_fr", 0.0)),
                                   Float64(get(br, "g_to", 0.0)),
                                   Float64(get(br, "b_to", 0.0)),
                                   tap, Float64(get(br, "shift", 0.0)),
                                   rate_pu,
                                   Float64(get(br, "angmin", -pi)),
                                   Float64(get(br, "angmax",  pi))))
    end

    # ── Loads (in-service only) ───────────────────────────────────────────────
    load_raw = collect(values(data["load"]))
    filter!(l -> Int(get(l, "status", 1)) == 1, load_raw)
    sort!(load_raw, by = l -> Int(l["index"]))
    loads = LoadData[]
    load_id_to_pos = Dict{Int,Int}()
    bus_pd = zeros(Float64, length(buses))
    bus_qd = zeros(Float64, length(buses))
    for (pos, l) in enumerate(load_raw)
        id = Int(l["index"])
        bus_id = Int(l["load_bus"])
        haskey(bus_id_to_pos, bus_id) ||
            error("load $id references an out-of-service/unknown bus $bus_id")
        bpos = bus_id_to_pos[bus_id]
        pd = Float64(get(l, "pd", 0.0)) * s
        qd = Float64(get(l, "qd", 0.0)) * s
        load_id_to_pos[id] = pos
        push!(loads, LoadData(id, bus_id, bpos, pd, qd))
        bus_pd[bpos] += pd
        bus_qd[bpos] += qd
    end

    return NetworkData(String(case_name), sbase, buses, gens, branches, loads,
                       ref_bus_positions, bus_id_to_pos, gen_id_to_pos,
                       branch_id_to_pos, load_id_to_pos, bus_pd, bus_qd, per_unit)
end

"""
    load_pglib_network(name) -> (NetworkData, parse_meta)

Resolve `name` in the pinned PGLib artifact, parse the `.m` file with
PowerModels, and return the typed [`NetworkData`] plus a small metadata
NamedTuple (`canonical_name`, `filepath`, `pglib_version`, `powermodels_version`)
recorded in the manifest.
"""
function load_pglib_network(name::AbstractString)
    canonical, filepath = resolve_pglib_case(name)
    data = PowerModels.parse_file(filepath)
    nd = parse_network(data, canonical)
    meta = (canonical_name = canonical,
            filepath = filepath,
            # SHA-256 of the exact MATPOWER source bytes (see manifest.jl); this
            # pins provenance to the resolved network file.
            matpower_sha256 = bytes2hex(open(SHA.sha256, filepath)),
            pglib_version = _pkg_version("PGLib"),
            powermodels_version = _pkg_version("PowerModels"))
    return nd, meta
end

# Resolve an installed dependency's version string by reading the active
# environment's Manifest.toml with the TOML stdlib. We deliberately avoid
# `using Pkg` at runtime: on this cluster Pkg's native precompile image is
# broken, so the example is kept Pkg-free (Pkg is only used, with
# --pkgimages=no, by the one-shot setup_env.jl).
function _pkg_version(name::AbstractString)
    manifest = Base.active_project() === nothing ? nothing :
               joinpath(dirname(Base.active_project()), "Manifest.toml")
    (manifest === nothing || !isfile(manifest)) && return "unknown"
    data = TOML.parsefile(manifest)
    deps = get(data, "deps", data)  # manifest format 2.0 nests under "deps"
    entry = get(deps, name, nothing)
    entry === nothing && return "unknown"
    # Each package maps to a 1-element array of tables.
    rec = entry isa AbstractVector ? first(entry) : entry
    return string(get(rec, "version", "unknown"))
end
