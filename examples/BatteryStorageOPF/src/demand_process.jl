# demand_process.jl
#
# Seeded, finite-support, PURE demand-uncertainty process for the battery-storage
# TS-DDR example (BATTERY_STORAGE_OPF_PLAN.md §6). The process is deterministic
# given its parameters and a seed: it turns integer atom indices into realized
# per-stage demand multipliers with NO dependence on generator prices and NO
# renewables.
#
# Structure of the uncertainty at one stage t:
#
#     w_t = [ s_t , r_{1,t} , … , r_{R,t} ]            (length nw = 1 + nregion)
#
#   * s_t   = base_shape[t] · L^{a_t}  is the realized SYSTEM-WIDE multiplier,
#             the deterministic hourly time-of-day shape times the atom's random
#             system load factor L;
#   * r_{k,t} = R_k^{a_t}              is the realized multiplier of REGION k.
#
# Region k of a bus is fixed by a deterministic, price-free, topology-derived
# rule (nearest graph-anchor, see [`assign_regions`](@ref)). The realized demand
# at bus p, stage t is
#
#     pd_realized = bus_pd[p] · s_t · r_{region(p),t},
#     qd_realized = bus_qd[p] · s_t · r_{region(p),t},
#
# so active and reactive demand are scaled by the SAME factor and every bus keeps
# its original power factor. Both the policy (which observes `w_t`) and the
# ExaModels NLP (which multiplies `bus_pd`/`bus_qd` by the parameter entries)
# consume exactly this `w_t`, so there is one canonical uncertainty vector.
#
# The support is a small finite set of joint atoms `(L, R_1, …, R_R)` with
# explicit probabilities, drawn i.i.d. across stages (stagewise independent — the
# form ordinary SDDP backward passes need in a later phase). Atom-index matrices
# are generated with StableRNGs so `(same params, same seed) ⇒ identical indices
# and hashes` across Julia versions and machines.

using StableRNGs
using Random
using SHA
using JSON

# ── Atoms and the load process ────────────────────────────────────────────────

"""
    LoadAtom

One joint realization of the finite demand support.

* `system_factor`     : the system-wide random load factor `L` (dimensionless).
* `regional_factors`  : per-region multipliers `R_k`, length `nregion`.

The realized system multiplier at a stage is `base_shape[t]·system_factor`; each
region's multiplier is its `regional_factors` entry. Active and reactive demand
are scaled by the same product, preserving every bus's power factor.
"""
struct LoadAtom
    system_factor::Float64
    regional_factors::Vector{Float64}
end

"""
    LoadProcess

A pure, seeded, finite-support demand process for a fixed [`BatteryCase`] network.

Fields:
* `nregion`         : number of demand regions `R` (≥ 1).
* `region_of_bus`   : length-`nbus` vector; entry `p` is the region (1..R) of the
  bus at array position `p`. Deterministic and topology-derived.
* `anchor_bus_ids`  : the `R` original PGLib bus ids used as region anchors, in
  region order (region `k`'s anchor is `anchor_bus_ids[k]`).
* `region_rule`     : short string naming the assignment rule (provenance).
* `base_shape`      : deterministic time-of-day shape, length `period`, values
  around 1. Stage `t` uses `base_shape[((t-1) mod period) + 1]`.
* `period`          : base-shape period in stages (e.g. 24 for hourly).
* `atoms`           : the finite support, a vector of [`LoadAtom`].
* `probs`           : atom probabilities, same length as `atoms`, summing to 1.
* `train_seed`      : declared RNG seed for TRAINING scenario generation.
* `eval_seed`       : declared RNG seed for EVALUATION scenario generation
  (distinct from `train_seed`).

The per-stage uncertainty dimension is `nw = 1 + nregion`.
"""
struct LoadProcess
    nregion::Int
    region_of_bus::Vector{Int}
    anchor_bus_ids::Vector{Int}
    region_rule::String
    base_shape::Vector{Float64}
    period::Int
    atoms::Vector{LoadAtom}
    probs::Vector{Float64}
    train_seed::Int
    eval_seed::Int
    preset::Symbol            # calibration preset name (:D0/:D1/:D2/:custom)
end

# ── Calibration ladder (fixed, declared; strongest → weakest) ─────────────────
#
# The demand amplitude is a CASE-DESIGN parameter: too heavy a process makes the
# realized demand unservable and forces nonzero active recourse (deficit d⁺ /
# surplus d⁻), which invalidates a scientific run. These three presets are the
# declared ladder; the first that passes the targetless-diagnostic and soft
# zero-active-recourse gates on case14 AND case300 becomes the public default.
# Rejected (stronger) presets remain available as named experimental presets. No
# interpolation, no unrestricted search.
"""
    DEMAND_PRESETS

Fixed calibration ladder for the default demand process, strongest first:

| preset | `base_amplitude` | high system factor | scarce regional | off-region |
|:--|:--|:--|:--|:--|
| `:D0` | 0.05 | 1.03 | 1.05 | 0.99  |
| `:D1` | 0.04 | 1.02 | 1.04 | 0.99  |
| `:D2` | 0.03 | 1.01 | 1.03 | 0.995 |

The finite system/regional atom structure and the calm probability are unchanged
across presets. Passing this ladder is a per-case, per-experiment obligation —
see [`DEFAULT_DEMAND_PRESET`](@ref) for the current (non-passing) status.
"""
const DEMAND_PRESETS = Dict(
    :D0 => (base_amplitude = 0.05, high_factor = 1.03, scarce_factor = 1.05, off_scarce_factor = 0.99),
    :D1 => (base_amplitude = 0.04, high_factor = 1.02, scarce_factor = 1.04, off_scarce_factor = 0.99),
    :D2 => (base_amplitude = 0.03, high_factor = 1.01, scarce_factor = 1.03, off_scarce_factor = 0.995),
)

"""
    DEFAULT_DEMAND_PRESET

The weakest declared preset (`:D2`), used as the convenient **tutorial / API
default** so the documented one-command examples run.

!!! warning "Not a scientifically accepted default"
    `:D2` is a **conservative tutorial default**, not a gate-passing scientific
    benchmark. No member of the declared ladder (`:D0`, `:D1`, `:D2`) passed the
    case300 zero-active-recourse acceptance checks.

    * `:D2` passes the current **case14** checks.
    * `:D2` does **not** pass the current **case300** zero-active-recourse checks.
    * Every scientific experiment must record its preset explicitly and
      independently pass the zero-active-recourse gates (both deficit d⁺ and
      surplus d⁻ zero within tolerance) for its own case.
    * The current **case300 battery placement/configuration is a REJECTED
      candidate**, not a frozen benchmark; screening it is Phase-3 work.
"""
const DEFAULT_DEMAND_PRESET = :D2

"""
    n_uncertainty(process) -> Int

Per-stage uncertainty dimension `nw = 1 + nregion` (the realized system
multiplier followed by the per-region multipliers).
"""
n_uncertainty(p::LoadProcess) = 1 + p.nregion

"""
    natom(process) -> Int

Number of atoms in the finite support.
"""
natom(p::LoadProcess) = length(p.atoms)

# ── Deterministic, price-free region assignment ───────────────────────────────

"""
    _bus_adjacency(nd) -> Vector{Vector{Int}}

Undirected adjacency list over bus ARRAY POSITIONS built from the in-service
branches of `nd` (self-loops and duplicates are harmless for BFS). Used only for
the topology-derived region assignment; carries no electrical weighting.
"""
function _bus_adjacency(nd::NetworkData)
    adj = [Int[] for _ in 1:nbus(nd)]
    for br in nd.branches
        push!(adj[br.f_pos], br.t_pos)
        push!(adj[br.t_pos], br.f_pos)
    end
    return adj
end

"""
    _bfs_distances(adj, source) -> Vector{Int}

Unweighted shortest-path (hop) distances from bus position `source` to every bus
over adjacency `adj`. Unreachable buses get `typemax(Int)`.
"""
function _bfs_distances(adj::Vector{Vector{Int}}, source::Int)
    n = length(adj)
    dist = fill(typemax(Int), n)
    dist[source] = 0
    queue = Int[source]
    head = 1
    while head <= length(queue)
        u = queue[head]; head += 1
        du = dist[u]
        for v in adj[u]
            if dist[v] == typemax(Int)
                dist[v] = du + 1
                push!(queue, v)
            end
        end
    end
    return dist
end

"""
    assign_regions(nd, nregion) -> (region_of_bus, anchor_bus_ids)

Partition the buses of `nd` into `nregion` connected regions using a
deterministic, physically defensible, price-free topology rule:

1. anchor 1 is the highest-degree bus (ties broken by smallest original id);
2. each further anchor is the bus whose minimum hop-distance to the already
   chosen anchors is largest (farthest-point sampling; ties by smallest id), so
   anchors are spread across the network;
3. every bus is assigned to its nearest anchor by hop distance (ties broken by
   anchor order); buses unreachable from all anchors fall back to region 1.

Returns the length-`nbus` region vector (over array positions) and the anchor
original bus ids in region order.

Depends only on branch topology and bus ids — never on generator prices, so the
region geography is stable under any price change (see the plan's prohibition on
price-derived case design).
"""
function assign_regions(nd::NetworkData, nregion::Int)
    nb = nbus(nd)
    (1 <= nregion <= nb) ||
        error("nregion must satisfy 1 ≤ nregion ≤ nbus=$nb; got $nregion")
    adj = _bus_adjacency(nd)
    deg = [length(a) for a in adj]

    # Anchor 1: highest degree, ties → smallest original id.
    order = sortperm(1:nb; by = p -> (-deg[p], nd.buses[p].id))
    anchors = Int[order[1]]
    # Farthest-point sampling for the remaining anchors.
    mindist = _bfs_distances(adj, anchors[1])
    while length(anchors) < nregion
        # Pick the bus maximizing distance to the nearest existing anchor; break
        # ties by smallest original id. Cap unreachable (typemax) at nb+1 so an
        # isolated component does not silently win every round.
        best = 0; best_key = (typemin(Int), 0)
        for p in 1:nb
            p in anchors && continue
            d = mindist[p] == typemax(Int) ? nb + 1 : mindist[p]
            key = (d, -nd.buses[p].id)
            if key > best_key
                best_key = key; best = p
            end
        end
        push!(anchors, best)
        dnew = _bfs_distances(adj, best)
        @inbounds for p in 1:nb
            mindist[p] = min(mindist[p], dnew[p])
        end
    end

    # Assign each bus to its nearest anchor (ties → lower region index).
    anchor_dists = [_bfs_distances(adj, a) for a in anchors]
    region_of_bus = Vector{Int}(undef, nb)
    for p in 1:nb
        best_r = 1; best_d = anchor_dists[1][p]
        for r in 2:nregion
            if anchor_dists[r][p] < best_d
                best_d = anchor_dists[r][p]; best_r = r
            end
        end
        # Unreachable from every anchor ⇒ region 1 (deterministic fallback).
        region_of_bus[p] = best_d == typemax(Int) ? 1 : best_r
    end
    anchor_bus_ids = [nd.buses[a].id for a in anchors]
    return region_of_bus, anchor_bus_ids
end

# ── Default building blocks ───────────────────────────────────────────────────

"""
    default_base_shape(period; amplitude=0.15) -> Vector{Float64}

A smooth deterministic time-of-day shape of length `period`, one sinusoidal peak
per period with mean 1:

`base_shape[h] = 1 + amplitude·sin(2π(h - 1)/period − π/2)`  (min at h=1, peak at
mid-period). `amplitude` (default 0.15) is the peak fractional deviation. Values
stay in `[1 − amplitude, 1 + amplitude] > 0`.
"""
function default_base_shape(period::Int; amplitude::Real = 0.15)
    period >= 1 || error("period must be ≥ 1; got $period")
    (0 <= amplitude < 1) || error("amplitude must satisfy 0 ≤ amplitude < 1; got $amplitude")
    return [1.0 + amplitude * sin(2pi * (h - 1) / period - pi / 2) for h in 1:period]
end

"""
    default_load_atoms(nregion; high_factor=1.05, scarce_factor=1.15,
                       off_scarce_factor=1.0, calm_prob=0.4) -> (atoms, probs)

Build the default finite support: `1 + nregion` atoms.

* atom 1 (`calm`)         : `L = 1`, every region multiplier `= 1`; probability
  `calm_prob`.
* atom `1+k` (`scarce_k`) : `L = high_factor`, region `k` multiplier
  `= scarce_factor`, all other regions `= off_scarce_factor`; probability
  `(1 − calm_prob)/nregion` each.

A small, SDDP-friendly support in which scarcity MOVES between regions across
atoms (the mechanism a later phase exploits) while active/reactive power factor
is preserved. No factor depends on generator prices.
"""
function default_load_atoms(nregion::Int;
                            high_factor::Real = 1.05,
                            scarce_factor::Real = 1.15,
                            off_scarce_factor::Real = 1.0,
                            calm_prob::Real = 0.4)
    nregion >= 1 || error("nregion must be ≥ 1; got $nregion")
    (0 < calm_prob < 1) || error("calm_prob must be in (0,1); got $calm_prob")
    atoms = LoadAtom[LoadAtom(1.0, ones(Float64, nregion))]
    for k in 1:nregion
        R = fill(Float64(off_scarce_factor), nregion)
        R[k] = Float64(scarce_factor)
        push!(atoms, LoadAtom(Float64(high_factor), R))
    end
    each = (1.0 - calm_prob) / nregion
    probs = vcat(Float64(calm_prob), fill(each, nregion))
    return atoms, probs
end

# ── Construction ──────────────────────────────────────────────────────────────

"""
    make_load_process(case; kwargs...) -> LoadProcess

Build the seeded finite-support demand process for a [`BatteryCase`].

# Keywords
- `nregion::Int = 3`: number of demand regions.
- `period::Int = 24`: base-shape period in stages.
- `base_amplitude::Real = 0.15`: peak fractional deviation of the base shape.
- `train_seed::Int = 20260722`: declared training seed.
- `eval_seed::Int = 970122`: declared evaluation seed (must differ from
  `train_seed`).
- `atoms`, `probs`: optional explicit finite support; by default
  [`default_load_atoms`](@ref)`(nregion)` is used.
- `base_shape`: optional explicit base shape (length `period`).

All parameters are validated: probabilities must be nonnegative, finite, and sum
to 1; atom regional-factor vectors must have length `nregion`; seeds must differ;
factors must be finite and positive.
"""
function make_load_process(case::BatteryCase;
                           preset::Symbol = DEFAULT_DEMAND_PRESET,
                           nregion::Int = 3,
                           period::Int = 24,
                           base_amplitude::Union{Nothing,Real} = nothing,
                           calm_prob::Real = 0.4,
                           train_seed::Int = 20260722,
                           eval_seed::Int = 970122,
                           atoms::Union{Nothing,AbstractVector{LoadAtom}} = nothing,
                           probs::Union{Nothing,AbstractVector{<:Real}} = nothing,
                           base_shape::Union{Nothing,AbstractVector{<:Real}} = nothing)
    nd = case.network
    region_of_bus, anchor_bus_ids = assign_regions(nd, nregion)

    # Resolve the calibration preset. Explicit `atoms`/`base_amplitude` override
    # it and mark the process `:custom` so the manifest never claims a preset the
    # numbers do not match.
    haskey(DEMAND_PRESETS, preset) || preset === :custom ||
        error("unknown demand preset :$preset; known: $(sort(collect(keys(DEMAND_PRESETS))))")
    cal = get(DEMAND_PRESETS, preset, DEMAND_PRESETS[DEFAULT_DEMAND_PRESET])
    amp = base_amplitude === nothing ? cal.base_amplitude : Float64(base_amplitude)
    resolved_preset = (atoms !== nothing || base_shape !== nothing ||
                       base_amplitude !== nothing) ? :custom : preset

    shape = base_shape === nothing ? default_base_shape(period; amplitude = amp) :
            Float64.(collect(base_shape))
    length(shape) == period ||
        error("base_shape must have length period=$period; got $(length(shape))")
    all(x -> isfinite(x) && x > 0, shape) ||
        error("base_shape entries must be finite and > 0")

    if atoms === nothing
        atoms_v, probs_v = default_load_atoms(nregion;
            high_factor = cal.high_factor,
            scarce_factor = cal.scarce_factor,
            off_scarce_factor = cal.off_scarce_factor,
            calm_prob = calm_prob)
    else
        atoms_v = collect(LoadAtom, atoms)
        probs_v = probs === nothing ?
            error("explicit atoms require explicit probs") : Float64.(collect(probs))
    end

    _validate_support(atoms_v, probs_v, nregion)
    train_seed != eval_seed ||
        error("train_seed and eval_seed must differ (got $train_seed for both)")

    return LoadProcess(nregion, region_of_bus, anchor_bus_ids,
                       "nearest_graph_anchor_farthest_point", shape, period,
                       atoms_v, probs_v, Int(train_seed), Int(eval_seed),
                       resolved_preset)
end

"""
    load_process_from_fields(; nregion, region_of_bus, anchor_bus_ids, region_rule,
                             base_shape, period, atoms, probs, train_seed,
                             eval_seed, preset) -> LoadProcess

Rebuild a [`LoadProcess`](@ref) from EXACT stored field values, with no defaults
and no recomputation (no re-derivation of `base_shape` from an amplitude, no
re-running of the region assignment). This is the reconstruction path used by
the stochastic manifest so a restored process is bit-identical and reproduces its
recorded hash. The support is revalidated before the process is returned.
"""
function load_process_from_fields(; nregion::Integer,
                                  region_of_bus::AbstractVector{<:Integer},
                                  anchor_bus_ids::AbstractVector{<:Integer},
                                  region_rule::AbstractString,
                                  base_shape::AbstractVector{<:Real},
                                  period::Integer,
                                  atoms::AbstractVector{LoadAtom},
                                  probs::AbstractVector{<:Real},
                                  train_seed::Integer, eval_seed::Integer,
                                  preset::Union{Symbol,AbstractString})
    atoms_v = collect(LoadAtom, atoms); probs_v = Float64.(collect(probs))
    _validate_support(atoms_v, probs_v, Int(nregion))
    length(base_shape) == period ||
        error("base_shape length $(length(base_shape)) ≠ period $period")
    return LoadProcess(Int(nregion), Int.(collect(region_of_bus)),
                       Int.(collect(anchor_bus_ids)), String(region_rule),
                       Float64.(collect(base_shape)), Int(period),
                       atoms_v, probs_v, Int(train_seed), Int(eval_seed),
                       Symbol(preset))
end

"""
    demand_multiplier_summary(process) -> NamedTuple

Peak demand multipliers implied by the process, reported by the calibration
gates:

* `max_system_multiplier` — `max_t max_a base_shape[t]·L^a`, the largest
  system-wide scaling of nominal demand;
* `max_bus_multiplier` — `max_t max_a max_r base_shape[t]·L^a·R_r^a`, the largest
  scaling seen by any individual bus (the quantity that actually decides
  servability);
* `min_bus_multiplier` — the corresponding minimum.
"""
function demand_multiplier_summary(process::LoadProcess)
    max_sys = -Inf; max_bus = -Inf; min_bus = Inf
    for h in process.base_shape, a in process.atoms
        s = h * a.system_factor
        max_sys = max(max_sys, s)
        for r in a.regional_factors
            max_bus = max(max_bus, s * r); min_bus = min(min_bus, s * r)
        end
    end
    return (max_system_multiplier = max_sys,
            max_bus_multiplier = max_bus,
            min_bus_multiplier = min_bus)
end

"""
    _validate_support(atoms, probs, nregion)

Validate a finite support: matching lengths, correct regional-factor
dimensions, finite positive factors, and probabilities that are nonnegative,
finite, and sum to 1 (within 1e-9). Throws on the first violation.
"""
function _validate_support(atoms::AbstractVector{LoadAtom}, probs::AbstractVector{<:Real}, nregion::Int)
    length(atoms) == length(probs) ||
        error("atoms ($(length(atoms))) and probs ($(length(probs))) length mismatch")
    isempty(atoms) && error("finite support must contain at least one atom")
    for (a, at) in enumerate(atoms)
        length(at.regional_factors) == nregion ||
            error("atom $a has $(length(at.regional_factors)) regional factors, expected nregion=$nregion")
        (isfinite(at.system_factor) && at.system_factor > 0) ||
            error("atom $a has non-positive/non-finite system_factor $(at.system_factor)")
        all(x -> isfinite(x) && x > 0, at.regional_factors) ||
            error("atom $a has a non-positive/non-finite regional factor")
    end
    all(p -> isfinite(p) && p >= 0, probs) ||
        error("atom probabilities must be finite and ≥ 0")
    s = sum(probs)
    abs(s - 1) <= 1e-9 || error("atom probabilities must sum to 1; got $s")
    return nothing
end

# ── Scenario index generation (stage-major) ───────────────────────────────────

"""
    scenario_index_matrix(process, horizon, paths; seed) -> Matrix{Int}

Draw a `horizon × paths` matrix of atom indices (each in `1:natom(process)`),
i.i.d. across stages and paths according to `process.probs`, using
`StableRNG(seed)`.

Ordering is **stage-major**: entry `[t, p]` is the atom of stage `t` on path `p`,
column `p` being one full scenario. The same `(process, horizon, paths, seed)`
always produces the identical matrix (and hence the identical protocol hash).
"""
function scenario_index_matrix(process::LoadProcess, horizon::Int, paths::Int; seed::Integer)
    horizon >= 1 || error("horizon must be ≥ 1; got $horizon")
    paths >= 1 || error("paths must be ≥ 1; got $paths")
    rng = StableRNG(UInt64(unsigned(Int64(seed))))
    A = natom(process)
    # Precompute the cumulative distribution once; sample by inverse-CDF so the
    # draw shape (one Float64 per (stage,path)) is fixed and never varies with A.
    cdf = cumsum(process.probs)
    cdf[end] = 1.0  # guard against fp drift so u ≤ cdf[end] always hits an atom
    idx = Matrix{Int}(undef, horizon, paths)
    for p in 1:paths, t in 1:horizon      # column-major loop matches stage-major storage
        u = rand(rng)
        a = 1
        @inbounds while a < A && u > cdf[a]
            a += 1
        end
        idx[t, p] = a
    end
    return idx
end

# ── Materialization: atom indices → realized w_flat (stage-major) ─────────────

"""
    materialize_scenario(process, index_row; horizon=length(index_row)) -> Vector{Float64}

Turn one scenario's atom-index column (`index_row`, length `horizon`) into the
flat per-stage uncertainty vector `w_flat` of length `horizon·nw`, stage-major:
for stage `t` the block is `[ s_t , r_{1,t} , … , r_{R,t} ]` with
`s_t = base_shape[t]·L^{a_t}` and `r_{k,t} = R_k^{a_t}`.

`index_row` may be a vector or a matrix column view. Every index must be a valid
atom (`1:natom`).
"""
function materialize_scenario(process::LoadProcess, index_row::AbstractVector{<:Integer};
                              horizon::Int = length(index_row))
    length(index_row) == horizon ||
        error("index_row length $(length(index_row)) ≠ horizon $horizon")
    nw = n_uncertainty(process)
    A = natom(process)
    P = process.period
    w = Vector{Float64}(undef, horizon * nw)
    for t in 1:horizon
        a = Int(index_row[t])
        (1 <= a <= A) || error("atom index $a at stage $t out of range 1:$A")
        atom = process.atoms[a]
        base = process.base_shape[((t - 1) % P) + 1]
        off = (t - 1) * nw
        w[off + 1] = base * atom.system_factor
        @inbounds for k in 1:process.nregion
            w[off + 1 + k] = atom.regional_factors[k]
        end
    end
    return w
end

"""
    materialize_all(process, index_matrix) -> Vector{Vector{Float64}}

Materialize every column of a stage-major `index_matrix` into its `w_flat`,
returning one vector per path (the paired-evaluation scenario set).
"""
function materialize_all(process::LoadProcess, index_matrix::AbstractMatrix{<:Integer})
    horizon = size(index_matrix, 1)
    return [materialize_scenario(process, view(index_matrix, :, p); horizon = horizon)
            for p in 1:size(index_matrix, 2)]
end

# ── Serialization: the scenario protocol ──────────────────────────────────────

"""
    process_canonical_content(process) -> String

Ordered, human-inspectable string capturing every scientific parameter of the
demand process: regions and their anchors, the base shape, the atoms and
probabilities, the per-stage dimension, and the declared seeds. Its SHA-256 is
[`process_hash`](@ref). The process is fully deterministic given this content.
"""
function process_canonical_content(process::LoadProcess)
    io = IOBuffer()
    println(io, "schema=battery_load_process/2")
    println(io, "preset=", process.preset)
    println(io, "nregion=", process.nregion)
    println(io, "region_rule=", process.region_rule)
    println(io, "anchor_bus_ids=", join(process.anchor_bus_ids, ","))
    println(io, "region_of_bus=", join(process.region_of_bus, ","))
    println(io, "period=", process.period)
    println(io, "base_shape=", join(_fmt.(process.base_shape), ","))
    println(io, "nw=", n_uncertainty(process))
    println(io, "natom=", natom(process))
    println(io, "train_seed=", process.train_seed)
    println(io, "eval_seed=", process.eval_seed)
    for (a, atom) in enumerate(process.atoms)
        println(io, "atom ", a,
                    " prob=", _fmt(process.probs[a]),
                    " L=", _fmt(atom.system_factor),
                    " R=", join(_fmt.(atom.regional_factors), ","))
    end
    return String(take!(io))
end

"""
    process_hash(process) -> String

Lowercase hex SHA-256 of [`process_canonical_content`](@ref). Identical process
parameters (regions, base shape, atoms, probabilities, seeds) reproduce the same
hash.
"""
process_hash(process::LoadProcess) = bytes2hex(sha256(process_canonical_content(process)))

"""
    index_matrix_hash(index_matrix) -> String

Lowercase hex SHA-256 of a stage-major atom-index matrix, computed from its
shape and column-major integer contents so it is stable across runs.
"""
function index_matrix_hash(index_matrix::AbstractMatrix{<:Integer})
    io = IOBuffer()
    print(io, size(index_matrix, 1), "x", size(index_matrix, 2), ":")
    for v in index_matrix           # column-major traversal, deterministic
        print(io, Int(v), ",")
    end
    return bytes2hex(sha256(take!(io)))
end

"""
    write_scenario_protocol(path, process, index_matrix; kind, seed,
                            extra=Dict()) -> String

Serialize a paired-scenario protocol to JSON at `path`: all process parameters,
the stage-major atom-index matrix (as nested arrays), the generating `seed` and
`kind` (`"train"`/`"eval"`), the per-stage dimension and units, and the process
and index-matrix hashes. Returns `path`.

The stored index matrix IS the paired protocol: every method materializes it
with [`materialize_scenario`](@ref) to obtain byte-identical `w_flat`s.
"""
function write_scenario_protocol(path::AbstractString, process::LoadProcess,
                                 index_matrix::AbstractMatrix{<:Integer};
                                 kind::AbstractString, seed::Integer,
                                 extra::AbstractDict = Dict{String,Any}())
    horizon, paths = size(index_matrix)
    doc = Dict{String,Any}(
        "schema" => "battery_load_process/1",
        "kind" => String(kind),
        "seed" => Int(seed),
        "horizon" => horizon,
        "paths" => paths,
        "nw_per_stage" => n_uncertainty(process),
        "process_hash_sha256" => process_hash(process),
        "index_matrix_hash_sha256" => index_matrix_hash(index_matrix),
        "process" => Dict{String,Any}(
            "preset" => String(process.preset),
            "nregion" => process.nregion,
            "region_rule" => process.region_rule,
            "anchor_bus_ids" => process.anchor_bus_ids,
            "region_of_bus" => process.region_of_bus,
            "period" => process.period,
            "base_shape" => process.base_shape,
            "train_seed" => process.train_seed,
            "eval_seed" => process.eval_seed,
            "atoms" => [Dict("prob" => process.probs[a],
                             "system_factor" => process.atoms[a].system_factor,
                             "regional_factors" => process.atoms[a].regional_factors)
                        for a in 1:natom(process)],
        ),
        # Stage-major: outer list = stages, inner list = per-path atom indices.
        "index_matrix" => [Int.(index_matrix[t, :]) for t in 1:horizon],
        "units" => "atom indices (1-based); w_t = [base_shape·L, R_1..R_nregion]",
    )
    for (k, v) in extra
        doc[k] = v
    end
    open(io -> JSON.print(io, doc, 2), path, "w")
    return path
end

"""
    reconstruct_scenario_protocol(path) -> (process, index_matrix, meta)

Rebuild a demand process and its stage-major atom-index matrix from a JSON
protocol written by [`write_scenario_protocol`](@ref), verifying BOTH recorded
hashes:

1. the reconstructed process reproduces `process_hash_sha256`;
2. the reconstructed index matrix reproduces `index_matrix_hash_sha256`.

Any mismatch raises an error. `meta` is a NamedTuple with `kind`, `seed`,
`horizon`, `paths`, and `nw_per_stage`.
"""
function reconstruct_scenario_protocol(path::AbstractString)
    doc = JSON.parsefile(path)
    pr = doc["process"]
    nregion = Int(pr["nregion"])
    atoms = LoadAtom[LoadAtom(Float64(a["system_factor"]),
                              Float64.(a["regional_factors"])) for a in pr["atoms"]]
    probs = Float64[Float64(a["prob"]) for a in pr["atoms"]]
    # EXACT-field reconstruction: no defaults, no recomputation of base_shape or
    # of the region assignment.
    process = load_process_from_fields(;
        nregion = nregion,
        region_of_bus = Int.(pr["region_of_bus"]),
        anchor_bus_ids = Int.(pr["anchor_bus_ids"]),
        region_rule = String(pr["region_rule"]),
        base_shape = Float64.(pr["base_shape"]),
        period = Int(pr["period"]),
        atoms = atoms, probs = probs,
        train_seed = Int(pr["train_seed"]), eval_seed = Int(pr["eval_seed"]),
        preset = get(pr, "preset", "custom"))

    got_ph = process_hash(process)
    want_ph = String(doc["process_hash_sha256"])
    got_ph == want_ph ||
        error("process-hash mismatch: protocol recorded $want_ph but rebuilt $got_ph")

    horizon = Int(doc["horizon"]); paths = Int(doc["paths"])
    rows = doc["index_matrix"]
    length(rows) == horizon || error("index_matrix has $(length(rows)) stages, expected $horizon")
    index_matrix = Matrix{Int}(undef, horizon, paths)
    for t in 1:horizon
        row = rows[t]
        length(row) == paths || error("stage $t has $(length(row)) paths, expected $paths")
        @inbounds for p in 1:paths
            index_matrix[t, p] = Int(row[p])
        end
    end
    got_ih = index_matrix_hash(index_matrix)
    want_ih = String(doc["index_matrix_hash_sha256"])
    got_ih == want_ih ||
        error("index-matrix-hash mismatch: protocol recorded $want_ih but rebuilt $got_ih")

    meta = (kind = String(doc["kind"]), seed = Int(doc["seed"]),
            horizon = horizon, paths = paths,
            nw_per_stage = Int(doc["nw_per_stage"]))
    return process, index_matrix, meta
end
