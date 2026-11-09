# hydro_power_data.jl
#
# Parses a HydroPowerModels case from standard input files:
#   PowerModels.json  — power system topology (PowerModels.jl JSON format)
#   hydro.json        — hydro unit parameters
#   inflows.csv       — inflow scenario data (stages × scenarios per hydro)
#   demand.csv        — optional per-stage bus demands (stages × loads)
#
# The power system part uses the same data conventions as ExaModelsPower.jl
# (DC-OPF susceptance formula, branch-variable formulation).

using JSON, CSV, Tables, Statistics, Random
using StableRNGs   # seeded demand-noise draws for the paired eval protocol

# ── Power system data structures ─────────────────────────────────────────────

struct PowerBusData
    idx::Int        # bus index (1-based, same as position for Bolivia)
    bus_type::Int   # 1=PQ, 2=PV, 3=REF/swing
    gs::Float64     # shunt conductance (pu)
    bs::Float64     # shunt susceptance (pu)
    vmin::Float64   # minimum voltage magnitude (pu)
    vmax::Float64   # maximum voltage magnitude (pu)
end

struct PowerGenData
    idx::Int        # generator index
    bus::Int        # bus index this generator is connected to
    pmin::Float64   # minimum active power (pu)
    pmax::Float64   # maximum active power (pu)
    qmin::Float64   # minimum reactive power (pu)
    qmax::Float64   # maximum reactive power (pu)
    cost1::Float64  # linear cost coefficient ($/pu·h)
    cost2::Float64  # quadratic cost coefficient ($/pu²·h); 0 for DC linear
end

struct PowerBranchData
    idx::Int        # branch index
    f_bus::Int      # from-bus index
    t_bus::Int      # to-bus index
    b::Float64      # DC susceptance = −br_x / (br_r² + br_x²)  [ExaModelsPower formula]
    rate_a::Float64 # thermal limit (pu); pf ∈ [−rate_a, rate_a]
    angmin::Float64 # minimum angle difference (rad)
    angmax::Float64 # maximum angle difference (rad)
    # AC parameters (raw MATPOWER/PowerModels fields)
    br_r::Float64   # resistance
    br_x::Float64   # reactance
    g_fr::Float64   # from-bus shunt conductance
    b_fr::Float64   # from-bus shunt susceptance
    g_to::Float64   # to-bus shunt conductance
    b_to::Float64   # to-bus shunt susceptance
    tap::Float64    # transformer turns ratio (1.0 = no transformer)
    shift::Float64  # transformer phase shift (rad)
end

struct PowerLoadData
    idx::Int   # load index in PowerModels.json
    bus::Int   # bus this load belongs to
end

"""
    PowerData

Parsed power system data from PowerModels.json.

- `buses`, `gens`, `branches`, `loads` are sorted by index.
- Bus indices are 1-based and consecutive for Bolivia (index == position).
- `ref_buses` holds bus positions (indices) of swing/reference buses (type 3).
- `cost_deficit` is the load-shedding penalty in \$/pu·h.
"""
struct PowerData
    nBus::Int
    nGen::Int
    nBranch::Int
    nLoad::Int
    buses::Vector{PowerBusData}
    gens::Vector{PowerGenData}
    branches::Vector{PowerBranchData}
    loads::Vector{PowerLoadData}
    ref_buses::Vector{Int}     # bus indices of reference buses
    baseMVA::Float64
    cost_deficit::Float64
    # Default per-bus demand (pu) from PowerModels.json load.pd / load.qd
    default_bus_demand::Vector{Float64}           # length nBus (active)
    default_bus_reactive_demand::Vector{Float64}  # length nBus (reactive)
end

# ── Hydro data structures ─────────────────────────────────────────────────────

struct HydroUnitData
    pos::Int        # 1-based position in hydro array (matches hydro.json order)
    gen_pos::Int    # position of this unit's generator in power_data.gens (sorted by gen index)
    max_vol::Float64
    min_vol::Float64
    max_turn::Float64   # maximum turbine outflow (m³/s equivalent)
    min_turn::Float64
    pf::Float64         # production factor (turbine coupling: baseMVA·pg = pf·outflow)
    spill_cost::Float64
    min_out_cost::Float64
    min_vol_cost::Float64
end

struct UpstreamTurn
    downstream_pos::Int   # position of the downstream hydro unit
    upstream_pos::Int     # position of the upstream hydro whose OUTFLOW contributes
end

struct UpstreamSpill
    downstream_pos::Int
    upstream_pos::Int     # position of the upstream hydro whose SPILL contributes
end

"""
    HydroData

Parsed hydro-unit data from hydro.json + inflows.csv.

`scenario_inflows[r]` is a (nStagesSample × nScenarios) matrix for hydro unit r.
"""
struct HydroData
    nHyd::Int
    units::Vector{HydroUnitData}
    upstream_turns::Vector{UpstreamTurn}
    upstream_spills::Vector{UpstreamSpill}
    K::Float64                             # water-balance conversion factor
    initial_volumes::Vector{Float64}       # length nHyd
    scenario_inflows::Vector{Matrix{Float64}}
    nScenarios::Int
    nStagesSample::Int
end

# ── Power data loader ─────────────────────────────────────────────────────────

"""
    load_power_data(pm_file) -> PowerData

Parse a PowerModels.jl JSON file into `PowerData`.
Includes both DC (b susceptance) and AC (br_r, br_x, g_fr, b_fr, tap, shift, etc.)
fields for use with either formulation.
"""
function load_power_data(pm_file::AbstractString)
    pm = JSON.parsefile(pm_file)

    baseMVA      = Float64(pm["baseMVA"])
    cost_deficit = Float64(get(pm, "cost_deficit", 0.0))

    # ── Buses ────────────────────────────────────────────────────────────────
    buses_raw = [(pm["bus"][k]["index"],
                  pm["bus"][k]["bus_type"],
                  Float64(get(pm["bus"][k], "gs", 0.0)),
                  Float64(get(pm["bus"][k], "bs", 0.0)),
                  Float64(get(pm["bus"][k], "vmin", 0.9)),
                  Float64(get(pm["bus"][k], "vmax", 1.1)))
                 for k in keys(pm["bus"])]
    sort!(buses_raw, by = x -> x[1])
    buses = [PowerBusData(b[1], b[2], b[3], b[4], b[5], b[6]) for b in buses_raw]
    nBus = length(buses)

    ref_buses = [b.idx for b in buses if b.bus_type == 3]

    # ── Generators (cost format: ncost=2 → [c1,c0]; ncost=3 → [c2,c1,c0]) ──
    gens_raw = Tuple[]
    for k in keys(pm["gen"])
        g = pm["gen"][k]
        cost_arr = Float64.(g["cost"])
        ncost = Int(g["ncost"])
        c1 = ncost >= 2 ? cost_arr[ncost == 2 ? 1 : 2] : 0.0   # linear coefficient
        c2 = ncost >= 3 ? cost_arr[1] : 0.0                      # quadratic coefficient
        push!(gens_raw, (g["index"], g["gen_bus"],
                         Float64(get(g, "pmin", 0.0)), Float64(g["pmax"]),
                         Float64(get(g, "qmin", -Inf)), Float64(get(g, "qmax", Inf)),
                         c1, c2))
    end
    sort!(gens_raw, by = x -> x[1])
    gens = [PowerGenData(g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8]) for g in gens_raw]
    nGen = length(gens)

    # ── Branches ─────────────────────────────────────────────────────────────
    brs_raw = Tuple[]
    for k in keys(pm["branch"])
        br = pm["branch"][k]
        br_r = Float64(get(br, "br_r", 0.0))
        br_x = Float64(br["br_x"])
        r2x2 = br_r^2 + br_x^2
        b_val = r2x2 > 0 ? -br_x / r2x2 : 0.0   # DC susceptance (ExaModelsPower formula)
        tap_val   = Float64(get(br, "tap",   1.0)); tap_val = tap_val ≈ 0 ? 1.0 : tap_val
        shift_val = Float64(get(br, "shift", 0.0))
        push!(brs_raw, (br["index"], br["f_bus"], br["t_bus"],
                        b_val,
                        Float64(get(br, "rate_a", Inf)),
                        Float64(get(br, "angmin", -π)),
                        Float64(get(br, "angmax",  π)),
                        br_r, br_x,
                        Float64(get(br, "g_fr", 0.0)),
                        Float64(get(br, "b_fr", 0.0)),
                        Float64(get(br, "g_to", 0.0)),
                        Float64(get(br, "b_to", 0.0)),
                        tap_val, shift_val))
    end
    sort!(brs_raw, by = x -> x[1])
    branches = [PowerBranchData(b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                                b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15])
                for b in brs_raw]
    nBranch = length(branches)

    # ── Loads ─────────────────────────────────────────────────────────────────
    loads_raw = Tuple[]
    for k in keys(pm["load"])
        l = pm["load"][k]
        push!(loads_raw, (l["index"], l["load_bus"],
                          Float64(get(l, "pd", 0.0)),
                          Float64(get(l, "qd", 0.0))))
    end
    sort!(loads_raw, by = x -> x[1])
    loads = [PowerLoadData(l[1], l[2]) for l in loads_raw]
    nLoad = length(loads)

    # Default per-bus demand: sum loads at each bus from PowerModels.json
    default_bus_demand          = zeros(Float64, nBus)
    default_bus_reactive_demand = zeros(Float64, nBus)
    for (l_idx, l_bus, l_pd, l_qd) in loads_raw
        default_bus_demand[l_bus]          += l_pd
        default_bus_reactive_demand[l_bus] += l_qd
    end

    return PowerData(nBus, nGen, nBranch, nLoad,
                     buses, gens, branches, loads,
                     ref_buses, baseMVA, cost_deficit,
                     default_bus_demand, default_bus_reactive_demand)
end

# ── Hydro data loader ─────────────────────────────────────────────────────────

"""
    load_hydro_data(hydro_file, inflows_file, power_data; num_stages=nothing)
              -> HydroData

Parse `hydro.json` and `inflows.csv` into `HydroData`.

`power_data` is needed to map `index_grid` (generator index) to `gen_pos`
(position in `power_data.gens`).
"""
function load_hydro_data(hydro_file::AbstractString,
                          inflows_file::AbstractString,
                          power_data::PowerData;
                          num_stages::Union{Int,Nothing} = nothing)

    hydro_root = JSON.parsefile(hydro_file)
    hydro_json = hydro_root["Hydrogenerators"]
    nHyd = length(hydro_json)

    # Build gen_index → gen_pos map
    gen_idx_to_pos = Dict(g.idx => pos for (pos, g) in enumerate(power_data.gens))

    # Build hydro.index → hydro_pos map (for upstream connections)
    hydro_idx_to_pos = Dict(Int(h["index"]) => pos for (pos, h) in enumerate(hydro_json))

    initial_volumes = [Float64(h["initial_volume"]) for h in hydro_json]

    # HydroUnitData
    units = HydroUnitData[]
    for (pos, h) in enumerate(hydro_json)
        ig = Int(h["index_grid"])
        gen_pos = get(gen_idx_to_pos, ig, -1)
        gen_pos > 0 || @warn "Hydro unit $pos ($(h["name"])): index_grid=$ig not found in generators"
        push!(units, HydroUnitData(
            pos, gen_pos,
            Float64(h["max_volume"]), Float64(h["min_volume"]),
            Float64(h["max_turn"]),   Float64(h["min_turn"]),
            Float64(h["production_factor"]),
            Float64(h["spill_cost"]),
            Float64(get(h, "minimal_outflow_violation_cost", 0.0)),
            Float64(get(h, "minimal_volume_violation_cost", 0.0)),
        ))
    end

    # Upstream connections: downstream_turn[i] uses hydro.index values
    upstream_turns  = UpstreamTurn[]
    upstream_spills = UpstreamSpill[]
    for (pos, h) in enumerate(hydro_json)
        for ds_idx in h["downstream_turn"]
            ds_pos = get(hydro_idx_to_pos, Int(ds_idx), nothing)
            ds_pos !== nothing && push!(upstream_turns, UpstreamTurn(ds_pos, pos))
        end
        for ds_idx in h["downstream_spill"]
            ds_pos = get(hydro_idx_to_pos, Int(ds_idx), nothing)
            ds_pos !== nothing && push!(upstream_spills, UpstreamSpill(ds_pos, pos))
        end
    end

    # Inflows
    allinflows = CSV.read(inflows_file, Tables.matrix; header=false)
    nrows, ncols = size(allinflows)
    nScenarios   = div(ncols, nHyd)
    nStagesSample = isnothing(num_stages) ? nrows : num_stages

    if !isnothing(num_stages) && num_stages > nrows
        repeats   = div(num_stages, nrows) + 1
        allinflows = vcat([allinflows for _ in 1:repeats]...)
    end
    allinflows = allinflows[1:nStagesSample, :]

    scenario_inflows = Vector{Matrix{Float64}}(undef, nHyd)
    for r in 1:nHyd
        scenario_inflows[r] = Float64.(allinflows[:, ((r-1)*nScenarios+1):(r*nScenarios)])
    end

    # Water-balance conversion factor K = 0.0036 · stage_hours.
    # 0.0036 converts a flow of m³/s to reservoir volume (hm³) accumulated over one
    # HOUR (3600 s/h · 1e-6 hm³/m³ = 0.0036). Multiplying by the stage duration
    # `stage_hours` (hours per stage) gives the per-stage flow→volume factor, so the
    # reservoir balance `V_{t+1} = V_t + K·(inflow − outflow) − spill(+upstream)`
    # is dimensionally correct for a `stage_hours`-long stage.
    # `stage_hours` is read from hydro.json and defaults to 1 (⇒ K = 0.0036) for
    # backward compatibility with cases predating the field. This exactly mirrors
    # HydroPowerModels.jl `constraint_hydro_balance` (k = 0.0036, coefficient
    # k · params["stage_hours"]) so the Exa and JuMP engines share one water balance.
    stage_hours = Int(get(hydro_root, "stage_hours", 1))
    K = 0.0036 * stage_hours

    return HydroData(nHyd, units, upstream_turns, upstream_spills,
                     K, initial_volumes, scenario_inflows, nScenarios, nStagesSample)
end

# ── Demand loader ─────────────────────────────────────────────────────────────

"""
    load_demand(demand_file, power_data; T=nothing) -> Matrix{Float64}

Load demand from a CSV file (rows = stages, cols = load indices in order).
Returns a [T × nBus] matrix of per-bus demands (pu).

If `T` is specified, truncates or repeats rows to match T stages.
If the file has only 1 row, it is repeated for all stages.
"""
function load_demand(demand_file::AbstractString,
                     power_data::PowerData;
                     T::Union{Int,Nothing} = nothing)
    raw = CSV.read(demand_file, Tables.matrix; header=false)
    nrows, ncols = size(raw)
    nLoad = power_data.nLoad
    ncols == nLoad || @warn "demand file has $ncols cols but nLoad=$nLoad; using min"

    nT = isnothing(T) ? nrows : T
    # Repeat rows if needed
    if nrows < nT
        repeats = div(nT, nrows) + 1
        raw = vcat([raw for _ in 1:repeats]...)
    end
    raw = raw[1:nT, :]

    # Map load columns to buses
    bus_demand = zeros(Float64, nT, power_data.nBus)
    for (j, load) in enumerate(power_data.loads)
        j > ncols && break
        for t in 1:nT
            bus_demand[t, load.bus] += Float64(raw[t, j])
        end
    end
    return bus_demand  # [T × nBus]
end

# ── Scenario sampling ─────────────────────────────────────────────────────────

"""
    sample_scenario(hydro_data, T) -> Vector{Float64}

Sample one inflow trajectory of length `T*nHyd` (flat, stage-major order).

Uses **joint** scenario sampling: at each stage one scenario index `ω` is drawn
and applied to all hydro reservoirs, preserving the spatial correlation present
in the historical inflow data. This matches SDDP's `SDDP.parameterize` semantics.
"""
function sample_scenario(hydro_data::HydroData, T::Int)
    nHyd = hydro_data.nHyd
    w = Vector{Float64}(undef, T * nHyd)
    for t in 1:T
        t_row = mod1(t, hydro_data.nStagesSample)
        # One scenario index per stage — all reservoirs share it (joint sampling).
        j = rand(1:hydro_data.nScenarios)
        for r in 1:nHyd
            w[(t-1)*nHyd + r] = hydro_data.scenario_inflows[r][t_row, j]
        end
    end
    return w
end

"""
    mean_inflow(hydro_data, T) -> Vector{Float64}

Return the mean inflow trajectory over scenarios as a flat T*nHyd vector.
"""
function mean_inflow(hydro_data::HydroData, T::Int)
    nHyd = hydro_data.nHyd
    w = Vector{Float64}(undef, T * nHyd)
    for t in 1:T
        t_row = mod1(t, hydro_data.nStagesSample)
        for r in 1:nHyd
            w[(t-1)*nHyd + r] = mean(hydro_data.scenario_inflows[r][t_row, :])
        end
    end
    return w
end

# ── Stochastic demand (demand_scenarios.csv) ──────────────────────────────────
#
# Demand model for the STOCHASTIC-demand variant of this case (see
# the MAIN repo): an i.i.d. per-stage MULTIPLICATIVE factor on every bus's
# active demand,
#
#     ξ_t ∈ {1 − s, 1, 1 + s},  P = 1/3 each,  independent of the inflow noise,
#
# with the spread s read from `<case>/demand_scenarios.csv` (single line
# `s,<value>`). On the ExaModels side ξ_t travels INSIDE the per-stage
# uncertainty vector: an augmented scenario is stage-major
# `[w_t; ξ_t]` (length T·(nHyd+1)), so the policy observes ξ_t exactly like it
# observes the stage inflow, and `prepare_solve!` applies base_demand·ξ_t to
# the p_demand parameter via `set_demand!` before every solve.

# Seed of the column-keyed demand-noise protocol: eval scenario column c draws
# its demand path from StableRNG(DEMAND_NOISE_SEED + c). Shared by
# train_hydro_exa_strict.jl (protocol eval set) and eval_paired_exa_strict.jl
# (paired-500 protocol), so both see IDENTICAL demand paths per inflow column.
const DEMAND_NOISE_SEED = 20260714

"""
    load_demand_spread(path::AbstractString) -> Union{Float64, Nothing}

Read the demand-noise spread `s` from a `demand_scenarios.csv` file.

The file holds a single data line `s,<value>` (e.g. `s,0.10`) defining the
three-atom multiplicative demand distribution

```math
\\xi_t \\in \\{1 - s,\\; 1,\\; 1 + s\\}, \\qquad P = \\tfrac{1}{3} \\text{ each}.
```

# Arguments
- `path::AbstractString`: path to `demand_scenarios.csv`.

# Returns
- `Float64` spread `s ∈ [0, 1)` when the file exists.
- `nothing` when the file does not exist (deterministic demand — every code
  path then behaves bit-identically to the pre-demand-noise implementation).
"""
function load_demand_spread(path::AbstractString)
    # Missing file ⇒ deterministic demand (backwards-compatible default).
    isfile(path) || return nothing
    # Exactly one non-empty line carries the single `s,<value>` record.
    lines = [strip(line) for line in eachline(path) if !isempty(strip(line))]
    length(lines) == 1 || error(
        "demand_scenarios.csv must contain exactly one non-empty line `s,<value>`; " *
        "found $(length(lines))",
    )
    line = only(lines)
    # Split into the key token and the numeric value.
    parts = split(line, ',')
    # Enforce the exact two-field `s,<value>` format shared by both engines.
    length(parts) == 2 && strip(parts[1]) == "s" ||
        error("demand_scenarios.csv must contain a single line `s,<value>`; got `$line`")
    # Parse the spread value.
    s = parse(Float64, strip(parts[2]))
    # A spread ≥ 1 would make the low atom non-positive demand; forbid it.
    0.0 <= s < 1.0 || error("demand spread must satisfy 0 ≤ s < 1; got $s")
    return s
end

"""
    demand_noise_atoms(spread::Real) -> Vector{Float64}

Return the three equiprobable multiplicative demand atoms

```math
\\{1 - s,\\; 1,\\; 1 + s\\}
```

for spread `s` (each with probability 1/3, i.i.d. across stages).
"""
demand_noise_atoms(spread::Real) = begin
    0.0 <= spread < 1.0 || error("demand spread must satisfy 0 ≤ s < 1; got $spread")
    [1.0 - Float64(spread), 1.0, 1.0 + Float64(spread)]
end

"""Return the demand-atom IDs for paired-protocol inflow column `column`."""
function protocol_demand_atom_indices(T::Integer, column::Integer;
                                      seed::Integer=DEMAND_NOISE_SEED)
    T >= 0 || throw(ArgumentError("T must be nonnegative; got $T"))
    column >= 1 || throw(ArgumentError("column must be positive; got $column"))
    return rand(StableRNG(seed + column), 1:3, T)
end

"""
    sample_demand_factors(rng, T::Int, spread::Real) -> Vector{Float64}

Draw `T` i.i.d. per-stage demand factors `ξ_t` from the three-atom
distribution `{1−s, 1, 1+s}` (probability 1/3 each) using `rng`.

The draws are made SEQUENTIALLY (one `rand` per stage), so for a fixed seed the
length-`T₁` path is a prefix of the length-`T₂ ≥ T₁` path — training with
`T = 126` and evaluating with `T = 96` therefore share the first 96 factors of
each protocol column.

# Arguments
- `rng`: any `AbstractRNG` (pass `StableRNG(DEMAND_NOISE_SEED + column)` for
  protocol-paired draws).
- `T::Int`: number of stages.
- `spread::Real`: demand spread `s`.

# Returns
- `Vector{Float64}` of length `T` with entries in `{1−s, 1, 1+s}`.
"""
function sample_demand_factors(rng, T::Int, spread::Real)
    # The three equiprobable atoms.
    atoms = demand_noise_atoms(spread)
    # One sequential draw per stage (prefix property — see docstring).
    return [atoms[rand(rng, 1:3)] for _ in 1:T]
end

"""
    protocol_demand_factors(spread::Real, T::Int, column::Int) -> Vector{Float64}

Seeded demand-factor path for paired-protocol inflow column `column`:

```math
\\xi^{(c)} = \\mathrm{sample\\_demand\\_factors}(\\mathrm{StableRNG}(\\mathrm{seed} + c),\\; T,\\; s).
```

Column-keyed seeding pairs the demand path with the inflow column: any script
evaluating column `c` (training-time protocol eval, paired-500 Exa eval, or
paired SDDP Historical eval) sees the IDENTICAL atom path, making the complete
joint uncertainty trajectory reproducible and paired across methods.

# Arguments
- `spread::Real`: demand spread `s`.
- `T::Int`: number of stages.
- `column::Int`: paired-protocol scenario column id.

# Returns
- `Vector{Float64}` of length `T`.
"""
protocol_demand_factors(spread::Real, T::Int, column::Int) =
    demand_noise_atoms(spread)[protocol_demand_atom_indices(T, column)]

"""
    augment_scenario(w::AbstractVector, ξ::AbstractVector) -> Vector{Float64}

Interleave a flat stage-major inflow trajectory `w` (length `T·nHyd`) with
per-stage demand factors `ξ` (length `T`) into the augmented stage-major
uncertainty vector

```math
[w_1;\\, \\xi_1;\\; w_2;\\, \\xi_2;\\; \\ldots;\\; w_T;\\, \\xi_T]
```

of length `T·(nHyd+1)`, i.e. per-stage blocks `[w_t; ξ_t]`. This is the layout
the demand-noise DE builder sizes `p_inflow` for and the layout
`HydroReachablePolicy` slices (`n_uncertainty = nHyd + 1`, physical inflow =
first `nHyd` entries of each block).

# Arguments
- `w::AbstractVector`: flat inflow trajectory, length divisible by `length(ξ)`.
- `ξ::AbstractVector`: per-stage demand factors, length `T`.

# Returns
- `Vector{Float64}` of length `T·(nHyd+1)`.
"""
function augment_scenario(w::AbstractVector, ξ::AbstractVector)
    # Number of stages comes from the factor vector.
    T = length(ξ)
    # Per-stage inflow width must divide the flat inflow length exactly.
    nHyd, rem = divrem(length(w), T)
    rem == 0 || error("length(w)=$(length(w)) is not divisible by T=$T")
    # Allocate the augmented stage-major output.
    out = Vector{Float64}(undef, T * (nHyd + 1))
    for t in 1:T
        # Copy the stage-t inflow block.
        out[(t-1)*(nHyd+1)+1 : (t-1)*(nHyd+1)+nHyd] = @view w[(t-1)*nHyd+1 : t*nHyd]
        # Append the stage-t demand factor as the block's last entry.
        out[t*(nHyd+1)] = ξ[t]
    end
    return out
end

"""
    sample_scenario(hydro_data, T, demand_spread; rng = Random.default_rng())
        -> Vector{Float64}

Demand-noise variant of [`sample_scenario`](@ref): draws the joint inflow
trajectory AND i.i.d. per-stage demand factors

```math
\\xi_t \\sim \\mathrm{Uniform}\\{1-s,\\; 1,\\; 1+s\\}
```

(independent of the inflow draw), returning the augmented stage-major vector
`[w_t; ξ_t]` of length `T·(nHyd+1)` (see [`augment_scenario`](@ref)).

# Arguments
- `hydro_data::HydroData`: inflow scenario data.
- `T::Int`: number of stages.
- `demand_spread::Real`: demand spread `s`.

# Keywords
- `rng`: random number generator (defaults to the task-global RNG, so
  `Random.seed!` seeds it exactly like the 2-arg method).

# Returns
- `Vector{Float64}` of length `T·(nHyd+1)`.
"""
function sample_scenario(hydro_data::HydroData, T::Int, demand_spread::Real;
                         rng = Random.default_rng())
    nHyd = hydro_data.nHyd
    # The three equiprobable demand atoms.
    atoms = demand_noise_atoms(demand_spread)
    # Augmented per-stage width: nHyd inflows + 1 demand factor.
    nu = nHyd + 1
    w = Vector{Float64}(undef, T * nu)
    for t in 1:T
        # Cyclic raw-row mapping (same convention as the 2-arg method).
        t_row = mod1(t, hydro_data.nStagesSample)
        # One joint inflow index per stage — all reservoirs share it.
        j = rand(rng, 1:hydro_data.nScenarios)
        for r in 1:nHyd
            w[(t-1)*nu + r] = hydro_data.scenario_inflows[r][t_row, j]
        end
        # Independent demand draw for the same stage (separate rand call).
        w[t*nu] = atoms[rand(rng, 1:3)]
    end
    return w
end
