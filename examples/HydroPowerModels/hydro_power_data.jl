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

    hydro_json = JSON.parsefile(hydro_file)["Hydrogenerators"]
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

    # Water balance conversion factor K = 0.0036 (standard HydroPowerModels.jl value)
    # Converts turbine outflow (m³/s equivalent) to reservoir volume per stage.
    K = 0.0036

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
