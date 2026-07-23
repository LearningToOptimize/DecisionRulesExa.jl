# battery_data.jl
#
# Validated battery structures and the seeded, reproducible case-construction
# API `make_battery_case`. All battery quantities are per unit on the network
# `baseMVA` (power in pu, energy in pu·h), consistent with the one conversion
# layer in network_data.jl.

using StableRNGs
using Random

# ── Battery model (see BATTERY_STORAGE_OPF_PLAN.md §4) ─────────────────────────

"""
    BatteryData

One battery, per unit on the network `baseMVA`.

* `id`               : 1-based battery index; equals placement order (stable).
* `bus_id`,`bus_pos` : host bus original PGLib id and array position.
* `p_charge_max`     : max charge power  p̄ᶜʰ (pu), applied as `0 ≤ pᶜʰ ≤ p̄ᶜʰ`.
* `p_discharge_max`  : max discharge power p̄ᵈⁱˢ (pu), `0 ≤ pᵈⁱˢ ≤ p̄ᵈⁱˢ`.
* `e_min`,`e_max`    : energy bounds e (pu·h), `e_min ≤ e ≤ e_max`.
* `e_init`           : initial state of charge (pu·h).
* `eta_ch`,`eta_dis` : charge / discharge efficiencies in (0, 1].
* `sigma`            : self-discharge rate σ (per hour); state keeps `(1 − σΔt)`.
* `cycle_cost_per_mwh`: nonnegative throughput/degradation price (\$/MWh). The
  objective charges `cycle_cost_per_mwh · baseMVA · Δt · (pᶜʰ + pᵈⁱˢ)` — the
  dollar degradation cost of the throughput energy — which keeps the model
  continuous and removes any incentive for simultaneous charge/discharge.

State equation (linear):
  `e_{t+1} = (1 − σΔt)·e_t + η_ch·Δt·pᶜʰ_t − (Δt/η_dis)·pᵈⁱˢ_t`,
active injection into the host bus: `p_bat = pᵈⁱˢ − pᶜʰ` (unity power factor).
"""
struct BatteryData
    id::Int
    bus_id::Int
    bus_pos::Int
    p_charge_max::Float64
    p_discharge_max::Float64
    e_min::Float64
    e_max::Float64
    e_init::Float64
    eta_ch::Float64
    eta_dis::Float64
    sigma::Float64
    cycle_cost_per_mwh::Float64
end

"""
    BatteryCase

A network plus its reproducibly-placed battery fleet and the full record needed
to rebuild it byte-identically (seed, sampling rule, sizing parameters, units).
"""
struct BatteryCase
    network::NetworkData
    batteries::Vector{BatteryData}
    # placement / configuration record (mirrored into the manifest)
    case_name::String
    seed::Int
    number_of_batteries::Int
    duration_hours::Float64
    initial_soc::Float64
    charge_efficiency::Float64
    discharge_efficiency::Float64
    fleet_power_fraction::Float64
    self_discharge_rate::Float64
    e_min_fraction::Float64
    cycle_cost_per_mwh::Float64
    eligible_bus_rule::String
    eligible_bus_ids::Vector{Int}
    selected_bus_ids::Vector{Int}   # stable placement order
    explicit_buses::Bool
    total_load_pu::Float64
    parse_meta::NamedTuple
end

nbattery(bc::BatteryCase) = length(bc.batteries)

# ── Eligible-bus rule ─────────────────────────────────────────────────────────

"""
    eligible_load_bus_ids(network) -> Vector{Int}

Default battery placement pool: original ids of buses hosting at least one
in-service load with strictly positive active demand, sorted ascending. Sorting
by original id makes the sampling reproducible independent of parse order.
"""
function eligible_load_bus_ids(network::NetworkData)
    ids = Int[]
    for (pos, b) in enumerate(network.buses)
        network.bus_pd[pos] > 0 && push!(ids, b.id)
    end
    return sort!(ids)
end

const ELIGIBLE_BUS_RULE = "in_service_load_buses_with_positive_active_demand"

# ── Case construction ─────────────────────────────────────────────────────────

"""
    make_battery_case(case_name;
        number_of_batteries = 20,
        seed                = 20260722,
        duration_hours      = 4.0,
        initial_soc         = 0.5,
        charge_efficiency   = 0.95,
        discharge_efficiency= 0.95,
        fleet_power_fraction= 0.25,
        self_discharge_rate = 0.0,
        e_min_fraction      = 0.0,
        cycle_cost_per_mwh  = 2.0,
        buses               = nothing,
    ) -> BatteryCase

Build a reproducible battery-storage AC-OPF case from a PGLib benchmark.

Steps: resolve `case_name` in the pinned PGLib artifact; parse it into per-unit
[`NetworkData`]; determine the eligible-bus pool; sample `number_of_batteries`
distinct host buses **uniformly without replacement** with `StableRNG(seed)`
(unless explicit `buses` are given); derive per-battery power/energy ratings
from declared system quantities; validate every bound; and return a typed
[`BatteryCase`].

Sizing rule (declared, price-free): the fleet charge/discharge power is
`fleet_power_fraction · Σ load` (pu), split equally across batteries; each
battery's energy capacity is `power · duration_hours` (pu·h). This depends only
on system load and the base, never on generator prices.

`buses` (optional) is an explicit vector of original PGLib bus ids used in the
given order instead of sampling; its length must equal `number_of_batteries`.

All numeric inputs are validated; invalid counts, buses, efficiencies,
capacities, SoC, self-discharge, or ambiguous case names raise actionable errors.
"""
function make_battery_case(case_name::AbstractString;
                           number_of_batteries::Integer = 20,
                           seed::Integer = 20260722,
                           duration_hours::Real = 4.0,
                           initial_soc::Real = 0.5,
                           charge_efficiency::Real = 0.95,
                           discharge_efficiency::Real = 0.95,
                           fleet_power_fraction::Real = 0.25,
                           self_discharge_rate::Real = 0.0,
                           e_min_fraction::Real = 0.0,
                           cycle_cost_per_mwh::Real = 2.0,
                           buses = nothing)

    # ── Validate scalar inputs up front (actionable messages) ─────────────────
    number_of_batteries >= 0 ||
        error("number_of_batteries must be ≥ 0; got $number_of_batteries")
    (isfinite(duration_hours) && duration_hours > 0) ||
        error("duration_hours must be finite and > 0; got $duration_hours")
    (0 <= e_min_fraction < 1) ||
        error("e_min_fraction must satisfy 0 ≤ e_min_fraction < 1; got $e_min_fraction")
    (e_min_fraction <= initial_soc <= 1) ||
        error("initial_soc must satisfy e_min_fraction ($e_min_fraction) ≤ initial_soc ≤ 1; " *
              "got $initial_soc")
    (0 < charge_efficiency <= 1) ||
        error("charge_efficiency must satisfy 0 < η_ch ≤ 1; got $charge_efficiency")
    (0 < discharge_efficiency <= 1) ||
        error("discharge_efficiency must satisfy 0 < η_dis ≤ 1; got $discharge_efficiency")
    (isfinite(fleet_power_fraction) && fleet_power_fraction >= 0) ||
        error("fleet_power_fraction must be finite and ≥ 0; got $fleet_power_fraction")
    (isfinite(self_discharge_rate) && 0 <= self_discharge_rate < 1) ||
        error("self_discharge_rate σ must satisfy 0 ≤ σ < 1 (per hour); got $self_discharge_rate")
    (isfinite(cycle_cost_per_mwh) && cycle_cost_per_mwh >= 0) ||
        error("cycle_cost_per_mwh must be finite and ≥ 0; got $cycle_cost_per_mwh")

    # ── Resolve + parse the PGLib network ─────────────────────────────────────
    network, parse_meta = load_pglib_network(case_name)
    eligible = eligible_load_bus_ids(network)
    total_load_pu = sum(network.bus_pd)

    # ── Determine host buses (explicit or sampled) ────────────────────────────
    explicit = buses !== nothing
    local selected::Vector{Int}
    if explicit
        selected = Int.(collect(buses))
        length(selected) == number_of_batteries ||
            error("explicit `buses` has $(length(selected)) entries but " *
                  "number_of_batteries = $number_of_batteries")
        length(unique(selected)) == length(selected) ||
            error("explicit `buses` contains duplicate bus ids")
        for bid in selected
            haskey(network.bus_id_to_pos, bid) ||
                error("explicit bus id $bid is not an in-service bus of \"$(network.case_name)\"")
        end
    else
        number_of_batteries <= length(eligible) ||
            error("cannot place $number_of_batteries batteries: only " *
                  "$(length(eligible)) eligible load buses in \"$(network.case_name)\". " *
                  "Reduce number_of_batteries or pass explicit `buses`.")
        # Uniform sampling without replacement: a seeded shuffle of the eligible
        # pool, first k entries. The shuffle order IS the stable placement order.
        rng = StableRNG(UInt64(unsigned(Int64(seed))))
        selected = Random.shuffle(rng, eligible)[1:number_of_batteries]
    end

    # ── Derive per-battery ratings from declared system quantities ────────────
    fleet_power = fleet_power_fraction * total_load_pu
    per_power = number_of_batteries == 0 ? 0.0 : fleet_power / number_of_batteries
    e_max = per_power * duration_hours
    e_min = e_min_fraction * e_max
    e_init = initial_soc * e_max

    batteries = BatteryData[]
    for (i, bid) in enumerate(selected)
        push!(batteries, BatteryData(i, bid, network.bus_id_to_pos[bid],
                                     per_power, per_power,
                                     e_min, e_max, e_init,
                                     Float64(charge_efficiency),
                                     Float64(discharge_efficiency),
                                     Float64(self_discharge_rate),
                                     Float64(cycle_cost_per_mwh)))
    end

    bc = BatteryCase(network, batteries, network.case_name, Int(seed),
                     Int(number_of_batteries), Float64(duration_hours),
                     Float64(initial_soc), Float64(charge_efficiency),
                     Float64(discharge_efficiency), Float64(fleet_power_fraction),
                     Float64(self_discharge_rate), Float64(e_min_fraction),
                     Float64(cycle_cost_per_mwh),
                     explicit ? "explicit_buses" : ELIGIBLE_BUS_RULE,
                     eligible, selected, explicit, total_load_pu, parse_meta)

    validate_battery_case(bc)
    return bc
end

"""
    validate_battery_case(bc) -> BatteryCase

Post-construction invariants (defence in depth on top of `make_battery_case`'s
input checks): distinct valid host buses, ordered non-degenerate energy bounds,
initial SoC inside its band, sane efficiencies, nonnegative powers and
throughput cost, and a stable-order selection consistent with the batteries.
Throws on the first violation; returns `bc` when all hold.
"""
function validate_battery_case(bc::BatteryCase)
    nd = bc.network
    seen = Set{Int}()
    for b in bc.batteries
        haskey(nd.bus_id_to_pos, b.bus_id) ||
            error("battery $(b.id) is on unknown bus $(b.bus_id)")
        nd.bus_id_to_pos[b.bus_id] == b.bus_pos ||
            error("battery $(b.id) bus_pos $(b.bus_pos) inconsistent with map")
        b.bus_id in seen && error("duplicate battery bus $(b.bus_id)")
        push!(seen, b.bus_id)
        (b.p_charge_max >= 0 && b.p_discharge_max >= 0) ||
            error("battery $(b.id) has negative power rating")
        (b.e_min <= b.e_max) ||
            error("battery $(b.id) has e_min ($(b.e_min)) > e_max ($(b.e_max))")
        (b.e_min - 1e-12 <= b.e_init <= b.e_max + 1e-12) ||
            error("battery $(b.id) initial SoC $(b.e_init) outside [$(b.e_min), $(b.e_max)]")
        (0 < b.eta_ch <= 1 && 0 < b.eta_dis <= 1) ||
            error("battery $(b.id) has an efficiency outside (0, 1]")
        (0 <= b.sigma < 1) ||
            error("battery $(b.id) has self-discharge σ = $(b.sigma) outside [0, 1)")
        (b.cycle_cost_per_mwh >= 0) ||
            error("battery $(b.id) has negative cycle cost")
    end
    length(bc.selected_bus_ids) == length(bc.batteries) ||
        error("selected_bus_ids length disagrees with batteries")
    all(bc.selected_bus_ids[i] == bc.batteries[i].bus_id for i in 1:length(bc.batteries)) ||
        error("selected_bus_ids order disagrees with battery placement order")
    return bc
end
