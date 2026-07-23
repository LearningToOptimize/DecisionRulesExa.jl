# Battery-Storage AC-OPF (deterministic foundation)

This example builds a **deterministic AC optimal-power-flow (AC-OPF) problem with
batteries** on top of any [PGLib-OPF](https://github.com/power-grid-lib/pglib-opf)
benchmark network, entirely in [ExaModels](https://github.com/exanauts/ExaModels.jl),
and solves it on the CPU with [MadNLP](https://github.com/MadNLP/MadNLP.jl).

It is **Phase 1** of a larger project: a reproducible battery-case generator plus
the ExaModels model that later phases train (TS-DDR) and benchmark (SDDP) on. This
phase contains **no** training, no uncertainty, and no SDDP — just the case
generator, the model, a reference check, and tests.

If you have never used Julia or power systems before: you can change three
numbers (the case name, the number of batteries, and a random seed), run one
command, and get a solved battery dispatch with a manifest that lets anyone
reproduce it exactly.

---

## 1. The battery model and units

Every quantity is **per unit (pu)** on the network's `baseMVA` power base. Energy
is in **pu·hours (pu·h)** and time in **hours**. All unit conversion happens in a
single data layer (`src/network_data.jl`), so the model code never re-scales.

For each battery `b` and stage `t` (stage length `Δt` hours) the model uses
continuous, non-negative charge/discharge powers and a linear state of charge
`e` (the battery SoC — *not* the SOC-WR relaxation used later by SDDP):

$$
0 \le p^{ch}_{t,b} \le \bar p^{ch}_b, \qquad
0 \le p^{dis}_{t,b} \le \bar p^{dis}_b,
$$

$$
e_{t+1,b} = (1-\sigma_b\,\Delta t)\,e_{t,b}
          + \eta^{ch}_b\,\Delta t\,p^{ch}_{t,b}
          - \frac{\Delta t}{\eta^{dis}_b}\,p^{dis}_{t,b},
\qquad
\underline e_b \le e_{t,b} \le \bar e_b,
$$

$$
p^{bat}_{t,b} = p^{dis}_{t,b} - p^{ch}_{t,b}.
$$

`p_bat` enters the **active** power balance at the battery's bus. Batteries run at
**unity power factor** (no reactive injection).

**Cost units (read carefully).** The stage objective is

$$
\Delta t \sum_g \big(c^{(2)}_g\,pg_{t,g}^2 + c^{(1)}_g\,pg_{t,g} + c^{(0)}_g\big)
\;+\;
\sum_b c^{cycle}_b \,\big(p^{ch}_{t,b} + p^{dis}_{t,b}\big).
$$

* **Generator cost** — the PGLib polynomial coefficients
  $(c^{(2)}_g, c^{(1)}_g, c^{(0)}_g)$ are **USD per hour** at the stage dispatch
  (with `pg` in pu on `baseMVA`). Because a stage lasts $\Delta t$ hours, the
  *physical* stage cost is $\Delta t\,(c^{(2)}pg^2 + c^{(1)}pg + c^{(0)})$ — the
  factor $\Delta t$ multiplies **every** term, including the constant $c^{(0)}$.
  The reported objective is therefore dollars over the whole horizon. (For the
  default $\Delta t = 1$ h this equals the raw hourly cost.)
* **Battery cycle cost** — a documented, non-negative **throughput/degradation**
  price, computed **per battery**:
  $c^{cycle}_b = \text{cycle\_cost\_per\_mwh}_b \cdot \text{baseMVA} \cdot \Delta t$
  (USD/MWh × MW/pu × h), so $c^{cycle}_b\,(p^{ch}+p^{dis})$ is the dollar cost of
  that battery's throughput **energy** over the stage. $\Delta t$ is already
  folded into $c^{cycle}_b$. Each battery uses its own price — the fleet does not
  share a single coefficient.

The strictly-positive cycle cost keeps the formulation continuous and removes any
incentive to charge and discharge simultaneously (audited by the tests).

The full network model is the standard **AC-polar OPF**: bus voltage magnitudes
and angles, generator active/reactive dispatch with the original PGLib cost
functions and limits, the four branch power-flow equations, angle-difference
limits, apparent-power thermal limits at both branch ends, and **hard** active
and reactive power balance. There is **no load shedding and no reactive slack** —
balance is enforced exactly.

Sizing is derived from **declared system quantities only** (never from generator
prices): the fleet charge/discharge power is `fleet_power_fraction · Σ load`,
split equally across batteries, and each battery's energy capacity is
`power · duration_hours`.

---

## 2. File inventory

```
examples/BatteryStorageOPF/
├── Project.toml              # pinned example environment
├── Manifest.toml             # resolved versions (created by setup)
├── setup_env.jl              # one-shot dependency resolver
├── run_case300.jl            # CPU smoke runner (any case via env vars)
├── README.md                 # this file
├── src/
│   ├── BatteryStorageOPF.jl  # module; includes the files below
│   ├── network_data.jl       # typed per-unit PGLib parsing + stable id maps
│   ├── battery_data.jl       # battery structs + make_battery_case + validation
│   ├── manifest.jl           # manifest write / hash / reconstruct
│   ├── battery_opf_exa.jl    # ExaModels AC-polar deterministic equivalent
│   └── reference_powermodels.jl # independent PowerModels/JuMP+Ipopt ACP check
└── test/
    └── runtests.jl           # example-local test suite
```

---

## 3. Environment setup

This example has its own environment; it does **not** use the parent package's.

```bash
module load julia                       # Julia 1.11+
cd examples/BatteryStorageOPF
julia --pkgimages=no --project=. setup_env.jl
```

> **Why `--pkgimages=no`?** On this cluster the system Julia cannot build the
> native precompile image for the `Pkg` standard library (a MadNLP dependency),
> which aborts a normal load. Disabling native package images sidesteps that; the
> model and solver are otherwise unaffected. **Use `--pkgimages=no` on every
> command below.** (The first run of each command JIT-compiles from source and
> may take a few minutes; subsequent runs in the same session are fast.)

The dependencies download the PGLib-OPF **23.07** benchmark artifact once; no
network access is needed afterwards.

---

## 4. Run case300 (one command)

```bash
julia --pkgimages=no --project=. run_case300.jl
```

This builds `case300_ieee` with 20 reproducibly-placed batteries (seed
`20260722`), writes a manifest, builds a 4-stage ExaModels AC-polar model, and
solves it on the CPU.

### Expected output (abridged)

```
Building battery case "case300_ieee" (batteries=20, seed=20260722)
  network: 300 buses, 69 gens, 411 branches, 201 loads, baseMVA=100.0
  total load = 235.2585 pu ; eligible load buses = 191
  battery buses (stable order) = [247, 104, 215, 211, 178, 201, 141, 135, 79, 21, 11, 234, 156, 73, 77, 175, 209, 9533, 10, 167]
  per-battery: p̄=2.9407 pu, e_max=11.7629 pu·h, e_init=5.8815 pu·h, η=0.95/0.95
  manifest hash = 002af1ee461dae3cec4dc1b29e0582e3bac81916ca9997853bd2869a5beeb93e
  wrote results/manifest_case300_ieee_seed20260722.json
         results/batteries_case300_ieee_seed20260722.csv

Building ExaModels AC-polar DE (T=4, Δt=1.0 h) ...
Solving on CPU with MadNLP ...

── Results ─────────────────────────────────────────────
status              : SOLVE_SUCCEEDED
accepted            : true
objective (USD)     : 1903529.258329
solve time (s)      : ~30
max primal residual : 3.350e-07
max |battery balance|: 7.550e-15
max simultaneous charge/discharge power : 1.658e-08 pu
SoC[:,1] (init)     : [5.8815, 5.8815, ... ]
SoC[:,end] (final)  : [-0.0, -0.0, ... , 4.6325, -0.0, -0.0]
Σ discharge (pu)    : 107.3469 ; Σ charge (pu) : 0.0000
────────────────────────────────────────────────────────
```

The batteries discharge their stored energy over the horizon (the SoC drains
from 5.88 to ~0 pu·h); with a flat/gentle load shape there is no arbitrage
incentive to charge. The tiny **max simultaneous charge/discharge power** (the
smaller of the two powers per battery, in pu — not a product) and the
machine-precision battery-balance residual confirm the continuous formulation
behaves as intended. Exact numbers can vary slightly across solver builds.

The two files under `results/` are the machine-readable manifest and a
human-readable battery table.

---

## 5. Change the case, battery count, and seed

`run_case300.jl` reads four environment variables:

```bash
# A different network, 3 batteries, a different seed, a 6-stage horizon:
BAT_CASE=case14_ieee BAT_NBAT=3 BAT_SEED=1 BAT_HORIZON=6 \
  julia --pkgimages=no --project=. run_case300.jl
```

Or from the Julia API directly:

```julia
include("src/BatteryStorageOPF.jl"); using .BatteryStorageOPF
case = make_battery_case("case300_ieee";
    number_of_batteries = 20,
    seed                = 20260722,
    duration_hours      = 4.0,     # energy/power ratio (a "4-hour battery")
    initial_soc         = 0.5,     # fraction of energy capacity
    charge_efficiency   = 0.95,
    discharge_efficiency= 0.95)
prob   = build_battery_de(case, 4; stage_hours = 1.0)
result = solve_de!(prob)   # named solve_de! (avoids clashing with CommonSolve.solve!)
```

List every available benchmark with `available_pglib_cases()`. Names resolve
leniently (`"case300_ieee"`, `"pglib_opf_case300_ieee"`, or the `.m` file name);
missing or ambiguous names raise an actionable error.

---

## 6. A second, small PGLib example

`case14_ieee` (14 buses) solves in a second and is handy for experimentation:

```bash
BAT_CASE=case14_ieee BAT_NBAT=3 BAT_SEED=7 BAT_HORIZON=4 \
  julia --pkgimages=no --project=. run_case300.jl
```

The same API works for **any** PGLib case — only the name changes.

---

## 7. Tests

```bash
julia --pkgimages=no --project=. test/runtests.jl
```

The suite checks reproducible placement and manifest hashing, reconstruction
from a manifest, input validation, non-consecutive identifier mapping, the
case300 structure (300 buses, 20 distinct valid battery buses), an accepted
solve with small primal and battery-balance residuals, the absence of
simultaneous charge/discharge, and **base-ACP parity**: with zero batteries or
zero battery power the ExaModels objective matches the independent
PowerModels/Ipopt reference.

---

## 8. Attribution and reproducibility

**Network data & license.** Networks come from the *Power Grid Library for
Benchmarking AC Optimal Power Flow Algorithms* (PGLib-OPF), release **23.07**,
distributed via `PGLib.jl`. PGLib-OPF is licensed under the **Creative Commons
Attribution 4.0 International** license
(<https://creativecommons.org/licenses/by/4.0/>). Please cite
S. Babaeinejadsarookolaee *et al.*, "The Power Grid Library for Benchmarking AC
Optimal Power Flow Algorithms," arXiv:1908.02788. The manifest records the exact
MATPOWER filename, its SHA-256, the upstream release, the license name and URL,
the attribution/citation, and the PGLib.jl / PowerModels.jl / Julia versions.

**Reproducibility.** Battery placement is a seeded, uniform-without-replacement
sample of eligible load buses using `StableRNGs` (stable across Julia versions).
`make_battery_case` records the exact case, the **SHA-256 of the MATPOWER source
bytes**, the seed, eligible-bus rule, selected buses (in stable order), all
battery parameters and units, package/artifact versions, and a **SHA-256 content
hash** (which itself includes the source-file hash). `write_manifest` serializes
this to JSON. `reconstruct_case(manifest)` does not claim a "byte-identical"
rebuild loosely — it verifies three concrete things and errors on any mismatch:

1. **source network bytes** — the SHA-256 of the currently resolved MATPOWER file
   equals the recorded `matpower_sha256` (checked first, so a changed or tampered
   network file is rejected outright);
2. **ordered placement** — the rebuilt battery buses match `selected_bus_ids` in
   order;
3. **battery parameters** — the rebuilt content hash equals the recorded one.

The same seed always yields the same ordered placement and hash; a different seed
yields a reproducibly different placement.
