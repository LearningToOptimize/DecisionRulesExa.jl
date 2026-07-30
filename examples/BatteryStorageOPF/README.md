# Battery-Storage AC-OPF with TS-DDR

This example builds a **stochastic AC optimal-power-flow (AC-OPF) problem with
batteries** on top of any [PGLib-OPF](https://github.com/power-grid-lib/pglib-opf)
benchmark network, entirely in [ExaModels](https://github.com/exanauts/ExaModels.jl),
and trains a **decision-rule policy (TS-DDR)** on the *true* AC-polar model with
[DecisionRulesExa](../../). It solves on CPU with [MadNLP](https://github.com/MadNLP/MadNLP.jl)
and on GPU with MadNLP + CUDSS.

If you have never used Julia or power systems before: you can change a few numbers
(the case name, the number of batteries, a random seed), run one command, and get
a trained battery policy with a manifest and trajectory that let anyone reproduce
it exactly.

* **Phase 1** — deterministic AC-polar foundation + reproducible PGLib battery-case
  generator + manifest (no uncertainty, no training).
* **Phase 2** *(this document also covers)* — seeded demand uncertainty, a paired
  scenario protocol, a battery-SoC **reachable policy**, a target-constrained
  ExaModels problem (strict / soft), CPU/GPU training, rollout evaluation,
  checkpointing, and compact trajectories.

There is **no claim here that TS-DDR beats SDDP** — that comparison is not part of
this example.

---

## 1. The battery model and units

Every quantity is **per unit (pu)** on the network's `baseMVA` power base. Energy
is in **pu·hours (pu·h)** and time in **hours**. All unit conversion happens in a
single data layer (`src/network_data.jl`); the model code never re-scales.

For each battery `b` and stage `t` (stage length `Δt` hours) the model uses
continuous, non-negative charge/discharge powers and a linear state of charge
`e` — the **battery SoC** (*not* the SOC-WR power-flow relaxation used by SDDP in
a later phase):

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

`p_bat` enters the **active** power balance at the battery's bus; batteries run at
**unity power factor** (no reactive injection). A strictly-positive throughput cost
`c_cycle·(p_ch + p_dis)` keeps the formulation continuous and removes any incentive
for simultaneous charge/discharge (audited by the tests). The network model is the
standard **AC-polar OPF** (voltages, the four branch-end flow equations with taps,
shifts and charging shunts, angle-difference limits, apparent-power limits at both
ends) with **hard** active and reactive balance and **no reactive slack**. These
equations live in ONE shared implementation (`src/acp_core.jl`) used by both the
deterministic Phase-1 builder and the Phase-2 operational builder.

The Phase-1 deterministic foundation has **no active recourse** — it is the
base-ACP parity artifact (hard balance). The Phase-2 **operational** model adds
the two-sided active recourse described in §3.

**Defaults & units** (all configurable in `make_battery_case`): power/energy in pu
/ pu·h on `baseMVA`; `duration_hours = 4.0` (a "4-hour battery"); `initial_soc =
0.5`; `charge_efficiency = discharge_efficiency = 0.95`; `fleet_power_fraction =
0.25` (fleet power = 0.25·Σload, split equally, never derived from generator
prices); `self_discharge_rate = 0.0` per hour; `cycle_cost_per_mwh = 2.0` USD/MWh.
`stage_hours` (Δt) defaults to `1.0` h at model-build time.

---

## 2. Demand uncertainty and what the policy observes

The only stochastic driver is **demand** (no renewables, no battery inflow). The
process is a **pure seeded function** (`src/demand_process.jl`):

* a deterministic hourly **base shape** `base_shape[t]` (time-of-day, mean 1);
* a small finite set of **joint atoms**, each an explicit
  `(L, R_1, …, R_R)` — a system-wide load factor `L` and per-region factors — with
  **explicit probabilities**;
* buses are split into `R` **regions** by a deterministic, price-free topology rule
  (nearest graph-anchor by hop distance, `assign_regions`), so scarcity can move
  spatially between atoms.

At stage `t`, given the atom `a_t`, the realized per-stage uncertainty vector is

```
w_t = [ s_t , r_{1,t} , … , r_{R,t} ],   s_t = base_shape[t]·L^{a_t},  r_{k,t} = R_k^{a_t}
```

(length `nw = 1 + R`). The realized demand at bus `p` is
`pd_p·s_t·r_{region(p),t}` and `qd_p·s_t·r_{region(p),t}` — active and reactive
scaled by the **same** factor, so every bus keeps its **power factor**. The policy
**observes `w_t` and the current battery SoC**, and chooses the next SoC target; it
**never sees future atoms**. Atoms are drawn i.i.d. across stages (a small,
SDDP-friendly support). Default: `nregion = 3`, `period = 24`, `1 + nregion`
atoms (a calm atom plus one per-region scarcity atom). Training and evaluation
use **distinct declared seeds** (`train_seed`, `eval_seed`).

### Demand calibration presets

Demand amplitude is a **case-design** parameter: too heavy a process makes the
realized demand unservable and forces nonzero active recourse (deficit `d⁺` or
surplus `d⁻`), which invalidates a scientific run. `DEMAND_PRESETS` is a fixed
ladder, strongest first:

| preset | `base_amplitude` | high system factor | scarce regional | off-region |
|:--|:--|:--|:--|:--|
| `:D0` | 0.05 | 1.03 | 1.05 | 0.99  |
| `:D1` | 0.04 | 1.02 | 1.04 | 0.99  |
| `:D2` | 0.03 | 1.01 | 1.03 | 0.995 |

The public default (`DEFAULT_DEMAND_PRESET = :D2`) is a **conservative tutorial
default**, chosen so the documented one-command examples run — **not** a
case300 gate-passing benchmark. `:D2` passes the current case14
zero-active-recourse checks but **does not** pass them on case300, and **the
current case300 battery placement/configuration is not yet a scientifically
accepted candidate** (screening it is Phase-3 work). Every scientific experiment
must record its preset explicitly and independently pass the zero-active-recourse
gates for its own case — **both** directions (`d⁺` and `d⁻`) numerically zero
within tolerance over the fixed stored evaluation paths, with every declared atom
checked individually. Rejected (stronger) presets remain available as named
experimental presets via `make_load_process(case; preset = :D0)`.
`demand_multiplier_summary(process)` reports the peak system and per-bus demand
multipliers.

**Targetless feasibility diagnostic.** `build_targetless_diagnostic_de` builds
the full-horizon ACP with batteries freely optimized and the two-sided active
recourse as the only valve. It is **genuinely targetless**: the model contains
no target variables, constraints, slacks, or penalties (not a zero-weight target
— no target data at all). Interpretation is deliberately asymmetric: a positive
active recourse at the returned *local* solution does **not** prove the demand is
physically infeasible (ACP is nonconvex; it is a local optimum); a successful
**recourse-disabled** solve (`allow_active_recourse = false`) proves a
zero-recourse feasible point *was found*; a **failed** recourse-disabled solve is
inconclusive and proves nothing. This separates demand-design signals from policy
failures without over-claiming.

### Paired protocol

`scenario_index_matrix(process, horizon, paths; seed)` produces a **stage-major**
`horizon × paths` matrix of atom indices. `write_scenario_protocol` serializes it
(with all process parameters, seeds, units, and hashes); `materialize_scenario`
turns one column into `w_flat`. **Every method shares the same stored index
matrix** — no method re-draws a differently-shaped random array.
`reconstruct_scenario_protocol` rebuilds and **verifies both** the process hash and
the index-matrix hash.

```julia
train_mat = scenario_index_matrix(process, T, 64; seed = process.train_seed)
eval_mat  = scenario_index_matrix(process, T, 64; seed = process.eval_seed)   # fixed held-out
write_scenario_protocol("results/eval_protocol.json", process, eval_mat; kind="eval", seed=process.eval_seed)
```

---

### Artifacts and exact reconstruction

`run_tiny_training.jl` writes the artifact set that defines an experiment, and
`evaluate_checkpoint.jl` reconstructs from those artifacts **alone** — it never
regenerates the demand process from defaults or environment variables:

| artifact | contents |
|:--|:--|
| `stochastic_manifest_<case>.json` | the case, the **exact** demand process (base-shape vector, `region_of_bus`, anchors, ordered atoms + probabilities, period, seeds, preset), horizons, stage duration, target mode, active-recourse price, penalty coefficients, policy architecture + activation, and five hashes |
| `train_protocol_<case>.json` / `eval_protocol_<case>.json` | the stage-major scenario-index matrices |
| `checkpoint_<case>.jls` | policy state + architecture + case/source/process hashes |
| `trajectory_<case>.json` | compact per-stage records |

Five **distinct** hashes are stored, none reused for another role: the
load-process content hash, the train and eval **index-matrix** hashes, and the
train and eval **protocol-file** (exact-bytes) hashes.
`reconstruct_stochastic_manifest` rebuilds the process field-for-field and
verifies its hash; `verify_protocol_file` checks the file bytes. Tampering with
the base shape, an atom, the scenario-index order, or a protocol file is
detected (see `test/test_artifacts.jl`, which also runs a **fresh-Julia-process
round trip** reproducing indices, statuses, costs, and active recourse).

## 3. Two-sided active recourse — the complete-recourse slack (always present)

The physical model carries a **two-sided active nodal recourse** — always present,
never a mode, and *not* a target-tracking device. At **every** bus there are two
nonnegative, **UNBOUNDED** variables entering the **active** balance:

```
d⁺[t,i] ≥ 0   (active deficit / injection   — covers an active shortfall)
d⁻[t,i] ≥ 0   (active surplus / absorption  — absorbs an active excess)
active balance:  p_d − d⁺ + d⁻ + gs·vm² − Σpg − Σ(p_dis−p_ch) + Σp_fr + Σp_to = 0
```

This is the classical multistage recourse device that gives **relatively
complete recourse**: because `d⁺` can inject arbitrary local power and `d⁻` can
absorb it, the stage subproblem is **feasible for every incoming SoC and every
dynamically reachable battery target**, in both the charging and discharging
directions. Concretely: a target that forces a battery to **charge** at a
network-constrained bus is served by local `d⁺`; a target that forces it to
**discharge** into a bus whose outgoing branches are saturated is absorbed by
local `d⁻`. They are an **artificial active-balance recourse, not curtailed
customer load**: `d⁺` may exceed local demand and may be positive at a bus with
`p_d = 0` — they exist at buses with no load too, which is exactly what makes the
recourse complete and why they are never reported as a per-load fraction.

The recourse is **active-only**: it does **not** touch reactive power, so reactive
KCL stays a **hard equality** with no reactive slack. Batteries are unity-power-
factor, so the battery target only moves active injection; reactive feasibility is
a property of the base network and the (feasible) demand process, independent of
the target.

Both directions are priced at `active_recourse_cost_per_mwh = 10 000` USD/MWh
(default), stage cost `cost · baseMVA · Δt · Σ(d⁺ + d⁻)`, **included in physical
operating cost**. They are a safety valve: **an accepted run leaves both
directions at ~0 within tolerance** — a nonzero deficit/surplus on an otherwise
sensible target is a case-design signal (the target is not network-deliverable),
not a solver failure. The solve **always succeeds**; the cost tells you whether
the target was deliverable. A scientific candidate path requires **both** `d⁺`
and `d⁻` numerically zero within tolerance.

## 4. Strict vs soft target mode

The interstage state is battery SoC. The policy outputs a **target next SoC**
`ê_{t+1}`; the operational problem (`build_battery_tsddr_de`) ties it to the
realized SoC in one of two **clearly separated** modes (orthogonal to active recourse):

* **strict** (default, **PRIMARY**) — `ê_{t+1,b} − e_{t+1,b} = 0`; **no target
  slack, no target penalty**. This is the production operational and training
  target: its multiplier is an economic shadow price uncontaminated by a penalty.
  Backed by the two-sided active recourse (§3), strict has **complete recourse**
  — for every supported PGLib case and every dynamically reachable target
  (including the exact reachable endpoints), the strict stage NLP solves and
  reproduces the target to `≤ 1e-5`.
* **soft** (diagnostic/fallback only) — `ê_{t+1,b} − e_{t+1,b} − δ⁺ + δ⁻ = 0` with
  `δ⁺,δ⁻ ≥ 0` and a documented **training-only** penalty
  `ρ1·Σ(δ⁺+δ⁻) + (ρ2/2)·Σ((δ⁺)²+(δ⁻)²)`.

The target penalty is **never** counted as physical operating cost, and target
violations are reported separately. Target slacks (`δ±`) and the active
recourse (`d±`) are distinct mechanisms and never share a name.

The default is **strict** — it is the primary mode. Soft is retained only as a
diagnostic/fallback.

### Strict start sequence (deterministic warm-start helper)

Strict feasibility is guaranteed by the two-sided active recourse (§3), not by the
start. This helper only makes the accepted solve **cheaper** (fewer iterations):
`solve_stage_with_starts` tries a **fixed** deterministic sequence and accepts the
**first** solver-accepted result (never the cheapest):

1. flat start (`vm = 1`);
2. **target-consistent battery start** — with `Δe = ê − (1−σΔt)·e_prev`,
   `Δe ≥ 0 ⇒ p_charge = Δe/(η_ch·Δt), p_discharge = 0`; otherwise
   `p_charge = 0, p_discharge = −Δe·η_dis/Δt`, clipped to the power bounds;
3. seed from the corresponding solved **targetless-diagnostic** ACP point;
4. the **previous stage's** accepted solution.

Only the starting point varies — no tolerance change, no iteration-limit
manipulation, no equation/bound/generator/network change, no dropped constraint.
Every attempted start and its solver status is logged.

### Reachable target policy

Every target lies in the one-stage physical reachability interval

```
ℓ = max(e_min, (1−σΔt)·e_t − (Δt/η_dis)·p̄_dis)
u = min(e_max, (1−σΔt)·e_t + η_ch·Δt·p̄_ch)
```

(battery dynamics only — **not** a network-feasibility proof), via
`ê = ℓ + (u−ℓ)·y`. The canonical default activation is

```
stretchedsigmoid(z) = clamp((sigmoid(z) − 0.03)/0.94, 0, 1 − 1e-3)
```

which attains the **lower** edge exactly at finite weights while keeping a small
margin **below** the exact upper edge: an exact `y = 1` (store-max) target under a
strict equality drives the stage NLP onto a measure-zero set that interior-point
solvers cannot converge into. `hardsigmoidsafe(z) = clamp(0.5 + 0.5z, 0, 1 − 1e-3)`
is a documented option with the same safe margin. Reachability bounds are physical
projection **data**: gradients stop through `ℓ` and `u` and flow only through the
normalized output. Recurrent state is reset at every scenario boundary.

---

## 5. Physical cost vs training-only penalty, reporting vs look-ahead

`decompose_costs(prob, result)` reports, recomputed independently from the primal
solution:

| quantity | meaning |
|---|---|
| `total_solver_objective` | the raw solver objective |
| `generator_cost` | `Δt·Σ(c2·pg² + c1·pg + c0)` (original PGLib costs, `c0` scaled too) |
| `battery_throughput_cost` | `baseMVA·Δt·Σ c_cycle(p_ch + p_dis)` |
| `active_recourse_cost` | `cost·baseMVA·Δt·Σ(d⁺ + d⁻)` — **physical** |
| `active_deficit_pu`, `active_surplus_pu`, `active_deficit_energy_mwh`, `active_surplus_energy_mwh`, `total_active_recourse_energy_mwh`, `max_active_deficit_pu`, `max_active_surplus_pu` | recourse audit, by direction (no per-load fraction) |
| **`physical_operating_cost`** | generator + throughput + **active-recourse** cost |
| `target_penalty`, `target_violation` | soft mode only — **training only, never physical** |
| `reporting_physical_cost` | physical cost over stages `1:reporting_horizon` |
| `lookahead_physical_cost` | physical cost over the look-ahead buffer |

and `total_solver_objective ≈ physical_operating_cost + target_penalty` (checked in
the tests). **Improvement is always judged on reporting-window physical cost,
never the total objective.**

The horizon is a **reporting horizon** followed by a **look-ahead buffer**:
`T = reporting_horizon + lookahead`. Both are recorded in the manifest and the
checkpoint and are identical for every method that uses the case. Only the reported
(reporting-horizon) physical cost is used to compare policies.

---

## 6. Environment setup

This example has its own environment and uses the parent `DecisionRulesExa` package
through a relative `[sources]` path in `Project.toml`.

```bash
module load julia                       # Julia 1.12.x (matches the parent package)
cd examples/BatteryStorageOPF
julia --pkgimages=no --project=. setup_env.jl
```

> **Why `--pkgimages=no`?** On this cluster the system Julia cannot build the
> native precompile image for the `Pkg` stdlib (a MadNLP dependency). Disabling
> native package images sidesteps that; the model and solver are unaffected. **Use
> `--pkgimages=no` on every command below.** The first run of each command JITs
> from source and may take a few minutes.

---

## 7. Run tiny CPU training (one command)

```bash
julia --pkgimages=no --project=. run_tiny_training.jl
```

This builds a battery case + demand process, writes the paired **train** and
**eval** protocols and the stochastic manifest to `results/`, evaluates a freshly
initialized policy on the fixed held-out eval scenarios (the "before" physical
cost), trains the reachable policy on the full-horizon target-constrained DE with
deterministic scenario replay, re-evaluates (the "after" physical cost), and saves
a checkpoint and a compact trajectory. It prints the initial and final held-out
**physical operating cost** and the improvement.

Override defaults via environment variables, e.g. a different case / battery count
/ seed / horizon / mode:

```bash
BAT_CASE=case14_ieee BAT_NBAT=3 BAT_SEED=1 BAT_NREGION=3 \
BAT_REPORT=4 BAT_LOOKAHEAD=1 BAT_MODE=strict BAT_BATCHES=40 \
  julia --pkgimages=no --project=. run_tiny_training.jl
```

## 8. Evaluate a saved checkpoint (no retraining)

```bash
BAT_CKPT=results/checkpoint_case14_ieee.jls \
  julia --pkgimages=no --project=. evaluate_checkpoint.jl
```

Reloads the policy (verifying the case-manifest, MATPOWER-source, and load-process
hashes), runs the non-anticipative stage-wise rollout on the **fixed** eval
scenarios, prints the held-out physical cost, and writes a trajectory.

## 9. Run on a GPU (when available)

```bash
# On a GPU node (see the GPU sbatch recipe), with a functional CUDA device:
julia --pkgimages=no --project=. run_gpu_training.jl
```

Builds the ExaModels model with a `CUDABackend()` and solves with MadNLP's CUDSS
GPU linear solver via `MadNLPGPU`. The CPU and GPU builders represent the **same**
mathematical problem (a structural-parity test asserts equal variable/constraint
counts and target-multiplier slice).

---

## 10. Change the case, battery count, and seed (Julia API)

```julia
include("src/BatteryStorageOPF.jl"); using .BatteryStorageOPF
using DecisionRulesExa, Flux, MadNLP, Random

case    = make_battery_case("case300_ieee"; number_of_batteries = 20, seed = 20260722)
process = make_load_process(case; nregion = 3)
T = 5; report = 4; lookahead = 1
de      = build_battery_tsddr_de(case, process; reporting_horizon = report,
                                 lookahead = lookahead, mode = :strict, stage_hours = 1.0)
stage   = build_battery_stage_problem(case, process; mode = :strict, stage_hours = 1.0)
Random.seed!(1)
policy  = battery_reachable_policy(case, process; dt = 1.0, layers = [64, 64], combiner_layers = [64])

train_mat = scenario_index_matrix(process, T, 64; seed = process.train_seed)
eval_mat  = scenario_index_matrix(process, T, 64; seed = process.eval_seed)
before = evaluate_paired(policy, stage, process, eval_mat; reporting_horizon = report)
train_battery_tsddr(policy, de, process, train_mat; num_batches = 100, num_train_per_batch = 16)
after  = evaluate_paired(policy, stage, process, eval_mat; reporting_horizon = report)
```

`available_pglib_cases()` lists every benchmark; names resolve leniently. The same
API works for **any** PGLib case — only the name changes.

---

## 11. File inventory

```
examples/BatteryStorageOPF/
├── Project.toml              # example env (DecisionRulesExa via [sources])
├── setup_env.jl              # resolve + instantiate + precompile
├── run_case300.jl            # Phase-1 deterministic CPU smoke runner
├── run_tiny_training.jl      # Phase-2 tiny CPU TS-DDR training + eval
├── evaluate_checkpoint.jl    # Phase-2 checkpoint evaluation (no retraining)
├── run_gpu_training.jl       # Phase-2 GPU training smoke
├── src/
│   ├── network_data.jl       # typed per-unit PGLib parsing + stable id maps
│   ├── battery_data.jl       # battery structs + make_battery_case + validation
│   ├── manifest.jl           # Phase-1 manifest write / hash / reconstruct
│   ├── acp_core.jl           # SHARED AC-polar equations (single source of truth)
│   ├── battery_opf_exa.jl    # Phase-1 deterministic model (uses acp_core)
│   ├── reference_powermodels.jl # PowerModels/Ipopt ACP parity check
│   ├── demand_process.jl     # seeded finite-support demand + paired protocol
│   ├── battery_tsddr.jl      # target-constrained DE (strict/soft) + cost decomposition
│   ├── battery_policy.jl     # battery-SoC reachable policy (edge-reaching map)
│   ├── battery_training.jl   # train/rollout entrypoints + checkpoint + trajectory
│   └── stochastic_manifest.jl# Phase-2 manifest (case + process + horizons + seeds + hashes)
└── test/
    ├── runtests.jl           # Phase-1 suite
    ├── runtests_phase2.jl    # Phase-2 suite (process/policy/DE/gradients/checkpoint/parity)
    ├── test_artifacts.jl     # exact reconstruction, 5 hashes, tampering, round trip
    └── test_e2e_training.jl  # tiny fixed-seed end-to-end training test
```

---

## 12. Tests

```bash
julia --pkgimages=no --project=. test/runtests.jl           # Phase 1
julia --pkgimages=no --project=. test/runtests_phase2.jl    # Phase 2
julia --pkgimages=no --project=. test/test_artifacts.jl     # artifacts + round trip
julia --pkgimages=no --project=. test/test_e2e_training.jl  # tiny end-to-end training
```

The Phase-2 suite checks exact demand replay, distinct train/eval protocols,
probability/input validation, deterministic region assignment, power-factor
preservation, protocol serialization/hashing/reconstruction, stochastic-manifest
reconstruction, reachability bounds and target containment, recurrent reset and
determinism, finite/finite-difference-checked policy gradients, the target
multiplier's finite-difference sign and magnitude, strict target equality, the
soft physical/penalty decomposition, zero active recourse in both directions and battery-balance residuals,
absence of simultaneous charge/discharge (soft mode), exact checkpoint reload, and
CPU/GPU structural parity (GPU checks are gated on `CUDA.functional()` and reported
as skipped on a CPU node).

---

## 13. Attribution and reproducibility

**Network data & license.** Networks come from the *Power Grid Library for
Benchmarking AC Optimal Power Flow Algorithms* (PGLib-OPF), release **23.07**,
distributed via `PGLib.jl`, licensed **CC BY 4.0**
(<https://creativecommons.org/licenses/by/4.0/>). Please cite
S. Babaeinejadsarookolaee *et al.*, arXiv:1908.02788. The manifest records the
exact MATPOWER filename, its SHA-256, the upstream release, the license, and the
package/Julia versions.

**Reproducibility.** Battery placement and scenario generation are seeded
(`StableRNGs`, stable across Julia versions). `make_battery_case` and
`make_load_process` record every parameter, seed, and hash; `write_stochastic_manifest`
serializes the case content hash (incl. the MATPOWER source hash), the process
hash, the horizon / reporting-horizon / look-ahead treatment, the target mode, the
train/eval seeds, and the protocol hashes. `reconstruct_stochastic_manifest`
rebuilds and verifies all of them. Checkpoints identify the Flux state,
architecture, target mode, horizons, seeds, and the case/source/process hashes, and
reload to reproduce policy outputs **exactly** on a fixed input.
