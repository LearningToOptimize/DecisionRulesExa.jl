# Battery-storage AC-OPF — ExaModels / GPU engine

This example is the GPU half of the multistage battery-storage study: a
true-ACP, strict-target deterministic equivalent written directly in ExaModels,
the strict reachable policy, and the TS-DDR trainer that drives them.

The scientific narrative lives in the documentation. This file says how to run
things and what each file is for.

The CPU half — the PGLib case builder, the PowerModels battery problem
specification and the stock SDDP baseline — lives in
`DecisionRules.jl/examples/BatteryStorageOPF`. The two packages are independent:
neither loads the other. They share the frozen case bytes and two source files
(`battery_case.jl`, `battery_solution_schema.jl`) as copies whose byte identity
is asserted whenever the case is rebuilt there.

## What is written by hand here, and why

This is the ONLY manually written network formulation in the project, and it
exists because there is no equally validated PowerModels-to-ExaModels bridge.
Its correctness is not asserted, it is measured: every physical variable is
differenced against the actual `PowerModels.ACPPowerModel` built from the same
frozen case, and the physical residuals of both solutions are recomputed
independently of either engine.

There is no SOC-WR model here. SDDP does not run through this engine, and TS-DDR
trains and evaluates on true ACP throughout.

There is exactly ONE formulation: strict. No soft-target, no penalized-target
and no target-deficit variant exists here or anywhere else in the supported
workflow.

## Files

| file | role |
|---|---|
| `battery_case.jl` | the frozen case contract, including the FROZEN FINITE DEMAND SUPPORT this engine trains from. **Byte-identical copy in the JuMP package**, where it is built. |
| `battery_solution_schema.jl` | the shared solution schema and the engine-neutral physical residuals. **Byte-identical copy in the JuMP package.** |
| `battery_exa.jl` | the network parse (arbitrary, nonconsecutive component identifiers) and the multistage strict true-ACP deterministic equivalent, its parameter updates, its target multipliers and its solution extraction |
| `battery_reachable_policy.jl` | the strict policy: recurrent encoder over observed demand, state-conditioned head, and the differentiable affine map into the one-stage reachable interval |
| `train_battery_exa_strict.jl` | the single parameterized training entry point, plus the rollout, the panel evaluation, checkpointing and device handling that the correctness gates share |
| `test/runtests.jl` | the consolidated regression suite |
| `case/<name>/` | the frozen artifacts, mirrored from the JuMP package |

## Commands

```bash
# consolidated regression suite (CPU)
julia --project=. test/runtests.jl

# the same suite on a GPU
DR_BAT_DEVICE=gpu julia --project=. test/runtests.jl

# a short strict training smoke
DR_BAT_STAGES=4 DR_BAT_EPOCHS=1 DR_BAT_BATCHES=2 DR_BAT_TRAJ=1 \
DR_BAT_ENCODER=8 DR_BAT_HEAD=12 DR_BAT_EVAL_EVERY=2 DR_BAT_EVAL_COLS=1,2 \
  julia --project=. -t auto train_battery_exa_strict.jl

# a full strict training stage on a GPU
DR_BAT_DEVICE=gpu julia --project=. -t auto train_battery_exa_strict.jl
```

## Environment variables

One training stage is fully parameterized, which is what will let a declarative
lineage driver replay a published schedule rather than a narrative.

| variable | meaning | default |
|---|---|---|
| `DR_BAT_CASE_DIR` | frozen case directory | `case/pglib_opf_case14_ieee` |
| `DR_BAT_STAGES` | horizon `T` | 24 |
| `DR_BAT_EPOCHS`, `DR_BAT_BATCHES` | update budget (`epochs × batches` gradient steps) | 2, 5 |
| `DR_BAT_TRAJ` | trajectories per gradient step | 2 |
| `DR_BAT_LR`, `DR_BAT_LR_FINAL` | cosine learning-rate ramp | 1e-3, 1e-4 |
| `DR_BAT_ENCODER`, `DR_BAT_HEAD` | encoder and head widths, comma separated | `64,64`, `128,128` |
| `DR_BAT_EVAL_EVERY`, `DR_BAT_EVAL_COLS` | screening-panel cadence and its protocol columns | 5, `1,2,3,4` |
| `DR_BAT_MAX_RECOURSE` | physical admissibility tolerance, pu | 1e-6 |
| `DR_BAT_SEED` | training seed | 20260804 |
| `DR_BAT_DEVICE` | `cpu` or `gpu` | `cpu` |
| `DR_BAT_CHECKPOINT` | checkpoint path | `battery_policy.jld2` |

## The demand this engine trains on

This engine never sees a demand sampler. What it reads is the frozen finite
support in `demand.json`: for every stage, a list of JOINT multiplier vectors
over the case's loads with explicit probabilities, hashed and mirrored
byte-identically from the JuMP package.

That is not a convenience. Both methods of the study train from finite support —
SDDP enumerates it in its backward pass, TS-DDR samples atom indices from it in
its trajectories — and if each were allowed to discretize a continuous authoring
law on its own, the two would face two different stochastic programs while every
report still said "the same demand process". `support_digest` is recomputed here
on load and checked against the manifest, so "the two engines consumed the same
support" is verified rather than intended.

Three consequences for this engine:

- the support is STAGE-DEPENDENT in general (`K_t` may differ across stages), so
  a training trajectory draws each stage's atom from that stage's own
  probabilities;
- the multiplier is per LOAD, so the realized per-bus demand is the scaled loads
  AGGREGATED to the bus, not a bus-level factor — anything else would average
  away a regional or per-load structure and silently change the problem;
- the stage clock feature the policy is given uses the period the support
  records, and stage indices outside the frozen horizon are an error rather than
  a wrap.

## Generators that exist in some stages and not others

A case may declare, for any generator, a per-stage AVAILABILITY schedule: a
vector of multipliers under the `stage_availability` key of that generator's row
in `network.json`, applied to `pmin`, `pmax`, `qmin` and `qmax` alike, so an
entry of `0` takes the unit out of service completely — active and reactive —
rather than leaving something that cannot generate but can still hold up a
voltage for free.

The convention is OPTIONAL and additive: a generator that carries no schedule is
available in every stage, and a case built before the convention existed produces
bit-identical variable bounds today. Because the schedule lives inside
`network.json`, it is covered by the case digest and travels with the case.

The JuMP package applies it to the parsed network just before PowerModels
instantiates one stage. This engine builds every stage of the horizon in ONE
model, so it applies the multiplier to each stage's generator variable bounds
instead — `build_battery_exa` takes a `stages` keyword naming which CASE stage
each of its `T` model positions is (`1:T` by default, a window such as `[2]` for
a continuation problem solved on its own). Three consequences:

- the model's SHAPE never depends on the schedule. The unit keeps its variables,
  its cost row and its position in every flat array, and only its bounds close;
  `gen_status` is deliberately untouched, because PowerModels drops an
  out-of-service generator from `ref` and that would change the variable set from
  one stage to the next in the JuMP engine.
- the schedule is DATA. It reaches the model through `lvar`/`uvar` and nothing
  else — no objective term, no constraint coefficient, no parameter — so it lies
  on no automatic-differentiation path, and the trajectory multipliers the
  trainer consumes are still exactly the derivative of the solved value.
- a case WITH a schedule may only be solved on the window it was built for.
  `assert_stage_window`, called from `strict_solve!`, fails closed rather than
  silently solving stage 5 with stage 1's availability; a case without a schedule
  accepts every offset exactly as it always did.

A schedule that is not a non-empty vector, a multiplier that is not finite or
lies outside `[0, 1]`, inconsistent bounds on a scheduled unit, and a stage the
schedule does not cover are all errors at parse or build time.

## Selection rule

A checkpoint is written only when a COMPLETE screening-panel evaluation improves
on the best complete evaluation so far. Complete means every panel column
solved AND the worst physical recourse on every column is within
`DR_BAT_MAX_RECOURSE`. Averaging the columns that happened to succeed would
report a policy that does not exist, and a policy that leans on the recourse is
not admissible however cheap it looks.

The training loss, the screening-panel rollout and an SDDP bound are three
distinct signals and are never compared in absolute level.

## Devices and precision

`DR_BAT_DEVICE=gpu` moves the policy with `Flux.gpu`, builds the ExaModels core
on a CUDA backend and selects `MadNLPGPU.CUDSSSolver`. Three failure modes are
checked rather than assumed:

- `Flux.gpu` is a silent no-op without cuDNN, so `cuDNN` is imported and
  `assert_device` verifies that every trainable array and the recurrent state
  really are device arrays;
- `MadNLPGPU.CUDSSSolver` is `nothing` unless CUDSS.jl has been loaded, which is
  checked before it is passed to the solver;
- the ExaModels model stays in `Float64` on both devices; the policy is
  `Float32` for training and can be promoted with `Flux.f64` for
  finite-difference work.

On a cluster whose NVIDIA driver is newer than the runtime CUDA.jl can
auto-select, `CUDA.functional()` returns false with a "JLLs were precompiled
without an NVIDIA driver present" message. Pin the runtime once per environment
with `CUDA.set_runtime_version!(v"12.6")` — this writes a machine-local
`LocalPreferences.toml`, which is not part of the published example.

## The solver is fresh on every solve

`solve!` builds a new MadNLP solver each time, deliberately. PGLib cases contain
synchronous condensers whose active-power box is exactly `[0, 0]`, MadNLP's
default `fixed_variable_treatment` removes such variables from its internal
primal vector, and its re-solve path then fails to map a full-length starting
point into the reduced one. The only re-solve configuration that works widens
those boxes by about `1e-8`, which makes this engine's feasible set larger than
the PowerModels model it is validated against. Correctness wins; the model
itself (sparsity pattern, derivative kernels) is still built once and reused,
and a fresh 24-stage solve of the correctness-phase case takes about 0.2 s.
