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
| `battery_reachable_policy.jl` | the two strict policy architectures — recurrent encoder over observed demand, head, and the shared differentiable affine map into the one-stage reachable interval |
| `train_battery_exa_strict.jl` | the single parameterized training entry point, plus the rollout, the panel evaluation, checkpointing, device handling and the study's four method identifiers |
| `portfolio_runner.jl` | the production runner: one preemptible SEGMENT of one long run of `tsddr_nonlinear` or `tsldr_recurrent_linear`, with identity binding, verified checkpoints, resume and a stop protocol |
| `test/runtests.jl` | the consolidated regression suite |
| `battery_portfolio.json` | the frozen PGLib panel manifest. **Byte-identical copy in the JuMP package**, where it is produced. |
| `case/<name>/` | the frozen artifacts, mirrored from the JuMP package |

## The two policy architectures

Both are strict, both emit an outgoing storage-state target into the same
feasibility layer, and both train through the same dual-gradient path. What
differs is the trainable map, and nothing else.

| `DR_BAT_ARCH` | encoder | head | raw target is |
|---|---|---|---|
| `tsddr_nonlinear` (default) | `Flux.LSTM` | nonlinear, bounded output, reads `[h_t; e_{t-1}]` | a nonlinear function of the history AND the state |
| `tsldr_recurrent_linear` | `Flux.RNN(·, identity)` | affine, identity output, reads `[h_t; ξ_t]` | an **affine causal** function of the observed history |

`ξ_t = [context_t; observation_t]` is the stage's clock features
`(sin 2πt/P, cos 2πt/P)` concatenated with the realized per-bus active demand —
the same input both architectures see.

### What "recurrent linear" means, exactly

`tsldr_recurrent_linear` is a **structured recurrent parameterization of a
time-series linear decision rule for the storage-state targets**:

```
h_t = A h_{t-1} + B ξ_t + b          (Flux.RNN with identity activation)
z_t = C h_t     + D ξ_t + d          (a chain of identity Dense layers)
```

Unrolled from `h_0 = 0`,

```
z_t = Σ_{k=1..t} C A^{t-k} B ξ_k  +  D ξ_t  +  (Σ_{j=0..t-1} C A^j b) + d,
```

so the raw target is affine in `ξ_1, …, ξ_t` and depends on nothing later. The
recurrence is a *coefficient family*, not a different function class: it ties the
`T·(T+1)/2` coefficient blocks of a general TSLDR to a shared `(A, B, C, D)`,
which is what keeps the parameter count independent of the horizon. Stage
dependence stays explicit through `ξ_t`'s clock features, in both `B` and `D`.

The trainable map does **not** read the incoming energy. It cannot: the incoming
energy is the previous stage's squashed target, so a head that read it would make
the raw target a nonlinear function of the history and the name would be false.
The state enters where a decision rule with feasibility restoration puts it — in
the feasibility layer, whose interval endpoints are functions of `e_{t-1}`.

### The feasibility layer is the same object in both

```
ê_t = r̲_t(e_{t-1}) + (r̄_t(e_{t-1}) − r̲_t(e_{t-1})) · s(z_t),   s = stretchedsigmoid
```

`s` is bounded and boundary-attaining, so every emitted target is reachable **by
construction** and the stage problem's hard target equality is always attainable.
There is no target slack, no penalty and no projection. The nonlinear
architecture applies `s` as its head's own output activation (as it always has);
the linear one may not — that would be a trainable nonlinearity in a map required
to be affine — so it applies `s` in the layer. The composite is the same map, and
the nonlinear architecture's parameters and rollout are bit-identical to what
they were before the second architecture existed.

### Only the target follows a rule

In **both** architectures the stage ACP problem optimizes every recourse
variable — generation, the charge/discharge split, voltages, angles and the two
nodal recourse injections. Only the outgoing storage-state target is produced by
a decision rule. This README will not call the stage's dispatch variables linear
decision rules, because they are not.

### What the suite proves about it

An activation audit (every function-valued field in the trainable tree is
`identity`, **and** every encoder cell is an `RNNCell` — an `LSTMCell` hides its
gates in its forward pass and carries no activation field); causality (changing a
future atom leaves earlier raw targets bit-identical); history (perturbing each
earlier atom separately moves the last raw target); affinity
`f(αx + (1−α)y) = α f(x) + (1−α) f(y)` in Float64; an explicit hand-unrolling of
the recurrence and its closed form; reachability over `T = 24`; and the actor
gradient against centered finite differences at `T = 4` and `T = 24`. Each has a
**null control** against the nonlinear architecture, which must fail it.

## Solver accuracy, and what a reported cost is

Two settings of this engine are load-bearing and neither is a model choice.

`DEFAULT_SOLVER_OPTIONS` pins `tol = 1e-10` **and** `bound_relax_factor = 0.0`.
MadNLP's default relaxes every variable bound by `1e-8` before solving, so it
converges on a slightly larger feasible set than the model declares and reports
the primal infeasibility of the RELAXED problem. Measured on
`pglib_opf_case1354_pegase`: MadNLP reported `1.18e-12` while the residual
recomputed from its own solution by the shared schema was `2.92e-06` — four
orders above the study's `1e-7` gate. Tightening `tol` does not touch it, because
the solver already believes it has converged. Zeroing the relaxation takes the
residual to `2.11e-12` and ran `5.7×` faster.

`physical_stage_cost`, from the byte-identical `battery_solution_schema.jl`, is
the only function that may produce a headline cost. This engine parks the two
nodal recourse injections a bound-relaxation BELOW zero; at a recourse price of
1e5–1e6 per pu that is tens of cost units of barrier residue in the raw
objective. The contract projects every element within `1e-6` pu to exactly zero,
marks the solve inadmissible if any element is outside it, and keeps the raw
objective for diagnostics. Both engines run that same code.

## The frozen PGLib panel

This engine trains and evaluates on a preregistered panel of canonical PGLib
systems. It cannot construct one: it has no PGLib and no PowerModels dependency
by design, and acquiring a benchmark, placing storage and calibrating a demand
level are the JuMP package's business. What ships here is the panel MANIFEST — a
small, self-hashing JSON file, byte-identical to the JuMP package's copy —
and the case artifacts materialized from it.

```bash
# in the JuMP package, once per case
julia --project=. battery_portfolio.jl --case pglib_opf_case118_ieee \
      --out /path/to/DecisionRulesExa.jl/examples/BatteryStorageOPF/case

# here
DR_BAT_CASE_DIR=case/pglib_opf_case118_ieee julia --project=. test/runtests.jl
```

The manifest records, per case: the canonical network digest, the storage buses,
the region sizes and demand shares, the calibrated demand level `κ_case`, the
support digest, and the screening/final protocol seeds and digests. The
regression suite reads it with nothing but `JSON` and `SHA`, recomputes its
self-digest, and asserts byte identity against the JuMP package's copy — because
a manifest that had drifted between the two engines would let them evaluate two
different panels while both reported success.

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

# the same, with the recurrent linear decision rule
DR_BAT_ARCH=tsldr_recurrent_linear \
  julia --project=. -t auto train_battery_exa_strict.jl

# a full strict training stage on a GPU
DR_BAT_DEVICE=gpu julia --project=. -t auto train_battery_exa_strict.jl
```

Or through the study's stable method identifiers:

```julia
include("train_battery_exa_strict.jl")
battery_method(:tsldr_recurrent_linear)      # the descriptor and the shared invariants
run_battery_method(:tsldr_recurrent_linear; num_stages = 24)
run_battery_method(:sddp_soc)                # refused here, by name: it is the JuMP engine's
```

## Environment variables

One training stage is fully parameterized, which is what will let a declarative
lineage driver replay a published schedule rather than a narrative.

| variable | meaning | default |
|---|---|---|
| `DR_BAT_CASE_DIR` | frozen case directory | `case/pglib_opf_case14_ieee` |
| `DR_BAT_ARCH` | `tsddr_nonlinear` or `tsldr_recurrent_linear` | `tsddr_nonlinear` |
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

Only the SCREENING protocol is used for selection. The final 500-column protocol
is not opened by anything in this example.

## What a checkpoint carries

Schema `battery_storage_opf/checkpoint/3`: the trainable state, the
**architecture**, the optimizer state and the training trajectory. Reachability
metadata is deliberately not restored from it — that belongs to the frozen case,
and a checkpoint that could override a battery rating would let a stale file
silently redefine the problem.

`load_checkpoint!` refuses three things rather than continuing while reporting
something untrue: a file without the schema tag (it predates the architecture
field and the optimizer state), a file from a different network, and a file from
a different **architecture** — checked by name, before any array is touched, so
it does not rely on two encoders happening to have incompatible weight shapes.

The optimizer state is saved because a checkpoint that restores only the
parameters does not restore the RUN: Adam's moment estimates are as much of the
optimizer's position as the weights are, and a resumed stage that silently
restarts them takes a different first step than an uninterrupted one would have.

## The study's four method identifiers

`BATTERY_METHODS` carries the four the study compares, with the same rows and the
same invariant fields in **both** public engines:

| identifier | engine | what varies |
|---|---|---|
| `tsddr_nonlinear` | this one | LSTM encoder, nonlinear head |
| `tsldr_recurrent_linear` | this one | affine recurrence, affine head |
| `sddp_soc` | the JuMP engine | `SOCWRConicPowerModel` backward cuts |
| `sddp_dc` | the JuMP engine | `DCPPowerModel` backward cuts |

Every row declares the same horizon (24), protocol (screening), strict target
semantics, recourse and admissibility rule, cost contract
(`physical_stage_cost`) and comparison path (true ACP on paired protocol
columns), and each suite asserts it. `run_battery_method` runs the two this
engine owns and refuses the other two by name, naming the engine that owns them:
neither package loads the other, by design.

## Running a long study: `portfolio_runner.jl`

`train_strict` is one training call in one process. A study run is longer than
any queue reservation and can be killed at any moment, so it is executed as a
sequence of SEGMENTS, each a separate invocation of `portfolio_runner.jl` that
continues the previous one from a verified checkpoint. The runner adds no
science: the policy, the stage model, the gradient, the cost contract and the
screening panel are the same objects this file's other sections describe.

```bash
julia --project=. portfolio_runner.jl \
    --case-manifest case/pglib_opf_case118_ieee/case_manifest.json \
    --method        tsddr_nonlinear \
    --config        config.toml \
    --protocol      screening.toml \
    --output        run/seg001 \
    --resume-from   none
```

Those six flags are the whole contract; `--run-id`, `--segment`, `--attempt`,
`--stop-file` and `--max-seconds` exist for an automated caller and all default.
To continue, point `--resume-from` at the previous segment's newest checkpoint.
The protocol descriptor is written once per case with

```julia
include("portfolio_runner.jl")
write_protocol_descriptor("case/pglib_opf_case118_ieee", "screening.toml")
```

and a descriptor naming the FINAL protocol is refused, both when writing one and
when a run is launched against one — before any scenario is solved.

**What makes a resumed run the same run.** The learning rate is
`cosine_lr(i, target_index, …)`, a function of the GLOBAL update index rather
than of a per-segment counter; the scenario sampler's state and the optimizer's
moment estimates are checkpointed and restored exactly; and every coordinate
that defines the run — the case manifest digest, the case content digest, the
method, the config digest, the protocol digest and kind, the horizon, the seed,
the architecture and the common `ACP_BOUND_RELAX_FACTOR` — is hashed into an
identity record that every checkpoint carries and every resume re-derives. A
mismatch on any one of them refuses the resume and names the field.

**What a segment writes.** `checkpoints/ck_XXXXXXXX.jld2` with a `.meta.toml`
sidecar naming its digest (payload written and hashed first, sidecar second, so
no sidecar can ever vouch for an unfinished file); `history.csv`,
`trajectory.csv` and `evaluation.csv`; `result.toml`; and `identity.toml`. The
stop file is polled between complete updates: on a stop request the current
update finishes, a checkpoint is written and an honest `preempted` result is
recorded. `complete` is reported only when the configured target index was
reached.

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
