# Bolivia hydro — ExaModels engine

The GPU half of the long-term hydrothermal planning case study: this directory
trains and evaluates the TS-DDR policy and the linear decision rule. The case, the
SDDP and reinforcement-learning baselines and the figures live in the companion
package, `DecisionRules.jl/examples/HydroPowerModels`. A run of the shipped
schedule reaches a policy of the same quality as the published one, not the same
policy: the stage problems are nonconvex and training is stochastic.

**The science is in the documentation** of that package, under *Case studies →
Long-term hydrothermal planning*. This file is the operating manual.

The two packages share the case bytes and two source files **byte for byte**:
`bolivia/{PowerModels.json, hydro.json, inflows.csv, *.mof.json,
case_manifest.json}`, `generate_canonical_case_artifacts.jl` and
`hydro_solution_schema.jl`. The case files are mirrored by the other package's
`export_subproblem_mof.jl --exa-root=…`; the source files are copies whose
identity is the point — both engines assert the same case contract and write
their solutions in the same format, without either depending on the other.

## Layout

| file | role |
|---|---|
| `hydro_power_data.jl` | parses `PowerModels.json` / `hydro.json` / `inflows.csv` into the flat arrays the ExaModels builder consumes |
| `hydro_power_exa.jl` | builds the `ExaModel`: AC-polar or DC, strict or penalized targets, with `hydro_solution` to unpack a solved point into named blocks |
| `hydro_reachable_policy.jl` | the feasibility-guaranteeing policy (LSTM encoder over inflow, state-conditioned head, targets mapped into the one-stage reachable interval) |
| `hydro_solution_schema.jl` | the long format in which a full physical solution is written; byte-identical to the JuMP engine's copy |
| `hydro_training_utils.jl` | small shared helpers for the training scripts |
| `train_hydro_exa_strict.jl` | ONE training stage, fully parameterized by environment variables |
| `run_tsddr_lineage.jl` | the lineage driver: runs a declared multi-stage schedule end to end, chaining only selected checkpoints |
| `lineage_from_scratch.json` | the from-scratch TS-DDR training schedule, as data: one entry per phase |
| `lineage_tsldr_from_scratch.json` | the same schedule for the linear decision rule; only the architecture differs |
| `eval_paired_exa.jl` | paired evaluation of a checkpoint, with per-stage physical recording and an optional full-solution dump |
| `generate_canonical_case_artifacts.jl` | the frozen-case contract and its verifier |

## Commands

Run from this directory with `--project=.`.

**1. Verify the case.**

```bash
julia --project=. generate_canonical_case_artifacts.jl --verify
```

**2. A short GPU smoke run** — a few updates on a short horizon, to confirm the
GPU stack (MadNLPGPU + CUDSS + cuDNN) is working before committing hours:

```bash
DR_NUM_STAGES=8 DR_NUM_ROLLOUT_STAGES=8 \
DR_NUM_EPOCHS=1 DR_NUM_BATCHES=3 DR_NUM_TRAIN_PER_BATCH=2 \
DR_NUM_EVAL_SCENARIOS=2 DR_EVAL_PROTOCOL_IDS=2,39 DR_EVAL_EVERY=3 \
  julia --project=. -t auto train_hydro_exa_strict.jl
```

**3. The full from-scratch training recipe.** The schedule is declared in
`lineage_from_scratch.json` and executed stage by stage; the linear decision rule
runs the same schedule from its own file:

```bash
julia --project=. run_tsddr_lineage.jl                 # full lineage
julia --project=. run_tsddr_lineage.jl --dry-run       # print the plan only
julia --project=. run_tsddr_lineage.jl --stages=phase3  # resume one phase
julia --project=. run_tsddr_lineage.jl --config=lineage_tsldr_from_scratch.json
```

Each stage runs as its own process, so a stage boundary is a real restart: the
optimizer state, the cosine learning-rate phase and the warm-up counter all
begin again. The driver chains only checkpoints that a COMPLETE, non-shedding
panel evaluation selected, hashes every parent before use, refuses `_latest`
snapshots outright, and stops the lineage — rather than falling back — if a
stage produces nothing selectable. Re-running resumes: a stage whose record
exists and whose checkpoint still hashes correctly is skipped, and a stage that
was interrupted continues its own trajectory from the full training state it
saved, rather than starting the phase again.

Records land in `bolivia/ACPPowerModel/lineage/`: one JSON per stage with the
resolved environment, ancestry, checkpoint hashes, update count and both the
process wall time and the trainer's own training-loop seconds, plus a
lineage-level ledger.

Neither W&B nor a workload manager is required. W&B logging is off unless
`DR_ENABLE_WANDB=true`; nothing in the driver reads a scheduler variable.

**4. Paired evaluation of a checkpoint.**

```bash
DR_EVAL_CKPT=/path/to/checkpoint.jld2 DR_EVAL_LABEL=my_policy \
  julia --project=. -t auto eval_paired_exa.jl                    # the 10-column panel

DR_EVAL_CKPT=… DR_EVAL_LABEL=shard_1_50 \
DR_EVAL_COL_FIRST=1 DR_EVAL_COL_LAST=50 \
  julia --project=. -t auto eval_paired_exa.jl                    # one shard of the 500
```

Adding `DR_SOLUTION_DUMP=1` additionally writes the full primal solution of every
stage and the decision trace that reproduces it, in the shared long format of
`hydro_solution_schema.jl` — the per-bus, per-branch physics the stagewise
figures are built from. (Nodal prices are duals and come from the JuMP engine's
evaluators, which have them directly.)

**5. Timing an update, and spreading it over many cores.** With strict targets
every reservoir volume is pinned to the policy's target, ``x_t = \hat x_t``, so the
deterministic equivalent of one scenario stops coupling its weeks: week ``t`` sees
the previous target ``\hat x_{t-1}`` only as data. An update over ``S`` scenarios of
``T`` weeks is therefore ``S \times T`` independent one-week problems, and the
target multiplier the training needs is assembled from the weekly ones,
``\lambda_t = y_t - y_{t+1}``, where ``y_t`` is the multiplier of week ``t``'s target
constraint (and ``y_{T+1} = 0``). The trainer exposes four timing modes, each of
which exits after printing its measurement:

| `DR_TIMING_ONLY` | what is timed |
|---|---|
| `1` | one warm solve of the full ``T``-week problem and of a single week, on GPU and CPU |
| `2` | every week alone, warm, on GPU and CPU: the slowest and the mean week |
| `3` | the network part of an update (rollout over ``T`` weeks and its reverse pass) and one decision |
| `4` | complete updates with rollout, weekly solves and gradient on worker processes (`hydro_fulliter.jl`) |

Mode 4 is the one to copy when an update must be fast on a CPU machine. It starts
`DR_FI_WORKERS` single-threaded worker processes (one per core; threads of one
process do not run interior-point solves concurrently), each holding one
one-week model built by the trainer's own builder (`hydro_fulliter_worker.jl`),
and then runs `DR_FI_BATCHES` updates for every batch size in `DR_FI_TRAJ`:

```bash
DR_USE_GPU=false DR_TIMING_ONLY=4 DR_FI_WORKERS=24 DR_FI_TRAJ=16+24 DR_FI_NET=pipelined \
  julia --project=. -t 1 train_hydro_exa_strict.jl
```

With `DR_FI_NET=workers` the three parts run one after another; with
`DR_FI_NET=pipelined` a scenario's reverse pass is queued as soon as its ``T``
weekly multipliers are back, so the gradients overlap the remaining solves and the
update costs little more than its solves. The first update of every batch size is
checked against the trainer's own path, the full ``T``-week problem solved per
scenario and differentiated in one process: the log prints the cosine between the
two gradients and the largest relative gaps in objective and multipliers, so a
change that breaks the decomposition shows up at once. Results are written to
`DR_FI_OUT` (default `logs/fulliter_cpu.json`).

## Configuration surface of one training stage

`train_hydro_exa_strict.jl` is driven entirely by environment variables, whose
defaults are the first phase of the from-scratch schedule; the lineage driver
simply sets them. The ones that define a stage:

| variable | meaning |
|---|---|
| `DR_NUM_TRAIN_PER_BATCH` | `nt`, trajectories sampled per gradient step |
| `DR_LR`, `DR_LR_FINAL`, `DR_LR_WARMUP` | cosine learning-rate schedule and its warm-up |
| `DR_NUM_EPOCHS` × `DR_NUM_BATCHES` | the update budget |
| `DR_MAX_TRAIN_SECONDS` | wall budget for the training loop |
| `DR_EVAL_EVERY`, `DR_EVAL_PROTOCOL_IDS`, `DR_NUM_EVAL_SCENARIOS` | the fixed evaluation panel and its cadence |
| `DR_SAVE_METRIC=rollout` | select checkpoints on the panel, not on the training loss |
| `DR_MAX_DEFICIT_PU` | reject an evaluation that shed load |
| `DR_ROLLOUT_PARALLEL`, `DR_ROLLOUT_RETRY_FAILED` | pooled vs sequential evaluation, and whether a failed scenario is retried sequentially |
| `DR_PRETRAINED_MODEL`, `DR_SEED_BEST`, `DR_PARENT_REPRO_TOL` | the parent checkpoint, its recorded value, and how exactly it must reproduce |
| `DR_STAGE_SUMMARY` | where to write the machine-readable end-of-stage record |

`DR_STOP_AFTER_STALE_EVALS` exists but defaults off, and should stay off unless a
stage is expected to improve monotonically: raising the learning rate at a
restart reliably degrades the policy before it recovers, and a small stale count
terminates the stage inside that dip.
