# Bolivia hydro — ExaModels engine

The GPU half of the long-term hydrothermal planning case study: this directory
trains and evaluates the policy. The case, the SDDP baseline and the figures live
in the companion package, `DecisionRules.jl/examples/HydroPowerModels`.

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
| `lineage_from_scratch.json` | the published from-scratch training schedule, as data: one entry per phase |
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
DR_ENABLE_WANDB=false \
  julia --project=. -t auto train_hydro_exa_strict.jl
```

**3. The full from-scratch training recipe.** This is the published schedule,
declared in `lineage_from_scratch.json` and executed stage by stage:

```bash
julia --project=. run_tsddr_lineage.jl                 # full lineage
julia --project=. run_tsddr_lineage.jl --dry-run       # print the plan only
julia --project=. run_tsddr_lineage.jl --stages=phase3  # resume one phase
```

Each stage runs as its own process, so a stage boundary is a real restart: the
optimizer state, the cosine learning-rate phase and the warm-up counter all
begin again. The driver chains only checkpoints that a COMPLETE, non-shedding
panel evaluation selected, hashes every parent before use, refuses `_latest`
snapshots outright, and stops the lineage — rather than falling back — if a
stage produces nothing selectable. Re-running resumes: a stage whose record
exists and whose checkpoint still hashes correctly is skipped.

Records land in `bolivia/ACPPowerModel/lineage/`: one JSON per stage with the
resolved environment, ancestry, checkpoint hashes, update count and both the
process wall time and the trainer's own training-loop seconds, plus a
lineage-level ledger.

Neither W&B nor a workload manager is required. `DR_ENABLE_WANDB=false` turns
logging off; nothing in the driver reads a scheduler variable.

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

## Configuration surface of one training stage

`train_hydro_exa_strict.jl` is driven entirely by environment variables; the
lineage driver simply sets them. The ones that define a stage:

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
