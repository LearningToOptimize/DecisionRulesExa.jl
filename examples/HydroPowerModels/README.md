# HydroPowerModels Example

Multi-stage hydrothermal scheduling using DecisionRulesExa.jl with DC or AC OPF formulations.

## Problem description

A hydro-dominated power system (Bolivia test case) is operated over a planning horizon of up to 96 stages. At each stage, the operator must decide generator dispatch, reservoir outflows, and spillage subject to:

- **Power flow constraints** (DC linearization or full AC polar OPF)
- **Reservoir dynamics** (water balance with stochastic inflows)
- **Generator and transmission limits**

The TS-DDR policy (an LSTM network) predicts target reservoir levels at each stage. The deterministic-equivalent NLP projects these targets onto the feasible set via slack-penalized target constraints. Training uses envelope-theorem gradients: dual multipliers on the target constraints give the policy gradient without differentiating through the solver.

## Formulations

Set `FORMULATION` in `train_hydro_exa.jl`:

| Formulation | `FORMULATION` | Variables per stage | Description |
|---|---|---|---|
| DC OPF | `:dc` | ~500 | Linear power flow, fast solves |
| AC Polar OPF | `:ac_polar` | ~2000 | Full nonlinear AC power flow |

## Data

The `bolivia/` directory contains:

- `PowerModels.json` — power system topology (39 buses, 55 branches, 19 generators)
- `hydro.json` — hydro unit parameters (7 reservoirs)
- `inflows.csv` — historical inflow scenarios (144 stages x 200 scenarios x 7 reservoirs)
- `_demand.csv` — per-stage bus demand scaling

Pre-solved deterministic-equivalent references (MOF format) are provided for validation:
- `DCPPowerModel.mof.json`
- `ACPPowerModel.mof.json`

## Files

| File | Description |
|---|---|
| `train_hydro_exa.jl` | Main training script with penalty scheduling, parallel GPU solves, and W&B logging |
| `hydro_power_data.jl` | Data parsing (PowerModels JSON, hydro JSON, inflows CSV) |
| `hydro_power_exa.jl` | ExaModels problem builder for DC and AC OPF formulations |
| `eval_exa_de.jl` | Validation script comparing ExaModels results against JuMP reference |
| `Project.toml` | Example-specific dependencies (W&B, JLD2, CUDA, etc.) |

## Running

### GPU training (recommended)

```julia
# From this directory:
julia --project -t auto train_hydro_exa.jl
```

Set `USE_GPU = true` in `train_hydro_exa.jl` (default). Requires a CUDA-capable GPU.

### CPU training

Set `USE_GPU = false` in `train_hydro_exa.jl`, then run the same command.

### Configuration

Key parameters in `train_hydro_exa.jl`:

| Parameter | Default | Description |
|---|---|---|
| `FORMULATION` | `:ac_polar` | OPF formulation (`:dc` or `:ac_polar`) |
| `NUM_STAGES` | 96 | Planning horizon |
| `NUM_EPOCHS` | 20 | Training epochs |
| `NUM_BATCHES` | 100 | Gradient steps per epoch |
| `NUM_WORKERS` | 4 | Parallel GPU solver instances |
| `LAYERS` | `[128, 128]` | LSTM hidden layer sizes |
| `LR` | 1e-3 | Learning rate |
| `DEFICIT_COST` | 1e5 | Load-shedding penalty ($/pu) |

### Training features

- **Penalty scheduling**: target penalty multiplier ramps through phases (0.1 -> 1.0 -> 10.0 -> 30.0) over training
- **Sample scheduling**: `num_train_per_batch` increases from `NUM_WORKERS` to `8 * NUM_WORKERS`
- **Evaluation scheduling**: rollout evaluation starts with 4 scenarios and ramps to 32 at halfway
- **Parallel solves**: independent NLP copies solved concurrently via `Threads.@spawn` worker pool
- **Parallel rollout**: evaluation scenarios distributed across CPU stage-problem copies
- **W&B logging**: training loss, rollout objectives, violation share, penalty multiplier

## Validation

Compare the ExaModels formulation against a JuMP/MadNLP reference:

```julia
julia --project -t auto eval_exa_de.jl
```

This loads a pre-solved JuMP reference and solves the same problem in ExaModels, printing a side-by-side comparison of objectives and reservoir trajectories.
