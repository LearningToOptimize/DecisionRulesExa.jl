# HydroPowerModels Example

Multi-stage hydrothermal scheduling using DecisionRulesExa.jl with DC or AC
OPF formulations.  This is the GPU-accelerated counterpart of the
[DecisionRules.jl hydro example](https://github.com/LearningToOptimize/DecisionRules.jl/tree/main/examples/HydroPowerModels).

## Problem description

A hydro-dominated power system (Bolivia test case) is operated over a
planning horizon of up to 126 stages (96 operational + 30 look-ahead).
At each stage, the operator must decide generator dispatch, reservoir
outflows, and spillage subject to:

- **Power flow constraints** (DC linearization or full AC polar OPF)
- **Reservoir dynamics** (water balance with stochastic inflows)
- **Generator and transmission limits**

The TS-DDR policy (an LSTM network) predicts target reservoir levels at
each stage.  The deterministic-equivalent NLP projects these targets onto
the feasible set.  Training uses envelope-theorem gradients: dual
multipliers on the target constraints give the policy gradient without
differentiating through the solver.

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
- `_demand.csv` — inactive per-stage demand candidate; rename to `demand.csv`
  when you intentionally want training scripts to use it

Pre-solved deterministic-equivalent references (MOF format) are provided for validation:
- `DCPPowerModel.mof.json`
- `ACPPowerModel.mof.json`

## Files

| File | Description |
|---|---|
| `Project.toml` | Example-specific environment with CUDA, CUDSS, MadNLPGPU, W&B, and JLD2 |
| `README.md` | This guide |
| `bolivia/PowerModels.json` | Bolivia network topology and generator data |
| `bolivia/hydro.json` | Hydro unit limits, initial volumes, cascade metadata, and stage duration |
| `bolivia/inflows.csv` | Historical inflow scenarios |
| `bolivia/_demand.csv` | Inactive per-stage demand candidate; ignored unless renamed to `demand.csv` |
| `bolivia/DCPPowerModel.mof.json` | Pre-exported DC OPF stage template |
| `bolivia/ACPPowerModel.mof.json` | Pre-exported AC polar OPF stage template |
| `bolivia/SOCWRConicPowerModel.mof.json` | Convex relaxation template used for validation/baselines |
| `hydro_power_data.jl` | Data parsing for PowerModels JSON, hydro JSON, inflows, and demand |
| `hydro_power_exa.jl` | Regular open-loop ExaModels DE builder; supports `strict_targets=true` |
| `hydro_reachable_policy.jl` | Reachable hydro target policy, cascade clamps, and reachability proof sketches |
| `hydro_power_exa_embedded.jl` | Embedded-policy DE builder and nonlinear oracle |
| `hydro_training_utils.jl` | Shared training-entrypoint helpers such as environment layer parsing |
| `train_hydro_exa.jl` | Open-loop DE training with penalty scheduling, parallel GPU solves, and W&B logging |
| `train_hydro_exa_embedded.jl` | Embedded (closed-loop) DE training; supports `DR_STRICT_EMBEDDED_TARGETS=true` for penalty-free strict mode |
| `train_hydro_exa_strict.jl` | Regular strict DE training using reachable-policy target rollout and no target slack penalty |
| `train_hydro_exa_critic.jl` | Critic/control-variate variant; adds a scalar critic with replay buffer and cheap rollout samples |
| `eval_exa_de.jl` | Validation script comparing ExaModels results against JuMP reference |

## Running

### Strict regular DE training (recommended for the current experiments)

```bash
DR_ENCODER_LAYERS=128,128 \
DR_HEAD_LAYERS=128,128 \
julia --project -t auto train_hydro_exa_strict.jl
```

This path removes all target slack penalties from the regular deterministic
equivalent. The target trajectory is generated before the solve, but it is not a
generic open-loop trajectory: `HydroReachablePolicy` starts from the known
initial reservoir state and feeds each previous target into the next policy call.
Because every emitted target is one-stage reachable from the previous target,
the full strict DE target path is feasible by induction. The strict solve then
forces the realized reservoir path to equal that reachable target path.

Use:

- `DR_ENCODER_LAYERS`: recurrent LSTM layers over inflows only.
- `DR_HEAD_LAYERS`: optional feed-forward hidden layers over
  `[encoded_inflow; reservoir_state]`. This is the nonlinear state-to-target
  map; it does not add recurrence over the state input.

### Strict embedded training

```bash
DR_STRICT_EMBEDDED_TARGETS=true julia --project -t auto train_hydro_exa_embedded.jl
```

Strict mode embeds the policy inside the NLP and enforces hard equality
targets — no penalty tuning needed.  Requires a reachable-set policy
(built automatically from the hydro data). The current embedded oracle manually
codes the reachable-policy Jacobian for the default single Dense target head;
use the regular strict DE script for multilayer state-conditioned heads unless
the embedded oracle is extended.

### Open-loop DE training (GPU)

```bash
julia --project -t auto train_hydro_exa.jl
```

Set `USE_GPU = true` in `train_hydro_exa.jl` (default). Requires a CUDA-capable GPU.

### GPU training with critic control variate

```bash
julia --project -t auto train_hydro_exa_critic.jl
```

The critic script keeps the dual-multiplier actor update but adds a damped
control variate (`critic_cv_weight = 0.5`) trained on the stage-wise rollout
objective without target penalty.

### CPU training

Set `USE_GPU = false` in the training script, then run the same command.

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
| `DR_HEAD_LAYERS` | `""` | Optional nonrecurrent state-conditioned target-head hidden sizes |
| `LR` | 1e-3 | Learning rate |
| `DEFICIT_COST` | 1e5 | Load-shedding penalty ($/pu) |
| `DR_TARGET_PENALTY_MULT` | `8.0` | Bolivia hydro default multiplier applied to the `:auto` target-penalty coefficients |
| `DR_PENALTY_SCHEDULE` | `const` | Penalty schedule; `const` uses `DR_TARGET_PENALTY_MULT` at every stage |

### Bolivia target-penalty calibration

The hydro builders keep `target_penalty = :auto`; `:auto` is the package helper
that computes the base L2/L1 target-slack penalty coefficients from the case
data. For the Bolivia AC hydro case, the case scripts then multiply those base
coefficients by `DR_TARGET_PENALTY_MULT`, which defaults to `8.0`.

This default came from the SDDP bridge diagnostic in
`compare_sddp_policy_rollout.jl`. We loaded the trained SDDP cuts, simulated one
scenario, took the SDDP stage-1 reservoir `out` states as targets, and solved the
corresponding one-stage Exa target-penalty problem. The first multiplier that
enforced the SDDP targets to solver tolerance was `8.0`; larger multipliers did
not materially improve target matching and started to worsen numerical behavior.

Representative stage-1 sweep:

| Multiplier | Exa no-target obj. | Gap vs SDDP | Target violation share | Max target slack |
|---:|---:|---:|---:|---:|
| 1 | 1568.300 | -1599.767 | 1.62e-1 | 2.018e-2 |
| 3 | 1648.143 | -1519.924 | 3.25e-1 | 1.995e-2 |
| 5 | 1669.268 | -1498.799 | 4.38e-1 | 1.994e-2 |
| 6 | 2727.777 | -440.291 | 1.29e-1 | 7.877e-3 |
| 7 | 3082.389 | -85.679 | 2.53e-2 | 1.345e-3 |
| 8 | 3163.132 | -4.936 | 5.38e-7 | 1.159e-8 |
| 10 | 3163.129 | -4.939 | 2.77e-7 | 3.105e-9 |

The same SDDP-target bridge was also run as a horizon-10 deterministic
equivalent: all ten SDDP reservoir `out` states were injected as Exa targets and
the Exa problem was solved once over the ten-stage horizon. With multiplier
`8.0` and no target-penalty discount (`gamma = 1.0`), the target trajectory
matched to numerical tolerance and the Exa no-target objective was within about
`49.4` of the SDDP rollout objective over a `31668.1` objective. Mild discounting
at `gamma = 0.99` was similar; stronger discounting degraded target enforcement.

| Horizon | Multiplier | Penalty gamma | Exa no-target obj. | Gap vs SDDP | Target violation share | Max target slack |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 8 | 1.00 | 31618.626 | -49.434 | 7.85e-8 | 1.18e-8 |
| 10 | 8 | 0.99 | 31618.629 | -49.431 | 1.87e-7 | 6.37e-8 |
| 10 | 8 | 0.97 | 31617.250 | -50.810 | 3.77e-5 | 2.27e-5 |
| 10 | 8 | 0.95 | 31127.120 | -540.940 | 1.39e-2 | 1.02e-2 |

The full 126-stage SDDP-target deterministic equivalent gives the same
directional result. Flat multiplier `8.0` enforces the complete target
trajectory to numerical tolerance; discounting the target penalty creates
increasing target leakage and worsens the no-target objective gap.

| Horizon | Multiplier | Penalty gamma | Exa no-target obj. | Gap vs SDDP | Target violation share | Max target slack |
|---:|---:|---:|---:|---:|---:|---:|
| 126 | 8 | 1.00 | 375724.588 | -615.045 | 6.65e-8 | 6.32e-9 |
| 126 | 8 | 0.99 | 375721.696 | -617.938 | 7.54e-6 | 1.45e-4 |
| 126 | 8 | 0.97 | 375477.644 | -861.990 | 2.86e-4 | 4.47e-3 |
| 126 | 8 | 0.95 | 375175.342 | -1164.292 | 4.89e-4 | 1.55e-2 |

### Strict reachable hydro targets

The hydro builders support strict target mode when paired with
`HydroReachablePolicy`:

```julia
policy = hydro_reachable_policy(hydro_data, LAYERS;
    activation = sigmoid,
    encoder_type = Flux.LSTM,
    combiner_layers = HEAD_LAYERS,
)

prob = build_embedded_hydro_de(policy, power_data, hydro_data, T;
    formulation = :ac_polar,
    strict_targets = true,
)
```

The training script exposes the same path with:

```bash
export DR_STRICT_EMBEDDED_TARGETS=true
julia --project -t auto train_hydro_exa_embedded.jl
```

In embedded strict mode the oracle enforces

```text
reservoir[t+1, r] = pi_theta(inflow[t], reservoir[t])[r]
```

directly. No `delta_pos`, `delta_neg`, L2 target penalty, or L1 target penalty
is created for the target equality. This is intended for closed-loop embedded
training: the policy is evaluated with the realized previous reservoir state, so
it can produce a physically reachable next reservoir by construction.

In regular strict DE, `train_hydro_exa_strict.jl` instead constructs the target
trajectory externally:

```text
target[0] = initial_volume
target[t] = pi_theta(inflow[t], target[t-1])
```

Since `pi_theta` maps into the one-stage reachable set from its input state, the
target trajectory is feasible by induction. This is the specific reason strict
mode is valid for regular DE here, despite regular DE target generation usually
being open-loop after the initial state.

For the current hydro water balance,

```text
x[t+1, r] = x[t, r] - K*outflow[t, r] - spill[t, r]
            + K*inflow[t, r] + upstream_contributions[t, r],
```

the helper maps the network's normalized output `y in [0, 1]` into a conservative
one-stage reachable interval:

```text
upper_r = min(max_vol_r, x_r + K*inflow_r - K*min_turn_r)

lower_r = min_vol_r
```

The lower bound is `min_vol` because this data model has nonnegative spill with
no finite upper bound. If a user supplies finite spill caps through
`spill_max`, the helper instead uses:

```text
lower_r = max(min_vol_r,
              x_r + K*inflow_r - K*max_turn_r - spill_max_r)
```

and still emits:

```text
target_r = lower_r + (upper_r - lower_r) * y_r.
```

This is a smooth affine map in the network output, so incoming gradients are not
destroyed by post-hoc clipping. During the actor update, the reachability bounds
are treated as constants because they depend on realized state, inflow, and case
limits rather than trainable weights; the incoming adjoint still reaches the
network as `(upper_r - lower_r) * adjoint_r`. The bound derivatives with respect
to the previous reservoir state are included in the embedded oracle Jacobian and
VJP for the NLP solve.

The helper is deliberately conservative for cascaded reservoirs: it does not
rely on optional upstream releases to make a target feasible. If a new hydro
case has finite spill limits, nonlinear transition physics, travel times, or
other coupled state-transition constraints, the strict option should only be
used after providing a case-specific reachable-set map. Otherwise keep the
slack-penalty builder and tune the target penalty.

### Training features

- **Penalty scheduling**: default is constant multiplier `8.0` on top of the
  `:auto` base penalty; annealed schedules remain available as explicit
  ablations
- **Sample scheduling**: `num_train_per_batch` increases from `NUM_WORKERS` to `8 * NUM_WORKERS`
- **Evaluation scheduling**: rollout evaluation starts with 4 scenarios and ramps to 32 at halfway
- **Parallel solves**: independent NLP copies solved concurrently via `Threads.@spawn` worker pool
- **Parallel rollout**: evaluation scenarios distributed across CPU stage-problem copies
- **Critic variant**: optional scalar critic with value and gradient matching,
  replay-buffer training, and cheap critic actor samples
- **W&B logging**: training loss, rollout objectives, violation share, penalty multiplier

## Validation

Compare the ExaModels formulation against a JuMP/MadNLP reference:

```julia
julia --project -t auto eval_exa_de.jl
```

This loads a pre-solved JuMP reference and solves the same problem in ExaModels, printing a side-by-side comparison of objectives and reservoir trajectories.
