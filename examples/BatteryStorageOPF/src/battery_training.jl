# battery_training.jl
#
# Beginner-usable TS-DDR training and evaluation entrypoints for the
# battery-storage example (Phase-2 prompt §5), built on the parent
# DecisionRulesExa APIs (`train_tsddr`, `rollout_tsddr`) and the target-constrained
# problem in battery_tsddr.jl.
#
#   * `make_replay_sampler`   — deterministic scenario replay from a stored
#                               atom-index matrix (training uses the train
#                               protocol; evaluation uses the eval protocol).
#   * `train_battery_tsddr`   — one call that trains a reachable policy on the
#                               full-horizon target-constrained DE, auditing and
#                               COUNTING failed solves (never silently dropping).
#   * `evaluate_battery_policy` / `evaluate_paired` — honest, non-anticipative
#                               STAGE-WISE rollout on fixed paired scenarios, with
#                               the full physical/penalty cost decomposition and a
#                               compact machine-readable trajectory.
#   * checkpoint save/load with exact reload; trajectory serialization.
#
# Physical operating cost and training-only target penalty are kept strictly
# separate everywhere (see `decompose_costs`); improvement is always judged on
# physical operating cost, never on the total objective.

using DecisionRulesExa
using Flux
using MadNLP
using JSON
using Serialization
using Statistics

# ── Deterministic scenario replay ─────────────────────────────────────────────

"""
    make_replay_sampler(process, index_matrix) -> (sampler, scenarios)

Return a zero-argument `sampler()` that yields materialized `w_flat` vectors by
cycling deterministically through the columns of a stage-major `index_matrix`
(one scenario per column). The same matrix always replays the same sequence, so
training is exactly reproducible. `scenarios` is the pre-materialized vector.
"""
function make_replay_sampler(process::LoadProcess, index_matrix::AbstractMatrix{<:Integer})
    scenarios = materialize_all(process, index_matrix)
    npaths = length(scenarios)
    npaths >= 1 || error("index_matrix must have ≥ 1 path")
    i = Ref(0)
    sampler = function ()
        i[] += 1
        return copy(scenarios[((i[] - 1) % npaths) + 1])
    end
    return sampler, scenarios
end

# ── Training ──────────────────────────────────────────────────────────────────

"""
    train_battery_tsddr(policy, de, process, train_index_matrix; kwargs...) -> NamedTuple

Train a [`BatteryReachablePolicy`] on a full-horizon [`BatteryTSDDRProblem`] `de`
via the parent `train_tsddr`, replaying the training scenarios in
`train_index_matrix` deterministically.

# Keywords
- `initial_state = policy_initial_state(de.case)`: initial SoC state vector.
- `num_batches::Int = 20`, `num_train_per_batch::Int = 4`.
- `optimizer = Adam(1e-3)` (Flux optimizer/chain).
- `madnlp_kwargs = (print_level=MadNLP.ERROR, tol=1e-6)`.
- `record_loss`: `(iter, model, loss, tag) -> Bool` callback (default prints).
- `warmstart::Bool = true`, `retry_on_failure::Bool = true`.

Returns `(model, n_ok, n_total, n_failed, failure_counts)` — every failed solve
is counted and reported, never silently discarded. `n_ok`/`n_total` count solves
across all batches.
"""
function train_battery_tsddr(policy, de::BatteryTSDDRProblem, process::LoadProcess,
                             train_index_matrix::AbstractMatrix{<:Integer};
                             initial_state = policy_initial_state(de.case),
                             num_batches::Int = 20,
                             num_train_per_batch::Int = 4,
                             optimizer = Flux.Adam(1f-3),
                             madnlp_kwargs = (print_level = MadNLP.ERROR, tol = 1e-6),
                             record_loss = (iter, model, loss, tag) -> begin
                                 println("  iter=$iter  mean_physical_obj=$(round(loss; digits = 3))")
                                 return false
                             end,
                             warmstart::Bool = true,
                             retry_on_failure::Bool = true)
    is_targetless(de) &&
        error("train_battery_tsddr requires a target mode (:strict or :soft); " *
              "a targetless diagnostic problem has no target constraints to differentiate")
    size(train_index_matrix, 1) == de.horizon ||
        error("train_index_matrix has $(size(train_index_matrix,1)) stages but de.horizon=$(de.horizon)")
    sampler, _ = make_replay_sampler(process, train_index_matrix)

    failure_counts = Dict{String,Int}()
    n_ok = Ref(0); n_total = Ref(0)
    diag = (iter, stats) -> begin
        n_ok[] += get(stats, "n_ok", 0)
        n_total[] += get(stats, "n_total", 0)
        for (k, v) in get(stats, "failure_counts", Dict{String,Int}())
            failure_counts[k] = get(failure_counts, k, 0) + v
        end
        return nothing
    end

    train_tsddr(policy, initial_state, de, de.p_x0, de.p_target, de.p_w, sampler;
                num_batches = num_batches, num_train_per_batch = num_train_per_batch,
                optimizer = optimizer, madnlp_kwargs = madnlp_kwargs,
                warmstart = warmstart, retry_on_failure = retry_on_failure,
                batch_diagnostics = diag, record_loss = record_loss)

    return (model = policy, n_ok = n_ok[], n_total = n_total[],
            n_failed = n_total[] - n_ok[], failure_counts = failure_counts)
end

# ── Honest stage-wise rollout evaluation + compact trajectory ─────────────────

"""
    evaluate_battery_policy(policy, stage_problem, process, w_flat, atom_row;
        reporting_horizon, madnlp_kwargs, warmstart, retry_on_failure)
        -> Union{Nothing, NamedTuple}

Evaluate `policy` on ONE materialized scenario `w_flat` by a non-anticipative
STAGE-WISE rollout of the single-stage `stage_problem` (built with `horizon = 1`).
At each stage the policy sees only the current `w_t` and the realized SoC, sets a
one-stage-reachable target, and the stage AC-OPF is solved; the realized next SoC
feeds the next stage.

`atom_row` is the scenario's atom-index column (for the trajectory record).
`reporting_horizon` splits reported physical cost from the look-ahead buffer.

Returns `nothing` if any stage solve fails after retry, otherwise a NamedTuple:
- `reporting_physical_cost`, `lookahead_physical_cost`, `physical_operating_cost`,
  `generator_cost`, `battery_throughput_cost`, `target_penalty`;
- `final_soc`;
- `trajectory`: a vector of per-stage records (stage, atom, soc_in, target,
  soc_out, p_charge, p_discharge, generator_cost, physical_cost, target_penalty,
  status, max_primal_residual, max_balance_residual).

Uses the parent `rollout_tsddr` for the vetted stage loop (retry, projection),
capturing per-stage physical costs and solutions through its callbacks.
"""
function evaluate_battery_policy(policy, stage_problem::BatteryTSDDRProblem,
                                 process::LoadProcess, w_flat::AbstractVector,
                                 atom_row::AbstractVector{<:Integer};
                                 reporting_horizon::Int,
                                 madnlp_kwargs = (print_level = MadNLP.ERROR, tol = 1e-6),
                                 warmstart::Bool = false,
                                 retry_on_failure::Bool = true)
    stage_problem.horizon == 1 ||
        error("evaluate_battery_policy needs a single-stage stage_problem (horizon=1)")
    is_targetless(stage_problem) &&
        error("evaluate_battery_policy requires a target mode; got a targetless diagnostic")
    nw = stage_problem.nw
    T = length(w_flat) ÷ nw
    length(w_flat) == T * nw || error("w_flat length not a multiple of nw=$nw")
    length(atom_row) == T || error("atom_row length $(length(atom_row)) ≠ horizon $T")
    1 <= reporting_horizon <= T ||
        error("reporting_horizon must satisfy 1 ≤ R ≤ $T; got $reporting_horizon")
    nK = stage_problem.nBat
    e0 = policy_initial_state(stage_problem.case; float_type = Float32)

    # Per-stage capture buffers (filled inside the rollout callbacks).
    stage_ctr = Ref(0)
    phys_stage = zeros(Float64, T)
    gen_stage = zeros(Float64, T)
    cyc_stage = zeros(Float64, T)
    rec_cost_stage = zeros(Float64, T)
    deficit_mwh_stage = zeros(Float64, T)
    surplus_mwh_stage = zeros(Float64, T)
    max_deficit_stage = zeros(Float64, T)
    max_surplus_stage = zeros(Float64, T)
    raw_rec_cost_stage = zeros(Float64, T)
    proj_corr_stage = zeros(Float64, T)
    lb_viol_stage = zeros(Float64, T)
    pen_stage = zeros(Float64, T)
    viol_stage = zeros(Float64, T)
    status_stage = Vector{Any}(undef, T)
    prim_stage = zeros(Float64, T)
    bal_stage = zeros(Float64, T)
    pch_stage = [zeros(Float64, nK) for _ in 1:T]
    pdis_stage = [zeros(Float64, nK) for _ in 1:T]
    soc_out_stage = [zeros(Float64, nK) for _ in 1:T]

    set_params! = function (prob, state, w_t, target, stage)
        stage_ctr[] = stage
        set_tsddr_initial_soc!(prob, state)
        set_tsddr_uncertainty!(prob, w_t)   # also realizes p_pd/p_qd for this stage
        set_tsddr_targets!(prob, target)
        return nothing
    end
    # Physical stage cost (generator + throughput + active-recourse cost; the
    # training-only target penalty is stripped). Also records the decomposition,
    # keeping the two recourse directions (deficit d⁺ and surplus d⁻) separate.
    no_penalty = function (prob, result)
        s = stage_ctr[]
        d = decompose_costs(prob, result)
        gen_stage[s] = d.generator_cost
        cyc_stage[s] = d.battery_throughput_cost
        rec_cost_stage[s] = d.active_recourse_cost
        deficit_mwh_stage[s] = d.active_deficit_energy_mwh
        surplus_mwh_stage[s] = d.active_surplus_energy_mwh
        max_deficit_stage[s] = d.max_active_deficit_pu
        max_surplus_stage[s] = d.max_active_surplus_pu
        raw_rec_cost_stage[s] = d.raw_active_recourse_cost
        proj_corr_stage[s] = d.active_recourse_projection_correction
        lb_viol_stage[s] = d.maximum_active_recourse_lower_bound_violation_pu
        phys_stage[s] = d.physical_operating_cost
        pen_stage[s] = d.target_penalty
        viol_stage[s] = d.target_violation
        status_stage[s] = result.status
        prim_stage[s] = tsddr_max_primal_residual(prob, result)
        solv = tsddr_solution(prob, result)
        bal_stage[s] = nK > 0 ? maximum(abs, tsddr_balance_residuals(prob, solv)) : 0.0
        if nK > 0
            pch_stage[s] .= Float64.(solv.p_ch[:, 1])
            pdis_stage[s] .= Float64.(solv.p_dis[:, 1])
            soc_out_stage[s] .= Float64.(solv.soc[:, 2])
        end
        return d.physical_operating_cost
    end
    realized = function (prob, result)
        # The realized next SoC (e[:,2]) is the state passed to the next stage.
        solv = tsddr_solution(prob, result)
        return nK > 0 ? Float32.(solv.soc[:, 2]) : Float32[]
    end

    out = rollout_tsddr(policy, e0, stage_problem, Float64.(w_flat);
                        horizon = T, n_uncertainty = nw,
                        set_stage_parameters! = set_params!,
                        realized_state = realized,
                        objective_no_target_penalty = no_penalty,
                        madnlp_kwargs = madnlp_kwargs,
                        warmstart = warmstart, policy_state = :realized,
                        retry_on_failure = retry_on_failure)
    out === nothing && return nothing

    # Assemble the compact trajectory and the reporting/look-ahead split.
    traj = Vector{Dict{String,Any}}(undef, T)
    for s in 1:T
        soc_in = Float64.(out.state_trajectory[s])
        target = Float64.(out.target_trajectory[s])
        traj[s] = Dict{String,Any}(
            "stage" => s,
            "atom" => Int(atom_row[s]),
            "soc_in" => soc_in,
            "target" => target,
            "soc_out" => soc_out_stage[s],
            "p_charge" => pch_stage[s],
            "p_discharge" => pdis_stage[s],
            "generator_cost" => gen_stage[s],
            "battery_throughput_cost" => cyc_stage[s],
            "active_recourse_cost" => rec_cost_stage[s],
            "active_deficit_energy_mwh" => deficit_mwh_stage[s],
            "active_surplus_energy_mwh" => surplus_mwh_stage[s],
            "max_active_deficit_pu" => max_deficit_stage[s],
            "max_active_surplus_pu" => max_surplus_stage[s],
            "raw_active_recourse_cost" => raw_rec_cost_stage[s],
            "active_recourse_projection_correction" => proj_corr_stage[s],
            "maximum_active_recourse_lower_bound_violation_pu" => lb_viol_stage[s],
            "physical_cost" => phys_stage[s],
            "target_penalty" => pen_stage[s],
            "target_violation" => viol_stage[s],
            "status" => string(status_stage[s]),
            "max_primal_residual" => prim_stage[s],
            "max_balance_residual" => bal_stage[s],
        )
    end
    reporting = sum(@view phys_stage[1:reporting_horizon])
    lookahead = T > reporting_horizon ? sum(@view phys_stage[reporting_horizon+1:T]) : 0.0
    return (reporting_physical_cost = reporting,
            lookahead_physical_cost = lookahead,
            physical_operating_cost = sum(phys_stage),
            generator_cost = sum(gen_stage),
            battery_throughput_cost = sum(cyc_stage),
            active_recourse_cost = sum(rec_cost_stage),
            active_deficit_energy_mwh = sum(deficit_mwh_stage),
            active_surplus_energy_mwh = sum(surplus_mwh_stage),
            total_active_recourse_energy_mwh = sum(deficit_mwh_stage) + sum(surplus_mwh_stage),
            max_active_deficit_pu = maximum(max_deficit_stage),
            max_active_surplus_pu = maximum(max_surplus_stage),
            raw_active_recourse_cost = sum(raw_rec_cost_stage),
            active_recourse_projection_correction = sum(proj_corr_stage),
            maximum_active_recourse_lower_bound_violation_pu = maximum(lb_viol_stage),
            target_penalty = sum(pen_stage),
            target_violation = sum(viol_stage),
            final_soc = Float64.(out.final_state),
            trajectory = traj)
end

"""
    evaluate_paired(policy, stage_problem, process, index_matrix;
        reporting_horizon, madnlp_kwargs, keep_trajectories=false) -> NamedTuple

Evaluate `policy` on the fixed paired scenario set defined by a stage-major
`index_matrix` (every method shares the SAME matrix), returning:
- `mean_reporting_physical_cost` (over successful paths),
- `reporting_physical_costs` (per successful path),
- `n_ok`, `n_failed`, `failed_paths`,
- `trajectories` (per path) when `keep_trajectories=true`.

Improvement is judged on `mean_reporting_physical_cost` — the physical operating
cost over the reporting horizon, with NO target penalty.
"""
function evaluate_paired(policy, stage_problem::BatteryTSDDRProblem, process::LoadProcess,
                         index_matrix::AbstractMatrix{<:Integer};
                         reporting_horizon::Int,
                         madnlp_kwargs = (print_level = MadNLP.ERROR, tol = 1e-6),
                         keep_trajectories::Bool = false)
    horizon, npaths = size(index_matrix)
    costs = Float64[]
    deficit_mwh = Float64[]
    surplus_mwh = Float64[]
    max_deficit = Float64[]
    max_surplus = Float64[]
    raw_rec_cost = Float64[]
    proj_corr = Float64[]
    lb_viol = Float64[]
    failed_paths = Int[]
    trajectories = keep_trajectories ? Vector{Any}(undef, npaths) : nothing
    for p in 1:npaths
        w_flat = materialize_scenario(process, view(index_matrix, :, p); horizon = horizon)
        res = evaluate_battery_policy(policy, stage_problem, process, w_flat,
                                      view(index_matrix, :, p);
                                      reporting_horizon = reporting_horizon,
                                      madnlp_kwargs = madnlp_kwargs)
        if res === nothing
            push!(failed_paths, p)
            keep_trajectories && (trajectories[p] = nothing)
        else
            push!(costs, res.reporting_physical_cost)
            push!(deficit_mwh, res.active_deficit_energy_mwh)
            push!(surplus_mwh, res.active_surplus_energy_mwh)
            push!(max_deficit, res.max_active_deficit_pu)
            push!(max_surplus, res.max_active_surplus_pu)
            push!(raw_rec_cost, res.raw_active_recourse_cost)
            push!(proj_corr, res.active_recourse_projection_correction)
            push!(lb_viol, res.maximum_active_recourse_lower_bound_violation_pu)
            keep_trajectories && (trajectories[p] = res)
        end
    end
    tot_def = isempty(deficit_mwh) ? 0.0 : sum(deficit_mwh)
    tot_sur = isempty(surplus_mwh) ? 0.0 : sum(surplus_mwh)
    return (mean_reporting_physical_cost = isempty(costs) ? NaN : mean(costs),
            reporting_physical_costs = costs,
            n_ok = length(costs), n_failed = length(failed_paths),
            failed_paths = failed_paths,
            total_active_deficit_energy_mwh = tot_def,
            total_active_surplus_energy_mwh = tot_sur,
            total_active_recourse_energy_mwh = tot_def + tot_sur,
            max_active_deficit_pu = isempty(max_deficit) ? 0.0 : maximum(max_deficit),
            max_active_surplus_pu = isempty(max_surplus) ? 0.0 : maximum(max_surplus),
            total_raw_active_recourse_cost = isempty(raw_rec_cost) ? 0.0 : sum(raw_rec_cost),
            total_active_recourse_projection_correction = isempty(proj_corr) ? 0.0 : sum(proj_corr),
            maximum_active_recourse_lower_bound_violation_pu = isempty(lb_viol) ? 0.0 : maximum(lb_viol),
            trajectories = trajectories)
end

# ── Checkpointing (exact reload) ──────────────────────────────────────────────

"""
    battery_checkpoint(policy, de; case, process, extra=Dict()) -> Dict

Assemble a checkpoint dictionary that identifies everything needed to reload the
policy and reproduce its outputs: the Flux model state, the architecture and
target mode, the reporting/look-ahead horizons, the train/eval seeds, and the
case-manifest / MATPOWER-source / load-process hashes.
"""
function battery_checkpoint(policy::BatteryReachablePolicy, de::BatteryTSDDRProblem;
                            case::BatteryCase, process::LoadProcess,
                            extra::AbstractDict = Dict{String,Any}())
    layers = _encoder_layer_sizes(policy)
    combiner = _combiner_layer_sizes(policy)
    doc = Dict{String,Any}(
        "schema" => "battery_tsddr_checkpoint/1",
        "flux_state" => Flux.state(policy),
        "architecture" => Dict{String,Any}(
            "policy" => "BatteryReachablePolicy",
            "n_uncertainty" => policy.n_uncertainty,
            "nbat" => policy.nbat,
            "encoder_layers" => layers,
            "combiner_layers" => combiner,
            "target_mode" => String(de.mode),
            "dt" => de.dt,
            # Target-head activation and its safe upper margin: the normalized
            # target never reaches exactly 1 (interior-point degeneracy under a
            # strict equality). Checkpoints are only comparable across runs with
            # the same activation.
            "activation" => string(_activation_name(policy)),
            "safe_upper_margin" => 1e-3,
        ),
        "horizons" => Dict{String,Any}(
            "reporting_horizon" => de.reporting_horizon,
            "lookahead" => de.lookahead,
            "horizon" => de.horizon,
        ),
        "seeds" => Dict{String,Any}(
            "train_seed" => process.train_seed,
            "eval_seed" => process.eval_seed,
        ),
        "hashes" => Dict{String,Any}(
            "case_manifest_content_hash" => manifest_hash(case),
            "matpower_sha256" => case.parse_meta.matpower_sha256,
            "load_process_hash" => process_hash(process),
        ),
        # Training-only target-penalty coefficients (soft mode) and the price used
        # by the physical two-sided active-recourse cost.
        "target_penalty" => Dict{String,Any}("rho1" => de.rho1, "rho2" => de.rho2),
        "active_recourse_cost_per_mwh" => de.active_recourse_cost_per_mwh,
    )
    for (k, v) in extra
        doc[k] = v
    end
    return doc
end

"""
    save_checkpoint(path, policy, de; case, process, extra=Dict()) -> String

Serialize a [`battery_checkpoint`](@ref) to `path` with the `Serialization`
stdlib. Returns `path`.
"""
function save_checkpoint(path::AbstractString, policy::BatteryReachablePolicy,
                         de::BatteryTSDDRProblem; case::BatteryCase, process::LoadProcess,
                         extra::AbstractDict = Dict{String,Any}())
    doc = battery_checkpoint(policy, de; case = case, process = process, extra = extra)
    open(io -> Serialization.serialize(io, doc), path, "w")
    return path
end

"""
    load_checkpoint(path, case, process; dt=nothing, float_type=Float32)
        -> (policy, meta)

Rebuild a [`BatteryReachablePolicy`] from a checkpoint written by
[`save_checkpoint`](@ref): construct a fresh policy with the recorded architecture
(for `case`/`process`), then load the saved Flux state into it exactly. Verifies
that the recorded case-manifest, MATPOWER-source, and load-process hashes match
the supplied `case`/`process`. Reloading reproduces the policy's outputs exactly
on any fixed CPU input. `meta` is the checkpoint dict (without the raw state).
"""
function load_checkpoint(path::AbstractString, case::BatteryCase, process::LoadProcess;
                         dt::Union{Nothing,Real} = nothing,
                         float_type::Type{<:AbstractFloat} = Float32)
    doc = open(Serialization.deserialize, path)
    arch = doc["architecture"]
    h = doc["hashes"]
    manifest_hash(case) == String(h["case_manifest_content_hash"]) ||
        error("checkpoint case-manifest hash mismatch")
    case.parse_meta.matpower_sha256 == String(h["matpower_sha256"]) ||
        error("checkpoint MATPOWER-source hash mismatch")
    process_hash(process) == String(h["load_process_hash"]) ||
        error("checkpoint load-process hash mismatch")

    Δt = dt === nothing ? Float64(arch["dt"]) : Float64(dt)
    policy = battery_reachable_policy(case, process;
                                      dt = Δt,
                                      layers = Int.(arch["encoder_layers"]),
                                      combiner_layers = Int.(arch["combiner_layers"]),
                                      activation = _activation_from_name(String(arch["activation"])),
                                      float_type = float_type)
    load_battery_policy!(policy, doc["flux_state"])
    meta = Dict(k => v for (k, v) in doc if k != "flux_state")
    return policy, meta
end

# Target-head activation recorded in (and restored from) the checkpoint. The
# head applies the bounded activation at every layer, so reading it off the
# output layer identifies the whole head.
function _activation_name(policy::BatteryReachablePolicy)
    comb = policy.combiner
    layer = comb isa Flux.Dense ? comb : comb.layers[end]
    return layer.σ
end

# Resolve a recorded activation name back to the callable.
function _activation_from_name(name::AbstractString)
    name == string(stretchedsigmoid) && return stretchedsigmoid
    name == string(hardsigmoidsafe) && return hardsigmoidsafe
    error("unknown target activation \"$name\" in checkpoint; expected " *
          "$(string(stretchedsigmoid)) or $(string(hardsigmoidsafe))")
end

# Encoder/combiner size introspection for the checkpoint architecture record.
# The encoder is a Chain of recurrent layers; the combiner is a Dense or a Chain
# of Denses. Sizes are read from the weight matrices so a reloaded policy is
# rebuilt with an identical structure.
function _encoder_layer_sizes(policy::BatteryReachablePolicy)
    sizes = Int[]
    for layer in policy.encoder.layers
        cell = DecisionRulesExa._as_cell(layer)
        # LSTM/GRU/RNN cell Wh is (factor·hidden) × hidden; hidden = out features.
        push!(sizes, size(cell.Wh, 2))
    end
    return sizes
end

function _combiner_layer_sizes(policy::BatteryReachablePolicy)
    comb = policy.combiner
    comb isa Flux.Dense && return Int[]        # single linear head → no hidden layers
    sizes = Int[]
    layers = comb.layers
    for i in 1:length(layers) - 1              # all but the output layer are "hidden"
        push!(sizes, size(layers[i].weight, 1))
    end
    return sizes
end

# ── Trajectory serialization ──────────────────────────────────────────────────

"""
    write_trajectory(path, evaluations; meta=Dict()) -> String

Write a compact, machine-readable JSON trajectory file. `evaluations` is a vector
whose entries are either `nothing` (a failed path) or the NamedTuple returned by
[`evaluate_battery_policy`](@ref). Each successful path stores its per-stage
records (stage & atom indices, battery SoC in/out, targets, charge/discharge,
generator and physical costs, target penalty, solver status, residuals) plus its
reporting / look-ahead physical costs. Returns `path`.
"""
function write_trajectory(path::AbstractString, evaluations::AbstractVector; meta::AbstractDict = Dict{String,Any}())
    paths = Any[]
    for (p, ev) in enumerate(evaluations)
        if ev === nothing
            push!(paths, Dict{String,Any}("path" => p, "status" => "failed"))
        else
            push!(paths, Dict{String,Any}(
                "path" => p,
                "status" => "ok",
                "reporting_physical_cost" => ev.reporting_physical_cost,
                "lookahead_physical_cost" => ev.lookahead_physical_cost,
                "physical_operating_cost" => ev.physical_operating_cost,
                "generator_cost" => ev.generator_cost,
                "battery_throughput_cost" => ev.battery_throughput_cost,
                "active_recourse_cost" => ev.active_recourse_cost,
                "active_deficit_energy_mwh" => ev.active_deficit_energy_mwh,
                "active_surplus_energy_mwh" => ev.active_surplus_energy_mwh,
                "total_active_recourse_energy_mwh" => ev.total_active_recourse_energy_mwh,
                "max_active_deficit_pu" => ev.max_active_deficit_pu,
                "max_active_surplus_pu" => ev.max_active_surplus_pu,
                "raw_active_recourse_cost" => ev.raw_active_recourse_cost,
                "active_recourse_projection_correction" => ev.active_recourse_projection_correction,
                "maximum_active_recourse_lower_bound_violation_pu" => ev.maximum_active_recourse_lower_bound_violation_pu,
                "target_penalty" => ev.target_penalty,
                "target_violation" => ev.target_violation,
                "final_soc" => ev.final_soc,
                "stages" => ev.trajectory,
            ))
        end
    end
    doc = Dict{String,Any}("schema" => "battery_tsddr_trajectory/1", "paths" => paths)
    for (k, v) in meta
        doc[k] = v
    end
    open(io -> JSON.print(io, doc, 2), path, "w")
    return path
end
