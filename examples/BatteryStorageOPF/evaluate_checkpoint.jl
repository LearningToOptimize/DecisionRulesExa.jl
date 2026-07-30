#!/usr/bin/env julia
# evaluate_checkpoint.jl
#
# Evaluate a saved TS-DDR checkpoint using ONLY the saved artifacts. It does NOT
# regenerate the demand process from defaults or environment variables:
#
#   1. reconstruct the case + demand process FIELD-FOR-FIELD from the stochastic
#      manifest and verify the load-process content hash;
#   2. verify the evaluation protocol FILE bytes against the manifest hash, then
#      read the stored evaluation scenario-index matrix and verify ITS hash;
#   3. reload the policy from the checkpoint (case/source/process hashes checked);
#   4. run the non-anticipative stage-wise rollout on exactly those scenarios.
#
# Run (from examples/BatteryStorageOPF):
#   module load julia
#   BAT_MANIFEST=results/stochastic_manifest_case14_ieee.json \
#   BAT_EVAL_PROTOCOL=results/eval_protocol_case14_ieee.json \
#   BAT_CKPT=results/checkpoint_case14_ieee.jls \
#     julia --pkgimages=no --project=. evaluate_checkpoint.jl

include(joinpath(@__DIR__, "src", "BatteryStorageOPF.jl"))
using .BatteryStorageOPF
using MadNLP
using Flux
using Printf

const RES      = joinpath(@__DIR__, "results")
const CASE     = get(ENV, "BAT_CASE", "case14_ieee")
const MANIFEST = get(ENV, "BAT_MANIFEST", joinpath(RES, "stochastic_manifest_$(CASE).json"))
const EVALPROT = get(ENV, "BAT_EVAL_PROTOCOL", joinpath(RES, "eval_protocol_$(CASE).json"))
const CKPT     = get(ENV, "BAT_CKPT", joinpath(RES, "checkpoint_$(CASE).jls"))
const OUTDIR   = get(ENV, "BAT_OUTDIR", RES)

function main()
    for f in (MANIFEST, EVALPROT, CKPT)
        isfile(f) || error("missing artifact: $f (run run_tiny_training.jl first)")
    end

    # 1. Experiment definition, reconstructed exactly and hash-verified.
    case, process, meta = reconstruct_stochastic_manifest(MANIFEST)
    @printf("Reconstructed experiment from %s\n", MANIFEST)
    @printf("  load-process hash verified : %s\n", process_hash(process))
    @printf("  preset=%s  target_mode=%s  report=%d  lookahead=%d  dt=%.3g\n",
            process.preset, meta.mode, meta.reporting_horizon, meta.lookahead, meta.stage_hours)
    @printf("  active recourse=%.1f USD/MWh  rho1=%.3g rho2=%.3g  activation=%s\n",
            meta.active_recourse_cost_per_mwh, meta.rho1, meta.rho2, meta.activation)

    # 2. Evaluation protocol: verify FILE bytes, then the index-matrix content.
    file_hash = verify_protocol_file(EVALPROT, meta.eval_protocol_file_sha256)
    @printf("  eval protocol file hash    : %s (verified)\n", file_hash)
    proto_process, eval_mat, pmeta = reconstruct_scenario_protocol(EVALPROT)
    process_hash(proto_process) == process_hash(process) ||
        error("evaluation protocol describes a different demand process than the manifest")
    if meta.eval_index_matrix_hash !== nothing
        index_matrix_hash(eval_mat) == String(meta.eval_index_matrix_hash) ||
            error("evaluation index-matrix hash mismatch vs manifest")
    end
    @printf("  eval index-matrix hash     : %s (verified)  paths=%d horizon=%d\n",
            index_matrix_hash(eval_mat), pmeta.paths, pmeta.horizon)

    # 3. Policy from the checkpoint.
    policy, ck = load_checkpoint(CKPT, case, process)
    @printf("  checkpoint target_mode=%s activation=%s\n",
            ck["architecture"]["target_mode"], ck["architecture"]["activation"])

    # 4. Rollout on exactly the stored scenarios.
    stage = build_battery_stage_problem(case, process; mode = meta.mode,
                                        rho1 = meta.rho1, rho2 = meta.rho2,
                                        active_recourse_cost_per_mwh = meta.active_recourse_cost_per_mwh,
                                        stage_hours = meta.stage_hours)
    ev = evaluate_paired(policy, stage, process, eval_mat;
                         reporting_horizon = meta.reporting_horizon, keep_trajectories = true)
    @printf("\n  mean held-out physical cost = %.6f  (n_ok=%d, n_failed=%d)\n",
            ev.mean_reporting_physical_cost, ev.n_ok, ev.n_failed)
    @printf("  active deficit = %.6g MWh   active surplus = %.6g MWh   (max deficit %.6g / surplus %.6g pu)\n",
            ev.total_active_deficit_energy_mwh, ev.total_active_surplus_energy_mwh,
            ev.max_active_deficit_pu, ev.max_active_surplus_pu)
    for (p, c) in enumerate(ev.reporting_physical_costs)
        @printf("    path %-3d reporting physical cost = %.6f\n", p, c)
    end

    traj = joinpath(OUTDIR, "eval_trajectory_$(CASE).json")
    write_trajectory(traj, ev.trajectories;
                     meta = Dict("case" => CASE, "target_mode" => String(meta.mode),
                                 "reconstructed_from" => MANIFEST))
    @printf("  wrote %s\n", traj)
    return nothing
end

main()
