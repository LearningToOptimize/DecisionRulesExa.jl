# Bolivia hydro example (ExaModels)

This directory is the ExaModels counterpart of the canonical Bolivia MAIN
hydro example in DecisionRules.jl.

Canonical invariants:

- identical MAIN input bytes in both repositories;
- `pd_scale = qd_scale = 0.6`;
- weekly stages (`stage_hours = 168`) and `K = 0.6048`;
- operational active deficit cost `6000 USD/(pu·stage)`;
- strict reachable targets using the stretched sigmoid with a safe `1e-3`
  upper margin;
- hard reactive balance and apparent-power thermal limits at both branch ends;
- 96 reporting stages, 30 look-ahead stages, and one saved 126-by-500
  stage-major joint inflow-and-demand protocol.

The ACP, DC, and SOC-WR MOFs are generated once by the DecisionRules.jl
exporter and copied here without reserialization. Corresponding files must
therefore be byte-identical across the repositories.

`bolivia/case_manifest.json` records hashes, constants, topology, objective
metadata, and protocol seeds. `bolivia/joint_protocol_500.csv` records the
paired inflow-scenario and demand-atom indices.

The single retained checkpoint and historical strict result are provenance
references, not final paired evidence. Phase 2A does not recreate missing SDDP
cuts or run production training.
