---
number: 37
title: "Add multi-node support."
state: open
labels:
- enhancement
---

## Summary
Enable multi-node training for existing training tasks (`clm_training`, `mlm_training`, `instruction`) under SLURM using Lightning Fabric strategies, with HPC Leonardo as the primary target, and without changing single-node behavior.

## Background / Current State
- Task dispatch maps `clm_training`, `mlm_training`, and `instruction` to `src/tasks/training/` via `src/main.py`.
- Training orchestration lives in `src/tasks/training/orchestrator.py` and uses Lightning Fabric trainers in `src/tasks/training/fabric/`.
- `FabricTrainerBase` calls `fabric.launch(self._pipeline)` in `src/tasks/training/fabric/trainer/base.py`.
- SLURM jobs are submitted via `slurm/submit_job.sh` and executed by `slurm/p.slurm`, which currently runs a single Python process (no `srun`/`torchrun`).
- Training schemas under `config/schemas/training/` do not expose node count or devices-per-node settings.

## Problem
Multi-node allocations are not leveraged because the SLURM entrypoint launches only a single process, so distributed strategies cannot coordinate across nodes.

## Goals
- Support multi-node training with existing Fabric strategies (`ddp`, `fsdp`, `deep_speed`).
- Add explicit configuration and validation for node counts and devices-per-node.
- Preserve single-node behavior for all existing configs.
- Document SLURM launch requirements and expected logs.

## Non-goals
- No new job submission logic in Python; continue to use `slurm/submit_job.sh`.
- No Horovod or model-parallel redesigns unless explicitly approved.
- No changes to dataset loading or tokenization behavior.

## Prior Art / References
- Lightning Fabric 2.5.1 SLURM environment behavior (requires `srun`, validates `devices` and `num_nodes` against `SLURM_NTASKS_PER_NODE` and `SLURM_NNODES`):
  - https://raw.githubusercontent.com/Lightning-AI/lightning/2.5.1/src/lightning/fabric/plugins/environments/slurm.py
  - https://raw.githubusercontent.com/Lightning-AI/lightning/2.5.1/src/lightning/fabric/strategies/launchers/subprocess_script.py
- Training orchestration: `src/tasks/training/orchestrator.py`.
- Fabric trainer base: `src/tasks/training/fabric/trainer/base.py`.
- SLURM scripts: `slurm/submit_job.sh`, `slurm/p.slurm`, and `slurm/tests/run_tests.sh`.

## Proposed Approach (High-Level)
- Update SLURM launch script to use `srun` when multi-node or multi-process is requested and keep `fabric.launch` unchanged (Lightning Fabric uses SLURMEnvironment to validate settings without spawning processes).
- Add optional `num_nodes` and `devices_per_node` to training schemas and validate configuration/environment mismatches.
- Pass resolved node/device settings into `L.Fabric` and log `world_size`/rank info for verification.
- Add a multi-node smoke config under `config/tests/` and document SLURM validation steps.

## Partition Selection (Leonardo)
- Run on a Leonardo login node:
  - `sinfo -o "%P %a %l %D %N"`
  - `scontrol show partition postiguet1`
  - `scontrol show partition allen`
  - `scontrol show partition lovelace`
- Choose the partition per the rule in Decisions, and record the `sinfo` output and chosen partition in this issue.

## Environment Reporting
- Update `slurm/p.slurm` to print Python, PyTorch, Lightning, CUDA, and NCCL availability with an `ENV` prefix after `scripts/set_environment.sh` runs so the SLURM logs capture the conda environment versions.

## Scope & Invariants (Approval Required if Expanded)
Likely touched files:
- `slurm/submit_job.sh`
- `slurm/p.slurm`
- `slurm/tests/run_tests.sh`
- `config/schemas/training/components/training_args.schema.yaml`
- `src/tasks/training/orchestrator.py`
- `src/tasks/training/fabric/trainer/base.py`
- `config/tests/`
- `slurm/README.md`

Invariants:
- Task dispatch and schema validation remain authoritative.
- No auto-submission or bypass of SLURM validation.
- Single-node training behavior remains unchanged.

## Decisions
- Multi-node SLURM runs use `srun` and keep `fabric.launch` unchanged; do not introduce `torchrun`.
- Partition selection rule for Leonardo validation: prefer `postiguet1` if it has at least two GPU nodes available, otherwise use `allen`, and if `allen` is unavailable use `lovelace`. Record the `sinfo` output and the chosen partition in the issue card.
- For multi-node configs (`num_nodes > 1`), `devices_per_node` must be explicit and must match `--ntasks-per-node`. For single-node configs, `devices_per_node` can remain null and default to `torch.cuda.device_count()`.

## Acceptance Criteria
- Multi-node SLURM runs report `fabric.world_size == num_nodes * devices_per_node` in logs.
- Existing single-node configs behave exactly as before.
- Invalid multi-node configurations fail fast with clear error messages.
- SLURM logs capture Python, PyTorch, and Lightning versions with an `ENV` prefix for auditability (GPU presence optional).
- Issue notes include the partition selection evidence (`sinfo` output) and the chosen partition name.

## ExecPlan
- `issues/execPlans/37-add-multi-node-support.execplan.md`
