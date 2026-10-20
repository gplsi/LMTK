# Add multi-node training support for Lightning Fabric on SLURM

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

PLANS.md lives at `PLANS.md` from the repository root; this document must be maintained in accordance with it.

## Purpose / Big Picture

Enable multi-node training for the existing training tasks (`clm_training`, `mlm_training`, `instruction`) by wiring SLURM multi-node launches into Lightning Fabric, while preserving the current single-node behavior. After this change, users can submit a job with `--nodes` and `--ntasks-per-node` and see logs that confirm the correct world size and rank coordination, and they can complete a short smoke run without code changes to their configs.

## Progress

- [x] (2026-01-22 09:55Z) Drafted ExecPlan and aligned issue card 37.
- [x] (2026-01-22 10:05Z) Confirmed Lightning Fabric 2.5.1 SLURM launcher behavior and updated the decision log.
- [x] (2026-01-22 10:39Z) Added partition selection and environment reporting steps to make the plan junior-proof.
- [x] (2026-01-22 10:39Z) Confirmed `.agent/PLANS.md` is missing; proceeded with `PLANS.md` at repo root and noted in Decision Log.
- [x] (2026-01-22 10:39Z) Updated SLURM scripts, training schema/runtime, and docs; added multi-node configs and example.
- [x] (2026-01-22 10:55Z) Aligned DDP backend selection with schema and updated multi-node configs for validation.
- [x] (2026-01-22 11:05Z) Added multi-node FSDP example and ensured FSDP config reads `parallelization_config` defaults.
- [x] (2026-01-22 11:26Z) Added multi-node SLURM submission tutorial at `slurm/MULTINODE_TUTORIAL.md`.
- [ ] Run SLURM multi-node smoke test and record job/log evidence in the issue card.

## Surprises & Discoveries

- Observation: Lightning Fabric 2.5.1 SLURMEnvironment requires `srun` and validates `devices` and `num_nodes` against `SLURM_NTASKS_PER_NODE` and `SLURM_NNODES`, and marks `creates_processes_externally=True`, so `fabric.launch` should not spawn child processes under SLURM.
  Evidence: https://raw.githubusercontent.com/Lightning-AI/lightning/2.5.1/src/lightning/fabric/plugins/environments/slurm.py
- Observation: `_SubprocessScriptLauncher.launch` skips spawning when `creates_processes_externally=True` but still calls `validate_settings`, so `fabric.launch` is safe under SLURM and should remain in place.
  Evidence: https://raw.githubusercontent.com/Lightning-AI/lightning/2.5.1/src/lightning/fabric/strategies/launchers/subprocess_script.py
- Observation: Local validation failed due to missing Python dependency `box`, so validation must run inside the configured Conda/SLURM environment.
  Evidence: `ModuleNotFoundError: No module named 'box'` from `python3 -m src.main --config config/tests/clm_training_smoke.yaml --validate`.

## Decision Log

- Decision: Use `srun` for multi-node or multi-process SLURM jobs and keep `fabric.launch` in the training code. Do not add torchrun for SLURM.
  Rationale: Lightning Fabric 2.5.1 SLURMEnvironment expects `srun`, validates `devices` and `num_nodes` against SLURM variables, and declares external process creation, so `fabric.launch` becomes a no-op launcher but still performs environment setup and validation.
  Date/Author: 2026-01-22 Codex.
- Decision: For Leonardo multi-node validation, prefer `postiguet1` when at least two GPU nodes are available, otherwise use `allen`, then `lovelace`. Record the `sinfo` output and chosen partition in the issue card.
  Rationale: This keeps default testing aligned with existing SLURM defaults while providing a deterministic fallback order when multi-node capacity is limited.
  Date/Author: 2026-01-22 Codex.
- Decision: Require `devices_per_node` to be explicitly set when `num_nodes > 1`.
  Rationale: This prevents accidental mismatches with `SLURM_NTASKS_PER_NODE` and avoids silent under-utilization or hanging.
  Date/Author: 2026-01-22 Codex.
- Decision: Use `PLANS.md` in repo root because `.agent/PLANS.md` is not present.
  Rationale: The repo provides `PLANS.md` as the canonical plan requirements; proceeding avoids blocking while preserving alignment with documented requirements.
  Date/Author: 2026-01-22 Codex.

## Outcomes & Retrospective

Not started.

## Context and Orientation

LMTK is YAML-driven. `src/main.py` loads a YAML config, validates it via `src.config.config_loader.ConfigValidator`, and dispatches the task to a module under `src/tasks/`. Training tasks (`clm_training`, `mlm_training`, `instruction`) are routed to `src/tasks/training/__init__.py`, which instantiates `ContinualOrchestrator` in `src/tasks/training/orchestrator.py`. The orchestrator selects a distributed strategy (FSDP/DDP/DeepSpeed/DP) and constructs a Lightning Fabric trainer implemented in `src/tasks/training/fabric/`. `FabricTrainerBase` in `src/tasks/training/fabric/trainer/base.py` builds an `L.Fabric` instance and calls `fabric.launch(self._pipeline)`.

SLURM jobs are submitted through `slurm/submit_job.sh`, which builds an `sbatch` command with `--nodes` and `--ntasks-per-node`, and executes `slurm/p.slurm` as the job script. Today, `slurm/p.slurm` runs a single Python process, so multi-node allocations are not actually used.

Leonardo is an HPC SLURM environment with multiple GPU partitions (for example: `postiguet1`, `allen`, and `lovelace`). Availability varies over time, so the plan includes explicit commands to discover partitions and to record the chosen partition in the issue card. Multi-node validation must use a partition with at least two GPU nodes available.

Local development does not have GPU access, so environment verification must be captured from SLURM job logs. The plan adds a version report block in `slurm/p.slurm` to make this auditable without local CUDA.

Definitions used in this plan: a node is one machine in a SLURM allocation, a process or rank is one training worker, and world size is the total number of ranks across all nodes. Multi-node training means world size is greater than the number of devices in a single node. An external launcher is a tool such as `srun` that starts one process per GPU across nodes and populates environment variables such as `SLURM_PROCID`, `SLURM_LOCALID`, `SLURM_NTASKS`, and `SLURM_NTASKS_PER_NODE`. Lightning Fabric's SLURMEnvironment requires `srun` and uses `SLURM_NTASKS_PER_NODE` and `SLURM_NNODES` to validate `devices` and `num_nodes`.

## Milestones

Milestone 1 documents how Lightning Fabric 2.5.x expects multi-node runs to be launched on SLURM and confirms that `fabric.launch` is safe to call when processes are already spawned by `srun`. This milestone is complete because the Decision Log records the final launcher behavior and evidence links to Lightning Fabric source.

Milestone 2 implements the SLURM and training code changes: the job script launches with `srun` when multi-node or multi-process is requested, the config schemas expose `num_nodes` and `devices_per_node` with explicit validation, and the training code resolves these values against the SLURM environment. This milestone is complete when configs validate locally and logs show the resolved node/device settings without altering single-node defaults.

Milestone 3 adds a multi-node smoke config, updates SLURM documentation, and runs a SLURM smoke test. This milestone is complete when the SLURM job finishes successfully and the logs show the expected world size, ranks, and finite training loss, and the job ID and log paths are recorded in the issue card.

## Plan of Work

Lightning Fabric 2.5.1 requires `srun` for SLURM, so the plan keeps `fabric.launch` in the training code and introduces an `srun` execution path in `slurm/p.slurm` only when multi-node or multi-process is requested. Avoid torchrun under SLURM to minimize behavioral change and rely on SLURMEnvironment validation to catch mismatched `devices` and `num_nodes`.

Next, update the SLURM entrypoints. In `slurm/submit_job.sh`, add a `--ntasks-per-node` CLI option and include it in the usage text, argument parsing, and summary output. Ensure the value is exported into the job environment (it already is, but the CLI cannot set it today). In `slurm/tests/run_tests.sh`, add a matching `--ntasks-per-node` option and pass it through to `slurm/submit_job.sh` so multi-node tests can be scheduled without editing defaults. In `slurm/p.slurm`, detect multi-node or multi-process runs (for example, `NODES > 1` or `NTASKS_PER_NODE > 1`), invoke the training command via `srun` with the allocated task geometry, and leave the single-node path unchanged when `NODES == 1` and `NTASKS_PER_NODE == 1`. Only set `MASTER_ADDR` and `MASTER_PORT` if they are unset, letting SLURMEnvironment defaults stand when available.

Then add configuration support. Extend `config/schemas/training/components/training_args.schema.yaml` with optional `num_nodes` and `devices_per_node` fields (nullable integers, minimum 1, default null). Ensure the schema descriptions explain that null means "use SLURM or local auto-detection." For multi-node configs (`num_nodes > 1`), require `devices_per_node` to be explicitly set and fail validation at runtime if it is missing. Add a multi-node example config under `config/examples/` or `config/experiments/` that sets `parallelization_strategy: ddp` and uses the new fields, and update any documentation references if necessary.

Update the training runtime. Add a small helper in `src/tasks/training/utils.py` (or `src/tasks/training/orchestrator.py` if you keep it local) to resolve distributed settings from config and environment. The helper should read `num_nodes`, `devices_per_node`, and SLURM environment variables, decide whether an external launcher is in use (`SLURM_PROCID` or `LOCAL_RANK` present), and compute `devices` for Fabric such that `devices == SLURM_NTASKS_PER_NODE` when SLURM is active. Validate mismatches explicitly: if `num_nodes > 1` and `devices_per_node` is missing, raise `ValueError`; if config and SLURM disagree on `num_nodes` or `devices_per_node`, raise `ValueError` with a clear message; if `num_nodes > 1` but `parallelization_strategy` is `none` or `dp`, fail fast; if multi-node is requested without GPUs, fail fast. Store the resolved values on the orchestrator for logging.

Wire the resolved settings into the trainer. Update `FabricTrainerBase.__init__` to accept `num_nodes` and `devices_per_node`, store them, and pass `num_nodes` into `L.Fabric` construction. Ensure logging reports `num_nodes`, `devices_per_node`, `fabric.world_size`, `SLURM_NTASKS`, and `SLURM_NTASKS_PER_NODE` early in the run so SLURM logs can confirm distributed setup. Keep the single-node defaults unchanged when these settings are null and no SLURM multi-node allocation is detected.

Finally, add tests and documentation. Create a `config/tests/clm_training_multinode_smoke.yaml` that uses the small defaults, sets `parallelization_strategy: ddp`, and specifies `num_nodes` and `devices_per_node` (or leaves them null to rely on SLURM). Update `slurm/README.md` with a short section that shows how to submit a multi-node training job and explains required flags (`--nodes`, `--gpus`, `--ntasks-per-node`) and expected log lines. Include an environment report in job logs by printing Python, PyTorch, Lightning, CUDA, and NCCL versions after `scripts/set_environment.sh` runs, so the execution environment is auditable even without local GPU access. Add a guard in `slurm/p.slurm` to fail fast with a clear error if `srun` is required but not found.

## Concrete Steps

On a Leonardo login node, choose a partition for multi-node validation. Record the chosen partition in the issue card before submitting tests. Prefer `postiguet1` if it has at least two GPU nodes available; otherwise use `allen`, and if `allen` is unavailable, use `lovelace`.

    sinfo -o "%P %a %l %D %N"
    scontrol show partition postiguet1
    scontrol show partition allen
    scontrol show partition lovelace

If the `sinfo` or `scontrol` commands are unavailable (for example, outside the cluster), note that the partition choice must be confirmed when the SLURM environment is accessible and do not proceed with multi-node submission.

Validate schema changes and config parsing locally.

    python3 -m src.main --config config/tests/clm_training_smoke.yaml --validate

Run a negative validation check to confirm `devices_per_node` is required when `num_nodes > 1`.

    cp config/tests/clm_training_smoke.yaml /tmp/clm_training_smoke_bad.yaml
    python3 - <<'PY'
    import yaml
    path = "/tmp/clm_training_smoke_bad.yaml"
    with open(path, "r") as handle:
        data = yaml.safe_load(handle)
    data["num_nodes"] = 2
    data.pop("devices_per_node", None)
    with open(path, "w") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    PY
    python3 -m src.main --config /tmp/clm_training_smoke_bad.yaml --validate

If the tokenized test dataset is missing, generate it with the existing smoke config.

    python3 -m src.main --config config/tests/tokenization_smoke.yaml

Dry-run the SLURM submission to confirm the sbatch geometry includes nodes and tasks-per-node.

    ./slurm/submit_job.sh --config config/tests/clm_training_multinode_smoke.yaml --nodes 2 --gpus 1 --ntasks-per-node 1 --dry-run

Submit the multi-node smoke test via the SLURM test runner (ensure your user is in `slurm/tests/slurm_test.env` allowed list).

    ./slurm/tests/run_tests.sh --config config/tests/clm_training_multinode_smoke.yaml --nodes 2 --gpus 1 --ntasks-per-node 1

When running on SLURM, capture environment versions from the job log. The log should include the output of a small Python snippet that prints versions for `python`, `torch`, `lightning`, `torch.version.cuda`, and `torch.distributed.is_nccl_available()`. If GPU hardware is unavailable, `nvidia-smi` will fail; the log should still show the software versions. Use a stable prefix such as `ENV` in each line so the logs are greppable.

## Validation and Acceptance

Run `python3 -m src.main --config config/tests/clm_training_smoke.yaml --validate` and expect "Configuration is valid!" to confirm the new schema fields are optional and do not break existing configs. Add a negative check by setting `num_nodes: 2` and omitting `devices_per_node` in a temporary local copy of the config and expect a `ValueError` stating that `devices_per_node` must be set when `num_nodes > 1`. For the multi-node SLURM run, inspect the job log under `slurm/tests/logs/` and confirm that the log shows the resolved `num_nodes`, `devices_per_node`, and `fabric.world_size` values and that `fabric.world_size` equals `num_nodes * devices_per_node`. The log must also include the Python, PyTorch, and Lightning versions from the environment report so the execution environment is documented, and it must record the chosen partition and node list. The training loss and validation loss must be finite and within a broad sanity range (for the tiny random model, expect 0 < loss < 100) to confirm no NaNs or infinities are introduced by distributed setup. Record the SLURM job ID, log paths, chosen partition, and the exact submission command in the issue card.

## Idempotence and Recovery

All edits are additive or guarded; running the steps multiple times should be safe. If a SLURM run fails, use the logged command and `scontrol show job <job_id>` to diagnose allocation issues, and rerun with the same config after fixing the root cause. To recover to single-node behavior, submit with `--nodes 1 --ntasks-per-node 1` and omit the new config fields.

## Artifacts and Notes

Capture minimal evidence in this section as work progresses. Example snippets:

    Partition: postiguet1 (selected via sinfo on 2026-01-22)
    SLURM job ID: 123456
    Logs: slurm/tests/logs/tests-123456.out
    world_size: 2 (num_nodes=2, devices_per_node=1)
    ENV python=3.10.13
    ENV torch=2.2.1
    ENV lightning=2.5.1
    ENV cuda=12.1
    ENV nccl=True
    train_loss: 8.42

    Local validation attempt failed:
    ModuleNotFoundError: No module named 'box'

## Interfaces and Dependencies

Use Lightning Fabric 2.5.x as already declared in `pyproject.toml`. No new external dependencies are required.

Add the following interfaces and fields:

In `config/schemas/training/components/training_args.schema.yaml`, add optional fields:

    num_nodes: integer or null, default null, minimum 1
    devices_per_node: integer or null, default null, minimum 1

In `src/tasks/training/utils.py`, add a resolver (or keep it local to the orchestrator if preferred):

    def resolve_distributed_settings(config: Box) -> dict:
        """Return devices, devices_per_node, num_nodes, and is_external_launcher; raise ValueError on mismatches or missing devices_per_node when num_nodes > 1."""

In `src/tasks/training/orchestrator.py`, store the resolved settings on the orchestrator and pass them into the trainer.

In `src/tasks/training/fabric/trainer/base.py`, update the initializer signature and Fabric construction:

    def __init__(self, devices: int | str, config: Box, dataset: HFDataset, checkpoint_path: str = None, num_nodes: int | None = None, devices_per_node: int | None = None) -> None:
        ...

    fabric = L.Fabric(devices=self.devices, num_nodes=self.num_nodes, strategy=strategy, precision=self.config.precision, loggers=loggers)

In `slurm/submit_job.sh`, add `--ntasks-per-node` to usage and argument parsing, and ensure it flows to `SBATCH_CMD` and exported vars.

In `slurm/tests/run_tests.sh`, add `--ntasks-per-node` and forward it to `slurm/submit_job.sh`.

In `slurm/p.slurm`, add a conditional `srun` execution path for multi-node or multi-process runs, and set `MASTER_ADDR` and `MASTER_PORT` before launching.

In `slurm/p.slurm`, add a small environment report block after `scripts/set_environment.sh`:

    python - <<'PY'
    import sys
    import torch
    import lightning
    print(f"ENV python={sys.version.split()[0]}")
    print(f"ENV torch={torch.__version__}")
    print(f"ENV lightning={lightning.__version__}")
    print(f"ENV cuda={torch.version.cuda}")
    print(f"ENV nccl={torch.distributed.is_nccl_available()}")
    PY

## LMTK Project Patterns to Embed in ExecPlans

LMTK workflows are YAML-driven. Each YAML configuration denotes a single job and maps to exactly one task module under `src/tasks/`. This plan keeps the training tasks under `src/tasks/training/` and updates only schema and orchestration layers. Tests for the new behavior live under `config/tests/` and should be submitted through `slurm/tests/run_tests.sh` with `slurm/tests/slurm_test.env` and an allowed submitter list. Configs should be reviewed before SLURM submission and must not auto-submit or bypass validation.

Plan update note (2026-01-22): Recorded Lightning Fabric 2.5.1 SLURM launcher evidence, finalized the srun + fabric.launch decision, and added explicit environment version reporting requirements to make the plan junior-proof on SLURM-only infrastructure.
Plan update note (2026-01-22): Added explicit partition selection steps, environment log prefixes, and acceptance evidence requirements so a junior can execute and audit multi-node validation on Leonardo.
Plan update note (2026-01-22): Required explicit `devices_per_node` for multi-node configs and added a negative validation check to prevent silent misconfiguration.
Plan update note (2026-01-22): Added Leonardo partition context and clarified that environment verification must come from SLURM logs due to local GPU unavailability.
Plan update note (2026-01-22): Recorded the absence of `.agent/PLANS.md` and the use of `PLANS.md` for plan compliance.
Plan update note (2026-01-22): Marked implementation steps complete for SLURM scripts, schema/runtime updates, and added multi-node configs and docs; SLURM validation remains pending.
Plan update note (2026-01-22): Recorded local validation failure due to missing `box` dependency, reinforcing the need to validate inside the SLURM/Conda environment.
Plan update note (2026-01-22): Updated DDP backend selection fallback to honor `parallelization_config.backend` and added required `parallelization_config` entries to multi-node configs.
Plan update note (2026-01-22): Added a multi-node FSDP example and updated FSDP config resolution to read `parallelization_config` values when top-level keys are unset.
