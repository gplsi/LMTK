# Token-budget multi-dataset mixing + replay (homogeneous packed blocks)

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

PLANS.md is checked into the repo at `PLANS.md`, and this document must be maintained in accordance with it.

## Purpose / Big Picture

Enable `task: clm_training` to train from multiple doc-level tokenized sources with deterministic block-level mixing and replay, while preserving issue-42 guarantees:

- each packed block belongs to exactly one source,
- distributed execution avoids silent duplication/padding side effects,
- accounting is persisted and auditable.

After this change, a novice can run one config and get:

1. multi-source packed training from `dataset.sources`,
2. deterministic realized allocation over `dataset.mixture.total_blocks`,
3. a rank-0 `mixture_report.json` that proves what was actually trained.

## Progress

- [ ] (2026-02-06 00:00Z) Add schema support for `dataset.sources` and `dataset.mixture`.
- [ ] (2026-02-06 00:00Z) Add red tests for deterministic allocation and distributed invariants.
- [ ] (2026-02-06 00:00Z) Implement mixture dataset and trainer integration.
- [ ] (2026-02-06 00:00Z) Implement per-source compatibility checks and fail-fast guards.
- [ ] (2026-02-06 00:00Z) Implement rank-safe accounting and `mixture_report.json`.
- [ ] (2026-02-06 00:00Z) Add smoke configs and integration testing config.
- [ ] (2026-02-06 00:00Z) Run local validation + SLURM smoke; record job IDs/logs in issue 43.

## Surprises & Discoveries

- Observation: current trainer validation cadence is epoch-driven (`validations_per_epoch`, `checkpoints_per_epoch`), and `validate_after_k_steps` is not consumed in `src/tasks/training/fabric/trainer/base.py`.
  Evidence: `_try_validate` logic in `src/tasks/training/fabric/trainer/base.py`.

- Observation: `src/tasks/training/orchestrator.py` currently loads exactly one source via `dataset.nameOrPath`.
  Evidence: `load_dataset` in `src/tasks/training/orchestrator.py`.

## Decision Log

- Decision: v1 mixing stays map-style and block-level only; no token-level mixed blocks.
  Rationale: keeps determinism, accounting, and distributed correctness simple and auditable.
  Date/Author: 2026-02-06 / Codex

- Decision: enforce explicit divisibility/step-budget fail-fast instead of silent correction.
  Rationale: silent drops/padding would make reported ratios untrustworthy.
  Date/Author: 2026-02-06 / Codex

- Decision: validation cadence for mixture runs uses epoch-based knobs (`validations_per_epoch`, `checkpoints_per_epoch`) in v1.
  Rationale: aligns with existing trainer behavior and avoids introducing a second cadence mechanism in this issue.
  Date/Author: 2026-02-06 / Codex

## Outcomes & Retrospective

- Pending implementation.

## Context and Orientation

LMTK is YAML-driven. `src/main.py` validates configs with `src/config/config_loader.py` and dispatches to task modules under `src/tasks/`.

Current relevant behavior:

- issue 42 already provides doc-level CLM tokenization output (`input_ids`, `length`, `ends_with_eos`) and training-time packing primitives:
  - `src/tasks/training/data/packing.py`
  - `src/tasks/training/data/packing_index.py`
- `src/tasks/training/orchestrator.py` currently loads one dataset path (`dataset.nameOrPath`) and passes it into Fabric trainers.
- `src/tasks/training/fabric/trainer/base.py` supports single-source packing and explicit distributed sampler policies.

This plan extends that existing path. It does not add new tasks or new SLURM submission logic.

Terms used in this plan:

- packed block: one fixed-length training sample of size `sequence_length`,
- homogeneous block: packed block attributable to one dataset source only,
- effective_total_blocks: actual executable block budget after enforcing distributed and optimization-step invariants.

## Interfaces and Dependencies

At completion, the following must exist:

1. Schema surface in `config/schemas/training/components/data.schema.yaml`:
   - support either:
     - single-source (`dataset.nameOrPath`), or
     - multi-source (`dataset.sources` + `dataset.mixture`).
   - `dataset.sources` item fields:
     - `dataset_id: string` (required, unique),
     - `nameOrPath: string` (required),
     - `weight: number` (required, `> 0`),
     - `tokenizer_name: string | null` (optional compatibility metadata),
     - `eos_token_id: integer | null` (optional compatibility metadata),
     - `index_cache_dir: string | null` (optional).
   - `dataset.mixture` fields:
     - `enabled: boolean` (required when `sources` exists),
     - `total_blocks: integer >= 1` (required),
     - `schedule_seed: integer | null` (optional),
     - `report_path: string | null` (optional; default `<output_dir>/mixture_report.json`),
     - `validation_mode: string | null` with enum `first_source` / `concat_all`.

2. Runtime dataset in `src/tasks/training/data/mixture.py`:
   - `class MixturePackedDataset(torch.utils.data.Dataset)`,
   - `__len__ -> effective_total_blocks`,
   - `__getitem__(i)` returns `input_ids`, `attention_mask`, `labels`, `dataset_id`.

3. Trainer integration in `src/tasks/training/fabric/trainer/base.py`:
   - detect mixture mode from config,
   - build per-source packed datasets and compose with `MixturePackedDataset`,
   - enforce distributed and step-budget invariants with hard errors,
   - aggregate per-source counters across ranks and write `mixture_report.json` on rank 0.

4. Orchestrator loading updates in `src/tasks/training/orchestrator.py`:
   - support `dataset.sources` loading path (DatasetDict per source) while preserving existing single-source behavior.

## Milestones

### Milestone 1: Schema and config surface

Add schema support in `config/schemas/training/components/data.schema.yaml` for `dataset.sources` and `dataset.mixture`, while preserving existing single-source configs.

At the end of this milestone, these commands succeed:

    python src/main.py --validate --config config/tests/clm_training_packing_smoke.yaml
    python src/main.py --validate --config config/tests/clm_training_packing_mixture_smoke.yaml

Acceptance:

- single-source training configs still validate unchanged,
- invalid mixture configs fail with explicit messages (missing `sources`, duplicate `dataset_id`, non-positive `weight`, missing `total_blocks`).

### Milestone 2: Red tests for mixture semantics

Before implementation, add failing tests in `tests/unit/training/test_mixture_packing.py` for:

- deterministic exact allocation (`weights -> target_blocks`) with deterministic remainder handling,
- no `O(total_blocks)` schedule materialization in dataset state,
- deterministic mapping for fixed seed,
- distributed invariant failures (non-divisible budgets),
- compatibility failures for mismatched tokenizer/eos metadata.

Acceptance:

- tests fail initially for missing implementation (red state is explicit and expected).

### Milestone 3: Implement mixture runtime

Implement:

- `src/tasks/training/data/mixture.py`,
- orchestration hooks in `src/tasks/training/orchestrator.py`,
- trainer integration and accounting in `src/tasks/training/fabric/trainer/base.py`.

Required runtime behavior:

- exact deterministic target allocation per source over configured budget,
- local block selection with replacement: `local_idx = hash(seed, global_idx, dataset_id) % source_num_blocks`,
- no token mixing inside blocks,
- fail-fast on incompatible distributed/optimizer settings.

Acceptance:

- red tests from Milestone 2 pass (green),
- existing packing tests still pass.

### Milestone 4: Auditable report and smoke configs

Add:

- `config/tests/clm_training_packing_mixture_smoke.yaml`,
- `config/tests/clm_training_packing_mixture_multinode_smoke.yaml`,
- `config/tests/mixture_packing_integration_smoke.yaml` (`task: testing`) that runs tokenization + mixture smoke.

The report file (rank 0) must include:

- config summary (`dataset_id`, weight, sequence_length, total_blocks, effective_total_blocks, seed/schedule_seed),
- runtime context (world_size, global batch settings, drop policies),
- `target_blocks_per_dataset`,
- `realized_blocks_per_dataset`,
- `realized_tokens_per_dataset`,
- `realized_ratios`,
- `deviation_from_target_blocks`.

Acceptance:

- smoke run produces report with internally consistent totals.

### Milestone 5: SLURM validation and evidence capture

Use the existing SLURM test runner path:

- `slurm/tests/run_tests.sh`,
- `slurm/tests/slurm_test.env`,
- optional secrets from `slurm/tests/test_secrets.env`.

Ensure submitter guard is satisfied (`ALLOWED_SUBMITTERS` in `slurm/tests/slurm_test.env`).

Acceptance:

- SLURM job exits successfully,
- job ID, log paths, exact command, and report path are recorded in `issues/43-token-budget-mixture-replay.md`.

## Plan of Work

1. Update `config/schemas/training/components/data.schema.yaml` with mutually exclusive single-source vs multi-source support and strict validation rules.
2. Add failing unit tests in `tests/unit/training/test_mixture_packing.py` for allocation, determinism, distributed invariants, and compatibility guards.
3. Implement `src/tasks/training/data/mixture.py` with deterministic index mapping and no full schedule array.
4. Extend `src/tasks/training/orchestrator.py` to load sources for mixture mode while preserving existing code path for `dataset.nameOrPath`.
5. Extend `src/tasks/training/fabric/trainer/base.py` to build mixture dataloaders, enforce fail-fast invariants, aggregate counters, and write report.
6. Add/update smoke configs under `config/tests/` using the tiny-model defaults currently used by test configs.
7. Execute local tests/validation first, then SLURM smoke, then record evidence in issue 43.

## Concrete Steps

Run from repository root (`/home/gplsi/GPLSI/codigos/LMTK`).

1. Run focused red tests:

    python3 -m pytest -q tests/unit/training/test_mixture_packing.py

2. Run existing packing regression tests:

    python3 -m pytest -q tests/unit/training/test_packing_index.py tests/unit/training/test_packing_sampler.py tests/test_packed_sequence_dataset.py

3. Validate configs:

    python src/main.py --validate --config config/tests/clm_training_packing_mixture_smoke.yaml
    python src/main.py --validate --config config/tests/mixture_packing_integration_smoke.yaml

4. Run local integration smoke:

    python src/main.py --config config/tests/mixture_packing_integration_smoke.yaml

5. Run SLURM multi-node smoke:

    ./slurm/tests/run_tests.sh --config config/tests/clm_training_packing_mixture_multinode_smoke.yaml --nodes 2 --ntasks-per-node 1

Expected observable outputs:

- schema validation prints `Configuration is valid!`,
- local smoke writes `mixture_report.json` under configured output dir,
- SLURM runner prints `Submitted batch job <ID>`, followed by resolved stdout/stderr log paths.

## Validation and Acceptance

A change is accepted only when all are true:

1. Determinism:
   - same config + seed + world size produces the same `target_blocks_per_dataset` and same deterministic schedule mapping.

2. Budget/accounting:
   - `sum(realized_blocks_per_dataset) == effective_total_blocks`,
   - `sum(realized_tokens_per_dataset) == effective_total_blocks * sequence_length`,
   - each dataset’s deviation from deterministic target is exactly explained by documented distributed constraints (no unexplained drift).

3. Safety:
   - incompatible source/tokenizer/eos configs fail fast,
   - incompatible distributed step budgets fail fast,
   - no rank hangs when artifact/index build fails.

4. Regression:
   - existing online-packing tests continue to pass.

## Idempotence and Recovery

- Re-running schema validation/tests is idempotent.
- Packing index artifacts remain reusable. Rebuild only when intentionally invalidating caches (delete source `.packing_index` directories).
- If a run fails during artifact creation, retry only after confirming no active job holds the relevant lock.
- Do not add fallback paths that silently alter budgets or compatibility outcomes.

## Artifacts and Notes

Minimal expected `mixture_report.json` structure:

    {
      "sequence_length": 128,
      "total_blocks": 1000,
      "effective_total_blocks": 1000,
      "world_size": 2,
      "sources": [{"dataset_id": "A", "weight": 2.0}, {"dataset_id": "B", "weight": 1.0}],
      "target_blocks_per_dataset": {"A": 667, "B": 333},
      "realized_blocks_per_dataset": {"A": 667, "B": 333},
      "realized_tokens_per_dataset": {"A": 85376, "B": 42624},
      "realized_ratios": {"A": 0.667, "B": 0.333}
    }

## Revision Note (2026-02-06)

Reworked this ExecPlan to be implementation-ready for a junior developer by removing ambiguity around validation cadence, adding explicit distributed budget invariants, defining exact report/accounting requirements, adding test-first sequencing, and including SLURM runbook details required by AGENTS.md and PLANS.md.
