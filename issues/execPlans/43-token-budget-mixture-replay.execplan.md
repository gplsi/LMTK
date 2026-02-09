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
2. deterministic realized allocation over a resolved block budget (`anchor_epochs` default or `explicit_blocks`),
3. a rank-0 `mixture_report.json` that proves what was actually trained.

## Progress

- [x] (2026-02-06 12:20Z) Add schema support for `dataset.sources` and `dataset.mixture`.
- [x] (2026-02-06 12:20Z) Add red tests for deterministic allocation and distributed invariants.
- [x] (2026-02-06 12:20Z) Implement mixture dataset and trainer integration.
- [x] (2026-02-06 12:20Z) Implement per-source compatibility checks and fail-fast guards.
- [x] (2026-02-06 12:20Z) Implement rank-safe accounting and `mixture_report.json`.
- [x] (2026-02-06 12:20Z) Add smoke configs and integration testing config.
- [x] (2026-02-09 10:40Z) Implement hardening pass 2 for resume integrity and metadata auditability (`mixture_runtime` checkpointing, `mixture_meta` v2, run metadata persistence, and end-of-run mixture summary logging).
- [x] (2026-02-09 12:15Z) Implement hardening pass 3 for scheduler LR bounds + metadata observability (`min_lr`/`max_lr`, bounded scheduler behavior, preserved default param-group LR semantics, and run-metadata/WandB scheduler-step logging).
- [x] (2026-02-09 13:05Z) Implement hardening pass 4 for release readiness (multi-node smoke config absolute paths, final run metadata refresh before report write, and on-device mixture realized-block accounting in the training hot path).
- [x] (2026-02-09 14:05Z) Implement hardening pass 5 for bounded-scheduler correctness on short runs (`warmup_steps == 1` now starts at `min_lr`) with regression test coverage.
- [x] (2026-02-09 15:15Z) Implement hardening pass 6 for review-closure fail-fast contracts (`dataset.mixture -> dataset.sources` schema/runtime enforcement and malformed `mixture_meta` dictionary validation), with regression tests.
- [ ] (2026-02-06 12:20Z) Run local validation + SLURM smoke; record job IDs/logs in issue 43. (completed: local pytest + schema validation + strict-budget/config fail-fast follow-up including pass 6 guardrails; remaining: SLURM smoke run and evidence capture)

## Surprises & Discoveries

- Observation: current trainer validation cadence is primarily epoch-driven (`validations_per_epoch`, `checkpoints_per_epoch`), and `_try_validate` also supports `validate_after_k_steps` for global-step triggers.
  Evidence: `_try_validate` logic in `src/tasks/training/fabric/trainer/base.py`.

- Observation: `src/tasks/training/orchestrator.py` currently loads exactly one source via `dataset.nameOrPath`.
  Evidence: `load_dataset` in `src/tasks/training/orchestrator.py`.

- Observation: `.agent/PLANS.md` is not present in this repository, so execution follows `PLANS.md` at repo root.
  Evidence: `sed: can't read .agent/PLANS.md: No such file or directory` during preflight.

- Observation: local config validation via `src/main.py` required `PYTHONPATH=.` in this environment.
  Evidence: `ModuleNotFoundError: No module named 'src'` without `PYTHONPATH=.`.

- Observation: JSON Schema Draft7 cannot enforce uniqueness of a property within array objects (`dataset.sources[].dataset_id`) without custom extensions.
  Evidence: Schema-level constraints can enforce shape but not field-level uniqueness in standard Draft7.

- Observation: the initial budget-mode conditional schema required both `anchor_epochs` and `total_blocks` when `budget_mode` was omitted, which contradicted the intended default-anchor behavior.
  Evidence: local validation raised both required-property errors for `dataset.mixture: { enabled: true }` before schema conditional fix.

- Observation: local environment lacks the `datasets` package for trainer/orchestrator unit imports, so new contract tests are dependency-gated and skip locally unless training dependencies are installed.
  Evidence: `ModuleNotFoundError: No module named 'datasets'` during initial collection before adding `importorskip`.

- Observation: pre-hardened mixture resume checkpoints preserved optimizer/model iteration counters but not source-attributed realized accounting state, which made resumed `mixture_report.json` totals incomplete.
  Evidence: `_mixture_realized_blocks_local` was runtime-only and absent from checkpoint state before pass 2.

- Observation: scheduler pass 3 initially rewrote optimizer param-group learning rates even when no LR bounds were requested, which could silently alter multi-group optimizer behavior.
  Evidence: `select_scheduler` set every param-group LR to `peak_lr` unconditionally before the bounded/default path split.

- Observation: multi-node smoke configs fail early when source dataset paths are relative, because the mixture trainer enforces absolute/shared filesystem paths under `SLURM_NNODES>1`.
  Evidence: explicit guard in `FabricTrainerBase._build_mixture_packing_dataloaders` rejects non-absolute or `/tmp`/`/dev/shm` paths.

## Decision Log

- Decision: v1 mixing stays map-style and block-level only; no token-level mixed blocks.
  Rationale: keeps determinism, accounting, and distributed correctness simple and auditable.
  Date/Author: 2026-02-06 / Codex

- Decision: enforce explicit divisibility/step-budget fail-fast instead of silent correction.
  Rationale: silent drops/padding would make reported ratios untrustworthy.
  Date/Author: 2026-02-06 / Codex

- Decision: v1 budget interface supports two modes: `anchor_epochs` (default) and `explicit_blocks`; both resolve to `requested_total_blocks` then `effective_total_blocks`.
  Rationale: preserves user ergonomics for “train one anchor epoch” while keeping strict token-budget accounting.
  Date/Author: 2026-02-06 / Codex

- Decision: in v1, validation datasets in mixture mode are auto-generated per source independently (when missing), not from a mixed stream.
  Rationale: avoids source leakage and gives source-attributed validation metrics before dedicated validation corpora exist.
  Date/Author: 2026-02-06 / Codex

- Decision: v1 determinism uses explicit algorithms: `blake2b_u64` for hashing and Hamilton largest-remainder for exact allocation with lexicographic tie-break.
  Rationale: prevents runtime/process-dependent behavior and removes implementation ambiguity for junior contributors.
  Date/Author: 2026-02-06 / Codex

- Decision: `anchor_epochs` uses canonical anchor-fixed math (`target_anchor_per_epoch = B_anchor`; non-anchor counts derived from `B_anchor * w_j / w_anchor`, then Hamilton with explicit half-up total).
  Rationale: removes ambiguity about epoch budget derivation while preserving intended replay proportions.
  Date/Author: 2026-02-06 / Codex

- Decision: validation cadence for mixture runs stays epoch-based (`validations_per_epoch`, `checkpoints_per_epoch`) in v1.
  Rationale: aligns with current trainer behavior and avoids introducing a second cadence mechanism in this issue.
  Date/Author: 2026-02-06 / Codex

- Decision: per-source `weight` becomes optional only in all-or-none mode; when omitted for all sources, effective weights are derived from per-source train packed-block capacities.
  Rationale: this preserves natural corpus proportions with zero manual tuning while keeping deterministic, auditable semantics.
  Date/Author: 2026-02-06 / Codex

- Decision: mixture mode fails fast when any source has zero train packed blocks after per-source validation split construction.
  Rationale: a zero-capacity source makes deterministic sampling invalid and would otherwise create hidden fallback behavior.
  Date/Author: 2026-02-06 / Codex

- Decision: deterministic mixture primitives are implemented in `src/tasks/training/data/mixture.py` and reused by trainer integration.
  Rationale: centralizes math-heavy allocation/scheduling logic in one testable module and keeps trainer changes focused on orchestration.
  Date/Author: 2026-02-06 / Codex

- Decision: strict budget execution is enforced through explicit executable-block resolution (`resolve_effective_total_blocks`) before dataloader construction.
  Rationale: this prevents silent block drops from `drop_last_batch` while preserving the `effective_total_blocks == requested_total_blocks` contract.
  Date/Author: 2026-02-06 / Codex

- Decision: config-time fail-fast for duplicate `dataset_id` and mixed source `weight` presence is implemented in `ConfigValidator` custom constraints.
  Rationale: Draft7 schema cannot encode these two mixture semantics robustly; validation must fail before runtime.
  Date/Author: 2026-02-06 / Codex

- Decision: schema conditionals now treat omitted `budget_mode` as anchor mode for validation purposes, requiring only `anchor_epochs` by default while `explicit_blocks` requires `total_blocks` only when explicitly selected.
  Rationale: restores contract consistency between issue semantics, runtime defaults, and `--validate` behavior.
  Date/Author: 2026-02-06 / Codex

- Decision: mixture checkpoint metadata is versioned (`mixture_meta_version=v2`) and includes `effective_total_blocks` and `dataset_index_map`, while legacy checkpoints without a version are still accepted via subset compatibility checks plus warning.
  Rationale: preserves backward compatibility without giving up strict resume guarantees for new checkpoints.
  Date/Author: 2026-02-09 / Codex

- Decision: persist run-level metadata and mixture runtime accounting into checkpoint state and report summary metrics at end-of-run.
  Rationale: closes auditability gaps for resume scenarios and ensures CSV/WandB backends capture final realized-vs-target mixture behavior.
  Date/Author: 2026-02-09 / Codex

- Decision: bounded scheduler controls are implemented with explicit `min_lr`/`max_lr` fields and fail-fast validation, while preserving legacy scheduler semantics when bounds are not configured.
  Rationale: enables deterministic start/end LR control without changing existing training behavior by default.
  Date/Author: 2026-02-09 / Codex

- Decision: optimizer param-group learning rates are preserved unless `max_lr` is explicitly provided.
  Rationale: avoids hidden regressions for multi-group optimizers and keeps blast radius low in HPC runs.
  Date/Author: 2026-02-09 / Codex

- Decision: scheduler step/bound metadata is persisted in run metadata and logged once via Fabric metrics.
  Rationale: guarantees auditable checkpoint metadata and visibility in CSV/WandB backends without per-step overhead.
  Date/Author: 2026-02-09 / Codex

- Decision: refresh `run_metadata` immediately before writing `mixture_report.json`.
  Rationale: prevents stale end-of-run counters when no final checkpoint save occurs.
  Date/Author: 2026-02-09 / Codex

- Decision: keep mixture realized-block counters on device during training, with dict synchronization only at checkpoint/report boundaries.
  Rationale: removes unnecessary per-step CPU synchronization from the training hot path while preserving existing checkpoint/report contracts.
  Date/Author: 2026-02-09 / Codex

- Decision: mixture configuration is now treated as invalid unless `dataset.sources` is present whenever `dataset.mixture.enabled` is true.
  Rationale: prevents silent fallback to single-source behavior and preserves auditable mixture intent.
  Date/Author: 2026-02-09 / Codex

- Decision: resume compatibility checks fail fast when checkpoint `mixture_meta` is not a dictionary.
  Rationale: malformed metadata must not crash with implicit attribute errors or continue in an unsafe state.
  Date/Author: 2026-02-09 / Codex

## Outcomes & Retrospective

- Implemented schema, runtime, tests, and local validation for mixture mode with all-or-none source weights, zero-capacity fail-fast guards, and strict executable-budget enforcement.
- Added config-time fail-fast for duplicate source IDs and mixed manual/omitted weights.
- Fixed schema budget-mode default semantics and added regression tests for omitted `budget_mode`.
- Added trainer/orchestrator contract tests that validate moving parts without running full training loops.
- Added hardening pass 2: checkpointed mixture runtime accounting for resume integrity, versioned mixture metadata with dataset index mapping, run metadata persistence, and final mixture summary logging.
- Added hardening pass 3: bounded LR scheduler support (`min_lr`/`max_lr`), config fail-fast validation for LR bounds, preserved default optimizer-group LR semantics, and persisted/logged scheduler step metadata for auditability.
- Added hardening pass 4: fixed multi-node smoke config path compliance, refreshed final run metadata before report emission, and moved mixture realized-block accounting to on-device counters for better HPC runtime efficiency.
- Added hardening pass 5: corrected bounded scheduler behavior for the `warmup_steps == 1` edge case so LR starts from `min_lr`, with targeted regression coverage.
- Added hardening pass 6: enforced `dataset.mixture -> dataset.sources` fail-fast behavior across schema/trainer/orchestrator and hardened resume metadata dictionary validation with targeted regression tests.
- Remaining work from this plan is operational validation on SLURM and recording job/log/report evidence in issue 43.

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
- training batch: `batch_size` packed blocks per rank (global optimizer-step batch also depends on `world_size` and `gradient_accumulation_steps`),
- homogeneous block: packed block attributable to one dataset source only,
- virtual mixture epoch: accounting unit used by `budget_mode=anchor_epochs` to derive `requested_total_blocks`; this is not trainer `number_epochs`,
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
     - `weight: number | null` (optional, `> 0` when set),
     - `tokenizer_name: string | null` (optional compatibility metadata),
     - `eos_token_id: integer | null` (optional compatibility metadata),
     - `index_cache_dir: string | null` (optional).
   - source-weight rule:
     - either every source sets `weight` (manual mode), or every source omits `weight` (derive from source blocks),
     - mixed provided/omitted source weights are invalid.
   - `dataset.mixture` fields:
     - `enabled: boolean` (required when `sources` exists),
     - `budget_mode: string` with enum `anchor_epochs` / `explicit_blocks` (default `anchor_epochs`),
     - `anchor_epochs: integer >= 1` (required when `budget_mode=anchor_epochs`),
     - `anchor_dataset_id: string | null` (optional; default selects source with max block capacity),
     - `total_blocks: integer >= 1` (required when `budget_mode=explicit_blocks`),
     - `schedule_seed: integer | null` (optional),
     - `report_path: string | null` (optional; default `<output_dir>/mixture_report.json`),
     - v1 validation behavior is fixed: per-source independent auto-split when `valid` is missing.

2. Runtime dataset in `src/tasks/training/data/mixture.py`:
   - `class MixturePackedDataset(torch.utils.data.Dataset)`,
   - `__len__ -> effective_total_blocks`,
   - `__getitem__(i)` returns `input_ids`, `attention_mask`, `labels`, `dataset_idx` (`int64` tensor),
   - deterministic `dataset_idx <-> dataset_id` mapping is stored in report/checkpoint metadata.

3. Trainer integration in `src/tasks/training/fabric/trainer/base.py`:
   - detect mixture mode from config,
   - build per-source packed datasets and compose with `MixturePackedDataset`,
   - resolve requested budget from selected budget mode (`anchor_epochs` or `explicit_blocks`),
   - resolve effective source weights before allocation:
     - manual mode when all `weight` values are provided,
     - inferred mode when all `weight` values are omitted (`effective_weight_i = source_blocks_i` from train split),
     - fail fast on mixed provided/omitted weights,
   - resolve default anchor as source with max **train-split** packed blocks when not specified,
   - auto-build per-source validation splits independently before mixture composition, reusing `FabricTrainerBase._ensure_validation_split` semantics per source (no duplicate splitter),
   - enforce `number_epochs == 1` in mixture mode (hard error otherwise),
   - keep existing epoch-driven validation/checkpoint cadence (`validations_per_epoch`, `checkpoints_per_epoch`) over this single trainer epoch,
   - enforce `effective_total_blocks == requested_total_blocks` in v1 (hard error otherwise),
   - emit `metric/val_loss_<dataset_id>` and `metric/val_loss_weighted`,
   - enforce non-dropping validation dataloader policy and fail fast if any source validation loader has zero batches,
   - enforce resume-compatibility checks for accounting-critical mixture metadata,
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
- invalid mixture configs fail with explicit messages (missing `sources`, duplicate `dataset_id`, non-positive `weight` when provided, mixed provided/omitted weights, invalid budget-mode conditionals such as missing `anchor_epochs` or missing `total_blocks` when required).

### Milestone 2: Red tests for mixture semantics

Before implementation, add failing tests in `tests/unit/training/test_mixture_packing.py` for:

- deterministic exact allocation (`weights -> target_blocks`) with deterministic remainder handling,
- weight resolution behavior:
  - all-manual weights are preserved,
  - all-omitted weights derive from source block counts,
  - mixed weight presence fails fast,
  - any source with zero train packed blocks fails fast with `dataset_id`,
- no `O(total_blocks)` schedule materialization in dataset state,
- deterministic mapping for fixed seed,
- deterministic mapping is stable across processes (no Python built-in `hash` dependency),
- Hamilton largest-remainder allocation with lexicographic tie-break reproducibility,
- canonical anchor-fixed per-epoch target derivation in `anchor_epochs`,
- budget resolution for `anchor_epochs` and `explicit_blocks`,
- default anchor selection (`argmax(source_blocks)`), plus explicit `anchor_dataset_id` override,
- fail-fast when mixture mode uses `number_epochs != 1`,
- fail-fast when `effective_total_blocks != requested_total_blocks` in v1 contract,
- distributed invariant failures (non-divisible budgets),
- compatibility failures for mismatched tokenizer/eos metadata,
- per-source deterministic validation auto-split construction,
- fail-fast when source lacks both `valid` and usable `validation_split`,
- fail-fast when any source validation dataloader has zero batches,
- fail-fast on resume metadata mismatch for accounting-critical fields,
- validation metrics emission (`val_loss_<dataset_id>` and `val_loss_weighted`).

Acceptance:

- tests fail initially for missing implementation (red state is explicit and expected).
- the test location remains discoverable and scoped: these tests stay under `tests/unit/training/` because they cross schema, orchestrator, and trainer boundaries; any helper classes specific to `MixturePackedDataset` should be colocated under `src/tasks/training/data/` when needed.

### Milestone 3: Implement mixture runtime

Implement:

- `src/tasks/training/data/mixture.py`,
- orchestration hooks in `src/tasks/training/orchestrator.py`,
- trainer integration and accounting in `src/tasks/training/fabric/trainer/base.py`.

Required runtime behavior:

- exact deterministic target allocation per source over resolved budget,
- effective weights are resolved deterministically (manual or derived from source block counts) before allocation,
- fail fast when any source has zero train packed blocks (`B_i == 0`) after validation split construction,
- local block selection with replacement: `local_idx = blake2b_u64(f"{schedule_seed}|{global_idx}|{dataset_id}") % source_num_blocks`,
- no token mixing inside blocks,
- in `anchor_epochs` mode, use canonical anchor-fixed derivation:
  - `target_anchor_per_epoch = B_anchor`,
  - `raw_non_anchor_j = B_anchor * w_j / w_anchor`,
  - `total_non_anchor = floor(sum(raw_non_anchor_j) + 0.5)` (explicit half-up),
  - Hamilton allocate non-anchor counts constrained to `total_non_anchor`,
  - `requested_total_blocks = anchor_epochs * (target_anchor_per_epoch + sum(target_non_anchor_j))`,
- in `explicit_blocks` mode, use configured `total_blocks` directly,
- validation split creation is per source, deterministic, and happens before mixture composition,
- validation split creation reuses existing `_ensure_validation_split` behavior per source,
- mixture mode fails fast unless `number_epochs == 1`,
- v1 fails fast unless `effective_total_blocks == requested_total_blocks`,
- validation execution emits per-source losses and weighted aggregate loss,
- validation uses non-dropping dataloader policy and fails fast when any source has zero validation batches,
- resume fails fast if accounting-critical mixture metadata differs from checkpoint metadata,
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

- config summary (`dataset_id`, configured_weight, effective_weight, weight_mode, sequence_length, budget_mode, anchor_dataset_id, requested_total_blocks, effective_total_blocks, seed/schedule_seed),
- source capacities (`source_blocks_available`),
- runtime context (world_size, global batch settings, drop policies),
- `target_blocks_per_dataset`,
- `realized_blocks_per_dataset`,
- `realized_tokens_per_dataset`,
- `realized_ratios`,
- `deviation_from_target_blocks`,
- validation provenance (`existing_valid` vs `auto_generated_valid`) per source,
- `val_loss_per_dataset`,
- `val_loss_weighted`,
- reproducibility metadata (`hash_algorithm`, `allocation_algorithm`, `split_seed_algorithm`).

Acceptance:

- smoke run produces report with internally consistent totals.

### Milestone 5: SLURM validation and evidence capture

Use the existing SLURM test runner path and keep runtime defaults aligned with `slurm/tests/slurm_test.env`:

- `slurm/tests/run_tests.sh`,
- `slurm/tests/slurm_test.env`,
- optional secrets from `slurm/tests/test_secrets.env`.

Run command:

    ./slurm/tests/run_tests.sh --config config/tests/clm_training_packing_mixture_multinode_smoke.yaml --nodes 2 --ntasks-per-node 1 --partition postiguet1 --gpus 1 --cpus 8 --memory 32G --time 02:00:00

Ensure submitter guard is satisfied (`ALLOWED_SUBMITTERS` in `slurm/tests/slurm_test.env`). If the runner script is unavailable, use:

    ./slurm/submit_job.sh --config config/tests/clm_training_packing_mixture_multinode_smoke.yaml --partition postiguet1 --gpus 1 --nodes 2 --ntasks-per-node 1 --cpus 8 --memory 32G --time 02:00:00

After submission, record completion and logs with:

    sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed

When using `run_tests.sh`, also record the printed `Stdout log` and `Stderr log` paths.

Acceptance:

- SLURM job exits successfully,
- job ID, log paths, exact command, and report path are recorded in `issues/43-token-budget-mixture-replay.md`.

## Plan of Work

1. Update `config/schemas/training/components/data.schema.yaml` with mutually exclusive single-source vs multi-source support and strict validation rules.
2. Add failing unit tests in `tests/unit/training/test_mixture_packing.py` for allocation, determinism, budget resolution, distributed invariants, compatibility guards, and per-source validation auto-split/metric behavior.
3. Implement `src/tasks/training/data/mixture.py` with deterministic index mapping, dual budget-mode support, explicit `blake2b_u64` hashing, Hamilton allocation, canonical anchor-fixed derivation, and no full schedule array.
4. Extend `src/tasks/training/orchestrator.py` to load sources for mixture mode while preserving existing code path for `dataset.nameOrPath`.
5. Extend `src/tasks/training/fabric/trainer/base.py` to build mixture dataloaders, auto-generate per-source validation splits, resolve/validate budgets, enforce resume/validation fail-fast guards, aggregate counters, and write report.
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
   - budget resolution is deterministic and auditable:
     - in `explicit_blocks`, requested budget equals configured `total_blocks`,
     - in `anchor_epochs`, requested budget equals derived per-epoch total times `anchor_epochs`,
   - source weight resolution is deterministic and auditable:
     - all-manual weights use configured values,
     - all-omitted weights use train packed-block counts as effective weights,
     - mixed provided/omitted weights fail fast,
     - any source with zero train packed blocks fails fast with `dataset_id`,
   - mixture mode enforces `number_epochs == 1`,
   - `effective_total_blocks == requested_total_blocks`,
   - `sum(realized_blocks_per_dataset) == effective_total_blocks`,
   - `sum(realized_tokens_per_dataset) == effective_total_blocks * sequence_length`,
   - each dataset’s deviation from deterministic target is exactly explained by documented distributed constraints (no unexplained drift).

3. Validation construction:
   - when a source has no `valid`, validation is auto-generated from that source only (no cross-source leakage),
   - validation split decisions are deterministic and persisted in report metadata,
   - validation logs/report include `val_loss_<dataset_id>` and `val_loss_weighted`,
   - run fails fast if any source has neither `valid` nor usable `validation_split`,
   - run fails fast if any source validation dataloader has zero batches.

4. Safety:
   - incompatible source/tokenizer/eos configs fail fast,
   - incompatible distributed step budgets fail fast,
   - no rank hangs when artifact/index build fails,
   - resume fails fast when accounting-critical mixture metadata differs from checkpoint metadata.

5. Regression:
   - existing online-packing tests continue to pass.
6. Reproducibility and metric checks:
   - each source in `mixture_report.json` includes reproducibility identity fields (`dataset_id`, `nameOrPath`, and dataset fingerprint/revision when available),
   - report includes `weight_mode`, per-source `configured_weight`, and per-source `effective_weight`,
   - logged `seed`, `schedule_seed`, and split-seed algorithm id are present,
   - smoke metric gate passes: `val_loss_weighted` is finite and in `(0, 30)`.

## Idempotence and Recovery

- Re-running schema validation/tests is idempotent.
- Packing index artifacts remain reusable. Rebuild only when intentionally invalidating caches (delete source `.packing_index` directories).
- If a run fails during artifact creation, retry only after confirming no active job holds the relevant lock.
- Do not add fallback paths that silently alter budgets or compatibility outcomes.

## Artifacts and Notes

Reference snippets (normative for v1 behavior):

    def blake2b_u64(text: str) -> int:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)

    def resolve_effective_weights(
        configured_weights_by_id: dict[str, float | None],
        source_blocks: dict[str, int],
    ) -> tuple[dict[str, float], str]:
        ids = sorted(configured_weights_by_id)
        present = [configured_weights_by_id[i] is not None for i in ids]
        if all(present):
            return ({i: float(configured_weights_by_id[i]) for i in ids}, "manual")
        if not any(present):
            return ({i: float(source_blocks[i]) for i in ids}, "from_source_blocks")
        raise ValueError("All sources must either set weight or omit weight.")

    def allocate_exact_counts(total: int, weights_by_id: dict[str, float]) -> dict[str, int]:
        ids = sorted(weights_by_id)  # deterministic tie-break
        z = sum(weights_by_id[i] for i in ids)
        raw = {i: total * weights_by_id[i] / z for i in ids}
        base = {i: int(math.floor(raw[i])) for i in ids}
        rem = total - sum(base.values())
        # largest remainder first; lexicographic dataset_id ascending on ties
        order = sorted(ids, key=lambda i: (-(raw[i] - base[i]), i))
        for i in order[:rem]:
            base[i] += 1
        return base

    def derive_anchor_epoch_targets(
        source_blocks: dict[str, int],
        weights_by_id: dict[str, float],
        anchor_id: str,
    ) -> dict[str, int]:
        b_anchor = source_blocks[anchor_id]
        w_anchor = weights_by_id[anchor_id]
        raw_non_anchor = {
            did: b_anchor * weights_by_id[did] / w_anchor
            for did in sorted(weights_by_id)
            if did != anchor_id
        }
        total_non_anchor = int(math.floor(sum(raw_non_anchor.values()) + 0.5))
        targets_non_anchor = allocate_exact_counts(total_non_anchor, raw_non_anchor)
        return {anchor_id: b_anchor, **targets_non_anchor}

    def permute_index(i: int, total: int, schedule_seed: int) -> int:
        if total <= 1:
            return 0
        stride = (blake2b_u64(f"perm_stride|{schedule_seed}") % total) or 1
        while math.gcd(stride, total) != 1:
            stride = (stride + 1) % total or 1
        offset = blake2b_u64(f"perm_offset|{schedule_seed}") % total
        return (i * stride + offset) % total

    split_seed_i = blake2b_u64(f"valsplit|{base_seed}|{dataset_id}") % 2147483647

    mixture_meta = {
      "weight_mode": weight_mode,
      "sources": [
        {
          "dataset_id": did,
          "configured_weight": configured_weights_by_id[did],
          "effective_weight": weights_by_id[did]
        }
        for did in sorted(weights_by_id)
      ],
      "budget_mode": budget_mode,
      "anchor_dataset_id": anchor_dataset_id,
      "requested_total_blocks": requested_total_blocks,
      "world_size": world_size,
      "batch_size": batch_size,
      "gradient_accumulation_steps": grad_accum,
      "hash_algorithm": "blake2b_u64_v1",
      "allocation_algorithm": "hamilton_lr_lexicographic_v1"
    }

Minimal expected `mixture_report.json` structure:

    {
      "sequence_length": 128,
      "weight_mode": "manual",
      "budget_mode": "explicit_blocks",
      "anchor_dataset_id": null,
      "source_blocks_available": {"A": 1000, "B": 300},
      "requested_total_blocks": 1000,
      "effective_total_blocks": 1000,
      "world_size": 2,
      "dataset_index_map": {"0": "A", "1": "B"},
      "sources": [
        {"dataset_id": "A", "configured_weight": 2.0, "effective_weight": 2.0},
        {"dataset_id": "B", "configured_weight": 1.0, "effective_weight": 1.0}
      ],
      "target_blocks_per_dataset": {"A": 667, "B": 333},
      "realized_blocks_per_dataset": {"A": 667, "B": 333},
      "realized_tokens_per_dataset": {"A": 85376, "B": 42624},
      "realized_ratios": {"A": 0.667, "B": 0.333},
      "val_loss_per_dataset": {"A": 2.10, "B": 2.45},
      "val_loss_weighted": 2.22,
      "validation_sources": {
        "A": {"mode": "auto_generated_valid", "seed": 42},
        "B": {"mode": "existing_valid"}
      },
      "hash_algorithm": "blake2b_u64_v1",
      "allocation_algorithm": "hamilton_lr_lexicographic_v1",
      "split_seed_algorithm": "blake2b_u64_mod_2147483647_v1"
    }

## Revision Note (2026-02-06)

Reworked this ExecPlan to be implementation-ready for a junior developer by removing ambiguity around validation cadence, adding explicit distributed budget invariants, defining exact report/accounting requirements, adding test-first sequencing, and including SLURM runbook details required by AGENTS.md and PLANS.md.

## Revision Note (2026-02-06, update 2)

Updated the plan to support two budget interfaces with one auditable resolved budget (`anchor_epochs` default with largest-source implicit anchor, and `explicit_blocks`), and locked v1 validation behavior to deterministic per-source auto-generated splits when dedicated validation datasets are unavailable.

## Revision Note (2026-02-06, update 3)

Aligned the ExecPlan with the hardened issue contracts by pinning deterministic algorithms (`blake2b_u64` hashing + Hamilton allocation), adding strict mixture-mode invariants (`number_epochs == 1`, `effective_total_blocks == requested_total_blocks`), and making per-source validation metrics/logging requirements explicit.

## Revision Note (2026-02-06, update 4)

Finalized junior-proof runtime semantics by clarifying virtual-epoch budgeting vs trainer epochs, standardizing tensor-safe source identity (`dataset_idx` + mapping), fixing Hamilton tie-break ordering in snippets, and adding reference snippets for deterministic permutation and resume-metadata compatibility checks.

## Revision Note (2026-02-06, update 5)

Locked non-overengineered validation behavior by requiring reuse of existing `FabricTrainerBase._ensure_validation_split` semantics per source instead of introducing a parallel split implementation.

## Revision Note (2026-02-06, update 6)

Corrected the `validate_after_k_steps` observation to match current trainer behavior, added explicit SLURM partition/resource and log-retrieval instructions (with manual fallback), and added reproducibility/metric acceptance gates required for ML-facing validation evidence.

## Revision Note (2026-02-06, update 7)

Added the all-or-none source-weight contract: manual weights remain supported, and when all weights are omitted the plan now requires deterministic derivation from per-source train packed-block capacities, with explicit fail-fast behavior for mixed presence and report/resume metadata updates.

## Revision Note (2026-02-06, update 8)

Added an explicit zero-capacity guard: mixture runs must fail fast when any source has zero train packed blocks after per-source validation splitting, and the failing `dataset_id` must be surfaced in tests and runtime behavior.

## Revision Note (2026-02-06, update 9)

Implemented milestones 1-4 in code: added schema/config surface for `dataset.sources` + `dataset.mixture`, introduced deterministic mixture primitives and `MixturePackedDataset`, integrated orchestrator/trainer mixture flows (including per-source validation construction, strict budget guards, resume metadata checks, and rank-0 report writing), and added local unit/schema validation evidence. SLURM evidence capture remains pending.

## Revision Note (2026-02-09, update 10)

Implemented resume/accounting hardening requested by review findings: persisted and restored `mixture_runtime` counters, versioned and expanded `mixture_meta` (`effective_total_blocks`, `dataset_index_map`), added backward-compatible legacy checkpoint handling, hardened rank-0 index build failure propagation to prevent hangs, persisted explicit `run_metadata`, and logged end-of-run per-source mixture summary metrics for CSV/WandB observability. SLURM evidence capture remains pending.

## Revision Note (2026-02-09, update 11)

Implemented scheduler/metadata hardening requested after follow-up review: added `min_lr`/`max_lr` training controls, fail-fast LR bound validation, bounded scheduler behavior that preserves legacy defaults when bounds are unset, fixed the default-path optimizer param-group LR rewrite regression risk, and persisted/logged scheduler step metadata (`optimizer_steps_per_epoch`, `total_optimizer_steps`, `warmup_steps`, min/peak LR) for checkpoint + CSV/WandB auditability.

## Revision Note (2026-02-09, update 12)

Implemented full-sweep release hardening: corrected multi-node smoke config to use absolute dataset paths compatible with strict SLURM path guards, refreshed `run_metadata` immediately before final mixture report generation to avoid stale counters, and moved mixture realized-block accounting to device-resident counters in the training loop (with checkpoint/report-time sync back to dict state) to reduce HPC hot-path CPU overhead.

## Revision Note (2026-02-09, update 13)

Fixed bounded scheduler short-run behavior by making single-step warmup (`warmup_steps == 1`) start at `min_lr` rather than peak LR, and added a dedicated scheduler regression test to prevent recurrence.

## Revision Note (2026-02-09, update 14)

Closed remaining code-level review findings by making mixture configuration fail fast when `dataset.mixture` is enabled without `dataset.sources` (schema + trainer + orchestrator) and by hardening resume compatibility checks to raise explicit errors when checkpoint `mixture_meta` is malformed (non-dict). Added focused regression coverage for both paths and updated issue tracking with a SLURM evidence closure checklist.
