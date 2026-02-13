---
number: 45
title: "Extract Mixture Orchestration from FabricTrainerBase without behavior regressions"
state: open
labels:
  - enhancement
  - training
  - refactor
  - maintainability
  - continual-pretraining
---

## Summary

Refactor mixture-specific orchestration logic out of `src/tasks/training/fabric/trainer/base.py` into focused trainer modules, while preserving the exact runtime behavior introduced in issues 42 and 43 (and follow-up hardening).

This issue is explicitly about maintainability and testability, not new user-facing features.

This issue is the explicit approval for a significant internal refactor (new-file extraction) under AGENTS.md constraints.

## Status (2026-02-13)

- Refactor implementation is in progress and aligned with approved scope.
- Mixture orchestration has been extracted from `src/tasks/training/fabric/trainer/base.py` into canonical trainer modules:
  - stable mixin entrypoints:
    - `src/tasks/training/fabric/trainer/mixture_setup_mixin.py`
    - `src/tasks/training/fabric/trainer/mixture_runtime_mixin.py`
  - internal implementation package:
    - `src/tasks/training/fabric/trainer/mixture/setup.py`
    - `src/tasks/training/fabric/trainer/mixture/runtime.py`
    - `src/tasks/training/fabric/trainer/mixture/resume.py`
    - `src/tasks/training/fabric/trainer/mixture/reporting.py`
    - `src/tasks/training/fabric/trainer/mixture/constants.py`
- `FabricTrainerBase` now composes both mixins and was reduced from 3507 lines to 2050 lines while preserving method call sites.
- Characterization coverage for report contracts was added in `tests/unit/training/test_mixture_trainer_contracts.py`:
  - `test_mixture_report_contract_keys_and_totals_are_stable`
  - `test_mixture_report_anchor_window_fields_are_stable`
- Local targeted pytest remains blocked in this workspace by a Python 3.13/Torch import segmentation fault; deterministic compile/format checks pass.
- SLURM smoke validation remains recommended before issue closure.

## Problem

`FabricTrainerBase` currently contains a large and growing amount of mixture-specific logic (budget resolution, alignment handling, metadata/resume compatibility, report generation, and validation wiring). This increases risk for future changes and makes regressions harder to prevent.

Recent work confirms the mixture path is functionally valuable in production-like SLURM runs, but the code concentration in a single class is now a long-term maintenance risk.

## Background / Current State

- Mixture functionality is implemented in:
  - `src/tasks/training/fabric/trainer/base.py` (orchestration and state management),
  - `src/tasks/training/data/mixture.py` (allocation/scheduling primitives),
  - related trainer/config tests in `tests/unit/training/` and `tests/unit/config/`.
- Current contracts include:
  - strict behavior for `explicit_blocks`,
  - anchor-mode alignment behavior,
  - deterministic source allocation/reporting,
  - resume metadata compatibility checks,
  - distributed fail-fast invariants.
- Issue 43 and later hardening passes introduced behavior that must remain stable.
- Before extraction, `src/tasks/training/fabric/trainer/base.py` was 3507 lines and contained both generic Fabric trainer responsibilities and a large mixture-specific method cluster, including setup, runtime counters, replay/anchor-window accounting, resume checks, and report generation.

## Goals

1. Reduce responsibility concentration in `FabricTrainerBase` by extracting mixture-specific orchestration code into dedicated modules/classes.
2. Preserve existing runtime behavior and metadata/report outputs.
3. Keep checkpoint/resume compatibility logic intact (no silent contract drift).
4. Improve test surface clarity so future mixture changes are safer.

## Non-goals

- No changes to mixture semantics or budget math.
- No schema redesign beyond what is required for refactor wiring.
- No training-loop feature additions.
- No SLURM submission workflow changes.
- No checkpoint format migrations for this issue.

## Scope

Primary code paths:

- `src/tasks/training/fabric/trainer/base.py`
- New extracted trainer modules under `src/tasks/training/fabric/trainer/` (canonical for this issue):
  - Stable mixin entrypoints:
    - `src/tasks/training/fabric/trainer/mixture_setup_mixin.py`
    - `src/tasks/training/fabric/trainer/mixture_runtime_mixin.py`
  - Internal implementation package:
    - `src/tasks/training/fabric/trainer/mixture/__init__.py`
    - `src/tasks/training/fabric/trainer/mixture/setup.py`
    - `src/tasks/training/fabric/trainer/mixture/runtime.py`
    - `src/tasks/training/fabric/trainer/mixture/resume.py`
    - `src/tasks/training/fabric/trainer/mixture/reporting.py`
- `src/tasks/training/data/mixture.py` (only if small mechanical moves are needed)
- Tests:
  - `tests/unit/training/test_mixture_trainer_contracts.py`
  - `tests/unit/training/test_mixture_packing.py`
  - `tests/unit/training/test_orchestrator_mixture_loading.py`
  - `tests/unit/config/test_online_packing_schema.py`
  - any new focused trainer-module tests created by the implementation

Documentation/process artifacts:

- `issues/execPlans/45-extract-mixture-orchestration-from-fabric-base.execplan.md`

Method-level extraction map (current as of 2026-02-13):

- Setup/config path methods exposed via `MixtureSetupMixin` (implementation may be delegated to `trainer/mixture/setup.py`):
  - `_get_mixture_config`
  - `_get_sources_config`
  - `_is_mixture_enabled`
  - `_derive_validation_split_seed`
  - `_collect_source_config_map`
  - `_validate_mixture_source_compatibility`
  - `_build_mixture_packing_dataloaders`
- Runtime/resume/report path methods exposed via `MixtureRuntimeMixin` (implementation may be delegated across `trainer/mixture/runtime.py`, `trainer/mixture/resume.py`, and `trainer/mixture/reporting.py`):
  - `_validate_mixture`
  - `_ensure_mixture_realized_counter_tensor`
  - `_ensure_mixture_replayed_counter_tensor`
  - `_sync_mixture_realized_blocks_local_from_tensor`
  - `_sync_mixture_replayed_draws_local_from_tensor`
  - `_sync_mixture_anchor_window_start_local_from_tensors`
  - `_advance_mixture_anchor_window_progress`
  - `_build_mixture_runtime_state`
  - `_restore_mixture_runtime_state`
  - `_build_mixture_resume_meta`
  - `_validate_mixture_resume_compatibility`
  - `_reduce_mixture_realized_blocks`
  - `_reduce_mixture_replayed_draws`
  - `_reduce_mixture_anchor_window_start_realized`
  - `_reduce_mixture_anchor_window_start_replayed`
  - `_write_mixture_report`

Code organization policy for this issue:

- Do not create large scripts that contain trainer domain logic.
- Scripts (if any are needed) must remain thin entrypoints only (argument parsing, invocation, exit), while domain logic lives in importable modules under `src/tasks/...`.
- Prefer cohesive subdirectories over one large file when a concern has multiple stable responsibilities.
- Avoid over-fragmentation: do not split into one-file-per-function.
- Any new file/subdirectory must include a short rationale in implementation notes (issue updates or ExecPlan progress entries) describing why that location is correct.

## Explicit Boundaries

Allowed:

- Mechanical method extraction and import rewiring for mixture-specific trainer methods.
- Test additions/updates strictly for behavior locking and non-regression proof.

Not allowed in this issue:

- Changes to config schemas under `config/schemas/`.
- Changes to task dispatch in `src/main.py`.
- Changes to dataset allocation math in `src/tasks/training/data/mixture.py` (except trivial import/typing cleanup if strictly needed).
- Changes to SLURM submission scripts beyond test-command usage.
- Any fallback behavior replacing current fail-fast errors.
- Moving mixture trainer logic into large ad-hoc scripts instead of `src/tasks/...` modules.

## Refactor Constraints (must hold)

- No behavior regression in:
  - resolved requested/effective blocks,
  - alignment policy outcomes,
  - dataloader construction semantics,
  - report output keys/meanings,
  - resume compatibility checks and failure modes.
- No fallback paths that mask invalid state.
- Fail-fast behavior must remain explicit and auditable.
- Refactor must be incremental and test-backed.
- Resulting code layout must improve organization (cohesive module boundaries) while keeping call-site behavior unchanged.

## Invariants (must remain true)

- Existing runtime configs for mixture training continue to run without YAML changes.
- Existing checkpoint resume compatibility rules remain unchanged.
- Existing report schema/keys in `mixture_report.json` remain unchanged.
- Existing strict fail-fast behavior for invalid mixture states remains unchanged.
- Existing source-allocation determinism remains unchanged for fixed seeds.

## Regression Traceability (issues 42 and 43)

- Contract: explicit-block and anchor-mode budget/allocation behavior remains unchanged.
  - Source: `issues/42-online-packing.md`, `issues/43-token-budget-mixture-replay.md`.
  - Required checks: `tests/unit/training/test_mixture_packing.py` (allocation/alignment tests) and no changes in `src/tasks/training/data/mixture.py` math.
- Contract: mixture resume metadata/runtime compatibility remains fail-fast and strict.
  - Source: issue 43 hardening passes.
  - Required checks: `tests/unit/training/test_mixture_trainer_contracts.py` resume tests and strict mismatch tests.
- Contract: report schema/semantics stay stable, including replay and anchor-window accounting.
  - Source: issue 43 report and runtime accounting contracts.
  - Required checks: trainer contract tests asserting report keys/relations and end-to-end smoke report generation.
- Contract: `dataset.mixture -> dataset.sources` fail-fast behavior remains intact.
  - Source: issue 43 pass-6 guardrails.
  - Required checks: `tests/unit/config/test_online_packing_schema.py` + `tests/unit/training/test_orchestrator_mixture_loading.py`.
- Contract: no fallback behavior masking invalid state is introduced during extraction.
  - Source: AGENTS.md and issue 42/43 design constraints.
  - Required checks: existing negative-path tests continue to assert explicit `ValueError`/validation failures.

## Acceptance Criteria

1. Mixture orchestration responsibilities are moved out of `FabricTrainerBase` into dedicated modules with clear interfaces.
2. Full method extraction scope (including replay and anchor-window runtime helpers) is covered as listed in this issue and mirrored in the ExecPlan.
3. Existing unit tests for mixture behavior remain green; new characterization tests cover extracted boundaries not currently locked.
4. End-to-end mixture smoke configs still validate and execute with unchanged semantics.
5. Report and metadata fields for mixture runs remain backward-compatible for existing consumers.
6. Changes are documented in the ExecPlan with a concrete non-regression validation path tied to issue 42/43 contracts.
7. Validation evidence is captured with exact commands, expected outputs, and (for SLURM) job ID + stdout/stderr log paths.
8. No changes are introduced outside the explicit boundaries for this issue.
9. Final structure is organized and refactorable: trainer logic resides in `src/tasks/...` modules with a canonical `trainer/mixture/` split by responsibility plus stable mixin entrypoints; no large script-based implementation is introduced.

## Validation Evidence Required

- Recommended local targeted tests:
  - `PYTHONPATH=. pytest -q -p no:debugging tests/unit/training/test_mixture_packing.py`
  - `PYTHONPATH=. pytest -q -p no:debugging tests/unit/training/test_mixture_trainer_contracts.py`
  - `PYTHONPATH=. pytest -q -p no:debugging tests/unit/training/test_orchestrator_mixture_loading.py`
  - `PYTHONPATH=. pytest -q -p no:debugging tests/unit/config/test_online_packing_schema.py`
- Recommended SLURM smoke:
  - `bash slurm/tests/run_tests.sh --config config/tests/mixture_packing_integration_smoke.yaml --memory 64G`
  - Optional multinode: `bash slurm/tests/run_tests.sh --config config/tests/clm_training_packing_mixture_multinode_smoke.yaml --memory 96G --nodes 2 --ntasks-per-node 4`
- Recommended records in issue updates:
  - Local pass/fail summary
  - SLURM job ID
  - SLURM stdout/stderr paths
  - `mixture_report.json` output path used for verification
  - Any deviations and rationale

Execution note:

- Local execution in this workspace may fail due to a Python 3.13/Torch import segmentation fault; if so, document the exact traceback and continue with available non-regression evidence (SLURM evidence is recommended but not mandatory for issue closure).

Latest execution evidence (2026-02-13):

- `python -m black --check src/tasks/training/fabric/trainer/mixture_setup_mixin.py src/tasks/training/fabric/trainer/mixture_runtime_mixin.py src/tasks/training/fabric/trainer/mixture/*.py tests/unit/training/test_mixture_trainer_contracts.py` -> pass
- `python -m isort --check src/tasks/training/fabric/trainer/mixture_setup_mixin.py src/tasks/training/fabric/trainer/mixture_runtime_mixin.py src/tasks/training/fabric/trainer/mixture/*.py tests/unit/training/test_mixture_trainer_contracts.py` -> pass
- `python -m py_compile src/tasks/training/fabric/trainer/base.py src/tasks/training/fabric/trainer/mixture_setup_mixin.py src/tasks/training/fabric/trainer/mixture_runtime_mixin.py src/tasks/training/fabric/trainer/mixture/__init__.py src/tasks/training/fabric/trainer/mixture/constants.py src/tasks/training/fabric/trainer/mixture/setup.py src/tasks/training/fabric/trainer/mixture/runtime.py src/tasks/training/fabric/trainer/mixture/resume.py src/tasks/training/fabric/trainer/mixture/reporting.py tests/unit/training/test_mixture_trainer_contracts.py` -> pass
- `PYTHONPATH=. pytest -q -p no:debugging tests/unit/training/test_mixture_packing.py tests/unit/training/test_mixture_trainer_contracts.py tests/unit/training/test_orchestrator_mixture_loading.py tests/unit/config/test_online_packing_schema.py` -> fails before collection with `Fatal Python error: Segmentation fault` while importing torch under Python 3.13

## Risks / Mitigations

- Risk: subtle drift in resume/report metadata fields.
  - Mitigation: snapshot/contract assertions and strict test checks before/after extraction.
- Risk: accidental changes in alignment/budget flow.
  - Mitigation: preserve pure allocation/alignment functions and add explicit boundary tests.
- Risk: over-refactor beyond approved scope.
  - Mitigation: keep extraction mechanical and module-local; avoid public API changes.

## Related Work

- `issues/42-online-packing.md`
- `issues/43-token-budget-mixture-replay.md`
- `issues/execPlans/42-online-packing.execplan.md`
- `issues/execPlans/43-token-budget-mixture-replay.execplan.md`
