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
- New extracted trainer modules under `src/tasks/training/fabric/trainer/`:
  - `src/tasks/training/fabric/trainer/mixture_setup_mixin.py`
  - `src/tasks/training/fabric/trainer/mixture_runtime_mixin.py`
- `src/tasks/training/data/mixture.py` (only if small mechanical moves are needed)
- Tests:
  - `tests/unit/training/test_mixture_trainer_contracts.py`
  - `tests/unit/training/test_mixture_packing.py`
  - `tests/unit/config/test_online_packing_schema.py`
  - any new focused trainer-module tests created by the implementation

Documentation/process artifacts:

- `issues/execPlans/45-extract-mixture-orchestration-from-fabric-base.execplan.md`

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

## Invariants (must remain true)

- Existing runtime configs for mixture training continue to run without YAML changes.
- Existing checkpoint resume compatibility rules remain unchanged.
- Existing report schema/keys in `mixture_report.json` remain unchanged.
- Existing strict fail-fast behavior for invalid mixture states remains unchanged.
- Existing source-allocation determinism remains unchanged for fixed seeds.

## Acceptance Criteria

1. Mixture orchestration responsibilities are moved out of `FabricTrainerBase` into dedicated modules with clear interfaces.
2. Existing unit tests for mixture behavior remain green; new tests cover extracted boundaries.
3. End-to-end mixture smoke config still validates and executes with unchanged semantics.
4. Report and metadata fields for mixture runs remain backward-compatible for existing consumers.
5. Changes are documented in the ExecPlan with a concrete non-regression validation path.
6. Validation evidence is captured with exact commands, expected outputs, and (for SLURM) job ID + stdout/stderr log paths.
7. No changes are introduced outside the explicit boundaries for this issue.

## Validation Evidence Required

- Local targeted tests:
  - `PYTHONPATH=. pytest -q -p no:debugging tests/unit/training/test_mixture_packing.py`
  - `PYTHONPATH=. pytest -q -p no:debugging tests/unit/training/test_mixture_trainer_contracts.py`
  - `PYTHONPATH=. pytest -q -p no:debugging tests/unit/config/test_online_packing_schema.py`
- SLURM smoke:
  - `bash slurm/tests/run_tests.sh --config config/tests/mixture_packing_integration_smoke.yaml --memory 64G`
- Required records in issue updates:
  - Local pass/fail summary
  - SLURM job ID
  - SLURM stdout/stderr paths
  - Any deviations and rationale

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
