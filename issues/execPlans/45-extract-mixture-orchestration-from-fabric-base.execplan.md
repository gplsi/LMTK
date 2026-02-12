# Extract Mixture Orchestration from `FabricTrainerBase` with Zero Behavioral Drift

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

PLANS.md is checked into the repo at `PLANS.md`, and this document must be maintained in accordance with it.

Issue reference: `issues/45-extract-mixture-orchestration-from-fabric-base.md`.

## Purpose / Big Picture

After this refactor, mixture functionality behaves exactly as it does today for users and SLURM jobs, but the mixture orchestration code is no longer concentrated in `src/tasks/training/fabric/trainer/base.py`. A junior engineer should be able to locate mixture setup logic and mixture runtime/resume/report logic in dedicated modules and modify one area without scanning the full base trainer.

A successful implementation is observable as follows: mixture configs that worked before still work without YAML changes, mixture reports keep the same keys and semantics, and the targeted tests pass before and after extraction with no behavior deltas.

## Progress

- [x] (2026-02-11 21:26Z) Created issue card `issues/45-extract-mixture-orchestration-from-fabric-base.md`.
- [x] (2026-02-11 21:26Z) Created initial ExecPlan with extraction milestones.
- [x] (2026-02-11 22:40Z) Hardened this ExecPlan for junior-proof execution: explicit terms, file-level test targets, SLURM guard/env instructions, deterministic acceptance checks, and removal of ambiguous implementation instructions.
- [ ] Add characterization tests in existing mixture test modules so behavior is locked before method moves.
- [ ] Extract mixture setup/config methods into `src/tasks/training/fabric/trainer/mixture_setup_mixin.py`.
- [ ] Extract mixture runtime/resume/report methods into `src/tasks/training/fabric/trainer/mixture_runtime_mixin.py`.
- [ ] Run local targeted tests and SLURM smoke runs; record job IDs/log paths and report evidence in issue 45.

## Surprises & Discoveries

- Observation: mixture logic currently spans configuration parsing, dataloader construction, runtime counters, resume metadata checks, and report generation in one class.
  Evidence: method cluster in `src/tasks/training/fabric/trainer/base.py` includes `_get_mixture_config`, `_build_mixture_packing_dataloaders`, `_build_mixture_resume_meta`, `_validate_mixture_resume_compatibility`, and `_write_mixture_report`.

- Observation: local Python/Torch environment instability has occurred in this workspace during some test runs.
  Evidence: prior local `pytest` attempts have failed with torch import/runtime crashes, so this plan includes SLURM-based validation as required evidence.

- Observation: SLURM test submission is explicitly gated by allowed submitters.
  Evidence: `slurm/tests/run_tests.sh` requires `ALLOWED_SUBMITTERS` in `slurm/tests/slurm_test.env` and blocks unauthorized users.

## Decision Log

- Decision: perform a mechanical extraction into mixins without changing behavior, config schema, or public APIs.
  Rationale: this is the lowest-risk way to reduce `base.py` size while preserving issue 42/43 contracts.
  Date/Author: 2026-02-11 / Codex

- Decision: keep strict fail-fast semantics unchanged (`explicit_blocks` divisibility constraints, resume metadata incompatibility checks, invalid-state failures).
  Rationale: these guards are safety contracts and must not regress during refactor.
  Date/Author: 2026-02-11 / Codex

- Decision: require characterization tests before moving methods.
  Rationale: behavior locks are needed so extraction remains structural and reviewable.
  Date/Author: 2026-02-11 / Codex

## Outcomes & Retrospective

To be completed during implementation. Final entry must summarize:

- whether all invariants remained unchanged,
- which methods moved and where,
- what test/SLURM evidence proves no behavior regression,
- any follow-up issue required for further decomposition.

## Context and Orientation

LMTK is YAML-driven: `src/main.py` validates a config and dispatches to task modules under `src/tasks/`. Mixture training behavior and contracts were previously introduced and hardened in issues 42 and 43.

Scope guard for this plan:

- This is an internal trainer refactor only.
- Do not change config schemas in `config/schemas/`.
- Do not change task dispatch in `src/main.py`.
- Do not change mixture allocation math in `src/tasks/training/data/mixture.py`.
- Do not introduce compatibility fallbacks that hide invalid states.

Terms used in this plan:

- Behavioral drift: any externally observable difference in mixture training behavior, failure modes, report fields, or resume compatibility relative to pre-refactor behavior.
- Mixin: a Python class containing methods intended to be inherited by another class (`FabricTrainerBase`) to separate responsibilities without changing call sites.
- Characterization test: a test that locks current behavior before refactoring, so changes that alter behavior fail loudly.

Current mixture methods in `src/tasks/training/fabric/trainer/base.py`:

- Setup/config path:
  - `_get_mixture_config`
  - `_get_sources_config`
  - `_is_mixture_enabled`
  - `_derive_validation_split_seed`
  - `_collect_source_config_map`
  - `_validate_mixture_source_compatibility`
  - `_build_mixture_packing_dataloaders`
- Runtime/resume/report path:
  - `_ensure_mixture_realized_counter_tensor`
  - `_sync_mixture_realized_blocks_local_from_tensor`
  - `_build_mixture_runtime_state`
  - `_restore_mixture_runtime_state`
  - `_build_mixture_resume_meta`
  - `_validate_mixture_resume_compatibility`
  - `_reduce_mixture_realized_blocks`
  - `_write_mixture_report`
  - `_validate_mixture`

This refactor must preserve all existing call sites and output contracts.

## Milestones

### Milestone 1: Lock Current Behavior with Characterization Tests

Add or extend tests before code movement so behavior is frozen. Keep tests in existing cross-cutting paths because this refactor affects trainer internals rather than one isolated task module.

Files to edit:

- `tests/unit/training/test_mixture_packing.py`
- `tests/unit/training/test_mixture_trainer_contracts.py`
- `tests/unit/config/test_online_packing_schema.py` (only if schema-adjacent assertions are needed)

Minimum behaviors to lock:

- anchor-mode alignment behavior remains stable,
- explicit-block strict failure behavior remains stable,
- resume metadata incompatibility still fails with explicit error,
- mixture report includes required keys and invariant relationships.

Tests to add or update (use these names unless an equivalent existing test already covers the exact contract):

- `tests/unit/training/test_mixture_packing.py::test_anchor_mode_alignment_contract_is_stable`
- `tests/unit/training/test_mixture_packing.py::test_explicit_blocks_divisibility_contract_is_stable`
- `tests/unit/training/test_mixture_trainer_contracts.py::test_mixture_report_contract_keys_and_totals_are_stable`
- `tests/unit/training/test_mixture_trainer_contracts.py::test_resume_meta_incompatibility_contract_is_stable`

Acceptance for milestone 1:

- tests fail if behavior is intentionally changed,
- tests pass on baseline before extraction.

### Milestone 2: Extract Setup/Config Methods to a Dedicated Mixin

Create `src/tasks/training/fabric/trainer/mixture_setup_mixin.py` and move only setup/config methods listed in Context and Orientation. Preserve method names, signatures, logging, exceptions, and return semantics.

Update `src/tasks/training/fabric/trainer/base.py` class inheritance to include this mixin. Keep all pre-existing `self` attributes initialized where they are today unless a move is strictly mechanical and behavior-equivalent.

Required class shape after extraction:

- `class FabricTrainerBase(MixtureSetupMixin, MixtureRuntimeMixin, ABC):`
- No method name collisions between mixins are allowed; if a collision appears, stop and resolve structurally without changing runtime behavior.

Acceptance for milestone 2:

- no changed behavior in setup/config path,
- milestone 1 tests still pass.

### Milestone 3: Extract Runtime/Resume/Reporting Methods to a Dedicated Mixin

Create `src/tasks/training/fabric/trainer/mixture_runtime_mixin.py` and move runtime/resume/report methods listed in Context and Orientation. Preserve metadata keys and compatibility checks exactly.

Update `src/tasks/training/fabric/trainer/base.py` inheritance to include the runtime mixin. Preserve ordering and call behavior so method resolution stays deterministic.

Acceptance for milestone 3:

- checkpoint resume compatibility behavior is unchanged,
- report field names and semantics are unchanged,
- milestone 1 tests still pass.

### Milestone 4: Validate End-to-End and Capture Evidence

Run local targeted tests and then SLURM smoke tests through the existing test runner wrapper.

Acceptance for milestone 4:

- local targeted tests pass (or documented local limitation exists),
- SLURM smoke run completes and evidence is recorded,
- issue 45 includes exact commands, job IDs, log paths, and a short no-regression conclusion.

## Plan of Work

Start with tests, then perform a two-phase mechanical extraction. Do not change behavior while moving methods.

First, add characterization assertions in the listed test files. Then extract setup/config methods into `mixture_setup_mixin.py` and wire base trainer inheritance. Re-run targeted tests. Next, extract runtime/resume/report methods into `mixture_runtime_mixin.py` and wire inheritance. Re-run tests. Finally, run SLURM smoke validation and capture evidence in issue 45.

If any extraction step changes behavior, stop and restore the last known-good structural state before retrying with smaller moves.

## Concrete Steps

Run commands from repository root: `/Users/ernestoluisestevanellvalladares/Repositorios/LMTK`.

1. Baseline and characterization tests:

    PYTHONPATH=. pytest -q -p no:debugging tests/unit/training/test_mixture_packing.py
    PYTHONPATH=. pytest -q -p no:debugging tests/unit/training/test_mixture_trainer_contracts.py
    PYTHONPATH=. pytest -q -p no:debugging tests/unit/config/test_online_packing_schema.py

2. Implement Milestone 2 (setup mixin extraction), then rerun step 1.

3. Implement Milestone 3 (runtime mixin extraction), then rerun step 1.

4. SLURM precheck (required before submission):

    grep '^ALLOWED_SUBMITTERS=' slurm/tests/slurm_test.env
    grep -E '^(PARTITION|GPU_COUNT|TIME_LIMIT|MEMORY|NODES|NTASKS_PER_NODE)=' slurm/tests/slurm_test.env

   Confirm current user appears in `ALLOWED_SUBMITTERS`. Confirm default test resources are correct (defaults are expected to target `postiguet1` + 1x RTX 4090 unless the test needs overrides). If not, update `slurm/tests/slurm_test.env` before submitting tests.

5. SLURM integration smoke (single-node):

    bash slurm/tests/run_tests.sh --config config/tests/mixture_packing_integration_smoke.yaml --memory 64G

6. Optional multi-node smoke (only if shared absolute dataset paths are valid for this cluster):

    bash slurm/tests/run_tests.sh --config config/tests/clm_training_packing_mixture_multinode_smoke.yaml --memory 96G --nodes 2 --ntasks-per-node 4

Expected runner output should include:

    Selected config: <config path>
    Submitted batch job <job_id>
    Job ID: <job_id>
    Stdout log: <path>
    Stderr log: <path>

## Validation and Acceptance

The refactor is accepted only when all checks below pass.

Local deterministic checks:

- targeted tests pass,
- pre/post extraction report keys are unchanged,
- strict error contracts are unchanged for known invalid cases.

SLURM checks:

- integration smoke config completes via `slurm/tests/run_tests.sh`,
- job logs show no new mixture contract failures,
- evidence is recorded in issue 45.

Report invariants to verify from generated `mixture_report.json`:

- required keys exist: `requested_total_blocks`, `effective_total_blocks`, `alignment_policy`, `alignment_unit`, `alignment_applied`, per-source realized/target fields,
- sum of realized source blocks equals `effective_total_blocks`,
- for strict budget runs, `effective_total_blocks` matches expected resolved value and no silent correction occurs.

Runtime error contract checks:

- In explicit-block strict runs, non-divisible block budgets still fail with `ValueError` and clear message.
- Resume metadata incompatibility still fails with explicit error (no implicit fallback/recovery path).

## Idempotence and Recovery

This plan is idempotent: rerunning tests and SLURM smoke from a clean checkout should produce equivalent outcomes.

If a move introduces behavioral drift, revert only the most recent extraction chunk and rerun the characterization tests before attempting a smaller move. Do not rename report keys, do not weaken failure conditions, and do not alter checkpoint metadata schema in this issue.

If local environment instability blocks tests, record the exact local error and use SLURM validation evidence as the authoritative path.

## Artifacts and Notes

Record all evidence in `issues/45-extract-mixture-orchestration-from-fabric-base.md` during implementation:

- local test command outputs (pass/fail),
- SLURM submission command(s),
- SLURM job ID(s),
- stdout/stderr paths,
- brief statement confirming whether behavior matches baseline.

Keep evidence concise and outcome-focused.

## Interfaces and Dependencies

New internal files to create:

- `src/tasks/training/fabric/trainer/mixture_setup_mixin.py`
- `src/tasks/training/fabric/trainer/mixture_runtime_mixin.py`

`FabricTrainerBase` in `src/tasks/training/fabric/trainer/base.py` must continue to expose and call the same internal mixture methods after inheritance wiring. `src/tasks/training/data/mixture.py` remains the single source of truth for mixture allocation/alignment math; do not duplicate those algorithms in mixins.

Required interface contract:

- `src/tasks/training/fabric/trainer/mixture_setup_mixin.py` defines `class MixtureSetupMixin:`
- `src/tasks/training/fabric/trainer/mixture_runtime_mixin.py` defines `class MixtureRuntimeMixin:`
- `src/tasks/training/fabric/trainer/base.py` composes both mixins in `FabricTrainerBase` and preserves existing public/internal behavior.
- Existing method names listed in Context and Orientation remain callable from `FabricTrainerBase` call sites without renaming.

No new external dependencies are introduced.

### Revision Notes

- 2026-02-11 / Codex: Initial ExecPlan created for issue 45 with mechanical extraction strategy.
- 2026-02-11 / Codex: Revised for strict PLANS.md compliance and junior-proof execution (definitions, explicit test files, SLURM guard instructions, deterministic acceptance checks, and unambiguous execution steps).
- 2026-02-11 / Codex: Added explicit scope guardrails, required class/interface shape, and explicit runtime error-contract checks to remove remaining implementation ambiguity.
