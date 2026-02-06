# Add `dataset_analysis` task for exact tokenized dataset analytics

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

PLANS.md is checked into the repo at `PLANS.md`, and this document must be maintained in accordance with it.

## Purpose / Big Picture

After this change, users can run a YAML job with `task: dataset_analysis` and receive exact, reproducible statistics about tokenized datasets before training. The workflow must support fixed-window and doc-level tokenization outputs and provide both machine-readable and human-readable artifacts.

## Progress

- [x] (2026-02-06 11:20Z) Create issue card and ExecPlan for issue 44.
- [x] (2026-02-06 11:31Z) Add task wiring (base schema enum, `src/main.py` dispatch, `config/schemas/dataset_analysis.schema.yaml`).
- [x] (2026-02-06 11:36Z) Add failing tests first (`src/tasks/dataset_analysis/test_orchestrator.py`) and observe red due missing module.
- [x] (2026-02-06 11:49Z) Implement analysis orchestrator and module entrypoint; make tests green.
- [x] (2026-02-06 11:52Z) Add example and smoke configs for fixed/doc-level tokenized datasets.
- [x] (2026-02-06 11:56Z) Add config schema tests and run targeted validation/tests with recorded commands and outcomes.
- [x] (2026-02-06 12:06Z) Run an end-to-end CLI execution against a temporary tokenized dataset and verify JSON/Markdown report artifacts are created.

## Surprises & Discoveries

- Observation: `ConfigValidator` discovers schemas by task filename (`<task>.schema.yaml`) recursively under `config/schemas/`.
  Evidence: `src/config/config_loader.py` `_find_schema_file` searches for `f"{task_name}.schema.yaml"`.

- Observation: test/runtime environments in this workspace differ; some have no `datasets` package available by default.
  Evidence: importing `src.utils.dataset.storage` fails without `datasets`, while schema tests still run.

- Observation: `python3 src/main.py --validate ...` requires `PYTHONPATH=.` in this shell to resolve the namespace package `src`.
  Evidence: command fails with `ModuleNotFoundError: No module named 'src'` without the env var and succeeds with it.

## Decision Log

- Decision: Use exact counting for token frequencies and unique token count in v1.
  Rationale: correctness is more important than speed for pre-training data audits.
  Date/Author: 2026-02-06 / Codex

- Decision: Emit both JSON and Markdown in one run.
  Rationale: JSON enables scripting, markdown enables quick human review and issue-card attachments.
  Date/Author: 2026-02-06 / Codex

- Decision: lazily import `DatasetStorage` inside `_load_dataset` and raise a clear runtime error when `datasets` is missing.
  Rationale: keeps module import/test collection stable in lightweight environments while preserving fail-fast behavior at execution time.
  Date/Author: 2026-02-06 / Codex

- Decision: when `compute_padding_metrics` is disabled and effective-token columns are absent, count token-frequency on raw `input_ids`.
  Rationale: users explicitly disabled strict effective-length accounting; raw counts are still valuable and deterministic.
  Date/Author: 2026-02-06 / Codex

## Outcomes & Retrospective

Implemented the issue end-to-end:

- New task wiring and schema were added.
- New orchestrator computes exact split/global metrics and writes JSON + Markdown reports.
- Task-focused unit tests and config schema tests were added and are passing.
- Smoke configs for fixed/doc-level datasets were added and validate successfully.

Remaining optional follow-up: run the full repo test matrix (`tox`) in a fully provisioned environment.

## Context and Orientation

LMTK is YAML-driven: `src/main.py` validates a config with `src/config/config_loader.py` and dispatches by `config.task` to `src/tasks/<task>/`.

This feature introduces a new task, `dataset_analysis`, implemented under `src/tasks/dataset_analysis/`.

The task will analyze already-tokenized Hugging Face datasets loaded from disk and compute exact statistics over selected splits.

## Milestones

### Milestone 1: Wire the new task and schema

Add `dataset_analysis` to `config/schemas/base.schema.yaml`, add dispatch in `src/main.py`, and define `config/schemas/dataset_analysis.schema.yaml`.

Acceptance: config validation recognizes and validates `task: dataset_analysis`.

### Milestone 2: Implement exact analysis engine

Implement `src/tasks/dataset_analysis/orchestrator.py` and `src/tasks/dataset_analysis/__init__.py`.

Expected behavior:

- Load tokenized dataset from disk.
- Resolve requested splits.
- Compute per-split and global metrics.
- Fail fast on invalid inputs/unsupported metric requirements.
- Save `<report_name>.json` and `<report_name>.md` to `output.path`.

Acceptance: running a smoke config writes both artifacts with expected sections and values.

### Milestone 3: Add tests and smoke configs

Create task-focused tests in `src/tasks/dataset_analysis/test_orchestrator.py` and add configs:

- `config/examples/dataset_analysis_example.yaml`
- `config/tests/dataset_analysis_smoke_fixed.yaml`
- `config/tests/dataset_analysis_smoke_doclevel.yaml`

Acceptance: targeted unit tests pass and schema validation succeeds for smoke configs.

## Plan of Work

1. Add schema wiring for new task enum + task-specific schema.
2. Add failing tests for:
   - fixed-window metrics,
   - doc-level metrics,
   - split selection,
   - fail-fast when padding metrics are requested but effective tokens cannot be derived.
3. Implement orchestrator with deterministic, exact aggregation.
4. Add report writers (JSON + markdown).
5. Add example/smoke configs and run targeted validation/tests.

## Concrete Steps

Run from repository root:

    python3 -m pytest src/tasks/dataset_analysis/test_orchestrator.py tests/unit/config/test_dataset_analysis_schema.py -q
    PYTHONPATH=. python3 src/main.py --validate --config config/tests/dataset_analysis_smoke_fixed.yaml
    PYTHONPATH=. python3 src/main.py --validate --config config/tests/dataset_analysis_smoke_doclevel.yaml

If the full env is available:

    ./.venv/bin/python -m tox -e py310

## Validation and Acceptance

Acceptance is met when:

- New task validates through schema and dispatches correctly.
- Reports are produced in both formats.
- Metrics are exact and consistent with deterministic unit-test fixtures.
- Fail-fast conditions produce explicit error messages.

## Idempotence and Recovery

- Re-running analysis with the same config overwrites report files deterministically.
- If validation fails, no partial report artifacts should remain in output directory for that run.
- Any runtime error should surface clearly and stop execution.

## Artifacts and Notes

- Primary artifact paths:
  - `<output.path>/<report_name>.json`
  - `<output.path>/<report_name>.md`
- Validation evidence captured during implementation:
  - `python3 -m pytest src/tasks/dataset_analysis/test_orchestrator.py tests/unit/config/test_dataset_analysis_schema.py -q` => `6 passed`
  - `PYTHONPATH=. python3 src/main.py --validate --config config/tests/dataset_analysis_smoke_fixed.yaml` => `Configuration is valid!`
  - `PYTHONPATH=. python3 src/main.py --validate --config config/tests/dataset_analysis_smoke_doclevel.yaml` => `Configuration is valid!`
  - `PYTHONPATH=. ./.venv/bin/python src/main.py --config /tmp/<generated-config>.yaml` on a temporary 2-row tokenized dataset => reports created at:
    - `/tmp/lmtk_ds_analysis_is_uckra/out/tmp_report.json`
    - `/tmp/lmtk_ds_analysis_is_uckra/out/tmp_report.md`

## Interfaces and Dependencies

New task schema fields:

- `dataset`:
  - `source: local` (v1)
  - `nameOrPath: <path to tokenized dataset>`
  - `format: hf | dataset`
- `analysis`:
  - `splits: [string]` (optional)
  - `top_k: int` (default 100)
  - `compute_padding_metrics: bool` (default true)
  - `compute_token_frequency: bool` (default true)
  - `compute_unique_tokens: bool` (default true)
  - `report_name: string` (default `dataset_analysis`)
- `output.path` required.

### Revision Notes

- 2026-02-06 / Codex: Initial ExecPlan creation for issue 44 based on requested implementation scope.
- 2026-02-06 / Codex: Updated progress/decisions/discoveries after implementation and recorded concrete validation commands and results.
- 2026-02-06 / Codex: Added end-to-end runtime evidence from an actual CLI run on a temporary tokenized dataset.
