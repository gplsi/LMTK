---
number: 38
title: "Add SLURM-based test runner and test configs"
state: closed
labels:
---

### Description

Local test execution is incomplete because key dependencies and GPUs live inside the SLURM runtime. We need a consistent, auditable SLURM-based test runner and a set of minimal YAML test configs so every change can be validated on the cluster.

### Motivation

- Ensure every change can run tests even when local deps are missing.
- Make test runs reproducible and auditable (job IDs, logs, exact commands).
- Standardize on small Llama-family models and minimal datasets for fast feedback.

### Background / Investigation

- `slurm/p.slurm` currently requires `CONFIG_FILE` and sets cache env vars; test mode must reuse environment setup while skipping config validation.
- `slurm/submit_job.sh` shows the `sbatch`/`--export` pattern and output/error flags; the test runner should mirror this to avoid divergence.
- `src/main.py` validates YAML against JSON schemas; new configs must satisfy required fields in `config/schemas/training/clm_training.schema.yaml` and `config/schemas/tokenization/tokenization.schema.yaml`.
- `src/tasks/tokenization/orchestrator.py` requires `tokenizer.tokenizer_name`, plus `context_length` or `max_sequence_length`, and `output.path`.

### Proposed Solution

- Add `slurm/tests/slurm_test.env` with safe test defaults (partition `postiguet1`, 1x RTX 4090, short time limits).
- Add `slurm/tests/run_tests.sh` that:
  - sources `slurm/tests/slurm_test.env`,
  - optionally sources `slurm/tests/test_secrets.env`,
  - enforces an allowed submitter list,
-  - submits a test job via `slurm/submit_job.sh` and prints job ID + log paths.
- Add a `task: testing` configuration and schema so tests run through `src/main.py` like other tasks.
- Add `config/tests/unit_tests.yaml` for unit tests and `config/tests/` smoke configs for end-to-end testing (tokenization + training).
- Define shared test defaults (including model IDs) in `config/tests/defaults.yaml` and keep smoke configs aligned.
- Add `config/schemas/testing.schema.yaml` and register `testing` in the base task enum.
- Update docs (`AGENTS.md`, `PLANS.md`, `README.md`, and `slurm/README.md`) to codify the testing workflow.
- Add a gitignored `slurm/tests/test_secrets.env` and a committed `slurm/tests/test_secrets.env.example`.
- Update `.gitignore` to ignore `slurm/tests/test_secrets.env` (and `slurm/tests/logs/` if needed beyond existing `*.out`/`*.err` ignores).
- Add a unit test that asserts the smoke configs match `config/tests/defaults.yaml` for model/tokenizer/seed values.

### Plan / Milestones

1) Add SLURM test env + secrets template, update `.gitignore`, implement `slurm/tests/run_tests.sh` as a wrapper around `slurm/submit_job.sh`.
2) Add `config/tests/defaults.yaml`, smoke configs, and a unit test that enforces defaults alignment.
3) Update docs and capture a SLURM test run (job ID + log paths).

### Acceptance Criteria

- `slurm/tests/run_tests.sh` submits a SLURM job for unit tests and prints the job ID plus resolved log paths.
- The test runner refuses submission when `whoami` is not in the allowed submitter list configured in `slurm/tests/slurm_test.env`.
- `slurm/tests/run_tests.sh --dry-run` prints the exact `sbatch` command from `slurm/submit_job.sh` and the selected test config.
- A `task: testing` unit-test config runs successfully on `postiguet1` and logs a pytest summary.
- A minimal tokenization end-to-end config runs successfully on `postiguet1` with the Llama-family tokenizer from `config/tests/defaults.yaml`.
- A minimal CLM training end-to-end config runs successfully after tokenization output is created, using the Llama-family model from `config/tests/defaults.yaml`.
- Secrets are read from `slurm/tests/test_secrets.env` when present and are not committed.
- Docs explain when to run unit tests vs integration tests and where test configs live.

### Non-Goals

- Running all integration tests on every change.
- Replacing the existing `slurm/submit_job.sh` workflow for production jobs.

### Notes

Integration test configs should be minimal and only run when relevant to the change (feature or bug fix). Unit tests remain the default for most changes.

### Status / Notes

- SLURM config issue resolved via `SLURM_CONF` fallback; jobs now submit from `slurm/tests/run_tests.sh`.
- Unit run (`./slurm/tests/run_tests.sh --unit`) -> job 34855 failed because pytest was missing (`slurm/tests/logs/tests-34855.out`, `slurm/tests/logs/tests-34855.err`).
- Integration run (`./slurm/tests/run_tests.sh --integration`) -> job 34858 failed due to `parallelization_strategy: none` in `config/tests/clm_training_smoke.yaml` (`slurm/tests/logs/tests-34858.out`, `slurm/tests/logs/tests-34858.err`).
- Installed pytest in the `lmtk` conda env and updated `config/tests/clm_training_smoke.yaml` to `parallelization_strategy: dp`.
- Re-submitted unit job 34862 (pending Resources) and integration job 34864 (pending Priority); logs at `slurm/tests/logs/tests-34862.out`, `slurm/tests/logs/tests-34862.err`, `slurm/tests/logs/tests-34864.out`, `slurm/tests/logs/tests-34864.err`.
- Fixed `select_scheduler` to return a constant scheduler for `lr_scheduler: fixed` and added `tests/unit/training/test_scheduler.py` (passes locally in `lmtk` env).
- Fixed publish/version unit test expectations and revalidated locally (20 passed).
- Integration job 34881 completed successfully; logs at `slurm/tests/logs/tests-34881.out`, `slurm/tests/logs/tests-34881.err`.
- Unit job 34883 completed successfully (30 passed); logs at `slurm/tests/logs/tests-34883.out`, `slurm/tests/logs/tests-34883.err`.