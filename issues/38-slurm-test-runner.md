---
number: 38
title: "Add SLURM-based test runner and test configs"
state: open
labels:
- enhancement
- testing
- infra
---

### Description

Local test execution is incomplete because key dependencies and GPUs live inside the SLURM runtime. We need a consistent, auditable SLURM-based test runner and a set of minimal YAML test configs so every change can be validated on the cluster.

### Motivation

- Ensure every change can run tests even when local deps are missing.
- Make test runs reproducible and auditable (job IDs, logs, exact commands).
- Standardize on small Llama-family models and minimal datasets for fast feedback.

### Proposed Solution

- Add `slurm/tests/slurm_test.env` with safe test defaults (partition `postiguet1`, 1x RTX 4090, short time limits).
- Add `slurm/tests/run_tests.sh` that:
  - sources `slurm/tests/slurm_test.env`,
  - optionally sources `slurm/tests/test_secrets.env`,
  - enforces an allowed submitter list,
  - submits a test job to SLURM and prints job ID + log paths.
- Extend `slurm/p.slurm` with a `RUN_MODE=test` path that runs a provided test command.
- Add `config/tests/` with minimal YAML configs for end-to-end testing (tokenization + training).
- Define shared test defaults (including model IDs) in `config/tests/defaults.yaml`.
- Update docs (`AGENTS.md`, `PLANS.md`, `README.md`, and `slurm/README.md`) to codify the testing workflow.
- Add a gitignored `slurm/tests/test_secrets.env` and a committed `slurm/tests/test_secrets.env.example`.

### Acceptance Criteria

- `slurm/run_tests.sh` submits a SLURM job for unit tests and prints job ID + log paths.
- The test runner refuses submission when `whoami` is not on the allowed submitter list.
- A minimal tokenization end-to-end config runs successfully on `postiguet1` with a small Llama tokenizer.
- A minimal CLM training end-to-end config runs successfully using a small Llama model.
- Secrets are read from `slurm/test_secrets.env` when present and are not committed.
- Docs explain when to run unit tests vs integration tests and where test configs live.

### Non-Goals

- Running all integration tests on every change.
- Replacing the existing `slurm/submit_job.sh` workflow for production jobs.

### Notes

Integration test configs should be minimal and only run when relevant to the change (feature or bug fix). Unit tests remain the default for most changes.
