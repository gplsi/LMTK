---
number: 46
title: "Fail fast on SLURM submit when Conda initialization is broken"
state: closed
labels:
- slurm
- conda
- maintenance
---

## Summary
Make `slurm/submit_job.sh` refuse to call `sbatch` when `scripts/set_environment.sh` cannot initialize Conda, and surface the exact bad Conda path or environment name to the user.

## Background / Current State
- `slurm/p.slurm` owns runtime environment setup and sources `scripts/set_environment.sh` on the compute node.
- `scripts/set_environment.sh` previously returned a generic error when the hardcoded `conda.sh` path was missing.
- `slurm/submit_job.sh` validated the config path but did not preflight Conda initialization before submitting the job.

## Problem
Users could successfully submit a SLURM job that was guaranteed to fail on the compute node because the Conda bootstrap path or environment name was wrong, and the submit-time feedback did not identify the exact bad path.

## Goals
- Fail before `sbatch` when Conda initialization is broken.
- Print the exact missing `conda.sh` path and the failing environment name.
- Keep environment ownership in `scripts/set_environment.sh` and submit-time orchestration in `slurm/submit_job.sh`.

## Non-goals
- No fallback to a different Conda installation.
- No redesign of the SLURM launch flow or environment model.

## Prior Art / References
- `scripts/set_environment.sh`
- `slurm/submit_job.sh`
- `slurm/p.slurm`
- `issues/closed/32-conda.md`

## Approach
- Extend `scripts/set_environment.sh` with explicit `CONDA_SH_PATH` and `CONDA_ENV_NAME` inputs and fail with concrete messages.
- Add a submission preflight in `slurm/submit_job.sh` that sources `scripts/set_environment.sh` in a subshell and aborts before `sbatch` on failure.
- Add a focused pytest covering the fail-fast path and a successful preflight with fake `conda` and `sbatch` shims.
- Document the override variables and the new failure mode in `slurm/README.md`.

## Acceptance Criteria
- `slurm/submit_job.sh` exits non-zero before `sbatch` when `scripts/set_environment.sh` cannot source `conda.sh`.
- The submit-time error includes the exact missing `conda.sh` path or activation failure from `scripts/set_environment.sh`.
- Users can override the Conda bootstrap path or env name via `CONDA_SH_PATH` and `CONDA_ENV_NAME`.
- Automated tests cover both the preflight failure and success paths.
