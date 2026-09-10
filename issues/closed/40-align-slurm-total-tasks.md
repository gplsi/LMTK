---
number: 40
title: "Align SLURM total tasks with nodes and tasks-per-node"
state: closed
labels:
- maintenance
- docs
---

## Summary
Make SLURM total tasks explicit as `nodes * ntasks-per-node` in submission output and documentation to prevent warnings about unused nodes.

## Background / Current State
- `slurm/submit_job.sh` submits multi-node jobs using `--nodes` and `--ntasks-per-node`.
- SLURM warns when `--nodes` exceeds total tasks (`--ntasks`), leading to dropped nodes.
- `slurm/MULTINODE_TUTORIAL.md` documents `--nodes` and `--ntasks-per-node` but does not call out the derived total.

## Problem
Users can see warnings like "requested 3 nodes but only 2 tasks" when total tasks do not align with the node allocation.

## Goals
- Make total tasks explicit and consistent with `nodes * ntasks-per-node`.
- Surface the derived total in submission output and documentation.
- Keep single-node behavior unchanged.

## Non-goals
- No changes to training orchestration or distributed launch logic.
- No new CLI flag for total tasks; rely on existing `--nodes` and `--ntasks-per-node`.

## Prior Art / References
- Slurm sbatch documentation: https://slurm.schedmd.com/sbatch.html (ntasks and ntasks-per-node semantics)
- Multi-node guide: `slurm/MULTINODE_TUTORIAL.md`
- Submission helper: `slurm/submit_job.sh`

## Approach
- Validate that `--nodes` and `--ntasks-per-node` are positive integers in `slurm/submit_job.sh`.
- Compute `total_tasks = nodes * ntasks-per-node` and pass `--ntasks` to sbatch for alignment.
- Add a note to `slurm/MULTINODE_TUTORIAL.md` explaining the derived total tasks.

## Acceptance Criteria
- sbatch command includes `--ntasks` equal to `nodes * ntasks-per-node`.
- Submission summary displays total tasks alongside nodes and tasks-per-node.
- Multi-node tutorial mentions the derived total task count.
