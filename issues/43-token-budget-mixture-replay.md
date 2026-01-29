---
number: 43
title: "Token-budget multi-dataset mixing + replay (homogeneous packed blocks)"
state: open
labels:
  - enhancement
  - training
  - hpc
  - continual-pretraining
  - data
---

## Summary

Add a follow-up to issue 42 that enables continual pretraining with **multiple input datasets** and **token-budget mixing** (e.g. 0.5/0.3/0.2), including **replay**, while keeping online packing blocks **homogeneous per dataset**.

This feature is explicitly designed to layer on top of the packing work in issue 42.

## Problem

For continual pretraining and replay we often want to train on multiple corpora with target proportions such as:

- Dataset A: 0.5
- Dataset B: 0.3
- Replay dataset R: 0.2

Those proportions must be defined and enforced over **tokens (or fixed-length packed blocks)**, not over documents/examples, because example lengths vary drastically.

We also need auditable accounting: “what did we actually train on?” must be reported and persisted (especially for SLURM multi-node runs).

## Key Decisions (locked for v1 of this issue)

1) **Homogeneous packed blocks**
   - Each packed block must be attributable to exactly one dataset (`dataset_id`).
   - We do **not** mix tokens from multiple datasets inside a single packed block.
   - Rationale: makes determinism, accounting, debugging, and reporting straightforward.

2) **Repeat/shuffle exhaustion policy**
   - If a dataset runs out of content to produce more full blocks, we do **repeat/shuffle** (sampling with replacement / circular traversal) rather than “top-off with another dataset” or padding.
   - Rationale: token-budget continual pretraining is naturally a streaming process; per-dataset exhaustion should not break ratio targets.

3) **Token-budget ratios, not per-batch hard constraints**
   - Ratios are enforced over a window (e.g. per epoch or per N steps), not by forcing each individual batch to match exact ratios.
   - Rationale: avoids unnecessary complexity and maintains stochasticity while still achieving the intended expected-gradient mixture.

## Goals

1) **Config: multi-dataset + weights**
   - Training config supports specifying multiple datasets and weights interpreted as **token/blocks ratios**.
   - Include explicit `dataset_id` / `name` fields for reporting.

2) **Deterministic, DDP-safe mixing**
   - Deterministic schedule given the same seed, epoch, world size, and configuration.
   - No silent data duplication across ranks by default (consistent with issue 42 policy).

3) **Auditable accounting**
   - Log and persist realized:
     - blocks-per-dataset,
     - tokens-per-dataset (`blocks * sequence_length`),
     - realized ratios and deviation from targets,
     - optional counters like `eos_inserted_per_dataset` (if relevant).
   - In DDP, aggregate counters across ranks and persist once on rank 0.

4) **Works with issue 42 packing**
   - Each input dataset is expected to be a “tokenized-docs dataset” (variable-length `input_ids` + `length`) produced by issue 42 Milestone 1.
   - Packing stays in training; tokenization remains doc-level.

## Non-goals

- Do not implement mixed-source blocks (token-level mixing inside a block).
- Do not implement dynamic padding.
- Do not redesign task dispatch or SLURM submission logic.
- Do not implement “exact uniqueness” guarantees (replay implies repetition is allowed and expected).

## Proposed Approach (high level)

### A) Per-dataset packed-block datasets

For each configured dataset `Di`:

- Load the tokenized-docs HF dataset.
- Wrap it with the same packing logic from issue 42 to expose fixed-length blocks.
- Ensure it has a stable `__len__` (blocks available per “cycle”).

### B) Mixture dataset (map-style) that composes per-dataset block datasets

Create a map-style dataset (e.g. `MixturePackedDataset`) that:

- Exposes a global `__len__` (either:
  - per-epoch blocks, or
  - a token-budget-derived number of blocks),
- Maps each global block index `i` to:
  - `dataset_id = schedule[i]`,
  - `local_block_idx = f(i, dataset_id, seed, epoch)` with repeat/shuffle semantics.

The schedule should be deterministic and should approximate weights tightly over time (e.g. deficit round-robin / weighted fair scheduling), not purely random draws.

### C) Dataloader + sampler policy

- The mixture dataset is map-style so it works with explicit `DistributedSampler` (same policy as issue 42).
- Sampling/shuffling occurs at the block level; the mixing schedule can be:
  - fixed per epoch, with optional shuffling of blocks (controlled and deterministic).

### D) Accounting & persistence

- Each sample carries `dataset_id` (and optionally `dataset_name`) so we can count per-dataset blocks.
- Maintain counters per rank; reduce across ranks at safe synchronization points (end of epoch and end of run).
- Persist a `mixture_report.json` (or similar) into `output_dir` on rank 0.

## Acceptance Criteria

- A smoke config trains for a short run and logs realized mix within a small tolerance of target ratios.
- Unit tests cover:
  - schedule determinism given seed/epoch,
  - ratio accuracy over N blocks,
  - repeat/shuffle behavior when a dataset has fewer than one epoch’s worth of blocks,
  - DDP-safe `drop_last` policy is enforced (no silent duplication).
- SLURM multi-node smoke run completes with correct accounting persisted on rank 0.

## Integration Points / Invariants (must align with issue 42)

- Build on the issue 42 packing abstraction:
  - packing lives in training as a dataset wrapper (not a collator),
  - packed blocks are fixed-shape,
  - sampler policy defaults avoid duplication.
- Keep mixing logic separate from packing logic:
  - packing produces “blocks from one dataset”,
  - mixing selects which dataset’s blocks to emit next.
