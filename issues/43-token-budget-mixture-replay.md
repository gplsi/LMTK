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

This issue provides the canonical design for multi-source continual pretraining in LMTK. The implementation must be DDP/FSDP-safe under SLURM + Lightning Fabric and must be auditable (exactly what tokens/blocks came from where).

## Problem

For continual pretraining and replay we often want to train on multiple corpora with target proportions such as:

- Dataset A: 0.5
- Dataset B: 0.3
- Replay dataset R: 0.2

Those proportions must be defined and enforced over **tokens (or fixed-length packed blocks)**, not over documents/examples, because example lengths vary drastically.

We also need auditable accounting: “what did we actually train on?” must be reported and persisted (especially for SLURM multi-node runs).

## Background / Current State

- Issue 42 added:
  - doc-level CLM tokenization (`input_ids` variable length + `length` + `ends_with_eos`),
  - training-time online packing via a map-style dataset wrapper (`PackedSequenceDataset`),
  - a persisted memmap packing index (`PackingIndex`) and explicit distributed sampler/drop policies.
- Current training config supports **one** dataset via `dataset.nameOrPath` (schema: `config/schemas/training/components/data.schema.yaml`). There is no schema surface for `dataset.sources` today.
- The training loop in `src/tasks/training/fabric/trainer/base.py` is currently epoch-driven (`validations_per_epoch`, `checkpoints_per_epoch`, end-of-epoch/end-of-run). `validate_after_k_steps` exists in schema but is not currently consumed by this trainer.

## Terminology (definitions for v1)

- **Packed block**: one training sample of exactly `sequence_length` tokens produced by issue 42 packing (fixed shape `[sequence_length]`).
- **Homogeneous block**: a packed block attributable to exactly one dataset source (no token-level mixing within a block).
- **Interleaved mixing**: the training stream alternates blocks from different datasets over time to approximate a target ratio (e.g. A:A:B repeating/shuffled), rather than training in sequential phases.
- **Token budget**: the desired allocation of training tokens per dataset. In v1, this is implemented via packed blocks:
  - `tokens ~= blocks * sequence_length` (exact enough for fixed-length blocks).

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

4) **DDP/FSDP-friendly by construction**
   - Mixing is implemented as a map-style dataset with deterministic `__getitem__` and explicit distributed sampler policy (no stateful collators; no iterable-only mixing).
   - Rationale: correctness under SLURM multi-node + resume is more important than micro-optimizing the input pipeline.

5) **Effective-budget invariants are explicit**
   - In distributed runs we fail fast unless `total_blocks` is compatible with world-size sharding and optimizer-step settings.
   - Rationale: ratio accounting is only meaningful when executed blocks exactly match defined budgets.

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

## Non-negotiable contracts (junior-proof)

These contracts define “correct v1 behavior”. If any cannot be satisfied, the implementation must fail fast with a clear error.

### Contract A: Source compatibility

All sources in a mixture must be compatible:
- Same `sequence_length` for packing (batching requires fixed shape).
- Same tokenizer/vocab space for the target model (token IDs must be meaningful for a single model).
  - v1 requirement: all sources must use the same tokenizer name (or at least the same `eos_token_id` and vocabulary). The implementation must fail fast if sources specify incompatible tokenizer/eos settings.

### Contract B: Deterministic mapping from global index → (dataset_id, local_block_idx)

- Mixing must be a pure function: `MixturePackedDataset.__getitem__(i)` depends only on `i` and config/seed (no mutable worker state).
- The mapping must not require storing a schedule array in RAM proportional to the number of blocks.
- In DDP/FSDP, distributed sharding must be explicit and must not rely on Fabric’s implicit sampler behavior.

### Contract C: No distributed hangs

- If rank 0 fails while building any required artifact (packing index or mixture artifacts), all ranks must fail fast (no barrier hangs).

### Contract D: Accounting must match reality

- Every emitted sample carries `dataset_id` (as a tensor) so per-dataset blocks are countable.
- At end-of-run, rank 0 writes a report to disk including:
  - requested weights,
  - deterministic target block allocation per dataset,
  - realized blocks/tokens per dataset,
  - realized ratios and deviation.

### Contract E: Effective budget under distributed execution

- The implementation must define and persist `effective_total_blocks` (the exact executed block budget after applying distributed constraints).
- It must fail fast on incompatible configurations rather than silently padding/duplicating/dropping in ways that invalidate mixture accounting.
- Required minimum checks in v1:
  - `total_blocks % world_size == 0` in distributed runs.
  - per-rank block count is compatible with `batch_size` and `gradient_accumulation_steps` (no silent lost optimizer steps).

## Non-goals

- Do not implement mixed-source blocks (token-level mixing inside a block).
- Do not implement dynamic padding.
- Do not redesign task dispatch or SLURM submission logic.
- Do not implement “exact uniqueness” guarantees (replay implies repetition is allowed and expected).

## Proposed Approach (high level, exact semantics)

### A) Per-dataset packed-block datasets

For each configured dataset `Di`:

- Load the tokenized-docs HF dataset.
- Wrap it with the same packing logic from issue 42 to expose fixed-length blocks.
- Ensure it has a stable `__len__` (blocks available per “cycle”).

### B) Mixture dataset (map-style) that composes per-dataset block datasets

Create a map-style dataset (e.g. `MixturePackedDataset`) that:

- Exposes a global `__len__` defined by **a fixed total number of blocks** (`dataset.mixture.total_blocks`).
  - v1 recommendation: set `number_epochs: 1` for mixture runs and set `total_blocks` to the desired run length. Use `validations_per_epoch` / `checkpoints_per_epoch` for cadence.
- Maps each global block index `i` to:
  - `dataset_id` using an exact-counts interleaving policy (weights → exact blocks-per-dataset over the whole run),
  - `local_block_idx` using deterministic sampling with replacement: `local_block_idx = hash(seed, i, dataset_id) % num_blocks_in_dataset`.

The key property: we mix at the **block selection** level. Each block is still produced by the per-dataset packer and remains homogeneous.

The mixing schedule must satisfy:
- Exact global counts: for `total_blocks`, allocate exact `blocks_per_dataset` from weights with deterministic remainder handling.
- Order randomization: interleave sources by applying a deterministic permutation to `i` before dataset selection (to avoid long runs of one source).

### C) Dataloader + sampler policy

- The mixture dataset is map-style so it works with explicit `DistributedSampler` (same policy as issue 42).
- Sampling/shuffling occurs at the block level; the mixing schedule can be:
  - fixed per epoch, with optional shuffling of blocks (controlled and deterministic).

### D) Accounting & persistence

- Each sample carries `dataset_id` (and optionally `dataset_name`) so we can count per-dataset blocks.
- Maintain counters per rank; reduce across ranks at safe synchronization points (end of epoch and end of run).
- Persist a `mixture_report.json` (or similar) into `output_dir` on rank 0.

## Configuration (exact YAML surface for v1)

This issue adds an opt-in surface under `dataset.sources` + `dataset.mixture`. When `dataset.sources` is present, `dataset.nameOrPath` is not used.

Minimal example (two sources A/B, 2:1 mixing, one-epoch run):

    task: clm_training
    experiment_name: clm_training_mixture_smoke
    verbose_level: 2
    model_name: BSC-LT/salamandra-2b
    precision: bf16-true
    seed: 42

    dataset:
      source: local
      format: hf
      sources:
        - dataset_id: A
          nameOrPath: /shared/tokenized/dsA_doclevel_salamandra2b
          weight: 2
          tokenizer_name: BSC-LT/salamandra-2b
        - dataset_id: B
          nameOrPath: /shared/tokenized/dsB_doclevel_salamandra2b
          weight: 1
          tokenizer_name: BSC-LT/salamandra-2b
      packing:
        enabled: true
        sequence_length: 2048
        insert_eos: true
        tokenizer_name: BSC-LT/salamandra-2b
      mixture:
        enabled: true
        total_blocks: 10000
        # optional; defaults in code to `seed`
        schedule_seed: null

    number_epochs: 1
    batch_size: 1
    num_workers: 0
    validations_per_epoch: 5
    output_dir: output/mixture_smoke

Notes:
- `total_blocks` controls run length. Approx tokens trained: `total_blocks * sequence_length`.
- Replay is expressed as just another source entry with a weight (e.g., `dataset_id: R`).
- In distributed runs, the config must satisfy Contract E divisibility invariants; invalid combinations fail fast with clear errors.

## Acceptance Criteria

- A smoke config trains for a short run and persists deterministic target vs realized allocation in `mixture_report.json`.
- Unit tests cover:
  - schedule determinism given seed/epoch,
  - exact target allocation over `total_blocks` (including deterministic remainder handling),
  - repeat/shuffle behavior when a dataset has fewer than one epoch’s worth of blocks,
  - DDP-safe policy enforcement (no silent duplication and explicit fail-fast on incompatible budgets).
- For a successful run:
  - `sum(realized_blocks_per_dataset) == effective_total_blocks`
  - `sum(realized_tokens_per_dataset) == effective_total_blocks * sequence_length`
  - per-dataset deviation is exactly explained by deterministic allocation and documented rounding/distributed constraints.
- SLURM multi-node smoke run completes with correct accounting persisted on rank 0.

## Validation Runbook (required evidence)

1) Local schema validation:

    python src/main.py --validate --config config/tests/clm_training_packing_mixture_smoke.yaml

2) Local targeted unit tests:

    python3 -m pytest -q tests/unit/training/test_mixture_packing.py

3) SLURM smoke:

    ./slurm/tests/run_tests.sh --config config/tests/clm_training_packing_mixture_multinode_smoke.yaml --nodes 2 --ntasks-per-node 1

Required evidence to record in this issue:
- exact command,
- SLURM job ID,
- stdout/stderr log paths,
- path to produced `mixture_report.json`,
- short log/report excerpt proving totals and realized ratios.

## Integration Points / Invariants (must align with issue 42)

- Build on the issue 42 packing abstraction:
  - packing lives in training as a dataset wrapper (not a collator),
  - packed blocks are fixed-shape,
  - sampler policy defaults avoid duplication.
- Keep mixing logic separate from packing logic:
  - packing produces “blocks from one dataset”,
  - mixing selects which dataset’s blocks to emit next.

## ExecPlan

- `issues/execPlans/43-token-budget-mixture-replay.execplan.md`
