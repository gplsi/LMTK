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
- The training loop in `src/tasks/training/fabric/trainer/base.py` is primarily epoch-oriented (`validations_per_epoch`, `checkpoints_per_epoch`, end-of-epoch/end-of-run), but it also consumes `validate_after_k_steps` for additional global-step validation triggers. This issue must preserve that existing behavior.

## Terminology (definitions for v1)

- **Packed block**: one training sample of exactly `sequence_length` tokens produced by issue 42 packing (fixed shape `[sequence_length]`).
- **Training batch**: a set of packed blocks (`batch_size` per rank; global optimizer-step batch also depends on `world_size` and `gradient_accumulation_steps`).
- **Homogeneous block**: a packed block attributable to exactly one dataset source (no token-level mixing within a block).
- **Interleaved mixing**: the training stream alternates blocks from different datasets over time to approximate a target ratio (e.g. A:A:B repeating/shuffled), rather than training in sequential phases.
- **Token budget**: the desired allocation of training tokens per dataset. In v1, this is implemented via packed blocks:
  - `tokens ~= blocks * sequence_length` (exact enough for fixed-length blocks).
- **`explicit_blocks` budget mode**: run length is set directly by `dataset.mixture.total_blocks`.
- **`anchor_epochs` budget mode**: run length is derived from an anchor dataset and epoch count.
- **Anchor dataset**: the source whose packed-block count defines one epoch in `anchor_epochs` mode. Default anchor is the source with the largest packed-block count unless `dataset.mixture.anchor_dataset_id` is provided.
- **Virtual mixture epoch**: in `anchor_epochs` mode, an accounting unit used only to derive `requested_total_blocks`; it is not the trainer's `number_epochs`.

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

5) **Two budget interfaces, one executable budget**
   - v1 supports:
     - `anchor_epochs` (default): derive budget from anchor source + epochs,
     - `explicit_blocks`: direct block budget via `total_blocks`.
   - Both modes must resolve to one concrete executable budget: `effective_total_blocks`.

6) **Effective-budget invariants are explicit**
   - In distributed runs we fail fast unless the resolved requested budget is compatible with world-size sharding and optimizer-step settings.
   - Rationale: ratio accounting is only meaningful when executed blocks exactly match defined budgets.

7) **Validation is block-based and per-source in v1**
   - Validation uses the same packing path (fixed-length blocks) as training.
   - Validation is evaluated independently per source; we do not build a mixed/proportion-sampled validation stream in v1.
   - A single comparable scalar is reported as a weighted aggregate over per-source losses.
   - Rationale: this keeps validation auditable and simple while preserving source-level visibility for replay/forgetting.

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
- Hashing must be stable across processes/Python runs:
  - do **not** use Python built-in `hash(...)` for schedule decisions,
  - v1 uses `hashlib.blake2b(..., digest_size=8)` over UTF-8 bytes of `"schedule_seed|global_idx|dataset_id"` and interprets the digest as unsigned 64-bit integer in big-endian order.
- Deterministic remainder/tie handling for exact allocation must be explicit and stable:
  - for equal remainders, break ties by lexicographic `dataset_id`.

### Contract C: No distributed hangs

- If rank 0 fails while building any required artifact (packing index or mixture artifacts), all ranks must fail fast (no barrier hangs).

### Contract D: Accounting must match reality

- Every emitted sample carries a tensor-safe source identity (`dataset_idx: int64`), with a deterministic `dataset_idx <-> dataset_id` mapping persisted in report/checkpoint metadata.
- At end-of-run, rank 0 writes a report to disk including:
  - requested weights,
  - deterministic target block allocation per dataset,
  - realized blocks/tokens per dataset,
  - realized ratios and deviation.

### Contract E: Effective budget under distributed execution

- The implementation must resolve and persist:
  - `requested_total_blocks`,
  - `effective_total_blocks` (the exact executed block budget after applying distributed constraints).
- Budget resolution rules:
  - `explicit_blocks`: `requested_total_blocks = dataset.mixture.total_blocks`.
  - `anchor_epochs`: build/load per-source **train-split** packing index lengths `B_i`; choose anchor:
    - `dataset.mixture.anchor_dataset_id` when provided,
    - otherwise the source with max `B_i`.
    Then derive per-epoch target allocation from weights with anchor constraint (`target_anchor_per_epoch = B_anchor`) and exact deterministic allocation, and set:
    - `requested_total_blocks = anchor_epochs * total_blocks_per_epoch`.
- Exact deterministic allocation algorithm in v1:
  - Global default allocator: Hamilton / largest-remainder:
    - `raw_i = total * weight_i / sum(weights)`,
    - `base_i = floor(raw_i)`,
    - distribute remaining blocks to largest fractional remainders,
    - tie-break by lexicographic `dataset_id`.
  - `anchor_epochs` per-epoch allocator (canonical, anchor-fixed):
    - `target_anchor_per_epoch = B_anchor`,
    - for each non-anchor source `j`: `raw_j = B_anchor * weight_j / weight_anchor`,
    - `total_non_anchor = floor(sum(raw_j) + 0.5)` (explicit half-up),
    - allocate non-anchor integer counts with Hamilton over `{raw_j}` constrained to `total_non_anchor`,
    - `total_blocks_per_epoch = target_anchor_per_epoch + sum(target_non_anchor_j)`.
- In mixture mode, `number_epochs` must be exactly `1`.
  - Rationale: run length is controlled only by mixture budget (`anchor_epochs` or `explicit_blocks`) to avoid accidental double-scaling.
- `anchor_epochs` does not control trainer epochs. It only defines `requested_total_blocks`; trainer epoch count remains fixed by the mixture contract (`number_epochs == 1`).
- v1 policy is strict budget execution:
  - `effective_total_blocks` must equal `requested_total_blocks`.
- It must fail fast on incompatible configurations rather than silently padding/duplicating/dropping in ways that invalidate mixture accounting.
- Required minimum checks in v1:
  - `requested_total_blocks % world_size == 0` in distributed runs.
  - per-rank block count is compatible with `batch_size` and `gradient_accumulation_steps` (no silent lost optimizer steps).

### Contract F: Validation construction is per-source and deterministic

- For mixture mode, validation is constructed independently per source when a source lacks a `valid` split.
- Auto-validation splitting must happen before mixing and must never split across sources.
- Reuse existing validation split behavior from `FabricTrainerBase._ensure_validation_split` (applied per source), not a parallel custom splitter.
- Split parameters come from existing `validation_split` settings and must be deterministic from seed + source identity.
- If a source has neither:
  - a provided `valid` split, nor
  - a usable `validation_split` configuration,
  the run must fail fast with a clear error.
- Deterministic source-specific split seed must be explicit:
  - derive `split_seed_i` from `blake2b_u64("valsplit|base_seed|dataset_id") % 2147483647` (range `0..2147483646`) for `datasets.train_test_split`.
- Validation execution contract in v1:
  - validation data is block-based (same `sequence_length` packing semantics as training),
  - report `metric/val_loss_<dataset_id>` for each source,
  - report `metric/val_loss_weighted` using normalized training weights over per-source losses:
    - `val_loss_weighted = sum_i(normalized_weight_i * val_loss_i)`.
- Validation dataloader policy in v1:
  - use non-dropping validation behavior (`sampler_drop_last=false`, `drop_last_batch=false`) to avoid empty/biased validation by construction.
- If any source validation dataloader has zero batches after setup, fail fast with the offending `dataset_id`.
- v1 must not introduce a sampled mixed validation stream or per-batch validation ratio constraints.
- v1 reporting must include whether each source used:
  - an existing `valid` split, or
  - an auto-generated per-source validation split.

### Contract G: Resume compatibility for budget/accounting integrity

- In mixture mode, resume must fail fast unless these are unchanged from checkpoint metadata:
  - source set and weights,
  - `budget_mode`, `anchor_dataset_id`, `requested_total_blocks`,
  - `world_size`, `batch_size`, `gradient_accumulation_steps`,
  - hashing/allocation algorithm identifiers.
- Rationale: changing these invalidates deterministic accounting guarantees.

## Non-goals

- Do not implement mixed-source blocks (token-level mixing inside a block).
- Do not implement dynamic padding.
- Do not redesign task dispatch or SLURM submission logic.
- Do not implement “exact uniqueness” guarantees (replay implies repetition is allowed and expected).
- Do not implement proportional sampled mixed validation streams in v1.

## Proposed Approach (high level, exact semantics)

### A) Per-dataset packed-block datasets

For each configured dataset `Di`:

- Load the tokenized-docs HF dataset.
- Wrap it with the same packing logic from issue 42 to expose fixed-length blocks.
- Ensure it has a stable `__len__` (blocks available per “cycle”).

### B) Mixture dataset (map-style) that composes per-dataset block datasets

Create a map-style dataset (e.g. `MixturePackedDataset`) that:

- Exposes a global `__len__` defined by resolved `effective_total_blocks`.
- Supports two budget interfaces:
  - `anchor_epochs` (default):
    - load per-source **train-split** block capacities `B_i` from packing indexes,
    - choose anchor (`anchor_dataset_id` or `argmax(B_i)`),
    - define one epoch as full anchor pass in blocks (`target_anchor_per_epoch = B_anchor`),
    - derive non-anchor per-epoch targets with canonical anchor-fixed algorithm (Contract E),
    - derive `requested_total_blocks = anchor_epochs * total_blocks_per_epoch`.
  - `explicit_blocks`:
    - use configured `total_blocks` as `requested_total_blocks`.
- Maps each global block index `i` to:
  - `dataset_idx`/`dataset_id` using an exact-counts interleaving policy (weights → exact blocks-per-dataset over the whole run),
  - `local_block_idx` using deterministic sampling with replacement from a stable hash:
    - `local_block_idx = blake2b_u64(f"{schedule_seed}|{i}|{dataset_id}") % num_blocks_in_dataset`.

The key property: we mix at the **block selection** level. Each block is still produced by the per-dataset packer and remains homogeneous.

The mixing schedule must satisfy:
- Exact global counts: for resolved `requested_total_blocks`, allocate exact `blocks_per_dataset` from weights with deterministic remainder handling.
- Order randomization: interleave sources by applying a deterministic permutation to `i` before dataset selection (to avoid long runs of one source).

### C) Dataloader + sampler policy

- The mixture dataset is map-style so it works with explicit `DistributedSampler` (same policy as issue 42).
- Sampling/shuffling occurs at the block level; the mixing schedule can be:
  - fixed per epoch, with optional shuffling of blocks (controlled and deterministic).

### D) Validation strategy (v1)

- Build/evaluate validation per source, not from a mixed stream.
- Validation uses packed blocks with the same `sequence_length` contract as training.
- Emit:
  - per-source `val_loss_<dataset_id>`,
  - weighted aggregate `val_loss_weighted` from configured training weights.
- Keep v1 simple:
  - no separate validation ratio scheduler,
  - no sampled mixed validation stream.
- Fail fast if any source produces zero validation batches after dataloader setup.

### E) Accounting & persistence

- Each sample carries `dataset_idx` (int tensor), and rank-0 metadata maps `dataset_idx -> dataset_id` for reporting/counters.
- Maintain counters per rank; reduce across ranks at safe synchronization points (end of epoch and end of run).
- Persist `mixture_report.json` into `output_dir` on rank 0 (or the explicit `dataset.mixture.report_path` when provided).
- Persist budget resolution fields so runs are auditable:
  - `budget_mode`, `anchor_dataset_id`, `source_blocks_available`, `requested_total_blocks`, `effective_total_blocks`.
- Persist validation provenance per source:
  - existing-valid vs auto-generated-valid, and split parameters used.
- Persist validation metrics:
  - `val_loss_per_dataset`,
  - `val_loss_weighted`.

### F) Reference snippets (normative for v1 behavior)

Use these snippets as the exact algorithmic reference during implementation.

    # 1) Stable hash used for schedule and local source block sampling
    def blake2b_u64(text: str) -> int:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)

    # 2) Exact deterministic allocation (Hamilton / largest remainder)
    def allocate_exact_counts(total: int, weights_by_id: dict[str, float]) -> dict[str, int]:
        ids = sorted(weights_by_id)  # deterministic tie-break by dataset_id
        z = sum(weights_by_id[i] for i in ids)
        raw = {i: total * weights_by_id[i] / z for i in ids}
        base = {i: int(math.floor(raw[i])) for i in ids}
        rem = total - sum(base.values())
        # largest remainder first; lexicographic dataset_id ascending on ties
        order = sorted(ids, key=lambda i: (-(raw[i] - base[i]), i))
        for i in order[:rem]:
            base[i] += 1
        return base

    # 3) Anchor-fixed per-epoch allocation for budget_mode=anchor_epochs
    def derive_anchor_epoch_targets(
        source_blocks: dict[str, int],
        weights_by_id: dict[str, float],
        anchor_id: str,
    ) -> dict[str, int]:
        b_anchor = source_blocks[anchor_id]
        w_anchor = weights_by_id[anchor_id]
        non_anchor = {k: v for k, v in weights_by_id.items() if k != anchor_id}
        raw_non_anchor = {k: b_anchor * non_anchor[k] / w_anchor for k in non_anchor}
        total_non_anchor = int(math.floor(sum(raw_non_anchor.values()) + 0.5))  # half-up
        targets_non_anchor = allocate_exact_counts(total_non_anchor, raw_non_anchor)
        return {anchor_id: b_anchor, **targets_non_anchor}

    # 4) Weighted validation metric
    val_loss_weighted = sum(normalized_weight[i] * val_loss_per_dataset[i] for i in dataset_ids)

    # 5) Deterministic per-source validation split seed
    split_seed_i = blake2b_u64(f"valsplit|{base_seed}|{dataset_id}") % 2147483647

    # 6) Deterministic permutation of [0, total) without materializing an O(total) array
    # (used before cumulative lookup to avoid long source runs while keeping exact counts)
    def permute_index(i: int, total: int, schedule_seed: int) -> int:
        if total <= 1:
            return 0
        stride = (blake2b_u64(f"perm_stride|{schedule_seed}") % total) or 1
        while math.gcd(stride, total) != 1:
            stride = (stride + 1) % total or 1
        offset = blake2b_u64(f"perm_offset|{schedule_seed}") % total
        return (i * stride + offset) % total

    # 7) O(log S) dataset-id lookup without materializing full schedule
    # counts_by_id: exact per-source counts for the epoch or run (sum == total)
    ordered_ids = sorted(counts_by_id)
    cumulative = []
    running = 0
    for did in ordered_ids:
        running += counts_by_id[did]
        cumulative.append((running, did))
    # map permutation index p = permute_index(i, total, schedule_seed) in [0, total)
    # did = first dataset with p < upper_bound using binary search on cumulative

    # 8) Resume guard metadata (must match at resume time)
    mixture_meta = {
        "sources": [{"dataset_id": did, "weight": weights_by_id[did]} for did in sorted(weights_by_id)],
        "budget_mode": budget_mode,
        "anchor_dataset_id": anchor_id,
        "requested_total_blocks": requested_total_blocks,
        "world_size": world_size,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "hash_algorithm": "blake2b_u64_v1",
        "allocation_algorithm": "hamilton_lr_lexicographic_v1",
    }

## Configuration (exact YAML surface for v1)

This issue adds an opt-in surface under `dataset.sources` + `dataset.mixture`. When `dataset.sources` is present, `dataset.nameOrPath` is not used.

Minimal example (two sources A/B, 2:1 mixing, implicit anchor + epoch budget):

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
        budget_mode: anchor_epochs
        anchor_epochs: 1
        # optional; defaults to source with max blocks
        anchor_dataset_id: null
        # optional; defaults in code to `seed`
        schedule_seed: null

    validation_split:
      proportion: 0.1
      shuffle: true
      seed: 42

    # hard requirement in mixture mode: number_epochs must be 1
    number_epochs: 1
    batch_size: 1
    num_workers: 0
    validations_per_epoch: 5
    output_dir: output/mixture_smoke

Notes:
- `anchor_epochs` defines run length relative to anchor capacity.
- In `anchor_epochs`, anchor capacity means train-split packed blocks after per-source validation split construction.
- For exact direct budgets, set:
  - `budget_mode: explicit_blocks`
  - `total_blocks: <N>`
- Replay is expressed as just another source entry with a weight (e.g., `dataset_id: R`).
- In distributed runs, the config must satisfy Contract E divisibility invariants; invalid combinations fail fast with clear errors.
- Validation in v1 is per-source auto-generated when source `valid` is absent (Contract F), using the existing `validation_split` config deterministically.
- Validation cadence remains the current trainer cadence: with mixture mode (`number_epochs: 1`), `validations_per_epoch` and `checkpoints_per_epoch` are distributed over the full resolved block budget.
- In mixture mode, `number_epochs != 1` is invalid and must fail fast.

## Acceptance Criteria

- A smoke config trains for a short run and persists deterministic target vs realized allocation in `mixture_report.json`.
- Unit tests cover:
  - schedule determinism given seed/epoch,
  - exact target allocation over resolved budget (including deterministic remainder handling),
  - stable-hash schedule behavior (no dependency on Python `hash()`),
  - exact Hamilton allocation with lexicographic tie-break on equal remainders,
  - anchor resolution (`anchor_dataset_id` override and default largest-source anchor),
  - budget derivation correctness for `anchor_epochs` and `explicit_blocks`,
  - repeat/shuffle behavior when a dataset has fewer than one epoch’s worth of blocks,
  - DDP-safe policy enforcement (no silent duplication and explicit fail-fast on incompatible budgets).
- Validation tests cover:
  - per-source deterministic auto-split creation,
  - no cross-source leakage in validation construction,
  - fail-fast when a source has neither `valid` nor usable `validation_split`,
  - fail-fast when a source validation dataloader yields zero batches,
  - per-source validation provenance persisted in report,
  - `val_loss_<dataset_id>` and `val_loss_weighted` emitted and persisted,
  - weighted aggregate is computed from per-source losses (not from sampled mixed validation batches).
- Resume safety tests cover:
  - fail-fast when resume metadata mismatches any Contract G field.
- Reproducibility evidence includes:
  - per-source dataset identity recorded in report metadata (`dataset_id`, `nameOrPath`, and dataset fingerprint/revision when available),
  - deterministic split/schedule inputs (`seed`, `schedule_seed`, split-seed algorithm id),
  - smoke metric sanity threshold: `val_loss_weighted` is finite and falls in `(0, 30)` for the test config.
- For a successful run:
  - `effective_total_blocks == requested_total_blocks`
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

   Use the SLURM test defaults from `slurm/tests/slurm_test.env` unless explicitly overridden:
   - `PARTITION=postiguet1`,
   - `GPU_COUNT=1` (default target: 1x RTX 4090),
   - `CPUS_PER_TASK=8`, `MEMORY=32G`, `TIME_LIMIT=02:00:00`.

   The submitter guard must pass before submission:
   - ensure your username is present in `ALLOWED_SUBMITTERS` in `slurm/tests/slurm_test.env`.

4) Fallback submission path (if `run_tests.sh` is unavailable):

    ./slurm/submit_job.sh --config config/tests/clm_training_packing_mixture_multinode_smoke.yaml --partition postiguet1 --gpus 1 --nodes 2 --ntasks-per-node 1 --cpus 8 --memory 32G --time 02:00:00

5) Log retrieval and completion check:

    sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed

   If `run_tests.sh` was used, also collect the printed log paths (`Stdout log`, `Stderr log`).
   If `submit_job.sh` was used directly, collect paths from the emitted `--output` / `--error` values.

Required evidence to record in this issue:
- exact command,
- SLURM job ID,
- stdout/stderr log paths,
- path to produced `mixture_report.json`,
- source reproducibility manifest (`dataset_id`, `nameOrPath`, fingerprint/revision when available),
- short log/report excerpt proving:
  - requested/effective block equality,
  - realized totals/ratios,
  - per-source validation provenance and validation metrics (including `val_loss_weighted` threshold check).

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
