---
number: 42
title: "Online packing (training-time packaging) for CLM via variable-length tokenization"
state: open
labels:
  - enhancement
  - training
  - tokenization
  - hpc
---

## Summary

Add a first-class “online packing” path for continual pretraining (CLM) where tokenization can output variable-length token sequences (no fixed length / no padding) and training performs the actual packaging (packing) into fixed-length blocks suitable for the model’s context window. The implementation must be safe, deterministic, resumable, and correct under multi-node SLURM + Lightning Fabric.

## Background / Current State

- CLM tokenization currently produces fixed-length windows using `return_overflowing_tokens=True` and `padding="max_length"` with a required `context_length` plus `overlap` (stride). This creates redundant tokens when `overlap > 0`. See `src/tasks/tokenization/tokenizer/causal.py`.
- Training (`clm_training`) loads a tokenized dataset from disk and assumes rectangular tensors. Dataloaders are created with default PyTorch collation (no `collate_fn`) and the Fabric pipeline expects columns `input_ids`, `attention_mask`, `labels`. See `src/tasks/training/fabric/trainer/base.py`.
- Multi-node support relies on SLURM launches (`srun`) and Lightning Fabric’s SLURM environment integration (see issue 37). The data pipeline must not silently duplicate data across ranks/nodes.
- In DDP-style distributed training, correct shuffling requires calling `DistributedSampler.set_epoch(epoch)` each epoch; we will make this explicit (at least for packing mode) as part of the data pipeline work.
- For packing, distributed samplers must be constructed after Fabric has initialized rank/world size (inside the Fabric pipeline) so the behavior is correct both for SLURM external launches and for local Fabric-spawned runs.

## Terminology (avoid confusion)

- **Dynamic padding**: padding each batch to that batch’s maximum sequence length (variable batch shapes). This issue does **not** implement dynamic padding.
- **Overlap / stride**: sliding-window tokenization that repeats tokens across consecutive windows (the current CLM tokenization path).
- **(Offline) packaging**: producing fixed-length training blocks ahead of time (what we do today).
- **Online packing** (this issue): tokenization produces variable-length doc-level `input_ids`; training packs those into fixed-length blocks (no per-batch padding; every sample has length `sequence_length`).

## Problem

We want to move away from offline “padding + packaging” for CLM because:

- It makes experimentation with token budgets and dataset replay/mixtures harder (data is already baked into fixed blocks).
- It can waste compute via overlap-based sliding windows (e.g., `context_length=8192`, `overlap=2048` repeats ~33% tokens for long docs).
- We want packaging logic to live with training so it can be shared, tested, and later extended (e.g., token-budget training, replay, multi-source mixing).

## Goals

1. **Tokenization: variable-length mode**
   - For CLM tokenization, when no size is specified (`context_length`/`max_sequence_length` omitted), tokenize each document without fixed-length padding/truncation and output variable-length sequences.
   - Enforce explicit validation: overlap/stride is invalid in variable-length mode (fail fast if `overlap` is set to a non-zero value).
   - Output should include a `length` field (token count) to support downstream packing length accounting and packing dataset sizing.

2. **Training: online packing**
   - Add an opt-in training configuration (`dataset.packing`) that packs variable-length token sequences into fixed-length blocks at training time.
   - Packing must:
     - Produce fixed-length blocks (`sequence_length`) with `attention_mask` all ones and `labels == input_ids` (CLM).
     - Insert an EOS token between documents by default (configurable).
     - Avoid double-EOS: if a document already ends with EOS, do not insert an extra EOS.
     - Be deterministic and safe for multi-node/multi-process SLURM runs (no silent data duplication across ranks).
     - Work with existing Fabric strategies (FSDP/DDP/DeepSpeed) and current training loop (resume/checkpointing).
     - Avoid distributed hangs by ensuring every rank processes the same number of batches; prefer dropping tail blocks over repeating samples to “pad” ranks.

3. **Testing and HPC validation**
   - Add focused unit tests for packing correctness and sharding behavior.
   - Add YAML-driven smoke configs under `config/tests/` for:
     - variable-length CLM tokenization,
     - CLM training with online packing enabled.
   - Ensure the feature can be validated via the SLURM test runner (`slurm/tests/run_tests.sh`) and record job ID/log paths in this issue when executed.

## Non-goals

- Do not implement dynamic padding for CLM batches (this feature is packing, not padding).
- Do not redesign model architecture, attention kernels, or introduce new job submission logic in Python.
- Do not implement full token-budget scheduling or multi-dataset mixing in this change (but design must leave clean extension points).
- Do not do a repo-wide “training config simplification” refactor (defaults/inheritance/presets). We will only improve ergonomics in the new `dataset.packing` surface for this issue.

## Prior Art / References

- CLM tokenization sliding window + overlap: `src/tasks/tokenization/tokenizer/causal.py`.
- Tokenization orchestration and config handling: `src/tasks/tokenization/orchestrator.py`.
- Training dataloaders and training loop: `src/tasks/training/fabric/trainer/base.py`.
- Multi-node launch requirements and validation patterns: `issues/37-add-multi-node-support.md`.
- Existing SLURM test runner patterns: `slurm/tests/run_tests.sh`, `config/tests/integration_smoke.yaml`.

## Proposed Approach (High-Level)

- Extend CLM tokenization to support a “document tokenization” mode when no fixed length is provided. In this mode, the tokenizer outputs variable-length `input_ids` and a `length` field (token count).
- Add a training-time packer abstraction that consumes the tokenized-doc dataset and yields fixed-length blocks, with deterministic sharding across distributed ranks and dataloader workers.
- Add config + schema support so users can opt into packing without impacting existing workflows.

## Config ergonomics (in-scope vs out-of-scope)

In-scope for this issue:

- Make the new `dataset.packing` configuration **easy to use** by providing safe code defaults for optional packing parameters so a minimal packing config can be short.
- Keep smoke/test configs explicit and verbose so behavior is unambiguous and reproducible.

Out-of-scope for this issue (follow-up candidate):

- “Simplify training configs” globally by relying on schema `default:` values or adding presets/includes. Today `ConfigValidator` validates but does **not** apply JSON-schema defaults, so making fields optional at schema-level without a larger config-system change is risky.

## Decisions (Initial)

- Online packing is opt-in via training config (no behavior change for existing pipelines).
- Variable-length CLM tokenization is opt-in via omission of `context_length` / `max_sequence_length` (and requires `overlap=0`/unset).
- Packing inserts EOS between docs by default to preserve document boundaries.
- Packing output is strictly fixed-shape blocks; no padding masks.
- For v1, tail tokens that cannot fill a full block are dropped (no EOS-padding of partial blocks). Token-budget training can revisit “no tokens dropped” semantics later.
- For distributed runs, we will prefer **dropping remainder** over **duplicating samples** to avoid silent data duplication across ranks (standard trade-off: correctness vs full-coverage per epoch).
- While implementing packing, we will also tighten two training-loop invariants that matter for HPC correctness:
  - epoch-level sampler reseeding (`set_epoch`) for any distributed sampler used by packing (and ideally for all DDP runs),
  - scheduler sizing uses the effective training dataloader length (number of executed batches / optimizer steps), not a floor-division approximation from dataset length. This prevents `total_steps == 0` bugs on small datasets and keeps schedules correct under `drop_last` policies.

## Acceptance Criteria

- A CLM tokenization config with no size specified validates and runs, producing a dataset with variable-length `input_ids` and `length`.
- A CLM training config with `dataset.packing.enabled: true` can train for a short smoke run and receives fixed-length batches (shape `[batch_size, sequence_length]`).
- Multi-process (or multi-node) SLURM smoke run completes without errors, and logs confirm correct world size and rank coordination (per issue 37 patterns).
- Unit tests cover packing correctness (EOS insertion, block size, labels) and distributed sharding invariants (no overlap between ranks for a fixed seed).
- CI-like tests pass via `python -m tox -e py310`.

## Scope & Invariants (Approval Required if Expanded)

Likely touched files:

- Tokenization:
  - `src/tasks/tokenization/orchestrator.py`
  - `src/tasks/tokenization/tokenizer/config.py`
  - `src/tasks/tokenization/tokenizer/causal.py`
  - `config/schemas/tokenization/tokenization.clm_training.schema.yaml`
  - `config/tests/`
- Training:
  - `src/tasks/training/fabric/trainer/base.py`
  - `config/schemas/training/components/data.schema.yaml`
  - `config/tests/`

Invariants:

- No new task types; keep existing `task: tokenization` and `task: clm_training`.
- Do not bypass config validation; configs must remain reviewable before SLURM submission.
- No new Python-based SLURM submission logic; continue using `slurm/submit_job.sh` and the existing Fabric workflow.
- Default behavior remains unchanged unless the new config flags are enabled.

## ExecPlan

- `issues/execPlans/42-online-packing.execplan.md`

## Design Contracts (Junior-proof)

These are non-negotiable “contracts” that define correct behavior for v1. If an implementation cannot satisfy one of these, it must fail fast with a clear error.

### Where packing lives

- Online packing is implemented as a **map-style dataset wrapper** (e.g. `PackedSequenceDataset`) that yields already-packed fixed-length blocks.
- Packing is **not** implemented in a `collate_fn` / HF `DataCollator` (collators only see pre-sampled items and become stateful/fragile under `num_workers>0`, DDP ranks, and resume).
- Packing must not materialize a global “token stream” in memory. The token stream definition is conceptual; the implementation must precompute only lightweight offsets (prefix sums) and produce each packed block on-demand via indexing.

### Packing input contract (tokenized-docs dataset)

- Source HF dataset rows must contain:
  - `input_ids: List[int]` (variable-length per row)
  - `length: int` (token count; required for efficient packing + accounting)
- For correctness, `length` must equal `len(input_ids)` for every row. The packer must validate this invariant on a small deterministic sample and fail fast on mismatch (stale/corrupt `length` silently causes out-of-bounds packing bugs).
- Source dataset must **not** be required to contain `attention_mask` or `labels` in packing mode.
- Empty documents (`length == 0`) are skipped; the implementation must log how many were skipped (rank 0).

### Packing output contract (what the trainer sees)

- Every packed sample has fixed shape `[sequence_length]`.
- Batches have fixed shape `[batch_size, sequence_length]`.
- For CLM packing v1:
  - `attention_mask` is all ones (no padding)
  - `labels == input_ids`

### EOS insertion semantics (document boundary)

- If `insert_eos: true`, the packer appends exactly one EOS token *after each document* (unless the document already ends in EOS). This naturally creates a single EOS separator between documents and may also leave the token stream ending with EOS after the final document.
- Avoid double-EOS: if a document already ends in EOS, do not insert an extra EOS.
- The packer does not strip or rewrite other special tokens (e.g. BOS); it only controls optional EOS insertion between documents.
- EOS insertion never happens “inside” a document; documents longer than `sequence_length` may span multiple blocks.
- If `insert_eos: true`, `eos_token_id` must be resolvable (either provided directly or via `tokenizer_name`); otherwise error (no silent fallback).

### Determinism / resume-safety

- `PackedSequenceDataset.__getitem__(i)` must be a pure function of `i` and configuration (no RNG, no mutable cross-worker state).
- Shuffling (when enabled) is done via the sampler over packed block indices. In DDP, `DistributedSampler.set_epoch(epoch)` must be called every epoch.
- In distributed runs, the default policy is “no silent duplication across ranks”: prefer dropping remainder over repeating samples.
  - Concretely in packing mode: default `sampler_drop_last: true` when `world_size > 1`, and apply it consistently across splits (train/valid) unless explicitly overridden for full validation coverage.

## Fail-fast Conditions (explicit errors)

- Variable-length tokenization + `tokenizer.overlap > 0` is invalid (hard error).
- Packing enabled but source dataset does not have `input_ids` (hard error).
- Packing enabled but source dataset does not have `length` (hard error).
- Packing enabled and `insert_eos: true` but neither `packing.eos_token_id` nor `packing.tokenizer_name` is provided (hard error).
- Packing enabled but the dataset appears already “offline packed” (e.g. fixed-length `input_ids` with existing `attention_mask/labels`) → hard error with a remediation hint (“disable packing or point to doc-level tokenization output”).

## Verification Recipes (copy/paste)

### 1) Validate configs

    python src/main.py --validate --config config/tests/tokenization_doclevel_smoke.yaml
    python src/main.py --validate --config config/tests/clm_training_packing_smoke.yaml

### 2) Run doc-level tokenization

    python src/main.py --config config/tests/tokenization_doclevel_smoke.yaml

Expected: output at `output/tests/tokenized_doclevel` with variable-length `input_ids` and `length`.

### 3) Run packing training smoke

    python src/main.py --config config/tests/clm_training_packing_smoke.yaml

Expected: logs (rank 0) confirm packing enabled and the first batch tensor shapes are `[batch_size, sequence_length]`.

### 3b) Run CI-like tests (recommended)

    python -m tox -e py310

### 4) SLURM multi-node smoke (shared filesystem required)

Important: `output/tests/tokenized_doclevel` must be on a filesystem shared across nodes.

    ./slurm/tests/run_tests.sh --config config/tests/tokenization_doclevel_smoke.yaml
    ./slurm/tests/run_tests.sh --config config/tests/clm_training_packing_multinode_smoke.yaml --nodes 2 --ntasks-per-node 1

## Out of Scope (v1) + Follow-ups

- Multi-input dataset mixing / replay (token-budget ratios) is explicitly out of scope for this issue (follow-up: `issues/43-token-budget-mixture-replay.md`).
- Extension pathway requirement (v1 must enable v2 cleanly):
  - Keep packing logic isolated as “doc-level tokens → fixed-length blocks” (single-source).
  - Ensure the training integration consumes a generic `torch.utils.data.Dataset` of packed blocks + an explicit sampler policy, so a later `MixturePackedDataset` can compose multiple per-dataset packed-block datasets without changing model/training code.
- Note: doc-level tokenized datasets are not compatible with existing workflows that assume fixed-shape `attention_mask/labels` (e.g. `dataset_merge`); packing mode is the intended consumer.

## Recommended additions for large datasets (v1.1)

We want online packing to remain correct and deterministic while scaling to tens of millions of documents. The current v1 implementation is correct but can become startup- and RAM-heavy because it scans every row and stores Python lists for offsets/metadata.

Recommended additions (low-risk, incremental):

1) **Tokenization emits `ends_with_eos: bool` (doc-level only)**  
   - In doc-level CLM tokenization mode, add an `ends_with_eos` column computed at tokenization time.  
   - Rationale: packing index build should not need to inspect `input_ids[-1]` for every document at training startup.

2) **Persisted packing index (memory-mapped) next to the dataset**  
   - Introduce an on-disk, versioned index artifact stored alongside the tokenized dataset directory, e.g.:
     - `<dataset_path>/.packing_index/v1/<split>/offsets.int64`
     - `<dataset_path>/.packing_index/v1/<split>/doc_lengths.int32`
     - `<dataset_path>/.packing_index/v1/<split>/doc_indices.int64`
     - `<dataset_path>/.packing_index/v1/<split>/doc_stream_lengths.int32` (or derive via `insert_eos` + `ends_with_eos`)
     - `<dataset_path>/.packing_index/v1/<split>/meta.json` (packing params + dataset fingerprint)
   - Use `numpy.memmap` for arrays so RAM usage is bounded and startup is fast.
   - In SLURM/DDP: only rank 0 builds the index, other ranks wait on a barrier and then load it. Fail fast with a clear error if the dataset path is not on a shared filesystem.

3) **Decouple rank-evenness from batch-evenness (drop_last policy)**  
   - Keep `sampler_drop_last` as the “no duplication across ranks” control.
   - Add a separate `drop_last_batch` setting for dropping partial batches (coverage vs. stability trade-off). Default recommendation:
     - train: `drop_last_batch: true` in distributed runs
     - valid: `drop_last_batch: false`

Acceptance for v1.1:
- Training startup time does not scale linearly with number of docs once the index exists (index reuse).
- Peak RAM does not scale with `num_docs` (bounded by memmap window + small buffers).
- Index is invalidated/rebuilt when any of these change: dataset fingerprint, `sequence_length`, `insert_eos`, `eos_token_id`, index version.

These additions are explicitly intended to avoid “reinventing the wheel” at the storage layer: Hugging Face Datasets remains the storage/row-access abstraction; we only add the missing deterministic packing index artifact.

## Implementation status (2026-02-05)

Implemented the v1 workflow end-to-end (tokenization → training packing), including schemas, docs, smoke configs, and unit tests.

Milestone 4 (large-dataset support) implementation is in progress in this branch/worktree:
- `ends_with_eos` emitted by doc-level tokenization.
- Persisted packing index (memory-mapped) under `dataset.nameOrPath/.packing_index` (configurable via `packing.index_cache_dir`).
- Decoupled `sampler_drop_last` (rank-evenness) from `drop_last_batch` (partial batch dropping).

Key code locations:

- Tokenization doc-level mode:
  - `src/tasks/tokenization/orchestrator.py`
  - `src/tasks/tokenization/tokenizer/config.py`
  - `src/tasks/tokenization/tokenizer/causal.py`
- Training-time packing:
  - `src/tasks/training/data/packing.py`
  - `src/tasks/training/fabric/trainer/base.py`
  - `src/tasks/training/utils.py`
- Schemas/configs/docs/tests:
  - `config/schemas/training/components/data.schema.yaml`
  - `config/schemas/tokenization/tokenization.clm_training.schema.yaml`
  - `config/tests/tokenization_doclevel_smoke.yaml`
  - `config/tests/clm_training_packing_smoke.yaml`
  - `config/tests/clm_training_packing_multinode_smoke.yaml`
  - `config/tests/online_packing_integration_smoke.yaml`
  - `docs/TOKENIZATION.md`, `docs/CLM_TRAINING.md`
  - `tests/test_packed_sequence_dataset.py`, `tests/unit/config/test_online_packing_schema.py`

### Test evidence / remaining validation

- Added schema-level unit tests that validate:
  - doc-level CLM tokenization config can omit `overlap`;
  - packing requires `sequence_length` when enabled.
- Local evidence (2026-02-05):
  - `python3 -m pytest -q tests/unit/config/test_online_packing_schema.py` → `3 passed`
  - `python3 -m pytest -q tests/unit/config/test_test_configs_defaults.py` → `4 passed`
- Local evidence (2026-02-05, Milestone 4):
  - `python3 -m pytest -q tests/unit/training/test_packing_index.py` → `3 passed`
- Full runtime tests and smoke runs must be executed in a real repo environment (Conda/Poetry/tox) where HF `datasets` + `transformers` are installed.
- SLURM validation is still required for multi-node evidence: record job IDs + logs here once run.

### Known risks (v1)

- `PackedSequenceDataset` precomputes prefix-sum offsets by scanning the full source dataset once at init time and stores per-document metadata in memory (`O(num_docs)`). This is correct but can be slow/heavy on very large corpora; if it becomes a bottleneck, we should consider a chunked/blocked prefix-sum index or a binary index file stored alongside the dataset.
- EOS resolution uses `AutoTokenizer.from_pretrained(tokenizer_name)` when `eos_token_id` is not provided. In offline/HPC environments, ensure the tokenizer is already cached locally or set `packing.eos_token_id` explicitly.
