# Token-budget multi-dataset mixing + replay (homogeneous packed blocks)

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

PLANS.md is checked into the repo at `PLANS.md`, and this document must be maintained in accordance with it.

## Purpose / Big Picture

Enable CLM continual pretraining on a mixture of multiple tokenized-doc datasets (produced by issue 42 doc-level tokenization) with a user-defined **token/blocks ratio** and optional replay, while keeping every packed training block **homogeneous** (attributable to exactly one dataset source).

After this change, a novice can:

1) Configure multiple sources under `dataset.sources` with weights (e.g. A:2, B:1, R:1).
2) Set a single run budget as a total number of packed blocks (`dataset.mixture.total_blocks`) and run training with `number_epochs: 1`.
3) Observe deterministic, DDP/FSDP-safe interleaving of dataset sources with auditable accounting written to `output_dir/mixture_report.json`.

This work must be safe under SLURM multi-node + Lightning Fabric. It must avoid silent sampler padding duplication, avoid distributed hangs, and fail fast on misconfiguration.

## Progress

- [ ] (2026-02-05 00:00Z) Add schema surface for multi-source dataset mixing.
- [ ] (2026-02-05 00:00Z) Implement multi-source loading + packing wrappers.
- [ ] (2026-02-05 00:00Z) Implement deterministic interleaving dataset (`MixturePackedDataset`).
- [ ] (2026-02-05 00:00Z) Implement auditable accounting + persisted report.
- [ ] (2026-02-05 00:00Z) Add unit tests + smoke configs + SLURM evidence.

## Surprises & Discoveries

- (fill in during implementation)

## Decision Log

- Decision: Mixing is **block-level interleaving** (no token-level mixing inside a packed block).
  Rationale: Makes determinism, accounting, debugging, and distributed safety straightforward. Token-level mixing changes semantics and is harder to audit.
  Date/Author: 2026-02-05 / Codex

- Decision: v1 run length is expressed as `dataset.mixture.total_blocks` and we recommend `number_epochs: 1`.
  Rationale: The existing trainer is epoch-driven and schema requires `number_epochs`. Expressing the whole run as one epoch avoids complex per-epoch mixture state updates across DataLoader workers.
  Date/Author: 2026-02-05 / Codex

- Decision: The mixture schedule must have exact global counts per dataset derived from weights, and must be randomized in order via a deterministic permutation.
  Rationale: Exact counts make accounting and debugging simple; deterministic permutation avoids long contiguous runs of a single dataset without storing a large schedule array in memory.
  Date/Author: 2026-02-05 / Codex

## Outcomes & Retrospective

- (fill in after milestones complete)

## Context and Orientation

LMTK is YAML-driven. `src/main.py` loads a YAML config, validates it with `src.config.config_loader.ConfigValidator` (schemas under `config/schemas/`), and dispatches to the task module under `src/tasks/` based on `config.task`.

This plan extends the `task: clm_training` data pipeline. Issue 42 introduced:

- Doc-level CLM tokenization output: variable-length `input_ids` per doc plus `length` and `ends_with_eos`.
- Training-time online packing: `PackedSequenceDataset` in `src/tasks/training/data/packing.py`, backed by a persisted memmap `PackingIndex` in `src/tasks/training/data/packing_index.py`.
- Fabric-safe distributed sampling policies in `src/tasks/training/fabric/trainer/base.py`.

The current training orchestrator (`src/tasks/training/orchestrator.py`) loads exactly one dataset from `dataset.nameOrPath`. For this issue, we must support multiple disk datasets under `dataset.sources`.

## Interfaces and Dependencies (what must exist at end)

### Configuration (schema)

Update `config/schemas/training/components/data.schema.yaml` so `dataset` supports one of:

1) Single dataset (existing behavior):
   - `dataset.nameOrPath` (required) + optional `dataset.packing`

2) Multi-source dataset mixing (new behavior):
   - `dataset.sources` (required; non-empty array)
   - `dataset.mixture` (required when `dataset.sources` is present)
   - `dataset.packing` remains the packing config applied to **all** sources (must be compatible).

Define `dataset.sources` item schema:

- `dataset_id: string` (required; unique within config; used for reporting)
- `nameOrPath: string` (required; path to tokenized-doc dataset on disk)
- `weight: number` (required; >0; mixing weight)
- `index_cache_dir: string | null` (optional; overrides default `<nameOrPath>/.packing_index`)

Define `dataset.mixture` schema:

- `enabled: boolean` (required; must be true when sources are present)
- `total_blocks: integer` (required; >= 1; global number of packed blocks for the run)
- `schedule_seed: integer | null` (optional; defaults in code to training `seed` or 0)
- `report_path: string | null` (optional; defaults in code to `<output_dir>/mixture_report.json`)
- `validation_mode: string | null` (optional; enum: `first_source`, `concat_all`; defaults in code to `first_source`)

### Runtime (Python)

Create a new dataset wrapper for mixing (map-style):

- `src/tasks/training/data/mixture.py:MixturePackedDataset`
  - `__len__ -> total_blocks`
  - `__getitem__(i) -> dict[str, torch.Tensor]` with keys:
    - `input_ids`, `attention_mask`, `labels` (from a per-source `PackedSequenceDataset`)
    - `dataset_id` (a scalar tensor encoding which source the block came from)

Implement a deterministic mapping without storing an `O(total_blocks)` schedule array:

1) Convert weights to exact block counts:
   - Let `W = sum(weights)`
   - For each dataset k:
     - `raw_k = total_blocks * weight_k / W`
     - `count_k = floor(raw_k)`
   - Distribute the remaining `R = total_blocks - sum(count_k)` blocks by largest fractional part.
   - Tie-break deterministically by `dataset_id` lexical order.

2) Randomize order by a permutation of `0..total_blocks-1`:
   - Use an affine permutation `j = (a*i + b) mod N` with `gcd(a, N) == 1`.
   - Derive `a` and `b` deterministically from `schedule_seed`.
   - Find `dataset_id` by mapping `j` into the prefix-sum ranges of `count_k`.

3) Choose local block index with replacement:
   - `local = hash64(schedule_seed, i, dataset_id) % num_blocks_in_source`

Per-source datasets:

- Load each tokenized-doc dataset from disk (`datasets.load_from_disk`).
- Ensure it has a `'train'` split (wrap single split as `DatasetDict({"train": ...})` like existing code).
- Apply `_ensure_validation_split` logic per source if validation is enabled.
- Build/reuse a `PackingIndex` per split under each source’s `index_cache_dir`.
- Wrap each split with `PackedSequenceDataset`.

Trainer integration:

- Extend `src/tasks/training/fabric/trainer/base.py` to support:
  - single-source packing (existing) and
  - multi-source mixing + packing (new).
- Ensure DDP/FSDP sampler behavior remains explicit:
  - create a `DistributedSampler` for the mixture dataset with `drop_last=True` when `world_size > 1`.
  - fail fast if `total_blocks` is not divisible by `world_size` when distributed (to avoid dropping/padding that would skew budgets).

Accounting:

- In the training loop, if `dataset_id` is present in the batch:
  - count blocks per dataset id (`torch.bincount` on CPU is fine),
  - aggregate across ranks at end-of-run,
  - write a JSON report on rank 0.

## Milestones

### Milestone 1: Schema + config surface

Scope:
- Update `config/schemas/training/components/data.schema.yaml` with `dataset.sources` + `dataset.mixture` and `anyOf`/`if-then` to require:
  - either `nameOrPath` (single) or `sources` (multi),
  - `mixture.total_blocks` when `sources` present.

Acceptance:
- `python src/main.py --validate --config <new smoke config>` succeeds.

### Milestone 2: Multi-source loading + packing wrappers

Scope:
- Add helper(s) to load multiple datasets from disk and apply issue-42 packing per source.
- Fail-fast checks:
  - all sources exist on disk,
  - each source has required columns (`input_ids`, `length`),
  - all sources share compatible packing config (`sequence_length`, EOS settings / tokenizer).

Acceptance:
- A unit test can create two fake in-memory splits and build per-source packing indices without errors.

### Milestone 3: MixturePackedDataset + deterministic mapping

Scope:
- Implement `MixturePackedDataset` in `src/tasks/training/data/mixture.py`.
- Ensure `__getitem__` is deterministic and does not allocate proportional to `total_blocks`.

Acceptance:
- Unit tests prove:
  - exact counts per dataset across `total_blocks`,
  - determinism for a fixed seed,
  - DDP sharding does not hang and does not silently pad/duplicate indices (we enforce divisibility).

### Milestone 4: Accounting + persisted report

Scope:
- Add per-dataset counters and write `mixture_report.json` at end-of-run on rank 0.
- Report must include:
  - config summary (dataset ids, weights, total_blocks, sequence_length),
  - realized blocks and tokens per dataset,
  - realized ratios.

Acceptance:
- Smoke run produces `mixture_report.json` with correct totals.

### Milestone 5: Tests + configs + SLURM evidence

Scope:
- Unit tests (no HF datasets dependency required):
  - allocation exactness + deterministic remainder rule,
  - permutation is bijective for N,
  - local index draw bounded by num_blocks.
- Add smoke config(s) under `config/tests/`:
  - `clm_training_packing_mixture_smoke.yaml` (single-node),
  - `clm_training_packing_mixture_multinode_smoke.yaml` (multi-node).
- Add a `task: testing` integration config that runs:
  - tokenization doc-level for two tiny sources (or reuse existing small tutorial dataset twice),
  - training mixture smoke.
- Run SLURM multi-node smoke via `slurm/tests/run_tests.sh` and record job ID/log paths in `issues/43-token-budget-mixture-replay.md`.

Acceptance:
- Unit tests pass locally.
- SLURM jobs exit 0 and report file exists on shared filesystem.

## Concrete Steps (commands)

Local unit tests:

    python3 -m pytest -q tests/unit/training/test_mixture_packing.py

Config validation:

    python src/main.py --validate --config config/tests/clm_training_packing_mixture_smoke.yaml

SLURM (example):

    ./slurm/tests/run_tests.sh --config config/tests/clm_training_packing_mixture_multinode_smoke.yaml --nodes 2 --ntasks-per-node 1

## Validation and Acceptance

The feature is accepted when:
- A mixture smoke run completes under DP and DDP/FSDP without hangs.
- `mixture_report.json` exists and totals match `total_blocks`:
  - `sum(blocks_per_dataset) == total_blocks`
  - `sum(tokens_per_dataset) == total_blocks * sequence_length`
- The observed per-dataset ratios match configured weights within expected rounding error.

## Idempotence and Recovery

- Index artifacts are reused (same semantics as issue 42). If you need to rebuild, delete the relevant `.packing_index/` directories.
- If a mixture run fails, ensure `BUILD_FAILED.json` and `LOCK` files are cleaned up only after confirming no active job is running.

