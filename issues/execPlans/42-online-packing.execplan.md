# Online packing (training-time packaging) for CLM via variable-length tokenization

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

PLANS.md is checked into the repo at `PLANS.md`, and this document must be maintained in accordance with it.

## Purpose / Big Picture

Enable a new CLM workflow where:

1) `task: tokenization` can produce a “tokenized-docs” dataset (variable-length `input_ids`, no fixed-length padding/truncation) when the config omits a size (`context_length` / `max_sequence_length`); and

2) `task: clm_training` can opt into `dataset.packing.enabled: true` to pack those variable-length token sequences into fixed-length training blocks (shape `[batch_size, sequence_length]`) at training time.

This is meant to be HPC-safe and multi-node-safe under SLURM + Lightning Fabric. The design explicitly avoids silent data duplication across ranks and avoids distributed hangs by ensuring all ranks see the same number of batches.

Terminology note (common source of confusion): this plan implements **packing**, not **dynamic padding**. Packing produces fixed-length blocks (every sample has length `sequence_length`) and avoids per-batch padding. Dynamic padding is the opposite: it pads each batch to a variable max length, yielding variable shapes and (in our prior experiments) degraded training performance. Dynamic padding is out of scope here.

After this change, a novice can run the new smoke configs under `config/tests/` and observe:

- tokenization output with variable-length examples (`input_ids` lengths differ) saved under `output/tests/tokenized_doclevel`;
- training receiving fixed-shape batches of size `[batch_size, sequence_length]` with `attention_mask` all ones and `labels == input_ids`.

## Progress

- [x] (2026-01-28 13:00Z) Drafted initial issue 42 + ExecPlan.
- [x] (2026-01-28 13:20Z) Revised plan for Fabric correctness: explicit sampler/drop policy, avoid Fabric sampler duplication, corrected CLI commands (`python src/main.py`).
- [x] (2026-02-05 00:00Z) Implemented Milestone 1 (doc-level CLM tokenization) with schema + smoke config + docs.
- [x] (2026-02-05 00:00Z) Implemented Milestone 2 (training-time packing) with deterministic dataset/sampler integration and fail-fast guards.
- [x] (2026-02-05 00:00Z) Implemented Milestone 3 (unit + integration scaffolding): unit tests + smoke configs + integration runner config + docs.
- [x] (2026-02-05 00:00Z) Implemented Milestone 4 (large-dataset support): persisted packing index (memmap) + `ends_with_eos` + explicit drop policies.
- [x] (2026-02-05 00:00Z) Hardened packing-index operational safety: LOCK contains PID/host/time; distributed runs fail fast on rank0 index-build failure (no barrier hangs); added regression tests for sampler disjointness and `ends_with_eos` mismatch.
- [ ] (2026-02-05 00:00Z) Run repo test suite in a full env (`python -m tox -e py310`) and record results in issue 42.
- [ ] (2026-02-05 00:00Z) Run runtime smoke locally (tokenization + packing training) and record key logs (packing enabled + batch shapes).
- [ ] (2026-02-05 00:00Z) Run SLURM multi-node smoke (`clm_training_packing_multinode_smoke.yaml`) and record job IDs + log paths in issue 42.

## Surprises & Discoveries

- Observation: `src/main.py` must be executed as a script (`python src/main.py ...`) because it imports task modules as `tasks.<name>`, which relies on Python adding `src/` (the script directory) to `sys.path`.
  Evidence: `src/main.py` uses `__import__(f"tasks.{module_name}", ...)`.

- Observation: `FabricTrainerBase` currently creates dataloaders in `__init__` (before `fabric.launch`). This is fine when Fabric injects its own distributed sampler, but packing mode requires a rank/world-size-aware sampler that must be constructed **after** Fabric has initialized distributed state (works for both SLURM external launches and Fabric-spawned runs).
  Evidence: `src/tasks/training/fabric/trainer/base.py:FabricTrainerBase.__init__` calls `_load_fabric_datasets_dataloaders`, and `_pipeline` later calls `fabric.setup_dataloaders(...)`.

- Observation: Lightning Fabric only injects a distributed sampler for **map-style datasets**. For `IterableDataset`, Fabric will not auto-replace the sampler.
  Evidence: `lightning/fabric/fabric.py:_requires_distributed_sampler` returns `False` when `has_iterable_dataset(dataloader)` is true.

- Observation: Lightning Fabric’s default distributed sampler kwargs do not set `drop_last`, meaning PyTorch’s `DistributedSampler(drop_last=False)` behavior can **pad and repeat samples** when `len(dataset)` is not divisible by `world_size`.
  Evidence: `lightning/fabric/strategies/parallel.py:ParallelStrategy.distributed_sampler_kwargs` returns only `{"num_replicas": ..., "rank": ...}`; `lightning/fabric/fabric.py:_get_distributed_sampler` does not set `drop_last`.

- Observation: This Codex sandbox environment does not have Hugging Face `datasets` installed, so full `pytest` collection fails on any module that imports `datasets`.
  Evidence: Local `pytest` collection errors with `ModuleNotFoundError: No module named 'datasets'`.

These observations drive the design choices in this plan: we implement packing as a map-style dataset for deterministic sizing and we take control of distributed sampling (including `drop_last`) to avoid silent duplication and distributed hangs.

## Decision Log

- Decision: Implement online packing as a **map-style** `torch.utils.data.Dataset`, not an `IterableDataset`.
  Rationale: Map-style datasets work well with deterministic indexing, have a stable `__len__`, work with PyTorch samplers cleanly, and are easier to resume (the current trainer resumes by skipping batches in a dataloader iterator).
  Date/Author: 2026-01-28 / Codex

- Decision: Do **not** implement packing in a `collate_fn` / HF `DataCollator`.
  Rationale: Collators only see already-sampled items and become stateful/fragile under `num_workers>0`, distributed ranks, and checkpoint resume. Packing must be indexable + deterministic; that is naturally expressed as a dataset wrapper that yields fixed-length blocks.
  Date/Author: 2026-01-29 / Codex

- Decision: In packing mode, do **not** rely on Fabric’s `use_distributed_sampler=True` default. Instead, create an explicit sampler and call `fabric.setup_dataloaders(..., use_distributed_sampler=False)` for those dataloaders.
  Rationale: We must guarantee “no silent duplication across ranks” and “no distributed hangs”. PyTorch’s default `DistributedSampler(drop_last=False)` can repeat samples for padding; also we must enforce a consistent `drop_last` policy. Explicit sampler construction makes this behavior auditable and configurable.
  Date/Author: 2026-01-28 / Codex

- Decision: In distributed runs (`world_size > 1`), default to `sampler_drop_last: true` for packing mode.
  Rationale: If we don’t drop, the only way to keep all ranks at equal length is to pad by repeating samples. Dropping tail samples is the standard trade-off when correctness (no duplication) is more important than full-coverage-per-epoch.
  Date/Author: 2026-01-28 / Codex

- Decision: Variable-length CLM tokenization is enabled by omitting `context_length`/`max_sequence_length`, and overlap/stride will be rejected in that mode.
  Rationale: Overlap implies fixed windowing semantics and would be ambiguous/unsafe when producing doc-level sequences.
  Date/Author: 2026-01-28 / Codex

- Decision: Packing inserts an EOS token between documents by default.
  Rationale: Preserves document boundary semantics and prevents accidental cross-document continuation without an explicit delimiter.
  Date/Author: 2026-01-28 / Codex

- Decision: v1 packing always drops tail tokens that don’t fill a full block.
  Rationale: This keeps all samples strictly fixed-shape without reintroducing padding/masking semantics. “No tokens dropped” can be tackled later as a token-budget / streaming feature (stateful iterator), which is a larger design.
  Date/Author: 2026-01-28 / Codex

- Decision: Always call `set_epoch(epoch)` on any `torch.utils.data.DistributedSampler` used for training (packing and non-packing).
  Rationale: This is a PyTorch requirement for correct epoch-to-epoch shuffling in DDP; without it, the sampler repeats the exact same shuffle order every epoch.
  Date/Author: 2026-01-28 / Codex

- Decision: Scheduler sizing uses the dataset backing the train dataloader (the “effective dataset”), not the originally loaded HF dataset.
  Rationale: Packing changes the unit of training from “documents” to “packed blocks”; scheduler step counts must be based on what the dataloader actually yields or warmup/decay schedules will be wrong and can even error.
  Date/Author: 2026-01-28 / Codex

- Decision: Construct packing dataloaders (and their distributed samplers) inside `_pipeline` after Fabric initializes rank/world size.
  Rationale: This makes packing-mode sampling correct both when training is launched externally (SLURM `srun`) and when Fabric spawns processes locally.
  Date/Author: 2026-01-28 / Codex

- Decision: Only improve training-config ergonomics for the new `dataset.packing` surface in this plan (safe code defaults); do not attempt a repo-wide training-config simplification.
  Rationale: `src/config/config_loader.ConfigValidator` validates but does not apply JSON-schema defaults. Broad “make fields optional” changes would require a separate config-system plan to safely inject defaults across the entire training configuration.
  Date/Author: 2026-01-28 / Codex

- Decision: v1 packing supports a single input dataset only; multi-dataset mixing is deferred.
  Rationale: Mixing semantics (doc-level vs token-level), determinism, distributed sharding policy, scheduler sizing, and resume correctness need explicit design + tests. We will keep a clean extension point (“doc source”) but not ship mixing behavior in this issue.
  Date/Author: 2026-01-29 / Codex

- Decision: `dataset.packing` is supported only for `task: clm_training` in v1.
  Rationale: v1 packing generates `labels == input_ids` and enforces CLM-specific assumptions; enabling it for other tasks would silently produce incorrect supervision. We fail fast instead.
  Date/Author: 2026-02-05 / Codex

- Decision: For large datasets, persist a versioned packing index as memory-mapped arrays stored next to the tokenized dataset.
  Rationale: Re-scanning all documents and keeping Python lists in RAM at each training startup does not scale to large corpora. HF Datasets provides the storage and row access; we only add the missing deterministic prefix-sum index artifact.
  Date/Author: 2026-02-05 / Codex

- Decision: Decouple “no duplication across ranks” (`sampler_drop_last`) from “drop partial batches” (`drop_last_batch`).
  Rationale: Dropping tail *blocks* for rank-evenness is a distributed correctness policy; dropping the final partial *batch* is a throughput/shape policy. Keeping them separate makes coverage vs. stability explicit and avoids accidental extra data loss.
  Date/Author: 2026-02-05 / Codex

- Decision: Write PID/host/timestamp into the packing-index LOCK file and include existing LOCK contents in timeout errors.
  Rationale: On HPC filesystems, lock-related timeouts are hard to debug. Metadata makes it clear whether the lock is stale or an active build, without changing the locking mechanism.
  Date/Author: 2026-02-05 / Codex

- Decision: Prevent distributed hangs when rank 0 fails building the packing index by writing a build-failure sentinel and raising on all ranks after the barrier.
  Rationale: A rank0 exception before `fabric.barrier()` would otherwise hang all other ranks indefinitely. A sentinel makes the failure auditable and ensures all ranks exit promptly.
  Date/Author: 2026-02-05 / Codex

## Contracts

These contracts are the “definition of done” for correctness. If an implementation cannot satisfy one, it must fail fast.

### Where packing lives

- Packing is implemented as a **map-style dataset wrapper** that yields fixed-length blocks (not a collator).
- Trainer continues to consume rectangular tensors; default PyTorch collation should work in packing mode.

### Input contract (tokenized-docs dataset)

- Source HF dataset rows must contain:
  - `input_ids: List[int]` (variable-length)
  - `length: int` (token count; used for sizing/accounting and efficient mapping)
- For correctness, `length` must equal `len(input_ids)` for every row. The packer must validate this invariant on a small deterministic sample and fail fast on mismatch (stale/corrupt `length` silently causes out-of-bounds packing bugs).
- Packing mode must not require `attention_mask` or `labels` columns on the source dataset.
- Empty docs (`length == 0`) are skipped; rank 0 must log how many were skipped.

### Output contract (packed blocks)

- Each packed sample is fixed-length: `len(input_ids) == sequence_length`.
- For CLM packing v1:
  - `attention_mask` is all ones (no padding semantics)
  - `labels == input_ids`

### EOS insertion semantics

- If `insert_eos: true`, append exactly one EOS token *after each document* (unless the document already ends with EOS). This naturally creates a single EOS separator between documents and may also leave the token stream ending with EOS after the final document.
- Avoid double-EOS: if a doc already ends with EOS, do not insert an extra EOS.
- The packer does not strip or rewrite other special tokens (e.g. BOS); it only controls optional EOS insertion after documents.
- EOS is never inserted “inside” a document; docs longer than `sequence_length` may span multiple blocks.
- If `insert_eos: true`, EOS must be resolvable:
  - either `packing.eos_token_id` is provided, or
  - `packing.tokenizer_name` is provided and resolves `.eos_token_id`.
  Otherwise: hard error (no silent fallback).

### Determinism / resume-safety

- `PackedSequenceDataset.__getitem__(i)` must be a pure function of `i` and config (no RNG, no mutable cross-worker state).
- Shuffling is done via the sampler over **packed block indices** (not “reshuffle docs and repack” each epoch).
- In distributed runs, `DistributedSampler.set_epoch(epoch)` is called every epoch.
- In distributed runs, the default policy is “no silent duplication across ranks”: prefer dropping remainder over repeating samples.

## Outcomes & Retrospective

Implemented doc-level tokenization (`input_ids` + `length` + `ends_with_eos`) and training-time packing via a map-style dataset wrapper with deterministic distributed sampling. Added schemas, docs, smoke configs, integration runner config, and unit tests. Large-dataset support is implemented via a persisted, versioned memmap packing index. Remaining work is validation in a full repo environment (`tox`) plus SLURM multi-node evidence recorded in issue 42.

## Context and Orientation

LMTK is YAML-driven. Each YAML configuration denotes a single job and maps to exactly one task module under `src/tasks/`. `src/main.py` loads a YAML config, validates it via `src.config.config_loader.ConfigValidator` (schemas under `config/schemas/`), and dispatches based on `config.task`.

Important “how to run” detail for novices:

- Run configs as `python src/main.py --config <path>`. Do not use `python -m src.main` because `src/` is not installed as a standard package and `src/main.py` imports task modules as `tasks.<name>`, which requires `src/` to be on `sys.path` (true when running the script directly).

Relevant modules for this change:

- Tokenization task:
  - Orchestrator: `src/tasks/tokenization/orchestrator.py`
  - CLM tokenizer: `src/tasks/tokenization/tokenizer/causal.py`
  - Tokenizer config dataclass: `src/tasks/tokenization/tokenizer/config.py`
  - Tokenization schema for CLM: `config/schemas/tokenization/tokenization.clm_training.schema.yaml`

- Training task (CLM):
  - Fabric trainer base: `src/tasks/training/fabric/trainer/base.py`
  - Training data schema: `config/schemas/training/components/data.schema.yaml`
  - Testing orchestrator (runs multiple configs in order): `src/tasks/testing/orchestrator.py`

Definitions used in this plan:

- “Tokenized-docs dataset” means a Hugging Face dataset saved to disk where each row is a document/sample and contains at least:
  - `input_ids`: variable-length list of token IDs (ints)
  - `length`: integer length of `input_ids`
  It does not need `attention_mask` or `labels`; training-time packing will generate those.

- “Packing” means converting the sequence-of-documents into a token stream (optionally inserting EOS between documents) and slicing it into fixed-size blocks of `sequence_length`. Each block becomes one training sample.

- “No silent duplication across ranks” means we must avoid the default distributed-sampler behavior of repeating samples to pad out even splits. The correct alternative is dropping tail samples (documented and explicit).

## Milestones

### Milestone 1: Variable-length CLM tokenization (“tokenized docs”)

What will exist at the end:

- A tokenization config can omit `tokenizer.context_length` / `tokenizer.max_sequence_length` (size) for CLM and produce a dataset where `input_ids` are variable-length.
- The output dataset includes a `length` field.
- Setting `tokenizer.overlap > 0` in this mode fails fast with a clear error.
- A smoke config exists under `config/tests/tokenization_doclevel_smoke.yaml`.

Config snippet to add (exact file content):

    task: tokenization
    experiment_name: test_tokenization_doclevel_smoke
    verbose_level: 1
    seed: 42

    tokenizer:
      tokenizer_name: hf-internal-testing/llama-tokenizer
      task: clm_training
      # Intentionally omit context_length/max_sequence_length to enable doc-level tokenization
      # overlap must be 0/unset in this mode (overlap > 0 is a hard error)
      batch_size: 64
      num_proc: 2
      show_progress: false

    dataset:
      source: local
      nameOrPath: tutorials/data/raw_text_data
      format: files
      file_config:
        format: txt

    output:
      path: output/tests/tokenized_doclevel

    test_size: 0

Acceptance:

- Validate:
    python src/main.py --validate --config config/tests/tokenization_doclevel_smoke.yaml
  Expected: “Configuration is valid!”

- Execute:
    python src/main.py --config config/tests/tokenization_doclevel_smoke.yaml
  Expected: logs show tokenization running and saving to `output/tests/tokenized_doclevel`.

### Milestone 2: Training-time packing (single dataset, Fabric-safe)

What will exist at the end:

- Training schema supports `dataset.packing`.
- Training can consume the tokenized-docs dataset and produce fixed-length blocks for CLM.
- In packing mode:
  - data duplication across ranks is prevented by default in distributed runs (via `sampler_drop_last: true`);
  - Fabric does not auto-inject another sampler (`use_distributed_sampler=False`) for packing dataloaders;
  - sampler epoch is set each epoch for deterministic reshuffles when `shuffle: true`.

Config snippet to add (exact file content):

    task: clm_training
    experiment_name: test_clm_training_packing_smoke
    verbose_level: 1
    model_name: hf-internal-testing/tiny-random-LlamaForCausalLM
    precision: bf16-true
    seed: 42

    dataset:
      source: local
      format: hf
      nameOrPath: output/tests/tokenized_doclevel
      packing:
        enabled: true
        sequence_length: 128
        tokenizer_name: hf-internal-testing/llama-tokenizer
        # Optional packing parameters (defaults are applied in code, not by schema defaults):
        # insert_eos: true
        # shuffle: true
        # sampler_drop_last: true  (recommended for world_size > 1)

    validation_split:
      proportion: 0.1
      shuffle: true
      seed: 42

    number_epochs: 1
    batch_size: 1
    num_workers: 0
    validate_after_epoch: true
    validate_on_end: true
    validations_per_epoch: 1
    save_on_validate: false
    save_on_end: false
    output_dir: output/tests

    gradient_accumulation: false
    gradient_accumulation_steps: 1
    grad_clip: 1.0
    lr: 2.0e-05
    lr_decay: false
    weight_decay: 0.0
    beta1: 0.9
    beta2: 0.95

    lr_scheduler: fixed
    warmup_proportion: 0.0

    logging_config: none
    parallelization_strategy: dp

Acceptance:

- Validate:
    python src/main.py --validate --config config/tests/clm_training_packing_smoke.yaml

- Execute (after Milestone 1 output exists):
    python src/main.py --config config/tests/clm_training_packing_smoke.yaml

- Observability requirement: Add a debug log in the trainer (guarded by `verbose_level >= 4`) that prints:
  - packing enabled, sequence_length, eos_token_id, sampler type, sampler drop_last, and the first batch tensor shapes.
  This gives a junior a concrete “did packing actually happen?” signal.

### Milestone 3: Tests + SLURM validation (including multi-node)

What will exist at the end:

- Unit tests for packing correctness and distributed-sampler policy.
- A `task: testing` integration config that runs doc-level tokenization then packing training locally (single process).
- A multi-node smoke config for packing under `config/tests/` and explicit SLURM validation steps.

Integration config snippet to add (exact file content):

    task: testing
    experiment_name: test_online_packing_integration_smoke
    verbose_level: 1
    testing:
      mode: integration
      configs:
        - config/tests/tokenization_doclevel_smoke.yaml
        - config/tests/clm_training_packing_smoke.yaml
      stop_on_failure: true

Multi-node config snippet to add (exact file content, requires cluster allocation):

    task: clm_training
    experiment_name: test_clm_training_packing_multinode_smoke
    verbose_level: 1
    model_name: hf-internal-testing/tiny-random-LlamaForCausalLM
    precision: bf16-true
    seed: 42

    dataset:
      source: local
      format: hf
      nameOrPath: output/tests/tokenized_doclevel
      packing:
        enabled: true
        sequence_length: 128
        insert_eos: true
        tokenizer_name: hf-internal-testing/llama-tokenizer
        shuffle: true
        sampler_drop_last: true

    validation_split:
      proportion: 0.1
      shuffle: true
      seed: 42

    number_epochs: 1
    batch_size: 1
    num_nodes: 2
    devices_per_node: 1
    num_workers: 0
    validate_after_epoch: true
    validate_on_end: true
    validations_per_epoch: 1
    save_on_validate: false
    save_on_end: false
    output_dir: output/tests

    gradient_accumulation: false
    gradient_accumulation_steps: 1
    grad_clip: 1.0
    lr: 2.0e-05
    lr_decay: false
    weight_decay: 0.0
    beta1: 0.9
    beta2: 0.95

    lr_scheduler: fixed
    warmup_proportion: 0.0

    logging_config: none
    parallelization_strategy: ddp
    parallelization_config:
      backend: "nccl"

SLURM validation steps (junior-proof):

1) Submit doc-level tokenization smoke (single-node is fine):

    ./slurm/tests/run_tests.sh --config config/tests/tokenization_doclevel_smoke.yaml

   Record the printed job ID and log paths in issue 42.
   Important: the tokenized output path (`output/tests/tokenized_doclevel`) must be on a filesystem shared across nodes. If your cluster uses per-node local scratch, point `output.path` to a shared location instead, or the multi-node training job will fail to load the dataset.

2) After that job completes successfully, submit packing multi-node smoke:

    ./slurm/tests/run_tests.sh --config config/tests/clm_training_packing_multinode_smoke.yaml --nodes 2 --ntasks-per-node 1

   Record the printed job ID and log paths in issue 42. Confirm completion with:

    sacct -j <job_id> --format=JobID,State,ExitCode -P

Acceptance:

- `python -m pytest -q` passes unit tests locally when the environment is available.
- SLURM jobs complete with ExitCode `0:0` and logs show correct rank/world size and packing enabled.

### Milestone 4: Large-dataset support (index + policy clarity)

What will exist at the end:

- Doc-level tokenization outputs an additional `ends_with_eos: bool` column (CLM doc-level mode only).
- Packing can reuse or build a persisted index stored alongside the tokenized dataset on disk (memory-mapped arrays).
- In SLURM/DDP, only rank 0 builds the index and other ranks wait on a barrier and then load it. If the dataset output path is not on a shared filesystem, the run fails fast with an actionable error.
- Drop policy is explicit:
  - `sampler_drop_last` controls rank-evenness and “no duplication across ranks”.
  - `drop_last_batch` controls whether to drop the final partial batch.

Acceptance:

- With a warm index, packing dataset initialization avoids scanning all rows and peak RAM does not scale with `num_docs`.
- Index invalidates and rebuilds when any of these change: dataset fingerprint, `sequence_length`, `insert_eos`, `eos_token_id`, or index version.

## Plan of Work

This plan follows TDD for the behavior changes: add focused unit tests first (fail), implement packing/tokenization until green, then refactor for clarity and performance.

### Milestone 1 implementation details (tokenization doc-level)

Edits:

1) `src/tasks/tokenization/orchestrator.py`
   - Today it errors if `context_length` cannot be resolved. Change behavior for `task == "clm_training"`:
     - If neither `max_sequence_length` nor `context_length` is present, treat this as doc-level tokenization mode and allow it.
     - In this mode, reject `overlap` if set and > 0 with a clear error message.
   - Pass `context_length=None` into `TokenizerConfig` for doc-level mode.

2) `src/tasks/tokenization/tokenizer/config.py`
   - Make `context_length` optional in the dataclass so it can represent doc-level tokenization.
   - Keep backward compatibility: existing paths that expect an int must validate and raise a clear error if they receive `None`.

3) `src/tasks/tokenization/tokenizer/causal.py`
   - Support two modes:
     - Fixed-length windowed (existing behavior): when `context_length` is an int.
     - Doc-level variable-length (new): when `context_length is None`.
   - In doc-level mode:
     - Do not call the tokenizer with `return_tensors="np"` (variable-length batches are not rectangular).
     - Call with `padding=False`, `truncation=False`, and without `return_overflowing_tokens`.
     - Return only `input_ids` and `length` in the map function output.
     - Define dataset features as variable-length `Sequence(Value("int32"))` for `input_ids` and `Value("int32")` for `length`.

4) `config/schemas/tokenization/tokenization.clm_training.schema.yaml`
   - Stop requiring `overlap` at schema-level. Validation for overlap becomes runtime-based:
     - If `context_length` is present (fixed-length mode), treat missing `overlap` as `0` and ensure we pass an integer stride to HF (avoid `None`).
     - If `context_length` is absent (doc-level mode), require overlap to be absent or `0`; any overlap > 0 fails fast.
   - Rationale: the base tokenization schema currently allows additional tokenizer fields without strict typing, so the safest place to enforce these invariants is in code + clear runtime errors.

### Milestone 2 implementation details (packing dataset + trainer integration)

New code organization:

- Use the existing `src/tasks/training/data/` directory as a small, training-owned package for dataset/loader utilities that are not specific to Fabric strategies.
  - Note: the directory may currently contain only stale `__pycache__` artifacts; it is still the canonical location for this feature.
  - Add `src/tasks/training/data/__init__.py` (so it is an importable package).
  - Add `src/tasks/training/data/packing.py`.

Interfaces to implement (in `src/tasks/training/data/packing.py`):

    class PackedSequenceDataset(torch.utils.data.Dataset):
        """
        Map-style dataset that exposes packed fixed-length blocks built from variable-length token sequences.

        Required input column: input_ids (list[int])
        Required input column: length (int) (must equal len(input_ids))
        """

        def __init__(self, hf_dataset, sequence_length: int, insert_eos: bool, eos_token_id: int | None):
            ...

        def __len__(self) -> int:
            """Number of fixed-length blocks available."""

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            """Return a dict with input_ids, attention_mask, labels as torch.long tensors."""

    def build_packing_dataloader(
        dataset: torch.utils.data.Dataset,
        *,
        split: str,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        sampler_drop_last: bool,
        seed: int | None,
        rank: int,
        world_size: int,
    ) -> torch.utils.data.DataLoader:
        """
        Build a DataLoader for packing mode with an explicit sampler policy.

        Required behaviors:
        - If world_size == 1:
          - Prefer an explicit sampler instead of DataLoader(shuffle=...) so epoch-level shuffling and resume are deterministic.
          - If `shuffle` is true (train split): use DistributedSampler(num_replicas=1, rank=0, shuffle=True, seed=seed or 0, drop_last=False) and DataLoader(shuffle=False).
          - If `shuffle` is false (non-train splits): use DataLoader(shuffle=False) and no sampler.
        - If world_size > 1:
          - Always use DistributedSampler(num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=sampler_drop_last, seed=seed or 0) for every split in packing mode.
          - Use DataLoader(shuffle=False) (shuffling is entirely controlled by the sampler).
          - Use DataLoader(drop_last=True) for every split when `sampler_drop_last` is true. This ensures every rank sees the same number of batches and avoids sampler padding/duplication.
          - Trade-off note: `sampler_drop_last=true` can drop a small tail of blocks in validation too; this is acceptable for smoke tests and the “no silent duplication across ranks” contract. If full validation coverage is required, set `sampler_drop_last=false` explicitly and accept that the sampler may pad/duplicate to make per-rank lengths even.
        """

Packing algorithm (must be documented in the module docstring for junior readability):

- Define the logical token stream:
  - If `insert_eos` is false: `doc_0 + doc_1 + doc_2 + ...` (simple concatenation).
  - If `insert_eos` is true: append `eos_token_id` after each document **unless** the document already ends with `eos_token_id`.
    - This avoids producing `... EOS, EOS ...` sequences when upstream data already contains EOS markers.
  - Important implementation constraint: do not materialize this full stream in memory. It is a conceptual definition only. `PackedSequenceDataset` must produce blocks on-demand via indexing (prefix sums + slicing) so memory use is O(number_of_docs) for offsets, not O(total_tokens).
- Define block `i` as the slice `[i * sequence_length : (i+1) * sequence_length]` from the stream.
- Always drop tail tokens that don’t fit into a full block (floor division for `__len__`).
- Implement `__getitem__` using a prefix-sum offset array built from doc lengths (and EOS insertions) and `bisect` to find the starting document for a block.
  - Definitions (for junior-proofness):
    - “Prefix-sum / offsets array” means a cumulative-length array where `offsets[i]` is the total token count up to (but not including) document `i` in the logical stream (after accounting for EOS insertions between documents).
    - “bisect” means binary-searching `offsets` to find which document contains a global token index `pos` (i.e., find the greatest `i` such that `offsets[i] <= pos`).

Worked example (for docstring and tests):

    docs:
      doc0 = [1, 2, 3]
      doc1 = [4, 5]
      doc2 = [6, 7, 8, 9]
    eos_token_id = 0
    sequence_length = 4
    token stream = [1, 2, 3, 0, 4, 5, 0, 6, 7, 8, 9, 0]
    blocks:
      block0 = [1, 2, 3, 0]
      block1 = [4, 5, 0, 6]
      block2 = [7, 8, 9, 0]

Trainer integration edits (in `src/tasks/training/fabric/trainer/base.py`):

1) Detect packing enabled:
   - Read `packing = config.dataset.get("packing", None)` and `packing.enabled`.
   - If packing is enabled, relax the “required columns” check: require only `input_ids` and `length` on the source HF dataset.
   - Fail-fast guard: if packing is enabled but the dataset appears already “offline packed” (e.g. fixed-length `input_ids` with existing `attention_mask/labels`), raise with remediation (“disable packing or point to doc-level tokenization output”).
     - Implementation hint (to avoid false positives): treat it as “offline packed” only if (a) `attention_mask` or `labels` columns exist, and (b) a small deterministic sample of rows has `len(input_ids) == packing.sequence_length` (or equals a consistent fixed length). If the sample shows variable lengths, do not block packing purely because extra columns exist.

2) Build packed datasets + dataloaders:
   - Create `PackedSequenceDataset` for each split in the DatasetDict.
   - Require a `length` column in packing mode:
     - If `length` is missing, raise a clear error (“packing requires doc-level tokenization output with a length field; re-run tokenization with size omitted”).
     - Validate `length == len(input_ids)` for a small deterministic sample (e.g. first 256 non-empty rows per split) and raise if any mismatch is found. This prevents subtle corruption where a stale length column causes out-of-bounds packing bugs later.
   - Resolve `eos_token_id`:
     - If `insert_eos` is true, require either `packing.eos_token_id` or `packing.tokenizer_name` (load tokenizer with `AutoTokenizer.from_pretrained` to get `.eos_token_id`).
     - If neither provided, raise a clear error (do not silently disable EOS insertion).
   - Explicit edge cases (must be implemented + tested):
     - Empty docs (`length == 0`) are skipped; rank 0 logs how many were skipped.
     - Docs longer than `sequence_length` are allowed and may span multiple blocks; EOS insertion only happens between docs.
     - v1 always drops tail tokens that cannot form a full block; rank 0 logs dropped tail tokens.
   - Keep the underlying HF dataset in its default Python format in packing mode (do not call `set_format(type="torch", ...)` on variable-length doc rows). `PackedSequenceDataset` is responsible for emitting torch tensors with fixed shapes.
   - Do not create rank-aware samplers before Fabric knows rank/world_size:
     - Create the packed *datasets* here, but create the *dataloaders* inside `_pipeline` (after `fabric.launch`) using `build_packing_dataloader(...)` so the sampler policy is centralized and unit-testable.
   - Packing config defaults (must be implemented in code because schema defaults are not applied):
     - If `packing.insert_eos` is missing: default to `True`.
     - If `packing.shuffle` is missing: default to `True` for the train split and `False` otherwise.
     - If `packing.sampler_drop_last` is missing: default to `True` when `world_size > 1` (and ignore it when `world_size == 1`).

3) Ensure Fabric does not override the sampler:
   - In `_pipeline`, when calling `fabric.setup_dataloaders`, pass `use_distributed_sampler=False` for dataloaders created in packing mode.
   - Leave existing behavior unchanged for non-packing dataloaders.

3.5) Packing dataloaders must be created in `_pipeline`:
   - Add a small helper in `FabricTrainerBase` like `_build_packing_dataloaders(self, fabric: L.Fabric) -> dict[str, DataLoader]` that:
     - reads `self.datasets` (already loaded and validation-split ensured),
     - wraps splits with `PackedSequenceDataset`,
     - constructs per-split dataloaders via `build_packing_dataloader(..., rank=fabric.global_rank, world_size=fabric.world_size)`.
   - Replace the existing “FABRIC DATALOADERS SETUP” line so it can handle both modes:
     - if packing enabled: build dataloaders here, then call `fabric.setup_dataloaders(dataloader, use_distributed_sampler=False)` for each one.
     - else: keep current behavior (`fabric.setup_dataloaders(dataloader)` with defaults).

4) Deterministic epoch shuffling:
   - At the start of every epoch, call `set_epoch(epoch)` on any `DistributedSampler` used by the training dataloader.
   - This applies to both packing and non-packing distributed runs.
   - In packing mode, this also applies when `world_size == 1` if `shuffle: true`, because we use a `DistributedSampler(num_replicas=1)` for deterministic epoch shuffles.
   - Implementation note: after `fabric.setup_dataloaders`, access the sampler via `self.dataloaders["train"].sampler` (or `self.dataloaders["train"].batch_sampler.sampler` if needed) and guard with `hasattr(sampler, "set_epoch")`.

4.5) Gradient accumulation safety (packing mode):
   - Current `FabricTrainerBase._accumulate_training` does not flush a partial gradient-accumulation group at end-of-epoch. This is easy to miss and leads to silent “lost” optimizer steps.
   - Fail-fast guard (packing mode only, to keep blast radius small): if `gradient_accumulation_steps > 1`, assert `len(self.dataloaders["train"]) % gradient_accumulation_steps == 0`. If not, raise a clear error instructing the user to adjust `batch_size`, `sampler_drop_last`, or `gradient_accumulation_steps`.

5) Scheduler correctness:
   - Build the scheduler using the effective number of batches the training loop will actually execute.
     - Rationale: the current scheduler helper (`src/tasks/training/utils.py:select_scheduler`) computes steps via floor division on dataset length. This can produce `total_steps == 0` for small datasets when `drop_last=False` (DataLoader still yields 1 batch), and in general can diverge from the true number of batches when drop policies change.
   - After dataloaders are created and (if applicable) prepared by Fabric, compute:
     - `train_dataloader = self.dataloaders["train"]`
     - `train_num_batches = len(train_dataloader)` (PyTorch defines this deterministically for map-style datasets)
     - `gradient_accumulation_steps = int(self.config.get("gradient_accumulation_steps", 1) or 1)`
     - `optimizer_steps_per_epoch = train_num_batches // gradient_accumulation_steps` (packing mode already enforces divisibility; see 4.5)
     - `total_optimizer_steps = optimizer_steps_per_epoch * int(self.config.number_epochs)`
   - Update `src/tasks/training/utils.py:select_scheduler` to accept an explicit `total_steps` (or `steps_per_epoch`) override so the schedule matches `total_optimizer_steps` exactly.
     - Keep backward compatibility: if the override is not provided, fall back to the current dataset-length-based computation.
   - This fixes both:
     - packing mode (scheduler uses packed-block length), and
     - existing non-packing behavior where `train_data_ratio` / validation-split mutations can make `self.dataset["train"]` disagree with what the dataloader yields.
   - Fail-fast guard: after dataloaders are built, assert `len(self.dataloaders["train"]) > 0` (number of batches). If it is 0, raise a clear error instructing the user to reduce `batch_size`, disable `sampler_drop_last`, or use more data. This prevents downstream schedulers (e.g. cosine) from receiving `total_steps=0` and erroring in confusing ways.
   - Intra-epoch validation/checkpoint scheduling correctness: when packing is enabled, update any “steps_per_epoch” calculations used for validation/checkpoint cadence (see `FabricTrainerBase._try_validate`) to be derived from the effective train dataloader (or its dataset), not the pre-packing HF dataset. Otherwise, `validations_per_epoch` / `checkpoints_per_epoch` will be scheduled incorrectly under packing.
   - Add unit tests:
     - `test_scheduler_steps_use_dataloader_len_not_floor_div_dataset_len` (construct a tiny dataset where `len(dataset) < batch_size * world_size` and `drop_last=False`; assert total_steps is non-zero and matches `len(train_dataloader)`).
     - Keep the existing intent: also assert the trainer does not size the scheduler from the pre-packing HF dataset (`self.dataset["train"]`) when packing is enabled.

### Milestone 3 implementation details (tests, configs, docs)

Unit tests (in `tests/` because this is cross-cutting between tokenization + training):

- Add `tests/test_packed_sequence_dataset.py` with:
  - `test_packing_inserts_eos_and_blocks_are_fixed_length`
  - `test_len_drops_remainder_tokens_by_default`
  - `test_distributed_sampler_drop_last_is_enforced_in_packing_mode` (unit-test `build_packing_dataloader` by passing `rank/world_size` explicitly)
  - `test_empty_docs_are_skipped` (assert no blocks are produced from empty rows)
  - `test_scheduler_steps_use_dataloader_len_not_floor_div_dataset_len` (assert scheduler sizing uses dataloader length / optimizer steps, avoiding `total_steps == 0` on small datasets)

YAML configs (in `config/tests/`):

- Add:
  - `config/tests/tokenization_doclevel_smoke.yaml`
  - `config/tests/clm_training_packing_smoke.yaml`
  - `config/tests/online_packing_integration_smoke.yaml`
  - `config/tests/clm_training_packing_multinode_smoke.yaml`

Docs:

- Update:
  - `docs/TOKENIZATION.md` with a “doc-level tokenization” section and a warning about overlap being invalid when size is omitted.
  - `docs/CLM_TRAINING.md` with a “packing” section that explains the sampler/drop trade-offs in distributed runs.

### Milestone 4 implementation details (large datasets: index + drop_last)

Goal:

Make online packing viable for large corpora by avoiding full dataset scans at startup and bounding RAM usage with memory-mapped arrays.

Edits:

1) Doc-level tokenization emits `ends_with_eos`:
   - In `src/tasks/tokenization/tokenizer/causal.py`, doc-level mode only:
     - Add `ends_with_eos: Value("bool")` (or `Value("int8")` if bool is unsupported in your datasets version) to the doc-level `Features`.
     - Compute `ends_with_eos` in `_tokenize_doc_function` using the tokenizer’s EOS id:
       - hard error if EOS id cannot be resolved (doc-level packing depends on this signal when `insert_eos` is enabled).

2) Persisted packing index:
   - Add `src/tasks/training/data/packing_index.py` implementing:
     - a `PackingIndexMeta` struct saved as JSON with: index version, dataset fingerprint (or dataset path + split sizes as a fallback), split name, `sequence_length`, `insert_eos`, `eos_token_id`.
     - a `PackingIndex.load_or_build(...)` that:
       - checks for an existing matching index under `index_cache_dir`,
       - otherwise builds and writes `offsets`, `doc_indices`, `doc_lengths`, and `doc_stream_lengths` as `numpy.memmap`,
       - writes `meta.json`,
       - uses a simple file lock to prevent concurrent builders.
   - In DDP/SLURM: build index on rank 0 only, then `fabric.barrier()`, then load on all ranks. If multi-node and the index cache dir is not shared, fail fast with an error that instructs the user to use a shared path.

3) Use the index in packing:
   - Refactor `src/tasks/training/data/packing.py:PackedSequenceDataset` to:
     - accept an optional `PackingIndex`,
     - use memmapped arrays instead of Python lists when present.
   - Keep the in-memory index path as a fallback for small datasets (or when caching is disabled).

4) Decouple drop policies:
   - Extend `dataset.packing` with `drop_last_batch` (optional).
   - In `src/tasks/training/data/packing.py:build_packing_dataloader`:
     - keep `sampler_drop_last` controlling `DistributedSampler(drop_last=...)`,
     - set `DataLoader(drop_last=drop_last_batch)` instead of tying it to `sampler_drop_last`.

Schema updates:

- In `config/schemas/training/components/data.schema.yaml` under `dataset.packing`, add:
  - `index_cache_dir` (string or null): where to store/reuse the persisted index (default in code: `<dataset_path>/.packing_index`).
  - `drop_last_batch` (bool or null): whether to drop the final partial batch (default in code: train=true for distributed, valid=false).

Tests (requires `datasets` installed):

- Add unit tests to lock:
  - index build + reuse (rebuild only when params/fingerprint mismatch),
  - memmap-backed packing output equivalence with in-memory packing on a small dataset,
  - `drop_last_batch` affects `len(dataloader)` without changing sampler sharding decisions.

## Concrete Steps

All commands below run from the repository root.

If local Python deps are missing, first activate the repo’s conda environment:

    source scripts/set_environment.sh

Milestone 1 validation:

    python src/main.py --validate --config config/tests/tokenization_doclevel_smoke.yaml
    python src/main.py --config config/tests/tokenization_doclevel_smoke.yaml

Milestone 2 validation:

    python src/main.py --validate --config config/tests/clm_training_packing_smoke.yaml
    python src/main.py --config config/tests/clm_training_packing_smoke.yaml

Milestone 3 validation:

    python -m tox -e py310
    python src/main.py --config config/tests/online_packing_integration_smoke.yaml

SLURM validation (preferred for HPC correctness, run by allowed submitter per `slurm/tests/slurm_test.env`):

    ./slurm/tests/run_tests.sh --config config/tests/tokenization_doclevel_smoke.yaml
    ./slurm/tests/run_tests.sh --config config/tests/clm_training_packing_multinode_smoke.yaml --nodes 2 --ntasks-per-node 1

## Validation and Acceptance

Acceptance is behavioral and must be observable in logs:

- Doc-level tokenization:
  - Output dataset saved to `output/tests/tokenized_doclevel`.
  - `input_ids` are variable-length and `length` equals `len(input_ids)` for sampled rows.
  - Any attempt to set `overlap > 0` in this mode fails fast with a clear error.

- Packing training:
  - Training runs without requiring `attention_mask` or `labels` columns in the source dataset when packing is enabled.
  - The first logged batch shows shapes `[batch_size, sequence_length]`.
  - In distributed runs, logs show the configured rank/world size and that `sampler_drop_last` is enabled (or explicitly disabled by config).

- Automated:
  - `python -m tox -e py310` passes.
  - SLURM jobs complete with ExitCode `0:0`, and job IDs/log paths are recorded in issue 42.

## Idempotence and Recovery

- All new behavior is opt-in. If anything breaks, set `dataset.packing.enabled: false` (default) and the existing pipeline should behave unchanged.
- If doc-level tokenization is too large for a dataset, specify `context_length` again and revert to fixed-length tokenization.
- If SLURM multi-node validation is unavailable (no allocation), document why and validate single-node `dp` plus at least one multi-process `ddp` run when resources return.

## Artifacts and Notes

Keep evidence in issue 42 and optionally paste key snippets here as work proceeds:

    SLURM job ID: 123456
    Logs: slurm/tests/logs/tests-123456.out
    Observed: packing enabled, sequence_length=128
    Observed: first batch shape: torch.Size([1, 128])
    Exit: COMPLETED|0:0

## Interfaces and Dependencies

Dependencies:

- Hugging Face `datasets` for loading tokenized-doc datasets from disk.
- PyTorch `Dataset`, `DataLoader`, and samplers.
- Lightning Fabric (already in repo) for distributed training.

New interfaces that must exist:

In `src/tasks/training/data/packing.py`:

    class PackedSequenceDataset(torch.utils.data.Dataset):
        def __len__(self) -> int: ...
        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]: ...

    def build_packing_dataloader(...) -> torch.utils.data.DataLoader: ...

In `config/schemas/training/components/data.schema.yaml`, under `dataset`, define a `packing` object with:

- `enabled` (bool)
- `sequence_length` (int, required if enabled)
- `insert_eos` (bool or null; default in code is true)
- `tokenizer_name` (string; required at runtime if insert_eos true unless eos_token_id provided)
- `eos_token_id` (int or null)
- `shuffle` (bool or null; default in code is true for train)
- `sampler_drop_last` (bool or null; default in code is true when world_size > 1)
- `drop_last_batch` (bool or null; default in code: train=true for distributed, valid=false)
- `index_cache_dir` (string or null; default in code: `<dataset_path>/.packing_index`)

## LMTK Project Patterns to Embed in ExecPlans

LMTK workflows are YAML-driven. Each YAML configuration denotes a single job and maps to exactly one task module under `src/tasks/`. This plan adds opt-in behavior to existing tasks and validates through configs under `config/tests/`. Integration validation should be driven by a `task: testing` config and submitted through `slurm/tests/run_tests.sh` so results are auditable (job ID + log paths recorded in the issue card). Configs should be reviewed before SLURM submission and must not auto-submit or bypass validation.

Plan update note (2026-01-28): Reworked the plan to be Fabric- and HPC-correct by (1) choosing a map-style packing dataset for stable `__len__` and deterministic indexing, (2) explicitly controlling distributed sampling to avoid silent duplication (`drop_last` policy) and distributed hangs, and (3) correcting execution commands to `python src/main.py` to match current task import behavior.

Plan update note (2026-01-28): Tightened the spec to be more junior-proof by (1) explicitly scoping v1 to “drop remainder tokens” (no partial-block padding semantics), (2) removing the `drop_remainder_tokens` knob from the proposed public config, (3) making overlap/default-stride and sampler seeding requirements explicit, and (4) explicitly scoping `set_epoch` + scheduler sizing fixes needed for correct DDP/HPC behavior.

Plan update note (2026-01-28): Clarified config ergonomics scope: only the new `dataset.packing` surface gets code defaults in this plan; repo-wide training-config simplification is intentionally out-of-scope because schema defaults are not currently applied by `ConfigValidator`.

Plan update note (2026-02-05): Marked Milestones 1–2 as implemented, recorded the sandbox test limitation (missing `datasets`), and added a v1 guard to fail fast if `dataset.packing` is enabled for non-CLM tasks.

Plan update note (2026-02-05): Added Milestone 4 for large-dataset support (persisted memmap packing index + decoupled drop_last policy) to keep online packing viable on HPC-scale corpora.
