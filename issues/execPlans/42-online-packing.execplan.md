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
- [ ] Implement Milestone 1 (doc-level CLM tokenization) with tests and schema updates.
- [ ] Implement Milestone 2 (training-time packing) with deterministic sampler integration and resume-safe behavior.
- [ ] Implement Milestone 3 (tests + SLURM validation) and record job/log evidence in issue 42.

## Surprises & Discoveries

- Observation: `src/main.py` must be executed as a script (`python src/main.py ...`) because it imports task modules as `tasks.<name>`, which relies on Python adding `src/` (the script directory) to `sys.path`.
  Evidence: `src/main.py` uses `__import__(f"tasks.{module_name}", ...)`.

- Observation: Lightning Fabric only injects a distributed sampler for **map-style datasets**. For `IterableDataset`, Fabric will not auto-replace the sampler.
  Evidence: `lightning/fabric/fabric.py:_requires_distributed_sampler` returns `False` when `has_iterable_dataset(dataloader)` is true.

- Observation: Lightning Fabric’s default distributed sampler kwargs do not set `drop_last`, meaning PyTorch’s `DistributedSampler(drop_last=False)` behavior can **pad and repeat samples** when `len(dataset)` is not divisible by `world_size`.
  Evidence: `lightning/fabric/strategies/parallel.py:ParallelStrategy.distributed_sampler_kwargs` returns only `{"num_replicas": ..., "rank": ...}`; `lightning/fabric/fabric.py:_get_distributed_sampler` does not set `drop_last`.

These observations drive the design choices in this plan: we implement packing as a map-style dataset for deterministic sizing and we take control of distributed sampling (including `drop_last`) to avoid silent duplication and distributed hangs.

## Decision Log

- Decision: Implement online packing as a **map-style** `torch.utils.data.Dataset`, not an `IterableDataset`.
  Rationale: Map-style datasets work well with deterministic indexing, have a stable `__len__`, work with PyTorch samplers cleanly, and are easier to resume (the current trainer resumes by skipping batches in a dataloader iterator).
  Date/Author: 2026-01-28 / Codex

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

- Decision: Only improve training-config ergonomics for the new `dataset.packing` surface in this plan (safe code defaults); do not attempt a repo-wide training-config simplification.
  Rationale: `src/config/config_loader.ConfigValidator` validates but does not apply JSON-schema defaults. Broad “make fields optional” changes would require a separate config-system plan to safely inject defaults across the entire training configuration.
  Date/Author: 2026-01-28 / Codex

## Outcomes & Retrospective

Not started.

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

- Create `src/tasks/training/data/` as a small, training-owned package for dataset/loader utilities that are not specific to Fabric strategies.
  - `src/tasks/training/data/__init__.py`
  - `src/tasks/training/data/packing.py`

Interfaces to implement (in `src/tasks/training/data/packing.py`):

    def get_distributed_rank_info() -> tuple[int, int]:
        """
        Return (rank, world_size) in a way compatible with SLURM + srun launches.

        Prefer:
          - SLURM_PROCID / SLURM_NTASKS
        Fallback:
          - RANK / WORLD_SIZE
        Default:
          - (0, 1)
        """

    class PackedSequenceDataset(torch.utils.data.Dataset):
        """
        Map-style dataset that exposes packed fixed-length blocks built from variable-length token sequences.

        Required input column: input_ids (list[int])
        Optional input column: length (int) to avoid recomputing len(input_ids)
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
    ) -> torch.utils.data.DataLoader:
        """
        Build a DataLoader for packing mode with an explicit sampler policy.

        Required behaviors:
        - If world_size == 1: use DataLoader(shuffle=shuffle for train, else False).
        - If world_size > 1: use DistributedSampler(drop_last=sampler_drop_last, seed=seed or 0),
          DataLoader(shuffle=False), and drop_last=True for the train split.
        """

Packing algorithm (must be documented in the module docstring for junior readability):

- Define the logical token stream as: doc_0 + [EOS] + doc_1 + [EOS] + ... when insert_eos is enabled.
- Define block `i` as the slice `[i * sequence_length : (i+1) * sequence_length]` from the stream.
- Always drop tail tokens that don’t fit into a full block (floor division for `__len__`).
- Implement `__getitem__` using a prefix-sum offset array built from doc lengths (and EOS insertions) and `bisect` to find the starting document for a block.

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
   - If packing is enabled, relax the “required columns” check: require only `input_ids` (and optionally `length`) on the source HF dataset.

2) Build packed datasets + dataloaders:
   - Create `PackedSequenceDataset` for each split in the DatasetDict.
   - Resolve `eos_token_id`:
     - If `insert_eos` is true, require either `packing.eos_token_id` or `packing.tokenizer_name` (load tokenizer with `AutoTokenizer.from_pretrained` to get `.eos_token_id`).
     - If neither provided, raise a clear error (do not silently disable EOS insertion).
   - Keep the underlying HF dataset in its default Python format in packing mode (do not call `set_format(type="torch", ...)` on variable-length doc rows). `PackedSequenceDataset` is responsible for emitting torch tensors with fixed shapes.
   - Create DataLoaders using `build_packing_dataloader(...)` so the sampler policy is centralized and unit-testable.
   - Packing config defaults (must be implemented in code because schema defaults are not applied):
     - If `packing.insert_eos` is missing: default to `True`.
     - If `packing.shuffle` is missing: default to `True` for the train split and `False` otherwise.
     - If `packing.sampler_drop_last` is missing: default to `True` when `world_size > 1` (and ignore it when `world_size == 1`).

3) Ensure Fabric does not override the sampler:
   - In `_pipeline`, when calling `fabric.setup_dataloaders`, pass `use_distributed_sampler=False` for dataloaders created in packing mode.
   - Leave existing behavior unchanged for non-packing dataloaders.

4) Deterministic epoch shuffling:
   - At the start of every epoch, call `set_epoch(epoch)` on any `DistributedSampler` used by the training dataloader.
   - This applies to both packing and non-packing distributed runs.
   - Implementation note: after `fabric.setup_dataloaders`, access the sampler via `self.dataloaders["train"].sampler` (or `self.dataloaders["train"].batch_sampler.sampler` if needed) and guard with `hasattr(sampler, "set_epoch")`.

5) Scheduler correctness:
   - Build the scheduler using the dataset that backs the (already Fabric-prepared) training dataloader:
     - use `train_dataset_for_scheduler = self.dataloaders["train"].dataset`
   - This fixes both:
     - packing mode (scheduler uses packed-block length), and
     - existing non-packing behavior where `train_data_ratio` / validation-split mutations can make `self.dataset["train"]` disagree with what the dataloader yields.
   - Add a unit test that monkeypatches `select_scheduler` and asserts the trainer passes `self.dataloaders["train"].dataset` (not `self.dataset["train"]`).

### Milestone 3 implementation details (tests, configs, docs)

Unit tests (in `tests/` because this is cross-cutting between tokenization + training):

- Add `tests/test_packed_sequence_dataset.py` with:
  - `test_packing_inserts_eos_and_blocks_are_fixed_length`
  - `test_len_drops_remainder_tokens_by_default`
  - `test_distributed_sampler_drop_last_is_enforced_in_packing_mode` (unit-test `build_packing_dataloader` without running distributed by setting env vars for rank/world size)

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

    python -m pytest -q
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
  - `pytest` passes.
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

    def get_distributed_rank_info() -> tuple[int, int]: ...

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

## LMTK Project Patterns to Embed in ExecPlans

LMTK workflows are YAML-driven. Each YAML configuration denotes a single job and maps to exactly one task module under `src/tasks/`. This plan adds opt-in behavior to existing tasks and validates through configs under `config/tests/`. Integration validation should be driven by a `task: testing` config and submitted through `slurm/tests/run_tests.sh` so results are auditable (job ID + log paths recorded in the issue card). Configs should be reviewed before SLURM submission and must not auto-submit or bypass validation.

Plan update note (2026-01-28): Reworked the plan to be Fabric- and HPC-correct by (1) choosing a map-style packing dataset for stable `__len__` and deterministic indexing, (2) explicitly controlling distributed sampling to avoid silent duplication (`drop_last` policy) and distributed hangs, and (3) correcting execution commands to `python src/main.py` to match current task import behavior.

Plan update note (2026-01-28): Tightened the spec to be more junior-proof by (1) explicitly scoping v1 to “drop remainder tokens” (no partial-block padding semantics), (2) removing the `drop_remainder_tokens` knob from the proposed public config, (3) making overlap/default-stride and sampler seeding requirements explicit, and (4) explicitly scoping `set_epoch` + scheduler sizing fixes needed for correct DDP/HPC behavior.

Plan update note (2026-01-28): Clarified config ergonomics scope: only the new `dataset.packing` surface gets code defaults in this plan; repo-wide training-config simplification is intentionally out-of-scope because schema defaults are not currently applied by `ConfigValidator`.
