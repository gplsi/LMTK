# Token-budget multi-dataset mixing + replay (homogeneous packed blocks)

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

PLANS.md is checked into the repo at `PLANS.md`, and this document must be maintained in accordance with it.

## Purpose / Big Picture

Enable `task: clm_training` to train from multiple doc-level tokenized sources with deterministic block-level mixing and replay, while preserving issue-42 guarantees:

- each packed block belongs to exactly one source,
- distributed execution avoids silent duplication/padding side effects,
- accounting is persisted and auditable.

After this change, a novice can run one config and get:

1. multi-source packed training from `dataset.sources`,
2. deterministic realized allocation over a resolved block budget (`anchor_epochs` default or `explicit_blocks`),
3. a rank-0 `mixture_report.json` that proves what was actually trained.

## Progress

- [ ] (2026-02-06 00:00Z) Add schema support for `dataset.sources` and `dataset.mixture`.
- [ ] (2026-02-06 00:00Z) Add red tests for deterministic allocation and distributed invariants.
- [ ] (2026-02-06 00:00Z) Implement mixture dataset and trainer integration.
- [ ] (2026-02-06 00:00Z) Implement per-source compatibility checks and fail-fast guards.
- [ ] (2026-02-06 00:00Z) Implement rank-safe accounting and `mixture_report.json`.
- [ ] (2026-02-06 00:00Z) Add smoke configs and integration testing config.
- [ ] (2026-02-06 00:00Z) Run local validation + SLURM smoke; record job IDs/logs in issue 43.

## Surprises & Discoveries

- Observation: current trainer validation cadence is primarily epoch-driven (`validations_per_epoch`, `checkpoints_per_epoch`), and `_try_validate` also supports `validate_after_k_steps` for global-step triggers.
  Evidence: `_try_validate` logic in `src/tasks/training/fabric/trainer/base.py`.

- Observation: `src/tasks/training/orchestrator.py` currently loads exactly one source via `dataset.nameOrPath`.
  Evidence: `load_dataset` in `src/tasks/training/orchestrator.py`.

## Decision Log

- Decision: v1 mixing stays map-style and block-level only; no token-level mixed blocks.
  Rationale: keeps determinism, accounting, and distributed correctness simple and auditable.
  Date/Author: 2026-02-06 / Codex

- Decision: enforce explicit divisibility/step-budget fail-fast instead of silent correction.
  Rationale: silent drops/padding would make reported ratios untrustworthy.
  Date/Author: 2026-02-06 / Codex

- Decision: v1 budget interface supports two modes: `anchor_epochs` (default) and `explicit_blocks`; both resolve to `requested_total_blocks` then `effective_total_blocks`.
  Rationale: preserves user ergonomics for “train one anchor epoch” while keeping strict token-budget accounting.
  Date/Author: 2026-02-06 / Codex

- Decision: in v1, validation datasets in mixture mode are auto-generated per source independently (when missing), not from a mixed stream.
  Rationale: avoids source leakage and gives source-attributed validation metrics before dedicated validation corpora exist.
  Date/Author: 2026-02-06 / Codex

- Decision: v1 determinism uses explicit algorithms: `blake2b_u64` for hashing and Hamilton largest-remainder for exact allocation with lexicographic tie-break.
  Rationale: prevents runtime/process-dependent behavior and removes implementation ambiguity for junior contributors.
  Date/Author: 2026-02-06 / Codex

- Decision: `anchor_epochs` uses canonical anchor-fixed math (`target_anchor_per_epoch = B_anchor`; non-anchor counts derived from `B_anchor * w_j / w_anchor`, then Hamilton with explicit half-up total).
  Rationale: removes ambiguity about epoch budget derivation while preserving intended replay proportions.
  Date/Author: 2026-02-06 / Codex

- Decision: validation cadence for mixture runs stays epoch-based (`validations_per_epoch`, `checkpoints_per_epoch`) in v1.
  Rationale: aligns with current trainer behavior and avoids introducing a second cadence mechanism in this issue.
  Date/Author: 2026-02-06 / Codex

## Outcomes & Retrospective

- Pending implementation.

## Context and Orientation

LMTK is YAML-driven. `src/main.py` validates configs with `src/config/config_loader.py` and dispatches to task modules under `src/tasks/`.

Current relevant behavior:

- issue 42 already provides doc-level CLM tokenization output (`input_ids`, `length`, `ends_with_eos`) and training-time packing primitives:
  - `src/tasks/training/data/packing.py`
  - `src/tasks/training/data/packing_index.py`
- `src/tasks/training/orchestrator.py` currently loads one dataset path (`dataset.nameOrPath`) and passes it into Fabric trainers.
- `src/tasks/training/fabric/trainer/base.py` supports single-source packing and explicit distributed sampler policies.

This plan extends that existing path. It does not add new tasks or new SLURM submission logic.

Terms used in this plan:

- packed block: one fixed-length training sample of size `sequence_length`,
- training batch: `batch_size` packed blocks per rank (global optimizer-step batch also depends on `world_size` and `gradient_accumulation_steps`),
- homogeneous block: packed block attributable to one dataset source only,
- virtual mixture epoch: accounting unit used by `budget_mode=anchor_epochs` to derive `requested_total_blocks`; this is not trainer `number_epochs`,
- effective_total_blocks: actual executable block budget after enforcing distributed and optimization-step invariants.

## Interfaces and Dependencies

At completion, the following must exist:

1. Schema surface in `config/schemas/training/components/data.schema.yaml`:
   - support either:
     - single-source (`dataset.nameOrPath`), or
     - multi-source (`dataset.sources` + `dataset.mixture`).
   - `dataset.sources` item fields:
     - `dataset_id: string` (required, unique),
     - `nameOrPath: string` (required),
     - `weight: number` (required, `> 0`),
     - `tokenizer_name: string | null` (optional compatibility metadata),
     - `eos_token_id: integer | null` (optional compatibility metadata),
     - `index_cache_dir: string | null` (optional).
   - `dataset.mixture` fields:
     - `enabled: boolean` (required when `sources` exists),
     - `budget_mode: string` with enum `anchor_epochs` / `explicit_blocks` (default `anchor_epochs`),
     - `anchor_epochs: integer >= 1` (required when `budget_mode=anchor_epochs`),
     - `anchor_dataset_id: string | null` (optional; default selects source with max block capacity),
     - `total_blocks: integer >= 1` (required when `budget_mode=explicit_blocks`),
     - `schedule_seed: integer | null` (optional),
     - `report_path: string | null` (optional; default `<output_dir>/mixture_report.json`),
     - v1 validation behavior is fixed: per-source independent auto-split when `valid` is missing.

2. Runtime dataset in `src/tasks/training/data/mixture.py`:
   - `class MixturePackedDataset(torch.utils.data.Dataset)`,
   - `__len__ -> effective_total_blocks`,
   - `__getitem__(i)` returns `input_ids`, `attention_mask`, `labels`, `dataset_idx` (`int64` tensor),
   - deterministic `dataset_idx <-> dataset_id` mapping is stored in report/checkpoint metadata.

3. Trainer integration in `src/tasks/training/fabric/trainer/base.py`:
   - detect mixture mode from config,
   - build per-source packed datasets and compose with `MixturePackedDataset`,
   - resolve requested budget from selected budget mode (`anchor_epochs` or `explicit_blocks`),
   - resolve default anchor as source with max **train-split** packed blocks when not specified,
   - auto-build per-source validation splits independently before mixture composition, reusing `FabricTrainerBase._ensure_validation_split` semantics per source (no duplicate splitter),
   - enforce `number_epochs == 1` in mixture mode (hard error otherwise),
   - keep existing epoch-driven validation/checkpoint cadence (`validations_per_epoch`, `checkpoints_per_epoch`) over this single trainer epoch,
   - enforce `effective_total_blocks == requested_total_blocks` in v1 (hard error otherwise),
   - emit `metric/val_loss_<dataset_id>` and `metric/val_loss_weighted`,
   - enforce non-dropping validation dataloader policy and fail fast if any source validation loader has zero batches,
   - enforce resume-compatibility checks for accounting-critical mixture metadata,
   - enforce distributed and step-budget invariants with hard errors,
   - aggregate per-source counters across ranks and write `mixture_report.json` on rank 0.

4. Orchestrator loading updates in `src/tasks/training/orchestrator.py`:
   - support `dataset.sources` loading path (DatasetDict per source) while preserving existing single-source behavior.

## Milestones

### Milestone 1: Schema and config surface

Add schema support in `config/schemas/training/components/data.schema.yaml` for `dataset.sources` and `dataset.mixture`, while preserving existing single-source configs.

At the end of this milestone, these commands succeed:

    python src/main.py --validate --config config/tests/clm_training_packing_smoke.yaml
    python src/main.py --validate --config config/tests/clm_training_packing_mixture_smoke.yaml

Acceptance:

- single-source training configs still validate unchanged,
- invalid mixture configs fail with explicit messages (missing `sources`, duplicate `dataset_id`, non-positive `weight`, invalid budget-mode conditionals such as missing `anchor_epochs` or missing `total_blocks` when required).

### Milestone 2: Red tests for mixture semantics

Before implementation, add failing tests in `tests/unit/training/test_mixture_packing.py` for:

- deterministic exact allocation (`weights -> target_blocks`) with deterministic remainder handling,
- no `O(total_blocks)` schedule materialization in dataset state,
- deterministic mapping for fixed seed,
- deterministic mapping is stable across processes (no Python built-in `hash` dependency),
- Hamilton largest-remainder allocation with lexicographic tie-break reproducibility,
- canonical anchor-fixed per-epoch target derivation in `anchor_epochs`,
- budget resolution for `anchor_epochs` and `explicit_blocks`,
- default anchor selection (`argmax(source_blocks)`), plus explicit `anchor_dataset_id` override,
- fail-fast when mixture mode uses `number_epochs != 1`,
- fail-fast when `effective_total_blocks != requested_total_blocks` in v1 contract,
- distributed invariant failures (non-divisible budgets),
- compatibility failures for mismatched tokenizer/eos metadata,
- per-source deterministic validation auto-split construction,
- fail-fast when source lacks both `valid` and usable `validation_split`,
- fail-fast when any source validation dataloader has zero batches,
- fail-fast on resume metadata mismatch for accounting-critical fields,
- validation metrics emission (`val_loss_<dataset_id>` and `val_loss_weighted`).

Acceptance:

- tests fail initially for missing implementation (red state is explicit and expected).
- the test location remains discoverable and scoped: these tests stay under `tests/unit/training/` because they cross schema, orchestrator, and trainer boundaries; any helper classes specific to `MixturePackedDataset` should be colocated under `src/tasks/training/data/` when needed.

### Milestone 3: Implement mixture runtime

Implement:

- `src/tasks/training/data/mixture.py`,
- orchestration hooks in `src/tasks/training/orchestrator.py`,
- trainer integration and accounting in `src/tasks/training/fabric/trainer/base.py`.

Required runtime behavior:

- exact deterministic target allocation per source over resolved budget,
- local block selection with replacement: `local_idx = blake2b_u64(f"{schedule_seed}|{global_idx}|{dataset_id}") % source_num_blocks`,
- no token mixing inside blocks,
- in `anchor_epochs` mode, use canonical anchor-fixed derivation:
  - `target_anchor_per_epoch = B_anchor`,
  - `raw_non_anchor_j = B_anchor * w_j / w_anchor`,
  - `total_non_anchor = floor(sum(raw_non_anchor_j) + 0.5)` (explicit half-up),
  - Hamilton allocate non-anchor counts constrained to `total_non_anchor`,
  - `requested_total_blocks = anchor_epochs * (target_anchor_per_epoch + sum(target_non_anchor_j))`,
- in `explicit_blocks` mode, use configured `total_blocks` directly,
- validation split creation is per source, deterministic, and happens before mixture composition,
- validation split creation reuses existing `_ensure_validation_split` behavior per source,
- mixture mode fails fast unless `number_epochs == 1`,
- v1 fails fast unless `effective_total_blocks == requested_total_blocks`,
- validation execution emits per-source losses and weighted aggregate loss,
- validation uses non-dropping dataloader policy and fails fast when any source has zero validation batches,
- resume fails fast if accounting-critical mixture metadata differs from checkpoint metadata,
- fail-fast on incompatible distributed/optimizer settings.

Acceptance:

- red tests from Milestone 2 pass (green),
- existing packing tests still pass.

### Milestone 4: Auditable report and smoke configs

Add:

- `config/tests/clm_training_packing_mixture_smoke.yaml`,
- `config/tests/clm_training_packing_mixture_multinode_smoke.yaml`,
- `config/tests/mixture_packing_integration_smoke.yaml` (`task: testing`) that runs tokenization + mixture smoke.

The report file (rank 0) must include:

- config summary (`dataset_id`, weight, sequence_length, budget_mode, anchor_dataset_id, requested_total_blocks, effective_total_blocks, seed/schedule_seed),
- source capacities (`source_blocks_available`),
- runtime context (world_size, global batch settings, drop policies),
- `target_blocks_per_dataset`,
- `realized_blocks_per_dataset`,
- `realized_tokens_per_dataset`,
- `realized_ratios`,
- `deviation_from_target_blocks`,
- validation provenance (`existing_valid` vs `auto_generated_valid`) per source,
- `val_loss_per_dataset`,
- `val_loss_weighted`,
- reproducibility metadata (`hash_algorithm`, `allocation_algorithm`, `split_seed_algorithm`).

Acceptance:

- smoke run produces report with internally consistent totals.

### Milestone 5: SLURM validation and evidence capture

Use the existing SLURM test runner path and keep runtime defaults aligned with `slurm/tests/slurm_test.env`:

- `slurm/tests/run_tests.sh`,
- `slurm/tests/slurm_test.env`,
- optional secrets from `slurm/tests/test_secrets.env`.

Run command:

    ./slurm/tests/run_tests.sh --config config/tests/clm_training_packing_mixture_multinode_smoke.yaml --nodes 2 --ntasks-per-node 1 --partition postiguet1 --gpus 1 --cpus 8 --memory 32G --time 02:00:00

Ensure submitter guard is satisfied (`ALLOWED_SUBMITTERS` in `slurm/tests/slurm_test.env`). If the runner script is unavailable, use:

    ./slurm/submit_job.sh --config config/tests/clm_training_packing_mixture_multinode_smoke.yaml --partition postiguet1 --gpus 1 --nodes 2 --ntasks-per-node 1 --cpus 8 --memory 32G --time 02:00:00

After submission, record completion and logs with:

    sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed

When using `run_tests.sh`, also record the printed `Stdout log` and `Stderr log` paths.

Acceptance:

- SLURM job exits successfully,
- job ID, log paths, exact command, and report path are recorded in `issues/43-token-budget-mixture-replay.md`.

## Plan of Work

1. Update `config/schemas/training/components/data.schema.yaml` with mutually exclusive single-source vs multi-source support and strict validation rules.
2. Add failing unit tests in `tests/unit/training/test_mixture_packing.py` for allocation, determinism, budget resolution, distributed invariants, compatibility guards, and per-source validation auto-split/metric behavior.
3. Implement `src/tasks/training/data/mixture.py` with deterministic index mapping, dual budget-mode support, explicit `blake2b_u64` hashing, Hamilton allocation, canonical anchor-fixed derivation, and no full schedule array.
4. Extend `src/tasks/training/orchestrator.py` to load sources for mixture mode while preserving existing code path for `dataset.nameOrPath`.
5. Extend `src/tasks/training/fabric/trainer/base.py` to build mixture dataloaders, auto-generate per-source validation splits, resolve/validate budgets, enforce resume/validation fail-fast guards, aggregate counters, and write report.
6. Add/update smoke configs under `config/tests/` using the tiny-model defaults currently used by test configs.
7. Execute local tests/validation first, then SLURM smoke, then record evidence in issue 43.

## Concrete Steps

Run from repository root (`/home/gplsi/GPLSI/codigos/LMTK`).

1. Run focused red tests:

    python3 -m pytest -q tests/unit/training/test_mixture_packing.py

2. Run existing packing regression tests:

    python3 -m pytest -q tests/unit/training/test_packing_index.py tests/unit/training/test_packing_sampler.py tests/test_packed_sequence_dataset.py

3. Validate configs:

    python src/main.py --validate --config config/tests/clm_training_packing_mixture_smoke.yaml
    python src/main.py --validate --config config/tests/mixture_packing_integration_smoke.yaml

4. Run local integration smoke:

    python src/main.py --config config/tests/mixture_packing_integration_smoke.yaml

5. Run SLURM multi-node smoke:

    ./slurm/tests/run_tests.sh --config config/tests/clm_training_packing_mixture_multinode_smoke.yaml --nodes 2 --ntasks-per-node 1

Expected observable outputs:

- schema validation prints `Configuration is valid!`,
- local smoke writes `mixture_report.json` under configured output dir,
- SLURM runner prints `Submitted batch job <ID>`, followed by resolved stdout/stderr log paths.

## Validation and Acceptance

A change is accepted only when all are true:

1. Determinism:
   - same config + seed + world size produces the same `target_blocks_per_dataset` and same deterministic schedule mapping.

2. Budget/accounting:
   - budget resolution is deterministic and auditable:
     - in `explicit_blocks`, requested budget equals configured `total_blocks`,
     - in `anchor_epochs`, requested budget equals derived per-epoch total times `anchor_epochs`,
   - mixture mode enforces `number_epochs == 1`,
   - `effective_total_blocks == requested_total_blocks`,
   - `sum(realized_blocks_per_dataset) == effective_total_blocks`,
   - `sum(realized_tokens_per_dataset) == effective_total_blocks * sequence_length`,
   - each dataset’s deviation from deterministic target is exactly explained by documented distributed constraints (no unexplained drift).

3. Validation construction:
   - when a source has no `valid`, validation is auto-generated from that source only (no cross-source leakage),
   - validation split decisions are deterministic and persisted in report metadata,
   - validation logs/report include `val_loss_<dataset_id>` and `val_loss_weighted`,
   - run fails fast if any source has neither `valid` nor usable `validation_split`,
   - run fails fast if any source validation dataloader has zero batches.

4. Safety:
   - incompatible source/tokenizer/eos configs fail fast,
   - incompatible distributed step budgets fail fast,
   - no rank hangs when artifact/index build fails,
   - resume fails fast when accounting-critical mixture metadata differs from checkpoint metadata.

5. Regression:
   - existing online-packing tests continue to pass.
6. Reproducibility and metric checks:
   - each source in `mixture_report.json` includes reproducibility identity fields (`dataset_id`, `nameOrPath`, and dataset fingerprint/revision when available),
   - logged `seed`, `schedule_seed`, and split-seed algorithm id are present,
   - smoke metric gate passes: `val_loss_weighted` is finite and in `(0, 30)`.

## Idempotence and Recovery

- Re-running schema validation/tests is idempotent.
- Packing index artifacts remain reusable. Rebuild only when intentionally invalidating caches (delete source `.packing_index` directories).
- If a run fails during artifact creation, retry only after confirming no active job holds the relevant lock.
- Do not add fallback paths that silently alter budgets or compatibility outcomes.

## Artifacts and Notes

Reference snippets (normative for v1 behavior):

    def blake2b_u64(text: str) -> int:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)

    def allocate_exact_counts(total: int, weights_by_id: dict[str, float]) -> dict[str, int]:
        ids = sorted(weights_by_id)  # deterministic tie-break
        z = sum(weights_by_id[i] for i in ids)
        raw = {i: total * weights_by_id[i] / z for i in ids}
        base = {i: int(math.floor(raw[i])) for i in ids}
        rem = total - sum(base.values())
        # largest remainder first; lexicographic dataset_id ascending on ties
        order = sorted(ids, key=lambda i: (-(raw[i] - base[i]), i))
        for i in order[:rem]:
            base[i] += 1
        return base

    def derive_anchor_epoch_targets(
        source_blocks: dict[str, int],
        weights_by_id: dict[str, float],
        anchor_id: str,
    ) -> dict[str, int]:
        b_anchor = source_blocks[anchor_id]
        w_anchor = weights_by_id[anchor_id]
        raw_non_anchor = {
            did: b_anchor * weights_by_id[did] / w_anchor
            for did in sorted(weights_by_id)
            if did != anchor_id
        }
        total_non_anchor = int(math.floor(sum(raw_non_anchor.values()) + 0.5))
        targets_non_anchor = allocate_exact_counts(total_non_anchor, raw_non_anchor)
        return {anchor_id: b_anchor, **targets_non_anchor}

    def permute_index(i: int, total: int, schedule_seed: int) -> int:
        if total <= 1:
            return 0
        stride = (blake2b_u64(f"perm_stride|{schedule_seed}") % total) or 1
        while math.gcd(stride, total) != 1:
            stride = (stride + 1) % total or 1
        offset = blake2b_u64(f"perm_offset|{schedule_seed}") % total
        return (i * stride + offset) % total

    split_seed_i = blake2b_u64(f"valsplit|{base_seed}|{dataset_id}") % 2147483647

    mixture_meta = {
      "budget_mode": budget_mode,
      "anchor_dataset_id": anchor_dataset_id,
      "requested_total_blocks": requested_total_blocks,
      "world_size": world_size,
      "batch_size": batch_size,
      "gradient_accumulation_steps": grad_accum,
      "hash_algorithm": "blake2b_u64_v1",
      "allocation_algorithm": "hamilton_lr_lexicographic_v1"
    }

Minimal expected `mixture_report.json` structure:

    {
      "sequence_length": 128,
      "budget_mode": "explicit_blocks",
      "anchor_dataset_id": null,
      "source_blocks_available": {"A": 1000, "B": 300},
      "requested_total_blocks": 1000,
      "effective_total_blocks": 1000,
      "world_size": 2,
      "dataset_index_map": {"0": "A", "1": "B"},
      "sources": [{"dataset_id": "A", "weight": 2.0}, {"dataset_id": "B", "weight": 1.0}],
      "target_blocks_per_dataset": {"A": 667, "B": 333},
      "realized_blocks_per_dataset": {"A": 667, "B": 333},
      "realized_tokens_per_dataset": {"A": 85376, "B": 42624},
      "realized_ratios": {"A": 0.667, "B": 0.333},
      "val_loss_per_dataset": {"A": 2.10, "B": 2.45},
      "val_loss_weighted": 2.22,
      "validation_sources": {
        "A": {"mode": "auto_generated_valid", "seed": 42},
        "B": {"mode": "existing_valid"}
      },
      "hash_algorithm": "blake2b_u64_v1",
      "allocation_algorithm": "hamilton_lr_lexicographic_v1",
      "split_seed_algorithm": "blake2b_u64_mod_2147483647_v1"
    }

## Revision Note (2026-02-06)

Reworked this ExecPlan to be implementation-ready for a junior developer by removing ambiguity around validation cadence, adding explicit distributed budget invariants, defining exact report/accounting requirements, adding test-first sequencing, and including SLURM runbook details required by AGENTS.md and PLANS.md.

## Revision Note (2026-02-06, update 2)

Updated the plan to support two budget interfaces with one auditable resolved budget (`anchor_epochs` default with largest-source implicit anchor, and `explicit_blocks`), and locked v1 validation behavior to deterministic per-source auto-generated splits when dedicated validation datasets are unavailable.

## Revision Note (2026-02-06, update 3)

Aligned the ExecPlan with the hardened issue contracts by pinning deterministic algorithms (`blake2b_u64` hashing + Hamilton allocation), adding strict mixture-mode invariants (`number_epochs == 1`, `effective_total_blocks == requested_total_blocks`), and making per-source validation metrics/logging requirements explicit.

## Revision Note (2026-02-06, update 4)

Finalized junior-proof runtime semantics by clarifying virtual-epoch budgeting vs trainer epochs, standardizing tensor-safe source identity (`dataset_idx` + mapping), fixing Hamilton tie-break ordering in snippets, and adding reference snippets for deterministic permutation and resume-metadata compatibility checks.

## Revision Note (2026-02-06, update 5)

Locked non-overengineered validation behavior by requiring reuse of existing `FabricTrainerBase._ensure_validation_split` semantics per source instead of introducing a parallel split implementation.

## Revision Note (2026-02-06, update 6)

Corrected the `validate_after_k_steps` observation to match current trainer behavior, added explicit SLURM partition/resource and log-retrieval instructions (with manual fallback), and added reproducibility/metric acceptance gates required for ML-facing validation evidence.
