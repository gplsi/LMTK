---
number: 44
title: "Add dataset_analysis task for exact tokenized dataset statistics"
state: open
labels:
  - enhancement
  - tokenization
  - data-quality
  - training
---

## Summary

Add a first-class `task: dataset_analysis` workflow that reads tokenized datasets from disk and produces exact, auditable statistics needed before training (padding amount, effective sequence lengths, top tokens, and unique-token count).

## Background / Current State

- Tokenized datasets are produced by `task: tokenization` and consumed by training tasks.
- We do not currently have a dedicated task for post-tokenization quality analysis.
- Users are forced to inspect datasets ad-hoc, which is slow and inconsistent.

## Problem

We need a stable and reproducible way to answer:

- How many samples and tokens are in a tokenized dataset?
- How much padding exists?
- What are the effective (non-padded) sequence lengths?
- Which token IDs dominate the corpus?
- How many unique token IDs exist?

This must work for both fixed-window CLM tokenization and doc-level tokenization.

## Goals

1. Add `dataset_analysis` to the supported `task` enum and dispatch path.
2. Implement exact metric computation with deterministic outputs.
3. Emit both machine-readable and human-readable reports.
4. Support split selection and strict validation/fail-fast semantics.
5. Cover the behavior with task-focused unit tests and smoke configs.

## Non-goals

- No changes to tokenization output format.
- No training-loop changes.
- No approximate/sketch-based counting in v1.
- No automatic correction of malformed datasets.

## Proposed Metrics (v1)

Per split and global aggregate:

- `num_examples`
- `total_tokens_raw`
- `total_tokens_effective` (non-padding tokens when derivable)
- `padding_tokens` / `padding_ratio` (when derivable)
- `avg_non_padded_seq_len`
- `non_padded_seq_len_stats` (`min`, `p50`, `p95`, `p99`, `max`)
- `unique_token_count` (exact)
- `top_k_tokens` (exact `token_id`, `count`, `frequency`)

## Fail-fast Conditions

- Dataset rows missing `input_ids`.
- Requested split not present.
- `compute_padding_metrics: true` while effective tokens cannot be derived from dataset columns.
- Unsupported dataset source/format for this task.

## Scope

- `config/schemas/base.schema.yaml`
- `config/schemas/dataset_analysis.schema.yaml`
- `src/main.py`
- `src/tasks/dataset_analysis/__init__.py`
- `src/tasks/dataset_analysis/orchestrator.py`
- `src/tasks/dataset_analysis/test_orchestrator.py`
- `config/examples/dataset_analysis_example.yaml`
- `config/tests/dataset_analysis_smoke_fixed.yaml`
- `config/tests/dataset_analysis_smoke_doclevel.yaml`
- `issues/execPlans/44-tokenized-dataset-analysis.execplan.md`

## Acceptance Criteria

- `task: dataset_analysis` validates and dispatches through `src/main.py`.
- A fixed-window tokenized dataset run outputs JSON + Markdown reports with padding and effective-length metrics.
- A doc-level tokenized dataset run outputs JSON + Markdown reports with exact token-frequency and unique-token metrics.
- Unit tests cover fixed-window metrics, doc-level metrics, split selection, and fail-fast behavior.

## Implementation Status (2026-02-06)

- Implemented in code:
  - task wiring + schema,
  - orchestrator and report generation,
  - task-focused unit tests,
  - schema tests,
  - example + smoke configs.
- Local validation executed:
  - `python3 -m pytest src/tasks/dataset_analysis/test_orchestrator.py tests/unit/config/test_dataset_analysis_schema.py -q` => `6 passed`.
  - smoke config validation for fixed/doc-level configs succeeded with `PYTHONPATH=. python3 src/main.py --validate --config ...`.
  - end-to-end CLI run on a temporary tokenized dataset created both artifacts:
    - `/tmp/lmtk_ds_analysis_is_uckra/out/tmp_report.json`
    - `/tmp/lmtk_ds_analysis_is_uckra/out/tmp_report.md`

## Related Work

- `issues/42-online-packing.md` (doc-level tokenization context)
- `issues/43-token-budget-mixture-replay.md` (future consumers of dataset quality metrics)
