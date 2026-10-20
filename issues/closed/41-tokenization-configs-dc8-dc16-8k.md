---
number: 41
title: "Add 8k/2k tokenization configs for DC8-DC16"
state: closed
labels:
- enhancement
---

## Summary
Generate new tokenization configs under `config/experiments/base-datasets/` for DC8-DC16 using context length 8192 and overlap 2048, while keeping the raw dataset sources the same and assigning new incremental IDs to avoid overwriting existing outputs.

## Background / Current State
- `config/experiments/base-datasets/dc8.yaml` through `config/experiments/base-datasets/dc16.yaml` tokenize the DC8-DC16 raw datasets at context length 2048 with overlap 512.
- Each config writes to `data/tokenized/dc8` through `data/tokenized/dc16`.

## Goals
- Add a new tokenization config per DC8-DC16 dataset with context length 8192 and overlap 2048.
- Use new incremental IDs so output paths are distinct from the existing 2k-tokenized datasets.

## Non-goals
- No schema or task changes.
- No dataset merge updates.
- No changes to the existing DC8-DC16 configs.

## Prior Art / References
- `config/experiments/base-datasets/dc8.yaml`
- `config/experiments/base-datasets/dc9.yaml`
- `config/experiments/base-datasets/dc10.yaml`
- `config/experiments/base-datasets/dc11.yaml`
- `config/experiments/base-datasets/dc12.yaml`
- `config/experiments/base-datasets/dc13.yaml`
- `config/experiments/base-datasets/dc14.yaml`
- `config/experiments/base-datasets/dc15.yaml`
- `config/experiments/base-datasets/dc16.yaml`

## Acceptance Criteria
- New configs exist under `config/experiments/base-datasets/` with context length 8192 and overlap 2048.
- Each new config keeps `dataset.nameOrPath` pointing to `data/downloaded/dc8` through `data/downloaded/dc16`.
- Each new config uses a new incremental ID for `experiment_name` and `output.path`.

## Validation
- Manually run `python -m src.main --config <new_config>` for one of the new configs to confirm the tokenization job starts and writes output to the new `data/tokenized/dcXX` path.

## Status / Notes
- Added `config/experiments/base-datasets/dc20.yaml` through `config/experiments/base-datasets/dc28.yaml` with context length 8192 and overlap 2048 mapped to DC8-DC16 sources.
