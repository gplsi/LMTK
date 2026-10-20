---
number: 26
title: "Fixed mlm"
state: closed
labels:
---

- Added support for parquet files, commonly used in Datasets from HuggingFace. This way, we can seamlessly use our own datasets from HuggingFace without needing to change the format.
- Introduced new scripts for dataset visualisation. These scripts are intended to be used after the tokenisation to check if it has been done correctly. The scripts let us visualise a sample of input_ids, attention_mask and labels for human evaluation.
- Fixed some training issues with masked language modelling tasks.