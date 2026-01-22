---
number: 3
title: "Slurm preparation"
state: closed
labels:
---

This pull request introduces several changes to improve configuration management, task flexibility, and script usability. Key updates include modifications to schema files to support additional tasks, updates to the Slurm script for dynamic user handling, and enhancements to the tokenization orchestrator to align with new task configurations.

### Configuration Updates:
* [`config/schemas/clm_training.schema.yaml`](diffhunk://#diff-c86fd4da13ea2d188141de45c3d1b7baea77a25034f263d6e4342808f9f5ec00L2-R2): Updated the `$id` field to reflect the new schema name (`clm_training.schema.yaml`) for better clarity and consistency.
* [`config/schemas/tokenization.schema.yaml`](diffhunk://#diff-d6389c9801940a088f7ce7d4df5f8685e75a7e4a88b81f83a1e1b06887185798L22-R23): Expanded the `task` enum to include `instruction_finetuning` and `mlm_training` alongside `clm_training`, with the default updated to `clm_training`. This enhances the flexibility of the schema for multiple training tasks.

### Script Enhancements:
* [`p1-dgx.slurm`](diffhunk://#diff-916b5d7d629b84b695313b267833a5cf92803c7497114d3cd75dc3de2b618593R24-R31): Introduced dynamic retrieval of user and group IDs, replacing hardcoded paths with user-specific variables for improved portability and usability in shared environments. [[1]](diffhunk://#diff-916b5d7d629b84b695313b267833a5cf92803c7497114d3cd75dc3de2b618593R24-R31) [[2]](diffhunk://#diff-916b5d7d629b84b695313b267833a5cf92803c7497114d3cd75dc3de2b618593L81-L84)

### Code Functionality:
* [`src/tasks/tokenization/orchestrator.py`](diffhunk://#diff-cb2a6e45352f372f6e32126a210913fd14a765df4a7ddfa737aba06d4c7b13bbR87-R93): Added task-based logic to the `tokenize_dataset` method, allowing it to select the appropriate tokenizer (`CausalLMTokenizer`) based on the task configuration. An error is raised for unsupported tasks, ensuring robustness.

### Miscellaneous:
* [`.dockerignore`](diffhunk://#diff-2f754321d62f08ba8392b9b168b83e24ea2852bb5d815d63e767f6c3d23c6ac5L5): Removed the `**/outputs/` pattern from the ignore list, potentially allowing output directories to be included in Docker builds.