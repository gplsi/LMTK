---
number: 16
title: "Tokenization instruction tuning"
state: closed
labels:
---

This pull request introduces a new task for tokenization instruction workflows, including support for multi-language datasets, chat templates, and instruction-specific preprocessing. The changes span configuration files, schema updates, orchestration logic, and preprocessing utilities. Additionally, minor updates were made to resource configurations and dependencies.

### Tokenization Instruction Task Implementation:

* **Configuration for Tokenization Instruction Task**: Added a new YAML configuration file (`config/experiments/tokenization_instruction_example.yaml`) for the ALIA dataset, specifying model, tokenizer, multi-language support, dataset paths, and training arguments.
* **Tokenization Instruction Schema**: Introduced a dedicated schema (`config/schemas/tokenization_instruction.schema.yaml`) to validate configurations for the tokenization instruction task, including tokenizer properties, dataset paths, and instruction tuning-specific settings.
* **Task Orchestration**: Implemented the `TokenizationInstructionOrchestrator` class in `src/tasks/tokenization_instruction/orchestrator.py` to manage the workflow, including validation, multi-language dataset processing, and saving tokenized outputs.
* **Preprocessing Utilities**: Added preprocessing logic in `src/tasks/tokenization_instruction/src/prepare_data/preprocess.py` to tokenize multi-turn conversations with chat templates and mask prompt tokens for instruction tuning.

### Schema and Configuration Updates:

* **Base Schema Update**: Extended the `task` enum in `config/schemas/base.schema.yaml` to include the new `tokenization_instruction` task type.

### Resource and Dependency Updates:

* **SLURM Resource Configuration**: Adjusted SLURM job settings in `slurm/slurm_config.env` and `slurm/submit_job.sh`, reducing GPU count, memory allocation, and CPUs per task to optimize resource usage. [[1]](diffhunk://#diff-d3afedf7464ddf00de760bd57f3f889d2ec3c99c38544083958aa70901b2ea15L21-R25) [[2]](diffhunk://#diff-aa487ed4e39a31fc978dc502c191f4eddc6700bf73c9278ff6ba499020eec164L50-R50)
* **Dependency Addition**: Added `sentencepiece` to `pyproject.toml` to support tokenization tasks.

### Miscellaneous Updates:

* **Debugging Enhancements**: Improved logging in SLURM scripts (`slurm/p.slurm`, `slurm/validate.sh`) for better visibility during execution. [[1]](diffhunk://#diff-7ddb008bb275c81170a989df865f5bf288271d98d4e5b7e36b2728b6e0784093R119) [[2]](diffhunk://#diff-a0a7b6a49cf6db99fffd384274c5d02450bfa20ed3fb85e179cb3487bab2fd1aL54-R54)
* **Docker Command**: Introduced a `DOCKER_RUN` variable in the `makefile` for easier containerized execution.