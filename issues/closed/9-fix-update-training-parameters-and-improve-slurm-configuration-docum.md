---
number: 9
title: "fix: update training parameters and improve SLURM configuration docum…"
state: closed
labels:
---

This pull request includes updates to configuration files and project metadata to improve training setup, resource allocation, and documentation. The changes primarily focus on enhancing clarity, flexibility, and usability for machine learning experiments and SLURM job submissions.

### Training Configuration Updates:
* [`config/experiments/salamandra-2b-mixed-lr/continual-1.yaml`](diffhunk://#diff-fa25b6eff2de54e5dfc753da9500b8f8f2fd79cd093f7a0a583a327e1032b16eL21-R21): Increased `number_epochs` from 1 to 2 to extend the training duration.

### Resource Allocation Improvements:
* [`makefile`](diffhunk://#diff-beda42571c095172ab63437d050612a571d0d9ddd3ad4f2aecbce907a9b7e3d0L37-R37): Updated GPU allocation from `"device=0,3"` to `all` for broader compatibility and resource utilization.

### Metadata and Documentation Enhancements:
* [`pyproject.toml`](diffhunk://#diff-50c86b7ed8ac2cf95bd48334961bf0530cdc77b5a56f852c5c61b89d735fd711L4-R5): Updated project description to "LMTK" and replaced placeholder author information with real names and emails.
* [`slurm/slurm_config.env`](diffhunk://#diff-d3afedf7464ddf00de760bd57f3f889d2ec3c99c38544083958aa70901b2ea15L18-R18): Added detailed comments and reorganized sections for better clarity, including SLURM output file patterns, project paths, and Docker configuration. Introduced new environment variables like `OUTPUT_FILE_PATTERN` and `ERROR_FILE_PATTERN` for SLURM job outputs. [[1]](diffhunk://#diff-d3afedf7464ddf00de760bd57f3f889d2ec3c99c38544083958aa70901b2ea15L18-R18) [[2]](diffhunk://#diff-d3afedf7464ddf00de760bd57f3f889d2ec3c99c38544083958aa70901b2ea15L28-R77