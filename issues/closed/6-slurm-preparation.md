---
number: 6
title: "Slurm preparation"
state: closed
labels:
---

This pull request introduces significant updates to the project's configuration, documentation, and execution workflows. Key changes include the addition of new experiment configurations for training and tokenization, improvements to SLURM cluster execution documentation, and the removal of legacy files related to Docker-based execution. 

Our most important change was the setup of an official SLURM execution protocol, which includes documentation, environment configurations and execution scripts that handle all the SLURM execution pipeline, including the automatic creation of the docker image (if necessary), the container, and executing the main script with the specified experiment configuration.

Below is a categorized summary of the most important changes:

### Configuration Updates:
* Added `config/experiments/gpt-2/continual.yaml` and `config/experiments/gpt-2/tokenizer.yaml` for GPT-2 continual training and tokenization tasks, including settings for dataset paths, model parameters, logging, and distributed training strategies. [[1]](diffhunk://#diff-8a01fecf7007e2926cd39ec785259ad128d3f588541599bad823580a648d4db1R1-R65) [[2]](diffhunk://#diff-befb446bca57bd6bd8117f59dd0662b5477ca57f3393403d0758e995e5874d7dR1-R23)
* Added `config/experiments/salamandra-2b-mixed-lr/continual-1.yaml` and `config/experiments/salamandra-2b-mixed-lr/continual-2.yaml` for Salamandra-2B mixed learning rate experiments, with configurations for checkpointing, gradient accumulation, and validation. [[1]](diffhunk://#diff-fa25b6eff2de54e5dfc753da9500b8f8f2fd79cd093f7a0a583a327e1032b16eR1-R65) [[2]](diffhunk://#diff-e2e4c2ec4a5b5397f259156c991b938c691019a30db0dde3e7d2b5aeed19bbf3R1-R66)
* Added `config/experiments/salamandra-2b-mixed-lr/tokenizer.yaml` for Salamandra-2B tokenization tasks, specifying tokenizer settings, dataset formats, and output paths.

### Documentation Enhancements:
* Updated `README.md` to include a detailed section on SLURM cluster execution, highlighting configurable job scripts, Docker integration, WandB tracking, and debugging support.

### File Removals:
* Removed legacy files `execute.sh` and `p1-dgx.slurm`, which were used for Docker-based execution and SLURM job submission, respectively. These workflows are now replaced with updated SLURM documentation and helper scripts. [[1]](diffhunk://#diff-ce75fc79763c9e26a2a7465a435f9e552b3d3283fb2440296c803d8c28fc5996L1-L68) [[2]](diffhunk://#diff-916b5d7d629b84b695313b267833a5cf92803c7497114d3cd75dc3de2b618593L1-L191)

### Codebase Simplification:
* Modified `docker/Dockerfile` to update workspace permissions from `chown` to `chmod`, simplifying ownership management.

### Miscellaneous:
* Removed `CHANGELOG.md` as part of cleanup, possibly indicating a shift to a different version tracking approach.