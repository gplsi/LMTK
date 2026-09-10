---
number: 1
title: "Slurm preparation"
state: closed
labels:
---

This pull request introduces significant changes to transition the project from a "pretraining" task to a "causal language model (CLM) training" task. It also includes updates to configuration files, schemas, Docker setup, and documentation to align with the new task. Below is a summary of the most important changes:

### Task Transition and Configuration Updates:
* Updated task type from `pretraining` to `clm_training` in various configuration files, including `continual.yaml` and `tokenizer.yaml`, and adjusted related parameters such as `batch_size`, `validate_on_end`, and `validate_after_epoch` to better suit the CLM training process. [[1]](diffhunk://#diff-95f4f408aba1a7dff21720980afc117171c9fe594b51409910bdc866ad0b10aaL1-R3) [[2]](diffhunk://#diff-95f4f408aba1a7dff21720980afc117171c9fe594b51409910bdc866ad0b10aaL16-R26) [[3]](diffhunk://#diff-5c51d74b57a94ff43d72c078fbdc51e72e4668805c7c7ac3f1d98b1fd259e0f0L9-R9)
* Extended the `task` enum in `base.schema.yaml` to include new task types like `clm_training`, `mlm_training`, and others.

### Schema Refactoring:
* Renamed schema files from `pretraining.*.schema.yaml` to `clm_training.*.schema.yaml` to reflect the new task, and updated `$id` and references accordingly. [[1]](diffhunk://#diff-44e92c6261609376f31711682bfa4f12cdfabcde4b234290ff6e40bc023023ddL2-R2) [[2]](diffhunk://#diff-cdf9a47cb5a670454e85aed19ab889ec2016d3a45df3cab2a8cd8a280c1c2eb4L2-R2) [[3]](diffhunk://#diff-c3e394bb6bd04dc799faba34c2482a7e6422f1dac338756f077f93eae90660eaL2-R9) [[4]](diffhunk://#diff-3c9988b4549c3c1ee801d28ccfd33f9b38bce1a7f2a2a254624df7c51d1e2fc3L2-R2)
* Removed unused or commented-out conditional schemas and redundant lines in `clm_training.schema.yaml` for better maintainability.

### Dockerfile Enhancements:
* Modified the Dockerfile to create a new user with configurable UID/GID, added `sudo` to the installed packages, and updated the Poetry installation process to use version 1.8.4 with caching optimizations. [[1]](diffhunk://#diff-f34da55ca08f1a30591d8b0b3e885bcc678537b2a9a4aadea4f190806b374ddcL7-R14) [[2]](diffhunk://#diff-f34da55ca08f1a30591d8b0b3e885bcc678537b2a9a4aadea4f190806b374ddcR28-R59)

### Documentation Updates:
* Updated API references, guides, and examples in the documentation to replace `pretraining` with `clm_training` in module paths and code snippets. [[1]](diffhunk://#diff-ed526a86f397b43bd6dc7663a2724623d38fbbaf7e208905df1214653edc8f1fL11-R72) [[2]](diffhunk://#diff-008dcb3426febd767787b1521f1fe33086313b927ea37eaab86df5fa88a51698L67-R68) [[3]](diffhunk://#diff-734c1cc58bbe15f4080a8ae3ac71c15dd0a66097e3ece30985bc497c63ee829fL33-R33) [[4]](diffhunk://#diff-734c1cc58bbe15f4080a8ae3ac71c15dd0a66097e3ece30985bc497c63ee829fL56-R56) [[5]](diffhunk://#diff-734c1cc58bbe15f4080a8ae3ac71c15dd0a66097e3ece30985bc497c63ee829fL170-R170) [[6]](diffhunk://#diff-734c1cc58bbe15f4080a8ae3ac71c15dd0a66097e3ece30985bc497c63ee829fL198-R198)

### Miscellaneous:
* Added new patterns to `.dockerignore` to exclude test and output files (`*.out`, `*.err`, `test_checkpoint_naming.py`).