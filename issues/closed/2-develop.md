---
number: 2
title: "Develop"
state: closed
labels:
---

This pull request introduces significant updates to the codebase, primarily transitioning from "pretraining" to "clm_training" for the task nomenclature. It includes changes to configuration files, schemas, documentation, and Docker setup. These updates standardize terminology, improve configurability, and enhance the development environment.

### Task Transition: Pretraining to CLM Training
* Updated task name in `config/experiments/salamandra-2b-from-epoch-1/continual.yaml` and `tokenizer.yaml` to "clm_training" [[1]](diffhunk://#diff-95f4f408aba1a7dff21720980afc117171c9fe594b51409910bdc866ad0b10aaL1-R3) [[2]](diffhunk://#diff-5c51d74b57a94ff43d72c078fbdc51e72e4668805c7c7ac3f1d98b1fd259e0f0L9-R9).
* Extended task options in `base.schema.yaml` to include "clm_training" and other new tasks.
* Renamed schema files from "pretraining" to "clm_training" to reflect the new task naming convention [[1]](diffhunk://#diff-44e92c6261609376f31711682bfa4f12cdfabcde4b234290ff6e40bc023023ddL2-R2) [[2]](diffhunk://#diff-cdf9a47cb5a670454e85aed19ab889ec2016d3a45df3cab2a8cd8a280c1c2eb4L2-R2) [[3]](diffhunk://#diff-c3e394bb6bd04dc799faba34c2482a7e6422f1dac338756f077f93eae90660eaL2-R9) [[4]](diffhunk://#diff-3c9988b4549c3c1ee801d28ccfd33f9b38bce1a7f2a2a254624df7c51d1e2fc3L2-R2) [[5]](diffhunk://#diff-c86fd4da13ea2d188141de45c3d1b7baea77a25034f263d6e4342808f9f5ec00L165-L167).

### Configuration Updates
* Adjusted experiment parameters in `continual.yaml`, such as reducing `batch_size` and enabling validation at the end of training and after each epoch.
* Fixed a typo in the `precision` field in `continual.yaml`.

### Documentation Updates
* Updated API references and guides to replace "pretraining" with "clm_training" in module paths and examples [[1]](diffhunk://#diff-ed526a86f397b43bd6dc7663a2724623d38fbbaf7e208905df1214653edc8f1fL11-R72) [[2]](diffhunk://#diff-008dcb3426febd767787b1521f1fe33086313b927ea37eaab86df5fa88a51698L67-R68) [[3]](diffhunk://#diff-734c1cc58bbe15f4080a8ae3ac71c15dd0a66097e3ece30985bc497c63ee829fL33-R33) [[4]](diffhunk://#diff-734c1cc58bbe15f4080a8ae3ac71c15dd0a66097e3ece30985bc497c63ee829fL56-R56) [[5]](diffhunk://#diff-734c1cc58bbe15f4080a8ae3ac71c15dd0a66097e3ece30985bc497c63ee829fL170-R170) [[6]](diffhunk://#diff-734c1cc58bbe15f4080a8ae3ac71c15dd0a66097e3ece30985bc497c63ee829fL198-R198).

### Dockerfile Enhancements
* Improved Dockerfile by switching to a non-interactive frontend, adding user creation with configurable UID/GID, and optimizing Poetry setup for dependency management [[1]](diffhunk://#diff-f34da55ca08f1a30591d8b0b3e885bcc678537b2a9a4aadea4f190806b374ddcL7-R14) [[2]](diffhunk://#diff-f34da55ca08f1a30591d8b0b3e885bcc678537b2a9a4aadea4f190806b374ddcR28-R59).

### Miscellaneous
* Updated `.dockerignore` to exclude additional temporary and test files.