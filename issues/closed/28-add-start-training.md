---
number: 28
title: "Add start training"
state: closed
labels:
---

This pull request adds and updates a comprehensive set of experiment configuration files for dataset tokenization, merging, model training, and checkpoint conversion, primarily targeting the Salamandra-2b model. It introduces new YAML files for various datasets and training scenarios, improves dataset merging capabilities, and updates the schema to support the new merge task. Additionally, there are minor improvements to `.dockerignore` and validation/checkpoint logic in an existing experiment config.

**Experiment and Dataset Configuration Additions:**

* Added multiple new YAML files under `config/experiments/base-datasets/` for tokenization of datasets DC1 through DC6 and DC4/DC5/DC6, specifying tokenizer parameters, data sources, and output paths. [[1]](diffhunk://#diff-c5c6e4b36b3d7e289106f9ba4f1cf2a0b037422da64975e770d9f15d68f471e5R1-R23) [[2]](diffhunk://#diff-10cfdd949bcf5040a079b40da85603a3ae09bf51fbd3ed9b2599c0b4a1ddf530R1-R23) [[3]](diffhunk://#diff-8fb92bfbb9799104336a7d0b72242b19f7306687cd1dec71e6d2135e39dac1e0R1-R32) [[4]](diffhunk://#diff-ff5fcab03aa67f1cf51747128ab90ee9ef7b296d01bed63bfdcbd3e98e3d3181R1-R23) [[5]](diffhunk://#diff-9b945fbbd588fc7a9e13bce11b213bf2c8e4d90440fdeae005dc74954fbc4bf2R1-R23) [[6]](diffhunk://#diff-735b3098551a2bc4e371414d621ca5336d5a296fa2a1eb7bcdccbf6b1f6eaabcR1-R23)
* Added dataset merge configuration files for merging datasets with specific sampling ratios, including `dc3.yaml`, `dc3_corrected.yaml`, and `100_va_15_md.yaml`, supporting reproducible shuffling and compatibility checks. [[1]](diffhunk://#diff-8fb92bfbb9799104336a7d0b72242b19f7306687cd1dec71e6d2135e39dac1e0R1-R32) [[2]](diffhunk://#diff-e09b37846ab1ee05919449fe8fbf4e438ba5bd344fb3b62eaf3854c41d1b8658R1-R29)

**Model Training and Conversion Configurations:**

* Introduced several new experiment configs for Salamandra-2b continual and transfer learning training, including `c0dc1.yaml`, `c1dc2.yaml`, `c3dc1.yaml`, `c3dc1-continuation.yaml`, `c3dc1-convert.yaml`, `c3dc3.yaml`, `c3dc3-continuation.yaml`, and others, covering different datasets, training schedules, validation strategies, and checkpointing. [[1]](diffhunk://#diff-42d1c7268e6ad37b523c6f80cfb1d005c2ddabab00ae9220b9b4baedbcca2612R1-R65) [[2]](diffhunk://#diff-93e8c7d1aafdf894f9a65f4fabf81794f60fb64b8cdb1475647668a74f37953eR1-R76) [[3]](diffhunk://#diff-d62fd262e0a4c2c4e9448c9f909031cc4d5e84fb9f4d706d639a9c942936d5c1R1-R66) [[4]](diffhunk://#diff-9629c4a4b304b94bb3f5a72d1a12d11b74539e33ef55b8423e7529a7785cd35bR1-R69) [[5]](diffhunk://#diff-8ade406435ead8ce3e2d816806f7a803c945ae4e09274501d4d50ea35cf793c7R1-R10) [[6]](diffhunk://#diff-3bbed7d6e551977303b9798432506ad8b0ff567beac715cddc39b291fea1b972R1-R66) [[7]](diffhunk://#diff-5da2f4b8318c265b3d9036f08f1a129f47acb318e0d2931a85cf2851c88ef7feR1-R70)
* Added conversion configs to transform FSDP checkpoints to HuggingFace format for easier downstream use. [[1]](diffhunk://#diff-8ade406435ead8ce3e2d816806f7a803c945ae4e09274501d4d50ea35cf793c7R1-R10) [[2]](diffhunk://#diff-4433174bf4610c012aec6186dae22144d67d375eaa9d1729d52b8171d1f0c912R1-R10)

**Schema and Logic Improvements:**

* Updated the schema in `base.schema.yaml` to add `"dataset_merge"` as a valid task type, enabling validation of new merge configurations.
* Improved validation and checkpoint parameters in existing experiment files, switching from step-based to epoch-based validation and checkpoint logic.

**Miscellaneous:**

* Updated `.dockerignore` to exclude the `output/` directory, preventing experiment results from being included in Docker builds.

**References:**  
[[1]](diffhunk://#diff-c5c6e4b36b3d7e289106f9ba4f1cf2a0b037422da64975e770d9f15d68f471e5R1-R23) [[2]](diffhunk://#diff-10cfdd949bcf5040a079b40da85603a3ae09bf51fbd3ed9b2599c0b4a1ddf530R1-R23) [[3]](diffhunk://#diff-8fb92bfbb9799104336a7d0b72242b19f7306687cd1dec71e6d2135e39dac1e0R1-R32) [[4]](diffhunk://#diff-ff5fcab03aa67f1cf51747128ab90ee9ef7b296d01bed63bfdcbd3e98e3d3181R1-R23) [[5]](diffhunk://#diff-9b945fbbd588fc7a9e13bce11b213bf2c8e4d90440fdeae005dc74954fbc4bf2R1-R23) [[6]](diffhunk://#diff-735b3098551a2bc4e371414d621ca5336d5a296fa2a1eb7bcdccbf6b1f6eaabcR1-R23) [[7]](diffhunk://#diff-e09b37846ab1ee05919449fe8fbf4e438ba5bd344fb3b62eaf3854c41d1b8658R1-R29) [[8]](diffhunk://#diff-42d1c7268e6ad37b523c6f80cfb1d005c2ddabab00ae9220b9b4baedbcca2612R1-R65) [[9]](diffhunk://#diff-93e8c7d1aafdf894f9a65f4fabf81794f60fb64b8cdb1475647668a74f37953eR1-R76) [[10]](diffhunk://#diff-d62fd262e0a4c2c4e9448c9f909031cc4d5e84fb9f4d706d639a9c942936d5c1R1-R66) [[11]](diffhunk://#diff-9629c4a4b304b94bb3f5a72d1a12d11b74539e33ef55b8423e7529a7785cd35bR1-R69) [[12]](diffhunk://#diff-8ade406435ead8ce3e2d816806f7a803c945ae4e09274501d4d50ea35cf793c7R1-R10) [[13]](diffhunk://#diff-3bbed7d6e551977303b9798432506ad8b0ff567beac715cddc39b291fea1b972R1-R66) [[14]](diffhunk://#diff-5da2f4b8318c265b3d9036f08f1a129f47acb318e0d2931a85cf2851c88ef7feR1-R70) [[15]](diffhunk://#diff-9f8dcf28d1ce18b7bb0b7be270a3305377bc7ff9a620c7e695f92672dacfb8d1L7-R7) [[16]](diffhunk://#diff-39b04f9fa224cb021ae116996bfc0a976e4181bb4fa0609f4537e36eb9532043L24-R25) [[17]](diffhunk://#diff-2f754321d62f08ba8392b9b168b83e24ea2852bb5d815d63e767f6c3d23c6ac5R5)