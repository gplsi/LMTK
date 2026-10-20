---
number: 32
title: "Conda"
state: closed
labels:
---

This pull request removes Docker support from the project and fully migrates environment management and job execution to Conda. The documentation, SLURM scripts, and environment setup have all been updated to reflect this transition, simplifying workflows and reducing complexity. The most significant changes are grouped below.

**Environment and Job Execution Migration:**

* All Docker-related code, configuration, and documentation have been removed from the SLURM scripts (`slurm/p.slurm`) and replaced with Conda-based environment setup using a new `scripts/set_environment.sh` script. Jobs are now launched directly with Python instead of inside Docker containers. [[1]](diffhunk://#diff-7ddb008bb275c81170a989df865f5bf288271d98d4e5b7e36b2728b6e0784093L39-L44) [[2]](diffhunk://#diff-7ddb008bb275c81170a989df865f5bf288271d98d4e5b7e36b2728b6e0784093L116-L121) [[3]](diffhunk://#diff-7ddb008bb275c81170a989df865f5bf288271d98d4e5b7e36b2728b6e0784093L134-L135) [[4]](diffhunk://#diff-7ddb008bb275c81170a989df865f5bf288271d98d4e5b7e36b2728b6e0784093L157-L162) [[5]](diffhunk://#diff-7ddb008bb275c81170a989df865f5bf288271d98d4e5b7e36b2728b6e0784093L207-R238) [[6]](diffhunk://#diff-7ddb008bb275c81170a989df865f5bf288271d98d4e5b7e36b2728b6e0784093L332-R252) [[7]](diffhunk://#diff-99f4dcd2317f60dd3888788acee00f8311651efdaba7247bc5030880539ea56aR1-R18)

**Documentation Updates:**

* The main `README.md` and `slurm/README.md` have been updated to remove Docker references and instructions, replacing them with Conda-based setup and job execution guidance. This includes installation steps, SLURM workflow explanations, troubleshooting, and validation processes. [[1]](diffhunk://#diff-b335630551682c19a781afebcf4d07bf978fb1f8ac04c6bf87428ed5106870f5R7-R22) [[2]](diffhunk://#diff-b335630551682c19a781afebcf4d07bf978fb1f8ac04c6bf87428ed5106870f5L160) [[3]](diffhunk://#diff-b335630551682c19a781afebcf4d07bf978fb1f8ac04c6bf87428ed5106870f5L223) [[4]](diffhunk://#diff-b335630551682c19a781afebcf4d07bf978fb1f8ac04c6bf87428ed5106870f5L269-L290) [[5]](diffhunk://#diff-aef058ffbc2c8716c6779c337792d449077683528cb410c667997362e13ee45bL3-R3) [[6]](diffhunk://#diff-aef058ffbc2c8716c6779c337792d449077683528cb410c667997362e13ee45bL61) [[7]](diffhunk://#diff-aef058ffbc2c8716c6779c337792d449077683528cb410c667997362e13ee45bL111-R111) [[8]](diffhunk://#diff-aef058ffbc2c8716c6779c337792d449077683528cb410c667997362e13ee45bL126-R126) [[9]](diffhunk://#diff-aef058ffbc2c8716c6779c337792d449077683528cb410c667997362e13ee45bL492-L548) [[10]](diffhunk://#diff-aef058ffbc2c8716c6779c337792d449077683528cb410c667997362e13ee45bL885) [[11]](diffhunk://#diff-aef058ffbc2c8716c6779c337792d449077683528cb410c667997362e13ee45bL927-R868) [[12]](diffhunk://#diff-aef058ffbc2c8716c6779c337792d449077683528cb410c667997362e13ee45bL954-R884) [[13]](diffhunk://#diff-aef058ffbc2c8716c6779c337792d449077683528cb410c667997362e13ee45bL978-L990) [[14]](diffhunk://#diff-aef058ffbc2c8716c6779c337792d449077683528cb410c667997362e13ee45bL1069-L1071) [[15]](diffhunk://#diff-aef058ffbc2c8716c6779c337792d449077683528cb410c667997362e13ee45bL1090-L1096) [[16]](diffhunk://#diff-aef058ffbc2c8716c6779c337792d449077683528cb410c667997362e13ee45bL1127)

**Script and Configuration Cleanup:**

* All Docker-specific environment variables and checks have been removed from the SLURM scripts and configuration files. The new environment setup script ensures the Conda environment is properly initialized and activated for job runs. [[1]](diffhunk://#diff-7ddb008bb275c81170a989df865f5bf288271d98d4e5b7e36b2728b6e0784093L39-L44) [[2]](diffhunk://#diff-7ddb008bb275c81170a989df865f5bf288271d98d4e5b7e36b2728b6e0784093L116-L121) [[3]](diffhunk://#diff-99f4dcd2317f60dd3888788acee00f8311651efdaba7247bc5030880539ea56aR1-R18)

**SLURM Workflow and Troubleshooting:**

* SLURM job templates and troubleshooting sections now focus on Conda environment issues rather than Docker image problems. The validation and debugging instructions have been updated accordingly. [[1]](diffhunk://#diff-aef058ffbc2c8716c6779c337792d449077683528cb410c667997362e13ee45bL111-R111) [[2]](diffhunk://#diff-aef058ffbc2c8716c6779c337792d449077683528cb410c667997362e13ee45bL126-R126) [[3]](diffhunk://#diff-aef058ffbc2c8716c6779c337792d449077683528cb410c667997362e13ee45bL492-L548) [[4]](diffhunk://#diff-aef058ffbc2c8716c6779c337792d449077683528cb410c667997362e13ee45bL927-R868) [[5]](diffhunk://#diff-aef058ffbc2c8716c6779c337792d449077683528cb410c667997362e13ee45bL954-R884) [[6]](diffhunk://#diff-aef058ffbc2c8716c6779c337792d449077683528cb410c667997362e13ee45bL978-L990) [[7]](diffhunk://#diff-aef058ffbc2c8716c6779c337792d449077683528cb410c667997362e13ee45bL1069-L1071) [[8]](diffhunk://#diff-aef058ffbc2c8716c6779c337792d449077683528cb410c667997362e13ee45bL1090-L1096)

**New Environment Setup Script:**

* Added `scripts/set_environment.sh` to initialize and activate the Conda environment, ensuring all required variables and dependencies are available for job execution.

These changes modernize and simplify the project's infrastructure, making it easier to use and maintain without relying on Docker.