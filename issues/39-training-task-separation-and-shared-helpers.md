---
number: 39
title: "Refactor training task layout and shared helpers"
state: open
labels:
- refactor
- maintenance
---

## Summary
The training code under `src/tasks/training/` mixes task-specific models, distributed strategies, and shared helpers in a way that makes it hard to extend and reason about. We need to separate training task types into their own directories, extract shared logic into explicit helper modules, and reduce duplication while keeping behavior unchanged.

## Background / Current State
- `src/main.py` maps `clm_training`, `mlm_training`, and `instruction` to the shared `src/tasks/training` task module.
- Task-specific model classes live under `src/tasks/training/fabric/model/` and are selected via `MODEL_CLASS_MAP` inside `src/tasks/training/fabric/trainer/base.py`.
- Distributed strategies live in `src/tasks/training/fabric/trainer/distributed.py`.
- Shared helpers (optimizers, schedulers, seed control) are bundled in `src/tasks/training/utils.py`.
- FSDP policy logic is duplicated between `src/tasks/training/fabric/wrappers/policies.py` and `src/tasks/training/fabric/wrappers/fsdp_config.py`.

## Problem
The current layout hides task boundaries and spreads shared logic across framework-specific modules. This increases coupling between tasks and the Fabric implementation, makes it harder to add new training types, and encourages duplication.

## Goals
- Separate training task types (CLM, MLM, instruction) into their own modules/directories.
- Consolidate shared helper logic into explicit shared modules.
- Keep behavior, configs, and user-facing workflows unchanged.
- Align code layout with the training schema structure under `config/schemas/training/`.

## Non-goals
- No changes to training behavior, hyperparameter defaults, or schema semantics.
- No new tasks, configs, or SLURM submission changes.
- No cross-cutting redesign of the training pipeline or public API changes.

## Prior Art / References
- `issues/closed/23-mlm-training-instruction-fixing-training-schemas-re-estructure-and-training-task-reestructure.md` (prior training/task restructuring).
- Training schema composition under `config/schemas/training/components/` and `config/schemas/training/strategy/`.
- Existing task package structures (e.g., `src/tasks/publish/format` and `src/tasks/publish/upload`) for separation patterns.

## Proposed Approach (High-Level)
- Create explicit packages for task-specific training types and for shared helpers.
- Move CLM/MLM/Instruction model classes into task-type modules and introduce a registry for task-to-model mapping.
- Move optimizer/scheduler/seed helpers and FSDP auto-wrap policy helpers into shared modules.
- Keep Fabric-specific trainer/logging code under `src/tasks/training/fabric/` but update imports to use the new shared/task packages.

## Scope & Invariants (Approval Required)
This is a significant refactor. Implementation should only proceed after explicit approval.

Invariants:
- `task: clm_training`, `task: mlm_training`, and `task: instruction` must still dispatch through `src/main.py` without changes to YAML configs.
- Training outputs, logging, and validation behavior must remain unchanged.
- All existing unit tests and relevant SLURM smoke tests must continue to pass.

Likely touched files:
- `src/tasks/training/fabric/model/*`
- `src/tasks/training/fabric/trainer/base.py`
- `src/tasks/training/fabric/trainer/distributed.py`
- `src/tasks/training/fabric/wrappers/*`
- `src/tasks/training/utils.py`
- `src/tasks/training/orchestrator.py`
- `tests/unit/training/test_scheduler.py`
- Documentation references under `README.md` or `docs/` if they mention the old structure.

## Open Questions
- Should the task dispatch in `src/main.py` remain grouped under `training`, or should task modules be split while keeping the task names unchanged?
- Where should Fabric-specific model base classes live relative to task-specific modules to keep dependencies clean?

## Acceptance Criteria
- Clear task-type and shared-helper directories exist under `src/tasks/training/`.
- No duplicated FSDP policy logic remains.
- Training configs validate and a smoke run can execute without behavioral change.
- Updated unit tests pass.

## ExecPlan
- Create `issues/execPlans/39-training-task-layout.execplan.md` aligned with `PLANS.md`.
