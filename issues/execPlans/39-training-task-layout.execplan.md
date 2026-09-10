# Refactor Training Task Layout and Shared Helpers

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

PLANS.md is checked into the repo at `PLANS.md`, and this document must be maintained in accordance with it.

## Purpose / Big Picture

Training in LMTK currently works, but the code layout hides task boundaries and scatters shared helpers across framework-specific modules. After this change, the training task types (CLM, MLM, instruction) live in their own task-type modules, shared helpers are centralized, and distributed strategies live in a dedicated strategies package. Users should still be able to run `python -m src.main --config tutorials/configs/clm_training_tutorial.yaml` or the SLURM smoke tests with no behavioral differences. The visible outcome is maintainability: it becomes clear where task-specific logic lives, where shared helpers belong, and where distributed strategy logic is defined.

## Progress

- [x] (2026-01-19 12:56Z) Drafted ExecPlan and captured scope from issue 39.
- [ ] (2026-01-19 12:56Z) Inventory current training layout and confirm all import sites that will move.
- [ ] (2026-01-19 12:56Z) Introduce task-type package and registry, move model classes, update trainer/orchestrator imports.
- [ ] (2026-01-19 12:56Z) Extract shared helpers and consolidate FSDP policy utilities, update tests and docs.
- [ ] (2026-01-19 12:56Z) Run unit + integration validation (pytest + SLURM smoke) and record results.

## Surprises & Discoveries

None yet.

## Decision Log

- Decision: Keep task dispatch grouped under `src/tasks/training` with the existing `src/main.py` task module map intact.
  Rationale: This preserves the public task names and avoids schema or CLI changes while still allowing internal reorganization.
  Date/Author: 2026-01-19 / Codex
- Decision: Introduce `src/tasks/training/task_types` for task-specific model modules and `src/tasks/training/shared` for shared helpers, and move current code into those packages.
  Rationale: This separates task logic from framework code and eliminates the current "utils.py" dumping ground while keeping behavior unchanged.
  Date/Author: 2026-01-19 / Codex
- Decision: Move distributed strategy classes out of `fabric/trainer/distributed.py` into `src/tasks/training/strategies` to align code layout with the `config/schemas/training/strategy/` structure.
  Rationale: Strategies are a distinct concern from the Fabric trainer implementation and deserve their own package boundary.
  Date/Author: 2026-01-19 / Codex

## Outcomes & Retrospective

Not started.

## Context and Orientation

LMTK is a YAML-driven toolkit. Each YAML configuration denotes a single job and maps to exactly one task module under `src/tasks/`. `src/main.py` loads a config, validates it against schemas under `config/schemas/`, and dispatches to a task module. The authoritative task list lives in `config/schemas/base.schema.yaml`, and `src/main.py` maps `clm_training`, `mlm_training`, and `instruction` to the shared `src/tasks/training` task module.

Training is implemented under `src/tasks/training/`. The current layout has task-specific model classes under `src/tasks/training/fabric/model/`, distributed strategies under `src/tasks/training/fabric/trainer/distributed.py`, and shared helpers (optimizers, schedulers, deterministic seeding) under `src/tasks/training/utils.py`. The Fabric trainer base (`src/tasks/training/fabric/trainer/base.py`) holds the task-to-model registry (`MODEL_CLASS_MAP`) and imports the task models directly. FSDP policy logic is duplicated between `src/tasks/training/fabric/wrappers/policies.py` and `src/tasks/training/fabric/wrappers/fsdp_config.py`.

In this plan, a "training task type" means a task value such as `clm_training`, `mlm_training`, or `instruction` that maps to a model definition and any task-specific logic. A "strategy" means the distributed training strategy selected via `parallelization_strategy` (for example FSDP, DDP, DeepSpeed, or data parallel).

## Milestones

### Milestone 1: Task-type package and registry

Create the `src/tasks/training/task_types/` package, move the CLM/MLM/Instruction model classes and their base class into it, and introduce a registry that maps task names to the model classes. Update `FabricTrainerBase` to consult the registry instead of owning the map. At the end of this milestone, task-specific model code lives outside the Fabric trainer implementation, and a focused unit test verifies the registry mapping. Acceptance is a passing `python -m pytest -q tests/unit/training/test_task_registry.py` and successful config validation via `python -m src.main --validate --config config/tests/clm_training_smoke.yaml` (or the tutorial config if preferred).

### Milestone 2: Shared helper extraction and FSDP policy consolidation

Create `src/tasks/training/shared/` and move optimizer, scheduler, seeding, and auto-wrap policy helpers into explicit modules. Update all imports in the Fabric trainer and strategy code to use the new shared modules, and remove duplicated FSDP policy helpers by consolidating them in a single shared module. At the end, there is no duplicated policy logic and `tests/unit/training/test_scheduler.py` passes using the new import path.

### Milestone 3: Strategy relocation and end-to-end validation

Move the distributed strategy classes to `src/tasks/training/strategies/`, update `src/tasks/training/orchestrator.py` imports, and adjust any remaining module references. Then run unit tests and the SLURM integration smoke run to confirm that tokenization and training still execute. Acceptance includes a successful SLURM integration smoke job (or a documented reason it could not be run) and a metrics CSV file produced by the training smoke config with a numeric loss column.

## Plan of Work

Start by auditing `src/tasks/training/` and documenting every import site that references `fabric/model`, `fabric/trainer/distributed.py`, or `training/utils.py`. Create the `src/tasks/training/task_types/` package and move the CLM/MLM/Instruction model modules plus the shared `BaseModel` and `AVAILABLE_MODELS` helper into it. Add `src/tasks/training/task_types/__init__.py` with a registry that maps task names to classes and a `get_model_class` helper. Update `FabricTrainerBase._instantiate_model` to call the registry rather than referencing `MODEL_CLASS_MAP`, and update any imports to point to the new task-type modules.

Next, create `src/tasks/training/shared/` and move shared utilities out of `src/tasks/training/utils.py`. Break helpers into focused modules (optimizers, schedulers, seeding, auto-wrap policies), update imports in the trainer and strategy code, and remove duplicated policy helpers from `fabric/wrappers/policies.py` and `fabric/wrappers/fsdp_config.py` by pointing them to the shared module. Remove or empty `src/tasks/training/utils.py` once all imports have been updated.

Then relocate the distributed strategies into `src/tasks/training/strategies/` (keeping the class names and behavior identical) and update `src/tasks/training/orchestrator.py` to import from the new package. Scan the codebase for references to the old paths (including docs and tests) and update them to match the new layout. Finally, add or update unit tests: keep `tests/unit/training/test_scheduler.py` aligned with the new shared module, and add a new registry test for task-to-model mapping.

## Concrete Steps

From the repository root, create the new packages:

    mkdir -p src/tasks/training/task_types src/tasks/training/shared src/tasks/training/strategies

Move the task model files (use `git mv` when possible):

    git mv src/tasks/training/fabric/model/base.py src/tasks/training/task_types/base.py
    git mv src/tasks/training/fabric/model/utils.py src/tasks/training/task_types/available_models.py
    git mv src/tasks/training/fabric/model/clm.py src/tasks/training/task_types/clm.py
    git mv src/tasks/training/fabric/model/mlm.py src/tasks/training/task_types/mlm.py
    git mv src/tasks/training/fabric/model/instruction.py src/tasks/training/task_types/instruction.py

Create `src/tasks/training/task_types/__init__.py` with the registry and update imports in `src/tasks/training/fabric/trainer/base.py` to use it. Then move or split shared helpers from `src/tasks/training/utils.py` into `src/tasks/training/shared/` modules and update imports everywhere they were referenced. Remove `src/tasks/training/utils.py` once no imports remain.

Move the distributed strategy classes:

    git mv src/tasks/training/fabric/trainer/distributed.py src/tasks/training/strategies/distributed.py

Update `src/tasks/training/orchestrator.py` imports to use `src.tasks.training.strategies.distributed`. Update any other references by searching:

    grep -R "tasks.training.fabric.model" -n src tests
    grep -R "tasks.training.utils" -n src tests
    grep -R "fabric.trainer.distributed" -n src tests

Add the new unit test and update existing ones:

    mkdir -p tests/unit/training
    (edit tests/unit/training/test_task_registry.py)
    (update tests/unit/training/test_scheduler.py import path)

## Validation and Acceptance

Run the unit tests that cover the moved helpers and registry:

    python -m pytest -q tests/unit/training/test_scheduler.py
    python -m pytest -q tests/unit/training/test_task_registry.py

Validate a training config:

    python -m src.main --validate --config config/tests/clm_training_smoke.yaml

For ML-facing validation, submit the integration smoke test through SLURM (if available):

    ./slurm/tests/run_tests.sh --integration

Record the job ID printed by the submit script, then confirm completion with:

    sacct -j <job_id> --format=JobID,State,ExitCode -P

Acceptance is a completed SLURM job with ExitCode `0:0`, and a metrics CSV file under `output/tests/` for the training smoke config that contains a numeric `loss` column. If SLURM submission is unavailable, document why and run `python -m src.main --validate --config config/tests/integration_smoke.yaml` as a local validation substitute.

## Idempotence and Recovery

Directory creation and file moves are safe to repeat; repeated moves should be skipped once files are in their new locations. If a move is incorrect, move the file back and restore the prior import paths. Avoid destructive git commands; prefer targeted file moves and import fixes so the work can be retried safely.

## Artifacts and Notes

Expected unit test output (example):

    1 passed in 0.10s

Expected SLURM accounting output (example):

    <job_id>|COMPLETED|0:0

## Interfaces and Dependencies

In `src/tasks/training/task_types/__init__.py`, define a registry and accessor:

    TASK_MODEL_REGISTRY: dict[str, type[lightning.LightningModule]]

    def get_model_class(task: str) -> type[lightning.LightningModule]:
        ...

In `src/tasks/training/task_types/base.py`, keep the existing `BaseModel` class behavior and update imports to use `available_models.py` for `AVAILABLE_MODELS`.

In `src/tasks/training/shared/schedulers.py`, expose:

    def select_scheduler(
        optimizer: torch.optim.Optimizer,
        lr_scheduler: str,
        number_epochs: int,
        world_size: int,
        batch_size: int,
        train_dataset: datasets.Dataset,
        warmup_proportion: float,
        gradient_accumulation_steps: int | None = None,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        ...

In `src/tasks/training/shared/optimizers.py`, expose:

    def select_optimizer(
        optimizer: str,
        model: torch.nn.Module,
        lr: float,
        weight_decay: float,
        beta1: float,
        beta2: float,
    ) -> torch.optim.Optimizer:
        ...

In `src/tasks/training/shared/seeding.py`, expose:

    def deterministic(seed: int) -> None:
        ...

In `src/tasks/training/shared/fsdp_policies.py`, expose:

    def create_auto_wrap_policy(model_name: str, model: torch.nn.Module | None = None, min_num_params: int = 1_000_000) -> typing.Callable | None:
        ...

In `src/tasks/training/strategies/distributed.py`, keep the existing class names `FSDP`, `DeepSpeed`, `DistributedDataParallel`, and `DataParallel` with the same constructor signatures and behavior, and ensure they still extend `FabricTrainerBase` from `src/tasks/training/fabric/trainer/base.py`.
