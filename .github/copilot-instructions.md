# GitHub Copilot Instructions for LMTK

These guidelines help AI coding agents work effectively in this repository.

## Big Picture
- **Purpose**: LMTK is a modular toolkit for tokenizing data, (continual) language model training, and publishing models/datasets.
- **Entry point**: All user-facing workflows go through `src/main.py`, which:
  - loads a YAML config,
  - validates it via `src.config.config_loader.ConfigValidator`,
  - dispatches to a task module in `src/tasks` based on `config.task`.
- **Primary tasks** (YAML `task` values → modules):
  - `tokenization` → `src/tasks/tokenization/`
  - `clm_training`, `mlm_training`, `instruction` → `src/tasks/training/`
  - `publish` → `src/tasks/publish/`
- **Configs**: YAML files live under `config/`, `config/examples/`, `config/experiments/`, and `tutorials/configs/`; schemas are defined in `src/config/`.

## Development & CI Workflows
- **Environment**: Prefer Conda + Poetry (see root `README.md`). Typical setup:
  - `make install` (Poetry) or `make install-pip` (pure pip).
- **Core commands** (mirror CI in `.github/workflows/ci.yml`):
  - Lint: `python -m tox -e lint`
  - Type check: `python -m tox -e type`
  - Tests: `python -m tox -e py310`
  - Coverage: `python -m tox -e coverage`
  - Docs: `python -m tox -e docs` or `./build_docs.sh`.
- **Quick task runs** (local):
  - `python -m src.main --config tutorials/configs/tokenization_tutorial.yaml`
  - `python -m src.main --config tutorials/configs/clm_training_tutorial.yaml`
  - `python -m src.main --config tutorials/configs/publish_tutorial.yaml`
- **SLURM**: For cluster execution, use `slurm/submit_job.sh` (see `slurm/README.md`). Do not reimplement job submission logic in Python; respect the existing shell / env conventions.

## Architectural Patterns
- **Task abstraction**:
  - Each high-level task exposes an `execute(config)` function under `src/tasks/<task_group>/__init__.py` or similar.
  - New task types should follow this pattern and register a `task` string in `src/main.py`'s `task_module_map` and the config schemas.
- **Configuration**:
  - Validation and typing are centralized in `src/config/` via Pydantic models.
  - When changing config structure, update both the schema and any YAML examples under `config/` / `tutorials/configs/`.
- **Training (Fabric)**:
  - Training is implemented with Lightning Fabric under `src/tasks/training/fabric/`.
  - `FabricTrainerBase` in `src/tasks/training/fabric/trainer/base.py` encapsulates:
    - dataset loading and validation split logic,
    - strategy setup hooks,
    - logging (CSV + optional WandB),
    - training loop, checkpointing, and evaluation.
  - Model variants are mapped via `MODEL_CLASS_MAP` to classes like `FabricCLM`, `FabricMLM`, `FabricInstruction` in `src/tasks/training/fabric/model/`.
  - Distributed strategies (FSDP, DeepSpeed, DDP, DataParallel) are configured via subclasses in `src/tasks/training/fabric/trainer/distributed.py` and config fields.
- **Logging & Monitoring**:
  - CLI logging uses `utils.logging.get_logger` with a `verbose_level` from config.
  - Training logs use `step_csv_logger` and optional WandB (`create_wandb_logger`) from `src/tasks/training/fabric/logger.py`.
  - Speed / throughput metrics are implemented in `src/tasks/training/fabric/speed_monitor.py`.

## Project-Specific Conventions
- **Paths & caches**:
  - `src/main.py` sets HF and WandB cache dirs under project-local `.cache2/` and `tmp/`; new code should reuse these env vars instead of hardcoding cache locations.
- **Config objects**:
  - Runtime configs are often `Box` instances (from `box` library) allowing attribute access (`config.foo`) and dot paths.
  - When passing configs into models, convert to dict as needed (e.g., `dict(config)` in `FabricTrainerBase._instantiate_model`).
- **Datasets**:
  - Hugging Face `datasets` is the standard; training expects `Dataset` or `DatasetDict` objects, with splits named `"train"` and `"valid"`.
  - Validation split behavior is centralized in `FabricTrainerBase._ensure_validation_split`; avoid duplicating this logic.
- **Reproducibility**:
  - Configuration hashing, environment snapshots, and seed control are handled in utils/config layers (see `README.md` and `src/utils/`). New features should integrate with this instead of adding ad-hoc logging.

## When Modifying or Adding Code
- **Extending training**:
  - Prefer extending `FabricTrainerBase` (or its distributed subclasses) rather than writing standalone training loops.
  - Respect existing hooks and logging patterns; wire new metrics through the same logger interfaces.
- **Adding a new task**:
  - Define a Pydantic config in `src/config/` and expose it via `ConfigValidator`.
  - Implement `execute(config)` in a new or existing `src/tasks/<name>/` module.
  - Add example YAML under `config/experiments/` or `tutorials/configs/`.
- **Tests**:
  - Follow patterns in `tests/` and `src/main_test.py`. For config-heavy features, add validation tests similar to those described in `README.md`.

## Documentation
- Documentation is Sphinx-based under `docs/` with a PyData theme.
- Keep public APIs and major workflows documented in `docs/source/guides/` and `docs/source/api/` when adding or changing top-level behavior.

If any of these conventions are unclear or you’re implementing a new pattern, ask the maintainers (or the user) which existing module to mirror before creating a new one.