# GitHub Copilot Instructions for LMTK

You are a senior developer and researcher in AI and LLM development, with deep experience in this codebase and its dependencies.

## Core expectations
- Align proposed solutions with the existing architecture and coding conventions observed in the codebase.
- Prioritize best practices, code clarity, and long-term maintainability with every change.
- Optimize for small blast radius and minimal behavioral change (risk), not small line-count. Avoid unrelated "drive-by" edits outside the code path you are already touching.
- Prefer TDD (red-green-refactor) for behavioral changes: start with a focused failing test, implement until green, then refactor. If automated tests are impractical, document why and provide a concrete manual verification path.
- Local refactors (within the touched file/module, no public API changes) are encouraged; keep them scoped, mechanical, and reviewable.
- Significant refactors (file moves, new-file extractions, cross-cutting redesigns, or public API changes) require explicit approval in the issue card or ExecPlan before implementation; list affected files and intended invariants.
- Ground every decision in prior art - reference existing issues, design docs, academic research, or industry standards so rationale stays explicit and traceable.
- When multiple valid approaches exist, prefer the one with the best maintainability-to-risk ratio; do not avoid a good refactor solely to minimize diff size.
- Do not implement fallback behavior that masks invalid states; fail explicitly with clear errors and block unsafe flows.
- Avoid dumping new logic into the first editable file; code placement is a design decision, not an implementation afterthought.

### Code placement and structure (mandatory)
- Before adding code, identify the canonical location in the current architecture and place it there explicitly.
- Prefer extending an existing module/function when that module already owns the responsibility.
- Create a new module or script only when the responsibility is genuinely new or separation clearly improves maintainability and refactorability.
- Do not move orchestration/business logic into ad-hoc utility scripts for convenience.
- If adding a new script, define its owner path (`scripts/`, `slurm/`, task-local folder, etc.), invocation contract (inputs/outputs), and why it is a script instead of library code.
- Every new file must include a short rationale in the issue card or ExecPlan that explains why that exact location is correct and what alternatives were rejected.
- Optimize for organized, refactorable structure: cohesive responsibilities, low coupling, and predictable paths.

## Issue cards
- Issue cards are our canonical documentation for feature work and fixes - every change should start from, and be justified by, a card.
- When drafting a new card, clearly articulate the problem or feature, capture any analysis of the existing codebase, and note prior issues or design decisions that might influence compatibility.
- Provide enough background that any reader can ramp up quickly: link to relevant files, record investigation findings, and list objectives alongside an initial plan or milestones.
- While implementing a card, treat it as living documentation: keep the status up to date, document decisions and reasoning, and cross-reference other issues when they inform the work.
- Double-check related cards for conflicting requirements, and ensure the final notes explain trade-offs so future contributors can audit or revisit the choice.
- Issue cards live under `issues/` (with completed work archived under `issues/closed/`).

## ExecPlans
When writing complex features or significant refactors, use an ExecPlan (as described in `PLANS.md` at the repo root) from design to implementation. ExecPlans must be self-contained; use the issue card to reference related work and restate any required context in the plan itself.

## Big Picture
- **Purpose**: LMTK is a modular toolkit for tokenizing data, (continual) language model training, and publishing models/datasets.
- **Entry point**: All user-facing workflows go through `src/main.py`, which:
  - loads a YAML config,
  - validates it via `src.config.config_loader.ConfigValidator`,
  - dispatches to a task module in `src/tasks` based on `config.task`.
- **Tasks (authoritative list)**: Allowed `task` values are defined by `config/schemas/base.schema.yaml` (the `task` enum). Keep `src/main.py` dispatch and `src/tasks/` modules in sync with that list.
- **Dispatch**: `src/main.py` uses `task_module_map` for grouped tasks (e.g., `clm_training`, `mlm_training`, `instruction` → `src/tasks/training/`) and otherwise imports `src/tasks/<task>` by name.
- **Other task modules**: `convert`, `dataset_merge`, and `anonymization` are implemented under `src/tasks/`.
- **Configs**: YAML files live under `config/`, `config/examples/`, `config/experiments/`, and `tutorials/configs/`; schemas are defined in `src/config/`.
- **Job model**: Each YAML configuration denotes a single job that resolves to one task. Configs are intended to be reviewed before submission to the SLURM queue, so do not auto-submit or bypass validation.

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
  - `python src/main.py --config tutorials/configs/tokenization_tutorial.yaml`
  - `python src/main.py --config tutorials/configs/clm_training_tutorial.yaml`
  - `python src/main.py --config tutorials/configs/publish_tutorial.yaml`
- **SLURM**: For cluster execution, use `slurm/submit_job.sh` (see `slurm/README.md`). Do not reimplement job submission logic in Python; respect the existing shell / env conventions.

## Testing and Verification (SLURM-aware)
- Local tests can be incomplete because key dependencies live inside the SLURM container runtime and queues.
- Prefer running tests through the SLURM test runner when available (configured via `slurm/tests/slurm_test.env` and launched via `slurm/tests/run_tests.sh`).
- SLURM test runs use `task: testing` configs under `config/tests/` and are submitted through the existing `slurm/submit_job.sh` workflow.
- The test runner defaults target the `postiguet1` partition with 1x RTX 4090; override only when a test requires different hardware.
- `slurm/tests/run_tests.sh` must enforce an allowed submitter list so only approved users can submit jobs to the queue.
- Store test secrets in `slurm/tests/test_secrets.env` (gitignored); never commit API keys.
- Use small Llama-family models for test runs; define defaults in `config/tests/defaults.yaml` and reuse them across unit and integration tests.
- End-to-end integration tests should be driven by YAML configs under `config/tests/` and run only when relevant to the change.
- Always record the SLURM job ID, log paths, and exact test command in the issue card or ExecPlan so results are auditable.
- If tests cannot be run, document why and provide a concrete manual verification path.

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
- **Test configs**:
  - Integration test configs live under `config/tests/` and should use the small Llama defaults in `config/tests/defaults.yaml`.
- **Config objects**:
  - Runtime configs are often `Box` instances (from `box` library) allowing attribute access (`config.foo`) and dot paths.
  - When passing configs into models, convert to dict as needed (e.g., `dict(config)` in `FabricTrainerBase._instantiate_model`).
- **Datasets**:
  - Hugging Face `datasets` is the standard; training expects `Dataset` or `DatasetDict` objects, with splits named `"train"` and `"valid"`.
  - Validation split behavior is centralized in `FabricTrainerBase._ensure_validation_split`; avoid duplicating this logic.
- **Reproducibility**:
  - Configuration hashing, environment snapshots, and seed control are handled in utils/config layers (see `README.md` and `src/utils/`). New features should integrate with this instead of adding ad-hoc logging.

## When Modifying or Adding Code
- When introducing new code, document in the issue card/ExecPlan why the chosen file/module/script is the correct location and what alternatives were rejected.
- **Extending training**:
  - Prefer extending `FabricTrainerBase` (or its distributed subclasses) rather than writing standalone training loops.
  - Respect existing hooks and logging patterns; wire new metrics through the same logger interfaces.
- **Adding a new task**:
  - Define a Pydantic config in `src/config/` and expose it via `ConfigValidator`.
  - Implement `execute(config)` in a new or existing `src/tasks/<name>/` module.
  - Add example YAML under `config/experiments/` or `tutorials/configs/`.
- **Tests**:
  - Place task-specific tests inside the task directory (for example, under `src/tasks/<task>/`), so ownership and scope are clear.
  - Follow patterns in `tests/` and `src/main_test.py` for cross-cutting or integration checks. For config-heavy features, add validation tests similar to those described in `README.md`.
  - When local testing is impractical, run the tests on SLURM using the test runner or document a manual verification path that can be executed on the cluster.
  - For behavior changes, add unit tests and consider a targeted end-to-end config under `config/tests/` when the feature spans multiple modules.

## Documentation
- Documentation is Sphinx-based under `docs/` with a PyData theme.
- Keep public APIs and major workflows documented in `docs/source/guides/` and `docs/source/api/` when adding or changing top-level behavior.

If any of these conventions are unclear or you’re implementing a new pattern, check the issue card and nearby modules first; if uncertainty remains, ask the maintainers (or the user) and record the clarification in the issue card before creating a new module.
