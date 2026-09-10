# Add SLURM-Based Test Runner and Test Configs

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

PLANS.md is checked into the repo at `PLANS.md`, and this document must be maintained in accordance with it.

## Purpose / Big Picture

LMTK relies on SLURM-only runtimes, so local tests are often incomplete. This plan pivots to a config-driven testing task that runs through the same `src/main.py` entrypoint and `slurm/submit_job.sh` path as production jobs. The result is a repeatable, auditable test workflow that uses standard SLURM submission while keeping unit tests and integration smoke runs separate, lightweight, and consistent with the task-based architecture.

## Progress

- [x] (2026-01-19 08:57Z) Reviewed and updated `AGENTS.md` and `PLANS.md` to include the testing task workflow.
- [x] (2026-01-19 08:57Z) Added the `testing` task to `config/schemas/base.schema.yaml` and created `config/schemas/testing.schema.yaml`.
- [x] (2026-01-19 08:57Z) Implemented `src/tasks/testing/` with a testing orchestrator that runs unit or integration tests.
- [x] (2026-01-19 08:57Z) Added `config/tests/defaults.yaml`, `config/tests/unit_tests.yaml`, and smoke configs under `config/tests/`.
- [x] (2026-01-19 08:57Z) Added `slurm/tests/slurm_test.env`, `slurm/tests/test_secrets.env.example`, and updated `.gitignore` for `slurm/tests/test_secrets.env`/`slurm/tests/logs/`.
- [x] (2026-01-19 08:57Z) Updated `slurm/tests/run_tests.sh` to wrap `slurm/submit_job.sh` and enforce the allowed submitter list.
- [x] (2026-01-19 08:57Z) Added `tests/unit/config/test_test_configs_defaults.py` to enforce defaults alignment.
- [x] (2026-01-19 08:57Z) Updated `README.md` and `slurm/README.md` to document the testing workflow and secrets split.
- [x] (2026-01-19 09:02Z) Updated `slurm/submit_job.sh` to honor test log output patterns supplied via `slurm/tests/slurm_test.env`.
- [x] (2026-01-19 11:12Z) Added a `SLURM_CONF` fallback in `slurm/tests/run_tests.sh` for non-interactive shells.
- [x] (2026-01-19 11:18Z) Fixed `slurm/p.slurm` working directory and updated `src/tasks/testing/orchestrator.py` to run `src/main.py` directly.
- [x] (2026-01-19 11:20Z) Updated `config/tests/clm_training_smoke.yaml` to use `parallelization_strategy: dp`.
- [x] (2026-01-19 12:05Z) Fixed `select_scheduler` to return a constant scheduler for `lr_scheduler: fixed`.
- [x] (2026-01-19 12:07Z) Added `tests/unit/training/test_scheduler.py` to cover the fixed scheduler path.
- [x] (2026-01-19 12:08Z) Ran `python -m pytest -q tests/unit/training/test_scheduler.py` in the `lmtk` conda env (pass).
- [x] (2026-01-19 12:35Z) Installed `pytest` in the `lmtk` conda env for unit test execution.
- [x] (2026-01-19 13:00Z) Fixed publish validation defaults and HuggingFace upload validation to satisfy unit tests.
- [x] (2026-01-19 13:05Z) Ran targeted unit tests for publish/version utilities (pass).
- [x] (2026-01-19 13:35Z) SLURM integration job 34881 completed successfully (ExitCode 0:0).
- [x] (2026-01-19 13:41Z) SLURM unit job 34883 completed successfully (30 passed).

## Surprises & Discoveries

- Observation: `slurm/submit_job.sh` overwrote environment-provided output patterns, so test log paths were not applied until we explicitly honored them.
  Evidence: `run_tests.sh --dry-run` initially showed `--output=%j_lmtk.out` despite `slurm/tests/slurm_test.env` specifying `slurm/tests/logs/tests-%j.out`.
- Observation: Local config validation failed because the runtime lacks the `box` dependency required by `src.main`.
  Evidence: `ModuleNotFoundError: No module named 'box'` when running `python3 -m src.main --validate --config config/tests/unit_tests.yaml`.
- Observation: SLURM submissions failed in this environment because `sbatch` cannot locate the controller via DNS SRV.
  Evidence: `sbatch: error: resolve_ctls_from_dns_srv: res_nsearch error: Unknown host`.
- Observation: Non-interactive shells lacked `SLURM_CONF`, causing SLURM CLI tools to fall back to DNS SRV lookup.
  Evidence: `squeue` failed until `SLURM_CONF=/etc/slurm/slurm.conf` was set.
- Observation: `slurm/p.slurm` changed into the parent directory, so `python3 src/main.py` resolved to `/home/gplsi/GPLSI/codigos/src/main.py` and failed.
  Evidence: `python3: can't open file '/home/gplsi/GPLSI/codigos/src/main.py'`.
- Observation: `python -m src.main` failed because `src` is not a Python package.
  Evidence: `ERROR: No module named 'tasks'` when running the integration smoke config.
- Observation: `pytest` was missing in the `lmtk` conda env.
  Evidence: `/home/gplsi/rst29/anaconda3/envs/lmtk/bin/python3: No module named pytest`.
- Observation: `parallelization_strategy: none` is permitted by schema but rejected by the training orchestrator.
  Evidence: `Invalid parallelization strategy: none` in `slurm/tests/logs/tests-34858.out`.

## Decision Log

- Decision: `.agent/PLANS.md` is not present in this repository; follow `PLANS.md` at the repo root.
  Rationale: The plan must remain compliant with the repo's documented ExecPlan requirements, which live in `PLANS.md`.
  Date/Author: 2026-01-19 / Codex
- Decision: Pivot to a config-driven `task: testing` so test runs use the same `src/main.py` and `slurm/submit_job.sh` path as production jobs.
  Rationale: Aligns with the YAML task model and reuses the existing, tested submission flow.
  Date/Author: 2026-01-13 / Codex
- Decision: Keep `slurm/tests/run_tests.sh` as a thin wrapper to source test env/secrets and enforce the allowed submitter list before calling `slurm/submit_job.sh`.
  Rationale: The allowed submitter guard must run before submission, and `submit_job.sh` should remain unchanged.
  Date/Author: 2026-01-13 / Codex
- Decision: Allow `slurm/submit_job.sh` to honor `OUTPUT_FILE_PATTERN` and `ERROR_FILE_PATTERN` from the environment when set by `slurm/tests/slurm_test.env`.
  Rationale: Test log paths must resolve to the expected `slurm/tests/logs/` location without rewriting submit logic elsewhere.
  Date/Author: 2026-01-19 / Codex
- Decision: Set `SLURM_CONF` inside `slurm/tests/run_tests.sh` when it is missing.
  Rationale: Ensures `sbatch`/`squeue` work in non-interactive shells without relying on user shell init.
  Date/Author: 2026-01-19 / Codex
- Decision: Execute integration configs via `python <project_root>/src/main.py` instead of `python -m src.main`.
  Rationale: `src` lacks `__init__.py`, so module execution fails on the cluster.
  Date/Author: 2026-01-19 / Codex
- Decision: Change `slurm/p.slurm` to `cd "$HOST_PROJECT_ROOT"` before running the main script.
  Rationale: Avoids resolving `src/main.py` from the wrong directory.
  Date/Author: 2026-01-19 / Codex
- Decision: Use `parallelization_strategy: dp` in the CLM training smoke config.
  Rationale: `none` is rejected by the orchestrator, and `dp` works for single-device runs.
  Date/Author: 2026-01-19 / Codex
- Decision: Default test partition remains `postiguet1` with 1x RTX 4090.
  Rationale: Matches the known test hardware and keeps runs consistent.
  Date/Author: 2026-01-13 / Codex
- Decision: Store test secrets in `slurm/tests/test_secrets.env` and keep production secrets in normal env/CLI paths.
  Rationale: Keeps credentials isolated and avoids accidental reuse or commits.
  Date/Author: 2026-01-13 / Codex
- Decision: Keep integration smoke configs under `config/tests/` and run them only when relevant.
  Rationale: Preserves fast feedback while keeping end-to-end coverage available.
  Date/Author: 2026-01-13 / Codex

## Outcomes & Retrospective

Not started.

## Context and Orientation

LMTK is a YAML-driven toolkit. Each YAML configuration denotes a single job that maps to exactly one task module under `src/tasks/`. `src/main.py` loads the YAML config, validates it against JSON schemas in `config/schemas/`, and dispatches to the task module. The authoritative task list is in `config/schemas/base.schema.yaml`, so adding a new task requires adding it to that enum and providing a matching schema file. SLURM submission runs through `slurm/submit_job.sh`, which feeds `slurm/p.slurm` and ultimately calls `python -m src.main --config <yaml>`.

This plan adds a new `task: testing` that executes unit tests or integration smoke runs inside the SLURM job. Test-specific SLURM defaults and secrets live under `slurm/tests/`, while production secrets remain in the standard environment or command-line flags. `sacct` is the SLURM accounting command used to query job state and exit codes; this plan uses it to verify test runs completed successfully.

## Milestones

Milestone 1 adds the new testing task and schema. By the end, `config/schemas/testing.schema.yaml` exists, `config/schemas/base.schema.yaml` lists the `testing` task, and `src/tasks/testing/` can execute unit or integration runs. Validate with `python -m src.main --validate --config config/tests/unit_tests.yaml` and expect a successful validation message.

Milestone 2 adds test defaults, smoke configs, and the wrapper script. `slurm/tests/slurm_test.env`, `slurm/tests/test_secrets.env.example`, and `slurm/tests/run_tests.sh` exist, and `tests/unit/config/test_test_configs_defaults.py` enforces defaults alignment. Run `python -m pytest -q tests/unit/config/test_test_configs_defaults.py` and expect a passing test.

Milestone 3 documents the workflow and validates it on SLURM. Update `README.md` and `slurm/README.md` with the testing task and secrets split. Submit a unit test run with `./slurm/tests/run_tests.sh --unit` and an integration smoke run with `./slurm/tests/run_tests.sh --integration`. Capture job IDs and log paths in this plan.

## Plan of Work

Add the `testing` task to `config/schemas/base.schema.yaml` and create `config/schemas/testing.schema.yaml` with explicit validation for unit versus integration mode. Implement `src/tasks/testing/` with a `TestingOrchestrator` that validates the config, runs `pytest` for unit mode, and sequentially runs integration configs using `python -m src.main --config <config>` with explicit error handling and optional stop-on-failure behavior. Create `config/tests/unit_tests.yaml` for unit runs, `config/tests/defaults.yaml` for shared Llama defaults, and the smoke configs under `config/tests/`. Update `slurm/tests/slurm_test.env`, add `slurm/tests/test_secrets.env.example`, update `.gitignore`, and adjust `slurm/tests/run_tests.sh` to source the test env, enforce the allowed submitter list, and call `slurm/submit_job.sh` with the selected test config. Ensure `slurm/submit_job.sh` honors `OUTPUT_FILE_PATTERN` and `ERROR_FILE_PATTERN` when provided via `slurm/tests/slurm_test.env` so logs land under `slurm/tests/logs/`. Add `tests/unit/config/test_test_configs_defaults.py` to keep defaults aligned. Finally, update `README.md` and `slurm/README.md` to explain how to run unit versus integration tests and how to use test secrets.

## Concrete Steps

From the repository root, update schemas and add the testing task:

    cat > config/schemas/testing.schema.yaml
    (content from Artifacts and Notes)
    Ctrl-D

Update `config/schemas/base.schema.yaml` to include `testing` in the `task` enum.

Create the testing task module:

    mkdir -p src/tasks/testing
    cat > src/tasks/testing/__init__.py
    (implementation described in Interfaces and Dependencies)
    Ctrl-D

    cat > src/tasks/testing/orchestrator.py
    (implementation described in Interfaces and Dependencies)
    Ctrl-D

Add the test configs and defaults:

    mkdir -p config/tests
    cat > config/tests/defaults.yaml
    (content from Artifacts and Notes)
    Ctrl-D

    cat > config/tests/unit_tests.yaml
    (content from Artifacts and Notes)
    Ctrl-D

    cat > config/tests/integration_smoke.yaml
    (content from Artifacts and Notes)
    Ctrl-D

    cat > config/tests/tokenization_smoke.yaml
    (content from Artifacts and Notes)
    Ctrl-D

    cat > config/tests/clm_training_smoke.yaml
    (content from Artifacts and Notes)
    Ctrl-D

Add the test env files and wrapper:

    mkdir -p slurm/tests
    cat > slurm/tests/slurm_test.env
    (content from Artifacts and Notes)
    Ctrl-D

    cat > slurm/tests/test_secrets.env.example
    (content from Artifacts and Notes)
    Ctrl-D

Update `.gitignore` to ignore `slurm/tests/test_secrets.env`.

Create or update the wrapper script:

    cat > slurm/tests/run_tests.sh
    (implementation described in Interfaces and Dependencies)
    Ctrl-D
    chmod +x slurm/tests/run_tests.sh

Add the defaults alignment test:

    mkdir -p tests/unit/config
    cat > tests/unit/config/test_test_configs_defaults.py
    (implementation described in Interfaces and Dependencies)
    Ctrl-D

Update `README.md` and `slurm/README.md` with the new testing workflow and secrets split.

## Validation and Acceptance

Run `python -m src.main --validate --config config/tests/unit_tests.yaml` and expect validation to succeed. Run `python -m pytest -q tests/unit/config/test_test_configs_defaults.py` and expect a passing test. Submit a unit test run with `./slurm/tests/run_tests.sh --unit` and capture the job ID printed by `slurm/submit_job.sh`. Verify completion with `sacct -j <job_id> --format=JobID,State,ExitCode -P` and expect `COMPLETED|0:0`. Confirm the log file contains a pytest summary like `X passed`.

Submit an integration smoke run with `./slurm/tests/run_tests.sh --integration`, capture the job ID, and verify the logs show tokenization completing before training starts and that `output/tests/tokenized` exists. Confirm training finishes and the CSV metrics file exists at `output/tests/<model_name>/version_0/metrics.csv` with a numeric `loss` column. If SLURM submission is unavailable, run `python -m src.main --validate --config config/tests/tokenization_smoke.yaml` and `python -m src.main --validate --config config/tests/clm_training_smoke.yaml` and record why SLURM validation was skipped.

## Idempotence and Recovery

Re-running `slurm/tests/run_tests.sh` is safe and produces new job IDs and logs. If a run fails, adjust the test config or SLURM defaults in `slurm/tests/slurm_test.env` and resubmit. If a job needs to be canceled, use `scancel <jobid>`.

## Artifacts and Notes

Example `slurm/tests/slurm_test.env` defaults (edit as needed for your cluster):

    export JOB_NAME="lmtk-tests"
    export PARTITION="postiguet1"  # 1x RTX 4090 test partition
    export GPU_COUNT="1"
    export MEMORY="32G"
    export TIME_LIMIT="02:00:00"
    export CPUS_PER_TASK="8"
    export NODES="1"
    export NTASKS_PER_NODE="1"
    export OUTPUT_FILE_PATTERN="slurm/tests/logs/tests-%j.out"
    export ERROR_FILE_PATTERN="slurm/tests/logs/tests-%j.err"
    export ALLOWED_SUBMITTERS="estevanell"
    export TEST_UNIT_CONFIG="config/tests/unit_tests.yaml"
    export TEST_INTEGRATION_CONFIG="config/tests/integration_smoke.yaml"

Example `slurm/tests/test_secrets.env.example`:

    export WANDB_API_KEY="replace_me"
    export HUGGINGFACE_API_KEY="replace_me"

Example `config/schemas/testing.schema.yaml`:

    $schema: "http://json-schema.org/draft-07/schema#"
    $id: "testing.schema.yaml"
    description: "Schema for SLURM-based test runner configurations"

    allOf:
      - $ref: "file:///workspace/config/schemas/base.schema.yaml"
      - type: object
        properties:
          testing:
            type: object
            properties:
              mode:
                type: string
                enum: ["unit", "integration"]
              command:
                type: string
                description: "Shell command to run unit tests"
              configs:
                type: array
                items:
                  type: string
                description: "List of config paths to run for integration testing"
              stop_on_failure:
                type: boolean
                default: true
            required:
              - mode
            allOf:
              - if:
                  properties:
                    mode:
                      const: "unit"
                then:
                  required: ["command"]
              - if:
                  properties:
                    mode:
                      const: "integration"
                then:
                  required: ["configs"]
        required:
          - testing

Example `config/tests/unit_tests.yaml`:

    task: testing
    experiment_name: test_unit_tests
    verbose_level: 1
    testing:
      mode: unit
      command: "python3 -m pytest -q tests src/tasks"

Example `config/tests/integration_smoke.yaml`:

    task: testing
    experiment_name: test_integration_smoke
    verbose_level: 1
    testing:
      mode: integration
      configs:
        - config/tests/tokenization_smoke.yaml
        - config/tests/clm_training_smoke.yaml
      stop_on_failure: true

Example `config/tests/defaults.yaml`:

    model_name: hf-internal-testing/tiny-random-LlamaForCausalLM
    tokenizer_name: hf-internal-testing/llama-tokenizer
    precision: bf16-true
    context_length: 128
    overlap: 16
    seed: 42

Example `config/tests/tokenization_smoke.yaml`:

    task: tokenization
    experiment_name: test_tokenization_smoke
    verbose_level: 1
    seed: 42

    tokenizer:
      tokenizer_name: hf-internal-testing/llama-tokenizer
      task: clm_training
      context_length: 128
      overlap: 16
      batch_size: 64
      num_proc: 2
      show_progress: false

    dataset:
      source: local
      nameOrPath: tutorials/data/raw_text_data
      format: files

    output:
      path: output/tests/tokenized
      format: hf
      split: true
      shuffle: true
      seed: 42

    test_size: 0

Example `config/tests/clm_training_smoke.yaml`:

    task: clm_training
    experiment_name: test_clm_training_smoke
    verbose_level: 1
    model_name: hf-internal-testing/tiny-random-LlamaForCausalLM
    precision: bf16-true
    seed: 42

    dataset:
      source: local
      format: hf
      nameOrPath: output/tests/tokenized

    validation_split:
      proportion: 0.1
      shuffle: true
      seed: 42

    number_epochs: 1
    batch_size: 1
    num_workers: 0
    validate_after_epoch: true
    validate_on_end: true
    validations_per_epoch: 1
    save_on_validate: false
    save_on_end: false
    output_dir: output/tests

    gradient_accumulation: false
    gradient_accumulation_steps: 1
    grad_clip: 1.0
    lr: 2.0e-05
    lr_decay: false
    weight_decay: 0.0
    beta1: 0.9
    beta2: 0.95
    lr_scheduler: fixed
    warmup_proportion: 0.0

    log_iter_interval: 1
    logging_config: none
    parallelization_strategy: dp

Dry-run output (secrets redacted):

    ./slurm/tests/run_tests.sh --dry-run
    Selected config: config/tests/unit_tests.yaml
    ===== LMTK Job Submission Summary =====
    Job Name: lmtk-tests
    Partition: postiguet1
    GPU Count: 1
    Memory: 32G
    Time Limit: 02:00:00
    CPUs per Task: 8
    =====
    DRY RUN - Command that would be executed:
    sbatch --job-name=lmtk-tests --partition=postiguet1 --gres=gpu:1 --mem=32G --time=02:00:00 --cpus-per-task=8 --nodes=1 --ntasks-per-node=1 --output=slurm/tests/logs/tests-%j.out --error=slurm/tests/logs/tests-%j.err --export=CONFIG_FILE=config/tests/unit_tests.yaml,HOST_PROJECT_ROOT=/home/gplsi/GPLSI/codigos/LMTK,WANDB_API_KEY=[REDACTED],HUGGINGFACE_API_KEY=[REDACTED],GPU_COUNT=1,MEMORY=32G,TIME_LIMIT=02:00:00,PARTITION=postiguet1,JOB_NAME=lmtk-tests,CPUS_PER_TASK=8,NODES=1,NTASKS_PER_NODE=1 /home/gplsi/GPLSI/codigos/LMTK/slurm/p.slurm

Defaults alignment test:

    python3 -m pytest -q tests/unit/config/test_test_configs_defaults.py
    ..                                                                       [100%]
    2 passed in 0.49s

Scheduler unit test:

    python -m pytest -q tests/unit/training/test_scheduler.py
    .                                                                        [100%]
    1 passed in 72.33s

Config validation attempt (dependency missing in this environment):

    python3 -m src.main --validate --config config/tests/unit_tests.yaml
    ModuleNotFoundError: No module named 'box'

SLURM unit test attempt (secrets redacted):

    ./slurm/tests/run_tests.sh --unit
    Selected config: config/tests/unit_tests.yaml
    Running: sbatch --job-name=lmtk-tests --partition=postiguet1 --gres=gpu:1 --mem=32G --time=02:00:00 --cpus-per-task=8 --nodes=1 --ntasks-per-node=1 --output=slurm/tests/logs/tests-%j.out --error=slurm/tests/logs/tests-%j.err --export=CONFIG_FILE=config/tests/unit_tests.yaml,HOST_PROJECT_ROOT=/home/gplsi/GPLSI/codigos/LMTK,WANDB_API_KEY=[REDACTED],HUGGINGFACE_API_KEY=[REDACTED],GPU_COUNT=1,MEMORY=32G,TIME_LIMIT=02:00:00,PARTITION=postiguet1,JOB_NAME=lmtk-tests,CPUS_PER_TASK=8,NODES=1,NTASKS_PER_NODE=1 /home/gplsi/GPLSI/codigos/LMTK/slurm/p.slurm
    sbatch: error: resolve_ctls_from_dns_srv: res_nsearch error: Unknown host
    sbatch: error: fetch_config: DNS SRV lookup failed
    sbatch: error: _establish_config_source: failed to fetch config
    sbatch: fatal: Could not establish a configuration source

SLURM integration attempt (secrets redacted):

    ./slurm/tests/run_tests.sh --integration
    Selected config: config/tests/integration_smoke.yaml
    Running: sbatch --job-name=lmtk-tests --partition=postiguet1 --gres=gpu:1 --mem=32G --time=02:00:00 --cpus-per-task=8 --nodes=1 --ntasks-per-node=1 --output=slurm/tests/logs/tests-%j.out --error=slurm/tests/logs/tests-%j.err --export=CONFIG_FILE=config/tests/integration_smoke.yaml,HOST_PROJECT_ROOT=/home/gplsi/GPLSI/codigos/LMTK,WANDB_API_KEY=[REDACTED],HUGGINGFACE_API_KEY=[REDACTED],GPU_COUNT=1,MEMORY=32G,TIME_LIMIT=02:00:00,PARTITION=postiguet1,JOB_NAME=lmtk-tests,CPUS_PER_TASK=8,NODES=1,NTASKS_PER_NODE=1 /home/gplsi/GPLSI/codigos/LMTK/slurm/p.slurm
    sbatch: error: resolve_ctls_from_dns_srv: res_nsearch error: Unknown host
    sbatch: error: fetch_config: DNS SRV lookup failed
    sbatch: error: _establish_config_source: failed to fetch config
    sbatch: fatal: Could not establish a configuration source

SLURM unit test attempt (post `SLURM_CONF`, secrets redacted):

    ./slurm/tests/run_tests.sh --unit
    Submitted batch job 34855
    Logs: slurm/tests/logs/tests-34855.out, slurm/tests/logs/tests-34855.err
    Result: pytest missing in conda env (`No module named pytest`).

SLURM integration attempt (post `SLURM_CONF`, secrets redacted):

    ./slurm/tests/run_tests.sh --integration
    Submitted batch job 34858
    Logs: slurm/tests/logs/tests-34858.out, slurm/tests/logs/tests-34858.err
    Result: invalid `parallelization_strategy: none` in CLM smoke config.

SLURM unit test re-run (post pytest install, pending):

    ./slurm/tests/run_tests.sh --unit
    Submitted batch job 34862
    Logs: slurm/tests/logs/tests-34862.out, slurm/tests/logs/tests-34862.err
    Status: FAILED (missing publish/version test expectations; see 34862 logs).

SLURM integration re-run (post config fix, pending):

    ./slurm/tests/run_tests.sh --integration
    Submitted batch job 34864
    Logs: slurm/tests/logs/tests-34864.out, slurm/tests/logs/tests-34864.err
    Status: PENDING (Priority)

SLURM integration re-run (post scheduler fix, running):

    ./slurm/tests/run_tests.sh --integration
    Submitted batch job 34881
    Logs: slurm/tests/logs/tests-34881.out, slurm/tests/logs/tests-34881.err
    Status: COMPLETED (ExitCode 0:0)

SLURM unit test re-run (post publish/version fixes):

    ./slurm/tests/run_tests.sh --unit
    Submitted batch job 34883
    Logs: slurm/tests/logs/tests-34883.out, slurm/tests/logs/tests-34883.err
    Result: 30 passed, 2 warnings
    Status: COMPLETED (ExitCode 0:0)

## Interfaces and Dependencies

`src/tasks/testing/__init__.py` should expose `execute(config: Box) -> None` and call `TestingOrchestrator(config).execute()`. `src/tasks/testing/orchestrator.py` should define a `TestingOrchestrator` that validates the `testing` section, logs actions using the same logger patterns as other tasks, and runs tests with explicit error handling. For unit mode, it should execute the configured command via `subprocess.run(..., shell=True, check=True)` and raise a clear error if the command fails. For integration mode, it should run each config in order using `subprocess.run([sys.executable, "-m", "src.main", "--config", config_path], check=True)` from the project root, and it should stop on the first failure when `stop_on_failure` is true.

`slurm/tests/run_tests.sh` should source `slurm/tests/slurm_test.env` and optionally `slurm/tests/test_secrets.env`, check `ALLOWED_SUBMITTERS` before submission, select a config based on `--unit`, `--integration`, or `--config <path>`, and call `./slurm/submit_job.sh -c <config>` while passing through resource overrides (partition, GPUs, time, memory, CPUs, job name, nodelist) and `--dry-run`. `slurm/submit_job.sh` must honor `OUTPUT_FILE_PATTERN` and `ERROR_FILE_PATTERN` when they are set by `slurm/tests/slurm_test.env`. `slurm/p.slurm` does not need changes for this pivot.

Plan revision note: Pivoted from `RUN_MODE=test` to a config-driven `task: testing` that reuses `slurm/submit_job.sh`, and updated milestones, steps, and artifacts accordingly to keep the plan junior-proof.
Plan revision note: Marked completed implementation steps, recorded the missing `.agent/PLANS.md`, and updated the unit test command example.
Plan revision note: Recorded the `submit_job.sh` adjustment to honor test log output patterns.
Plan revision note: Added evidence for local validation, dry-run output, and SLURM submission failures with secrets redacted.
