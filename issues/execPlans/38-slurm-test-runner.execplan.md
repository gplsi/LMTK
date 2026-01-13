# Add SLURM-Based Test Runner and Test Configs

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

PLANS.md is checked into the repo at `PLANS.md`, and this document must be maintained in accordance with it.

## Purpose / Big Picture

LMTK depends on SLURM-only GPU runtimes and containers, so local tests are often incomplete or impossible. This plan creates a dedicated SLURM test runner, standard test defaults (including a small Llama-family model), and minimal end-to-end YAML configs so every change can be validated with predictable job submission, logs, and results. The outcome is a repeatable testing workflow that future changes can always use, while keeping integration tests targeted and fast.

## Progress

- [ ] (2026-01-13 11:59Z) Confirm whether `AGENTS.md` and `PLANS.md` already reflect the SLURM test runner policy; if not, update them.
- [ ] (2026-01-13 11:59Z) Add `slurm/tests/slurm_test.env` and `slurm/tests/test_secrets.env.example`, update `.gitignore` for `slurm/tests/test_secrets.env`.
- [ ] (2026-01-13 11:59Z) Implement `slurm/tests/run_tests.sh` with the submitter guard, CLI overrides, and `RUN_MODE=test`.
- [ ] (2026-01-13 11:59Z) Extend `slurm/p.slurm` for `RUN_MODE=test` while leaving the production path unchanged.
- [ ] (2026-01-13 11:59Z) Add `config/tests/defaults.yaml` and smoke configs under `config/tests/`.
- [ ] (2026-01-13 11:59Z) Add `tests/unit/config/test_test_configs_defaults.py` to enforce defaults alignment.
- [ ] (2026-01-13 11:59Z) Update `README.md` and `slurm/README.md` with the test runner workflow.
- [ ] (2026-01-13 11:59Z) Run a SLURM unit test job and a tokenization+training integration job; record job IDs and logs here.

## Surprises & Discoveries

None yet.

## Decision Log

- Decision: Use `slurm/tests/slurm_test.env` and `slurm/tests/test_secrets.env` to mirror the existing SLURM config pattern while keeping secrets out of git.
  Rationale: Reuses familiar `.env` conventions, keeps defaults explicit, and avoids committing credentials.
  Date/Author: 2026-01-13 / Codex
- Decision: Default test partition is `postiguet1` with 1x RTX 4090.
  Rationale: Matches the known test hardware and keeps runs consistent.
  Date/Author: 2026-01-13 / Codex
- Decision: Provide minimal integration configs under `config/tests/` and run them only when relevant.
  Rationale: Keeps end-to-end checks available without slowing down every change.
  Date/Author: 2026-01-13 / Codex
- Decision: Move all SLURM test artifacts under `slurm/tests/`.
  Rationale: Keeps test tooling separate from production SLURM scripts and improves organization.
  Date/Author: 2026-01-13 / Codex
- Decision: Keep `config/tests/defaults.yaml` as the canonical source of model/tokenizer defaults and enforce alignment via a unit test.
  Rationale: The current config loader uses plain PyYAML with no cross-file includes, so a test is the lowest-risk way to keep defaults aligned.
  Date/Author: 2026-01-13 / Codex
- Decision: Run integration configs in a fixed order (tokenization then training) when both are requested.
  Rationale: Training depends on the tokenized dataset path produced by tokenization.
  Date/Author: 2026-01-13 / Codex

## Outcomes & Retrospective

Not started.

## Context and Orientation

LMTK is a YAML-driven toolkit. Each YAML configuration denotes a single job that maps to exactly one task module under `src/tasks/`. Jobs are normally submitted to SLURM using `slurm/submit_job.sh`, which runs `slurm/p.slurm` and eventually calls `src/main.py --config <yaml>`. A SLURM partition is a named queue that specifies what hardware a job can run on. In this environment, tests should target the `postiguet1` partition with one RTX 4090 GPU. Unit tests live alongside task code under `src/tasks/<task>/` when they are task-specific, while cross-cutting tests live under `tests/`. Integration tests should be represented as YAML configs under `config/tests/`, and run only when the change touches the relevant behavior. The SLURM test runner introduced in this plan is a small shell wrapper that submits a job with a `RUN_COMMAND` rather than a task config, so unit tests can run inside the same environment as production jobs.
`sacct` is the SLURM accounting command used to query job state and exit codes; this plan uses it to verify test runs completed successfully.

## Milestones

Milestone 1 establishes the SLURM test runner plumbing. By the end, `slurm/tests/slurm_test.env`, `slurm/tests/test_secrets.env.example`, `slurm/tests/run_tests.sh`, and a `RUN_MODE=test` path in `slurm/p.slurm` exist. Run `./slurm/tests/run_tests.sh --dry-run` from the repo root and expect a printed `sbatch` command that includes `RUN_MODE=test`, a resolved `RUN_COMMAND`, and output/error paths under `slurm/tests/logs/`. The script must refuse submission if the current user is not in `ALLOWED_SUBMITTERS`.

Milestone 2 adds test defaults and smoke configs plus a unit test that enforces defaults alignment. Validate configs with `python -m src.main --validate --config config/tests/tokenization_smoke.yaml` and `python -m src.main --validate --config config/tests/clm_training_smoke.yaml`. Run `python -m pytest -q tests/unit/config/test_test_configs_defaults.py` and expect a passing test.

Milestone 3 documents the workflow and validates it on SLURM. Update `README.md` and `slurm/README.md` with the unit vs integration test policy. Submit a unit test run with `./slurm/tests/run_tests.sh` and an integration run with `./slurm/tests/run_tests.sh --integration tokenization_smoke,clm_training_smoke`. Capture job IDs and logs in this plan.

## Plan of Work

Create a dedicated SLURM test defaults file and secrets template under `slurm/tests/`, and update `.gitignore` so secrets are never committed. Implement `slurm/tests/run_tests.sh` to load the test defaults, load secrets if present, verify the submitter is allowed, and submit a job with a test command. Extend `slurm/p.slurm` to recognize `RUN_MODE=test` and execute `RUN_COMMAND` without requiring `CONFIG_FILE`, while still reusing the existing environment setup. Define a small Llama-family model in `config/tests/defaults.yaml` and add minimal YAML configs under `config/tests/` for tokenization and CLM training that run quickly. Add a unit test in `tests/unit/config/test_test_configs_defaults.py` that asserts the integration configs are aligned with `config/tests/defaults.yaml` so test defaults stay consistent. Update README and SLURM docs to explain the new workflow. Finally, validate by submitting a unit test run and a targeted integration test run on SLURM, recording job IDs and log paths.

## Concrete Steps

From the repository root, add the test defaults and secrets template:

    cat > slurm/tests/slurm_test.env
    (content from Artifacts and Notes)
    Ctrl-D

    cat > slurm/tests/test_secrets.env.example
    (content from Artifacts and Notes)
    Ctrl-D

Update `.gitignore` to ignore `slurm/tests/test_secrets.env`.

Create the test runner:

    cat > slurm/tests/run_tests.sh
    (implementation described in Interfaces and Dependencies)
    Ctrl-D
    chmod +x slurm/tests/run_tests.sh

Update `slurm/p.slurm` to support test mode by reading `RUN_MODE` and `RUN_COMMAND`, validating `RUN_COMMAND` when in test mode, and executing it in place of the task config path.

Add test defaults and configs:

    mkdir -p config/tests
    cat > config/tests/defaults.yaml
    (content from Artifacts and Notes)
    Ctrl-D

    cat > config/tests/tokenization_smoke.yaml
    (content from Artifacts and Notes)
    Ctrl-D

    cat > config/tests/clm_training_smoke.yaml
    (content from Artifacts and Notes)
    Ctrl-D

Add a unit test at `tests/unit/config/test_test_configs_defaults.py` that loads `config/tests/defaults.yaml` and validates that the integration configs use the same model, tokenizer, and seed defaults.

Update documentation in `README.md` and `slurm/README.md` to describe the new test runner, the test env files, and the integration test policy.

Submit a dry-run:

    ./slurm/tests/run_tests.sh --dry-run

Then submit an actual unit test run, and record the job ID and log paths:

    ./slurm/tests/run_tests.sh

Finally, submit a targeted integration test run when relevant (tokenization before training):

    ./slurm/tests/run_tests.sh --integration tokenization_smoke,clm_training_smoke

## Validation and Acceptance

Run `./slurm/tests/run_tests.sh --dry-run` from the repo root and confirm the output prints a full `sbatch` command that includes `--export=RUN_MODE=test,RUN_COMMAND=...` plus `--output=slurm/tests/logs/tests-<jobid>.out` and `--error=slurm/tests/logs/tests-<jobid>.err`. Submit unit tests with `./slurm/tests/run_tests.sh`. Capture the job ID from stdout. Verify completion with `sacct -j <job_id> --format=JobID,State,ExitCode -P`; expect `COMPLETED|0:0`. Confirm the log file in `slurm/tests/logs/tests-<jobid>.out` contains a pytest summary like `X passed`. Run integration tests in order with `./slurm/tests/run_tests.sh --integration tokenization_smoke,clm_training_smoke`. Confirm the tokenization log line "Tokenization workflow completed successfully" appears before training starts, and that `output/tests/tokenized` exists. Confirm training finishes and the CSV metrics file exists at `output/tests/<model_name>/version_0/metrics.csv` with a numeric `loss` column. If SLURM submission is unavailable, run `python -m src.main --validate --config config/tests/tokenization_smoke.yaml` and `python -m src.main --validate --config config/tests/clm_training_smoke.yaml`, then `python -m pytest -q tests/unit/config/test_test_configs_defaults.py`, and record why SLURM validation was skipped.

## Idempotence and Recovery

Re-running `slurm/tests/run_tests.sh` is safe and produces new job IDs and logs. If a run fails, update the resource settings in `slurm/tests/slurm_test.env` or adjust the test config and resubmit. If a job needs to be canceled, use `scancel <jobid>`.

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
    export TEST_COMMAND="python -m pytest -q tests src/tasks"
    export TEST_CONFIG_DIR="config/tests"
    export TEST_INTEGRATION_CONFIGS="config/tests/tokenization_smoke.yaml,config/tests/clm_training_smoke.yaml"

Example `slurm/tests/test_secrets.env.example`:

    export WANDB_API_KEY="replace_me"
    export HUGGINGFACE_API_KEY="replace_me"

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
    parallelization_strategy: none

## Interfaces and Dependencies

`slurm/tests/run_tests.sh` is a bash script that sources `slurm/tests/slurm_test.env` and optionally `slurm/tests/test_secrets.env`. It must refuse to run if `whoami` is not listed in the comma-separated `ALLOWED_SUBMITTERS` value. It should accept `--dry-run`, `--integration <names|all>`, and resource override flags (`--partition`, `--gpus`, `--time`, `--memory`, `--cpus`, `--job-name`, `--nodelist`). When `--integration` is provided, it should map names like `tokenization_smoke` to `config/tests/tokenization_smoke.yaml` using `TEST_CONFIG_DIR` (defaulting to `config/tests` if unset) and `TEST_INTEGRATION_CONFIGS` from the env file, then build a `RUN_COMMAND` that runs them in order with `python -m src.main --config <config>`. It must fail with a clear error if an integration name is unknown or if `RUN_COMMAND` is empty. `slurm/p.slurm` must treat `RUN_MODE=test` as a separate execution path that skips `CONFIG_FILE` validation and runs `RUN_COMMAND` after environment setup, while leaving the production `CONFIG_FILE` path untouched when `RUN_MODE` is not `test`.

Plan revision note: Added milestones, corrected progress, clarified validation and runner interfaces, and replaced schema-invalid config examples with validated smoke configs to make the plan junior-proof.
