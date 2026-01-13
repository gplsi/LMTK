# Add SLURM-Based Test Runner and Test Configs

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

PLANS.md is checked into the repo at `PLANS.md`, and this document must be maintained in accordance with it.

## Purpose / Big Picture

LMTK depends on SLURM-only GPU runtimes and containers, so local tests are often incomplete or impossible. This plan creates a dedicated SLURM test runner, standard test defaults (including a small Llama-family model), and minimal end-to-end YAML configs so every change can be validated with predictable job submission, logs, and results. The outcome is a repeatable testing workflow that future changes can always use, while keeping integration tests targeted and fast.

## Progress

- [x] (2026-01-13 10:16Z) Updated `AGENTS.md` and `PLANS.md` to codify SLURM test runner usage, test configs, and integration test policy.
- [x] (2026-01-13 10:16Z) Created issue card `issues/38-slurm-test-runner.md`.
- [x] (2026-01-13 10:22Z) Updated docs to place test tooling under `slurm/tests/` and align README references.
- [ ] Add SLURM test defaults (`slurm/tests/slurm_test.env`) and secrets template (`slurm/tests/test_secrets.env.example`) and update `.gitignore`.
- [ ] Implement `slurm/tests/run_tests.sh` with submitter guard, config parsing, and SLURM submission.
- [ ] Extend `slurm/p.slurm` to support `RUN_MODE=test` and a `RUN_COMMAND` path.
- [ ] Add `config/tests/defaults.yaml` and minimal task configs under `config/tests/`.
- [ ] Add unit tests that validate test configs use the shared defaults.
- [ ] Update `README.md` and `slurm/README.md` with the testing workflow.
- [ ] Validate a unit test run and a targeted integration test on SLURM, capturing job ID and logs.

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

## Outcomes & Retrospective

Not started.

## Context and Orientation

LMTK is a YAML-driven toolkit. Each YAML configuration denotes a single job that maps to exactly one task module under `src/tasks/`. Jobs are normally submitted to SLURM using `slurm/submit_job.sh`, which runs `slurm/p.slurm` and eventually calls `src/main.py --config <yaml>`. A SLURM partition is a named queue that specifies what hardware a job can run on. In this environment, tests should target the `postiguet1` partition with one RTX 4090 GPU. Unit tests live alongside task code under `src/tasks/<task>/` when they are task-specific, while cross-cutting tests live under `tests/`. Integration tests should be represented as YAML configs under `config/tests/`, and run only when the change touches the relevant behavior. The SLURM test runner introduced in this plan is a small shell wrapper that submits a job with a `RUN_COMMAND` rather than a task config, so unit tests can run inside the same environment as production jobs.

## Plan of Work

Create a dedicated SLURM test defaults file and secrets template under `slurm/tests/`, and update `.gitignore` so secrets are never committed. Implement `slurm/tests/run_tests.sh` to load the test defaults, load secrets if present, verify the submitter is allowed, and submit a job with a test command. Extend `slurm/p.slurm` to recognize `RUN_MODE=test` and execute `RUN_COMMAND` without requiring `CONFIG_FILE`, while still reusing the existing environment setup. Define a small Llama-family model in `config/tests/defaults.yaml` and add minimal YAML configs under `config/tests/` for tokenization and CLM training that run quickly. Add a unit test that asserts the integration configs are aligned with `config/tests/defaults.yaml` so test defaults stay consistent. Update README and SLURM docs to explain the new workflow. Finally, validate by submitting a unit test run and a targeted integration test run on SLURM, recording job IDs and log paths.

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

Add a unit test under `tests/` that loads `config/tests/defaults.yaml` and validates that the integration configs use the same model and tokenizer defaults.

Update documentation in `README.md` and `slurm/README.md` to describe the new test runner, the test env files, and the integration test policy.

Submit a dry-run:

    ./slurm/tests/run_tests.sh --dry-run

Then submit an actual unit test run, and record the job ID and log paths:

    ./slurm/tests/run_tests.sh

Finally, submit a targeted integration test run when relevant:

    ./slurm/tests/run_tests.sh --integration tokenization_smoke

## Validation and Acceptance

Validation requires a SLURM submission that runs unit tests in the cluster environment and reports a passing pytest summary in logs. A targeted integration run must complete successfully using the small Llama defaults and the specified test config. Acceptance is reached when `slurm/tests/run_tests.sh` prints a job ID, logs exist at the configured path, and `sacct` shows an exit code of `0:0` for the job. Evidence should include the job ID, the exact command, and the log excerpts showing the test summary or completion messages.

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
    export TEST_INTEGRATION_CONFIGS="config/tests/tokenization_smoke.yaml,config/tests/clm_training_smoke.yaml"

Example `slurm/tests/test_secrets.env.example`:

    export WANDB_API_KEY="replace_me"
    export HUGGINGFACE_API_KEY="replace_me"

Example `config/tests/defaults.yaml`:

    model_name: hf-internal-testing/tiny-random-LlamaForCausalLM
    tokenizer_name: hf-internal-testing/llama-tokenizer
    seed: 42
    max_steps: 5
    batch_size: 1

Example `config/tests/tokenization_smoke.yaml`:

    task: tokenization
    experiment_name: test_tokenization_smoke
    verbose_level: 1
    tokenizer:
      tokenizer_name: hf-internal-testing/llama-tokenizer
      use_fast: true
      task_type: clm_training
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
    model_name: hf-internal-testing/tiny-random-LlamaForCausalLM
    dataset:
      source: local
      format: hf
      nameOrPath: output/tests/tokenized
    batch_size: 1
    number_epochs: 1
    gradient_accumulation: false
    gradient_accumulation_steps: 1
    log_iter_interval: 1
    validations_per_epoch: 0
    lr: 2.0e-5
    weight_decay: 0.0
    output_dir: output/tests
    task: clm_training
    verbose_level: 1

## Interfaces and Dependencies

`slurm/tests/run_tests.sh` is a bash script that reads `slurm/tests/slurm_test.env` for defaults and optionally `slurm/tests/test_secrets.env` for credentials. It must validate that `whoami` appears in `ALLOWED_SUBMITTERS`, then submit `slurm/p.slurm` with `RUN_MODE=test` and `RUN_COMMAND` derived from `TEST_COMMAND` and any requested integration configs. It should accept flags to override partition, GPU count, and to enable integration configs by name. `slurm/p.slurm` must treat `RUN_MODE=test` as a separate execution path that runs `RUN_COMMAND` with the same environment setup as production jobs, and it must fail fast with a clear error message if `RUN_COMMAND` is missing.

Plan revision note: Updated all SLURM test artifacts to live under `slurm/tests/` to match the agreed organization and keep test tooling separate from production SLURM files.
