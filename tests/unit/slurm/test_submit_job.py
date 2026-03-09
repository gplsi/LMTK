from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMIT_SCRIPT = REPO_ROOT / "slurm" / "submit_job.sh"
CONFIG_PATH = REPO_ROOT / "config" / "tests" / "defaults.yaml"


def _write_fake_conda(tmp_path: Path) -> Path:
    conda_sh = tmp_path / "conda.sh"
    conda_sh.write_text(
        "\n".join(
            [
                'conda() {',
                '  if [[ "$1" == "activate" ]]; then',
                '    export CONDA_DEFAULT_ENV="$2"',
                "    return 0",
                "  fi",
                '  if [[ "$1" == "--version" ]]; then',
                '    echo "conda 99.0-test"',
                "    return 0",
                "  fi",
                '  echo "unsupported fake conda command: $*" >&2',
                "  return 1",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    return conda_sh


def _write_fake_sbatch(tmp_path: Path) -> Path:
    sbatch = tmp_path / "sbatch"
    sbatch.write_text("#!/bin/sh\necho \"Submitted batch job 12345\"\n", encoding="utf-8")
    sbatch.chmod(0o755)
    return sbatch


def _run_submit_job(tmp_path: Path, *, conda_sh_path: Path) -> subprocess.CompletedProcess[str]:
    _write_fake_sbatch(tmp_path)

    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    env["CONDA_SH_PATH"] = str(conda_sh_path)
    env["CONDA_ENV_NAME"] = "lmtk-test"

    return subprocess.run(
        [
            "bash",
            str(SUBMIT_SCRIPT),
            "-c",
            str(CONFIG_PATH.relative_to(REPO_ROOT)),
            "-g",
            "1",
            "--nodes",
            "1",
            "--ntasks-per-node",
            "1",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_submit_job_fails_fast_when_conda_initialization_script_is_missing(tmp_path: Path) -> None:
    missing_conda_path = tmp_path / "missing-conda.sh"

    result = _run_submit_job(tmp_path, conda_sh_path=missing_conda_path)

    combined_output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Conda environment initialization failed during SLURM submission preflight." in combined_output
    assert f"Error: conda initialization script not found at: {missing_conda_path}" in combined_output
    assert "Submitted batch job 12345" not in combined_output


def test_submit_job_submits_after_conda_preflight_passes(tmp_path: Path) -> None:
    conda_sh = _write_fake_conda(tmp_path)

    result = _run_submit_job(tmp_path, conda_sh_path=conda_sh)

    combined_output = result.stdout + result.stderr
    assert result.returncode == 0
    assert "Conda environment preflight passed." in combined_output
    assert "Submitted batch job 12345" in combined_output
    assert "Job submitted successfully!" in combined_output
