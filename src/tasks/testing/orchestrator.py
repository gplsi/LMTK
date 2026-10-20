"""Testing task orchestrator for unit and integration runs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from box import Box

from src.utils.logging import VerboseLevel, get_logger


class TestingOrchestrator:
    """Run unit or integration tests from a task config."""

    def __init__(self, config: Box) -> None:
        self.config = config
        self.verbose_level = VerboseLevel(
            self.config.get("verbose_level", VerboseLevel.INFO)
        )
        self.logger = get_logger(__name__, self.verbose_level)
        self.project_root = Path(__file__).resolve().parents[3]

    def execute(self) -> None:
        testing_config = self.config.get("testing")
        if not testing_config:
            raise ValueError("Testing configuration must be provided under 'testing'.")

        mode = testing_config.get("mode")
        if mode == "unit":
            self._run_unit_tests(testing_config)
        elif mode == "integration":
            self._run_integration_tests(testing_config)
        else:
            raise ValueError("Testing mode must be either 'unit' or 'integration'.")

    def _run_unit_tests(self, testing_config: Box) -> None:
        command = testing_config.get("command")
        if not command:
            raise ValueError("Unit testing requires a 'command' field.")

        self.logger.info("Running unit tests with command: %s", command)
        self._run_command(command)

    def _run_integration_tests(self, testing_config: Box) -> None:
        configs = testing_config.get("configs")
        if not configs:
            raise ValueError("Integration testing requires a non-empty 'configs' list.")

        stop_on_failure = testing_config.get("stop_on_failure", True)
        failures: list[str] = []

        for config_path in configs:
            resolved_path = self._resolve_config_path(str(config_path))
            self.logger.info("Running integration config: %s", resolved_path)
            try:
                subprocess.run(
                    [
                        sys.executable,
                        str(self.project_root / "src" / "main.py"),
                        "--config",
                        str(resolved_path),
                    ],
                    check=True,
                    cwd=self.project_root,
                )
            except subprocess.CalledProcessError as exc:
                self.logger.error("Integration config failed: %s", resolved_path)
                if stop_on_failure:
                    raise RuntimeError(
                        f"Integration run failed for config: {resolved_path}"
                    ) from exc
                failures.append(str(resolved_path))

        if failures:
            raise RuntimeError(
                "Integration runs failed for configs: " + ", ".join(failures)
            )

    def _run_command(self, command: str) -> None:
        subprocess.run(command, shell=True, check=True, cwd=self.project_root)

    def _resolve_config_path(self, config_path: str) -> Path:
        path = Path(config_path)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()
