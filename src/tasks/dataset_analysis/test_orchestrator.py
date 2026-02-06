from __future__ import annotations

import json
from pathlib import Path

import pytest
from box import Box

from src.tasks.dataset_analysis.orchestrator import DatasetAnalysisOrchestrator


class FakeSplit:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        return self._rows[idx]


class FakeDatasetDict(dict):
    pass


def _build_config(dataset_path: Path, output_path: Path, **analysis_overrides) -> Box:
    analysis = {
        "splits": [],
        "top_k": 5,
        "compute_padding_metrics": True,
        "compute_token_frequency": True,
        "compute_unique_tokens": True,
        "report_name": "report",
    }
    analysis.update(analysis_overrides)
    return Box(
        {
            "task": "dataset_analysis",
            "experiment_name": "test_dataset_analysis",
            "verbose_level": 1,
            "dataset": {
                "source": "local",
                "nameOrPath": str(dataset_path),
                "format": "hf",
            },
            "analysis": analysis,
            "output": {"path": str(output_path)},
        },
        box_dots=True,
    )


def _read_json_report(output_path: Path, report_name: str = "report") -> dict:
    report_path = output_path / f"{report_name}.json"
    return json.loads(report_path.read_text(encoding="utf-8"))


def test_fixed_window_metrics_are_computed_exactly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ds = FakeSplit(
        [
            {
                "input_ids": [10, 11, 0, 0],
                "attention_mask": [1, 1, 0, 0],
                "labels": [10, 11, -100, -100],
            },
            {
                "input_ids": [12, 13, 14, 0],
                "attention_mask": [1, 1, 1, 0],
                "labels": [12, 13, 14, -100],
            },
        ]
    )
    dataset_path = tmp_path / "tokenized_fixed"
    output_path = tmp_path / "analysis"
    monkeypatch.setattr(DatasetAnalysisOrchestrator, "_load_dataset", lambda _self: ds)

    config = _build_config(dataset_path, output_path)
    orchestrator = DatasetAnalysisOrchestrator(config)
    result = orchestrator.execute()

    assert result["global"]["num_examples"] == 2
    assert result["global"]["total_tokens_raw"] == 8
    assert result["global"]["total_tokens_effective"] == 5
    assert result["global"]["padding_tokens"] == 3
    assert pytest.approx(result["global"]["padding_ratio"]) == 3 / 8
    assert pytest.approx(result["global"]["avg_non_padded_seq_len"]) == 2.5
    assert result["global"]["unique_token_count"] == 5

    top_tokens = result["global"]["top_k_tokens"]
    assert len(top_tokens) == 5
    assert {entry["token_id"] for entry in top_tokens} == {10, 11, 12, 13, 14}

    persisted = _read_json_report(output_path)
    assert persisted["global"]["padding_tokens"] == 3


def test_doclevel_metrics_and_top_tokens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ds = FakeSplit(
        [
            {"input_ids": [1, 2, 3], "length": 3, "ends_with_eos": False},
            {"input_ids": [3, 4], "length": 2, "ends_with_eos": False},
        ]
    )
    dataset_path = tmp_path / "tokenized_doclevel"
    output_path = tmp_path / "analysis"
    monkeypatch.setattr(DatasetAnalysisOrchestrator, "_load_dataset", lambda _self: ds)

    config = _build_config(dataset_path, output_path)
    result = DatasetAnalysisOrchestrator(config).execute()

    assert result["global"]["total_tokens_raw"] == 5
    assert result["global"]["total_tokens_effective"] == 5
    assert result["global"]["padding_tokens"] == 0
    assert result["global"]["unique_token_count"] == 4
    assert result["global"]["top_k_tokens"][0]["token_id"] == 3
    assert result["global"]["top_k_tokens"][0]["count"] == 2
    assert pytest.approx(result["global"]["top_k_tokens"][0]["frequency"]) == 0.4


def test_split_filtering_uses_only_requested_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = FakeDatasetDict(
        {
            "train": FakeSplit([{"input_ids": [1, 2], "length": 2}]),
            "valid": FakeSplit([{"input_ids": [3, 4, 5], "length": 3}]),
        }
    )
    dataset_path = tmp_path / "tokenized_multisplit"
    output_path = tmp_path / "analysis"
    monkeypatch.setattr(DatasetAnalysisOrchestrator, "_load_dataset", lambda _self: ds)

    config = _build_config(dataset_path, output_path, splits=["valid"])
    result = DatasetAnalysisOrchestrator(config).execute()

    assert set(result["splits"].keys()) == {"valid"}
    assert result["global"]["num_examples"] == 1
    assert result["global"]["total_tokens_raw"] == 3


def test_fails_when_padding_metrics_requested_but_effective_tokens_not_derivable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = FakeSplit([{"input_ids": [1, 2, 0]}, {"input_ids": [3, 0, 0]}])
    dataset_path = tmp_path / "tokenized_ambiguous"
    output_path = tmp_path / "analysis"
    monkeypatch.setattr(DatasetAnalysisOrchestrator, "_load_dataset", lambda _self: ds)

    config = _build_config(dataset_path, output_path, compute_padding_metrics=True)

    with pytest.raises(ValueError, match="Cannot derive effective token lengths"):
        DatasetAnalysisOrchestrator(config).execute()
