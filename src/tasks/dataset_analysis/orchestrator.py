"""
Dataset analysis orchestrator for tokenized datasets.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from box import Box

from src.utils.logging import VerboseLevel, get_logger


class DatasetAnalysisOrchestrator:
    """Compute exact statistics over tokenized Hugging Face datasets."""

    def __init__(self, config: Box) -> None:
        self.config = config
        self.verbose_level = VerboseLevel(
            self.config.get("verbose_level", VerboseLevel.INFO)
        )
        self.logger = get_logger(__name__, self.verbose_level)

    def validate_config(self) -> None:
        if not self.config.get("dataset"):
            raise ValueError("Dataset configuration must be provided")

        dataset_cfg = self.config.dataset
        if dataset_cfg.get("source") != "local":
            raise ValueError(
                "dataset_analysis currently supports only dataset.source='local'"
            )
        if not dataset_cfg.get("nameOrPath"):
            raise ValueError("dataset.nameOrPath must be provided")
        if dataset_cfg.get("format") not in {"hf", "dataset"}:
            raise ValueError("dataset.format must be either 'hf' or 'dataset'")

        if not self.config.get("output") or not self.config.output.get("path"):
            raise ValueError("output.path must be provided")

        analysis = self.config.get("analysis", Box({}, box_dots=True))
        if "top_k" in analysis and int(analysis.top_k) < 1:
            raise ValueError("analysis.top_k must be >= 1")
        if "splits" in analysis and not isinstance(analysis.splits, list):
            raise ValueError("analysis.splits must be a list of split names")

    def execute(self) -> Dict[str, Any]:
        self.validate_config()
        options = self._analysis_options()
        dataset = self._load_dataset()
        selected_splits = self._resolve_splits(dataset, options["splits"])

        split_reports: Dict[str, Dict[str, Any]] = {}
        global_counter: Counter[int] = Counter()
        global_lengths: List[int] = []
        global_examples = 0
        global_tokens_raw = 0
        global_tokens_effective = 0
        all_effective_known = True

        for split_name, split_dataset in selected_splits.items():
            split_report, split_counter, split_lengths, effective_known = self._analyze_split(
                split_name, split_dataset, options
            )
            split_reports[split_name] = split_report
            global_counter.update(split_counter)
            global_lengths.extend(split_lengths)
            global_examples += split_report["num_examples"]
            global_tokens_raw += split_report["total_tokens_raw"]

            if effective_known and split_report["total_tokens_effective"] is not None:
                global_tokens_effective += int(split_report["total_tokens_effective"])
            else:
                all_effective_known = False

        global_report = self._build_global_report(
            num_examples=global_examples,
            total_tokens_raw=global_tokens_raw,
            total_tokens_effective=global_tokens_effective
            if all_effective_known
            else None,
            lengths=global_lengths if all_effective_known else [],
            token_counter=global_counter,
            options=options,
            effective_known=all_effective_known,
        )

        report: Dict[str, Any] = {
            "task": "dataset_analysis",
            "dataset_path": str(self.config.dataset.nameOrPath),
            "analysis_options": options,
            "splits": split_reports,
            "global": global_report,
        }

        self._write_reports(report, options["report_name"])
        self.logger.info("Dataset analysis completed: %s", self.config.dataset.nameOrPath)
        return report

    def _analysis_options(self) -> Dict[str, Any]:
        analysis = self.config.get("analysis", Box({}, box_dots=True))
        return {
            "splits": list(analysis.get("splits", []) or []),
            "top_k": int(analysis.get("top_k", 100)),
            "compute_padding_metrics": bool(
                analysis.get("compute_padding_metrics", True)
            ),
            "compute_token_frequency": bool(
                analysis.get("compute_token_frequency", True)
            ),
            "compute_unique_tokens": bool(analysis.get("compute_unique_tokens", True)),
            "report_name": str(analysis.get("report_name", "dataset_analysis")),
        }

    def _load_dataset(self):
        try:
            from src.utils.dataset import DatasetStorage
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The 'datasets' dependency is required to load tokenized datasets from disk. "
                "Install project dependencies before running dataset_analysis."
            ) from exc

        storage = DatasetStorage(verbose_level=self.verbose_level)
        dataset_path = str(self.config.dataset.nameOrPath)
        self.logger.info("Loading tokenized dataset from %s", dataset_path)
        return storage.load_from_disk(dataset_path)

    def _resolve_splits(
        self, dataset: Any, requested_splits: List[str]
    ) -> Dict[str, Any]:
        if self._is_dataset_dict(dataset):
            available = list(dataset.keys())
            splits_to_use = requested_splits or available
            missing = sorted(set(splits_to_use) - set(available))
            if missing:
                raise ValueError(
                    f"Requested splits {missing} are not available. Available splits: {sorted(available)}"
                )
            return {split: dataset[split] for split in splits_to_use}

        if requested_splits and requested_splits != ["train"]:
            raise ValueError(
                "Single-split datasets only support split name 'train'. "
                f"Received: {requested_splits}"
            )
        return {"train": dataset}

    @staticmethod
    def _is_dataset_dict(dataset: Any) -> bool:
        if isinstance(dataset, dict):
            return True
        return type(dataset).__name__ == "DatasetDict"

    def _analyze_split(
        self, split_name: str, split_dataset: Any, options: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Counter[int], List[int], bool]:
        num_examples = len(split_dataset)
        total_tokens_raw = 0
        total_tokens_effective = 0
        non_padded_lengths: List[int] = []
        token_counter: Counter[int] = Counter()
        effective_known = True

        need_token_counts = options["compute_token_frequency"] or options["compute_unique_tokens"]

        for row_index in range(num_examples):
            row = split_dataset[row_index]
            if "input_ids" not in row:
                raise ValueError(
                    f"Split '{split_name}' row {row_index} is missing required column 'input_ids'"
                )

            input_ids = list(row["input_ids"])
            raw_tokens = [int(token_id) for token_id in input_ids]
            total_tokens_raw += len(raw_tokens)

            effective_tokens, effective_len = self._derive_effective_tokens(
                row=row,
                raw_tokens=raw_tokens,
                split_name=split_name,
                row_index=row_index,
                require_effective=options["compute_padding_metrics"],
            )

            if effective_len is None:
                effective_known = False
                if need_token_counts:
                    token_counter.update(raw_tokens)
                continue

            total_tokens_effective += effective_len
            non_padded_lengths.append(effective_len)
            if need_token_counts:
                token_counter.update(effective_tokens)

        split_report = self._build_split_report(
            split_name=split_name,
            num_examples=num_examples,
            total_tokens_raw=total_tokens_raw,
            total_tokens_effective=total_tokens_effective
            if effective_known
            else None,
            non_padded_lengths=non_padded_lengths,
            token_counter=token_counter,
            options=options,
            effective_known=effective_known,
        )
        return split_report, token_counter, non_padded_lengths, effective_known

    def _derive_effective_tokens(
        self,
        row: Mapping[str, Any],
        raw_tokens: List[int],
        split_name: str,
        row_index: int,
        require_effective: bool,
    ) -> Tuple[List[int], Optional[int]]:
        raw_len = len(raw_tokens)

        if "attention_mask" in row and row["attention_mask"] is not None:
            mask = list(row["attention_mask"])
            if len(mask) != raw_len:
                raise ValueError(
                    f"Split '{split_name}' row {row_index} has attention_mask length {len(mask)} "
                    f"but input_ids length {raw_len}"
                )
            tokens = [tok for tok, keep in zip(raw_tokens, mask) if int(keep) == 1]
            return tokens, len(tokens)

        if "labels" in row and row["labels"] is not None:
            labels = list(row["labels"])
            if len(labels) != raw_len:
                raise ValueError(
                    f"Split '{split_name}' row {row_index} has labels length {len(labels)} "
                    f"but input_ids length {raw_len}"
                )
            tokens = [tok for tok, label in zip(raw_tokens, labels) if int(label) != -100]
            return tokens, len(tokens)

        if "length" in row and row["length"] is not None:
            declared_len = int(row["length"])
            if declared_len < 0 or declared_len > raw_len:
                raise ValueError(
                    f"Split '{split_name}' row {row_index} has invalid length={declared_len} "
                    f"for input_ids length {raw_len}"
                )
            tokens = raw_tokens[:declared_len]
            return tokens, declared_len

        if require_effective:
            raise ValueError(
                "Cannot derive effective token lengths for padding metrics. "
                "Provide attention_mask, labels, or length columns."
            )

        return [], None

    def _build_split_report(
        self,
        split_name: str,
        num_examples: int,
        total_tokens_raw: int,
        total_tokens_effective: Optional[int],
        non_padded_lengths: List[int],
        token_counter: Counter[int],
        options: Dict[str, Any],
        effective_known: bool,
    ) -> Dict[str, Any]:
        report = self._build_base_metrics(
            num_examples=num_examples,
            total_tokens_raw=total_tokens_raw,
            total_tokens_effective=total_tokens_effective,
            non_padded_lengths=non_padded_lengths,
            effective_known=effective_known,
        )
        report["split"] = split_name

        frequency_denominator = (
            report["total_tokens_effective"]
            if report["total_tokens_effective"] is not None
            else report["total_tokens_raw"]
        )
        if options["compute_unique_tokens"]:
            report["unique_token_count"] = int(len(token_counter))
        else:
            report["unique_token_count"] = None

        if options["compute_token_frequency"]:
            report["top_k_tokens"] = self._top_k_tokens(
                token_counter, options["top_k"], frequency_denominator
            )
        else:
            report["top_k_tokens"] = []

        return report

    def _build_global_report(
        self,
        num_examples: int,
        total_tokens_raw: int,
        total_tokens_effective: Optional[int],
        lengths: List[int],
        token_counter: Counter[int],
        options: Dict[str, Any],
        effective_known: bool,
    ) -> Dict[str, Any]:
        report = self._build_base_metrics(
            num_examples=num_examples,
            total_tokens_raw=total_tokens_raw,
            total_tokens_effective=total_tokens_effective,
            non_padded_lengths=lengths,
            effective_known=effective_known,
        )

        frequency_denominator = (
            report["total_tokens_effective"]
            if report["total_tokens_effective"] is not None
            else report["total_tokens_raw"]
        )
        if options["compute_unique_tokens"]:
            report["unique_token_count"] = int(len(token_counter))
        else:
            report["unique_token_count"] = None

        if options["compute_token_frequency"]:
            report["top_k_tokens"] = self._top_k_tokens(
                token_counter, options["top_k"], frequency_denominator
            )
        else:
            report["top_k_tokens"] = []

        return report

    @staticmethod
    def _build_base_metrics(
        num_examples: int,
        total_tokens_raw: int,
        total_tokens_effective: Optional[int],
        non_padded_lengths: List[int],
        effective_known: bool,
    ) -> Dict[str, Any]:
        if effective_known and total_tokens_effective is not None:
            padding_tokens = int(total_tokens_raw - total_tokens_effective)
            padding_ratio = (
                float(padding_tokens / total_tokens_raw) if total_tokens_raw > 0 else 0.0
            )
            avg_non_padded = (
                float(total_tokens_effective / num_examples) if num_examples > 0 else 0.0
            )
            length_stats = DatasetAnalysisOrchestrator._length_stats(non_padded_lengths)
        else:
            padding_tokens = None
            padding_ratio = None
            avg_non_padded = None
            length_stats = None

        return {
            "num_examples": int(num_examples),
            "total_tokens_raw": int(total_tokens_raw),
            "total_tokens_effective": int(total_tokens_effective)
            if total_tokens_effective is not None
            else None,
            "padding_tokens": padding_tokens,
            "padding_ratio": padding_ratio,
            "avg_non_padded_seq_len": avg_non_padded,
            "non_padded_seq_len_stats": length_stats,
            "effective_tokens_derived": bool(effective_known),
        }

    @staticmethod
    def _length_stats(lengths: List[int]) -> Dict[str, float]:
        if not lengths:
            return {"min": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0}
        sorted_lengths = sorted(int(v) for v in lengths)
        return {
            "min": int(sorted_lengths[0]),
            "p50": DatasetAnalysisOrchestrator._percentile(sorted_lengths, 50.0),
            "p95": DatasetAnalysisOrchestrator._percentile(sorted_lengths, 95.0),
            "p99": DatasetAnalysisOrchestrator._percentile(sorted_lengths, 99.0),
            "max": int(sorted_lengths[-1]),
        }

    @staticmethod
    def _percentile(sorted_values: List[int], percentile: float) -> float:
        if not sorted_values:
            return 0.0
        if len(sorted_values) == 1:
            return float(sorted_values[0])
        position = (percentile / 100.0) * (len(sorted_values) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return float(sorted_values[lower])
        lower_value = float(sorted_values[lower])
        upper_value = float(sorted_values[upper])
        weight = position - lower
        return lower_value + (upper_value - lower_value) * weight

    @staticmethod
    def _top_k_tokens(
        token_counter: Counter[int], top_k: int, denominator: Optional[int]
    ) -> List[Dict[str, Any]]:
        if not token_counter:
            return []
        denom = float(denominator) if denominator else 0.0
        sorted_items = sorted(token_counter.items(), key=lambda item: (-item[1], item[0]))
        result = []
        for token_id, count in sorted_items[:top_k]:
            result.append(
                {
                    "token_id": int(token_id),
                    "count": int(count),
                    "frequency": float(count / denom) if denom > 0 else 0.0,
                }
            )
        return result

    def _write_reports(self, report: Dict[str, Any], report_name: str) -> None:
        output_dir = Path(self.config.output.path)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / f"{report_name}.json"
        markdown_path = output_dir / f"{report_name}.md"

        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        markdown_path.write_text(self._build_markdown_report(report), encoding="utf-8")

    def _build_markdown_report(self, report: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append("# Dataset Analysis Report")
        lines.append("")
        lines.append(f"- Dataset path: `{report['dataset_path']}`")
        lines.append(f"- Task: `{report['task']}`")
        lines.append("")
        lines.append("## Global")
        lines.extend(self._render_metrics(report["global"]))
        lines.append("")
        lines.append("## Splits")
        for split_name, split_report in report["splits"].items():
            lines.append("")
            lines.append(f"### `{split_name}`")
            lines.extend(self._render_metrics(split_report))
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _render_metrics(metrics: Mapping[str, Any]) -> List[str]:
        lines: List[str] = []
        lines.append(f"- num_examples: `{metrics['num_examples']}`")
        lines.append(f"- total_tokens_raw: `{metrics['total_tokens_raw']}`")
        lines.append(f"- total_tokens_effective: `{metrics['total_tokens_effective']}`")
        lines.append(f"- padding_tokens: `{metrics['padding_tokens']}`")
        lines.append(f"- padding_ratio: `{metrics['padding_ratio']}`")
        lines.append(f"- avg_non_padded_seq_len: `{metrics['avg_non_padded_seq_len']}`")
        lines.append(f"- unique_token_count: `{metrics['unique_token_count']}`")
        top_tokens = metrics.get("top_k_tokens", [])
        if top_tokens:
            lines.append("- top_k_tokens:")
            for entry in top_tokens[:10]:
                lines.append(
                    f"  - token_id={entry['token_id']} count={entry['count']} "
                    f"frequency={entry['frequency']:.6f}"
                )
        else:
            lines.append("- top_k_tokens: `[]`")
        return lines
