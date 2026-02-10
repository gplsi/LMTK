#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

try:
    from datasets import DatasetDict, load_from_disk
except ModuleNotFoundError:  # pragma: no cover - script still supports metadata-only mode
    DatasetDict = None  # type: ignore[assignment]
    load_from_disk = None  # type: ignore[assignment]

from transformers import AutoTokenizer

from src.utils.dataset import (
    build_tokenization_metadata,
    write_tokenization_metadata,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill lmtk_tokenization_meta.json for tokenized datasets by resolving "
            "eos_token_id from a local tokenizer path."
        )
    )
    parser.add_argument(
        "--dataset-path",
        action="append",
        default=[],
        help="Dataset path to patch. Repeat flag for multiple datasets.",
    )
    parser.add_argument(
        "--dataset-list-file",
        type=str,
        default=None,
        help="Optional text file containing one dataset path per line.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Local tokenizer/model path used to resolve eos_token_id.",
    )
    parser.add_argument(
        "--eos-token-id",
        type=int,
        default=None,
        help="Explicit EOS token id. Overrides tokenizer-based resolution when provided.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print actions without writing metadata files.",
    )
    parser.add_argument(
        "--no-strict-ends-with-eos-check",
        action="store_true",
        help="Disable strict sampled consistency checks for ends_with_eos.",
    )
    parser.add_argument(
        "--skip-dataset-validation",
        action="store_true",
        help=(
            "Skip loading dataset shards and deep column checks. Useful in environments "
            "without the `datasets` package; still writes metadata files."
        ),
    )
    return parser.parse_args()


def _read_dataset_list_file(path: Path) -> list[str]:
    rows: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append(stripped)
    return rows


def _collect_dataset_paths(args: argparse.Namespace) -> list[Path]:
    raw_paths = list(args.dataset_path)
    if args.dataset_list_file:
        raw_paths.extend(_read_dataset_list_file(Path(args.dataset_list_file)))
    if not raw_paths:
        raise ValueError("Provide at least one --dataset-path or --dataset-list-file.")

    dedup: dict[str, Path] = {}
    for raw in raw_paths:
        path = Path(raw).expanduser().resolve()
        dedup[str(path)] = path
    return sorted(dedup.values(), key=lambda item: str(item))


def _iter_dataset_splits(dataset_obj: object) -> Iterable[tuple[str, object]]:
    if DatasetDict is None:
        raise RuntimeError("datasets package is not available.")
    if isinstance(dataset_obj, DatasetDict):
        for split_name in sorted(dataset_obj.keys()):
            yield split_name, dataset_obj[split_name]
        return
    yield "train", dataset_obj


def _validate_split_columns(split_name: str, split_dataset: object) -> None:
    column_names = getattr(split_dataset, "column_names", [])
    required = {"input_ids", "length"}
    missing = sorted(required - set(column_names))
    if missing:
        raise ValueError(
            f"Split {split_name!r} is missing required columns {missing}; "
            "expected a doc-level tokenized dataset."
        )


def _validate_ends_with_eos(
    split_name: str,
    split_dataset: object,
    *,
    eos_token_id: int,
) -> None:
    column_names = set(getattr(split_dataset, "column_names", []))
    if "ends_with_eos" not in column_names:
        return
    sample_n = min(256, len(split_dataset))
    for row_idx in range(sample_n):
        row = split_dataset[row_idx]
        input_ids = row["input_ids"]
        expected = bool(input_ids) and int(input_ids[-1]) == int(eos_token_id)
        observed = bool(row["ends_with_eos"])
        if expected != observed:
            raise ValueError(
                f"Split {split_name!r} has inconsistent ends_with_eos at row {row_idx}: "
                f"observed={observed} expected={expected} eos_token_id={eos_token_id}."
            )


def _process_dataset_path(
    dataset_path: Path,
    *,
    tokenizer_path: Path,
    eos_token_id: int,
    strict_ends_with_eos_check: bool,
    dry_run: bool,
    skip_dataset_validation: bool,
) -> None:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    if not skip_dataset_validation:
        if load_from_disk is None:
            raise RuntimeError(
                "datasets package is not installed; rerun with --skip-dataset-validation "
                "or execute in an environment with datasets available."
            )
        dataset = load_from_disk(str(dataset_path))
        for split_name, split_dataset in _iter_dataset_splits(dataset):
            _validate_split_columns(split_name, split_dataset)
            if strict_ends_with_eos_check:
                _validate_ends_with_eos(
                    split_name,
                    split_dataset,
                    eos_token_id=eos_token_id,
                )

    metadata = build_tokenization_metadata(
        tokenizer_name=str(tokenizer_path),
        eos_token_id=eos_token_id,
        task="clm_training",
        created_by="lmtk-backfill",
    )
    if dry_run:
        print(f"[DRY-RUN] Would write metadata for {dataset_path}")
        return

    metadata_path = write_tokenization_metadata(dataset_path, metadata)
    print(f"[OK] Wrote {metadata_path}")


def main() -> int:
    args = _parse_args()
    strict_check = not bool(args.no_strict_ends_with_eos_check)

    if args.eos_token_id is None and args.tokenizer_path is None:
        raise ValueError("Provide either --eos-token-id or --tokenizer-path.")

    tokenizer_path: Path | None = None
    if args.tokenizer_path is not None:
        tokenizer_path = Path(args.tokenizer_path).expanduser().resolve()
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer path does not exist: {tokenizer_path}")

    eos_token_id = args.eos_token_id
    if eos_token_id is None:
        assert tokenizer_path is not None
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            use_fast=True,
            local_files_only=True,
        )
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise ValueError(
                f"Tokenizer at {tokenizer_path} does not define eos_token_id; cannot backfill metadata."
            )

    tokenizer_name_for_metadata = (
        str(tokenizer_path) if tokenizer_path is not None else "manual-eos-token-id"
    )

    dataset_paths = _collect_dataset_paths(args)
    print(f"Tokenizer source: {tokenizer_name_for_metadata}")
    print(f"Resolved eos_token_id: {int(eos_token_id)}")
    print(f"Datasets to patch: {len(dataset_paths)}")

    failures: list[tuple[Path, str]] = []
    for dataset_path in dataset_paths:
        try:
            _process_dataset_path(
                dataset_path,
                tokenizer_path=Path(tokenizer_name_for_metadata),
                eos_token_id=int(eos_token_id),
                strict_ends_with_eos_check=strict_check,
                dry_run=bool(args.dry_run),
                skip_dataset_validation=bool(args.skip_dataset_validation),
            )
        except Exception as exc:
            failures.append((dataset_path, str(exc)))
            print(f"[FAIL] {dataset_path}: {exc}")

    if failures:
        print("\nBackfill finished with failures:")
        for dataset_path, message in failures:
            print(f" - {dataset_path}: {message}")
        return 1

    print("\nBackfill completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
