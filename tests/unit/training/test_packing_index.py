from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from src.tasks.training.data.packing_index import PackingIndex


@dataclass
class _FakeSplit:
    rows: list[dict]
    column_names: list[str]
    _fingerprint: str = "fakefp"

    def __len__(self) -> int:  # pragma: no cover
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:  # pragma: no cover
        return self.rows[idx]


def test_packing_index_build_and_reuse(tmp_path: Path) -> None:
    split = _FakeSplit(
        rows=[
            {"input_ids": [1, 2, 3], "length": 3, "ends_with_eos": False},
            {"input_ids": [], "length": 0, "ends_with_eos": False},
            {"input_ids": [4, 5, 0], "length": 3, "ends_with_eos": True},
        ],
        column_names=["input_ids", "length", "ends_with_eos"],
    )

    idx1 = PackingIndex.load_or_build(
        hf_split=split,
        split="train",
        sequence_length=4,
        insert_eos=True,
        eos_token_id=0,
        cache_dir=tmp_path,
    )
    assert idx1.meta.kept_docs == 2
    assert idx1.meta.used_ends_with_eos is True
    assert idx1.total_tokens == 7  # doc0 + eos + doc2 (already eos)
    assert idx1.num_blocks == 1

    meta_path = tmp_path / "v1" / "train" / "meta.json"
    first_meta = meta_path.read_text(encoding="utf-8")

    idx2 = PackingIndex.load_or_build(
        hf_split=split,
        split="train",
        sequence_length=4,
        insert_eos=True,
        eos_token_id=0,
        cache_dir=tmp_path,
        allow_build=False,
    )
    assert idx2.meta == idx1.meta
    assert meta_path.read_text(encoding="utf-8") == first_meta


def test_packing_index_rebuilds_on_param_change(tmp_path: Path) -> None:
    split = _FakeSplit(
        rows=[
            {"input_ids": [1, 2, 3], "length": 3, "ends_with_eos": False},
        ],
        column_names=["input_ids", "length", "ends_with_eos"],
    )

    idx1 = PackingIndex.load_or_build(
        hf_split=split,
        split="train",
        sequence_length=4,
        insert_eos=True,
        eos_token_id=0,
        cache_dir=tmp_path,
    )

    idx2 = PackingIndex.load_or_build(
        hf_split=split,
        split="train",
        sequence_length=4,
        insert_eos=True,
        eos_token_id=99,
        cache_dir=tmp_path,
    )

    assert idx1.meta.eos_token_id == 0
    assert idx2.meta.eos_token_id == 99


def test_packing_index_disallow_build_raises(tmp_path: Path) -> None:
    split = _FakeSplit(
        rows=[{"input_ids": [1], "length": 1}],
        column_names=["input_ids", "length"],
    )

    with pytest.raises(FileNotFoundError):
        PackingIndex.load_or_build(
            hf_split=split,
            split="train",
            sequence_length=4,
            insert_eos=False,
            eos_token_id=None,
            cache_dir=tmp_path,
            allow_build=False,
        )


def test_packing_index_ends_with_eos_mismatch_fails_fast(tmp_path: Path) -> None:
    split = _FakeSplit(
        rows=[
            {"input_ids": [1, 2, 0], "length": 3, "ends_with_eos": False},
        ],
        column_names=["input_ids", "length", "ends_with_eos"],
    )

    with pytest.raises(ValueError, match="ends_with_eos"):
        PackingIndex.load_or_build(
            hf_split=split,
            split="train",
            sequence_length=4,
            insert_eos=True,
            eos_token_id=0,
            cache_dir=tmp_path,
        )


def test_packing_index_length_mismatch_fails_fast(tmp_path: Path) -> None:
    split = _FakeSplit(
        rows=[
            {"input_ids": [1, 2, 3], "length": 2},
        ],
        column_names=["input_ids", "length"],
    )

    with pytest.raises(ValueError, match="row\\['length'\\] does not match len\\(row\\['input_ids'\\]\\)"):
        PackingIndex.load_or_build(
            hf_split=split,
            split="train",
            sequence_length=4,
            insert_eos=False,
            eos_token_id=None,
            cache_dir=tmp_path,
        )


def test_packing_index_rebuilds_on_dataset_fingerprint_change(tmp_path: Path) -> None:
    split_v1 = _FakeSplit(
        rows=[
            {"input_ids": [1, 2, 3], "length": 3, "ends_with_eos": False},
        ],
        column_names=["input_ids", "length", "ends_with_eos"],
        _fingerprint="fp-v1",
    )
    split_v2 = _FakeSplit(
        rows=[
            {"input_ids": [1, 2, 3], "length": 3, "ends_with_eos": False},
        ],
        column_names=["input_ids", "length", "ends_with_eos"],
        _fingerprint="fp-v2",
    )

    idx1 = PackingIndex.load_or_build(
        hf_split=split_v1,
        split="train",
        sequence_length=4,
        insert_eos=True,
        eos_token_id=0,
        cache_dir=tmp_path,
    )
    idx2 = PackingIndex.load_or_build(
        hf_split=split_v2,
        split="train",
        sequence_length=4,
        insert_eos=True,
        eos_token_id=0,
        cache_dir=tmp_path,
    )

    assert idx1.meta.dataset_fingerprint == "fp-v1"
    assert idx2.meta.dataset_fingerprint == "fp-v2"
