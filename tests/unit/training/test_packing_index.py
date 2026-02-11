from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import time

import pytest

from src.tasks.training.data.packing_index import PackingIndex


@dataclass
class _FakeSplit:
    rows: list[dict]
    column_names: list[str]
    _fingerprint: str = "fakefp"

    def __len__(self) -> int:  # pragma: no cover
        return len(self.rows)

    def __getitem__(self, idx):  # pragma: no cover
        if isinstance(idx, int):
            return self.rows[idx]
        if isinstance(idx, str):
            if idx not in self.column_names:
                raise KeyError(idx)
            return [row[idx] for row in self.rows]
        raise TypeError(f"Unsupported index type: {type(idx)}")

    def iter(self, batch_size: int = 1000):  # pragma: no cover
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        for start in range(0, len(self.rows), batch_size):
            chunk = self.rows[start : start + batch_size]
            yield {name: [row[name] for row in chunk] for name in self.column_names}


@dataclass
class _NoRowAccessAfterValidationSplit(_FakeSplit):
    allowed_row_accesses: int = 0
    _row_reads: int = 0

    def __getitem__(self, idx):  # pragma: no cover
        if isinstance(idx, int):
            if self._row_reads >= int(self.allowed_row_accesses):
                raise AssertionError(f"Unexpected row-wise access at idx={idx}")
            self._row_reads += 1
        return super().__getitem__(idx)


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
    assert idx2.cache_hit is True
    assert meta_path.read_text(encoding="utf-8") == first_meta


def test_packing_index_breaks_stale_lock_when_configured(tmp_path: Path) -> None:
    split = _FakeSplit(
        rows=[
            {"input_ids": [1, 2, 3], "length": 3, "ends_with_eos": False},
            {"input_ids": [4], "length": 1, "ends_with_eos": False},
        ],
        column_names=["input_ids", "length", "ends_with_eos"],
    )

    lock_path = tmp_path / "v1" / "train" / "LOCK"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text('{"created_at_unix": 0}\n', encoding="utf-8")

    idx = PackingIndex.load_or_build(
        hf_split=split,
        split="train",
        sequence_length=4,
        insert_eos=True,
        eos_token_id=0,
        cache_dir=tmp_path,
        lock_timeout_s=1,
        stale_lock_age_s=1,
    )
    assert idx.meta.split == "train"


def test_packing_index_does_not_break_active_lease(tmp_path: Path) -> None:
    split = _FakeSplit(
        rows=[
            {"input_ids": [1, 2, 3], "length": 3, "ends_with_eos": False},
        ],
        column_names=["input_ids", "length", "ends_with_eos"],
    )

    lock_path = tmp_path / "v1" / "train" / "LOCK"
    lease_path = tmp_path / "v1" / "train" / "LEASE.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text('{"created_at_unix": 0}\n', encoding="utf-8")
    lease_path.write_text(
        '{"created_at_unix": 0, "updated_at_unix": 4102444800}\n',
        encoding="utf-8",
    )

    with pytest.raises(TimeoutError):
        PackingIndex.load_or_build(
            hf_split=split,
            split="train",
            sequence_length=4,
            insert_eos=True,
            eos_token_id=0,
            cache_dir=tmp_path,
            lock_timeout_s=1,
            stale_lock_age_s=1,
        )

    assert lock_path.exists()


def test_packing_index_breaks_stale_lock_without_lease_by_mtime(tmp_path: Path) -> None:
    split = _FakeSplit(
        rows=[
            {"input_ids": [1, 2], "length": 2, "ends_with_eos": False},
        ],
        column_names=["input_ids", "length", "ends_with_eos"],
    )

    lock_path = tmp_path / "v1" / "train" / "LOCK"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("stale-lock\n", encoding="utf-8")
    old_ts = time.time() - 3600
    os.utime(lock_path, (old_ts, old_ts))

    idx = PackingIndex.load_or_build(
        hf_split=split,
        split="train",
        sequence_length=4,
        insert_eos=True,
        eos_token_id=0,
        cache_dir=tmp_path,
        lock_timeout_s=1,
        stale_lock_age_s=1,
    )
    assert idx.meta.split == "train"


def test_packing_index_allow_build_false_does_not_mutate_lock(tmp_path: Path) -> None:
    split = _FakeSplit(
        rows=[{"input_ids": [9], "length": 1}],
        column_names=["input_ids", "length"],
    )
    lock_path = tmp_path / "v1" / "train" / "LOCK"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text('{"created_at_unix": 0}\n', encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        PackingIndex.load_or_build(
            hf_split=split,
            split="train",
            sequence_length=4,
            insert_eos=False,
            eos_token_id=None,
            cache_dir=tmp_path,
            allow_build=False,
            lock_timeout_s=1,
            stale_lock_age_s=1,
        )

    assert lock_path.exists()


def test_packing_index_timeout_error_includes_lease_owner_details(tmp_path: Path) -> None:
    split = _FakeSplit(
        rows=[{"input_ids": [1], "length": 1}],
        column_names=["input_ids", "length"],
    )
    lock_path = tmp_path / "v1" / "train" / "LOCK"
    lease_path = tmp_path / "v1" / "train" / "LEASE.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text('{"created_at_unix": 1}\n', encoding="utf-8")
    lease_path.write_text(
        '{"owner_host": "node-a", "owner_pid": 1234, "updated_at_unix": 1}\n',
        encoding="utf-8",
    )

    with pytest.raises(TimeoutError) as exc_info:
        PackingIndex.load_or_build(
            hf_split=split,
            split="train",
            sequence_length=4,
            insert_eos=False,
            eos_token_id=None,
            cache_dir=tmp_path,
            lock_timeout_s=1,
            stale_lock_age_s=None,
        )

    message = str(exc_info.value)
    assert "Lease owner host='node-a' pid=1234" in message
    assert "timeout_s=1 stale_lock_age_s=None" in message


def test_packing_index_timeout_error_handles_malformed_lease(tmp_path: Path) -> None:
    split = _FakeSplit(
        rows=[{"input_ids": [1], "length": 1}],
        column_names=["input_ids", "length"],
    )
    lock_path = tmp_path / "v1" / "train" / "LOCK"
    lease_path = tmp_path / "v1" / "train" / "LEASE.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text('{"created_at_unix": 1}\n', encoding="utf-8")
    lease_path.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(TimeoutError) as exc_info:
        PackingIndex.load_or_build(
            hf_split=split,
            split="train",
            sequence_length=4,
            insert_eos=False,
            eos_token_id=None,
            cache_dir=tmp_path,
            lock_timeout_s=1,
            stale_lock_age_s=None,
        )
    assert "Timed out waiting for packing index lock" in str(exc_info.value)


def test_packing_index_wait_logs_expose_owner_diagnostics(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    split = _FakeSplit(
        rows=[{"input_ids": [1], "length": 1}],
        column_names=["input_ids", "length"],
    )
    lock_path = tmp_path / "v1" / "train" / "LOCK"
    lease_path = tmp_path / "v1" / "train" / "LEASE.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text('{"created_at_unix": 1}\n', encoding="utf-8")
    lease_path.write_text(
        '{"owner_host": "node-b", "owner_pid": 42, "updated_at_unix": 1}\n',
        encoding="utf-8",
    )

    caplog.set_level(logging.INFO, logger="src.tasks.training.data.packing_index")
    with pytest.raises(TimeoutError):
        PackingIndex.load_or_build(
            hf_split=split,
            split="train",
            sequence_length=4,
            insert_eos=False,
            eos_token_id=None,
            cache_dir=tmp_path,
            lock_timeout_s=1,
            stale_lock_age_s=None,
        )

    messages = [record.getMessage() for record in caplog.records if "Waiting for packing index lock" in record.getMessage()]
    assert messages
    assert any("lease_owner_host=node-b" in msg and "lease_owner_pid=42" in msg for msg in messages)


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


def test_packing_index_fast_path_without_insert_eos_avoids_full_row_reads(tmp_path: Path) -> None:
    split = _NoRowAccessAfterValidationSplit(
        rows=[
            {"input_ids": [1, 2, 3], "length": 3},
            {"input_ids": [4, 5], "length": 2},
            {"input_ids": [6, 7, 8, 9], "length": 4},
        ],
        column_names=["input_ids", "length"],
        allowed_row_accesses=1,
    )

    idx = PackingIndex.load_or_build(
        hf_split=split,
        split="train",
        sequence_length=4,
        insert_eos=False,
        eos_token_id=None,
        cache_dir=tmp_path,
        length_sample_validation=1,
    )
    assert idx.total_tokens == 9
    assert idx.num_blocks == 2


def test_packing_index_fast_path_with_ends_with_eos_avoids_full_row_reads(tmp_path: Path) -> None:
    split = _NoRowAccessAfterValidationSplit(
        rows=[
            {"input_ids": [1, 2, 3], "length": 3, "ends_with_eos": False},
            {"input_ids": [4, 5, 0], "length": 3, "ends_with_eos": True},
            {"input_ids": [7], "length": 1, "ends_with_eos": False},
        ],
        column_names=["input_ids", "length", "ends_with_eos"],
        allowed_row_accesses=1,
    )

    idx = PackingIndex.load_or_build(
        hf_split=split,
        split="train",
        sequence_length=4,
        insert_eos=True,
        eos_token_id=0,
        cache_dir=tmp_path,
        length_sample_validation=1,
    )
    assert idx.total_tokens == 9
    assert idx.num_blocks == 2


def test_packing_index_fallback_without_ends_with_eos_keeps_row_path(tmp_path: Path) -> None:
    split = _FakeSplit(
        rows=[
            {"input_ids": [10, 11], "length": 2},
            {"input_ids": [12, 0], "length": 2},
            {"input_ids": [], "length": 0},
        ],
        column_names=["input_ids", "length"],
    )

    idx = PackingIndex.load_or_build(
        hf_split=split,
        split="train",
        sequence_length=4,
        insert_eos=True,
        eos_token_id=0,
        cache_dir=tmp_path,
        length_sample_validation=1,
    )
    # Row0 adds EOS, row1 already ends with EOS, row2 empty.
    assert idx.meta.kept_docs == 2
    assert idx.total_tokens == 5
    assert idx.num_blocks == 1
