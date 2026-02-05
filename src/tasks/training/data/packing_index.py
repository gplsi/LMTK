from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional

import numpy as np


INDEX_VERSION = 1


@dataclass(frozen=True)
class PackingIndexMeta:
    version: int
    split: str
    sequence_length: int
    insert_eos: bool
    eos_token_id: int | None
    dataset_fingerprint: str | None
    num_rows: int
    kept_docs: int
    used_ends_with_eos: bool


@dataclass(frozen=True)
class PackingIndexStats:
    skipped_empty_docs: int
    tail_tokens_dropped: int


class PackingIndex:
    def __init__(
        self,
        *,
        meta: PackingIndexMeta,
        stats: PackingIndexStats,
        offsets: np.memmap,
        doc_indices: np.memmap,
        doc_lengths: np.memmap,
        doc_stream_lengths: np.memmap,
    ) -> None:
        self.meta = meta
        self.stats = stats

        kept = int(meta.kept_docs)
        self.offsets = offsets[: kept + 1]
        self.doc_indices = doc_indices[:kept]
        self.doc_lengths = doc_lengths[:kept]
        self.doc_stream_lengths = doc_stream_lengths[:kept]

    @property
    def total_tokens(self) -> int:
        return int(self.offsets[-1])

    @property
    def num_blocks(self) -> int:
        return self.total_tokens // int(self.meta.sequence_length)

    @staticmethod
    def _infer_dataset_fingerprint(hf_split: Any) -> Optional[str]:
        fp = getattr(hf_split, "_fingerprint", None)
        if isinstance(fp, str) and fp:
            return fp
        fp = getattr(hf_split, "fingerprint", None)
        if isinstance(fp, str) and fp:
            return fp
        # Fallback: derive a weak-but-deterministic signature so we don't silently reuse a stale index.
        # This is intentionally lightweight (samples only) to avoid scanning the whole dataset.
        try:
            num_rows = int(len(hf_split))
        except Exception:
            return None
        if num_rows <= 0:
            return "derived:empty"
        sample_n = min(256, num_rows)
        h = sha256()
        h.update(str(num_rows).encode("utf-8"))
        for i in range(sample_n):
            row = hf_split[i]
            h.update(str(int(row.get("length", 0))).encode("utf-8"))
            if "ends_with_eos" in getattr(hf_split, "column_names", []):
                h.update(b"1" if bool(row.get("ends_with_eos", False)) else b"0")
        if num_rows > sample_n:
            row = hf_split[num_rows - 1]
            h.update(b"last")
            h.update(str(int(row.get("length", 0))).encode("utf-8"))
            if "ends_with_eos" in getattr(hf_split, "column_names", []):
                h.update(b"1" if bool(row.get("ends_with_eos", False)) else b"0")
        return f"derived:{h.hexdigest()}"

    @staticmethod
    def _paths(root: Path) -> dict[str, Path]:
        return {
            "meta": root / "meta.json",
            "lock": root / "LOCK",
            "offsets": root / "offsets.int64",
            "doc_indices": root / "doc_indices.int64",
            "doc_lengths": root / "doc_lengths.int32",
            "doc_stream_lengths": root / "doc_stream_lengths.int32",
        }

    @staticmethod
    def _read_meta(path: Path) -> PackingIndexMeta:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return PackingIndexMeta(**payload)

    @staticmethod
    def _write_meta_atomic(path: Path, meta: PackingIndexMeta) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(meta), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    @staticmethod
    def _acquire_lock(lock_path: Path, *, timeout_s: int = 1800, poll_s: float = 0.5) -> int:
        deadline = time.time() + timeout_s
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                return fd
            except FileExistsError:
                if time.time() > deadline:
                    raise TimeoutError(f"Timed out waiting for packing index lock at {lock_path}.")
                time.sleep(poll_s)

    @staticmethod
    def _release_lock(fd: int, lock_path: Path) -> None:
        try:
            os.close(fd)
        finally:
            try:
                lock_path.unlink(missing_ok=True)
            except Exception:
                # Best-effort; stale locks are handled by timeouts and user intervention.
                pass

    @classmethod
    def load_or_build(
        cls,
        *,
        hf_split: Any,
        split: str,
        sequence_length: int,
        insert_eos: bool,
        eos_token_id: int | None,
        cache_dir: Path,
        length_sample_validation: int = 256,
        allow_build: bool = True,
    ) -> "PackingIndex":
        if int(sequence_length) <= 0:
            raise ValueError("sequence_length must be a positive integer.")
        if insert_eos and eos_token_id is None:
            raise ValueError("insert_eos is true but eos_token_id is None.")
        if not hasattr(hf_split, "column_names"):
            raise TypeError("hf_split must be a Hugging Face Dataset split (has column_names).")
        if "input_ids" not in hf_split.column_names:
            raise ValueError("Packing index requires an 'input_ids' column.")
        if "length" not in hf_split.column_names:
            raise ValueError("Packing index requires a 'length' column.")

        root = cache_dir / f"v{INDEX_VERSION}" / str(split)
        root.mkdir(parents=True, exist_ok=True)
        p = cls._paths(root)

        dataset_fingerprint = cls._infer_dataset_fingerprint(hf_split)
        num_rows = int(len(hf_split))
        used_ends_with_eos = bool(insert_eos and ("ends_with_eos" in hf_split.column_names))

        def meta_matches(meta: PackingIndexMeta) -> bool:
            return (
                int(meta.version) == INDEX_VERSION
                and meta.split == str(split)
                and int(meta.sequence_length) == int(sequence_length)
                and bool(meta.insert_eos) == bool(insert_eos)
                and (meta.eos_token_id if meta.eos_token_id is not None else None)
                == (int(eos_token_id) if eos_token_id is not None else None)
                and meta.dataset_fingerprint == dataset_fingerprint
                and int(meta.num_rows) == num_rows
                and bool(meta.used_ends_with_eos) == used_ends_with_eos
                and int(meta.kept_docs) >= 0
            )

        def load_existing(meta: PackingIndexMeta) -> "PackingIndex":
            expected_rows = int(meta.num_rows)
            offsets = np.memmap(p["offsets"], dtype=np.int64, mode="r", shape=(expected_rows + 1,))
            doc_indices = np.memmap(p["doc_indices"], dtype=np.int64, mode="r", shape=(expected_rows,))
            doc_lengths = np.memmap(p["doc_lengths"], dtype=np.int32, mode="r", shape=(expected_rows,))
            doc_stream_lengths = np.memmap(
                p["doc_stream_lengths"], dtype=np.int32, mode="r", shape=(expected_rows,)
            )

            total_tokens = int(offsets[int(meta.kept_docs)])
            tail = total_tokens % int(meta.sequence_length)
            stats = PackingIndexStats(skipped_empty_docs=expected_rows - int(meta.kept_docs), tail_tokens_dropped=tail)
            return cls(
                meta=meta,
                stats=stats,
                offsets=offsets,
                doc_indices=doc_indices,
                doc_lengths=doc_lengths,
                doc_stream_lengths=doc_stream_lengths,
            )

        if p["meta"].exists():
            try:
                meta = cls._read_meta(p["meta"])
            except Exception:
                meta = None
            if meta is not None and meta_matches(meta):
                return load_existing(meta)

        if not allow_build:
            raise FileNotFoundError(
                "Packing index is missing or invalid and building is disabled. "
                f"Expected index under {root}."
            )

        lock_fd = cls._acquire_lock(p["lock"])
        try:
            if p["meta"].exists():
                try:
                    meta = cls._read_meta(p["meta"])
                except Exception:
                    meta = None
                if meta is not None and meta_matches(meta):
                    return load_existing(meta)

            offsets = np.memmap(p["offsets"], dtype=np.int64, mode="w+", shape=(num_rows + 1,))
            doc_indices = np.memmap(p["doc_indices"], dtype=np.int64, mode="w+", shape=(num_rows,))
            doc_lengths = np.memmap(p["doc_lengths"], dtype=np.int32, mode="w+", shape=(num_rows,))
            doc_stream_lengths = np.memmap(p["doc_stream_lengths"], dtype=np.int32, mode="w+", shape=(num_rows,))

            offsets[0] = 0
            kept_docs = 0
            skipped_empty = 0
            validated = 0

            for row_idx in range(num_rows):
                row = hf_split[row_idx]
                row_length = int(row["length"])
                if row_length <= 0:
                    skipped_empty += 1
                    continue

                if validated < int(length_sample_validation):
                    input_ids = row["input_ids"]
                    if row_length != len(input_ids):
                        raise ValueError(
                            "Invalid packing input: row['length'] does not match len(row['input_ids']). "
                            f"Row={row_idx} length={row_length} len(input_ids)={len(input_ids)}"
                        )
                    if insert_eos and used_ends_with_eos:
                        expected = bool(input_ids) and int(input_ids[-1]) == int(eos_token_id)
                        if bool(row["ends_with_eos"]) != expected:
                            raise ValueError(
                                "Invalid packing input: row['ends_with_eos'] does not match the provided eos_token_id. "
                                "This usually indicates the dataset was tokenized with a different tokenizer/eos id. "
                                f"Row={row_idx} ends_with_eos={row['ends_with_eos']} expected={expected} eos_token_id={eos_token_id}"
                            )
                    validated += 1

                eos_extra = 0
                if insert_eos:
                    if used_ends_with_eos:
                        ends_with_eos = bool(row["ends_with_eos"])
                        eos_extra = 0 if ends_with_eos else 1
                    else:
                        input_ids = row["input_ids"]
                        last_token = int(input_ids[-1])
                        eos_extra = 0 if last_token == int(eos_token_id) else 1

                stream_len = row_length + eos_extra

                doc_indices[kept_docs] = int(row_idx)
                doc_lengths[kept_docs] = int(row_length)
                doc_stream_lengths[kept_docs] = int(stream_len)
                offsets[kept_docs + 1] = offsets[kept_docs] + int(stream_len)
                kept_docs += 1

            total_tokens = int(offsets[kept_docs])
            tail_tokens = total_tokens % int(sequence_length)

            meta = PackingIndexMeta(
                version=INDEX_VERSION,
                split=str(split),
                sequence_length=int(sequence_length),
                insert_eos=bool(insert_eos),
                eos_token_id=int(eos_token_id) if eos_token_id is not None else None,
                dataset_fingerprint=dataset_fingerprint,
                num_rows=num_rows,
                kept_docs=kept_docs,
                used_ends_with_eos=used_ends_with_eos,
            )
            cls._write_meta_atomic(p["meta"], meta)

            offsets.flush()
            doc_indices.flush()
            doc_lengths.flush()
            doc_stream_lengths.flush()

            stats = PackingIndexStats(skipped_empty_docs=skipped_empty, tail_tokens_dropped=tail_tokens)
            return cls(
                meta=meta,
                stats=stats,
                offsets=offsets,
                doc_indices=doc_indices,
                doc_lengths=doc_lengths,
                doc_stream_lengths=doc_stream_lengths,
            )
        finally:
            cls._release_lock(lock_fd, p["lock"])
