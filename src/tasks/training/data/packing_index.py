from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator, Optional

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
    def _acquire_lock(
        lock_path: Path,
        *,
        timeout_s: int = 1800,
        poll_s: float = 0.5,
        lock_payload: dict[str, Any] | None = None,
    ) -> int:
        deadline = time.time() + timeout_s
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                if lock_payload is not None:
                    try:
                        payload = dict(lock_payload)
                        payload.setdefault("pid", os.getpid())
                        payload.setdefault("host", socket.gethostname())
                        payload.setdefault("created_at_unix", time.time())
                        serialized = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
                        os.write(fd, serialized)
                        os.fsync(fd)
                    except Exception:
                        # Best-effort: the lock itself is the safety mechanism.
                        pass
                return fd
            except FileExistsError:
                if time.time() > deadline:
                    lock_contents = None
                    try:
                        lock_contents = lock_path.read_text(encoding="utf-8", errors="replace").strip()
                    except Exception:
                        lock_contents = None

                    details = f" Existing LOCK contents: {lock_contents}" if lock_contents else ""
                    raise TimeoutError(
                        f"Timed out waiting for packing index lock at {lock_path}.{details} "
                        "If you are sure no training job is running, delete the LOCK file and retry."
                    )
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

        lock_fd = cls._acquire_lock(
            p["lock"],
            lock_payload={
                "index_version": INDEX_VERSION,
                "split": str(split),
                "sequence_length": int(sequence_length),
                "insert_eos": bool(insert_eos),
                "eos_token_id": int(eos_token_id) if eos_token_id is not None else None,
                "cache_dir": str(cache_dir),
            },
        )
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
            max_validated = min(int(length_sample_validation), num_rows)
            for row_idx in range(max_validated):
                row = hf_split[row_idx]
                row_length = int(row["length"])
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

            def _iter_column_batches(batch_size: int = 65536) -> Iterator[dict[str, np.ndarray]]:
                if hasattr(hf_split, "iter"):
                    iterator = hf_split.iter(batch_size=batch_size)
                    for batch in iterator:
                        lengths = np.asarray(batch["length"], dtype=np.int64).reshape(-1)
                        payload: dict[str, np.ndarray] = {"length": lengths}
                        if used_ends_with_eos:
                            payload["ends_with_eos"] = np.asarray(batch["ends_with_eos"], dtype=np.bool_).reshape(-1)
                        yield payload
                    return

                try:
                    lengths_raw = hf_split["length"]
                    ends_raw = hf_split["ends_with_eos"] if used_ends_with_eos else None
                except Exception:
                    lengths_raw = [int(hf_split[row_idx]["length"]) for row_idx in range(num_rows)]
                    ends_raw = (
                        [bool(hf_split[row_idx]["ends_with_eos"]) for row_idx in range(num_rows)]
                        if used_ends_with_eos
                        else None
                    )

                lengths = np.asarray(lengths_raw, dtype=np.int64).reshape(-1)
                payload: dict[str, np.ndarray] = {"length": lengths}
                if used_ends_with_eos and ends_raw is not None:
                    payload["ends_with_eos"] = np.asarray(ends_raw, dtype=np.bool_).reshape(-1)
                yield payload

            kept_docs = 0
            total_tokens = 0

            # Fast path: avoid full-row `input_ids` reads when eos insertion can be derived from columns.
            if not insert_eos or used_ends_with_eos:
                row_cursor = 0
                for batch in _iter_column_batches():
                    lengths = batch["length"]
                    count = int(lengths.size)
                    if count <= 0:
                        continue

                    row_indices = np.arange(row_cursor, row_cursor + count, dtype=np.int64)
                    keep_mask = lengths > 0
                    kept_count = int(np.count_nonzero(keep_mask))
                    if kept_count > 0:
                        kept_rows = row_indices[keep_mask]
                        kept_lengths = lengths[keep_mask]

                        if insert_eos and used_ends_with_eos:
                            ends = batch["ends_with_eos"][keep_mask]
                            eos_extra = (~ends).astype(np.int64, copy=False)
                        else:
                            eos_extra = np.zeros((kept_count,), dtype=np.int64)

                        stream_lengths = kept_lengths + eos_extra
                        next_kept = kept_docs + kept_count

                        doc_indices[kept_docs:next_kept] = kept_rows
                        doc_lengths[kept_docs:next_kept] = kept_lengths.astype(np.int32, copy=False)
                        doc_stream_lengths[kept_docs:next_kept] = stream_lengths.astype(np.int32, copy=False)

                        cumulative = np.cumsum(stream_lengths, dtype=np.int64)
                        offsets[kept_docs + 1 : next_kept + 1] = total_tokens + cumulative

                        total_tokens += int(cumulative[-1])
                        kept_docs = next_kept

                    row_cursor += count

                if row_cursor != num_rows:
                    raise RuntimeError(
                        "Packing index build internal error: processed row count does not match dataset size "
                        f"({row_cursor} != {num_rows})."
                    )
            else:
                # Fallback path for datasets without `ends_with_eos`: requires per-row `input_ids[-1]`.
                for row_idx in range(num_rows):
                    row = hf_split[row_idx]
                    row_length = int(row["length"])
                    if row_length <= 0:
                        continue

                    input_ids = row["input_ids"]
                    last_token = int(input_ids[-1])
                    eos_extra = 0 if last_token == int(eos_token_id) else 1

                    stream_len = row_length + eos_extra
                    doc_indices[kept_docs] = int(row_idx)
                    doc_lengths[kept_docs] = int(row_length)
                    doc_stream_lengths[kept_docs] = int(stream_len)
                    offsets[kept_docs + 1] = offsets[kept_docs] + int(stream_len)
                    total_tokens = int(offsets[kept_docs + 1])
                    kept_docs += 1

            skipped_empty = num_rows - kept_docs
            tail_tokens = total_tokens % int(sequence_length)

            offsets.flush()
            doc_indices.flush()
            doc_lengths.flush()
            doc_stream_lengths.flush()

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
