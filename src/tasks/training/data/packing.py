from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from src.tasks.training.data.packing_index import PackingIndex


@dataclass(frozen=True)
class PackingStats:
    skipped_empty_docs: int
    tail_tokens_dropped: int


class PackedSequenceDataset(Dataset):
    """
    Map-style dataset that exposes packed fixed-length blocks built from variable-length token sequences.

    Required input columns on `hf_dataset`:
      - input_ids: list[int] (variable-length)
      - length: optional int (when present, must equal len(input_ids))
      - labels: optional list[int] (packed in lockstep; otherwise defaults to input_ids)
      - attention_mask: optional list[int] (packed in lockstep; otherwise defaults to ones)

    Packing algorithm (conceptual; do not materialize the full stream in memory):
      - Define the logical token stream:
        - If insert_eos is false: doc_0 + doc_1 + ...
        - If insert_eos is true: append eos_token_id after each document unless the document already ends in eos_token_id.
      - Define block i as stream[i * sequence_length : (i + 1) * sequence_length]
      - Drop tail tokens that do not fit into a full block.

    Implementation constraint:
      - The token stream definition is conceptual only. This dataset must not concatenate all tokens into memory.
        It precomputes only lightweight prefix sums (offsets) and produces each block on demand.
    """

    def __init__(
        self,
        *,
        hf_dataset: Any,
        index: PackingIndex,
        eos_token_id: int | None,
    ) -> None:
        self._hf_dataset = hf_dataset
        self._index = index
        self._sequence_length = int(index.meta.sequence_length)
        self._eos_token_id = int(eos_token_id) if eos_token_id is not None else None
        if bool(index.meta.insert_eos) and self._eos_token_id is None:
            raise ValueError("Packing index requires EOS insertion but eos_token_id is None.")

        self.stats = PackingStats(
            skipped_empty_docs=int(index.stats.skipped_empty_docs),
            tail_tokens_dropped=int(index.stats.tail_tokens_dropped),
        )

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    def __len__(self) -> int:
        return int(self._index.num_blocks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0:
            raise IndexError("PackedSequenceDataset does not support negative indices.")

        num_blocks = len(self)
        if idx >= num_blocks:
            raise IndexError(f"Index {idx} out of range for {num_blocks} packed blocks.")

        start_pos = idx * self._sequence_length
        end_pos = start_pos + self._sequence_length
        tokens: list[int] = []
        labels: list[int] = []
        attention_mask: list[int] = []

        offsets = self._index.offsets
        doc_ptr = bisect_right(offsets, start_pos) - 1
        pos = start_pos

        def _optional_sequence(row: dict[str, Any], name: str, input_len: int) -> list[int] | None:
            if name not in row:
                return None
            values = row[name]
            if len(values) != input_len:
                raise ValueError(
                    f"Invalid packing input: row[{name!r}] length does not match row['input_ids']. "
                    f"len({name})={len(values)} len(input_ids)={input_len}"
                )
            return values

        def _synthetic_eos_label(
            *,
            input_ids: list[int],
            row_labels: list[int] | None,
            doc_len: int,
        ) -> int:
            if self._eos_token_id is None:
                raise RuntimeError(
                    "Packing index indicates EOS insertion but eos_token_id is None."
                )
            if row_labels is None:
                return int(self._eos_token_id)
            labels_are_clm_targets = all(
                int(row_labels[i]) == int(input_ids[i]) for i in range(doc_len)
            )
            # Non-CLM labels come from instruction-style masking; the packer-inserted
            # EOS is a structural boundary token, not original supervised content.
            return int(self._eos_token_id) if labels_are_clm_targets else -100

        while pos < end_pos:
            doc_start = int(offsets[doc_ptr])
            doc_end = int(offsets[doc_ptr + 1])
            local_pos = pos - doc_start

            row_idx = int(self._index.doc_indices[doc_ptr])
            row = self._hf_dataset[row_idx]
            input_ids: list[int] = row["input_ids"]
            doc_len = int(self._index.doc_lengths[doc_ptr])
            stream_len = int(self._index.doc_stream_lengths[doc_ptr])
            eos_extra = int(stream_len - doc_len)
            input_len = len(input_ids)
            if doc_len != input_len:
                raise ValueError(
                    "Invalid packing input: index doc length does not match row['input_ids']. "
                    f"row={row_idx} length={doc_len} len(input_ids)={input_len}"
                )
            row_labels = _optional_sequence(row, "labels", input_len)
            row_attention_mask = _optional_sequence(row, "attention_mask", input_len)

            remaining = end_pos - pos

            if local_pos < doc_len:
                take = min(doc_len - local_pos, remaining)
                input_chunk = input_ids[local_pos : local_pos + take]
                tokens.extend(int(x) for x in input_chunk)
                if row_labels is None:
                    labels.extend(int(x) for x in input_chunk)
                else:
                    labels.extend(int(x) for x in row_labels[local_pos : local_pos + take])
                if row_attention_mask is None:
                    attention_mask.extend([1] * take)
                else:
                    attention_mask.extend(
                        int(x) for x in row_attention_mask[local_pos : local_pos + take]
                    )
                pos += take
                remaining -= take

            if remaining > 0 and eos_extra == 1 and pos < doc_end:
                if self._eos_token_id is None:
                    raise RuntimeError(
                        "Packing index indicates EOS insertion but eos_token_id is None."
                    )
                tokens.append(int(self._eos_token_id))
                labels.append(
                    _synthetic_eos_label(
                        input_ids=input_ids,
                        row_labels=row_labels,
                        doc_len=doc_len,
                    )
                )
                attention_mask.append(1)
                pos += 1
                remaining -= 1

            if pos >= doc_end:
                doc_ptr += 1

        if (
            len(tokens) != self._sequence_length
            or len(labels) != self._sequence_length
            or len(attention_mask) != self._sequence_length
        ):
            raise RuntimeError(
                "Packing produced an invalid block length. "
                f"Expected {self._sequence_length}, got input_ids={len(tokens)}, "
                f"labels={len(labels)}, attention_mask={len(attention_mask)}."
            )

        input_tensor = torch.tensor(tokens, dtype=torch.long)
        attention_tensor = torch.tensor(attention_mask, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        return {
            "input_ids": input_tensor,
            "attention_mask": attention_tensor,
            "labels": label_tensor,
        }


def build_packing_dataloader(
    dataset: Dataset,
    *,
    split: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    sampler_drop_last: bool,
    drop_last_batch: bool,
    seed: int | None,
    rank: int,
    world_size: int,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer.")
    if world_size <= 0:
        raise ValueError("world_size must be a positive integer.")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size.")

    seed_value = int(seed) if seed is not None else 0
    split_lower = split.lower()

    sampler = None

    if world_size == 1:
        if split_lower == "train" and shuffle:
            sampler = DistributedSampler(
                dataset,
                num_replicas=1,
                rank=0,
                shuffle=True,
                seed=seed_value,
                drop_last=False,
            )
    else:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=bool(shuffle),
            seed=seed_value,
            drop_last=bool(sampler_drop_last),
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=bool(drop_last_batch),
    )
