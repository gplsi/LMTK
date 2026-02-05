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
      - length: int (must equal len(input_ids))

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

        offsets = self._index.offsets
        doc_ptr = bisect_right(offsets, start_pos) - 1
        pos = start_pos

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

            remaining = end_pos - pos

            if local_pos < doc_len:
                take = min(doc_len - local_pos, remaining)
                tokens.extend(int(x) for x in input_ids[local_pos : local_pos + take])
                pos += take
                remaining -= take

            if remaining > 0 and eos_extra == 1 and pos < doc_end:
                if self._eos_token_id is None:
                    raise RuntimeError("Packing index indicates EOS insertion but eos_token_id is None.")
                tokens.append(int(self._eos_token_id))
                pos += 1
                remaining -= 1

            if pos >= doc_end:
                doc_ptr += 1

        if len(tokens) != self._sequence_length:
            raise RuntimeError(
                "Packing produced an invalid block length. "
                f"Expected {self._sequence_length}, got {len(tokens)}."
            )

        input_tensor = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.ones((self._sequence_length,), dtype=torch.long)
        labels = input_tensor.clone()
        return {"input_ids": input_tensor, "attention_mask": attention_mask, "labels": labels}


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
