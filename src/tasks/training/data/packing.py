from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


@dataclass(frozen=True)
class _DocMeta:
    dataset_index: int
    length: int
    effective_length: int


class PackedSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """
    Map-style dataset that exposes packed fixed-length blocks built from variable-length token sequences.

    Required input column: input_ids (list[int])
    Optional input column: length (int) to avoid recomputing len(input_ids)

    Packing algorithm:
    - Define the logical token stream by concatenating documents. If insert_eos is true, append eos_token_id
      after each document unless the document already ends with eos_token_id (avoid double-eos).
    - Define block i as the slice [i * sequence_length : (i + 1) * sequence_length] from the stream.
    - Always drop tail tokens that don't fit into a full block (floor division for __len__).
    """

    def __init__(
        self,
        hf_dataset: Any,
        *,
        sequence_length: int,
        insert_eos: bool,
        eos_token_id: int | None,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be a positive integer.")

        self._hf_dataset = hf_dataset
        self._sequence_length = int(sequence_length)
        self._insert_eos = bool(insert_eos)
        self._eos_token_id = eos_token_id

        if self._insert_eos and self._eos_token_id is None:
            raise ValueError("eos_token_id must be provided when insert_eos is true.")

        self.skipped_empty_docs = 0
        self.inserted_eos_tokens = 0

        # Select only non-empty docs for packing.
        doc_metas: list[_DocMeta] = []
        starts: list[int] = [0]

        has_length = hasattr(hf_dataset, "column_names") and "length" in hf_dataset.column_names

        for dataset_index in range(len(hf_dataset)):
            row = hf_dataset[dataset_index]
            input_ids = row["input_ids"]
            length = int(row["length"]) if has_length else len(input_ids)
            if length <= 0:
                self.skipped_empty_docs += 1
                continue

            effective_length = length
            if self._insert_eos:
                if not input_ids:
                    raise ValueError(
                        f"Invalid dataset row {dataset_index}: length={length} but input_ids is empty."
                    )
                if has_length and length != len(input_ids):
                    raise ValueError(
                        f"Invalid dataset row {dataset_index}: length={length} does not match len(input_ids)={len(input_ids)}."
                    )
                if int(input_ids[-1]) != int(self._eos_token_id):
                    effective_length += 1
                    self.inserted_eos_tokens += 1

            doc_metas.append(
                _DocMeta(
                    dataset_index=dataset_index,
                    length=length,
                    effective_length=effective_length,
                )
            )
            starts.append(starts[-1] + effective_length)

        self._docs = doc_metas
        self._doc_starts = starts  # len = num_docs + 1
        self._total_tokens = starts[-1]
        self._num_blocks = self._total_tokens // self._sequence_length
        self.dropped_tail_tokens = self._total_tokens - (self._num_blocks * self._sequence_length)

    def __len__(self) -> int:
        return self._num_blocks

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= self._num_blocks:
            raise IndexError(idx)

        start_pos = idx * self._sequence_length
        remaining = self._sequence_length

        out = torch.empty(self._sequence_length, dtype=torch.long)
        write_offset = 0

        while remaining > 0:
            doc_i = bisect.bisect_right(self._doc_starts, start_pos) - 1
            if doc_i < 0 or doc_i >= len(self._docs):
                raise RuntimeError("PackedSequenceDataset internal mapping error.")

            doc_meta = self._docs[doc_i]
            doc_start = self._doc_starts[doc_i]
            local_pos = start_pos - doc_start

            take = min(remaining, doc_meta.effective_length - local_pos)
            row = self._hf_dataset[doc_meta.dataset_index]
            input_ids: list[int] = row["input_ids"]

            if local_pos < doc_meta.length:
                take_from_doc = min(take, doc_meta.length - local_pos)
                out[write_offset : write_offset + take_from_doc] = torch.as_tensor(
                    input_ids[local_pos : local_pos + take_from_doc],
                    dtype=torch.long,
                )
                write_offset += take_from_doc
                start_pos += take_from_doc
                remaining -= take_from_doc
                take -= take_from_doc

            if take:
                # Must be an inserted EOS token (at most one).
                if not self._insert_eos or self._eos_token_id is None:
                    raise RuntimeError("Encountered inserted EOS without configuration.")
                out[write_offset] = int(self._eos_token_id)
                write_offset += 1
                start_pos += 1
                remaining -= 1

        attention_mask = torch.ones(self._sequence_length, dtype=torch.long)
        labels = out.clone()

        return {
            "input_ids": out,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def build_packing_dataloader(
    dataset: Dataset[Any],
    *,
    split: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    sampler_drop_last: bool,
    seed: int | None,
    rank: int,
    world_size: int,
) -> DataLoader[Any]:
    """
    Build a DataLoader for packing mode with an explicit sampler policy.

    Required behaviors:
    - If world_size == 1: use DataLoader(shuffle=shuffle for train, else False) and no sampler.
    - If world_size > 1: use DistributedSampler(num_replicas=world_size, rank=rank, drop_last=sampler_drop_last, seed=seed or 0),
      DataLoader(shuffle=False), and drop_last=True for the train split.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    is_train = split == "train"
    effective_shuffle = bool(shuffle) if is_train else False

    if world_size <= 1:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=effective_shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=is_train,
        )

    sampler = DistributedSampler(
        dataset,
        num_replicas=int(world_size),
        rank=int(rank),
        shuffle=effective_shuffle,
        drop_last=bool(sampler_drop_last),
        seed=int(seed or 0),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
    )
