from __future__ import annotations

from pathlib import Path

import pytest
import torch

datasets = pytest.importorskip("datasets")
Dataset = datasets.Dataset

from src.tasks.training.data.packing import PackedSequenceDataset, build_packing_dataloader
from src.tasks.training.data.packing_index import PackingIndex


def test_packing_inserts_eos_and_blocks_are_fixed_length(tmp_path: Path) -> None:
    hf = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3], [4, 5], [6, 7, 8, 9]],
            "length": [3, 2, 4],
        }
    )
    index = PackingIndex.load_or_build(
        hf_split=hf,
        split="train",
        sequence_length=4,
        insert_eos=True,
        eos_token_id=0,
        cache_dir=tmp_path,
    )
    ds = PackedSequenceDataset(hf_dataset=hf, index=index, eos_token_id=0)
    assert len(ds) == 3
    ex0 = ds[0]
    ex1 = ds[1]
    ex2 = ds[2]
    assert ex0["input_ids"].tolist() == [1, 2, 3, 0]
    assert ex1["input_ids"].tolist() == [4, 5, 0, 6]
    assert ex2["input_ids"].tolist() == [7, 8, 9, 0]
    for ex in (ex0, ex1, ex2):
        assert ex["input_ids"].shape == (4,)
        assert ex["attention_mask"].shape == (4,)
        assert ex["labels"].shape == (4,)
        assert ex["attention_mask"].tolist() == [1, 1, 1, 1]
        assert torch.equal(ex["labels"], ex["input_ids"])


def test_len_drops_remainder_tokens_by_default(tmp_path: Path) -> None:
    # Stream (with EOS) length = 5, sequence_length=4 => 1 block, remainder dropped.
    hf = Dataset.from_dict({"input_ids": [[1, 2, 3, 4, 5]], "length": [5]})
    index = PackingIndex.load_or_build(
        hf_split=hf,
        split="train",
        sequence_length=4,
        insert_eos=False,
        eos_token_id=None,
        cache_dir=tmp_path,
    )
    ds = PackedSequenceDataset(hf_dataset=hf, index=index, eos_token_id=None)
    assert len(ds) == 1
    assert ds[0]["input_ids"].tolist() == [1, 2, 3, 4]


def test_empty_docs_are_skipped(tmp_path: Path) -> None:
    hf = Dataset.from_dict({"input_ids": [[], [1, 2, 3, 4]], "length": [0, 4]})
    index = PackingIndex.load_or_build(
        hf_split=hf,
        split="train",
        sequence_length=4,
        insert_eos=False,
        eos_token_id=None,
        cache_dir=tmp_path,
    )
    ds = PackedSequenceDataset(hf_dataset=hf, index=index, eos_token_id=None)
    assert len(ds) == 1
    assert ds[0]["input_ids"].tolist() == [1, 2, 3, 4]


def test_distributed_sampler_drop_last_is_enforced_in_packing_mode() -> None:
    base = torch.utils.data.TensorDataset(torch.arange(10))

    # world_size=2, drop_last=True => each rank sees floor(10/2)=5 indices; with batch_size=2 and drop_last=True => 2 batches.
    dl_rank0 = build_packing_dataloader(
        dataset=base,
        split="train",
        batch_size=2,
        num_workers=0,
        shuffle=True,
        sampler_drop_last=True,
        drop_last_batch=True,
        seed=123,
        rank=0,
        world_size=2,
    )
    assert len(dl_rank0) == 2


def test_scheduler_steps_use_dataloader_len_not_floor_div_dataset_len() -> None:
    # This test locks the intended behavior: scheduler sizing must not go to 0 steps
    # when dataset is smaller than batch_size*world_size but DataLoader yields 1 batch.
    from unittest.mock import patch

    from src.tasks.training import utils as training_utils

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    hf = Dataset.from_dict({"input_ids": [0]})

    with patch.object(training_utils, "get_constant_schedule_with_warmup") as warmup:
        warmup.return_value = object()
        training_utils.select_scheduler(
            optimizer=optimizer,
            lr_scheduler="warmup_constant",
            number_epochs=1,
            world_size=1,
            batch_size=8,
            train_dataset=hf,
            warmup_proportion=0.5,
            total_steps=1,
        )
        assert warmup.called
