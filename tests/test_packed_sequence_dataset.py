import torch
from datasets import Dataset


def _ids(*docs: list[int]) -> Dataset:
    return Dataset.from_dict({"input_ids": list(docs), "length": [len(d) for d in docs]})


def test_packing_inserts_eos_and_blocks_are_fixed_length() -> None:
    from src.tasks.training.data.packing import PackedSequenceDataset

    docs = _ids([1, 2, 3], [4, 5], [6, 7, 8, 9])
    packed = PackedSequenceDataset(
        docs,
        sequence_length=4,
        insert_eos=True,
        eos_token_id=0,
    )

    assert len(packed) == 3

    first = packed[0]
    assert set(first.keys()) == {"input_ids", "attention_mask", "labels"}
    assert first["input_ids"].shape == (4,)
    assert torch.equal(first["attention_mask"], torch.ones(4, dtype=torch.long))
    assert torch.equal(first["labels"], first["input_ids"])

    assert torch.equal(packed[0]["input_ids"], torch.tensor([1, 2, 3, 0], dtype=torch.long))
    assert torch.equal(packed[1]["input_ids"], torch.tensor([4, 5, 0, 6], dtype=torch.long))
    assert torch.equal(packed[2]["input_ids"], torch.tensor([7, 8, 9, 0], dtype=torch.long))


def test_len_drops_remainder_tokens_by_default() -> None:
    from src.tasks.training.data.packing import PackedSequenceDataset

    docs = _ids([10, 11, 12, 13, 14])
    packed = PackedSequenceDataset(
        docs,
        sequence_length=4,
        insert_eos=True,
        eos_token_id=0,
    )

    # stream length = 6 (doc + eos), so only 1 full block is available
    assert len(packed) == 1
    assert torch.equal(packed[0]["input_ids"], torch.tensor([10, 11, 12, 13], dtype=torch.long))


def test_distributed_sampler_drop_last_is_enforced_in_packing_mode() -> None:
    from src.tasks.training.data.packing import build_packing_dataloader

    dataset = torch.utils.data.TensorDataset(torch.arange(10))

    loader = build_packing_dataloader(
        dataset,
        split="train",
        batch_size=2,
        num_workers=0,
        shuffle=True,
        sampler_drop_last=True,
        seed=123,
        rank=0,
        world_size=2,
    )

    assert loader.drop_last is True
    assert loader.sampler.__class__.__name__ == "DistributedSampler"
    assert getattr(loader.sampler, "drop_last") is True


def test_empty_docs_are_skipped() -> None:
    from src.tasks.training.data.packing import PackedSequenceDataset

    docs = _ids([], [1, 2, 3, 4], [])
    packed = PackedSequenceDataset(
        docs,
        sequence_length=4,
        insert_eos=False,
        eos_token_id=None,
    )

    assert len(packed) == 1
    assert torch.equal(packed[0]["input_ids"], torch.tensor([1, 2, 3, 4], dtype=torch.long))
