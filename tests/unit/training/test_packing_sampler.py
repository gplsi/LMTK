from __future__ import annotations

import torch
from torch.utils.data.distributed import DistributedSampler


def test_distributed_sampler_drop_last_produces_disjoint_index_sets() -> None:
    dataset = torch.utils.data.TensorDataset(torch.arange(101))

    sampler_rank0 = DistributedSampler(
        dataset,
        num_replicas=2,
        rank=0,
        shuffle=True,
        seed=123,
        drop_last=True,
    )
    sampler_rank1 = DistributedSampler(
        dataset,
        num_replicas=2,
        rank=1,
        shuffle=True,
        seed=123,
        drop_last=True,
    )

    idx0 = list(iter(sampler_rank0))
    idx1 = list(iter(sampler_rank1))

    assert len(idx0) == len(idx1)
    assert set(idx0).isdisjoint(set(idx1))
    assert len(set(idx0)) + len(set(idx1)) == 2 * len(idx0)

