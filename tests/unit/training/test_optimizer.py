import torch

from src.tasks.training.utils import select_optimizer


class ToyRMSNorm(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))
        self.bias = torch.nn.Parameter(torch.zeros(4))


class OptimizerGroupingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.linear = torch.nn.Linear(4, 4)
        self.layer_norm = torch.nn.LayerNorm(4)
        self.rms_norm = ToyRMSNorm()
        self.frozen = torch.nn.Linear(4, 4)
        for param in self.frozen.parameters():
            param.requires_grad = False


def _param_ids(params: list[torch.nn.Parameter]) -> set[int]:
    return {id(param) for param in params}


def test_select_optimizer_default_keeps_single_group_behavior() -> None:
    model = OptimizerGroupingModel()

    optimizer = select_optimizer(
        "adamw",
        model,
        lr=1e-4,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        no_decay_norms=False,
    )

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert _param_ids(optimizer.param_groups[0]["params"]) == _param_ids(list(model.parameters()))


def test_select_optimizer_can_exclude_norms_and_biases_from_weight_decay() -> None:
    model = OptimizerGroupingModel()

    optimizer = select_optimizer(
        "adamw",
        model,
        lr=1e-4,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        no_decay_norms=True,
    )

    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.1, 0.0]
    decay_ids = _param_ids(optimizer.param_groups[0]["params"])
    no_decay_ids = _param_ids(optimizer.param_groups[1]["params"])
    grouped_ids = decay_ids | no_decay_ids

    assert id(model.embedding.weight) in decay_ids
    assert id(model.linear.weight) in decay_ids
    assert id(model.linear.bias) in no_decay_ids
    assert id(model.layer_norm.weight) in no_decay_ids
    assert id(model.layer_norm.bias) in no_decay_ids
    assert id(model.rms_norm.weight) in no_decay_ids
    assert id(model.rms_norm.bias) in no_decay_ids
    assert id(model.frozen.weight) not in grouped_ids
    assert id(model.frozen.bias) not in grouped_ids
