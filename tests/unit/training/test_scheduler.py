import torch
import pytest

from src.tasks.training.utils import select_scheduler


class _ToyDataset:
    def __init__(self, n: int) -> None:
        self.n = int(n)

    def __len__(self) -> int:
        return self.n


def test_select_scheduler_fixed_returns_constant() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = _ToyDataset(4)

    scheduler = select_scheduler(
        optimizer=optimizer,
        lr_scheduler="fixed",
        number_epochs=1,
        world_size=1,
        batch_size=1,
        train_dataset=dataset,
        warmup_proportion=0.0,
        base_lr=1e-3,
    )

    assert scheduler.__class__.__name__ == "LambdaLR"


def _simulate_lrs(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_steps: int,
) -> list[float]:
    lrs: list[float] = []
    for _ in range(num_steps):
        lrs.append(float(optimizer.param_groups[0]["lr"]))
        optimizer.step()
        scheduler.step()
    return lrs


def _simulate_group_lrs(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_steps: int,
) -> list[list[float]]:
    lrs: list[list[float]] = []
    for _ in range(num_steps):
        lrs.append([float(group["lr"]) for group in optimizer.param_groups])
        optimizer.step()
        scheduler.step()
    return lrs


def test_warmup_linear_respects_min_lr_and_returns_to_min_lr() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = _ToyDataset(32)

    scheduler = select_scheduler(
        optimizer=optimizer,
        lr_scheduler="warmup_linear",
        number_epochs=1,
        world_size=1,
        batch_size=1,
        train_dataset=dataset,
        warmup_proportion=0.25,
        base_lr=1e-3,
        min_lr=2e-4,
        total_steps=12,
    )
    lrs = _simulate_lrs(optimizer, scheduler, num_steps=12)

    assert lrs[0] == pytest.approx(2e-4)
    assert max(lrs) <= 1e-3 + 1e-10
    assert max(lrs) >= 9e-4
    assert lrs[-1] == pytest.approx(2e-4, rel=1e-4, abs=1e-8)


def test_warmup_cosine_uses_explicit_max_lr_peak() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = _ToyDataset(64)

    scheduler = select_scheduler(
        optimizer=optimizer,
        lr_scheduler="warmup_cosine",
        number_epochs=1,
        world_size=1,
        batch_size=1,
        train_dataset=dataset,
        warmup_proportion=0.2,
        base_lr=1e-3,
        min_lr=3e-4,
        max_lr=2e-3,
        total_steps=20,
    )
    lrs = _simulate_lrs(optimizer, scheduler, num_steps=20)

    assert lrs[0] == pytest.approx(3e-4)
    assert max(lrs) <= 2e-3 + 1e-10
    assert max(lrs) >= 1.8e-3
    assert lrs[-1] == pytest.approx(3e-4, rel=1e-4, abs=1e-8)


def test_cosine_scheduler_with_min_lr_floor() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = _ToyDataset(24)

    scheduler = select_scheduler(
        optimizer=optimizer,
        lr_scheduler="cosine",
        number_epochs=1,
        world_size=1,
        batch_size=1,
        train_dataset=dataset,
        warmup_proportion=0.0,
        base_lr=1e-3,
        min_lr=4e-4,
        total_steps=10,
    )
    lrs = _simulate_lrs(optimizer, scheduler, num_steps=10)

    assert lrs[0] == pytest.approx(1e-3)
    assert lrs[-1] == pytest.approx(4e-4, rel=1e-4, abs=1e-8)
    assert min(lrs) >= 4e-4 - 1e-10


def test_bounded_cosine_ignores_warmup_proportion() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = _ToyDataset(24)

    scheduler = select_scheduler(
        optimizer=optimizer,
        lr_scheduler="cosine",
        number_epochs=1,
        world_size=1,
        batch_size=1,
        train_dataset=dataset,
        warmup_proportion=0.5,  # should be ignored for cosine
        min_lr=2e-4,
        total_steps=6,
    )
    lrs = _simulate_lrs(optimizer, scheduler, num_steps=6)

    assert lrs[0] == pytest.approx(1e-3)
    assert lrs[-1] == pytest.approx(2e-4, rel=1e-4, abs=1e-8)


def test_scheduler_rejects_invalid_lr_bounds() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = _ToyDataset(4)

    with pytest.raises(ValueError, match="min_lr"):
        select_scheduler(
            optimizer=optimizer,
            lr_scheduler="warmup_linear",
            number_epochs=1,
            world_size=1,
            batch_size=1,
            train_dataset=dataset,
            warmup_proportion=0.1,
            base_lr=1e-3,
            min_lr=2e-3,
            total_steps=10,
        )


def test_single_warmup_step_starts_from_min_lr() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = _ToyDataset(4)

    scheduler = select_scheduler(
        optimizer=optimizer,
        lr_scheduler="warmup_linear",
        number_epochs=1,
        world_size=1,
        batch_size=1,
        train_dataset=dataset,
        warmup_proportion=0.25,
        base_lr=1e-3,
        min_lr=2e-4,
        total_steps=4,  # warmup_steps == 1
    )
    lrs = _simulate_lrs(optimizer, scheduler, num_steps=4)

    assert lrs[0] == pytest.approx(2e-4)


def test_default_scheduler_path_preserves_param_group_learning_rates() -> None:
    layer_a = torch.nn.Linear(2, 2)
    layer_b = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(
        [
            {"params": layer_a.parameters(), "lr": 1e-3},
            {"params": layer_b.parameters(), "lr": 2e-3},
        ]
    )
    dataset = _ToyDataset(8)
    before = [float(group["lr"]) for group in optimizer.param_groups]

    _ = select_scheduler(
        optimizer=optimizer,
        lr_scheduler="warmup_linear",
        number_epochs=1,
        world_size=1,
        batch_size=1,
        train_dataset=dataset,
        warmup_proportion=0.0,
        total_steps=4,
    )
    after = [float(group["lr"]) for group in optimizer.param_groups]

    assert before == after


def test_bounded_scheduler_applies_absolute_min_lr_per_param_group() -> None:
    layer_a = torch.nn.Linear(2, 2)
    layer_b = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(
        [
            {"params": layer_a.parameters(), "lr": 1e-3},
            {"params": layer_b.parameters(), "lr": 2e-3},
        ]
    )
    dataset = _ToyDataset(16)

    scheduler = select_scheduler(
        optimizer=optimizer,
        lr_scheduler="warmup_linear",
        number_epochs=1,
        world_size=1,
        batch_size=1,
        train_dataset=dataset,
        warmup_proportion=0.5,
        min_lr=5e-4,
        total_steps=4,
    )
    lrs = _simulate_group_lrs(optimizer, scheduler, num_steps=4)

    assert lrs[0][0] == pytest.approx(5e-4)
    assert lrs[0][1] == pytest.approx(5e-4)
    assert lrs[1][0] == pytest.approx(1e-3)
    assert lrs[1][1] == pytest.approx(2e-3)
    assert lrs[-1][0] == pytest.approx(5e-4)
    assert lrs[-1][1] == pytest.approx(5e-4)
