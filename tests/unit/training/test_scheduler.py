import torch
from datasets import Dataset

from src.tasks.training.utils import select_scheduler


def test_select_scheduler_fixed_returns_constant() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = Dataset.from_dict({"input_ids": [0, 1, 2, 3]})

    scheduler = select_scheduler(
        optimizer=optimizer,
        lr_scheduler="fixed",
        number_epochs=1,
        world_size=1,
        batch_size=1,
        train_dataset=dataset,
        warmup_proportion=0.0,
    )

    assert scheduler.__class__.__name__ == "LambdaLR"
