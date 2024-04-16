from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Tuple, Optional


@dataclass
class Configuration:
    labels: list
    data_path: str
    n_classes: int

    patience: int = field(default=10000)
    min_delta: float = field(default=0.0)

    pretrained: bool = field(default=True)
    fine_tune: bool = field(default=True)

    batch_size: int = field(default=8, metadata={"help": "The number of batches for the training dataloader."})
    n_epochs: int = field(default=10, metadata={"help": "The number of epochs to run the training."})
    lr: float = field(default=0.001, metadata={"help": "The learning rate to be used for the optimize."})
    max_lr: float = field(default=0.1)

    device: str = field(default='cuda')
    wandb: Optional[str] = field(default=None)

    loss_computed_by_model: bool = field(default=False)
