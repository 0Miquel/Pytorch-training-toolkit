from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Tuple, Optional


@dataclass
class Configuration:
    # Dataset config
    labels: Optional[list]
    data_path: str
    n_classes: int
    n_frames: Optional[int] = field(default=10)

    # Early stopping config
    patience: int = field(default=10000)
    min_delta: float = field(default=0.0)
    max_mode: bool = field(default=False)
    monitor: str = field(default="loss")

    # Model config
    pretrained: bool = field(default=True)
    fine_tune: bool = field(default=True)

    # Trainer config
    batch_size: int = field(default=8, metadata={"help": "The number of batches for the training dataloader."})
    n_epochs: int = field(default=10, metadata={"help": "The number of epochs to run the training."})
    device: str = field(default='cuda')
    wandb: Optional[str] = field(default=None)
    loss_computed_by_model: Optional[bool] = field(default=False)

    # Optimizer config
    lr: float = field(default=0.001, metadata={"help": "The learning rate to be used for the optimize."})
    max_lr: Optional[float] = field(default=0.1)



