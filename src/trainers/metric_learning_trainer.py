from src.utils import (
    plot_umap,
    load_batch_to_device
)
from src.metrics import MetricMonitor
from .base_trainer import BaseTrainer
from matplotlib.figure import Figure
from typing import Dict


class MetricLearningTrainer(BaseTrainer):
    def __init__(
            self,
            config,
            train_dl,
            val_dl,
            model,
            optimizer,
            miner=None,
            criterion=None,
            scheduler=None
    ):
        super().__init__(
            config=config,
            train_dl=train_dl,
            val_dl=val_dl,
            criterion=criterion,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler
        )
        self.miner = miner

    def generate_media(self) -> Dict[str, Figure]:
        """
        Generate media from output and sample.
        """
        self.model.eval()
        outputs = []
        labels = []

        for step, batch in enumerate(self.val_dl):
            batch = load_batch_to_device(batch, self.device)
            # forward
            output = self.model(batch["x"])
            outputs.append(output)
            labels.append(batch["label"].squeeze())

        umap_results = plot_umap(outputs, labels)
        return {"umap": umap_results}

    def compute_loss(self, output, sample):
        if self.loss is None:
            raise RuntimeError("`criterion` should not be None.")

        if self.miner is None:
            return self.loss(output, sample["label"].squeeze())
        else:
            miner_tuples = self.miner(output, sample["label"].squeeze())
            return self.loss(output, sample["label"].squeeze(), miner_tuples)

