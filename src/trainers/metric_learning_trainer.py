from src.utils import (
    plot_umap,
    plot_top_k_similar,
    load_batch_to_device,
)
from src.metrics import MetricMonitor
from .base_trainer import BaseTrainer
from matplotlib.figure import Figure
from typing import Dict
import torch
from tqdm import tqdm


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

        # UMAP
        outputs = []
        labels = []
        img_paths = []
        for step, batch in enumerate(self.val_dl):
            batch = load_batch_to_device(batch, self.device)
            # forward
            output = self.predict(self.model, batch)
            outputs.append(output)
            labels.append(batch["label"].squeeze())
            img_paths = img_paths + batch["img_path"]
        umap_results = plot_umap(outputs, labels)

        # qualitative results
        sample = next(self.val_dl.__iter__())
        sample = load_batch_to_device(sample, self.device)
        sample_output = self.predict(self.model, sample)
        sample_labels = sample["label"].squeeze()
        sample_img_paths = sample["img_path"]
        top_k_results = plot_top_k_similar(sample_labels, sample_output, sample_img_paths, outputs, labels, img_paths)

        return {"umap": umap_results, "top_k": top_k_results}

    def compute_loss(self, output, sample):
        if self.loss is None:
            raise RuntimeError("`criterion` should not be None.")

        if self.miner is None:
            return self.loss(output, sample["label"].squeeze())
        else:
            miner_tuples = self.miner(output, sample["label"].squeeze())
            return self.loss(output, sample["label"].squeeze(), miner_tuples)

