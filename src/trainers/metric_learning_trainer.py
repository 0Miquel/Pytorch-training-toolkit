from src.utils import (
    load_batch_to_device,
)
from .base_trainer import BaseTrainer
from matplotlib.figure import Figure
from typing import Dict
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import umap
import umap.plot
import cv2


class MetricLearningTrainer(BaseTrainer):
    """
    Simple metric learning trainer implementation. It computes the metric loss only from the embeddings.
    More advanced trainers can be found here:
    https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/
    """
    def __init__(
            self,
            config,
            train_dl,
            val_dl,
            model,
            optimizer,
            test_dl=None,
            miner=None,
            criterion=None,
            scheduler=None
    ):
        super().__init__(
            config=config,
            train_dl=train_dl,
            val_dl=val_dl,
            test_dl=test_dl,
            criterion=criterion,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler
        )
        self.miner = miner

    def compute_loss(self, output, batch):
        if self.criterion is None:
            raise RuntimeError("`criterion` should not be None.")

        if self.miner is None:
            return self.criterion(output, batch["label"].squeeze())
        else:
            miner_tuples = self.miner(output, batch["label"].squeeze())
            return self.criterion(output, batch["label"].squeeze(), miner_tuples)

    def generate_media(self) -> Dict[str, Figure]:
        """
        Generate media from output and batch.
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
        outputs = torch.cat(outputs)
        labels = torch.cat(labels)

        umap_results = self.plot_umap(outputs, labels)

        # qualitative results
        batch = next(self.val_dl.__iter__())
        batch = load_batch_to_device(batch, self.device)
        batch_output = self.predict(self.model, batch)
        batch_labels = batch["label"].squeeze()
        batch_img_paths = batch["img_path"]
        top_k_results = self.plot_top_k_similar(batch_labels, batch_output, batch_img_paths, outputs, labels, img_paths)

        return {"umap": umap_results, "top_k": top_k_results}

    @staticmethod
    def plot_top_k_similar(sample_labels, sample_output, sample_img_paths, outputs, labels, img_paths, k=5):
        outputs = outputs.detach().cpu().numpy()
        labels = labels.squeeze().detach().cpu().numpy()
        sample_labels = sample_labels.squeeze().detach().cpu().numpy()
        sample_output = sample_output.detach().cpu().numpy()

        cosine_similarities = cosine_similarity(sample_output, outputs)
        top_k_indices = np.argsort(cosine_similarities, axis=1)[:, -k:]

        num_samples = len(sample_labels)

        fig, axes = plt.subplots(num_samples, k + 1, figsize=(k + 1, num_samples))
        fig.tight_layout()

        for i, (top_k_idx_row, sample_label, sample_img_path, similarity_row) in enumerate(
                zip(top_k_indices, sample_labels, sample_img_paths, cosine_similarities)):

            # Load the sample image
            sample_img = cv2.imread(sample_img_path)[:, :, ::-1]  # Image.open(sample_img_path)
            axes[i, 0].imshow(sample_img)
            axes[i, 0].set_title(f"{sample_label}")
            axes[i, 0].axis('off')

            # Load and plot the k closest images
            for j, idx in enumerate(top_k_idx_row):
                img_path = img_paths[idx]
                img = cv2.imread(img_path)[:, :, ::-1]  # Image.open(img_path)
                ax = axes[i, j + 1]
                ax.imshow(img)
                ax.set_title(f"{labels[idx]} - {similarity_row[idx]:.4f}")
                ax.axis('off')

                # Highlight with green rectangle if the label is the same, otherwise with red
                rect_color = 'green' if labels[idx] == sample_label else 'red'
                width = img.shape[1]
                height = img.shape[0]
                rect = plt.Rectangle((0, 0), width, height, linewidth=4, edgecolor=rect_color, facecolor='none')
                ax.add_patch(rect)
                ax.set_xlim(0, width)
                ax.set_ylim(height, 0)

        plt.close('all')
        return fig

    @staticmethod
    def plot_umap(data, labels):
        data = data.detach().cpu().numpy()
        labels = labels.squeeze().detach().cpu().numpy()

        mapper = umap.UMAP().fit(data)
        fig, ax = plt.subplots(figsize=(10, 10))
        umap.plot.points(mapper, ax=ax, labels=labels)

        plt.close('all')
        return fig
