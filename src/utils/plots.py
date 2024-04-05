import numpy as np
import wandb
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torchvision.utils as vutils
import cv2
import umap
import umap.plot
from src.utils import save_figure


def plot_umap(data, labels):
    data = torch.cat(data)
    data = data.detach().cpu().numpy()
    labels = torch.cat(labels)
    labels = labels.squeeze().detach().cpu().numpy()

    mapper = umap.UMAP().fit(data)
    fig, ax = plt.subplots(figsize=(10, 10))
    umap.plot.points(mapper, ax=ax, labels=labels)

    plt.close('all')
    return fig


def plot_fake_imgs(generator, latent_vector_size):
    device = next(generator.parameters()).device
    fixed_noise = torch.randn(64, latent_vector_size, 1, 1, device=device)
    fake = generator(fixed_noise).detach().cpu()
    fig = vutils.make_grid(fake, padding=2, normalize=True)

    plt.close('all')
    return fig


def plot_segmentation_results(x, y_pred, y_true, thr=0.5):
    imgs = tensors_to_ims(x)
    y_pred = nn.Sigmoid()(y_pred)

    fig, axes = plt.subplots(nrows=y_pred.shape[0], ncols=3, figsize=(6, y_pred.shape[0]))
    fig.tight_layout()

    for i, (img, y_pred_, y_true_) in enumerate(zip(imgs, y_pred, y_true)):
        y_pred_ = y_pred_.permute((1, 2, 0)).cpu().detach().numpy()
        y_true_ = y_true_.permute((1, 2, 0)).cpu().detach().numpy()
        y_pred_ = (y_pred_ > thr).astype(np.float32)
        if i == 0:
            axes[i, 0].set_title("Image")
            axes[i, 1].set_title("Ground truth")
            axes[i, 2].set_title("Predicted")
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(y_true_)
        axes[i, 2].imshow(y_pred_)

    plt.close('all')
    return fig


def plot_classification_results(inputs, outputs, targets, labels):
    images = tensors_to_ims(inputs)
    outputs = nn.Softmax(dim=1)(outputs)

    fig, ax = plt.subplots(nrows=outputs.shape[0], ncols=2, figsize=(4, outputs.shape[0]))
    fig.tight_layout()

    for i, (img, output, target) in enumerate(zip(images, outputs, targets)):
        if i == 0:
            ax[i, 0].set_title("Image")
            ax[i, 1].set_title("Class Probabilities")

        output_label = labels[torch.argmax(output).item()]
        target_label = labels[torch.argmax(target).item()]

        max_idx = torch.argmax(output)
        bar_colors = ['g' if j == max_idx and output_label == target_label
                      else 'r' if j == max_idx and output_label != target_label else 'b' for j in range(len(labels))]
        ax[i, 0].imshow(img)
        ax[i, 1].bar(labels, output.cpu().detach().numpy(), color=bar_colors)
        ax[i, 1].set_ylim(0, 1.0)
        ax[i, 1].tick_params(axis='x', rotation=30)

    plt.close('all')
    return fig


def norm_tensor_to_original_im(norm_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    norm_array = norm_tensor.detach().cpu().numpy().transpose((1, 2, 0))
    reverted_img = (norm_array * std + mean) * 255
    im = reverted_img.astype("uint8")
    return im


def tensors_to_ims(tensor_ims):
    ims = []
    for tensor_im in tensor_ims:
        im = norm_tensor_to_original_im(tensor_im)
        ims.append(im)
    return ims
