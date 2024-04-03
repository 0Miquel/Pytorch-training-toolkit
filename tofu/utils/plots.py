import numpy as np
import wandb
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import hydra
import os
import cv2
from torchvision import transforms
import matplotlib.animation as animation


def save_fake_imgs(imgs, epoch):
    outputs_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    figs_dir = os.path.join(outputs_dir, 'figs')
    os.makedirs(figs_dir, exist_ok=True)
    imgs_path = os.path.join(figs_dir, f"fake_imgs_{epoch}.jpg")
    plt.imshow(np.transpose(imgs, (1, 2, 0)))
    plt.savefig(imgs_path)
    plt.close()


def create_animation_gif(img_list):
    # Create animation for the training process
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    # Save the animation as a GIF
    ani.save('animation.gif', writer='pillow', fps=2)  # 'pillow' is the writer for GIF format


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


def plot_segmentation_batch(y_pred, y_true, thr=0.5):
    y_pred = nn.Sigmoid()(y_pred)
    fig, axes = plt.subplots(nrows=y_pred.shape[0], ncols=2, figsize=(20, 20))
    for i, (y_pred_, y_true_) in enumerate(zip(y_pred, y_true)):
        y_pred_ = y_pred_.permute((1, 2, 0)).cpu().detach().numpy()
        y_true_ = y_true_.permute((1, 2, 0)).cpu().detach().numpy()
        y_pred_ = (y_pred_ > thr).astype(np.float32)
        if i == 0:
            axes[i, 0].set_title("Ground truth")
            axes[i, 1].set_title("Predicted")
        axes[i, 0].imshow(y_true_)
        axes[i, 1].imshow(y_pred_)
        axes[i, 0].set_axis_off()
        axes[i, 1].set_axis_off()
    plt.close('all')
    return fig


def segmentation_table(inputs, outputs, targets, labels):
    """
    Creates WandB table
    """
    table = wandb.Table(columns=["Prediction", "Ground truth"])

    for img, pred_mask, true_mask in zip(inputs, outputs, targets):
        pred_mask = nn.Sigmoid()(pred_mask)
        pred_mask = pred_mask.permute((1, 2, 0)).cpu().detach().numpy()
        true_mask = true_mask.permute((1, 2, 0)).cpu().detach().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.float32)
        true_mask = np.squeeze(true_mask).astype("uint8")
        pred_mask = np.squeeze(pred_mask).astype("uint8")

        pred_mask_img = wandb.Image(img, masks={"predictions": {"mask_data": pred_mask, "class_labels": labels}})
        true_mask_img = wandb.Image(img, masks={"ground_truth": {"mask_data": true_mask, "class_labels": labels}})

        table.add_data(pred_mask_img, true_mask_img)

    return table


def classificiation_table(inputs, outputs, targets, labels):
    table = wandb.Table(columns=["Input", "Prediction", "Ground truth", "Probabilities"])
    outputs = nn.Softmax(dim=1)(outputs)
    for img, output, target in zip(inputs, outputs, targets):
        # img = img.permute((1, 2, 0)).cpu().detach().numpy().astype("uint8")
        img = wandb.Image(img)
        output_label = labels[torch.argmax(output).item()]
        target_label = labels[torch.argmax(target).item()]

        # create the bar chart
        fig, ax = plt.subplots(figsize=(10, 10))
        max_idx = torch.argmax(output)
        bar_colors = ['g' if i == max_idx and output_label == target_label
                      else 'r' if i == max_idx and output_label != target_label else 'b' for i in range(len(labels))]
        ax.bar([*labels.values()], output.cpu().detach().numpy(), color=bar_colors)
        ax.set_ylabel('Probability')
        ax.set_title('Class Probabilities')
        ax.tick_params(axis='x', rotation=90)

        # save the chart as a WandB plot
        probabilities = wandb.Image(fig)
        plt.close(fig)
        table.add_data(img, output_label, target_label, probabilities)

    return table


def confussion_matrix_wandb(predictions, gt, labels):
    conf_matrix = wandb.plot.confusion_matrix(probs=None,
                                              y_true=gt, preds=predictions,
                                              class_names=labels)
    return conf_matrix

