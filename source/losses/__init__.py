

import torch.nn as nn
import segmentation_models_pytorch.losses as smp_losses


def get_loss(config):
    loss_name = config['loss_name']

    try:
        loss = getattr(nn, loss_name)()
    except AttributeError:
        try:
            loss = getattr(smp_losses, loss_name)()
        except AttributeError:
            try:
                loss = globals()[loss_name]()
            except KeyError:
                raise f"Loss with name {loss_name} not found"

    return loss
