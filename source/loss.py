import torch.nn as nn


def get_loss(config):
    loss_name = config['loss_name']

    try:
        loss = getattr(nn, loss_name)()
    except AttributeError:
        try:
            loss = globals()[loss_name]()
        except KeyError:
            raise f"Loss with name {loss_name} not found"

    return loss


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        pass

    def forward(self, x, target):
        pass
