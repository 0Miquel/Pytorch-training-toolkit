# from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
import torch.optim.lr_scheduler as lr_scheduler
from omegaconf import OmegaConf,open_dict


def get_scheduler(config, optimizer, steps_per_epoch, n_epochs):
    scheduler_name = config["scheduler_name"]
    settings = config["settings"]
    if scheduler_name == "OneCycleLR":
        # Need steps per epoch and n_epochs for OneCycleLR
        with open_dict(settings):
            settings.steps_per_epoch = steps_per_epoch
            settings.epochs = n_epochs

    try:
        scheduler = getattr(lr_scheduler, scheduler_name)(optimizer, **settings)
    except AttributeError:
        try:
            scheduler = globals()[scheduler_name](optimizer, **settings)
        except KeyError:
            raise f"Scheduler with name {scheduler_name} not found"

    return scheduler


class MyScheduler:
    def __init__(self, settings):
        pass
