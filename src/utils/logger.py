import wandb
import time
from omegaconf import OmegaConf

from src.utils import save_figure


class Logger:
    def __init__(self, cfg):
        self.start_time = time.time()
        self.cfg = cfg
        if self.cfg.wandb is not None:
            self.project_name = self.cfg.wandb
            cfg_dict = object_to_dict(OmegaConf.to_object(cfg))
            self.run = wandb.init(project=self.project_name, config=cfg_dict)
        else:
            self.project_name = None

    def upload_metrics(self, train_metrics, val_metrics, epoch):
        logs = {"epoch": epoch}
        for metric_name, metric in train_metrics.items():
            logs["train/"+metric_name] = metric
        for metric_name, metric in val_metrics.items():
            logs["val/"+metric_name] = metric

        # save metrics locally
        self.save_metrics(logs)

        # log to wandb
        if self.project_name is not None:
            wandb.log(logs)

    def upload_media(self, figures):
        # save metrics locally
        self.save_media(figures)

        # log to wandb
        if self.project_name is not None:
            wandb_logs = {}
            for figure_name, figure in figures.items():
                wandb_logs["media/"+figure_name] = wandb.Image(figure)
            wandb.log(wandb_logs)

    def save_media(self, figures):
        for figure_name, figure in figures.items():
            figure_name = figure_name + ".png"
            save_figure(figure, figure_name)

    def save_metrics(self, logs):
        pass

    def finish(self):
        time_elapsed = time.time() - self.start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        if self.project_name is not None:
            self.run.finish()


def object_to_dict(obj):
    # Get all attributes of the object
    attributes = vars(obj)
    # Create a dictionary to store attribute names and values
    attributes_dict = {}
    # Iterate over attributes and add them to the dictionary
    for attr_name, attr_value in attributes.items():
        attributes_dict[attr_name] = attr_value
    return attributes_dict
