import wandb
import time
from omegaconf import OmegaConf
import hydra
import os
import csv


class Logger:
    def __init__(self, cfg):
        self.results_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
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
        figs_dir = os.path.join(self.results_dir, 'figs')
        os.makedirs(figs_dir, exist_ok=True)
        for figure_name, figure in figures.items():
            fig_filename = os.path.join(figs_dir, figure_name + ".png")
            figure.savefig(fig_filename)

    def save_metrics(self, logs):
        csv_filename = os.path.join(self.results_dir, "metrics.csv")

        if not os.path.exists(csv_filename):
            # If the file doesn't exist, write the keys to the CSV file
            with open(csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(list(logs.keys()))

        # Write the values corresponding to each key
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(logs.keys()))
            writer.writerow(logs)

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
