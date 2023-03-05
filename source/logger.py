import wandb
from omegaconf import DictConfig
from omegaconf import OmegaConf
from source.utils.plots import *


def get_logger(cfg, wandb_name):
    return Logger(cfg, wandb_name)


class Logger:
    def __init__(self, cfg, wandb_name):
        self.project_name = wandb_name
        if cfg is not None:
            self.sweep = False
            self.cfg = OmegaConf.to_object(cfg)
            self.run = wandb.init(project=self.project_name, config=self.cfg)
        else:
            # If called by wandb.agent,
            # this config will be set by Sweep Controller
            self.sweep = True
            self.run = wandb.init(project=self.project_name, config=cfg)
            config = wandb.config
            self.cfg = nested_dict(config)

        self.labels = self.cfg["dataset"]["settings"]["labels"]
        self.task = self.cfg["trainer"]["task"]
        self.logs = {}

    def add(self, og_imgs, outputs, targets, metrics, phase):
        self.logs[phase] = metrics
        if self.task == "segmentation":
            table = segmentation_table(og_imgs, outputs, targets, self.labels)
            self.logs[phase+"_results"] = table
        elif self.task == "classification":
            table = classificiation_table(og_imgs, outputs, targets, self.labels)
            self.logs[phase+"_results"] = table

    def upload(self):
        wandb.log(self.logs)

    def log_model(self, model_path):
        if not self.sweep:
            art = wandb.Artifact(self.project_name, type="model")
            art.add_file(model_path)
            self.run.log_artifact(art)

    def finish(self):
        self.run.finish()


def nested_dict(original_dict):
    nested_dict = {}
    for key, value in original_dict.items():
        parts = key.split(".")
        d = nested_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return nested_dict
