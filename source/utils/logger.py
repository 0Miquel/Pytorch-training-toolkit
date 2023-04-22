import wandb
from omegaconf import DictConfig
from omegaconf import OmegaConf
from source.utils import *
from source.utils import segmentation_table


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
        self.labels = {i: label for i, label in enumerate(self.labels)}
        self.logs = {}

    # def add(self, og_imgs, outputs, targets, metrics, phase, exec_metrics):
    #     for metric_name, metric_value in metrics.items():
    #         self.logs[phase+"/"+metric_name] = metric_value
    #     # self.logs[phase] = metrics
    #     if self.task == "segmentation" and not self.sweep:
    #         table = segmentation_table(og_imgs, outputs, targets, self.labels)
    #         self.logs[phase+"/segm_results"] = table
    #     elif self.task == "classification" and not self.sweep:
    #         table = classificiation_table(og_imgs, outputs, targets, self.labels)
    #         self.logs[phase+"/class_results"] = table
    #         conf_matrix = confussion_matrix_wandb(exec_metrics["predictions"], exec_metrics["gt"], self.labels)
    #         self.logs[phase + "/conf_matrix"] = conf_matrix

    def add_classification_table(self, og_imgs, outputs, targets, phase):
        table = classificiation_table(og_imgs, outputs, targets, self.labels)
        self.logs[phase+"/class_results"] = table

    def add_segmentation_table(self, og_imgs, outputs, targets, phase):
        table = segmentation_table(og_imgs, outputs, targets, self.labels)
        self.logs[phase+"/segm_results"] = table

    def add(self, metrics, phase):
        for metric_name, metric_value in metrics.items():
            self.logs[phase+"/"+metric_name] = metric_value

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
