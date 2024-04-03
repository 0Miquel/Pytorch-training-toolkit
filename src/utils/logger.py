import wandb
from src.utils import tensors_to_ims, classificiation_table, segmentation_table


class Logger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.project_name = self.cfg["trainer"]["wandb"]
        self.run = wandb.init(project=self.project_name, config=cfg)
        self.logs = {}
        if "labels" in self.cfg["dataset"]["settings"].keys():
            self.labels = self.cfg["dataset"]["settings"]["labels"]
            self.labels = {i: label for i, label in enumerate(self.labels)}

    def add_classification_table(self, inputs, outputs, targets, phase):
        og_imgs = tensors_to_ims(inputs)
        table = classificiation_table(og_imgs, outputs, targets, self.labels)
        self.logs[phase+"/class_results"] = table

    def add_segmentation_table(self, inputs, outputs, targets, phase):
        og_imgs = tensors_to_ims(inputs)
        table = segmentation_table(og_imgs, outputs, targets, self.labels)
        self.logs[phase+"/segm_results"] = table

    def add(self, metrics, phase):
        for metric_name, metric_value in metrics.items():
            self.logs[phase+"/"+metric_name] = metric_value

    def upload(self):
        wandb.log(self.logs)

    def finish(self):
        self.run.finish()
