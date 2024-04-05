import wandb
from src.utils import save_figure


class Logger:
    def __init__(self, cfg):
        self.logs = {}
        self.cfg = cfg
        if self.cfg["trainer"]["wandb"] is not None:
            self.project_name = self.cfg["trainer"]["wandb"]
            self.run = wandb.init(project=self.project_name, config=cfg)
        else:
            self.project_name = None

    def add_metrics(self, metrics, phase):
        for metric_name, metric_value in metrics.items():
            self.logs[phase+"/"+metric_name] = metric_value

    def add_media(self, figures):
        for figure_name, figure in figures.items():
            self.logs["media/"+figure_name] = figure

    def upload(self, epoch):
        self.logs["epoch"] = epoch
        # log to wandb (both metrics and media)
        if self.project_name is not None:
            self.wandb_log()
        # save media locally
        self.save_media()
        # save metrics locally
        self.save_metrics()

    def wandb_log(self):
        wandb_logs = self.logs.copy()
        for key, value in wandb_logs.items():
            if "media" in key:
                # convert to wandb.Image if it is media
                wandb_logs[key] = wandb.Image(value)
        wandb.log(self.logs)

    def save_media(self):
        epoch = self.logs["epoch"]
        for key, value in self.logs.items():
            if "media" in key:
                figure_name = key.split("/")[1] + f"_epoch_{epoch}.png"
                save_figure(value, figure_name)

    def save_metrics(self):
        pass

    def finish(self):
        if self.project_name is not None:
            self.run.finish()
