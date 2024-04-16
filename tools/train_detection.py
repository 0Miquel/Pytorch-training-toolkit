import hydra
import torch
import torch.nn as nn

from hydra.core.config_store import ConfigStore
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import Configuration

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config_detection", node=Configuration)


@hydra.main(version_base=None, config_path="configs", config_name="config_detection.yaml")
def main(config: Configuration) -> None:
    pass


if __name__ == "__main__":
    main()
