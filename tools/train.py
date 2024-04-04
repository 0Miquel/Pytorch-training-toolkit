import sys
sys.path.append('../')  # Add the parent directory to the Python path

from src import get_trainer
import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="config_segmentation")
def main(cfg):
    cfg = OmegaConf.to_object(cfg)
    trainer = get_trainer(cfg)
    metric = trainer.fit()
    return metric


if __name__ == "__main__":
    main()
