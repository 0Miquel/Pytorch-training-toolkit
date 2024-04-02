from tofu import get_trainer
import hydra
from omegaconf import OmegaConf
import os
# change working directory to be in tools directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    cfg = OmegaConf.to_object(cfg)
    trainer = get_trainer(cfg)
    metric = trainer.fit()
    return metric


if __name__ == "__main__":
    main()
