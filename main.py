from hydra import compose, initialize
from source import train
import argparse
import sys


def main(cfg, wandb_pname):
    train(wandb_pname, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", default="config_classification")
    parser.add_argument("--wandb_name", default=None)
    args = parser.parse_args(sys.argv[1:])
    config_name = args.config_name
    wandb_name = args.wandb_name

    initialize(version_base=None, config_path="conf")
    config = compose(config_name=config_name)
    main(config, wandb_name)
