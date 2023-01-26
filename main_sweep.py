from source.trainer import train
import wandb
from source.utils.io import *
import argparse
import sys


def main(sweep_fname, sweep_count, wandb_pname):
    cfg_sweep = load_yaml_config(f"sweeps/{sweep_fname}")
    sweep_id = wandb.sweep(cfg_sweep, project=wandb_pname)
    wandb.agent(sweep_id, train, count=sweep_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", default="sweep.yaml")
    parser.add_argument("--n_runs", default=5, type=int)
    parser.add_argument("--wandb_name", default="test")
    args = parser.parse_args(sys.argv[1:])
    sweep = args.sweep
    n_runs = args.n_runs
    wandb_name = args.wandb_name

    main(sweep, n_runs, wandb_name)

