from tofu import get_trainer
import hydra


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    trainer = get_trainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
