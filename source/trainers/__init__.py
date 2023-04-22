import importlib
import pkgutil
# from .base_trainer import BaseTrainer

# get a list of all submodules of the current package
package_name = __name__
package_path = __path__
module_names = [name for _, name, _ in pkgutil.walk_packages(package_path)]


def train(wandb_name=None, cfg=None):
    trainer_name = cfg["trainer"]["trainer_name"]

    trainer = None
    for module_name in module_names:
        module = importlib.import_module(f'{package_name}.{module_name}')
        if hasattr(module, trainer_name):
            trainer = getattr(module, trainer_name)(cfg, wandb_name)
            break
            # return trainer
    if trainer is None:
        raise f"Trainer with name {trainer_name} not found"
    else:
        trainer.fit()
