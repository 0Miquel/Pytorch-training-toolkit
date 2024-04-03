import importlib
import pkgutil

# get a list of all submodules of the current package
package_name = __name__
package_path = __path__
module_names = [name for _, name, _ in pkgutil.walk_packages(package_path)]


def get_trainer(cfg=None):
    trainer_name = cfg["trainer"]["trainer_name"]

    for module_name in module_names:
        module = importlib.import_module(f'{package_name}.{module_name}')
        if hasattr(module, trainer_name):
            trainer = getattr(module, trainer_name)(cfg)
            return trainer

    raise AttributeError(f"Trainer with name {trainer_name} not found")
