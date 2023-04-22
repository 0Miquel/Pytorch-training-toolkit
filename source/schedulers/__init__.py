import torch.optim.lr_scheduler as lr_scheduler

import importlib
import pkgutil

# get a list of all submodules of the current package
package_name = __name__
package_path = __path__
module_names = [name for _, name, _ in pkgutil.walk_packages(package_path)]


def get_scheduler(config, optimizer, steps_per_epoch=None, n_epochs=None):
    scheduler_name = config["scheduler_name"]
    settings = config["settings"]
    if scheduler_name == "OneCycleLR":
        # Need steps per epoch and n_epochs for OneCycleLR
        settings["steps_per_epoch"] = steps_per_epoch
        settings["epochs"] = n_epochs

    if hasattr(lr_scheduler, scheduler_name):
        # get lr scheduler from torch.optim.lr_scheduler package
        scheduler = getattr(lr_scheduler, scheduler_name)(optimizer, **settings)
        return scheduler
    else:
        # get lr scheduler defined in this package
        for module_name in module_names:
            module = importlib.import_module(f'{package_name}.{module_name}')
            if hasattr(module, scheduler_name):
                scheduler = getattr(module, scheduler_name)(optimizer, **settings)
                return scheduler

    raise f"Scheduler with name {scheduler_name} not found"
