import torch.optim.lr_scheduler as lr_scheduler

import importlib
import pkgutil

# get a list of all submodules of the current package
package_name = __name__
package_path = __path__
module_names = [name for _, name, _ in pkgutil.walk_packages(package_path)]


def get_scheduler(cfg, optimizer, total_steps):
    scheduler_name = cfg["scheduler_name"]
    framework = cfg['framework'] if 'framework' in cfg.keys() else None
    settings = cfg['settings'] if 'settings' in cfg.keys() else {}

    if scheduler_name == "OneCycleLR":
        settings["total_steps"] = total_steps

    if framework == 'torch':
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

    raise AttributeError(f"Scheduler with name {scheduler_name} not found")
