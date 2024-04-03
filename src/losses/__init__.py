import torch.nn as nn
import segmentation_models_pytorch.losses as smp_losses
import pytorch_metric_learning.losses as pml_losses

import importlib
import pkgutil

# get a list of all submodules of the current package
package_name = __name__
package_path = __path__
module_names = [name for _, name, _ in pkgutil.walk_packages(package_path)]


def get_loss(cfg):
    loss_name = cfg['loss_name']
    framework = cfg['framework'] if 'framework' in cfg.keys() else None
    settings = cfg["settings"] if "settings" in cfg.keys() else {}

    if framework == 'torch':
        # get loss from torch.nn package
        loss = getattr(nn, loss_name)(**settings)
        return loss
    elif framework == 'segmentation_models_pytorch':
        # get loss from segmentation_models_pytorch.losses package
        loss = getattr(smp_losses, loss_name)(**settings)
        return loss
    elif framework == 'pytorch_metric_learning':
        # get loss from pytorch_metric_learning.losses package
        loss = getattr(pml_losses, loss_name)(**settings)
        return loss
    else:
        # get loss defined in this package
        for module_name in module_names:
            module = importlib.import_module(f'{package_name}.{module_name}')
            if hasattr(module, loss_name):
                loss = getattr(module, loss_name)(**settings)
                return loss

    raise AttributeError(f"Loss with name {loss_name} not found")
