import torch.nn as nn
import segmentation_models_pytorch.losses as smp_losses

import importlib
import pkgutil

# get a list of all submodules of the current package
package_name = __name__
package_path = __path__
module_names = [name for _, name, _ in pkgutil.walk_packages(package_path)]


def get_loss(config):
    loss_name = config['loss_name']

    if hasattr(nn, loss_name):
        # get loss from torch.nn package
        loss = getattr(nn, loss_name)()
        return loss
    elif hasattr(smp_losses, loss_name):
        # get loss from segmentation_models_pytorch.losses package
        loss = getattr(smp_losses, loss_name)()
        return loss
    else:
        # get loss defined in this package
        for module_name in module_names:
            module = importlib.import_module(f'{package_name}.{module_name}')
            if hasattr(module, loss_name):
                loss = getattr(module, loss_name)()
                return loss

    raise f"Loss with name {loss_name} not found"



# def get_loss(config):
#     loss_name = config['loss_name']
#
#     try:
#         # check if loss is found in torch.nn
#         loss = getattr(nn, loss_name)()
#     except AttributeError:
#         try:
#             # check if loss is found in segmentation_models_pytorch.losses
#             loss = getattr(smp_losses, loss_name)()
#         except AttributeError:
#             # check if loss is defined in one of the submodules of this package
#             try:
#                 loss = globals()[loss_name]()
#             except KeyError:
#                 raise f"Loss with name {loss_name} not found"
#
#     return loss
