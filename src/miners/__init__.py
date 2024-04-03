import pytorch_metric_learning.miners as pml_miners

import importlib
import pkgutil

# get a list of all submodules of the current package
package_name = __name__
package_path = __path__
module_names = [name for _, name, _ in pkgutil.walk_packages(package_path)]


def get_miner(cfg):
    miner_name = cfg['miner_name']
    framework = cfg['framework'] if 'framework' in cfg.keys() else None
    settings = cfg["settings"] if "settings" in cfg.keys() else {}

    if framework == 'pytorch_metric_learning':
        # get miner from pytorch_metric_learning.miners package
        miner = getattr(pml_miners, miner_name)(**settings)
        return miner
    else:
        # get miner defined in this package
        for module_name in module_names:
            module = importlib.import_module(f'{package_name}.{module_name}')
            if hasattr(module, miner_name):
                miner = getattr(module, miner_name)(**settings)
                return miner

    raise AttributeError(f"Miner with name {miner_name} not found")
