import importlib
import pkgutil

# get a list of all submodules of the current package
package_name = __name__
package_path = __path__
module_names = [name for _, name, _ in pkgutil.walk_packages(package_path)]


def get_model(config):
    model_name = config['model_name']
    settings = config['settings']

    for module_name in module_names:
        # get model defined in this package
        module = importlib.import_module(f'{package_name}.{module_name}')
        if hasattr(module, model_name):
            model = getattr(module, model_name)(settings)
            return model

    raise f"Model with name {model_name} not found"
