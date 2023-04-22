from yaml.loader import SafeLoader
import yaml


def load_yaml_config(path):
    with open(path) as f:
        data = yaml.load(f, Loader=SafeLoader)
    return data
