from .resnet50 import Resnet50
from .unet import Unet
from .customresnet import CustomResnet
# from . import *


def get_model(config):
    model_name = config['model_name']
    settings = config['settings']

    try:
        model = globals()[model_name](settings)
    except KeyError:
        raise f"Model with name {model_name} not found"

    return model
