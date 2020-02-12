import yaml
from attrdict import AttrDict

def get_cfg():
    with open('config/config.yaml') as file:
        config = AttrDict(yaml.safe_load(file))
    return config
