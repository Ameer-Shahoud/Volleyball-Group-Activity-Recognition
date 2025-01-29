import json


def init_config(path: str):
    global __config
    __config = _Config(path)


def get_config():
    return __config


class _Config:
    def __init__(self, path):
        self.json = json.load(open(path, 'r'))
        self.dataset_path: str = self.json.get('dataset_path')


__config: _Config = None
