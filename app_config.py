import json


def init_config(path: str):
    global _config
    _config = _Config(path)


def get_config():
    return _config


class _Config:
    def __init__(self, path):
        self.json = json.load(open(path, 'r'))
        self.output_path: str = self.json.get('output_path')
        self.is_notebook: bool = self.json.get('is_notebook')
        self.dataset: _DatasetConfig = _DatasetConfig(self.json.get('dataset'), self.output_path)


_config: _Config = None


class _DatasetConfig:
    def __init__(self, dataset_json, output):
        self.root: str = dataset_json.get('root')
        self.videos_path: str = f"{self.root}/{dataset_json.get('videos_path')}"
        self.tracking_boxes_annotation_path: str = f"{self.root}/{dataset_json.get('tracking_boxes_annotation_path')}"
        self.pkl_path: str = f"{output}/{dataset_json.get('pkl_path')}"
