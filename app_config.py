import json
import random
import numpy as np
import torch
from Enums.classification_level import ClassificationLevel
from Enums.dataset_type import DatasetType


def init_config(path: str):
    global _config
    _config = _Config(path)


def get_config():
    return _config


def is_available():
    return _config != None


class _Config:
    def __init__(self, path):
        self.json: dict = json.load(open(path, 'r'))
        self.output_dir: str = self.json.get('output_dir')
        self.is_notebook: bool = self.json.get('is_notebook')
        self.dataset: _DatasetConfig = _DatasetConfig(
            self.json.get('dataset'), self.output_dir)

        self.__seed()

    def __seed(self):
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


_config: _Config = None


class _DatasetConfig:
    def __init__(self, dataset_json: dict, output):
        self.root_dir: str = dataset_json.get('root_dir')
        self.videos_dir: str = f"{self.root_dir}/{dataset_json.get('videos_dir')}"
        self.tracking_boxes_annotation_dir: str = f"{self.root_dir}/{dataset_json.get('tracking_boxes_annotation_dir')}"
        self.__categories: dict[str, list[str]
                                ] = dataset_json.get('categories')
        self.__videos: dict[str, list[str]] = dataset_json.get('videos')

    def get_categories(self, level: ClassificationLevel):
        return self.__categories.get(level.value)

    def get_videos(self, type: DatasetType):
        return self.__videos.get(type.value)
