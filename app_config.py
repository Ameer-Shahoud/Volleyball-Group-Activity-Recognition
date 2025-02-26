import json
import random
import numpy as np
import torch
from Enums.classification_level import ClassificationLevel
from Enums.dataset_type import DatasetType

# Global configuration object
_config: _Config = None

def init_config(path: str):
    """ Initializes the global configuration from a JSON file."""
    global _config
    _config = _Config(path)


def get_config():
    """Retrieves the global configuration object."""
    return _config


def is_available():
    """Checks if the global configuration is available."""
    return _config != None


class _Config:
    """
    Class for managing application configuration settings.

    Attributes:
        json (dict): Parsed JSON data from the configuration file.
        output_dir (str): Directory for storing output files.
        is_notebook (bool): Flag to indicate if running in a notebook environment.
        dataset (_DatasetConfig): Configuration for dataset settings.
    """
    def __init__(self, path):
        """Initializes the _Config object by loading the JSON configuration file."""
        self.json: dict = json.load(open(path, 'r'))
        self.output_dir: str = self.json.get('output_dir')
        self.is_notebook: bool = self.json.get('is_notebook')
        self.dataset: _DatasetConfig = _DatasetConfig(
            self.json.get('dataset'), self.output_dir)
        #Seed setting for reproducibility
        self.__seed()

    def __seed(self):
        """Sets the random seed for reproducibility in PyTorch, NumPy, and Python's random module."""
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False




class _DatasetConfig:
    """
    Class for managing dataset configuration settings.

    Attributes:
        root_dir (str): Root directory of the dataset.
        videos_dir (str): Directory containing video files.
        tracking_boxes_annotation_dir (str): Directory containing tracking boxes annotations.
        __categories (dict): Categories for image and player-level classification.
        __videos (dict): Video IDs for training, validation, and testing.
    """
    def __init__(self, dataset_json: dict, output):
        """Initializes the _DatasetConfig object by loading dataset settings."""
        self.root_dir: str = dataset_json.get('root_dir')
        self.videos_dir: str = f"{self.root_dir}/{dataset_json.get('videos_dir')}"
        self.tracking_boxes_annotation_dir: str = f"{self.root_dir}/{dataset_json.get('tracking_boxes_annotation_dir')}"
        self.__categories: dict[str, list[str]
                                ] = dataset_json.get('categories')
        self.__videos: dict[str, list[str]] = dataset_json.get('videos')

    def get_categories(self, level: ClassificationLevel):
        """
        Retrieves categories for the specified classification level.

        Args:
            level (ClassificationLevel): Classification level (IMAGE or PLAYER).

        Returns:
            list[str]: List of categories for the classification level.
        """
        return self.__categories.get(level.value)

    def get_videos(self, type: DatasetType):
        """
        Retrieves video IDs for the specified dataset type.

        Args:
            type (DatasetType): Type of dataset (TRAIN, VAL, TEST).

        Returns:
            list[str]: List of video IDs for the specified dataset type.
        """
        return self.__videos.get(type.value)
