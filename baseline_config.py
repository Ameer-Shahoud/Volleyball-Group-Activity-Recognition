import json
import os
from Enums.classification_level import ClassificationLevel
from Enums.dataset_type import DatasetType
import app_config as cf
import torchvision.transforms as transforms

# Global baseline configuration object
_bl_config: '_BLConfig' = None


def init_bl_config(path: str):
    """Initializes the baseline configuration from a JSON file."""
    global _bl_config
    _bl_config = _BLConfig(path)


def get_bl_config():
    """Retrieves the global baseline configuration object."""
    return _bl_config


def is_available():
    return _bl_config != None


class _BLConfig:
    """
    Class for managing baseline configuration settings.

    Attributes:
        json (dict): Parsed JSON data from the baseline configuration file.
        id (str): Identifier for the baseline experiment.
        output_dir (str): Output directory for storing experiment results.
        dataset (_DatasetConfig): Configuration for dataset preprocessing.
        training (_TrainingConfig): Configuration for training settings.
    """

    def __init__(self, path):
        """Initializes the _BLConfig object by loading the baseline configuration."""
        self.json: dict = json.load(open(path, 'r'))
        self.id: str = self.json.get('id')
        self.output_dir: str = os.path.join(
            cf.get_config().output_dir, 'Output', self.json.get('output_dir'), self.id
        )
        self.dataset: _DatasetConfig = _DatasetConfig(self.json.get('dataset'))
        self.training: _TrainingConfig = _TrainingConfig(
            self.json.get('training'))

    def create_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def clear_output_dir(self):
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                file_path = os.path.join(self.output_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)


class _DatasetConfig:
    """
    Class for managing dataset preprocessing settings.

    Attributes:
        past_frames_count (int): Number of past frames for temporal modeling.
        post_frames_count (int): Number of post frames for temporal modeling.
        preprocess (_PreprocessConfig): Preprocessing transformations.
    """

    def __init__(self, dataset_json: dict):
        """Initializes the _DatasetConfig object for preprocessing settings."""
        self.past_frames_count: int = dataset_json.get('past_frames_count')
        self.post_frames_count: int = dataset_json.get('post_frames_count')
        self.preprocess: _PreprocessConfig = _PreprocessConfig(
            dataset_json.get('preprocess'))


class _PreprocessConfig:
    def __init__(self, preprocess_json: dict):
        self.__transforms: dict = preprocess_json.get('transforms')

    def get_transforms(self, level: ClassificationLevel, type: DatasetType):
        return transforms.Compose(
            list(map(lambda t: self.__get_transform__(t),
                 self.__transforms.get(level.value).get(type.value)))
        )

    def __get_transform__(self, transform: dict):
        args: dict = transform.get('args')
        match transform.get('type'):
            case "resize":
                return transforms.Resize((args.get('width'), args.get('height')))
            case "random_horizontal_flip":
                return transforms.RandomHorizontalFlip()
            case "center_crop":
                return transforms.CenterCrop((args.get('width'), args.get('height')))
            case "to_tensor":
                return transforms.ToTensor()
            case "normalize":
                return transforms.Normalize(mean=args.get('mean'), std=args.get('std'))


class _TrainingConfig:
    def __init__(self, training_json: dict):
        self.epochs: int = training_json.get('epochs')
        self.batch_size: int = training_json.get('batch_size')
        self.learning_rate: int = training_json.get('learning_rate')
