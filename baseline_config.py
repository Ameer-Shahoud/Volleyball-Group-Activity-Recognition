import json
from Enums.classification_level import ClassificationLevel
from Enums.dataset_type import DatasetType
import app_config as cf
import torchvision.transforms as transforms


def init_bl_config(path: str):
    global _bl_config
    _bl_config = _BLConfig(path)


def get_bl_config():
    return _bl_config


def is_available():
    return _bl_config != None


class _BLConfig:
    def __init__(self, path):
        self.json: dict = json.load(open(path, 'r'))
        self.id: str = self.json.get('id')
        self.output_dir: str = f"{cf.get_config().output_dir}/{self.json.get('output_dir')}/{self.id}"
        self.dataset: _DatasetConfig = _DatasetConfig(self.json.get('dataset'))
        self.training: _TrainingConfig = _TrainingConfig(
            self.json.get('training'))


_bl_config: _BLConfig = None


class _DatasetConfig:
    def __init__(self, dataset_json: dict):
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
