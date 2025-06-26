import json
import os
from Abstracts.with_update_state_mixin import WithUpdateFromDictMixin
from Types.metric_type import MetricType
from Types.mode_type import ModeType
from Utils.assign_not_none import assign_not_none
from Utils.logger import Logger
import config as cf
import torchvision.transforms as transforms
from torch.utils.tensorboard.writer import SummaryWriter

# Global baseline configuration object
_bl_config: '_BLConfig' = None


def init_bl_config(path: str, config_override_patch: dict = None):
    """Initializes the baseline configuration from a JSON file."""
    global _bl_config
    _bl_config = _BLConfig(path)
    if config_override_patch:
        _bl_config.patch_state(state=config_override_patch)


def get_bl_config():
    """Retrieves the global baseline configuration object."""
    return _bl_config


def is_available():
    return _bl_config != None


class _BLConfig(WithUpdateFromDictMixin):
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
        super().__init__(self.json)

        self.dataset: _DatasetConfig = _DatasetConfig(self.json.get('dataset'))
        self.training: _TrainingConfig = _TrainingConfig(
            self.json.get('training')
        )

        self.writer = SummaryWriter(
            os.path.join(self.output_dir, f"{self.id}_tensorboard")
        )
        self.logger = Logger(
            log_dir=self.output_dir,
            log_name=self.id,
            writer=self.writer
        )

    def update_from_dict(self, state: dict):
        self.id: str = assign_not_none(state.get('id'), self._getattr('id'))
        self.title: str = assign_not_none(
            state.get('title'), self._getattr('title')
        )
        self.output_dir: str = assign_not_none(
            os.path.join(
                cf.get_config().output_dir,
                state.get('output_dir'),
                self.id
            ) if state.get('output_dir')
            else None,
            self._getattr('output_dir')
        )
        self.is_temporal: bool = assign_not_none(
            state.get('is_temporal'), self._getattr('is_temporal')
        )
        self.is_joint: bool = assign_not_none(
            state.get('is_joint'), self._getattr('is_joint')
        )

    def create_baseline_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def clear_output_dir(self):
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                file_path = os.path.join(self.output_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)


class _DatasetConfig(WithUpdateFromDictMixin):
    """
    Class for managing dataset preprocessing settings.

    Attributes:
        past_frames_count (int): Number of past frames for temporal modeling.
        post_frames_count (int): Number of post frames for temporal modeling.
        preprocess (_PreprocessConfig): Preprocessing transformations.
    """

    def __init__(self, dataset_json: dict):
        """Initializes the _DatasetConfig object for preprocessing settings."""
        super().__init__(dataset_json)
        self.preprocess: _PreprocessConfig = _PreprocessConfig(
            dataset_json.get('preprocess'))

    def update_from_dict(self, state: dict):
        self.filter_missing_players_boxes_frames: bool = assign_not_none(
            state.get('filter_missing_players_boxes_frames'),
            self._getattr('filter_missing_players_boxes_frames')
        )
        self.past_frames_count: int = assign_not_none(
            state.get('past_frames_count'),
            self._getattr('past_frames_count'),
            0
        )
        self.post_frames_count: int = assign_not_none(
            state.get('post_frames_count'),
            self._getattr('post_frames_count'),
            0
        )

    def get_seq_len(self):
        return self.past_frames_count + self.post_frames_count + 1


class _PreprocessConfig(WithUpdateFromDictMixin):
    def __init__(self, preprocess_json: dict):
        super().__init__(preprocess_json)

    def update_from_dict(self, state: dict):
        self.transforms: list[dict] = assign_not_none(
            state.get('transforms'), self._getattr('transforms'), []
        )

    def get_transforms(self):
        return transforms.Compose(
            list(map(lambda t: self.__get_transform__(t), self.transforms))
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


class _TrainingConfig(WithUpdateFromDictMixin):
    def __init__(self, training_json: dict):
        super().__init__(training_json)
        self.early_stopping: _EarlyStoppingConfig = _EarlyStoppingConfig(
            training_json.get('early_stopping')
        )
        self.scheduler: _SchedulerConfig = _SchedulerConfig(
            training_json.get('scheduler')
        )

    def update_from_dict(self, state: dict):
        self.epochs: int = assign_not_none(
            state.get('epochs'), self._getattr('epochs')
        )
        self.batch_size: int = assign_not_none(
            state.get('batch_size'), self._getattr('batch_size')
        )
        self.learning_rate: float = assign_not_none(
            state.get('learning_rate'), self._getattr('learning_rate')
        )


class _EarlyStoppingConfig(WithUpdateFromDictMixin):
    def __init__(self, early_stopping_json: dict):
        super().__init__(early_stopping_json)

    def update_from_dict(self, state: dict):
        self.metric: MetricType = assign_not_none(
            state.get('metric'), self._getattr('metric')
        )
        self.patience: int = assign_not_none(
            state.get('patience'), self._getattr('patience')
        )
        self.delta: float = assign_not_none(
            state.get('delta'), self._getattr('delta')
        )
        self.mode: ModeType = assign_not_none(
            state.get('mode'), self._getattr('mode')
        )


class _SchedulerConfig(WithUpdateFromDictMixin):
    def __init__(self, scheduler_json: dict):
        super().__init__(scheduler_json)

    def update_from_dict(self, state: dict):
        self.patience: int = assign_not_none(
            state.get('patience'), self._getattr('patience')
        )
        self.factor: float = assign_not_none(
            state.get('factor'), self._getattr('factor')
        )
        self.mode: ModeType = assign_not_none(
            state.get('mode'), self._getattr('mode')
        )
