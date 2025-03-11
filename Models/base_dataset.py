import os
import pickle
from typing import Any
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from Enums.dataset_type import DatasetType
from Models.config_mixin import _ConfigMixin
from Models.video_annotations import VideoAnnotations
from Utils.dataset import get_frame_img_path


class _BaseDataset(Dataset, _ConfigMixin, ABC):
    """
    Abstract base class for creating custom datasets.
    It manages loading video annotations and preparing data for training and evaluation.

    Attributes:
        _type (DatasetType): Type of dataset (TRAIN, VAL, TEST).
        _videos (list): List of video IDs.
        _videos_annotations (dict): Dictionary of video IDs and corresponding VideoAnnotations objects.
    """

    def __init__(self, type: DatasetType):
        """
        Initializes the dataset by loading video annotations and configurations.

        Args:
            type (DatasetType): Type of dataset (TRAIN, VAL, TEST).
        """
        self._type = type
        self._videos = self.get_cf().dataset.get_videos(type)
        self._videos_annotations: dict[int, VideoAnnotations] = {}

        __videos_path = self.get_cf().dataset.videos_dir

        videos = [dir for dir in os.listdir(__videos_path) if os.path.isdir(
            os.path.join(__videos_path, dir))]
        videos = sorted(
            map(int, [dir for dir in videos if dir in self._videos]))

        _pkl_path = self.get_cf().dataset.get_pkl_path(self._type)
        if os.path.exists(_pkl_path):
            with open(_pkl_path, "rb") as f:
                self._videos_annotations = pickle.load(f)
        else:
            for idx, video in enumerate(videos):
                print(f'{idx}/{len(videos)} - Processing Dir {video}')
                self._videos_annotations[video] = VideoAnnotations(video)
            with open(_pkl_path, "wb") as f:
                pickle.dump(self._videos_annotations, f)

        self._flatten_dataset = self.get_flatten()

    @abstractmethod
    def get_flatten(self) -> list[list['_BaseDatasetItem']]:
        pass

    def __len__(self):
        return len(self._flatten_dataset)

    @abstractmethod
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def get_video_annotations(self, video: int) -> VideoAnnotations:
        return self._videos_annotations[video]

    def get_all_videos_annotations(self):
        return self._videos_annotations.items()


class _BaseDatasetItem(_ConfigMixin, ABC):
    def __init__(self, video: int, clip: int, frame: int, img_path: str):
        self.video = video
        self.clip = clip
        self.frame = frame
        self.img_path = img_path

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video,
            "clip": self.clip,
            "frame": self.frame,
            "img_path": self.img_path,
        }
