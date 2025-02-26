import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from abc import ABC, abstractmethod
from Enums.classification_level import ClassificationLevel
from Enums.dataset_type import DatasetType
from Models.config_mixin import _ConfigMixin
from Models.video_annotations import VideoAnnotations
from Utils.dataset import get_frame_img_path


class _Dataset(Dataset, _ConfigMixin, ABC):
    """
    Abstract base class for creating custom datasets.
    It manages loading video annotations and preparing data for training and evaluation.

    Attributes:
        _level (ClassificationLevel): Classification level (IMAGE or PLAYER).
        _type (DatasetType): Type of dataset (TRAIN, VAL, TEST).
        _videos (list): List of video IDs.
        _videos_annotations (dict): Dictionary of video IDs and corresponding VideoAnnotations objects.
    """
    _level: ClassificationLevel = None

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

        for idx, video in enumerate(videos):
            print(f'{idx}/{len(videos)} - Processing Dir {video}')
            self._videos_annotations[video] = VideoAnnotations(video)

        self._encoder = LabelEncoder()
        self._encoder.fit(self.get_cf().dataset.get_categories(self._level))

    @abstractmethod
    def get_flatten(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    def get_video_annotations(self, video: int) -> VideoAnnotations:
        return self._videos_annotations[video]

    def get_all_videos_annotations(self):
        return self._videos_annotations.items()


class ImageDataset(_Dataset):
    """
    Custom dataset for image-level classification.
    It loads images, applies transformations, and encodes labels.

    Attributes:
        _level (ClassificationLevel): Classification level set to IMAGE.
        __flatten_dataset (list): Flattened list of ImageDatasetItem objects.
    """
    def __init__(self, type):
        """Initializes the ImageDataset by setting classification level to IMAGE."""
        self._level = ClassificationLevel.IMAGE
        super().__init__(type)
        self.__flatten_dataset = self.get_flatten()

    def get_flatten(self):
        """
        Flattens the dataset into a list of ImageDatasetItem objects.

        Returns:
            list[ImageDatasetItem]: Flattened list of dataset items.
        """
        dataset: list[ImageDatasetItem] = []
        for _, v in self._videos_annotations.items():
            for __, c in v.get_all_clips_annotations():
                for frame_ID, boxes in c.get_within_range_frame_boxes():
                    item = ImageDatasetItem(
                        video=v.video,
                        clip=c.clip,
                        frame=frame_ID,
                        label=c.get_category(),
                        img_path=get_frame_img_path(v.video, c.clip, frame_ID)
                    )
                    dataset.append(item)
        return dataset

    def __len__(self):
        return len(self.__flatten_dataset)

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset by index.

        Args:
            index (int): Index of the item.

        Returns:
            tuple: A tuple containing the transformed image tensor and its label.
        """
        item: ImageDatasetItem = self.__flatten_dataset[index]

        try:
            img = Image.open(item.img_path).convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found at {item.img_path}")

        if self.has_bl_cf():
            img = self.get_bl_cf().dataset.preprocess.get_transforms(
                self._level, self._type)(img)
        else:
            img = transforms.ToTensor()(img)

        y_label = torch.Tensor(
            self._encoder.transform([item.label])
        ).to(torch.long)

        return img, y_label[0]


class ImageDatasetItem:
    """
    Class to organize individual dataset items for image-level classification.

    Attributes:
        video (int): Video ID.
        clip (int): Clip ID within the video.
        frame (int): Frame ID within the clip.
        label (str): Action label for the image.
        img_path (str): Path to the image file.
    """
    def __init__(self, video: int, clip: int, frame: int, label: str, img_path: str):
        self.video = video
        self.clip = clip
        self.frame = frame
        self.label = label
        self.img_path = img_path

    def to_dict(self):
        """Converts the ImageDatasetItem object to a dictionary."""
        return {
            "video": self.video,
            "clip": self.clip,
            "frame": self.frame,
            "label": self.label,
            "img_path": self.img_path
        }


class PlayerDataset(_Dataset):
    """
    Placeholder class for player-level classification dataset.
    Inherits from _Dataset.
    """
    def __init__(self, type):
        """
        Initializes the PlayerDataset by setting classification level to PLAYER.

        Args:
            type (DatasetType): Type of dataset (TRAIN, VAL, TEST).
        """
        self._level = ClassificationLevel.PLAYER
        super().__init__(type)
        self.__flatten_dataset = self.get_flatten()


class PlayerDatasetItem:
    pass
