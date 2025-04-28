from PIL import Image
from typing import Any
import torch
import torchvision.transforms as transforms
from Enums.classification_level import ClassificationLevel
from Enums.dataset_type import DatasetType
from Abstracts.base_dataset import _BaseDataset, _BaseDatasetItem
from Utils.dataset import get_frame_img_path


class ImageDataset(_BaseDataset):
    def __init__(self, type: DatasetType):
        super().__init__(type)

    def get_flatten(self) -> list[list['ImageDatasetItem']]:
        dataset: list[list[ImageDatasetItem]] = []
        for _, v in self._videos_annotations.items():
            for __, c in v.get_all_clips_annotations():
                items: list[ImageDatasetItem] = []
                for frame_ID, boxes in c.get_within_range_frame_boxes():
                    items += [ImageDatasetItem(
                        video=v.video,
                        clip=c.clip,
                        frame=frame_ID,
                        img_path=get_frame_img_path(v.video, c.clip, frame_ID),
                        label=c.get_category(),
                    )]
                dataset.append(items)
        return dataset

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        items: list[ImageDatasetItem] = self._flatten_dataset[index]

        imgs: list[torch.Tensor] = []
        for item in items:
            try:
                imgs += [Image.open(item.img_path).convert('RGB')]
            except FileNotFoundError:
                raise FileNotFoundError(f"Image not found at {item.img_path}")

        for i in range(len(imgs)):
            if self.has_bl_cf():
                imgs[i] = self.get_bl_cf().dataset.preprocess.get_transforms(
                    ClassificationLevel.IMAGE, self._type
                )(imgs[i])
            else:
                imgs[i] = transforms.ToTensor()(imgs[i])

        y_label = torch.Tensor(
            [self.get_cf().dataset.get_encoded_category(
                ClassificationLevel.IMAGE, item.label
            )]
        ).to(torch.long)
        return torch.stack(imgs), y_label[0]


class ImageDatasetItem(_BaseDatasetItem):
    def __init__(self, video: int, clip: int, frame: int, img_path: str, label: str):
        super().__init__(video=video, clip=clip, frame=frame, img_path=img_path)
        self.label = label

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict(dict([*super().to_dict().items(), ('label', self.label)]))
