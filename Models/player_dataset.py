from PIL import Image
import torch
import torchvision.transforms as transforms
from Enums.classification_level import ClassificationLevel
from Enums.dataset_type import DatasetType
from Models.base_dataset import _BaseDataset, _BaseDatasetItem
from Models.box import BoxInfo
from Utils.dataset import get_frame_img_path


class PlayerDataset(_BaseDataset):
    def __init__(self, type: DatasetType):
        super().__init__(type)

    def get_flatten(self) -> list[list['PlayerDatasetItem']]:
        dataset: list[list[PlayerDatasetItem]] = []
        for _, v in self._videos_annotations.items():
            for __, c in v.get_all_clips_annotations():
                items: dict[int, list[PlayerDatasetItem]] = {
                    i: [] for i in range(12)
                }
                for frame_ID, boxes in c.get_within_range_frame_boxes():
                    for box in boxes:
                        items[box.player_ID] += [PlayerDatasetItem(
                            video=v.video,
                            clip=c.clip,
                            frame=frame_ID,
                            img_path=get_frame_img_path(
                                v.video, c.clip, frame_ID),
                            box=box
                        )]
                for item in items.values():
                    dataset.append(item)
        return dataset

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        items: list[PlayerDatasetItem] = self._flatten_dataset[index]

        player_imgs: list[torch.Tensor] = []
        for item in items:
            try:
                player_imgs += [Image.open(item.img_path).convert(
                    'RGB').crop(item.box.box)]
            except FileNotFoundError:
                raise FileNotFoundError(f"Image not found at {item.img_path}")

        for i in range(len(player_imgs)):
            if self.has_bl_cf():
                player_imgs[i] = self.get_bl_cf().dataset.preprocess.get_transforms(
                    ClassificationLevel.PLAYER, self._type
                )(player_imgs[i])
            else:
                player_imgs[i] = transforms.ToTensor()(player_imgs[i])

        y_label = torch.Tensor(
            [self.get_cf().dataset.get_encoded_category(
                ClassificationLevel.PLAYER, item.box.category
            )]
        ).to(torch.long)
        return torch.stack(player_imgs), y_label[0]


class PlayerDatasetItem(_BaseDatasetItem):
    def __init__(self, video: int, clip: int, frame: int, img_path: str, box: BoxInfo):
        super().__init__(video=video, clip=clip, frame=frame, img_path=img_path)
        self.box = box

    def to_dict(self):
        return dict([*super().to_dict().items(), ('box', self.box)])
