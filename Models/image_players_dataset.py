from PIL import Image
import torch
import torchvision.transforms as transforms
from Enums.classification_level import ClassificationLevel
from Enums.dataset_type import DatasetType
from Abstracts.base_dataset import _BaseDataset, _BaseDatasetItem
from Models.box import BoxInfo
from Utils.dataset import get_frame_img_path


class ImagePlayersDataset(_BaseDataset):
    def __init__(self, type: DatasetType):
        super().__init__(type)

    def get_flatten(self) -> list[list['ImagePlayersDatasetItem']]:
        dataset: list[list[ImagePlayersDatasetItem]] = []
        for _, v in self._videos_annotations.items():
            for __, c in v.get_all_clips_annotations():
                items:  list[ImagePlayersDatasetItem] = []
                for frame_ID, boxes in c.get_within_range_frame_boxes():
                    if len(boxes) != 12:
                        continue
                    items += [ImagePlayersDatasetItem(
                        video=v.video,
                        clip=c.clip,
                        frame=frame_ID,
                        img_path=get_frame_img_path(v.video, c.clip, frame_ID),
                        boxes=boxes,
                        label=c.get_category(),
                    )]
                if len(items) == self.get_bl_cf().dataset.get_seq_len():
                    dataset.append(items)
        return dataset

    def __getitem__(self, index) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        items: list[ImagePlayersDatasetItem] = self._flatten_dataset[index]

        img_players: list[list[tuple[torch.Tensor, str]]] = []
        for item in items:
            try:
                _boxes = sorted(
                    map(lambda b: (b.player_ID, b.box, b.category), item.boxes)
                )
                img = Image.open(item.img_path).convert('RGB')
                imgs: list[torch.Tensor] = []
                for box in _boxes:
                    imgs += [(img.crop(box[1]), box[2])]
                img_players += [imgs]
            except FileNotFoundError:
                raise FileNotFoundError(f"Image not found at {item.img_path}")

        img_players_tensors = [[torch.Tensor() for _ in img_players[0]]
                               for __ in img_players
                               ]
        label_tensors = [torch.Tensor() for _ in img_players[0]]

        for i in range(len(img_players)):
            for j in range(len(img_players[i])):
                if self.has_bl_cf():
                    transformed = self.get_bl_cf().dataset.preprocess.get_transforms(
                        ClassificationLevel.PLAYER, self._type
                    )(img_players[i][j][0])
                else:
                    transformed = transforms.ToTensor()(img_players[i][j][0])

                label = torch.Tensor(
                    [self.get_cf().dataset.get_encoded_category(
                        ClassificationLevel.PLAYER, img_players[i][j][1]
                    )]
                ).to(torch.long)

                img_players_tensors[i][j] = transformed
                if i == self.get_bl_cf().dataset.past_frames_count or (i == len(img_players) - 1 and not len(label_tensors)):
                    label_tensors[j] = label

            img_players_tensors[i] = torch.stack(img_players_tensors[i])

        label_tensors = torch.stack(label_tensors)
        img_label = torch.Tensor(
            [self.get_cf().dataset.get_encoded_category(
                ClassificationLevel.IMAGE, item.label
            )]
        ).to(torch.long)

        return torch.stack(img_players_tensors), (label_tensors, img_label[0])


class ImagePlayersDatasetItem(_BaseDatasetItem):
    def __init__(self, video: int, clip: int, frame: int, img_path: str, boxes: list[BoxInfo], label: str):
        super().__init__(video=video, clip=clip, frame=frame,  img_path=img_path)
        self.boxes = boxes
        self.label = label

    def to_dict(self):
        return dict([*super().to_dict().items(), ('boxes', self.boxes)])
