from PIL import Image
import torch
import torchvision.transforms as transforms
from Enums.classification_level import ClassificationLevel
from Enums.dataset_type import DatasetType
from Abstracts.base_dataset import _BaseDataset, _BaseDatasetItem
from Models.box import BoxInfo
from Utils.dataset import get_frame_img_path
import torch.nn.functional as F


class ImagePlayersDataset(_BaseDataset):
    def __init__(self, type: DatasetType):
        super().__init__(type)

    def get_flatten(self) -> list[list['ImagePlayersDatasetItem']]:
        dataset: list[list[ImagePlayersDatasetItem]] = []
        for _, v in self._videos_annotations.items():
            for __, c in v.get_all_clips_annotations():
                items:  list[ImagePlayersDatasetItem] = []
                for frame_ID, boxes in c.get_within_range_frame_boxes():
                    if len(boxes) > 12 or (len(boxes) < 12 and self.get_bl_cf().dataset.filter_missing_players_boxes_frames):
                        continue
                    items += [ImagePlayersDatasetItem(
                        video=v.video,
                        clip=c.clip,
                        frame=frame_ID,
                        img_path=get_frame_img_path(v.video, c.clip, frame_ID),
                        boxes=boxes,
                        label=c.get_category(),
                    )]
                if self.get_bl_cf().is_temporal:
                    if len(items) == self.get_bl_cf().dataset.get_seq_len():
                        dataset.append(items)
                else:
                    dataset.extend(map(lambda x: [x], items))
        return dataset

    def __getitem__(self, index) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        items: list[ImagePlayersDatasetItem] = self._flatten_dataset[index]

        img_players: list[list[tuple[torch.Tensor, str]]] = []

        box_orders: dict = None
        if self.get_bl_cf().dataset.filter_missing_players_boxes_frames:
            item = items[0]
            box_orders = {id: None for id in range(0, 12)}
            orders = sorted(map(lambda b: (b.box[0], b.player_ID), item.boxes))
            for i, o in enumerate(orders):
                box_orders[o[1]] = i

        for item in items:
            try:
                if box_orders:
                    _boxes: list[tuple] = [None for _ in range(12)]
                    for b in item.boxes:
                        _boxes[box_orders[b.player_ID]] = (
                            b.player_ID, b.box, b.category
                        )
                else:
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
        label_tensors: list[torch.Tensor] = [None for _ in img_players[0]]

        for i in range(len(img_players)):
            for j in range(len(img_players[i])):
                if self.has_bl_cf():
                    transformed = self.get_bl_cf(
                    ).dataset.preprocess.get_transforms()(img_players[i][j][0])
                else:
                    transformed = transforms.ToTensor()(img_players[i][j][0])

                img_players_tensors[i][j] = transformed

                if not label_tensors[j]:
                    label = torch.Tensor(
                        [self.get_cf().dataset.get_encoded_category(
                            ClassificationLevel.PLAYER, img_players[i][j][1]
                        )]
                    ).to(torch.long)
                    label_tensors[j] = label

            players_tensors = torch.stack(img_players_tensors[i])
            players_count = players_tensors.shape[0]
            if players_count < 12 and not self.get_bl_cf().dataset.filter_missing_players_boxes_frames:
                players_tensors = F.pad(
                    players_tensors,
                    (0, 0, 0, 0, 0, 0, 0, 12-players_count)
                )
            img_players_tensors[i] = players_tensors

        label_tensors = torch.stack(label_tensors)
        img_label = torch.Tensor(
            [self.get_cf().dataset.get_encoded_category(
                ClassificationLevel.IMAGE, item.label
            )]
        ).to(torch.long)

        label = (label_tensors,
                 img_label[0]) if self.get_bl_cf().is_joint else img_label[0]

        return torch.stack(img_players_tensors), label


class ImagePlayersDatasetItem(_BaseDatasetItem):
    def __init__(self, video: int, clip: int, frame: int, img_path: str, boxes: list[BoxInfo], label: str):
        super().__init__(video=video, clip=clip, frame=frame,  img_path=img_path)
        self.boxes = boxes
        self.label = label

    def to_dict(self):
        return dict([*super().to_dict().items(), ('boxes', self.boxes)])
