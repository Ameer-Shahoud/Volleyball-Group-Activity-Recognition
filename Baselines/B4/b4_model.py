import torch
from Baselines.B1.b1_model import B1Model
from Abstracts.base_model import _BaseModel
from Enums.classification_level import ClassificationLevel
from Modules.lstm_head import LSTMHead


class B4Model(_BaseModel):
    def __init__(self, pretrained_img_model_path: str):
        super().__init__()

        self.pretrained_img_model: B1Model = _BaseModel.load_model(
            model_path=pretrained_img_model_path,
        )

        for p in self.pretrained_img_model.parameters():
            p.requires_grad = False

        self.lstm_head = LSTMHead(
            num_classes=len(self.get_cf().dataset.get_categories(
                ClassificationLevel.IMAGE)
            )
        )

    def forward(self, x: torch.Tensor):
        features = self.pretrained_img_model(x, return_features=True)
        return self.lstm_head(features)[0]

    def write_graph_to_tensorboard(self):
        self.get_bl_cf().writer.add_graph(
            self,
            input_to_model=torch.randn(
                self.get_bl_cf().training.batch_size,
                self.get_bl_cf().dataset.get_seq_len(),
                3,
                224,
                224
            )
        )
