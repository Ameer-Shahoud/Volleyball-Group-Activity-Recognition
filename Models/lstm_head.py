from torch import nn


class LSTMHead(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, num_classes: int = None):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return lstm_out, self.classifier(lstm_out[:, -1, :])
