"""Model definitions — one class per architecture family (same I/O contract).
Input convention: x shape (B, 128, 9). CNN transposes internally to (B, 9, 128).
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Shreyas Bibhuty — baseline. Flattened window -> dense layers (no temporal modelling)."""
    def __init__(self, seq_len=128, n_channels=9, n_classes=6, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * n_channels, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    """Ansh Aaditya Singh — temporal convolutions over sensor channels."""
    def __init__(self, n_channels=9, n_classes=6, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),          # 128 -> 64
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),          # 64 -> 32
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, 9, 128)
        return self.classifier(self.features(x))


class LSTMClassifier(nn.Module):
    """Aryan Ranpura — sequential modelling of the 128-step window."""
    def __init__(self, n_channels=9, hidden=128, layers=2, n_classes=6, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
