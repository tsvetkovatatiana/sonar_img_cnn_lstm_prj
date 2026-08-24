import time
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_ch, out_ch, kernel, 1, padding, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        self.residual = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )

    def forward(self, x):
        out = self.block(x)
        out = out + self.residual(x)
        return torch.relu(out)


class SonarCNN(nn.Module):

    def __init__(self, out_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(1, 16, 3, 2, 1),
            ResidualBlock(16, 32, 3, 2, 1),
            ResidualBlock(32, 64, 3, 2, 1),
            ResidualBlock(64, 64, 3, 1, 1),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        x = self.proj(x)
        return x  # (B, out_dim)


class CNN_LSTM(nn.Module):

    def __init__(self, cnn_out_dim=64, lstm_hidden=64, lstm_layers=1, fc_hidden=34):
        super().__init__()

        self.cnn = SonarCNN(out_dim=cnn_out_dim)

        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(lstm_hidden)

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        x = self.cnn(x)  # (B*T, cnn_out_dim)
        x = x.view(B, T, -1)  # (B, T, cnn_out_dim)

        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        h_last = out[:, -1]

        out = self.head(h_last).squeeze(-1)
        return out

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params    : {total:,}")
        print(f"Trainable params: {trainable:,}")


if __name__ == "__main__":
    print("--------- MODEL SUMMARY --------- ")
    model.count_params()
    print(model)
