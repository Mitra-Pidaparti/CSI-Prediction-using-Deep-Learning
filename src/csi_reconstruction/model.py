"""CNN architecture extracted from the original research notebooks."""

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, compression_ratio: int, height: int = 32, width: int = 32):
        super().__init__()
        if compression_ratio < 1 or 2 * height * width % compression_ratio:
            raise ValueError("compression_ratio must divide 2 * height * width")
        self.compressed_dim = 2 * height * width // compression_ratio
        self.conv = nn.Conv2d(2, 2, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(2 * height * width, self.compressed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.dense(self.flatten(x))


class Decoder(nn.Module):
    def __init__(self, compressed_dim: int, height: int = 32, width: int = 32, sigmoid: bool = True):
        super().__init__()
        self.height, self.width = height, width
        self.dense = nn.Linear(compressed_dim, 2 * height * width)
        self.conv1 = nn.Conv2d(2, 4, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu1 = nn.ReLU()
        self.conv2_upper = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.bn2_upper = nn.BatchNorm2d(4)
        self.conv2_upper2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.bn2_upper2 = nn.BatchNorm2d(8)
        self.conv2_lower = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.bn2_lower = nn.BatchNorm2d(4)
        self.conv2_lower2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.bn2_lower2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(16, 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(4)
        self.conv4 = nn.Conv2d(4, 2, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(2)
        self.reconstruction = nn.Conv2d(2, 2, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid() if sigmoid else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x).view(-1, 2, self.height, self.width)
        x = self.relu1(self.bn1(self.conv1(x)))
        upper = self.bn2_upper(self.conv2_upper(x))
        upper = self.bn2_upper2(self.conv2_upper2(upper))
        lower = self.bn2_lower(self.conv2_lower(x))
        lower = self.bn2_lower2(self.conv2_lower2(lower))
        x = self.bn3(self.conv3(torch.cat((upper, lower), dim=1)))
        x = self.bn4(self.conv4(x))
        x = self.reconstruction(x)
        # The notebook adds this tensor to itself. Keep that behavior for comparison.
        return self.sigmoid(x + x)


class LightweightCNN(nn.Module):
    def __init__(self, compression_ratio: int, height: int = 32, width: int = 32, sigmoid: bool = True):
        super().__init__()
        self.encoder = Encoder(compression_ratio, height, width)
        self.decoder = Decoder(self.encoder.compressed_dim, height, width, sigmoid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
