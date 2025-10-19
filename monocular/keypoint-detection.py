import torch
import os
import json
import numpy as np

from torch import nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

PRINT_MESSAGE_INTERVAL = 10
LEARNING_RATE = 5e-2
BATCH_SIZE = 64
EPOCHS = 10


class ConeDataset(Dataset):
    def __init__(self, folder):
        super().__init__()

        self.FILE_PATH = os.path.join("data", folder)

        with open(os.path.join(self.FILE_PATH, "annotations.coco.json")) as f:
            self.data = json.load(f)

        self.images = [] * len(self.data["images"])
        self.keypoints = [] * len(self.data["images"])

        # Note: this for loop assumes the image ids are 0-indexed without gaps, which seems to be true
        for img in self.data["images"]:
            img_path = os.path.join(self.FILE_PATH, img["file_name"])

            with Image.open(img_path) as i:
                self.images[img["id"]] = transforms.ToTensor(i)

        for kps in self.data["annotations"]:
            x = kps["keypoints"]
            if (
                x[2] != 2
                or x[5] != 2
                or x[8] != 2
                or x[11] != 2
                or x[14] != 2
                or x[17] != 2
                or x[20] != 2
                or x[23] != 2
            ):
                raise Exception("Issues encountered with processing keypoint.")

            # 6 7
            self.keypoints[kps["image_id"]] = torch.tensor(
                [
                    x[0],
                    x[1],
                    x[3],
                    x[4],
                    x[6],
                    x[7],
                    x[9],
                    x[10],
                    x[12],
                    x[13],
                    x[15],
                    x[16],
                    x[18],
                    x[19],
                    x[21],
                    x[22],
                ]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.keypoints[idx]


class BottleneckResidual(nn.Module):
    def __init__(self, c, g, stride=1, conv_1x1=False):
        super().__init__()

        self.bottleneck = nn.Sequential(
            nn.LazyConv2d(c, kernel_size=1, groups=g),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(c, kernel_size=3, stride=stride, groups=g, padding=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(c, kernel_size=1, groups=g),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
        )

        self.conv_1x1 = conv_1x1

        if self.conv_1x1:
            self.identity_conv = nn.LazyConv2d(c, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bottleneck(x)

        if self.conv_1x1:
            x = self.identity_conv(x)

        print(y.shape, x.shape)
        return F.relu(y + x)


class KeypointDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # Initial Block
            nn.LazyConv2d(32, kernel_size=7, stride=5, padding=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(64, kernel_size=3, stride=2, padding=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            # Bottleneck Blocks
            self.block(64, 2),
            self.block(128, 2),
            self.block(256, 2),
            self.block(512, 2),
            # Fully connected
            nn.Flatten(),
            # nn.LazyLinear(512),
            # nn.ReLU(),
            # nn.LazyLinear(256),
            # nn.ReLU(),
            nn.LazyLinear(16),
            nn.ReLU(),
        )

    def block(self, c: int, num_blocks: int) -> nn.Module:
        layers = []

        for i in range(num_blocks):
            layers.append(BottleneckResidual(c, 4, conv_1x1=(i == 0)))

        return nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        return self.model(x)


class KeypointLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def cross_ratio(p1, p2, p3, p4):
        return (torch.hypot(p1, p3) / torch.hypot(p1, p4)) / (
            torch.hypot(p2, p3) / torch.hypot(p2, p4)
        )

    def forward(self, predicted, target):
        gamma = 0.0001
        return (
            torch.sum((predicted - target) ** 2)
            + gamma * KeypointLoss.cross_ratio(*predicted[:4])
            + gamma * KeypointLoss.cross_ratio(*predicted[4:])
        )


def train_epoch(dl, model, loss, optimizer):
    model.train()

    for i, (X, y) in enumerate(dl):
        fwd = model(X)
        loss = loss(fwd, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % PRINT_MESSAGE_INTERVAL == 0:
            loss, current = loss.item(), i * BATCH_SIZE + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(dl.dataset):>5d}]")


def test_epoch(dl, model, loss):
    model.eval()

    # TODO fully implement
