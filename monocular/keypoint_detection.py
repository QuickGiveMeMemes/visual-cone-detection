import torch
import os
import json
import numpy as np

from torch import nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import random
import cv2


PRINT_MESSAGE_INTERVAL = 1
LEARNING_RATE = 0.00001
BATCH_SIZE = 64
EPOCHS = 10000
IMAGE_SIZE = 80

class ConeDataset(Dataset):
    def __init__(self, folder):
        super().__init__()

        self.FILE_PATH = os.path.join("../data", folder)

        with open(os.path.join(self.FILE_PATH, "_annotations.coco.json")) as f:
            self.data = json.load(f)

        self.images = [0] * len(self.data["images"])
        self.keypoints = [0] * len(self.data["images"])

        # Note: this for loop assumes the image ids are 0-indexed without gaps, which seems to be true
        for img in self.data["images"]:
            img_path = os.path.join(self.FILE_PATH, img["file_name"])

            with Image.open(img_path) as i:
                self.images[img["id"]] = transforms.ToTensor()(i)

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
            ) / IMAGE_SIZE

        

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
            self.block(256, 1),
            # self.block(512, 2),
            # Fully connected
            nn.Flatten(),
            # nn.LazyLinear(512),
            # nn.ReLU(),
            # nn.LazyLinear(256),
            # nn.ReLU(),
            nn.LazyLinear(16),
        )

    def block(self, c: int, num_blocks: int) -> nn.Module:
        layers = []

        for i in range(num_blocks):
            layers.append(BottleneckResidual(c, 4, conv_1x1=(i == 0)))

        return nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        y = self.model(x)
        return self.model(x)


class KeypointLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def distance(a, b):
        return torch.sum((a - b) ** 2, dim=0) ** 0.5

    @staticmethod
    def cross_ratio(p1, p2, p3, p4):
        d13 = KeypointLoss.distance(p1, p3)
        d14 = KeypointLoss.distance(p1, p4)
        d23 = KeypointLoss.distance(p2, p3)
        d24 = KeypointLoss.distance(p2, p4)

        eps = 1e-8
        ratio = (d13 / (d14 + eps)) / (d23 / (d24 + eps) + eps)
        return ratio

    def forward(self, predicted, target):
        gamma = 0.0001

        points = []

        for i in range(0, 16, 2):
            points.append(torch.stack([predicted[:, i], predicted[:, i + 1]]))
        return torch.sum(
            torch.sum((predicted - target) ** 2, dim=1)
            + gamma * (KeypointLoss.cross_ratio(*points[:4]) - 1.35) ** 2
            + gamma * (KeypointLoss.cross_ratio(*points[4:]) - 1.35) ** 2
        )

writer = SummaryWriter(log_dir="runs/keypoints")

def train_epoch(dl, model, loss_fn, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    for i, (X, y) in enumerate(dl):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        fwd = model(X)
        l = loss_fn(fwd, y)
        l.backward()
        optimizer.step()

        ll = l.item()
        total_loss += ll

        if i % PRINT_MESSAGE_INTERVAL == 0:
            current = i * dl.batch_size + len(X)
            print(f"loss: {ll}  [{current:>5d}/{len(dl.dataset):>5d}]")
    avg_loss = total_loss / len(dl)
    writer.add_scalar("train/loss", avg_loss, epoch)
    print(f"Avg loss: {avg_loss}")


def test_epoch(dl, model, loss_fn, device, epoch):
    model.eval()

    num_batches = len(dl)
    test_loss = 0.0

    with torch.no_grad():
        for X, y in dl:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    writer.add_scalar("test/loss", test_loss, epoch)
    print(f"Test Avg loss: {test_loss:>8f} \n")


def random_test(model, dataset, device, epoch):
    model.eval()
    idx = random.randint(0, len(dataset) - 1)
    img, true_kps = dataset[idx]

    X = img.unsqueeze(0).to(device)
    with torch.no_grad():
        pred_kps = model(X).cpu().squeeze()


    np_img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR) 

    for x, y in zip(true_kps[::2], true_kps[1::2]):
        cv2.circle(np_img, (int(x.item() * IMAGE_SIZE), int(y.item()* IMAGE_SIZE)), 3, (0, 255, 0), -1)

    for x, y in zip(pred_kps[::2], pred_kps[1::2]):
        cv2.circle(np_img, (int(x.item()* IMAGE_SIZE), int(y.item()* IMAGE_SIZE)), 3, (0, 0, 255), -1)

    cv2.imwrite(f"test_out/epoch{epoch}.png", np_img)
    cv2.imshow("keypoints", np_img)
    cv2.waitKey(1)

SAVE_FILE_HEADER = os.path.join("../save", "test_run")
LOAD_FILE = "../save/test_run_epoch_500.pth"

train = ConeDataset("train")
train_dl = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
test = ConeDataset("test")
test_dl = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# import matplotlib.pyplot as plt
# ratios = []
# for i, (X, y) in enumerate(train_dl):
#     for yy in y:
#         points = []

#         for i in range(0, 16, 2):
#             points.append(torch.stack([yy[i], yy[i + 1]]))
#         ratios.append(KeypointLoss.cross_ratio(*points[:4]))
#         ratios.append(KeypointLoss.cross_ratio(*points[4:]))

# for i, (X, y) in enumerate(test_dl):
#     for yy in y:
#         points = []

#         for i in range(0, 16, 2):
#             points.append(torch.stack([yy[i], yy[i + 1]]))
#         ratios.append(KeypointLoss.cross_ratio(*points[:4]))
#         ratios.append(KeypointLoss.cross_ratio(*points[4:]))
# plt.hist(ratios, bins=100)
# plt.show()

if __name__ == "__init__":
    model = KeypointDetector().to(device)
    loss = KeypointLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if LOAD_FILE is not None:
        print(f"Loading from: {LOAD_FILE}")
        model.load_state_dict(torch.load(LOAD_FILE, map_location=device))

    epochs = EPOCHS
    for i in range(epochs):
        print(f"Epoch {i+1}/{epochs}")
        train_epoch(train_dl, model, loss, optimizer, device, i)
        test_epoch(test_dl, model, loss, device, i)
        random_test(model, test, device, i)


        if i % 10 == 0:
            save_path = SAVE_FILE_HEADER + f"_epoch_{i}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved model to {save_path}")

    writer.close()