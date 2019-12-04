# Creator: Hoang-Dung Do

import torch.nn.functional as func
import torch.nn as nn


class CNN3Conv(nn.Module):
    def __init__(self):
        super(CNN3Conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # (32,16,16)
        x = self.pool(func.relu(self.conv1(x)))
        # (64,8,8)
        x = self.pool(func.relu(self.conv2(x)))
        # (128,4,4)
        x = self.pool(func.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        return x


class CNN4Conv(nn.Module):
    def __init__(self):
        super(CNN4Conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 10)

    def forward(self, x):
        # (32,16,16)
        x = self.pool(func.relu(self.conv1(x)))
        # (64,8,8)
        x = self.pool(func.relu(self.conv2(x)))
        # (128,4,4)
        x = self.pool(func.relu(self.conv3(x)))
        # (256,2,2)
        x = self.pool(func.relu(self.conv4(x)))
        x = x.view(-1, 256 * 2 * 2)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = func.relu(self.fc3(x))
        x = func.relu(self.fc4(x))
        return x
