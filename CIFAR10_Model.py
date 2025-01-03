
import torch.nn as nn
import torch.nn.functional as F


class ColorImgCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # in_channels = 3 because we have RGB image. So it's basically 3 matrixes,
        # matrix A representing the Red values of the image
        # matrix B representing the Green values of the image
        # matrix C representing the Blue values of the image
        # A + B + C = color image
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.conv3 = nn.Conv2d(36, 52, 3, 1)


        self.fc1 = nn.Linear(64 * 6 * 6, 180)
        self.fc2 = nn.Linear(180, 120)
        self.out = nn.Linear(120, 10)

        # avoiding from overfitting. we turn off randomly 50% of the neurons so the
        #  model will learn with its weight but he will not learn for the next iteration
        #self.dropout = nn.Dropout(0.2)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)


        x = x.view(x.size(0), -1) # flatten the image

        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        x = self.out(x)

        return x