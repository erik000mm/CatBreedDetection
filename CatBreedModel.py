import torch.nn as nn
import torch

from ConvUnit import *


class InceptionBlock_A(nn.Module):

    def __init__(self, in_channels, features):
        super(InceptionBlock_A, self).__init__()

        self.branch_1 = nn.Sequential(
            ConvUnit(in_channels=in_channels, out_channels=features, kernel_size=1)
        )

        self.branch_2 = nn.Sequential(
            ConvUnit(in_channels=in_channels, out_channels=32, kernel_size=1),
            ConvUnit(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            ConvUnit(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        )

        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1),
            ConvUnit(in_channels=in_channels, out_channels=48, kernel_size=1, padding=1)
        )

    def forward(self, x):
        branches = (self.branch_1, self.branch_2, self.branch_3)

        return torch.cat([branch(x) for branch in branches], 1)


class CatBreedModel(nn.Module):
    def __init__(self):
        super(CatBreedModel, self).__init__()
        self.conv_1 = ConvUnit(3, 32, kernel_size=5, stride=2)
        self.conv_2 = ConvUnit(32, 64, kernel_size=3)
        self.conv_3 = ConvUnit(64, 64, kernel_size=3)

        self.maxpool_1 = nn.MaxPool2d(3, stride=2)

        self.conv_4 = ConvUnit(64, 92, kernel_size=3)
        self.conv_5 = ConvUnit(92, 112, kernel_size=3)
        self.conv_6 = ConvUnit(112, 152, kernel_size=3)

        self.maxpool_2 = nn.MaxPool2d(3, stride=2)

        self.block_a1 = InceptionBlock_A(in_channels=152, features=92)
        self.block_a2 = InceptionBlock_A(in_channels=204, features=128)
        self.block_a3 = InceptionBlock_A(in_channels=240, features=182)

        self.maxpool_3 = nn.MaxPool2d(3, stride=2)

        self.conv_7 = ConvUnit(294, 312, kernel_size=3)
        self.conv_8 = ConvUnit(312, 330, kernel_size=3)
        self.conv_9 = ConvUnit(330, 342, kernel_size=3)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dr = nn.Dropout(p=0.2)

        self.fc = torch.nn.Linear(in_features=342, out_features=22)


    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.maxpool_1(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.maxpool_2(x)
        x = self.block_a1(x)
        x = self.block_a2(x)
        x = self.block_a3(x)
        x = self.maxpool_3(x)
        x = self.conv_7(x)
        x = self.conv_8(x)
        x = self.conv_9(x)
        x = self.adaptive_pool(x)
        x = x.flatten(start_dim=1)
        x = self.dr(x)
        x = self.fc(x)
        return x