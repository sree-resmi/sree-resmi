import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

class ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.functional.conv2d(3, 64, 7, 2)
        self.BatchNorm = nn.functional.batch_norm()
        self.relu = nn.functional.relu()
        self.maxpooling = nn.functional.max_pool2d(3, 2)
        self.resblock1 = ResBlock(64, 64, 1)
        self.resblock2 = ResBlock(64, 128, 2)
        self.resblock3 = ResBlock(128, 256, 2)
        self.resblock4 = ResBlock(256, 512, 2)
        self.globalavgpool = nn.functional.avg_pool2d()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 2)
        self.sigmoid = nn.Sigmoid()
