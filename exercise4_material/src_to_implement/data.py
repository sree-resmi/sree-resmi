from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        self._transform = tv.transforms.Compose([tv.transforms.ToPILImage, tv.transforms.ToTensor(), tv.transforms.Normalize(mean=train_mean, std=train_std)])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = imread(self.data.values[index][0])
        label = self.data.values[index][1:3]
        image = gray2rgb(image)
        img = self._transform.transforms[2](self._transform.transforms[1](np.array(self._transform.transforms[0](image).mode)))
        x = torch.tensor(img)
        y = torch.tensor([bool(label[0]), bool(label[1])])
        return (x, y)