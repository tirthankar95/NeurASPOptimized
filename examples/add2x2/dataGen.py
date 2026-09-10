import os

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class MNIST_Addition(Dataset):

    def __init__(self, dataset, examples):
        self.data = list()
        self.dataset = dataset
        with open(examples) as f:
            for line in f:
                line = line.strip().split(' ')
                self.data.append(tuple([int(i) for i in line]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        i1, i2, i3, i4, r1, r2, c1, c2 = self.data[index]
        return torch.cat((self.dataset[i1][0], self.dataset[i2][0], self.dataset[i3][0], self.dataset[i4][0]), 0).unsqueeze(1), r1, r2, c1, c2

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081, ))])

_BASE_DIR = os.path.dirname(__file__)
_DATA_DIR = os.path.abspath(os.path.join(_BASE_DIR, '..', 'data'))
_ADD2X2_TRAIN = os.path.join(_DATA_DIR, 'add2x2_train.txt')

trainDataset = MNIST_Addition(
    torchvision.datasets.MNIST(root=_DATA_DIR, train=True, download=True, transform=transform),
    _ADD2X2_TRAIN
)
# only randomly take 3000 data
np.random.seed(1) # fix the random seed for reproducibility
trainDataset = torch.utils.data.Subset(trainDataset, np.random.choice(len(trainDataset), 3000, replace=False))
testLoader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(_DATA_DIR, train=False, transform=transform),
    batch_size=1000,
    shuffle=True
)
dataList = []
obsList = []
BATCH_SIZE = 50
trainDataset = torch.utils.data.DataLoader(
    trainDataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True
)
for images, r1, r2, c1, c2 in trainDataset:
    dataList.append({'i': images})
    obsList.append([f':- not add2x2({r1},{r2},{c1},{c2}).' for r1, r2, c1, c2 in zip(r1, r2, c1, c2)])
