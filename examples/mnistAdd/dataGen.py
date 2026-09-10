import os

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms

_DIR = os.path.dirname(os.path.abspath(__file__))

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
        i1, i2, l = self.data[index]
        return self.dataset[i1][0], self.dataset[i2][0], l

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081, ))])
train_dataset = MNIST_Addition(torchvision.datasets.MNIST(root=os.path.join(_DIR, 'data'), train=True, download=True, transform=transform), os.path.join(_DIR, 'data/train_data.txt'))
test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(os.path.join(_DIR, 'data'), train=False, transform=transform), batch_size=1000, shuffle=True)

BATCH_SIZE = 50

def collate(batch):
    i1s, i2s, ls = zip(*batch)
    return torch.stack(i1s), torch.stack(i2s), ls

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
dataList = []
obsList = []
for i1, i2, ls in train_loader:
    dataList.append({'i1': i1, 'i2': i2})
    obsList.append([f':- not addition(i1, i2, {l}).' for l in ls])