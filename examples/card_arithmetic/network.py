from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.classifier = nn.Conv2d(64, 52, 1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def forward(self, x, marg_idx=None, type=1):
        if x.dim() == 5:
            x = x.flatten(0, 1)
        x = self.ReLU(self.conv1(x))
        x = self.ReLU(self.conv2(x))
        x = self.ReLU(self.conv3(x))
        x = self.ReLU(self.conv4(x))
        x = self.dropout(x)
        x = self.classifier(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.softmax(x)
        return x

