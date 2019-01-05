import torch
import torch.nn as nn


class NetCompletion(nn.Module):
    def __init__(self):
        super(NetCompletion, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # ------------------------------------------
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # ------------------------------------------
            nn.Conv2d(128, 256, 1, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 1, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 1, dilation=4),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 1, dilation=8),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 1, dilation=16),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #------------------------------------------
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #------------------------------------------
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 3, kernel_size=3),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


class _NetContext(nn.Module):
    def __init__(self):
        super(_NetContext, self).__init__()
        kernel_size = 5
        stride = 2
        self.__local = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size, stride, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size, stride, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size, stride, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size, stride, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size, stride, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.__global = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size, stride, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size, stride, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size, stride, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size, stride, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size, stride, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size, stride, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

    def forward(self, input):
        x1 = self.__local(input)
        x1 = x1.view(x1.size(0), -1)
        x1 = nn.Linear(x1.size(-1), 1024)(x1)
        x1 = nn.ReLU()(nn.BatchNorm1d(1024)(x1))

        x2 = self.__global(input)
        x2 = x2.view(x2.size(0), -1)
        x2 = nn.Linear(x2.size(-1), 1024)(x2)
        x2 = nn.ReLU()(nn.BatchNorm1d(1024)(x2))

        x = torch.cat([x1, x2])
        x = nn.Linear(1024, 1)(x)
        return nn.Sigmoid()(x)
