import torch
import torch.nn as nn


completionnet_places2 = nn.Sequential( # Sequential,
	nn.Conv2d(4,64,(5, 5),(1, 1),(2, 2)),
	nn.BatchNorm2d(64),
	nn.ReLU(),
	nn.Conv2d(64,128,(3, 3),(2, 2),(1, 1)),
	nn.BatchNorm2d(128),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.BatchNorm2d(128),
	nn.ReLU(),
	nn.Conv2d(128,256,(3, 3),(2, 2),(1, 1)),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),(1, 1),1),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),(1, 1),1),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(2, 2),(2, 2),1),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(4, 4),(4, 4),1),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(8, 8),(8, 8),1),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(16, 16),(16, 16),1),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),(1, 1),1),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),(1, 1),1),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.ConvTranspose2d(256,128,(4, 4),(2, 2),(1, 1),(0, 0)),
	nn.BatchNorm2d(128),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.BatchNorm2d(128),
	nn.ReLU(),
	nn.ConvTranspose2d(128,64,(4, 4),(2, 2),(1, 1),(0, 0)),
	nn.BatchNorm2d(64),
	nn.ReLU(),
	nn.Conv2d(64,32,(3, 3),(1, 1),(1, 1)),
	nn.BatchNorm2d(32),
	nn.ReLU(),
	nn.Conv2d(32,3,(3, 3),(1, 1),(1, 1)),
	nn.Sigmoid(),
)


class _NetCompletion(nn.Module):
    def __init__(self):
        super(_NetCompletion, self).__init__()
        self.main = completionnet_places2

    def forward(self, input):
        # Input is the image with the mask concatenated
        return self.main(input)


class _NetContext(nn.Module):
    def __init__(self):
        super(_NetContext, self).__init__()
        kernel_size = 5
        stride = 2
        self.__local = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size, stride),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size, stride),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size, stride),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size, stride),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size, stride),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.__global = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size, stride),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size, stride),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size, stride),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size, stride),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size, stride),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size, stride),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

    def forward(self, x1, x2):
        x1 = self.__local(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = nn.Linear(x1.size(-1), 1024)(x1)
        x1 = nn.ReLU()(nn.BatchNorm1d(1024)(x1))

        x2 = self.__global(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = nn.Linear(x2.size(-1), 1024)(x2)
        x2 = nn.ReLU()(nn.BatchNorm1d(1024)(x2))

        x = torch.cat([x1, x2], 1)
        x = nn.Linear(2048, 1)(x)
        return nn.Sigmoid()(x)
