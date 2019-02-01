import torch.nn as nn
import torch.nn.functional as F
import torch

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, opt.dim)
            # nn.Tanh()
        )

    def generate(self, batchlen):
        z = torch.normal(torch.zeros(batchlen, PRIOR_N), 1.0)
        return self.forward(z)

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), self.opt.dim)
        return img

class LinearDiscriminator(nn.Module):
    def __init__(self, opt):
        super(LinearDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity



class ConvDiscriminator(nn.Module):
    def __init__(self, opt):
        super(ConvDiscriminator, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(opt.dim, 1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 8, 3, stride=2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(392, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        img_flat = self.fc1(img)
        img = img_flat.view(img.size(0), 1, 32, 32)
        img = self.conv1(img)
        img_flat = img.view(img.size(0), -1)
        validity = self.fc2(img_flat)
        return validity
