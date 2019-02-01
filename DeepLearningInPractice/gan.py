import argparse
import os
import numpy as np
import math

import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=10002, help='number of epochs of training')
parser.add_argument('--n_train_gen', type=int, default=1, help='train/batch of generator')
parser.add_argument('--n_train_disc', type=int, default=1, help='train/batch of discriminator')
parser.add_argument('--batch_size', type=int, default=4096, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=32, help='dimensionality of the latent space')
parser.add_argument('--dim', type=int, default=2, help='dimension of the problem')
parser.add_argument('--sample_interval', type=int, default=500, help='interval betwen image samples')
parser.add_argument('--output', type=str, default="results", help='output dir')
parser.add_argument('--no-reload', action='store_true', help='reload from existing models')
parser.add_argument('--start', type=int, default=0, help='first epoch')
opt = parser.parse_args()
print(opt)


plots = os.path.join(opt.output, "plots")
models = os.path.join(opt.output, "models")
gen_filename = os.path.join(models, "generator.pt")
disc_filename = os.path.join(models, "discriminator.pt")
os.makedirs(opt.output, exist_ok=True)
os.makedirs(plots, exist_ok=True)
os.makedirs(models, exist_ok=True)


cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

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
        img = img.view(img.size(0), opt.dim)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

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

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
if not(opt.no_reload) and os.path.isfile(gen_filename) and \
    os.path.isfile(disc_filename):
    print("Loading models:\n - Generator:", gen_filename)
    print(" - Discriminator:", disc_filename)
    generator = torch.load(gen_filename)
    discriminator = torch.load(disc_filename)
else:
    print("Instanciating generator and discriminator")
    generator = Generator()
    discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
# os.makedirs('../../data/mnist', exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST('../../data/mnist', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                    ])),
#     batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------


def generate_batch(batchlen):
    """This function generates a batch of length 'batchlen' from the 25-gaussian dataset.

    return a torch tensor of dimensions (batchlen, 2)
    """
    # to sample from the gaussian mixture, we first sample the means for each point, then
    # add a gaussian noise with small variance
    samples = torch.multinomial(torch.tensor([0.2,0.2,0.2,0.2,0.2]), 2*batchlen, replacement=True)
    means = (2.0 * (samples - 2.0)).view(batchlen,2).type(torch.FloatTensor)
    return torch.normal(means, 0.05)



for epoch in range(opt.start, opt.n_epochs):
    batch = generate_batch(opt.batch_size)

    # Adversarial ground truths
    valid = Variable(Tensor(batch.size(0), 1).fill_(1.0), requires_grad=False)
    fake = Variable(Tensor(batch.size(0), 1).fill_(0.0), requires_grad=False)


    # Configure input
    real_batch = Variable(batch.type(Tensor))

    # -----------------
    #  Train Generator
    # -----------------
    for _ in range(opt.n_train_gen):
        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (batch.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_batch = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_batch), valid)

        g_loss.backward()
        optimizer_G.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------
    for _ in range(opt.n_train_disc):
        optimizer_D.zero_grad()
        batch = generate_batch(opt.batch_size)
        real_batch = Variable(batch.type(Tensor))
        z = Variable(Tensor(np.random.normal(0, 1, (batch.shape[0], opt.latent_dim))))
        gen_batch = generator(z)
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_batch), valid)
        fake_loss = adversarial_loss(discriminator(gen_batch.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()


    if (epoch - 1) % (opt.n_epochs // 30) == 0:
        print ("[Epoch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs,
                                                        d_loss.item(), g_loss.item()))

    if (epoch - 1) % opt.sample_interval == 0:
        real_batch = real_batch.cpu().data.numpy()
        gen_batch = gen_batch.cpu().data.numpy()
        plt.clf()
        plt.scatter(real_batch[:,0], real_batch[:,1], s=2.0, label='real data')
        plt.scatter(gen_batch[:,0], gen_batch[:,1], s=2.0, label='fake data')
        plt.legend()
        filename = os.path.join(plots, f"results-{epoch}.png")
        plt.savefig(filename)

        torch.save(generator, gen_filename)
        torch.save(discriminator, disc_filename)
