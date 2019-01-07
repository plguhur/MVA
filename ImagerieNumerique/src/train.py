# This code is inspired by:
# https://raw.githubusercontent.com/BoyuanJiang/context_encoder_pytorch/master/train.py


from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
from src.models import _NetContext, _NetCompletion
from src.inpaint import load_network, inpainting2

REAL_LABEL = 1
FAKE_LABEL = 0


def train_random_mask(n_samples, range_size=[96, 128],
    global_shape=(256, 256), local_shape=(128, 128), pad=5,
    datamean=torch.FloatTensor([0.4560, 0.4472, 0.4155])):
    """ Generate n random mask with sizes 256, 256 inside 96:128, 96:128 """
    W, H = global_shape
    local_shape = np.asarray(local_shape)
    global_shape = np.asarray(global_shape)
    half = local_shape // 2
    M = torch.FloatTensor(n_samples, 3, *global_shape).fill_(0)
    range_size = np.asarray(range_size)
    center = np.empty((n_samples, 2))
    size = np.empty((n_samples, 2))
    for i in range(n_samples):
        w, h = np.random.randint(*range_size, size=2)
        tl_y = np.random.randint(pad, H-h-pad)
        tl_x = np.random.randint(pad, W-w-pad)
        fill = datamean.repeat(h, w, 1).    \
            transpose(0, 2).transpose(1, 2)
        M[i, :, tl_y:tl_y+h, tl_x:w+tl_x] = fill
        center[i] = [tl_y + h//2, tl_x + w//2]
    clip = np.clip(center-half, [0,0], local_shape)
    patch = np.hstack([clip, clip + local_shape])
    return M, patch.astype(int)


def load_dataset(dataset="lsun", dataroot="", image_size=256, batch_size=32):
    if dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=dataroot,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    elif dataset == 'lsun':
        dataset = dset.LSUN(db_path=dataroot, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif dataset == 'cifar10':
        transform = transforms.Compose([transforms.Scale(image_size),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = dset.CIFAR10(root=dataroot, download=True,
                               transform=transform)
    elif dataset == 'streetview':
        transform = transforms.Compose([transforms.Scale(image_size),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = dset.ImageFolder(root=dataroot, transform=transform )
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
                                         # num_workers=int(opt.workers))
    return dataloader


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_networks(g_load="completionnet_places2.t7", d_load="", cuda=False):
    if g_load == "":
        raise NotImplementedError()
        netG = _NetCompletion()
        netG.apply(weights_init)
    else:
        netG = load_network(g_load)

    if d_load == "":
        netD = _NetContext()
        netD.apply(weights_init)
    else:
        netD = load_network(g_load)

    if cuda:
        return netG.cuda(), netD.cuda()
    else:
        return netG, netD


def init_variables(batch_size, image_size, cuda=False):
    input_real = Variable(torch.FloatTensor(batch_size, 3, image_size,
                                                imageSize))
    input_cropped = Variable(torch.FloatTensor(batch_size, 3, image_size,
                                                imageSize))
    label = Variable(torch.FloatTensor(batch_size))
    real_center = Variable(torch.FloatTensor(batch_size, 3, image_size/2, image_size/2))
    if cuda:
        return input_real.cuda(), input_cropped.cuda(), label.cuda(), \
            real_center.cuda()
    else:
        return input_real, input_cropped, label, real_center


def train_discriminator(G, D, dataloader,
        datamean=torch.FloatTensor([0.4560, 0.4472, 0.4155]),
        n_epochs=5, checkpointing=-1):
    """ Given a trained generator G, this function trained the  discriminator D """

    criterion = nn.BCELoss()
    criterionMSE = nn.MSELoss()
    optimizerD = optim.Adam(D.parameters())

    for epoch in range(n_epochs):
        for k, data in enumerate(dataloader):
            # init var
            batch, _ = data
            n_samples, n_ch, h, w = batch.size()
            _local = torch.zeros((n_samples, n_ch, 128, 128))
            _label = torch.FloatTensor(n_samples)
            M, patch = train_random_mask(n_samples, datamean=datamean)
            D.zero_grad()

            # train with real data
            _global = batch.resize(n_samples, 3, 256, 256)
            for i in range(n_samples):
                p = patch[i] #FIXME get random patch
                _local[i, :, :, :] = _global[i, :, \
                                        p[0]:p[2], p[1]:p[3]]
            _label.data.fill_(REAL_LABEL)
            discrimator = D(_local, _global).resize(n_samples)
            errD_real = criterion(discrimator, _label)
            errD_real.backward()
            D_x = discrimator.data.mean()

            # train with fake
            fake = inpainting2(G, datamean, batch, M)
            _global = fake.resize(n_samples, 3, 256, 256)
            for i in range(n_samples):
                p = patch[i]
                _local[i, :, :, :] = _global[i, :, \
                                                p[0]:p[2], p[1]:p[3]]
            _label.data.fill_(FAKE_LABEL)
            discrimator = D(_local, _global).resize(n_samples)
            errD_fake = criterion(discrimator, _label)
            errD_fake.backward()
            D_G_z1 = discrimator.data.mean()

            errD = errD_real + errD_fake
            optimizerD.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f l_D(x): %.4f, l_D(G(x)): %.4f'
              % (epoch, n_epochs, k, len(dataloader),
                 errD.data[0], D_x, D_G_z1))

        if checkpointing > 0 and (epoch+1) % checkpointing == 0:
            torch.save({'epoch':epoch+1,
                    'state_dict':D.state_dict()},
                    'model/net_context.pth' )

if __name__ == "__main__":
    os.makedirs("result/train/cropped", exist_ok=True)
    os.makedirs("result/train/real", exist_ok=True)
    os.makedirs("result/train/recon", exist_ok=True)
    os.makedirs("model", exist_ok=True)

    cuda = False
    _, netD = get_networks(cuda=cuda)
    netG, datamean = load_network()
    dataloader = load_dataset(dataset="cifar10", dataroot="dataset/cifar10", batch_size=2)
    train_discriminator(netG, netD, dataloader, n_epochs=5)
