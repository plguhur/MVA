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
from tqdm import tqdm_notebook as tqdm

cudnn.benchmark = True

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class DCGAN_Generator(nn.Module):
    def __init__(self, ngf, nc, nz):
        super(DCGAN_Generator, self).__init__()
        self.ngpu = 1
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output




class DCGAN_Discriminator(nn.Module):
    def __init__(self, ndf, nc):
        super(DCGAN_Discriminator, self).__init__()
        self.ngpu = 1
        self.ndf = ndf
        self.nc = nc
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


    
    

def build_dcgan(device, nz=100, nc=3, ngf=64, ndf=64):
    netG = DCGAN_Generator(ngf, nc, nz).to(device)
    netG.apply(weights_init)
    
    netD = DCGAN_Discriminator(ndf, nc).to(device)
    netD.apply(weights_init)
    
    optimizerD = optim.Adam(netD.parameters())
    optimizerG = optim.Adam(netG.parameters())

    return netG, optimizerG, netD, optimizerD





def train_gan_an_epoch(dataloader, netG, optimizerG, netD, optimizerD, criterion, 
                       device, nz=100, silent=False, output=""):
    
    real_label = 1
    fake_label = 0

    with tqdm(range(len(dataloader)), disable=silent) as t:
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device)
            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            t.set_postfix(err_disc=errD.item(), err_gen=errG.item(), 
                          disc=D_x, dgz1=D_G_z1, dgz2=D_G_z2)
            t.update()
            
            # do checkpointing
            if os.path.isdir(output):
                torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (output, epoch))
                torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (output, epoch))
    
    return errD.item(), errG.item()



class WGANGP_Generator(nn.Module):
    def __init__(self, ngf, nc, nz):
        super(WGANGP_Generator, self).__init__()
        self.img_size = ngf**2*nc
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(nz, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(self.img_size)),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], nc, ngf, ngf)
        return img


class WGANGP_Discriminator(nn.Module):
    def __init__(self, ndf, nc):
        super(WGANGP_Discriminator, self).__init__()
        self.img_size = ndf**2*nc
        
        self.model = nn.Sequential(
            nn.Linear(int(self.img_size), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity
    

    
def export_gan_result(dataloader, netG, netD, device, output, epoch, nz=100):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    netD.zero_grad()
    real_cpu = images.to(device)
    vutils.save_image(real_cpu,
        '%s/real_samples.png' % output,
        normalize=True)
    
    batch_size = real_cpu.size(0)
    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
    fake = netG(fixed_noise)
    vutils.save_image(fake.detach(),
        '%s/fake_samples_epoch_%03d.png' % (output, epoch),
           normalize=True)
