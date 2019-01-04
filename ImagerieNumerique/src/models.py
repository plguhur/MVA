import torch
import torch.nn as nn


class _netG(nn.Module):
    def __init__(self, opt):
        super(_netG, self).__init__()
        self.ngpu = opt.ngpu
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(opt.nc,opt.nef,4,2,1, bias=False),
            nn.ReLU(0.2, inplace=True),
            # state size: (nef) x 64 x 64
            nn.Conv2d(opt.nef,opt.nef,4,2,1, bias=False),
            nn.BatchNorm2d(opt.nef),
            nn.ReLU(0.2, inplace=True),
            # state size: (nef) x 32 x 32
            nn.Conv2d(opt.nef,opt.nef*2,4,2,1, bias=False),
            nn.BatchNorm2d(opt.nef*2),
            nn.ReLU(0.2, inplace=True),
            # state size: (nef*2) x 16 x 16
            nn.Conv2d(opt.nef*2,opt.nef*4,4,2,1, bias=False),
            nn.BatchNorm2d(opt.nef*4),
            nn.ReLU(0.2, inplace=True),
            # state size: (nef*4) x 8 x 8
            nn.Conv2d(opt.nef*4,opt.nef*8,4,2,1, bias=False),
            nn.BatchNorm2d(opt.nef*8),
            nn.ReLU(0.2, inplace=True),
            # state size: (nef*8) x 4 x 4
            nn.Conv2d(opt.nef*8,opt.nBottleneck,4, bias=False),
            # tate size: (nBottleneck) x 1 x 1
            nn.BatchNorm2d(opt.nBottleneck),
            nn.ReLU(0.2, inplace=True),
            # input is Bottleneck, going into a convolution
            nn.ConvTranspose2d(opt.nBottleneck, opt.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(opt.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(opt.ngf, opt.nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class _netD(nn.Module):
    def __init__(self, opt):
        super(_netlocalD, self).__init__()
        self.ngpu = opt.ngpu
        kernel_size = 5
        stride = 2
        self.local = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size, stride, bias=True),
            nn.BatchNorm2d(),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size, stride, bias=True),
            nn.BatchNorm2d(),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size, stride, bias=True),
            nn.BatchNorm2d(),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size, stride, bias=True),
            nn.BatchNorm2d(),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size, stride, bias=True),
            nn.BatchNorm2d(),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.Sigmoid()
        )

        self.global = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size, stride, bias=True),
            nn.BatchNorm2d(),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size, stride, bias=True),
            nn.BatchNorm2d(),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size, stride, bias=True),
            nn.BatchNorm2d(),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size, stride, bias=True),
            nn.BatchNorm2d(),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size, stride, bias=True),
            nn.BatchNorm2d(),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size, stride, bias=True),
            nn.BatchNorm2d(),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        torch.cat([self.local, self.global])
        return output.view(-1, 1)
