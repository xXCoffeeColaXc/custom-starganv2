import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from utils import *

class AdaINResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, sdim, up=False):
        super(AdaINResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.norm1 = AdaIN(sdim, in_ch)
        self.norm2 = AdaIN(sdim, out_ch)
        self.lrelu = nn.LeakyReLU(0.2)
        self.is_up = up
        self.is_sc = in_ch != out_ch
        if self.is_sc:
            self.sc = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)

    def up(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')

    def forward(self, x, s):
        res = self.lrelu(self.norm1(x, s))
        if self.is_up:
            x = self.up(x)
            res = self.up(res)
        if self.is_sc: x = self.sc(x)
        res = self.conv1(res)
        res = self.conv2(self.lrelu(self.norm2(res, s)))
        return (x + res) / math.sqrt(2)
    
class AdaIN(nn.Module):
    def __init__(self, sdim, nf):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm2d(nf)
        self.gamma = nn.Linear(sdim, nf)
        self.beta = nn.Linear(sdim, nf)
        self.apply(init_fc_weight_one)

    def forward(self, x, s):
        B, C, H, W = x.size()
        return (1 + self.gamma(s).view(B, C, 1, 1)) * self.norm(x) + self.beta(s).view(B, C, 1, 1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm=False, down=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)

        self.norm = norm
        if norm:
            self.norm1 = nn.InstanceNorm2d(in_ch, affine=True)
            self.norm2 = nn.InstanceNorm2d(in_ch, affine=True)

        self.lrelu = nn.LeakyReLU(0.2)
        self.is_down = down

        self.is_sc = in_ch != out_ch
        if self.is_sc:
            self.sc = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)

    def down(self, x):
        return F.avg_pool2d(x, 2)

    def forward(self, x):
        if self.norm: res = self.norm1(x)
        else: res = x
        res = self.conv1(self.lrelu(res))
        if self.is_sc: x = self.sc(x)
        if self.is_down:
            x = self.down(x)
            res = self.down(res)
        if self.norm: res = self.norm2(res)
        res = self.conv2(self.lrelu(res))
        return (x + res) / math.sqrt(2)


class Generator(nn.Module):
    def __init__(self, nf, sdim):
        super(Generator, self).__init__()
        self.conv_in = nn.Conv2d(3, nf, 3, 1, 1)
        self.enc = nn.Sequential(
            ResBlock(nf, 2*nf, norm=True, down=True),
            ResBlock(2*nf, 4*nf, norm=True, down=True),
            ResBlock(4*nf, 8*nf, norm=True, down=True),
            ResBlock(8*nf, 8*nf, norm=True),
            ResBlock(8*nf, 8*nf, norm=True)
        )
        self.dec = nn.ModuleList([
            AdaINResBlock(8*nf, 8*nf, sdim),
            AdaINResBlock(8*nf, 8*nf, sdim),
            AdaINResBlock(8*nf, 4*nf, sdim, up=True),
            AdaINResBlock(4*nf, 2*nf, sdim, up=True),
            AdaINResBlock(2*nf, nf, sdim, up=True)
        ])
        self.conv_out = nn.Sequential(
            nn.InstanceNorm2d(nf, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf, 3, 1, 1, 0)
        )
        self.apply(init_conv_weight)

    def forward(self, x, s):
        x = self.conv_in(x)
        x = self.enc(x)
        for layer in self.dec:
            x = layer(x, s)
        x = self.conv_out(x)
        return x
    
class MappingNetwork(nn.Module):
    def __init__(self, nz, nd, sdim):
        super(MappingNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(nz, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.unshared = nn.ModuleList()
        for i in range(nd):
            self.unshared.append(nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, sdim)
            ))

        self.apply(init_conv_weight)
        self.apply(init_fc_weight_zero)

    def forward(self, z, y):          # z: B x nz, y: B
        B = z.size(0)
        z = self.shared(z)            # B x 512
        s = [layer(z) for layer in self.unshared]
        s = torch.stack(s, dim=1)     # B x nd x sdim
        i = torch.LongTensor(range(B)).cuda()
        return s[i, y]                # B x sdim
    
class StyleEncoder(nn.Module):
    def __init__(self, nf, nd, sdim):
        super(StyleEncoder, self).__init__()
        self.conv_in = nn.Conv2d(3, nf, 3, 1, 1)
        self.res = nn.Sequential(
            ResBlock(nf, 2*nf, down=True),
            ResBlock(2*nf, 4*nf, down=True),
            ResBlock(4*nf, 8*nf, down=True),
            ResBlock(8*nf, 8*nf, down=True),
            ResBlock(8*nf, 8*nf, down=True)
        )
        self.conv_out = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(8*nf, 8*nf, 4, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8*nf, nd*sdim, 1, 1, 0)
        )
        self.nd = nd
        self.apply(init_conv_weight)

    def forward(self, x, y):            # x: B x 3 x 128 x 128, y: B
        B = x.size(0)
        x = self.conv_in(x)             # B x nf x 128 x 128
        x = self.res(x)                 # B x 8nf x 4 x 4
        x = self.conv_out(x)            # B x nd*sdim x 1 x 1
        x = x.view(B, self.nd, -1)      # B x nd x sdim
        i = torch.LongTensor(range(B)).cuda()
        return x[i, y]                  # B x sdim
    
class Discriminator(nn.Module):
    def __init__(self, nf, nd):
        super(Discriminator, self).__init__()
        self.conv_in = nn.Conv2d(3, nf, 3, 1, 1)
        self.res = nn.Sequential(
            ResBlock(nf, 2*nf, down=True),
            ResBlock(2*nf, 4*nf, down=True),
            ResBlock(4*nf, 8*nf, down=True),
            ResBlock(8*nf, 8*nf, down=True),
            ResBlock(8*nf, 8*nf, down=True)
        )
        self.conv_out = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(8*nf, 8*nf, 4, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8*nf, nd, 1, 1, 0)
        )
        self.apply(init_conv_weight)

    def forward(self, x, y):        # x: B x 3 x 128 x 128, y: B
        B = x.size(0)
        x = self.conv_in(x)         # B x nf x 128 x 128
        x = self.res(x)             # B x 8nf x 4 x 4
        x = self.conv_out(x)        # B x nd x 1 x 1
        x = x.view(B, -1)           # B x nd
        i = torch.LongTensor(range(B)).cuda()
        return x[i, y]              # B