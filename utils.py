import math

import torch
from torch import nn

from config import Config


device = torch.device("cuda")


# generator block for producing samples from noise
class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides, padding):
        super().__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, padding=padding, stride=strides, bias=False)
        self.b_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.b_norm(out)

        return self.relu(out)


# discrimantor block for identifying generator generated samples
class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, (4, 4), padding=1, stride=2, bias=False)
        self.b_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.b_norm(out)

        return self.relu(out)


# generator model
class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.batch_size = config.batch_size
        # must sqrt to an integer
        self.noise_dims = config.noise_dims

        self.gen_blocks = [GeneratorBlock(config.noise_dims, config.gen_in_channels, 1, 0).to(device)]
        for i in range(config.num_gen_blocks):
            self.gen_blocks += [GeneratorBlock(config.gen_in_channels, config.gen_in_channels, 2, 1).to(device)]
        
        self.gen_blocks += [GeneratorBlock(config.gen_in_channels, int(config.gen_in_channels/2), 2, 1).to(device)]

        self.final_conv_transpose = nn.ConvTranspose2d(int(config.gen_in_channels/2), 3, kernel_size=4, padding=1, stride=2, bias=False)

        self.tanh = nn.Tanh()
    
    def forward(self, x):
        # dims[0] = batch_size, dims[1] = channels
        out = torch.reshape(x, (self.batch_size, self.noise_dims, 1, 1))

        for i in range(len(self.gen_blocks)):
            out = self.gen_blocks[i](out)
        
        out = self.final_conv_transpose(out)

        return self.tanh(out)


# discriminator model
class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.batch_size = config.batch_size

        self.dis_blocks = [DiscriminatorBlock(3, config.dis_in_channels).to(device)]
        prev_in_channels = config.dis_in_channels
        for i in range(1, config.num_dis_blocks):
            if prev_in_channels <= 64:
                self.dis_blocks += [DiscriminatorBlock(prev_in_channels, prev_in_channels*2).to(device)]
            else:
                self.dis_blocks += [DiscriminatorBlock(prev_in_channels, prev_in_channels).to(device)]
            if prev_in_channels <= 64:
                prev_in_channels *= 2

        self.flatten = nn.Flatten()

        self.final_conv = nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0, bias=False)

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = x
        for i in range(len(self.dis_blocks)):
            out = self.dis_blocks[i](out)
        
        out = self.final_conv(out)

        out = self.flatten(out)

        return self.sigmoid(out)