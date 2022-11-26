import math

import torch
from torch import nn

from config import Config


# pulled from Dr. Karpathy's minGPT implementation
class GELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


# generator block for producing samples from noise
class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor)
        self.conv = nn.Conv2d(in_channels, out_channels, (3, 3), padding="same")
        self.b_norm = nn.BatchNorm2d(out_channels)
        self.gelu = GELU()
    
    def forward(self, x):
        out = self.upsample(x)
        out = self.conv(out)
        out = self.b_norm(out)

        return self.gelu(out)


# discrimantor block for identifying generator generated samples
class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, (3, 3), padding="same")
        self.pool = nn.MaxPool2d((2, 2))
        self.b_norm = nn.BatchNorm2d(out_channels)
        self.gelu = GELU()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.pool(out)
        out = self.b_norm(out)

        return self.gelu(out)


# generator model
class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.batch_size = config.batch_size
        # must sqrt to an integer
        self.noise_dims = config.noise_dims

        self.gen_blocks = [GeneratorBlock(1, config.gen_in_channels, 2)]
        for i in range(config.num_gen_upscale_blocks):
            self.gen_blocks += [GeneratorBlock(config.gen_in_channels, config.gen_in_channels, 2)]
        
        for i in range(config.num_gen_const_blocks):
            self.gen_blocks += [GeneratorBlock(config.gen_in_channels, config.gen_in_channels, 1)]

        self.final_conv = nn.Conv2d(config.gen_in_channels, 3, (3, 3), padding="same")
        self.tanh = nn.Tanh()

        self.apply(self.weights_init)
    
    # initialize weights according to a normal distribution
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        # dims[0] = batch_size, dims[1] = channels
        out = torch.reshape(x, (self.batch_size, 1, self.noise_dims, self.noise_dims))

        for i in range(len(self.gen_blocks)):
            out = self.gen_blocks[i](out)
        
        out = self.final_conv(out)

        return self.tanh(out)


# discriminator model
class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.batch_size = config.batch_size

        self.dis_blocks = [DiscriminatorBlock(3, config.dis_in_channels)]
        prev_in_channels = 8
        for i in range(1, config.num_dis_blocks):
            self.dis_blocks += [DiscriminatorBlock(prev_in_channels, prev_in_channels*2)]
            prev_in_channels *= 2

        self.flatten = nn.Flatten()

        # TODO fix hidden units
        self.linear1 = nn.Linear(int(config.flatten_dims), config.dis_hidden_units)
        self.linear2 = nn.Linear(config.dis_hidden_units, 1)

        self.dropout = nn.Dropout(config.dropout)

        self.sigmoid = nn.Sigmoid()

        self.apply(self.weights_init)
    
    # initialize weights according to a normal distribution
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        out = x
        for i in range(len(self.dis_blocks)):
            out = self.dis_blocks[i](out)
        
        out = self.flatten(out)

        out = self.linear1(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.dropout(out)

        return self.sigmoid(out)