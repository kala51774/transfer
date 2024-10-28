from torch import nn
from torch.nn.utils import spectral_norm


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, act=True):
        super().__init__()
        self.act = act
        self.sn_conv = spectral_norm(nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                padding_mode="zeros" # Author's code used slim.convolution2d, which is using SAME padding (zero padding in pytorch)
            ))
        self.LReLU = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        x = self.sn_conv(x)
        if self.act:
            x = self.LReLU(x)
        return x