# 定义卷积
from torch import nn

from Module.layer_norm import layer_norm
from utils.funs import init_weights


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=0, sn=False, bias=False,
                 padding_mode='reflect'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=strides, padding=padding, bias=bias,
                              padding_mode=padding_mode)
        self.ins_norm = layer_norm()
        self.activation = nn.LeakyReLU(0.2, True)

        init_weights(self)

    def forward(self, x):
        out = self.conv(x)
        out = self.ins_norm(out)
        out = self.activation(out)

        return out
