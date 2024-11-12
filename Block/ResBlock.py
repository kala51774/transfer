# 定义残差块
from torch import nn

from model.ILN import ILN


class ResBlock(nn.Module):
    def __init__(self, channels, use_bias=False):
        super().__init__()
        Res_block = []
        Res_block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(channels, channels, 3, 1, 0, bias=use_bias),
                      ILN(channels), nn.PReLU(num_parameters=1)]

        Res_block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(channels, channels, 3, 1, 0, bias=use_bias),
                      ILN(channels)]
        self.Res_block = nn.Sequential(*Res_block)

    def forward(self, x):
        return x + self.Res_block(x)
