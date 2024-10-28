# 定义解码器
from torch import nn

from Block.ResBlock import ResBlock
from model.ILN import ILN


class GenDecoder(nn.Module):
    def __init__(self, hw=64, out_channels=3, n_block=None, use_bias=False):
        super().__init__()
        # 残差块
        model = []
        res = hw * 4
        for i in range(n_block):
            model += [ResBlock(res)]
        # frist upsample
        mult = 2 ** (n_block // 2)
        model += [nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
                  nn.Conv2d(int(hw * mult), int(hw * mult / 2), kernel_size=3, stride=1, padding=1),
                  nn.Conv2d(int(hw * mult / 2), int(hw * mult / 2), kernel_size=3, stride=1, padding=1),
                  ILN(int(hw * mult / 2)),
                  nn.ReLU(True)]
        # second upsampling
        model += [nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
                  nn.Conv2d(int(hw * mult / 2), int(hw * mult / 4), kernel_size=3, stride=1, padding=1),
                  nn.Conv2d(int(hw * mult / 4), int(hw * mult / 4), kernel_size=3, stride=1, padding=1),
                  ILN(int(hw * mult / 4)),
                  nn.ReLU(True)]
        # addtional layer
        model += [nn.Conv2d(int(hw * mult / 4), int(hw * mult / 8), kernel_size=3, stride=1, padding=1),
                  nn.Conv2d(int(hw * mult / 8), int(hw * mult / 8), kernel_size=3, stride=1, padding=1),
                  ILN(int(hw * mult / 8)), nn.ReLU(True)]
        model += [nn.Conv2d(int(hw * mult / 8), int(hw * mult / 16), kernel_size=3, stride=1, padding=1),
                  nn.Conv2d(int(hw * mult / 16), int(hw * mult / 16), kernel_size=3, stride=1, padding=1),
                  ILN(int(hw * mult / 16)), nn.ReLU(True)]
        model += [nn.Conv2d(int(hw * mult / 16), out_channels, 7, 1, 3), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    # 定义生成器
