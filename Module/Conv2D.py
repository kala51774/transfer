from torch import nn


class Conv2D(nn.Module):
    def __init__(self,in_channels=256,out_channels=256, kernel_size=3, strides=1,padding=1,sn=False):
        super().__init__()
        self.sn=sn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=strides, padding=padding,
                     padding_mode="reflect")

    def forward(self,x):
        return self.conv(x)