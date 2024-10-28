from torch import nn

from Block.ConvBlock import ConvBlock
from Module.Conv2D import Conv2D


class Discriminator_T(nn.Module):
    def __init__(self):
        super( Discriminator_T,self).__init__()
        self.channels=[1,32,64,128,256]
        self.model=[]
        self.model+=[ConvBlock(in_channels=self.channels[0],out_channels=self.channels[1],padding=1,sn=True),
                     ConvBlock(in_channels=self.channels[1],out_channels=self.channels[2],strides=2,padding=1,sn=True),
                     ConvBlock(in_channels=self.channels[2],out_channels=self.channels[3],strides=2,padding=1,sn=True),
                     ConvBlock(in_channels=self.channels[3],out_channels=self.channels[4],strides=2,padding=1,sn=True),
                     Conv2D(in_channels=self.channels[4],out_channels=1,padding=1,sn=True)]
        self.model=nn.Sequential(*self.model)
    def forward(self,x):
        #3*1
        out=self.model(x)
        return out