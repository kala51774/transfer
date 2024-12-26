# 定义残差块
from torch import nn
import torch
from model.ILN import ILN

from WTConv.wtconv import WTConv2d

class ResBlock(nn.Module):
    def __init__(self, channels, use_bias=False):
        super().__init__()

        # self.wtconv = nn.Sequential(WTConv2d(in_channels=channels,out_channels=channels,kernel_size=3,stride=1,bias=use_bias,wt_levels=5),
    #    )
       

        Res_block = []
        Res_block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(channels, channels, 3, 1, 0, bias=use_bias),
                      ILN(channels), nn.PReLU(num_parameters=1)]
        
   
        Res_block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(channels, channels, 3, 1, 0, bias=use_bias),
                      ILN(channels)]
        self.Res_block = nn.Sequential(*Res_block)
        # self.w = nn.Parameter(torch.ones(3))

        # print(type(self.wtconv))
        # self.i = 0

    def forward(self, x):
        # w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        # w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        # w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))
        # self.i+=1
        # if self.i%100==0:
        #     print(w1,w2,w3)
        #     self.i=0
        return  x + self.Res_block(x)
