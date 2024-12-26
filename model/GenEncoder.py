import torch
from torch import nn

from Block.ResBlock import ResBlock
from model.ILN import ILN
from WTConv.wtconv import WTConv2d

class GenEncoder(nn.Module):
    def __init__(self, hw=64, n_block=None,norm=nn.InstanceNorm2d, use_bias=False):
        super().__init__()
        # 平面卷积
        model = []
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(3, hw, 7, 1, 0, bias=use_bias),

                  ]
        # 下采样
        down = 2
        for i in range(down):
            mult = 2 ** i
            model += [
                nn.Conv2d(hw * mult, hw * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    #   WTConv2d(hw * mult * 2),
                     ILN(hw * mult * 2), nn.ReLU(True)]
            # 残差块
        res = hw * 4
        for j in range(n_block):
            model += [ResBlock(res)]
            
        self.model = nn.Sequential(*model)
        self.wtconv = nn.Sequential(WTConv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,bias=use_bias,wt_levels=4),
       )
        self.w = nn.Parameter(torch.ones(2))
        self.i=0
    def forward(self, x):
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        self.i+=1
        if self.i%100==0:
            print(w1,w2)
            self.i=0
        t= self.model(x)
        return t*w1+self.wtconv(t)*w2  # 1,256,64,64

if __name__ == '__main__':
    x= torch.rand(3,3,256,256)
    model = GenEncoder(n_block=5)
    out = model(x)
    # print(x==out)
    print(out.shape)
