#残差的SCT
import torch
from torch import nn

from Module.scconv import ScConv
from utils.funs import calc_cov, nor_mean, nor_mean_std


class new_neck_v2(nn.Module):
    def __init__(self):
        super(new_neck_v2, self).__init__()

        # self.net = nn.Sequential(nn.Conv2d(256,128,1,1,0),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(128,32,1,1,0))
        self.net =ScConv(256)
        self.uncompress = nn.Conv2d(32,256,1,1,0)
        self.sm = nn.Softmax(dim=-1)
        chanle = 256

        self.bn_share =nn.BatchNorm2d(chanle)

    def calc_mean(self,feat):
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean

    def forward(self, content, style=None,noise=None,init=False):

        if init:
            # print("content.shape:", content.shape)

            cF_nor = nor_mean_std(content)

            # cF_nor = self.bn_style(content)
            cF = self.net(cF_nor)
            # cF = self.uncompress(cF)
            cF = cF +content
            # print("cf.shape:", cF.shape)
            return cF

        else:
            # print("content.shape:", content.shape,"style.shape:",style.shape)
            # cF_nor = nor_mean_std(content)

            sF_nor, smean = nor_mean(style)
            # smean  = self.calc_mean(style)
            cF_nor = self.bn_share(content)
            sF_nor = self.bn_share(style)

            cF = self.net(cF_nor)
            sF = self.net(sF_nor)
            b, c, w, h = cF.size()
            s_cov = calc_cov(sF)
            b1, c1, hw = s_cov.size()
            s_cov = self.sm(s_cov) * int(c1) ** (-0.5)#test
            gF = torch.bmm(s_cov, cF.flatten(2, 3)).view(b,c,w,h)
            # gF = self.uncompress(gF)
            if noise==None:
                gF = gF + smean.expand(cF_nor.size())+content
            else:
                gF = gF + smean.expand(cF_nor.size())+content
            # print("gF.shape",gF.shape)
            return gF

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    x = torch.randn(3,256,64,64)
    # print(x.shape)
    model= new_neck_v2()
    print(model(x,x).shape)
    print(count_parameters(model))
