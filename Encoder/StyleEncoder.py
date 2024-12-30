from torch import nn
from Module.PatchEmbed import PatchEmbed
from WTConv.wtconv import WTConv2d
import torch
class StyleEncoder(nn.Module):
    def __init__(self, img_channels=3, num_features=64, padding_mode="reflect", ):
        super().__init__()
        self.embed_dim=128
        self.img_size=512
        self.PatchEmbed = PatchEmbed(img_size=self.img_size,embed_dim=self.embed_dim)
        self.wtconv1 = nn.Sequential(WTConv2d(in_channels=num_features * 2,out_channels=num_features * 2,kernel_size=3,stride=1,bias=True,wt_levels=3))

        self.wtconv2 = nn.Sequential(WTConv2d(in_channels=num_features * 4,out_channels=num_features * 4,kernel_size=3,stride=1,bias=True,wt_levels=3))

        self.padding_mode = padding_mode

        self.initial_down = nn.Sequential(
            # k7n32s1
            nn.Conv2d(self.embed_dim, num_features, kernel_size=7, stride=1, padding=3,
                      padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # Down-convolution
        self.down1 = nn.Sequential(
            # k3n32s2
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=2, padding=1,
                      padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # k3n64s1
            nn.Conv2d(num_features, num_features * 2, kernel_size=3, stride=1, padding=1,
                      padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.down2 = nn.Sequential(
            # k3n64s2
            nn.Conv2d(num_features * 2, num_features * 2, kernel_size=3, stride=2, padding=1,
                      padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # k3n128s1
            nn.Conv2d(num_features * 2, num_features * 4, kernel_size=3, stride=1, padding=1,
                      padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.w_a = nn.Parameter(torch.ones(2))
        self.w_b = nn.Parameter(torch.ones(2))
        self.i=0


    def forward(self, x):
        
        wa_1 = torch.exp(self.w_a[0]) / torch.sum(torch.exp(self.w_a))
        wa_2 = torch.exp(self.w_a[1]) / torch.sum(torch.exp(self.w_a))

        wb_1 = torch.exp(self.w_b[0]) / torch.sum(torch.exp(self.w_b))
        wb_2 = torch.exp(self.w_b[1]) / torch.sum(torch.exp(self.w_b))
        
        # print(self.initial_down(x).shape)
        # print(x.shape)
        # torch.Size([1, 3, 512, 512])
        x = self.PatchEmbed(x)
        # torch.Size([1, 128, 64, 64])
        # print(x.shape)

        x = self.down1(self.initial_down(x))

        x = x * wa_1 + self.wtconv1(x) * wa_2
       
        x = self.down2(x)
        x = x * wb_1+ self.wtconv2(x) * wb_2
        
        self.i+=1
        if self.i%100==0:
            print("wa")
            print(wa_1,wa_2)
            print("wb")
            print(wb_1,wb_2)
            self.i=0

        # self.wtconv1
        # style_code = self.down2(self.down1())

        return x