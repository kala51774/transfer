from torch import nn

from Block.Block import Block


class Discriminator_S(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[32, 64, 128, 256]):
        super().__init__()

        self.model = nn.Sequential(

            # k3n32s2
            Block(in_channels, features[0], kernel_size=3, stride=2, padding=1),  # b,32,128,128
            # k3n32s1
            Block(features[0], features[0], kernel_size=3, stride=1, padding=1),  # b,32,128,128

            # k3n64s2
            Block(features[0], features[1], kernel_size=3, stride=2, padding=1),  # b,64,64,64
            # k3n64s1
            Block(features[1], features[1], kernel_size=3, stride=1, padding=1),  # b,64,64,64

            # k3n128s2
            Block(features[1], features[2], kernel_size=3, stride=2, padding=1),  # b,128,32,32
            # k3n128s1
            Block(features[2], features[2], kernel_size=3, stride=1, padding=1),  # b,128,32,32

            # k1n1s1
            Block(features[2], out_channels, kernel_size=1, stride=1, padding=0, act=False)  # b,3,32,32
        )

    def forward(self, x):
        x = self.model(x)
        return x
