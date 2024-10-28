# 定义生成器
from torch import nn

from Decoder.GenDecoder import GenDecoder
from model.GenEncoder import GenEncoder


class Generator(nn.Module):
    def __init__(self, n_domian=None, E_block=5, D_block=4):
        super(Generator, self).__init__()
        # 编码器
        self.Encoder = [GenEncoder(n_block=E_block)]
        self.Encoder = nn.Sequential(*self.Encoder)
        # 解码器
        self.Decoder = [GenDecoder(n_block=D_block)]
        self.Decoder = nn.Sequential(*self.Decoder)

    def encoder(self, x):
        return self.Encoder(x)  # type: ignore

    def decoders(self, x):
        return self.Decoder(x)

    def forward(self, x):
        encode = self.encoder(x)
        return self.decoders(encode)

