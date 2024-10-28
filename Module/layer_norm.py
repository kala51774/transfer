from torch import nn
from torch.utils._contextlib import F


class layer_norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.layer_norm(x, x.size()[1:])