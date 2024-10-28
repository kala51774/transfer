from torch import nn


class StyleEncoder(nn.Module):
    def __init__(self, img_channels=3, num_features=64, padding_mode="reflect", ):
        super().__init__()
        self.padding_mode = padding_mode

        self.initial_down = nn.Sequential(
            # k7n32s1
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3,
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

    def forward(self, x):
        style_code = self.down2(self.down1(self.initial_down(x)))

        return style_code