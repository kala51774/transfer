from torch import nn


class content_struct(nn.Module):
    def __init__(self):
        super(content_struct, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(128 * 16 * 16, 512)
#         self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
#         x = nn.functional.relu(self.conv1(x))
#         x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
#         x = x.view(-1, 128 * 16 * 16)
#         x = nn.functional.relu(self.fc1(x))
#         x = self.fc2(x)
        return x

