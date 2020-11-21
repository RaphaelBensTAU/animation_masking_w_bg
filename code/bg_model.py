import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d, Hourglass

class BgRefinementNetwork(torch.nn.Module):
    def __init__(self):
        super(BgRefinementNetwork, self).__init__()
        self.hourglass = Hourglass(block_expansion=64, in_features=11, max_features=1024, num_blocks=5)
        self.final_hourglass = nn.Conv2d(in_channels=self.hourglass.out_filters, out_channels=3, kernel_size=(7, 7),padding=(3, 3))

    def forward(self, x):
        out = self.hourglass(x)
        out = self.final_hourglass(out)
        return torch.sigmoid(out)

class BackgroundGenerator(torch.nn.Module):
    def __init__(self, num_channels, block_expansion, max_features, num_down_blocks, num_bottleneck_blocks, th):
        super(BackgroundGenerator, self).__init__()

        self.first = SameBlock2d(8, 256, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))
        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))

    def forward(self, x):
        out = self.first(x)
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        return torch.sigmoid(out)
    