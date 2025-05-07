import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3),
                      padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConcatUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                     kernel_size=(3, 3), stride=(2, 2),
                                     padding=1, output_padding=1)
        self.block = Block(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x_cat = torch.cat([x2, x1], dim=1)
        x = self.block(x_cat)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_classes):
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_classes

        self.max_pool = nn.MaxPool2d(2)

        self.block1 = Block(in_channels, 8)
        self.block2 = Block(8, 16)
        self.block3 = Block(16, 32)
        self.block4 = Block(32, 64)
        self.up1 = ConcatUp(64, 32)
        self.up2 = ConcatUp(32, 16)
        self.up3 = ConcatUp(16, 8)
        self.outc = nn.Conv2d(8, out_classes, kernel_size=(1, 1))

    def forward(self, x):
        x1 = self.block1(x)
        xm1 = self.max_pool(x1)
        x2 = self.block2(xm1)
        xm2 = self.max_pool(x2)
        x3 = self.block3(xm2)
        xm3 = self.max_pool(x3)
        x4 = self.block4(xm3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        out = torch.sigmoid(logits)
        return out
