import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_filters),
            nn.ReLU(True),
            nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_filters),
            nn.ReLU(True)
            )

    def forward(self, x):
        return self.block(x)


class TransposeConv(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(TransposeConv, self).__init__()
        self.trans = nn.ConvTranspose2d(in_filters, out_filters, kernel_size=3,
                                        output_padding=1, padding=1, stride=2)

    def forward(self, x):
        return self.trans(x)


class Unet(nn.Module):
    def __init__(self, base_channel_size):
        super(Unet, self).__init__()
        self.down1 = ConvBlock(1, base_channel_size)
        self.down2 = ConvBlock(base_channel_size, 2*base_channel_size)
        self.down3 = ConvBlock(2*base_channel_size, 4*base_channel_size)
        self.down4 = ConvBlock(4*base_channel_size, 8*base_channel_size)
        self.down5 = ConvBlock(8*base_channel_size, 16*base_channel_size)
        self.pool = nn.MaxPool2d(2)
        self.trans1 = TransposeConv(16*base_channel_size, 8*base_channel_size)
        self.up1 = ConvBlock(16*base_channel_size, 8*base_channel_size)
        self.trans2 = TransposeConv(8*base_channel_size, 4*base_channel_size)
        self.up2 = ConvBlock(8*base_channel_size, 4*base_channel_size)
        self.trans3 = TransposeConv(4*base_channel_size, 2*base_channel_size)
        self.up3 = ConvBlock(4*base_channel_size, 2*base_channel_size)
        self.trans4 = TransposeConv(2*base_channel_size, base_channel_size)
        self.up4 = ConvBlock(2*base_channel_size, base_channel_size)
        self.fc = nn.Conv2d(base_channel_size, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Encode (w,h,c)
        x = self.down1(x)
        skip1 = x  # 640x480x64

        x = self.pool(x)  # Downscaling to 320x240
        x = self.down2(x)
        skip2 = x  # 320x240x128

        x = self.pool(x)
        x = self.down3(x)
        skip3 = x  # 160x120x256

        x = self.pool(x)
        x = self.down4(x)
        skip4 = x  # 80x60x512

        x = self.pool(x)
        x = self.down5(x)  # 40x30x1024

        # Decode (w,h,c)
        x = self.trans1(x)  # 80x60x512
        x = torch.cat([x, skip4], dim=1)  # 80x60x(512+512)
        x = self.up1(x)  # 80x60x512

        x = self.trans2(x)  # 160x120x256
        x = torch.cat([x, skip3], dim=1)  # 160x120x(256+256)
        x = self.up2(x)  # 160x120x256

        x = self.trans3(x)  # 320x240x128
        x = torch.cat([x, skip2], dim=1)  # 320x240x(128+128)
        x = self.up3(x)  # 320x240x128

        x = self.trans4(x)  # 640x480x64
        x = torch.cat([x, skip1], dim=1)  # 640x480x(64+64)
        x = self.up4(x)  # 640x480x64

        x = self.fc(x)  # 640x480x1

        return F.sigmoid(x)
