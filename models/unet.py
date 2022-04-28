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
        # Encode (N, C, H, W)
        x = self.down1(x)
        skip1 = x  # (N, 64, 480, 640)

        x = self.pool(x)  # Downscaling to 240x320
        x = self.down2(x)
        skip2 = x  # (N, 128, 240, 320)

        x = self.pool(x)
        x = self.down3(x)
        skip3 = x  # (N, 256, 120, 160)

        x = self.pool(x)
        x = self.down4(x)
        skip4 = x  # (N, 512, 60, 80)

        x = self.pool(x)
        x = self.down5(x)  # (N, 1024, 30, 40)

        # Decode (N, C, H, W)
        x = self.trans1(x)  # (N, 512, 60, 80)
        x = torch.cat([x, skip4], dim=1)  # (N, (512+512), 60, 80)
        x = self.up1(x)  # (N, 512, 60, 80)

        x = self.trans2(x)  # (N, 256, 120, 160)
        x = torch.cat([x, skip3], dim=1)  # (N, (256+256), 120, 160)
        x = self.up2(x)  # (N, 256, 120, 160)

        x = self.trans3(x)  # (N, 128, 240, 320)
        x = torch.cat([x, skip2], dim=1)  # (N, (128+128), 240, 320)
        x = self.up3(x)  # (N, 128, 240, 320)

        x = self.trans4(x)  # (N, 64, 480, 640)
        x = torch.cat([x, skip1], dim=1)  # (N, (64+64), 480, 640)
        x = self.up4(x)  # (N, 64, 480, 640)

        x = self.fc(x)  # (N, 1, 480, 640)
        x = nn.Sigmoid(x)
        return x
