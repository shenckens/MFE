
import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):

    def __init__(self, input_channels: int, base_channel_size: int, output_dim: int, act_fn: object = nn.ReLu, stride=1):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For depth imgs, this parameter is 1
            - base_channel_size : Number of channels we use in the first conv layers. Deeper layers might use a duplicate of it.
            - output_dim : Dimensionality of output layer
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, c_hid, kernel_size=3,
                      padding=1, stride=1),  # 640x480 => 640x480
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1, stride=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=5),  # 320x160 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=5),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=5),  # 8x8 => 4x4
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(4*4*2*c_hid, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential()
        pass

    def forward(self, x):
        return self.decoder(x)


class ImgRecon(nn.Module):

    def __init__(self):
        super(ImgRecon, self).__init__()
        pass

    def forward(self, x):
        pass
