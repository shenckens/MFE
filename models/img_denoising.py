import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # encoder
        self.enc1 = nn.Conv2d(1, 512, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        # decoder
        self.dec1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec2 = nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2)
        self.dec3 = nn.ConvTranspose2d(128, 256, kernel_size=2, stride=2)
        self.dec4 = nn.ConvTranspose2d(256, 512, kernel_size=2, stride=2)
        self.out = nn.Conv2d(512, 1, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # encode
        x = F.relu(self.enc1(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = self.bn2(x)
        x = self.pool(x)
        x = F.relu(self.enc3(x))
        x = self.bn3(x)
        x = self.pool(x)
        x = F.relu(self.enc4(x))
        x = self.bn4(x)
        x = self.pool(x)  # the latent space representation

        # decode
        x = F.relu(self.dec1(x))
        x = (self.bn4(x))
        x = F.relu(self.dec2(x))
        x = (self.bn3(x))
        x = F.relu(self.dec3(x))
        x = (self.bn2(x))
        x = F.relu(self.dec4(x))
        x = (self.bn1(x))
        x = torch.sigmoid(self.out(x))
        return x
