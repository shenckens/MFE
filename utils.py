import numpy as np
import torch
import torch.nn as nn

# Loss functions


def SSIM_loss(x, y):
    loss = 0
    return loss


def L1(x, y):
    loss = nn.L1Loss(x, y)
    return loss


def L2(x, y):
    loss = 0
    return loss


def BCE(x, y):
    loss = 0
    return loss


def fill_recon_img(recon_img, gt_img, mask, zclip=False):
    img = np.where(mask, gt_img, recon_img)
    if zclip:
        img = np.where(img > zclip, 0.0, img)
    return torch.from_numpy(img)
