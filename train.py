import torch
from torch.utils.data import DataLoader
# from torch.ignite.metrics import SSIM
import pytorch_ssim
import numpy as np
# from models.img_denoising import DenoisingAutoencoder
from datasets.testset_neucon_depths import TestsetNeuconDepths
from datasets.neucon_depths import NeuconDepths
from models.unet import *
from utils import fill_recon_img
import os
import argparse

datapath = '/project/henckens/data/scannet'

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--lr', type=float, default=0.01,
                        help='The learning-rate used for training the model.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs used for training.')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Number of images in a batch.')
    parser.add_argument('--base_channel_size', type=int, default=64,
                        help='The size of the first (base) amount of convolutional filters, uses multiples of this number in deeper layers.')
    parser.add_argument('--zclip', type=float, default=False,
                        help='The maximum value (in meters) from which the depth is not counted and set to 0.')
    # parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer used during training.')

    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device.')

    # train_data = NeuconDepths('./Desktop/data', 'test')
    train_data = TestsetNeuconDepths(datapath, 'train', zclip=args.zclip)
    # valdata = TestsetNeuconDepths('./Desktop/data', 'val')
    train_dl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    model = Unet(args.base_channel_size)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Loss module
    ssim_loss = pytorch_ssim.SSIM()

    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs+1}')
        model.train()
        for recon_img, gt_img, mask in train_dl:
            # train batch
            input = fill_recon_img(recon_img, gt_img, mask)
            input = torch.unsqueeze(input, dim=1)
            print(f'Proceeding with data input of shape {input.shape}.')
            input.to(device)
            print(f'input is on {input.get_device()}')
            gt_img.to(device)
            print(f'gt is on {gt_img.get_device()}')
            print(f'model is on {model.get_device()}')

            # forward pass
            optimizer.zero_grad()
            print(f'Entering model forward pass.')
            output = model(input)

            # calculate loss
            print(f'Entering loss module.')
            loss = ssim_loss(output, gt_img)
            loss_value = loss.data[0]
            print(loss_value)

            # backward pass, optimizer step.
            loss.backward()
            optimizer.step()
            print(f'Completed one iteration')
        print(f'Completed one epoch')
        break
        model.eval()
        # do validation
        # update graph
