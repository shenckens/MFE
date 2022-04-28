import torch
from torch.utils.data import DataLoader
# from torch.ignite.metrics import SSIM
import pytorch_ssim
import numpy as np
# from models.img_denoising import DenoisingAutoencoder
from datasets.testset_neucon_depths import TestsetNeuconDepths
from datasets.neucon_depths import NeuconDepths
from utils import *
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
    # parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer used during training.')

    args = parser.parse_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train_data = NeuconDepths('./Desktop/data', 'test')
    traindata = TestsetNeuconDepths(datapath, 'train')
    # valdata = TestsetNeuconDepths('./Desktop/data', 'val')
    train_dl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    model = DenoisingAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Loss module
    ssim_loss = pytorch_ssim.SSIM()

    for epoch in range(args.epochs):
        model.train()
        for recon_img, gt_img, mask in train_dl:
            # train batch
            input = fill_recon_img(
                recon_img, gt_img, mask, zclip=3.0).to(device)

            # forward pass
            optimizer.zero_grad()
            output = model(input)

            # calculate loss
            loss = ssim_loss(output, gt_img)
            loss_value = loss.data[0]
            print(loss_value)

            # backward pass, optimizer step.
            loss.backward()
            optimizer.step()
        break
        model.eval()
        # do validation
        # update graph
