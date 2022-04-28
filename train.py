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
    parser.add_argument('--loss_fn', type=str, default='l1',
                        help="The loss function used (either 'ssim' or 'l1')")
    # weight decay?
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device.')

    # train_data = NeuconDepths('./Desktop/data', 'test')
    train_data = TestsetNeuconDepths(datapath, 'train', zclip=args.zclip)
    # valdata = TestsetNeuconDepths('./Desktop/data', 'val', zclip=args.zclip)
    train_dl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    model = Unet(args.base_channel_size)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Loss module
    if args.loss_fn == 'l1':
        loss_module = nn.L1Loss()
    elif arg.loss_fn == 'ssim':
        loss_module = pytorch_ssim.SSIM()

    epoch_loss = []
    for epoch in range(args.epochs):
        losses = []
        print(f'Epoch {epoch+1}/{args.epochs+1}')
        model.train()
        i = 0
        for recon_img, gt_img, mask in train_dl:
            # train batch
            input = fill_recon_img(recon_img, gt_img, mask)
            input = torch.unsqueeze(input, dim=1)
            input = input.to(device=device, dtype=torch.float)
            gt_img = torch.unsqueeze(gt_img, dim=1)
            gt_img = gt_img.to(device=device, dtype=torch.float)

            # forward pass
            optimizer.zero_grad()
            output = model(input)

            # calculate loss
            loss = loss_module(output, gt_img)
            losses.append(loss)

            # backward pass, optimizer step.
            loss.backward()
            optimizer.step()

            i += 1
            if i % 100 == 0:
                print(f'Completed {i}/{len(train_dl)} iterations\
                      on epoch {epoch+1}/{args.epochs+1}')

        epoch_loss.append(losses.mean())
        print(f'Completed one epoch')
        break
        model.eval()
        # do validation
        # update graph

    print(epoch_losses)
