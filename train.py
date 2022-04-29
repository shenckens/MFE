import torch
from torch.utils.data import DataLoader
import pytorch_ssim
import numpy as np
# from models.img_denoising import DenoisingAutoencoder
from datasets.testset_neucon_depths import TestsetNeuconDepths
from datasets.neucon_depths import NeuconDepths
from models.unet import *
from utils import fill_recon_img
import os
import argparse

# Goes in config file at later stage
datapath = '/project/henckens/data/scannet'


def evaluate(model, recon_img, gt_img, mask, epoch):

    input = fill_recon_img(recon_img, gt_img, mask)
    input = torch.unsqueeze(input, dim=1)
    input = input.to(device=device, dtype=torch.float)
    gt_img = torch.unsqueeze(gt_img, dim=1)
    gt_img = gt_img.to(device=device, dtype=torch.float)

    # forward pass
    output = model(input)

    # calculate loss
    loss = loss_module(output, gt_img)

    return loss


if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.001,
                        help='The learning-rate used for training the model.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs used for training.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Number of images in a batch.')
    parser.add_argument('--decay', type=float, default=0.01,
                        help='The weight decay (L2 reguralization) used for the model optimizer.')
    parser.add_argument('--base_channel_size', type=int, default=64,
                        help='The size of the first (base) amount of convolutional filters, uses multiples of this number in deeper layers.')
    parser.add_argument('--zclip', type=float, default=False,
                        help='The maximum value (in meters) from which the depth is not counted and set to 0.')
    parser.add_argument('--loss_fn', type=str, default='mse',
                        help="The loss function used (either 'mse', 'ssim' or 'l1')")
    parser.add_argument('--save_model', type=bool, default=True,
                        help='Boolean to indicate if the model parameters should be saved to disk after every epoch.')

    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device.')

    # Data and DataLoaders
    # train_data = NeuconDepths('./Desktop/data', 'test') # for local testing
    train_data = TestsetNeuconDepths(datapath, 'train', zclip=args.zclip)
    val_data = TestsetNeuconDepths(datapath, 'val', zclip=args.zclip)
    train_dl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    # Model and optimizer
    model = Unet(args.base_channel_size)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Loss module
    if args.loss_fn == 'l1':
        loss_module = nn.L1Loss()
    elif args.loss_fn == 'mse':
        loss_module = nn.MSELoss()
    elif args.loss_fn == 'ssim':
        loss_module = pytorch_ssim.SSIM()

    train_loss = []
    val_loss = []
    print(f'Training...')
    for epoch in range(args.epochs):
        losses = []
        print(f'Epoch {epoch+1}/{args.epochs}')
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
            if args.loss_fn == 'ssim':
                loss = 1 - loss_module(output, gt_img)
            else:
                loss = loss_module(output, gt_img)
            print(loss.item())
            losses.append(loss.item())

            # backward pass, optimizer step.
            loss.backward()
            optimizer.step()

            i += 1
            if i % 100 == 0:
                print(f'Completed {i}/{len(train_dl)} iterations\
                      on epoch {epoch+1}/{args.epochs}')
                print(f'Loss: {loss}')

        train_loss.append(losses.mean())

        # Validation
        losses_val = []
        model.eval()
        for recon_img, gt_img, mask in val_dl:
            loss_val = evaluate(model, recon_img, gt_img, mask, epoch)
            losses_val.append(loss_val.item())
        val_loss.append(losses_val.mean())

        print(f'Completed epoch {epoch+1}.')
        print(f'Training loss: {train_loss}')
        print(f'Validation loss: {val_loss}')

        # Saving model so far.
        if args.save_model:
            torch.save(model.state_dict(), './saved_parameters/{}_epoch{}_lr{}_bs{}_zclip{}.pt'.format(
                model.__class__.__name__, epoch+1, args.lr, args.batch_size, args.zclip))

    print(f'Train loss per epoch {train_loss}')
    print(f'Validation loss per epoch {val_loss}')
