import torch
from torch.utils.data import DataLoader
import torchgeometry as tgm
# import numpy as np
# from models.img_denoising import DenoisingAutoencoder
from datasets.testset_neucon_depths import TestsetNeuconDepths
# from datasets.neucon_depths import NeuconDepths
from models.unet import *
from utils import fill_recon_img
import os
import argparse

# Goes in config file at later stage
datapath = '/project/henckens/data/scannet'
saved_parameters = '/project/henckens/saved_parameters'

# SSIM loss parameters
window_size = 11
reduction = 'mean'  # 'none', 'mean', 'sum'


def evaluate(model, recon_img, gt_img, mask):

    if args.fill_imgs:
        input = fill_recon_img(recon_img, gt_img, mask)
    else:
        input = recon_img
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

    parser.add_argument('--batch_size', type=int, default=2,
                        help='Number of images in a batch.')
    parser.add_argument('--base_channel_size', type=int, default=64,
                        help='The size of the first (base) amount of convolutional filters, uses multiples of this number in deeper layers.')
    parser.add_argument('--zclip', type=float, default=False,
                        help='The maximum value (in meters) from which the depth is not counted and set to 0.')
    parser.add_argument('--loss_fn', type=str, default='ssim',
                        help="The loss function used (either 'mse', 'ssim' or 'l1')")
    parser.add_argument('--state_dict', type=str, default=None,
                        help="Filename of model's saved state dict.")

    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device.')

    # Checking if state dict exists.
    state_dict_path = os.path.join(saved_parameters, args.state_dict)
    if not os.path.isfile(state_dict_path):
        raise ValueError(
            f'The state_dict at {state_dict_path} does not exist.')

    # Data and DataLoader
    test_data = TestsetNeuconDepths(datapath, 'test', zclip=args.zclip)
    test_dl = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Setting the model
    model = Unet(args.base_channel_size)
    model.load_state_dict(os.path.join(saved_parameters, args.state_dict))
    model = model.to(device)
    model.eval()

    # Loss module
    if args.loss_fn == 'l1':
        loss_module = nn.L1Loss()
    elif args.loss_fn == 'mse':
        loss_module = nn.MSELoss()
    elif args.loss_fn == 'ssim':
        loss_module = tgm.losses.SSIM(window_size, reduction)

    test_loss = []
    print(f'Testing...')

    losses_test = []

    for recon_img, gt_img, mask, _ in test_dl:
        loss_test = evaluate(model, recon_img, gt_img, mask)
        losses_test.append(loss_test.item())

    test_loss.append(sum(losses_test)/len(losses_test))

    print(f'Testing loss using {args.loss_fn}-loss: {test_loss}')
