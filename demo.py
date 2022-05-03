import torch
from torch.utils.data import DataLoader
import numpy as np
from models.unet import *
from datasets.testset_neucon_depths import TestsetNeuconDepths
import matplotlib.pyplot as plt
import os
import argparse

# Goes in config file at later stage
datapath = '/project/henckens/data/scannet'
saved_parameters = 'project/henckens/saved_parameters'

# SSIM loss parameters
window_size = 11
reduction = 'mean'  # 'none', 'mean', 'sum'


def plot_input_output(input, output, filename):
    plt.subplot(1, 2, 1)
    plt.imread(input)
    plt.title('Input image')
    plt.subplot(1, 2, 2)
    plt.imread(output)
    plt.title('Output image')
    plt.tight_layout()
    return plt.show()


def process_input(input):
    recon_depth = np.load(input) / 1
    if args.zclip:
        recon_depth /= args.zclip
        recon_depth = np.where(recon_depth > 1.0, 0.0, recon_depth)

    return 0


if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None,
                        help="Path of input image to the model.")
    parser.add_argument('--state_dict', type=str, default=None,
                        help="Filename of model's saved state dict.")
    parser.add_argument('--visualize', type=bool, default=True,
                        help='Visualize the input and output images in a single plot.')
    parser.add_argument('--zclip', type=float, default=False,
                        help='The maximum value (in meters) from which the depth is not counted and set to 0.')
    parser.add_argument('--fill_imgs', type=bool, default=False,
                        help='Boolean value to determine if holes in recon images should be filled up with pixels from the gt images.')

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
    test_dl = DataLoader(test_data, batch_size=1, shuffle=True)

    # Setting the model
    model = Unet(args.base_channel_size)
    model.load_state_dict(os.path.join(saved_parameters, args.state_dict))
    model = model.to(device)
    model.eval()

    # TODO: process input image.
    # Image input size must be (1, 1, 480, 640).
    output = model(input)

    output = torch.squeeze(output, dim=1)
