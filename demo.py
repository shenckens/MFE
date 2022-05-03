import torch
import numpy as np
from models.unet import *
import matplotlib.pyplot as plt
import os
import argparse

# Goes in config file at later stage
saved_parameters = 'project/henckens/saved_parameters'

# SSIM loss parameters
window_size = 11
reduction = 'mean'  # 'none', 'mean', 'sum'


def plot_input_output(input, output):
    return plt.show()


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

    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device.')

    # Checking if state dict exists.
    state_dict_path = os.path.join(saved_parameters, args.state_dict)
    if not os.path.isfile(state_dict_path):
        raise ValueError(
            f'The state_dict at {state_dict_path} does not exist.')

    # Setting the model
    model = Unet(args.base_channel_size)
    model.load_state_dict(os.path.join(saved_parameters, args.state_dict))
    model = model.to(device)
    model.eval()

    # TODO: process input image.
    output = model(input)
