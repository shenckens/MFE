import torch
from torch.utils.data import DataLoader
import numpy as np
from models.unet import *
from utils import fill_recon_img
from datasets.testset_neucon_depths import TestsetNeuconDepths
import PIL.Image as Image
import matplotlib.pyplot as plt
import os
import argparse

# Goes in config file at later stage
datapath = '/project/henckens/data/scannet'
saved_parameters = '/project/henckens/saved_parameters'
# datapath = '../../Desktop/data'
# saved_parameters = '../../Desktop/saved_parameters'


def plot_input_output(input, output, gt):
    plt.suptitle(f'{args.scene} img{args.i}\n model={args.state_dict}')
    plt.subplot(2, 2, 1)
    plt.imshow(input)
    plt.title('Input image')
    plt.subplot(2, 2, 2)
    plt.imshow(output)
    plt.title('Output image')
    plt.subplot(2, 2, 3)
    plt.imshow(Image.open(os.path.join(
        datapath, 'scans_test', args.scene, 'color', '{}.jpg'.format(args.i))))
    plt.title('Color image')
    plt.subplot(2, 2, 4)
    plt.imshow(gt)
    plt.title('Ground-truth image')
    plt.tight_layout(pad=2)
    return plt.show()


def process_input_img(scene, n, zclip=False):
    # Takes .npy file as input image.
    recon_depth = np.load(os.path.join(
        datapath, 'scans_test', scene, 'recon_max_depth', '{}.npy'.format(n))) / 1
    gt_depth = np.asarray(Image.open(os.path.join(
        datapath, 'scans_test', scene, 'depth', '{}.png'.format(n)))) / 1000
    if not zclip:
        # Maximum depth value found in ScanNet test set.
        zclip = 12.211601257324219
        print(f'Set zclip to {zclip}')
    recon_depth /= zclip
    recon_depth = torch.from_numpy(
        np.where(recon_depth > 1.0, 0.0, recon_depth)).unsqueeze(dim=0)
    gt_depth /= zclip
    gt_depth = torch.from_numpy(
        np.where(gt_depth > 1.0, 0.0, gt_depth)).unsqueeze(dim=0)

    mask = torch.from_numpy(
        np.where(recon_depth > 0.0, 0.0, 1.0))
    print(recon_depth.shape, gt_depth.shape, mask.shape)
    return recon_depth, gt_depth, mask


if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=bool, default=False,
                        help="Boolean value that acts as a swicht for input given or not.")
    parser.add_argument('--scene', type=str, default='800_00',
                        help='The scene where the input image comes from.')
    parser.add_argument('--i', type=int, default=149,
                        help='The index of the image from the given scene.')
    parser.add_argument('--state_dict', type=str, default=None,
                        help="Filename of model's saved state dict.")
    parser.add_argument('--visualize', type=bool, default=True,
                        help='Visualize the input and output images in a single plot.')
    parser.add_argument('--zclip', type=float, default=False,
                        help='The maximum value (in meters) from which the depth is not counted and set to 0.')
    parser.add_argument('--fill_imgs', type=bool, default=False,
                        help='Boolean value to determine if holes in recon images should be filled up with pixels from the gt images.')
    parser.add_argument('--base_channel_size', type=int, default=64,
                        help='The size of the first (base) amount of convolutional filters, uses multiples of this number in deeper layers.')

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
    if device == torch.device('cpu'):
        model.load_state_dict(torch.load(
            os.path.join(saved_parameters, args.state_dict), map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(
            os.path.join(saved_parameters, args.state_dict)))
    model = model.to(device)
    model.eval()

    if not args.input:
        # Data and DataLoader
        test_data = TestsetNeuconDepths(datapath, 'test', zclip=args.zclip)
        test_dl = DataLoader(test_data, batch_size=1, shuffle=True)
        # Get single sample.
        recon_img, gt_img, mask = next(iter(test_dl))

    else:
        # Process input image to (1, H, W)
        recon_img, gt_img, mask = process_input_img(
            args.scene, args.i, args.zclip)

    if args.fill_imgs:
        input = fill_recon_img(recon_img, gt_img, mask)
    else:
        input = recon_img

    input = torch.unsqueeze(input, dim=1)
    input = input.to(device=device, dtype=torch.float)

    print("Feeding input through the model...")
    output = model(input)

    input = input.squeeze(dim=0)
    output = output.squeeze(dim=0)

    print(input.shape, output.shape, gt_img.shape)

    if args.visualize:
        if (output.shape == gt_img.shape):
            plot_input_output(input[0].detach().numpy(
            ), output[0].detach().numpy(), gt_img[0].detach().numpy())
        else:
            print('The shapes of the images are not equal.')

    # TODO: Save output image somewhere.
    # plt.imsave(f'./outputs/{args.state_dict}/{args.scene}/{args.i}', output)
