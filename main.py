# This is the main file, which will act as a testfile for now.

import numpy as np
import torch
import PIL
import matplotlib.pyplot as plt
from datasets.neucon_depths import NeuconDepths
from datasets.testset_neucon_depths import TestsetNeuconDepths
from torch.utils.data import DataLoader

data = NeuconDepths('./Desktop/data', 'test')
# data = TestsetNeuconDepths('./Desktop/data', 'train')
print(len(data))
dataloader = DataLoader(data, batch_size=1, shuffle=True)

for color, noisy, gt, mask in dataloader:
    print(noisy.shape)
    print(noisy[0])
    print(gt.shape)
    print(torch.max(gt[0]))
    print(torch.max(noisy[0]))
    gt3 = np.where(gt[0] > 1.0, 0.0, gt[0])
    plt.subplot(3, 3, 1)
    plt.imshow(noisy[0])
    plt.title('recon')
    plt.subplot(3, 3, 3)
    plt.imshow(gt3)
    plt.title('gt z=3.0')
    plt.subplot(3, 3, 2)
    plt.imshow(gt[0])
    plt.title('gt')
    plt.subplot(3, 3, 4)
    plt.imshow(np.where(mask[0], gt3, noisy[0]))
    plt.title('filled_in_recon')
    plt.subplot(3, 3, 5)
    plt.imshow(mask[0])
    plt.title('mask')
    plt.subplot(3, 3, 6)
    plt.imshow(np.where(mask[0], 0.0, gt[0]))
    plt.title('masked_gt')
    plt.subplot(3, 3, 7)
    plt.imshow(color[0])
    plt.title('color')
    plt.tight_layout()
    plt.show()
    break
