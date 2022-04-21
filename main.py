# This is the main file, which will act as a testfile for now.

import numpy as np
import torch
import PIL
import matplotlib.pyplot as plt
from datasets.neucon_depths import NeuconDepths
from torch.utils.data import DataLoader

testset = NeuconDepths('./Desktop/data', 'test')
print(len(testset))
testdataloader = DataLoader(testset, batch_size=1, shuffle=True)

for noisy, gt in testdataloader:
    print(noisy.shape)
    print(noisy)
    print(gt.shape)
    print(gt)
    plt.subplot(1, 2, 1)
    plt.imshow(noisy[0])
    plt.subplot(1, 2, 2)
    plt.imshow(gt[0])
    plt.show()
    break
