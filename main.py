# This is the main file, which will act as a testfile for now.

import numpy as np
import torch
import PIL
import matplotlib.pyplot as plt
from datasets.denoise_depths import DenoiseDepths
from torch.utils.data import DataLoader

testset = DenoiseDepths('./Desktop/data', 'test')
print(len(testset))
testdataloader = DataLoader(testset, batch_size=1, shuffle=True)

for noisy, gt in testdataloader:
    print(noisy.shape)
    print(noisy)
    print(gt.shape)
    print(gt)
    break
