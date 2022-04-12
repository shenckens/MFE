
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class DenoiseDepths(Dataset):
    def __init__(self, datapath, mode):
        super(DenoiseDepths, self).__init__()
        self.datapath = datapath
        self.mode = mode
        assert self.mode in ['train', 'val', 'test']
        if mode == 'test':
            self.source_path = 'scans_test'
        else:
            self.source_path = 'scans'
        self.all_imgs = self.build_list()

    def build_list(self):
        all_imgs = []
        path = os.path.join(self.datapath, self.source_path)
        all_scenes = sorted(
            [s for s in os.listdir(path) if not s.startswith('.')])
        for n, scene in enumerate(all_scenes):
            if os.path.isfile(os.path.join(path, scene, 'noisy_depth', '{}.png'.format(n))):
                noisy_depth = np.asarray(plt.imread(os.path.join(
                    path, scene, 'noisy_depth', '{}.png'.format(n))))
                gt_depth = np.asarray(plt.imread(os.path.join(
                    path, scene, 'depth', '{}.png'.format(n))))
                imgs = (noisy_depth, gt_depth)
                all_imgs.append(imgs)
        return all_imgs

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        noisy_depth, gt_depth = self.all_imgs[idx]
        return noisy_depth, gt_depth
