
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import PIL


class NeuconDepths(Dataset):
    def __init__(self, datapath, mode):
        super(NeuconDepths, self).__init__()
        self.datapath = datapath
        self.mode = mode
        assert self.mode in ['train', 'val', 'test']
        if mode == 'test':
            self.source_path = 'scans_test'
        else:
            self.source_path = 'scans'
        self.max_depth = 0.0
        self.all_imgs = self.build_list()

    def build_list(self):
        print(f'Building list of images...')
        all_imgs = []
        path = os.path.join(self.datapath, self.source_path)
        all_scenes = sorted(
            [s for s in os.listdir(path) if not s.startswith('.')])
        for i, scene in enumerate(all_scenes):
            print(f'scene {i+1}/{len(all_scenes)}')
            n_poses = len(os.listdir(os.path.join(
                path, scene, 'pose')))
            for n in range(n_poses):
                if os.path.isfile(os.path.join(path, scene, 'recon_max_depth', '{}.png'.format(n))):
                    idxs = scene, n
                    all_imgs.append(idxs)
                    depth_val1 = np.load(os.path.join(
                        path, scene, 'recon_depth', '{}.npy'.format(n))).max()
                    depth_val2 = np.asarray(PIL.Image.open(os.path.join(
                        path, scene, 'depth', '{}.png'.format(n)))).max() / 1000
                    for val in [depth_val1, depth_val2]:
                        if val > self.max_depth:
                            self.max_depth = val

            # break

        print(self.max_depth)
        return all_imgs

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        path = os.path.join(self.datapath, self.source_path)
        scene, n = self.all_imgs[idx]
        color = np.asarray(PIL.Image.open(os.path.join(
            path, scene, 'color', '{}.jpg'.format(n))))
        recon_depth = np.load(os.path.join(
            path, scene, 'recon_depth', '{}.npy'.format(n))) / 3
        gt_depth = np.asarray(PIL.Image.open(os.path.join(
            path, scene, 'depth', '{}.png'.format(n)))) / 3000
        mask = np.where(recon_depth > 0.0, False, True)
        # return color, recon_depth, gt_depth, mask
        return recon_depth, gt_depth, mask
