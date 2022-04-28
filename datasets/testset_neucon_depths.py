
import os
import numpy as np
from torch.utils.data import Dataset
import PIL


class TestsetNeuconDepths(Dataset):
    def __init__(self, datapath, mode):
        super(TestsetNeuconDepths, self).__init__()
        self.mode = mode
        self.path = os.path.join(datapath, 'scans_test')
        self.all_scenes = sorted(
            [s for s in os.listdir(self.path) if not s.startswith('.')])
        self.train, self.val, self.test = np.split(self.all_scenes, [int(
            len(self.all_scenes)*0.8), int(len(self.all_scenes)*0.9)])
        assert self.mode in ['train', 'val', 'test']
        if mode == 'train':
            self.scenes = self.train
        elif mode == 'val':
            self.scenes = self.val
        else:
            self.scenes = self.test
        self.all_imgs = self.build_list()
        self.max_depth = 0.0

    def build_list(self):
        print(f'Building list of images...')
        all_imgs = []
        for i, scene in enumerate(self.scenes):
            if i % 10 == 0:
                print(f'Pre-processing scene {i}/{len(self.scenes)}')
            n_poses = len(os.listdir(os.path.join(
                self.path, scene, 'pose')))
            for n in range(n_poses):
                if os.path.isfile(os.path.join(self.path, scene, 'recon_max_depth', '{}.png'.format(n))):
                    idxs = scene, n
                    all_imgs.append(idxs)
        return all_imgs

    def compute_max_depth(self):
        print(f'Computing max depth value...')
        print(f'This can take a while')
        for scene, n in self.all_imgs:
            depth_val1 = np.load(os.path.join(
                self.path, scene, 'recon_max_depth', '{}.npy'.format(n))).max()
            depth_val2 = np.asarray(PIL.Image.open(os.path.join(
                self.path, scene, 'depth', '{}.png'.format(n)))).max() / 1000
            for val in [depth_val1, depth_val2]:
                if val > self.max_depth:
                    self.max_depth = val
        return self.max_depth

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        scene, n = self.all_imgs[idx]
        recon_depth = np.load(os.path.join(
            self.path, scene, 'recon_depth', '{}.npy'.format(n))) / 1
        gt_depth = np.asarray(PIL.Image.open(os.path.join(
            self.path, scene, 'depth', '{}.png'.format(n)))) / 1000
        mask = np.where(recon_depth > 0.0, 0.0, 1.0)
        return recon_depth, gt_depth, mask
