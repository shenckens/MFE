import numpy as np
import torch


def fill_recon_img(recon_img, gt_img, mask):
    img = np.where(mask, gt_img, recon_img)
    return torch.from_numpy(img)


def stack_patch(patch, img):
    ''' Takes a mask img of a planar patch and original image.
        Returns the stacked points of the planar patch as a numpy array.
    '''
    pts = []
    for i in range(len(img)):
        for j in range(len(img[0])):
            if patch[i][j]:
                pts.append((i, j, img[i][j]))
    A = np.stack(pts, axis=0)
    return A


def calc_normal(A):
    '''An = b. Where A is a stacked matrix of 3d points in a patch,
       n is the normal vector and b a 3d vector of ones.
    '''
    b = np.ones((len(A), 1))
    n = np.linalg.inv(A.T @ A) @ A.T @ b
    n /= np.linalg.norm(n, ord=2)
    return n


def MPL(patch, img, img_gt):
    '''Computes the Mean Planar Loss for one single planar patch
       in a current and corresponding ground truth image.
    '''
    A, A_gt = stack_patch(patch, img), stack_patch(patch, img_gt)
    n, n_gt = calc_normal(A), calc_normal(A_gt)
    loss = np.linalg.norm(n - n_gt, ord=1)
    return loss
