import trimesh
import pyrender
import json
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import PIL
import os
# for no screen rendering
os.environ['PYOPENGL_PLATFORM'] = 'egl'


PATH = '/project/henckens/data/scannet/scans_test'  # put in config at later stage
RECONPATH = './results/scene_scannet_checkpoints_fusion_eval_47'

# PATH = './Desktop/data/scannet/scans_test'
# RECONPATH = './Desktop'


def load_recon_mesh(scene):
    recon_mesh = o3d.io.read_triangle_mesh(
        os.path.join(RECONPATH, '{}.ply'.format(scene)))  # Read the reconstructed mesh
    return recon_mesh


def load_recon_pcd(scene):
    recon_pcd = o3d.io.read_point_cloud(
        os.path.join(RECONPATH, '{}.ply'.format(scene)))  # Read the reconstructed pcd
    return recon_pcd


def offscreen_render(scene):

    scene_path = os.path.join(PATH, scene)
    scene_info = dict(np.loadtxt(
        f'{os.path.join(scene_path, scene)}.txt', delimiter=' = ', dtype=dict))

    # pyrender_path = os.path.join(scene_path, 'recon_max_depth')
    # if not os.path.exists(pyrender_path):
    #     os.makedirs(pyrender_path)

    pcd_path = os.path.join(scene_path, 'pcd_depth')
    if not os.path.exists(pcd_path):
        os.makedirs(pcd_path)

    n_poses = int(scene_info['numDepthFrames'])
    width = int(scene_info['depthWidth'])
    height = int(scene_info['depthHeight'])
    fx = float(scene_info['fx_depth'])
    fy = float(scene_info['fy_depth'])
    cx = float(scene_info['mx_depth'])
    cy = float(scene_info['my_depth'])
    intrinsic_matrix = np.genfromtxt(os.path.join(
        scene_path, 'intrinsic/intrinsic_depth.txt'))

    # Load mesh and set correct campose.
    mesh = trimesh.load_mesh(os.path.join(
        RECONPATH, '{}.ply'.format(scene)))
    print(mesh)
    # mesh = pyrender.Mesh.from_trimesh(mesh)
    rotation_x = np.array([[1,  0,  0, 0],
                           [0, -1,  0, 0],
                           [0,  0, -1, 0],
                           [0,  0,  0, 1]])

    for n in range(n_poses):
        if n % 100 == 0:
            print(f'{n}/{n_poses}')

        # Check if index exists in folder.
        if os.path.isfile(os.path.join(scene_path, 'pose', '{}.txt'.format(n))):
            extrinsic_matrix = np.genfromtxt(os.path.join(
                scene_path, 'pose', '{}.txt'.format(n)))
            if np.isfinite(extrinsic_matrix).all():

                renderer = pyrender.OffscreenRenderer(width, height)
                renderer.viewport_height = height
                renderer.viewport_width = width
                Scene = pyrender.Scene()
                Scene.clear()
                Scene.add(mesh)
                # NeuralRecon is limited to a depth of 3 meters.
                cam = pyrender.IntrinsicsCamera(
                    cx=cx, cy=cy, fx=fx, fy=fy)
                Scene.add(cam, pose=extrinsic_matrix@rotation_x)
                flags = pyrender.constants.RenderFlags.DEPTH_ONLY

                # Returns redered depth in meters.
                depth = np.asarray(renderer.render(
                    Scene, flags=flags)).astype(np.float32)
                renderer.delete()

                # Save depth as numpy array and image.
                # np.save(os.path.join(pyrender_path, '{}.npy'.format(n)), depth)
                # plt.imsave(os.path.join(pyrender_path,
                #                         '{}.png'.format(n)), depth)

                np.save(os.path.join(pcd_path, '{}.npy'.format(n)), depth)
                plt.imsave(os.path.join(pcd_path,
                                        '{}.png'.format(n)), depth)

            else:
                print(f'{scene}/pose/{n}.txt contains an invalid pose matrix.')

        else:
            print(
                f"File {os.path.join(scene_path, 'pose', '{}.txt'.format(n))} does not exist.")
    pass


if __name__ == "__main__":

    all_scenes = sorted([s for s in os.listdir(PATH) if not s.startswith('.')])

    for i, scene in enumerate(all_scenes):
        print(f'{scene}...')
        offscreen_render(scene)
        print(f'Done with {scene} ({i+1}/{len(all_scenes)})')
