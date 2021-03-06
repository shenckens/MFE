import trimesh
import pyrender
import json
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
# # for no screen rendering
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
# import scene_utils

PATH = '/project/henckens/data/scannet/scans_test'  # put in config at later stage
RECONPATH = './results/scene_scannet_checkpoints_fusion_eval_47'

# PATH = './Desktop/data/scannet/scans_test'
# RECONPATH = './Desktop'


def to_np(pointcloud) -> np.ndarray:
    return np.asarray(pointcloud.points)


def to_o3d(pointcloud: np.ndarray, pointcloud_normals: np.ndarray = None, pointcloud_colors: np.ndarray = None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    if pointcloud_normals:
        pcd.normals = o3d.utility.Vector3dVector(pointcloud_normals)
    if pointcloud_colors:
        pcd.colors = o3d.utility.Vector3dVector(pointcloud_colors)
    return pcd


def load_recon_mesh(scene):
    recon_mesh = o3d.io.read_triangle_mesh(
        os.path.join(RECONPATH, '{}.ply'.format(scene)))  # Read the reconstructed mesh
    return recon_mesh


def load_recon_pcd(scene):
    recon_pcd = o3d.io.read_point_cloud(
        os.path.join(RECONPATH, '{}.ply'.format(scene)))  # Read the reconstructed pcd
    return recon_pcd


def get_json_template():
    template = {
                  "class_name": "PinholeCameraParameters",
                  "extrinsic": [],
                  "intrinsic":
                  {
                     "height": 0,
                     "intrinsic_matrix": [],
                     "width": 0
                  },
                  "version_major": 1,
                  "version_minor": 0
                }
    return template


def make_pose_json(intrinsic_matrix, extrinsic_matrix, width, height, img_idx, path):
    '''3x3, 4x4, img_widht, img_height, pose_idx, path to o3d_parameters
    json file for each pose in a scene.
    '''
    intrinsic_matrix[0][2] = width / 2 - 0.5
    intrinsic_matrix[1][2] = height / 2 - 0.5
    f = get_json_template()
    f['extrinsic'] = np.linalg.inv(extrinsic_matrix).flatten('F').tolist()
    intrinsic = f['intrinsic']
    intrinsic['width'] = width
    intrinsic['height'] = height
    intrinsic['intrinsic_matrix'] = intrinsic_matrix[:3,
                                                     :3].flatten('F').tolist()
    with open(os.path.join(path, '{}.json'.format(img_idx)), 'w') as fp:
        json.dump(f, fp)
    pass


def headless_render(mesh, width, height, intrinsics, extrinsics):
    '''o3dmesh img_widht img_height img_index save_path'''
    renderer = o3d.visualization.rendering.OffscreenRenderer(
        width=width, height=height, headless=True)
    renderer.scene.add_model('mesh', mesh)

    pinhole = o3d.camera.PinholeCameraIntrinsic(
        width, height, fx=intrinsics[0, 0], fy=intrinsics[1, 1], cx=width / 2 - 0.5, cy=height / 2 - 0.5)

    renderer.setup_camera(pinhole, extrinsics)

    depth = renderer.render_to_depth_image(z_in_view_space=True)
    return depth


def make_noisy_depth(scene):
    mesh = load_recon_mesh(scene)
    pcd = load_recon_pcd(scene)
    scene_path = os.path.join(PATH, scene)
    scene_info = dict(np.loadtxt(
        f'{os.path.join(scene_path, scene)}.txt', delimiter=' = ', dtype=dict))
    json_path = os.path.join(scene_path, 'o3d_parameters')
    if not os.path.exists(json_path):
        os.makedirs(json_path)
    noisy_depth_path = os.path.join(scene_path, 'noisy_depth')
    if not os.path.exists(noisy_depth_path):
        os.makedirs(noisy_depth_path)
    noisy_pcd_path = os.path.join(scene_path, 'noisy_pcd')
    if not os.path.exists(noisy_pcd_path):
        os.makedirs(noisy_pcd_path)
    n_poses = int(scene_info['numDepthFrames'])
    width = int(scene_info['depthWidth'])
    height = int(scene_info['depthHeight'])
    intrinsic_matrix = np.genfromtxt(os.path.join(
        scene_path, 'intrinsic/intrinsic_depth.txt'))
    intrinsic_matrix[0][2] = width / 2 - 0.5
    intrinsic_matrix[1][2] = height / 2 - 0.5
    for n in range(n_poses):
        # Check if index exists in folder.
        if os.path.isfile(os.path.join(scene_path, 'pose', '{}.txt'.format(n))):
            extrinsic_matrix = np.genfromtxt(os.path.join(
                scene_path, 'pose', '{}.txt'.format(n)))
            if np.isfinite(extrinsic_matrix).all():
                make_pose_json(intrinsic_matrix, extrinsic_matrix,
                               width, height, n, json_path)
                depth = headless_render(
                    mesh, width, height, intrinsic_matrix[:3, :3], extrinsic_matrix)
                plt.imsave(os.path.join(noisy_depth_path,
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
        print(scene)
        make_noisy_depth(scene)
        print(f'Done with {scene} ({i+1}/{len(all_scenes)})')
        break
