import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import scene_utils

PATH = './data/scannet/scans_test'  # put in config at later stage


def load_recon_mesh_for_testing(scene):
    recon_mesh = o3d.io.read_triangle_mesh(
        f'./{scene}.ply')  # Read the reconstructed mesh
    return recon_mesh


def load_recon_pcd_for_testing(scene):
    recon_pcd = o3d.io.read_point_cloud(
        f'./{scene}.ply')  # Read the reconstructed pcd
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
    # if os.path.isfile(os.path.join(path, '{}.json'.format(img_idx))):
    #     return 0
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


def render_depth_img(pcd, parameter_file, img_idx, path):
    param = o3d.io.read_pinhole_camera_parameters(parameter_file)
    f = open(parameter_file)
    file = json.load(f)
    width = file["intrinsic"]["width"]
    height = file["intrinsic"]["height"]
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=param.intrinsic.width,
                      height=param.intrinsic.height)
    ctr = vis.get_view_control()
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
#     depth = vis.capture_depth_float_buffer(False)
#     plt.imsave('{}.png'.format("testdepth_1"), np.asarray(depth), cmap='gray')

    vis.capture_depth_image(os.path.join(
        path, '{}.png'.format(img_idx)), do_render=True)
#     vis.run()
#     vis.destroy_window()
    pass


def noisy_depth(scene):
    mesh = load_recon_mesh_for_testing(scene)  # Load mesh for testing
    pcd = load_recon_pcd_for_testing(scene)  # Load pcd for testing
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
    for n in range(n_poses):
        extrinsic_matrix = np.genfromtxt(os.path.join(
            scene_path, 'pose', '{}.txt'.format(n)))
        make_pose_json(intrinsic_matrix, extrinsic_matrix,
                       width, height, n, json_path)
        render_depth_img(mesh, os.path.join(
            json_path, '{}.json'.format(n)), n, noisy_depth_path)
        render_depth_img(pcd, os.path.join(
            json_path, '{}.json'.format(n)), n, noisy_pcd_path)
    pass


if __name__ == "__main__":
    all_scenes = sorted([s for s in os.listdir(PATH) if not s.startswith('.')])

    for scene in all_scenes:
        noisy_depth(scene)

        break
