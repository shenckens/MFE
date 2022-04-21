import trimesh
import pyrender
import json
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
# for no screen rendering
os.environ['PYOPENGL_PLATFORM'] = 'egl'
# import scene_utils

PATH = '/project/henckens/data/scannet/scans_test'  # put in config at later stage
RECONPATH = './results/scene_scannet_checkpoints_fusion_eval_47'

# PATH = './Desktop/data/scannet/scans_test'
# RECONPATH = './Desktop'


class Renderer():
    """OpenGL mesh renderer.
    Used to render depthmaps from a mesh for 2d evaluation and model training.
    """

    def __init__(self, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        self.render_flags = pyrender.RenderFlags.OFFSCREEN
        # self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=width / 2 - 0.5, cy=height / 2 - 0.5,
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=pose)
        # self.scene.add(cam, pose=np.linalg.inv(pose))
        return self.renderer.render(self.scene, self.render_flags)

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def pcd_opengl(self, mesh):
        return pyrender.Poin.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()

    # obj1_trimesh = trimesh.load('./models/obj_01.ply')
    # obj1_mesh = Mesh.from_trimesh(obj1_trimesh)
    # scene.add(obj1_mesh)


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


# def pyrender_depth_img(mesh, parameter_file, img_idx, path):
#     # param = o3d.io.read_pinhole_camera_parameters(parameter_file)
#     f = open(parameter_file)
#     file = json.load(f)
#     width = file["intrinsic"]["width"]
#     height = file["intrinsic"]["height"]
#     renderer = Renderer()
#     mesh_opengl = renderer.mesh_opengl(mesh)
#
#     _, depth_pred = renderer(height, width, cam_intr, cam_pose, mesh_opengl)
#     pass


def make_noisy_depth(scene):

    scene_path = os.path.join(PATH, scene)
    scene_info = dict(np.loadtxt(
        f'{os.path.join(scene_path, scene)}.txt', delimiter=' = ', dtype=dict))

    json_path = os.path.join(scene_path, 'o3d_parameters')
    if not os.path.exists(json_path):
        os.makedirs(json_path)
    pyrender_path = os.path.join(scene_path, 'recon_depth')
    if not os.path.exists(pyrender_path):
        os.makedirs(pyrender_path)

    n_poses = int(scene_info['numDepthFrames'])
    width = int(scene_info['depthWidth'])
    height = int(scene_info['depthHeight'])
    fx = float(scene_info['fx_depth'])
    fy = float(scene_info['fy_depth'])
    cx = float(scene_info['mx_depth'])
    cy = float(scene_info['my_depth'])
    intrinsic_matrix = np.genfromtxt(os.path.join(
        scene_path, 'intrinsic/intrinsic_depth.txt'))

    mesh = trimesh.load(os.path.join(
        RECONPATH, '{}.ply'.format(scene)))
    mesh = pyrender.Mesh.from_trimesh(mesh)
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
                make_pose_json(intrinsic_matrix, extrinsic_matrix,
                               width, height, n, json_path)
                # render_depth_img(mesh, os.path.join(
                #     json_path, '{}.json'.format(n)), n, noisy_depth_path)
                # render_depth_img(pcd, os.path.join(
                #     json_path, '{}.json'.format(n)), n, noisy_pcd_path)

                # EDIT
                # mesh = trimesh.load(os.path.join(
                #     RECONPATH, '{}.ply'.format(scene)))
                # mesh = pyrender.Mesh.from_trimesh(mesh)
                # renderer = Renderer(width, height)
                # mesh_opengl = renderer.mesh_opengl(mesh)
                #
                # _, depth_pred = renderer(
                #     height, width, intrinsic_matrix, extrinsic_matrix, mesh_opengl)
                # plt.imsave(os.path.join(pyrender_path, '{}.png'.format(n)),
                #            np.asarray(depth_pred), cmap='gray')
                # renderer.delete()

                renderer = Renderer()
                cam_pose = extrinsic_matrix@rotation_x
                _, depth_pred = renderer(
                    height, width, intrinsic_matrix, cam_pose, mesh)
                renderer.delete()
                plt.imsave(os.path.join(pyrender_path, '{}.png'.format(n)),
                           depth_pred)
                # depth_pred = o3d.geometry.Image(depth_pred)
                # o3d.io.write_image(os.path.join(
                #     pyrender_path, '{}.png'.format(n)), depth_pred)
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
        make_noisy_depth(scene)
        print(f'Done with {scene} ({i+1}/{len(all_scenes)})')
        break
