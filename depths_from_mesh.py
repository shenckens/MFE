import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import scene_utils

PATH = './data/scannet/scans_test'  # put in config at later stage

all_scenes = sorted(os.listdir(PATH))

for scene in all_scenes:
    scene_path = os.path.join(PATH, scene)
