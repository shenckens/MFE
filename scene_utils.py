import json
import os

PATH = './data/scannet/scans_test'  # put in config at later stage


def make_json_template(scene):
    f = {
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
    file = json.dump(f, 'scene_template.json')
    return file


def get_json_template():
    template = os.path.join(PATH, 'scene_template.json')
    if not template:
        template = make_json_template()
    return template
