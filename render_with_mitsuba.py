import imageio.v3 as imageio
import json
import mitsuba as mi
import numpy as np
import os
import torch
import torch.nn.functional as F
from render_utils import *
from material import *
from math_utils import *
from light import *
from render import process_dict

mi.set_variant("cuda_ad_rgb")


if __name__ == "__main__":

    scene_dir = "/home/lizongtai/research/scenes/combine/"
    scene_path = os.path.join(scene_dir, "scene.json")

    # load scene
    scene_dict = json.load(open(scene_path, "r"))
    print(scene_dict)

    process_dict(scene_dir, scene_dict)

    # render the scene
    scene = mi.load_dict(scene_dict)

    # load camera
    W, H = 640, 640
    fov_y = 40
    eye = torch.tensor([-40, 0, 0], dtype=torch.float32)
    at = torch.tensor([0, 0, 0], dtype=torch.float32)
    up = torch.tensor([0, 0, 1], dtype=torch.float32)

    sensor = mi.load_dict({
        "type": "perspective",
        "film": {
            "type": "hdrfilm",
            "width": W,
            "height": H,
            "rfilter": {
                "type": "box"
            }
        },
        "sampler": {
            "type": "independent",
            "sample_count": 16
        },
        "fov": fov_y,
        "fov_axis": "y",
        "to_world": mi.ScalarTransform4f.look_at(
            eye.tolist(), at.tolist(), up.tolist()),
    })

    integrator = mi.load_dict({
        "type": "path",
        "max_depth": 5
    })

    image = mi.render(scene, spp=16, integrator=integrator, sensor=sensor)
    mi.util.write_bitmap("mitsuba.png", image)

