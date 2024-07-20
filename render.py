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

mi.set_variant("cuda_ad_rgb")


# recursively look for "to_world" and "filename" keys
def process_dict(scene_dir, d):
    for k, v in d.items():
        if type(v) == dict and "to_world" in v.keys():
            d[k]["to_world"] = mi.ScalarTransform4f(v["to_world"])
        if type(v) == dict and "filename" in v.keys():
            d[k]["filename"] = os.path.join(scene_dir, v["filename"])
        if type(v) == dict:
            process_dict(scene_dir, v)


def ray_intersect(scene, xs, ds):
    """ warpper of mitsuba ray-mesh intersection
    Args:
        xs: Bx3 pytorch ray origin
        ds: Bx3 pytorch ray direction
    Return:
        positions: Bx3 intersection location
        normals: Bx3 normals
        uvs: Bx2 uv coordinates
        idx: B triangle indices, -1 indicates no intersection
        valid: B whether a valid intersection
    """
    xs_mi = mi.Point3f(xs[..., 0], xs[..., 1], xs[..., 2])
    ds_mi = mi.Vector3f(ds[..., 0], ds[..., 1], ds[..., 2])
    rays_mi = mi.Ray3f(xs_mi, ds_mi)

    ret = scene.ray_intersect_preliminary(rays_mi)
    idx = mi.Int(ret.prim_index).torch().long()  # triangle index
    ret = ret.compute_surface_interaction(rays_mi)

    positions = ret.p.torch()
    # normals = positions.clone() * 0.5
    normals = ret.n.torch()
    normals = F.normalize(normals, dim=-1)
    normals = double_sided(-ds, normals)

    # check if invalid intersection
    ts = ret.t.torch()
    valid = (~ts.isinf())
    idx[~valid] = -1
    return positions, normals, ret.uv.torch(), idx, valid


def path_tracing(scene,
                 material,
                 light,
                 rays_o,
                 rays_d,
                 dx_du,
                 dy_dv,
                 spp,
                 indir_depth,
                 validate=False):
    """ Path trace current scene
    Args:
        scene: mitsuba scene
        emitter_net: emitter object
        material_net: material object
        rays_o: Bx3 ray origin
        rays_d: Bx3 ray direction
        dx_du,dy_dv: Bx3 ray differential
        spp: sampler per pixel
        indir_depth: indirect illumination depth
    Return:
        L: Bx3 traced results
    """
    B = len(rays_o)
    device = rays_o.device

    # sample camera ray
    du, dv = torch.rand(2, len(rays_o), spp, 1, device=device) - 0.5
    rays_d = -rays_d # rays_d points to camera, -rays_d points to object
    rays_d = F.normalize(rays_d[:, None] + dx_du[:, None] * du + dy_dv[:, None] * dv,
                         dim=-1).reshape(-1, 3)
    position = rays_o.repeat_interleave(spp, 0)

    B = len(position)
    L = torch.zeros(B, 3, device=device) # [B, 3], estimated radiance
    beta = torch.ones(B, 3, device=device) # [B, 3], path throughput weight
    active = torch.ones(B, device=device).bool()

    # incident_light_dirs = material["neural_tex"].net.gen_light_incident_dirs(method="stratified_sampling").cuda()  # [envW * envH, 3]
    # envir_map_light_rgbs = material["neural_tex"].net.get_light_rgbs(incident_light_dirs).cuda()  # [light_num, envW * envH, 3]
    # envir_map_light_rgbs = envir_map_light_rgbs.squeeze(0)  # [envW * envH, 3] (light_num = 1)
    # light_area_weight = material["neural_tex"].net.light_area_weight.to(device)  # [envW * envH]
    # # print(light_area_weight.shape)

    # with torch.no_grad():
    # torch.autograd.set_detect_anomaly(True)
    positions = None # for validation

    with torch.no_grad():
        if True:
            for d in range(0, indir_depth):
                with torch.no_grad():
                    new_position, normal, _, _, valid = ray_intersect(
                        scene, position + mi.math.RayEpsilon * rays_d, rays_d)

                if validate:
                    if d == 0:
                        positions = new_position

                # outgoing rays
                invalid = ~valid
                nw_coloring = active.clone()
                nw_coloring[active.clone()] = invalid # rays going outside in this depth
                active[active.clone()] = valid # remaining rays

                outgoing_dir = rays_d[invalid]
                rays_d = rays_d[valid]
                L[nw_coloring] += beta[invalid] * light.Le(outgoing_dir)

                if not active.any():
                    break

                wo = -rays_d
                position = new_position[valid]
                normal = normal[valid]
                beta = beta[valid]
                B = len(position)

                if False:
                    # sampleBSDF
                    u = torch.rand(B, device=device)
                    u2 = torch.rand(B, 2, device=device)
                    wo, t1, t2 = world2local(wo, normal)
                    bs = mat.Sample_f(wo, u, u2)
                    bs.wi = local2world(bs.wi, normal, t1, t2)
                    # dot_prod = torch.einsum("bi,bi->b", bs.wi, normal).unsqueeze(1)
                    dot_prod = Dot(bs.wi, normal)
                    beta *= bs.f * dot_prod.abs() / bs.pdf
                    wi = bs.wi
                else:
                    # uniformly sample sphere
                    u2 = torch.rand(B, 2, device=device)
                    nw_wo, t1, t2 = world2local(wo, normal)
                    wi = SampleUniformHemisphere(u2)
                    f = mat.f(nw_wo, wi)
                    wi = local2world(wi, normal, t1, t2)
                    pdf = torch.tensor(0.5 / math.pi, device=device).repeat(wi.shape[0], 1)
                    # dot_prod = torch.einsum("bi,bi->b", wi, normal)
                    dot_prod = Dot(wi, normal)
                    beta *= f * dot_prod.abs() / pdf

                rays_d = wi

    L = L.reshape(-1, spp, 3).mean(1)
    # L = linear2srgb_torch(L)

    if validate and isinstance(positions, torch.Tensor):
        positions = positions.reshape(-1, spp, 3).mean(1)

    ret_dict = {
        "L": L,
    }
    if validate:
        ret_dict.update({
            "position": positions,
        })
    return ret_dict


if __name__ == "__main__":

    scene_dir = "/home/lizongtai/research/scenes/combine/"
    scene_path = os.path.join(scene_dir, "scene.json")

    # load scene
    scene_dict = json.load(open(scene_path, "r"))
    print(scene_dict)

    process_dict(scene_dir, scene_dict)

    # render the scene
    scene = mi.load_dict(scene_dict)

    # load envmap
    envmap_path = scene_dict["light"]["filename"]
    envmap = imageio.imread(envmap_path)
    envmap = torch.tensor(envmap, dtype=torch.float32)

    # load camera
    W, H = 640, 640
    fov_y = 40
    eye = torch.tensor([40, -12.5, 0], dtype=torch.float32)
    at = torch.tensor([0, -12.5, 0], dtype=torch.float32)
    up = torch.tensor([0, 0, 1], dtype=torch.float32)
    # eye = torch.tensor([10, 10, -5], dtype=torch.float32)
    # at = torch.tensor([0, 0, -1], dtype=torch.float32)
    # up = torch.tensor([0, 0, -1], dtype=torch.float32)

    fov_x = 2 * np.arctan(np.tan(fov_y * 0.5 / 180 * np.pi) * W / H) * 180 / np.pi
    focal = (H * 0.5) / np.tan(fov_y * 0.5 / 180 * np.pi)

    directions = get_ray_directions(H, W, [focal, focal])
    rays_o, rays_d, dxdu, dydv = get_rays(directions, lookAt(eye, at, up), focal)
    # rays_d is now going out from camera image plane to camera position
    # print(rays_d.shape)
    # print(rays_d[:, 0].min(), rays_d[:, 0].max(), rays_d[:, 1].min(), rays_d[:, 1].max(), rays_d[:, 2].min(), rays_d[:, 2].max())
    # breakpoint()
    device = torch.device("cuda")
    rays_o, rays_d = rays_o.to(device), rays_d.to(device)
    dxdu, dydv = dxdu.to(device), dydv.to(device)

    # NOTE: now cope with only one object, need to be modified
    mat = PrincipledBRDF({
        "base_color": [0.5, 0.5, 0.5],
        "roughness": 0.,
    })
    light = ImageInfiniteLight({
        "image": envmap,
        "scale": 1,
    })
    mat = mat.to(device)
    light = light.to(device)

    # render
    spp = 128
    depth = 4
    batch = 640 * 640
    result = torch.zeros((len(rays_o), 3), device=rays_o.device)
    for i in range(0, len(rays_o), batch):
        current_result = path_tracing(scene, mat, light, rays_o[i:i + batch],
                                      rays_d[i:i + batch], dxdu[i:i + batch],
                                      dydv[i:i + batch], spp, depth)
        result[i:i + batch] = current_result["L"]

    result = result.reshape(H, W, 3).cpu().numpy()
    result = np.clip(result, 0, 1)
    result = (result * 255).astype(np.uint8)
    imageio.imwrite("result.png", result)
