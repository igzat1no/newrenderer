import imageio.v3 as imageio
import json
import mitsuba as mi
import numpy as np
import os
import torch
import torch.nn.functional as F
from render_utils import *
from material import *

mi.set_variant("cuda_ad_rgb")


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
    wi = F.normalize(rays_d[:, None] + dx_du[:, None] * du + dy_dv[:, None] * dv,
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

    if True:
        for d in range(0, indir_depth):
            with torch.no_grad():
                new_position, normal, _, _, vis = ray_intersect(
                    scene, position + mi.math.RayEpsilon * wi, wi)

            if validate:
                if d == 0:
                    positions = new_position

            normal = material["neural_tex"].net.renderModule_normal(
                nw_position, intrinsic_feat)
            normal = double_sided(-wi, normal)
            normal = F.normalize(normal, dim=-1, eps=1e-6)

            nw_colored = active.clone()
            nw_colored[nw_colored.clone()] = ~vis
            active[active.clone()] = vis
            if d != 0:
                if True:
                    # with torch.no_grad():
                    # calculate the color of the light
                    position = position[~vis]
                    nw_wi = wi[~vis]  # [bs, 3]
                    # find the corresponding light on environment map using nw_wi
                    cosine = (nw_wi[:, None] * incident_light_dirs).sum(
                        -1)  # [bs, envW * envH]
                    # print(cosine.max(), cosine.min())
                    # exit(0)
                    indices = cosine.argmax(-1)  # [bs]
                    light_rgbs = envir_map_light_rgbs[indices]  # [bs, 3]
                    light_weight_nw = light_area_weight[indices]  # [bs]
                    light_weight_nw = torch.stack(
                        [light_weight_nw, light_weight_nw, light_weight_nw], dim=-1)
                    # calculate the surface color
                    surface_brdf = nw_brdf[~vis]
                    tmp_ind = torch.arange(len(surface_brdf), device=device)
                    cosine = cosine[tmp_ind, indices]
                    cosine = torch.stack([cosine, cosine, cosine], dim=-1)
                    # light_rgbs = light_rgbs * cosine
                    L[nw_colored] += surface_brdf * light_rgbs  # * cosine #* light_weight_nw

            if not active.any():
                break

            wo = -wi
            position = new_position[vis]
            wo = wo[vis]
            normal = normal[vis]
            nw_brdf = nw_brdf[vis]
            B = len(position)

            # mat = material["neural_tex"].sample(position, wo)
            # if validate:
            #     if d == 0:
            #         albedo[vis] = mat["albedo"]

            # breakpoint()
            sample1 = torch.rand(B, device=device)
            # sample2 = torch.rand(B, 2, device=device)

            sample2 = torch.rand(B, 2, device=device)
            wi = SampleCosineHemisphere(sample2)
            wi[(wi * normal).sum(-1) < 0, 2] *= -1
            # pdf = CosineHemispherePDF(AbsCosTheta(wi))
            # specular = GGX_specular(normal, wo, wi[:, None], mat["roughness"], mat["fresnel"]).squeeze(1)
            # nw_brdf *= (mat["albedo"] / math.pi + specular) * ((wi * normal).sum(-1).abs() / pdf).unsqueeze(1)
            nw_brdf *= material.f(wo, wi)

    L = L.reshape(-1, spp, 3).mean(1)
    L = torch.clamp(L, 0, 1)
    L = linear2srgb_torch(L)

    if validate:
        positions = positions.reshape(-1, spp, 3).mean(1)
        albedo = albedo.reshape(-1, spp, 3).mean(1)

    ret_dict = {
        "L": L,
    }
    if validate:
        ret_dict.update({
            "position": positions,
            'albedo': albedo,
        })
    return ret_dict


if __name__ == "__main__":
    scene_dir = "/home/lizongtai/research/scenes/simple/"
    scene_path = os.path.join(scene_dir, "scene.json")

    # load scene
    scene_dict = json.load(open(scene_path, "r"))
    print(scene_dict)

    for k, v in scene_dict.items():
        if type(v) == dict and "to_world" in v.keys():
            scene_dict[k]["to_world"] = mi.ScalarTransform4f(v["to_world"])
        if type(v) == dict and "filename" in v.keys():
            scene_dict[k]["filename"] = os.path.join(scene_dir, v["filename"])

    # render the scene
    scene = mi.load_dict(scene_dict)

    # load camera
    W, H = 640, 480
    fov_y = 40
    eye = torch.tensor([10, 0, 0], dtype=torch.float32)
    at = torch.tensor([0, 0, 0], dtype=torch.float32)
    up = torch.tensor([0, 0, 1], dtype=torch.float32)
    # eye = torch.tensor([10, 10, -5], dtype=torch.float32)
    # at = torch.tensor([0, 0, -1], dtype=torch.float32)
    # up = torch.tensor([0, 0, -1], dtype=torch.float32)

    # NOTE: now cope with only one object, need to be modified
    mat = DiffuseBRDF(dict())

    fov_x = 2 * np.arctan(np.tan(fov_y * 0.5) * W / H)
    focal = (H * 0.5) / np.tan(fov_y * 0.5)

    directions = get_ray_directions(H, W, [focal, focal])
    rays_o, rays_d, dxdu, dydv = get_rays(directions, lookAt(eye, at, up), focal)
    device = torch.device("cuda")
    rays_o = rays_o.to(device)
    rays_d = rays_d.to(device)
    dxdu = dxdu.to(device)
    dydv = dydv.to(device)
    # render
    spp = 16
    depth = 4
    batch = 640 * 480
    result = torch.zeros((len(rays_o), 3), device=rays_o.device)
    for i in range(0, len(rays_o), batch):
        current_result = path_tracing(scene, mat, rays_o[i:i + batch],
                                      rays_d[i:i + batch], dxdu[i:i + batch],
                                      dydv[i:i + batch], spp, depth)
        result[i:i + batch] = current_result["L"]

    result = result.reshape(H, W, 3).cpu().numpy()
    result = np.clip(result, 0, 1)
    result = (result * 255).astype(np.uint8)
    imageio.imwrite("result.png", result)
