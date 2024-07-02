import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from kornia.utils.grid import create_meshgrid


def same_hemisphere(W1, W2):
    return W1[..., 2] * W2[..., 2] > 0


def Reflect(v, n):
    return -v + 2 * torch.sum(v * n, dim=-1, keepdim=True) * n


def world2local(v, normal):
    """ convert world coordinate to local coordinate
    Args:
        v: Bx3 world coordinate
        normal: Bx3 normal
    Return:
        v_local: Bx3 local coordinate
    """
    nw = torch.tensor([0, 0, 1], dtype=torch.float32, device=v.device).expand_as(v)
    flag = (normal == nw)
    if flag.sum() > 0:
        nw[flag] = torch.tensor([0, 1, 0], dtype=torch.float32, device=v.device).expand_as(nw[flag])
    tangent = F.normalize(torch.cross(normal, nw, dim=-1), dim=-1)
    bitangent = torch.cross(normal, tangent, dim=-1)
    ret = torch.stack([torch.einsum("bi,bi->b", v, tangent),
                        torch.einsum("bi,bi->b", v, bitangent),
                        torch.einsum("bi,bi->b", v, normal)], dim=-1)
    return F.normalize(ret, dim=-1), tangent, bitangent


def local2world(v, normal, tangent, bitangent):
    """ convert local coordinate to world coordinate
    Args:
        v: Bx3 local coordinate
        normal: Bx3 normal
        tangent: Bx3 tangent
        bitangent: Bx3 bitangent
    Return:
        v_world: Bx3 world coordinate
    """
    return v[:, 0:1] * tangent + v[:, 1:2] * bitangent + v[:, 2:3] * normal


def double_sided(V,N):
    """ double sided normal
    Args:
        V: Bx3 viewing direction
        N: Bx3 normal direction
    Return:
        Bx3 flipped normal towards camera direction
    """
    NoV = (N * V).sum(-1)
    flipped = NoV < 0
    N_flipped = torch.where(flipped.unsqueeze(-1), -N, N)
    return N_flipped


def lookAt(eye, at, up):
    a = eye - at
    w = a / torch.linalg.norm(a)
    u = torch.cross(up, w, dim=-1)
    u = u / torch.linalg.norm(u)
    v = torch.cross(w, u, dim=-1)
    c2w = torch.tensor([[u[0], u[1], u[2], eye[0]],
                        [v[0], v[1], v[2], eye[1]],
                        [w[0], w[1], w[2], eye[2]],
                        [   0,    0,    0,      1]],
                        dtype=eye.dtype, device=eye.device)
    return c2w


def get_ray_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False).squeeze(0) + 0.5
    x, y = grid.unbind(-1)
    center = [W / 2, H / 2]
    directions = torch.stack([(x - center[0]) / focal[0],
        (y - center[1]) / focal[1], torch.ones_like(x)], -1)  # (H, W, 3)
    directions /= torch.norm(directions, dim=-1, keepdim=True)
    return directions


def get_rays(directions, c2w, focal=None):
    R = c2w[:3, :3]
    rays_d = directions @ R  # (H, W, 3)
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    if focal is None:
        return rays_o, rays_d, None, None

    dxdu = torch.tensor([1.0/focal,0,0], dtype=torch.float32)[None,None].expand_as(directions)@R.T
    dydv = torch.tensor([0,1.0/focal,0], dtype=torch.float32)[None,None].expand_as(directions)@R.T
    dxdu = dxdu.view(-1,3)
    dydv = dydv.view(-1,3)
    return rays_o, rays_d, dxdu, dydv


def _clip_0to1_warn_torch(tensor_0to1):
    """Enforces [0, 1] on a tensor/array that should be already [0, 1].
    """
    msg = "Some values outside [0, 1], so clipping happened"
    if isinstance(tensor_0to1, torch.Tensor):
        if torch.min(tensor_0to1) < 0 or torch.max(tensor_0to1) > 1:
            tensor_0to1 = torch.clamp(
                tensor_0to1, min=0, max=1)
    elif isinstance(tensor_0to1, np.ndarray):
        if tensor_0to1.min() < 0 or tensor_0to1.max() > 1:
            tensor_0to1 = np.clip(tensor_0to1, 0, 1)
    else:
        raise NotImplementedError(f'Do not support dtype {type(tensor_0to1)}')
    return tensor_0to1


def linear2srgb_torch(tensor_0to1):
    if isinstance(tensor_0to1, torch.Tensor):
        pow_func = torch.pow
        where_func = torch.where
    elif isinstance(tensor_0to1, np.ndarray):
        pow_func = np.power
        where_func = np.where
    else:
        raise NotImplementedError(f'Do not support dtype {type(tensor_0to1)}')

    srgb_linear_thres = 0.0031308
    srgb_linear_coeff = 12.92
    srgb_exponential_coeff = 1.055
    srgb_exponent = 2.4

    tensor_0to1 = _clip_0to1_warn_torch(tensor_0to1)

    tensor_linear = tensor_0to1 * srgb_linear_coeff

    tensor_nonlinear = srgb_exponential_coeff * (
        pow_func(tensor_0to1 + 1e-6, 1 / srgb_exponent)
    ) - (srgb_exponential_coeff - 1)

    is_linear = tensor_0to1 <= srgb_linear_thres
    tensor_srgb = where_func(is_linear, tensor_linear, tensor_nonlinear)

    return tensor_srgb


def GGX_specular(
        normal,
        pts2c,
        pts2l,
        roughness,
        fresnel
):
    L = F.normalize(pts2l, dim=-1)  # [nrays, nlights, 3]
    V = F.normalize(pts2c, dim=-1)  # [nrays, 3]
    H = F.normalize((L + V[:, None, :]) / 2.0, dim=-1)  # [nrays, nlights, 3]
    N = F.normalize(normal, dim=-1)  # [nrays, 3]

    NoV = torch.sum(V * N, dim=-1, keepdim=True)  # [nrays, 1]
    N = N * NoV.sign()  # [nrays, 3]

    NoL = torch.sum(N[:, None, :] * L, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1] TODO check broadcast
    NoV = torch.sum(N * V, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, 1]
    NoH = torch.sum(N[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]
    VoH = torch.sum(V[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]

    alpha = roughness * roughness  # [nrays, 3]
    alpha2 = alpha * alpha  # [nrays, 3]
    k = (alpha + 2 * roughness + 1.0) / 8.0
    FMi = ((-5.55473) * VoH - 6.98316) * VoH
    frac0 = fresnel[:, None, :] + (1 - fresnel[:, None, :]) * torch.pow(2.0, FMi)  # [nrays, nlights, 3]

    frac = frac0 * alpha2[:, None, :]  # [nrays, 1]
    nom0 = NoH * NoH * (alpha2[:, None, :] - 1) + 1

    nom1 = NoV * (1 - k) + k
    nom2 = NoL * (1 - k[:, None, :]) + k[:, None, :]
    nom = (4 * np.pi * nom0 * nom0 * nom1[:, None, :] * nom2).clamp_(1e-6, 4 * np.pi)
    spec = frac / nom
    return spec


if __name__ == '__main__':
    eye = torch.tensor([10, 10, -5], dtype=torch.float32)
    at = torch.tensor([0, 0, -1], dtype=torch.float32)
    up = torch.tensor([0, 0, -1], dtype=torch.float32)

    directions = get_ray_directions(32, 32, torch.tensor([1, 1], dtype=torch.float32))
    print(directions.shape)
    rays_o, rays_d, dxdu, dydv = get_rays(directions, lookAt(eye, at, up), 1)
    print(rays_o.shape, rays_d.shape, dxdu.shape, dydv.shape)
    print(dxdu, dydv)