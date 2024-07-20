import math
from re import T
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from kornia.utils.grid import create_meshgrid
from math_utils import Lerp


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



class MIPMap(nn.Module):

    def __init__(self, height, width, img, doTri=False, maxAniso=8.0):

        self.doTri = doTri
        self.maxAniso = maxAniso
        self.res = [height, width]

        nLevels = 1 + int(math.log2(max(height, width)))
        self.pyramid = []
        self.pyramid.append(img)
        pool_layer = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        for i in range(1, nLevels - 1):
            input_tensor = self.pyramid[i - 1]
            output_tensor = pool_layer(input_tensor.permute(2, 0, 1).unsqueeze(0))
            self.pyramid.append(output_tensor.squeeze(0).permute(1, 2, 0))
        last_tensor = self.pyramid[-1]
        self.pyramid.append(torch.mean(last_tensor, dim=1, keepdim=True))

        import imageio.v3 as imageio
        outimg = torch.zeros(1024, 1024, 3)
        ind = 0
        for i in range(0, nLevels):
            img = self.pyramid[i]
            img = torch.clamp(img, 0, 1)
            img = img * 255
            outimg[ind:ind + img.shape[0], :img.shape[1], :] = img
            ind += img.shape[0]

        self.weightLUTsize = 128
        self.weightLut = []
        for i in range(self.weightLUTsize):
            r2 = i / (self.weightLUTsize - 1)
            self.weightLut.append(math.exp(-r2 * 2) - math.exp(-2))

    def Levels(self):
        return len(self.pyramid)

    def triangle(self, level, st):
        level = np.clip(level, 0, self.Levels() - 1)
        s = st[..., 0] * self.pyramid[level].shape[0] - 0.5
        t = st[..., 1] * self.pyramid[level].shape[1] - 0.5
        s0 = torch.floor(s).to(torch.int32)
        t0 = torch.floor(t).to(torch.int32)
        ds = s - s0
        dt = t - t0
        ds = ds[..., None]
        dt = dt[..., None]
        ind1 = torch.stack((s0, t0), dim=-1)
        ind2 = torch.stack((s0 + 1, t0), dim=-1)
        ind3 = torch.stack((s0, t0 + 1), dim=-1)
        ind4 = torch.stack((s0 + 1, t0 + 1), dim=-1)

        def process(ind):
            print(ind.shape)
            flag = (ind[..., 0] == self.pyramid[level].shape[0])
            print(flag.shape)
            if flag.sum() > 0:
                ind[flag, 0] = 0
            flag = (ind[..., 1] == self.pyramid[level].shape[1])
            print(flag.shape)
            if flag.sum() > 0:
                ind[flag, 1] = 0

        process(ind1)
        process(ind2)
        process(ind3)
        process(ind4)

        return (1 - ds) * (1 - dt) * self.pyramid[level][ind1[..., 0], ind1[..., 1]] + \
                (1 - ds) * dt * self.pyramid[level][ind2[..., 0], ind2[..., 1]] + \
                ds * (1 - dt) * self.pyramid[level][ind3[..., 0], ind3[..., 1]] + \
                ds * dt * self.pyramid[level][ind4[..., 0], ind4[..., 1]]

    def Lookup(self, st, width=0.):
        print("shaithisnaidnmweimt")
        level = self.Levels() - 1 + math.log2(max(width, 1e-8))
        if level < 0:
            return self.triangle(0, st)
        elif level >= self.Levels() - 1:
            return self.pyramid[self.Levels() - 1][0, 0] * torch.ones_like(st)
        else:
            iLevel = int(level)
            delta = level - iLevel
            return Lerp(delta, self.triangle(iLevel, st), self.triangle(iLevel + 1, st))


if __name__ == '__main__':

    eye = torch.tensor([10, 10, -5], dtype=torch.float32)
    at = torch.tensor([0, 0, -1], dtype=torch.float32)
    up = torch.tensor([0, 0, -1], dtype=torch.float32)

    directions = get_ray_directions(32, 32, torch.tensor([1, 1], dtype=torch.float32))
    print(directions.shape)
    rays_o, rays_d, dxdu, dydv = get_rays(directions, lookAt(eye, at, up), 1)
    print(rays_o.shape, rays_d.shape, dxdu.shape, dydv.shape)
    print(dxdu, dydv)