import torch
from kornia.utils.grid import create_meshgrid
import math


def same_hemisphere(W1, W2):
    return W1[..., 2] * W2[..., 2] > 0


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
    rays_d = directions @ R.T  # (H, W, 3)
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


if __name__ == '__main__':
    eye = torch.tensor([10, 10, -5], dtype=torch.float32)
    at = torch.tensor([0, 0, -1], dtype=torch.float32)
    up = torch.tensor([0, 0, -1], dtype=torch.float32)

    directions = get_ray_directions(32, 32, torch.tensor([1, 1], dtype=torch.float32))
    print(directions.shape)
    rays_o, rays_d, dxdu, dydv = get_rays(directions, lookAt(eye, at, up), 1)
    print(rays_o.shape, rays_d.shape, dxdu.shape, dydv.shape)
    print(dxdu, dydv)