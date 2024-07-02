from drjit import device
import torch
import math


def SampleUniformDisk(sample2):
    ''' sampling on unit disk, return in [-1, 1] * [-1, 1]
    '''
    # print(sample2[:10])
    B = sample2.shape[0]
    sample2 = 2 * sample2 - 1
    ret = torch.zeros((B, 2), device=sample2.device)

    mask = (sample2[:, 0] == 0) & (sample2[:, 1] == 0)
    mask = ~mask
    nwmask = sample2[:, 0].abs() > sample2[:, 1].abs()
    mask1 = mask & nwmask
    ret[mask1] = torch.stack([
        sample2[mask1, 0] * ((math.pi / 4) * (sample2[mask1, 1] / sample2[mask1, 0])).cos(),
        sample2[mask1, 0] * ((math.pi / 4) * (sample2[mask1, 1] / sample2[mask1, 0])).sin(),
    ], dim=1)
    mask2 = mask & ~nwmask
    ret[mask2] = torch.stack([
        sample2[mask2, 1] * ((math.pi / 2) - (math.pi / 4) * (sample2[mask2, 0] / sample2[mask2, 1])).cos(),
        sample2[mask2, 1] * ((math.pi / 2) - (math.pi / 4) * (sample2[mask2, 0] / sample2[mask2, 1])).sin(),
    ], dim=1)
    return ret


def SampleCosineHemisphere(sample2):
    r = SampleUniformDisk(sample2)
    z = torch.clamp(1 - r[:, 0] * r[:, 0] - r[:, 1] * r[:, 1], min=1e-8).sqrt()
    z = z.unsqueeze(1)
    r = torch.concat([r, z], dim=-1)
    return r


def SampleUniformSphere(sample2):
    z = 1 - 2 * sample2[:, 0]
    r = (1 - z * z).clamp(min=1e-8).sqrt()
    phi = 2 * math.pi * sample2[:, 1]
    return torch.stack([r * phi.cos(), r * phi.sin(), z], dim=-1)


def SampleUniformHemisphere(sample2):
    z = sample2[:, 0]
    r = (1 - z * z).clamp(min=1e-8).sqrt()
    phi = 2 * math.pi * sample2[:, 1]
    return torch.stack([r * phi.cos(), r * phi.sin(), z], dim=-1)

