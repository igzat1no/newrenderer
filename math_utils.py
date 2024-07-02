import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def Dot(v1, v2):
    # v1, v2: (B, N)
    # return: (B, 1)
    v1 = v1.unsqueeze(1)
    v2 = v2.unsqueeze(2)
    return torch.bmm(v1, v2).squeeze(2)


def FD90(h, wo, roughness):
    dot_res = Dot(h, wo).abs()
    return 2 * roughness * (dot_res * dot_res) + 0.5


def FD(w, fd90):
    dot_res = w[:, 2].abs().pow(5)[:, None]
    return (fd90 - 1) * (1 - dot_res) + 1


def GSmith(w, alpha):
    wz = w[:, 2].pow(2).unsqueeze(-1)
    return 2 / (1 + torch.sqrt(1 - alpha * alpha + alpha * alpha / (wz + 1e-6)))


def luminance(s):
    return 0.2126 * s[:, 0] + 0.7152 * s[:, 1] + 0.0722 * s[:, 2]


if __name__ == "__main__":
    h = torch.ones((10, 3))
    wo = torch.ones((10, 3))
    roughness = 0.5
    fd90 = FD90(h, wo, roughness)
    ans = FD(wo, fd90)
    alpha = torch.tensor(0.5)
    nw = GSmith(wo, alpha)
    print(nw.shape)
