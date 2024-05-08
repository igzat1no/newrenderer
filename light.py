import torch
import torch.nn as nn
from sample_utils import *


class LightSampleContext:
    def __init__(self, p, n, ns):
        self.p = p
        self.n = n
        self.ns = ns # shading normal


class LightLiSample:
    def __init__(self, L, wi, pdf):
        self.L = L
        self.wi = wi
        self.pdf = pdf


class UniformInfiniteLight:
    def __init__(self, props: dict):
        self.color = nn.Parameter(torch.tensor(
            props.get("color", [1.0, 1.0, 1.0]), dtype=torch.float32))
        self.scale = nn.Parameter(torch.tensor(
            props.get("scale", 1.0), dtype=torch.float32))

    def Le(self, ray_dir):
        ret = self.color * self.scale
        ret = ret.expand(ray_dir.shape[0])
        return ret

    def SampleLi(self, ctx: LightSampleContext, u):
        wi = SampleUniformSphere(u)
        pdf = torch.tensor(1.0 / (4.0 * math.pi), dtype=torch.float32)
        pdf = pdf.expand(wi.shape[0])
        return LightLiSample(self.Le(wi), wi, pdf)

