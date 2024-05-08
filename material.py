import torch
import torch.nn as nn
import math
from render_utils import *
from sample_utils import *


class BSDFSample:
    def __init__(self, f, wi, pdf):
        self.f = f
        self.wi = wi
        self.pdf = pdf


class DiffuseBRDF:
    def __init__(self, props: dict):
        self.R = nn.Parameter(torch.tensor(
            props.get("R", 0.5), dtype=torch.float32))

    # TODO 1/pi can be saved as a constant
    def f(self, wo, wi):
        flag = same_hemisphere(wo, wi)
        ret = self.R * (1.0 / math.pi)
        ret = ret.expand(wo.shape[0]).to(wo.device)
        ret[~flag] = 0.0
        return ret

    def Sample_f(self, wo, uc, u):
        wi = SampleCosineHemisphere(u)
        wo[wo[:, 2] < 0] *= -1
        pdf = wi[:, 2].abs() * (1.0 / math.pi)
        f = self.R * (1.0 / math.pi)
        f = f.expand(wo.shape[0]).to(wo.device)
        return BSDFSample(f, wi, pdf)

    def PDF(self, wo, wi):
        flag = same_hemisphere(wo, wi)
        ret = wi[:, 2].abs() * (1.0 / math.pi)
        ret[~flag] = 0.0
        return ret


class PrincipledBRDF:
    def __init__(self, props: dict):
        self.base_color = nn.Parameter(torch.tensor(
            props.get("base_color", [1.0, 1.0, 1.0]), dtype=torch.float32))
        self.metallic = nn.Parameter(torch.tensor(
            props.get("metallic", 0.0), dtype=torch.float32))
        self.roughness = nn.Parameter(torch.tensor(
            props.get("roughness", 0.5), dtype=torch.float32))
        self.specular = nn.Parameter(torch.tensor(
            props.get("specular", 0.5), dtype=torch.float32))
