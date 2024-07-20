from re import I
import torch
import torch.nn as nn
import math
from math_utils import *
from render_utils import *
from sample_utils import *


class BSDFSample(nn.Module):
    def __init__(self, f, wi, pdf):
        super(BSDFSample, self).__init__()
        self.f = f
        self.wi = wi
        self.pdf = pdf


class DiffuseBRDF(nn.Module):
    def __init__(self, props: dict):
        super(DiffuseBRDF, self).__init__()
        self.R = nn.Parameter(torch.tensor(
            props.get("R", 0.5), dtype=torch.float32))

    # TODO 1/pi can be saved as a constant
    def f(self, wo, wi):
        flag = same_hemisphere(wo, wi)
        ret = self.R * (1.0 / math.pi)
        ret = ret.repeat(wo.shape[0], 1)
        ret[~flag] = 0.0
        return ret

    def Sample_f(self, wo, uc, u):
        wi = SampleCosineHemisphere(u)
        wo[wo[:, 2] < 0] *= -1
        pdf = (wi[:, 2].abs() * (1.0 / math.pi)).unsqueeze(1)
        f = self.R * (1.0 / math.pi)
        f = f.repeat(wo.shape[0], 1)
        return BSDFSample(f, wi, pdf) # [N, 1], [N, 3], [N, 1]

    def PDF(self, wo, wi):
        # flag = same_hemisphere(wo, wi)
        flag = (wo[:, 2] < 0) | (wi[:, 2] < 0)
        ret = wi[:, 2].abs() * (1.0 / math.pi)
        ret[flag] = 0.0
        return ret


class PrincipledBRDF(nn.Module):
    def __init__(self, props: dict):
        super(PrincipledBRDF, self).__init__()
        self.base_color = nn.Parameter(torch.tensor(
            props.get("base_color", [1.0, 1.0, 1.0]), dtype=torch.float32))
        self.metallic = nn.Parameter(torch.tensor(
            props.get("metallic", 0.0), dtype=torch.float32))
        self.roughness = nn.Parameter(torch.tensor(
            props.get("roughness", 0.5), dtype=torch.float32))
        self.specular = nn.Parameter(torch.tensor(
            props.get("specular", 0.5), dtype=torch.float32))
        self.fresnel = nn.Parameter(torch.tensor(
            props.get("fresnel", 1.0), dtype=torch.float32))

    def f(self, wo, wi):
        diffuse = self.base_color / math.pi
        diffuse = diffuse.repeat(wo.shape[0], 1)
        normals = torch.tensor([0, 0, 1], dtype=torch.float32).repeat(wo.shape[0], 1).to(wo.device)
        specular = GGX_specular(normals, wo, wi[:, None], self.roughness.repeat(wo.shape[0], 1), self.fresnel.repeat(wo.shape[0], 1)).squeeze(1)
        return diffuse + specular


class DisneyDiffuse(nn.Module):
    def __init__(self, props: dict):
        super(DisneyDiffuse, self).__init__()
        self.baseColor = nn.Parameter(torch.tensor(
            props.get("baseColor", [1.0, 1.0, 1.0]), dtype=torch.float32))
        self.roughness = nn.Parameter(torch.tensor(
            props.get("roughness", 0.5), dtype=torch.float32))

    def f(self, wo, wi):
        h = (wo + wi) / (wo + wi).norm(dim=1, keepdim=True)
        fd90 = FD90(h, wo, self.roughness)
        fin = FD(wi, fd90)
        fout = FD(wo, fd90)
        normals = torch.tensor([0, 0, 1], dtype=torch.float32).repeat(wo.shape[0], 1).to(wo.device)
        return self.baseColor * fin * fout * (normals @ wi[:, None]).abs() * (1.0 / math.pi)

    def PDF(self, wo: torch.Tensor, wi):
        ret = torch.zeros(wo.shape[0], 1).to(wo.device)
        flag = (wo[:, 2] < 0) | (wi[:, 2] < 0)
        ret = wo[:, 2].clamp_min(0) * (1.0 / math.pi)
        ret[flag] = 0.0
        return ret

    def Sample_f(self, wo, uc, u):
        wi = SampleCosineHemisphere(u)
        wo[wo[:, 2] < 0] *= -1
        pdf = (wi[:, 2].abs() * (1.0 / math.pi)).unsqueeze(1)
        f = self.f(wo, wi)
        return BSDFSample(f, wi, pdf)


# class DisneyMetal(nn.Module):
#     def __init__(self, props: dict):
#         super(DisneyMetal, self).__init__()
#         self.baseColor = nn.Parameter(torch.tensor(
#             props.get("baseColor", [1.0, 1.0, 1.0]), dtype=torch.float32))
#         self.roughness = nn.Parameter(torch.tensor(
#             props.get("roughness", 0.5), dtype=torch.float32))
#         self.specular = nn.Parameter(torch.tensor(
#             props.get("specular", 0.5), dtype=torch.float32))
#         self.metallic = nn.Parameter(torch.tensor(
#             props.get("metallic", 0.0), dtype=torch.float32))

#     def f(self, wo, wi):
#         h = (wo + wi) / (wo + wi).norm(dim=1, keepdim=True)
#         baseColor = self.baseColor.repeat(wo.shape[0], 1).to(wo.device)
#         Ks =
#         Fm = self.baseColor + (1 - self.baseColor) * (1 - Dot(h, wo).abs()).pow(5)
#         alpha = (self.roughness * self.roughness).clamp_min(0.001)
#         NoH = wo[:, 2].unsqueeze(-1)
#         tmp = NoH.pow(2) * (alpha.pow(2) - 1) + 1
#         Dm = alpha.pow(2) / (math.pi * tmp.pow(2))
#         Gm = GSmith(wi, alpha) * GSmith(wo, alpha)
#         return Fm * Dm * Gm / (4 * wi[:, 2].abs().unsqueeze(-1))

#     def PDF(self, wo, wi):
#         h = (wo + wi) / (wo + wi).norm(dim=1, keepdim=True)
#         NoI = wo[:, 2].unsqueeze(-1)
#         NoO = wi[:, 2].unsqueeze(-1)
#         NoH = h[:, 2].unsqueeze(-1)
#         flag = (NoO <= 0) | (NoH <= 0)
#         spec =


if __name__ == "__main__":
    # test_mat = DisneyMetal({})
    wi, wo = torch.ones((10, 3)), torch.ones((10, 3))
    # print(test_mat.f(wo, wi))