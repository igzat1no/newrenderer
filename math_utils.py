import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Distribution1d(nn.Module):

    def __init__(self, data, n):
        super(Distribution1d, self).__init__()
        self.data = torch.tensor(data)
        self.func = self.data
        self.cdf = torch.zeros(n + 1)
        self.cdf[0] = 0
        for i in range(1, n + 1):
            self.cdf[i] = self.cdf[i - 1] + self.func[i - 1] / n
        self.func_int = self.cdf[n]
        if self.func_int == 0:
            self.cdf = torch.tensor([i / n for i in range(1, n + 1)])
        else:
            self.cdf /= self.func_int

    def count(self):
        return self.func.size(0)

    def sample_continuous(self, u, pdf, off=None):
        offset = torch.searchsorted(self.cdf, u)
        if off is not None:
            off = offset
        du = u - self.cdf[offset]
        if self.cdf[offset + 1] - self.cdf[offset] > 0:
            du /= self.cdf[offset + 1] - self.cdf[offset]
        if pdf is not None:
            if self.func_int > 0:
                pdf = self.func[offset] / self.func_int
            else:
                pdf = 0
        return (offset + du) / self.count()

    def sample_discrete(self, u, pdf, u_remapped):
        offset = torch.searchsorted(self.cdf, u)
        if pdf is not None:
            if self.func_int > 0:
                pdf = self.func[offset] / (self.func_int * self.count())
            else:
                pdf = 0

        if u_remapped is not None:
            u_remapped = (u - self.cdf[offset]) / (self.cdf[offset + 1] - self.cdf[offset])
        return offset

    def discrete_pdf(self, index):
        return self.func[index] / (self.func_int * self.count())


class Distribution2d(nn.Module):

    def __init__(self, data, nu, nv):
        super(Distribution2d, self).__init__()
        self.data = data.reshape(nu * nv)

        self.p_conditional_v = []
        for v in range(nv):
            self.p_conditional_v.append(Distribution1d(data[v * nu: (v + 1) * nu], nu))

        marginal_func = []
        for v in range(nv):
            marginal_func.append(self.p_conditional_v[v].func_int)
        self.p_marginal = Distribution1d(marginal_func, nv)

    def sample_continuous(self, u, pdf):
        pdfs = [0, 0]
        v = 0
        d1 = self.p_marginal.sample_continuous(u[1], pdfs[1], v)
        d0 = self.p_conditional_v[v].sample_continuous(u[0], pdfs[0])
        pdf = pdfs[0] * pdfs[1]
        return d0, d1

    def pdf(self, p):
        iu = np.clip(int(p[0] * self.p_conditional_v[0].count()), 0, self.p_conditional_v[0].count() - 1)
        iv = np.clip(int(p[1] * self.p_marginal.count()), 0, self.p_marginal.count() - 1)
        return self.p_conditional_v[iv].func[iu] / self.p_marginal.func_int


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


def Lerp(t, v1, v2):
    return (1 - t) * v1 + t * v2


def illuminance(col):
    return torch.sum(col * col);


def SphericalPhi(v):
    p = torch.atan2(v[:, 1], v[:, 0])
    return torch.where(p < 0, p + 2 * math.pi, p)


def SphericalTheta(v):
    return torch.acos(torch.clamp(v[:, 2], -1, 1))


if __name__ == "__main__":
    h = torch.ones((10, 3))
    wo = torch.ones((10, 3))
    roughness = 0.5
    fd90 = FD90(h, wo, roughness)
    ans = FD(wo, fd90)
    alpha = torch.tensor(0.5)
    nw = GSmith(wo, alpha)
    print(nw.shape)
