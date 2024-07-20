import torch
import torch.nn as nn
from render_utils import MIPMap
from sample_utils import *
from math_utils import *


class LightSampleContext(nn.Module):
    def __init__(self, p, n, ns):
        super(LightSampleContext, self).__init__()
        self.p = p
        self.n = n
        self.ns = ns # shading normal


class LightLiSample(nn.Module):
    def __init__(self, L, wi, pdf):
        super(LightLiSample, self).__init__()
        self.L = L
        self.wi = wi
        self.pdf = pdf


class UniformInfiniteLight(nn.Module):
    def __init__(self, props: dict):
        super(UniformInfiniteLight, self).__init__()
        self.color = nn.Parameter(torch.tensor(
            props.get("color", [1.0, 1.0, 1.0]), dtype=torch.float32))
        self.scale = nn.Parameter(torch.tensor(
            props.get("scale", 1.0), dtype=torch.float32))

    def Le(self, ray_dir):
        ret = self.color * self.scale
        ret = ret.repeat(ray_dir.shape[0], 1)
        return ret

    def SampleLi(self, ctx: LightSampleContext, u):
        wi = SampleUniformSphere(u)
        pdf = torch.tensor(1.0 / (4.0 * math.pi), dtype=torch.float32)
        pdf = pdf.expand(wi.shape[0])
        return LightLiSample(self.Le(wi), wi, pdf)


class ImageInfiniteLight(nn.Module):
    def __init__(self, props: dict):
        super(ImageInfiniteLight, self).__init__()
        image = nn.Parameter(torch.tensor(
            props.get("image", []), dtype=torch.float32))
        self.scale = nn.Parameter(torch.tensor(
            props.get("scale", 1.0), dtype=torch.float32))
        self.scene_center = torch.tensor(
            props.get("scene_center", [0.0, 0.0, 0.0]), dtype=torch.float32)
        self.scene_radius = torch.tensor(
            props.get("scene_radius", 100.0), dtype=torch.float32)

        height = image.shape[0]
        width = image.shape[1]
        self.image = MIPMap(height, width, image)
        nwimg = torch.zeros(height, width, 3, dtype=torch.float32)
        vp = torch.arange(0, height, dtype=torch.float32) / height
        up = torch.arange(0, width, dtype=torch.float32) / width
        sinTheta = torch.sin(math.pi * (torch.arange(0, height, dtype=torch.float32) + 0.5) / height)
        st = torch.zeros(height, width, 2, dtype=torch.float32)
        st[:, :, 0] = vp.unsqueeze(1).expand(height, width)
        st[:, :, 1] = up.unsqueeze(0).expand(height, width)

        filter = 1. / max(width, height)
        nwimg = self.image.Lookup(st, filter) * sinTheta

        self.distribution = Distribution2d(nwimg, width, height)


    def Power(self):
        return math.pi * self.scene_radius * self.scene_radius * self.image.Lookup(
            torch.tensor([0.5, 0.5], dtype=torch.float32), 0.0) * self.scale

    def Le(self, ray_dir):
        s = SphericalPhi(ray_dir) * 0.5 / math.pi
        t = SphericalTheta(ray_dir) / math.pi
        st = torch.stack([s, t], dim=-1)
        return self.image.Lookup(st) * self.scale

    # def SampleLi(self, ctx: LightSampleContext, u):




# Spectrum InfiniteAreaLight::Sample_Li(const Interaction &ref, const Point2f &u,
#                                       Vector3f *wi, Float *pdf,
#                                       VisibilityTester *vis) const {
#     ProfilePhase _(Prof::LightSample);
#     // Find $(u,v)$ sample coordinates in infinite light texture
#     Float mapPdf;
#     Point2f uv = distribution->SampleContinuous(u, &mapPdf);
#     if (mapPdf == 0) return Spectrum(0.f);

#     // Convert infinite light sample point to direction
#     Float theta = uv[1] * Pi, phi = uv[0] * 2 * Pi;
#     Float cosTheta = std::cos(theta), sinTheta = std::sin(theta);
#     Float sinPhi = std::sin(phi), cosPhi = std::cos(phi);
#     *wi =
#         LightToWorld(Vector3f(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta));

#     // Compute PDF for sampled infinite light direction
#     *pdf = mapPdf / (2 * Pi * Pi * sinTheta);
#     if (sinTheta == 0) *pdf = 0;

#     // Return radiance value for infinite light direction
#     *vis = VisibilityTester(ref, Interaction(ref.p + *wi * (2 * worldRadius),
#                                              ref.time, mediumInterface));
#     return Spectrum(Lmap->Lookup(uv), SpectrumType::Illuminant);
# }

# Float InfiniteAreaLight::Pdf_Li(const Interaction &, const Vector3f &w) const {
#     ProfilePhase _(Prof::LightPdf);
#     Vector3f wi = WorldToLight(w);
#     Float theta = SphericalTheta(wi), phi = SphericalPhi(wi);
#     Float sinTheta = std::sin(theta);
#     if (sinTheta == 0) return 0;
#     return distribution->Pdf(Point2f(phi * Inv2Pi, theta * InvPi)) /
#            (2 * Pi * Pi * sinTheta);
# }
