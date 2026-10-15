"""Single-op micro models used to learn what the Pixel EdgeTPU (darwinn NNAPI HAL) accepts.

  python micro_ops.py OUTDIR
"""
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

C, T = 128, 64


class Conv(nn.Module):
    def __init__(self, k=3, d=1):
        super().__init__()
        self.c = nn.Conv1d(C, C, k, dilation=d, padding=d * (k - 1) // 2)

    def forward(self, x):
        return self.c(x)


class ConvMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv1d(C, C, 3, padding=1)

    def forward(self, x, m):
        return self.c(x * m)


class INorm(nn.Module):
    def forward(self, x, m):
        n = m.sum(-1, keepdim=True)
        mean = (x * m).sum(-1, keepdim=True) / n
        xc = (x - mean) * m
        var = (xc * xc).sum(-1, keepdim=True) / n
        return (x - mean) * torch.rsqrt(var + 1e-5)


class INormPool(nn.Module):
    """Masked instance norm whose time reductions are full-window average pools."""

    def forward(self, x, m, scale):
        L = x.shape[-1]
        mean = F.avg_pool1d(x * m, L) * scale  # scale = L / n_valid
        xc = (x - mean) * m
        var = F.avg_pool1d(xc * xc, L) * scale
        return (x - mean) * torch.rsqrt(var + 1e-5)


class Pool64(nn.Module):
    def forward(self, x):
        return F.avg_pool1d(F.avg_pool1d(x, 64), x.shape[-1] // 64)


class Sin(nn.Module):
    def __init__(self, shape=None):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        if self.shape is None:
            return torch.sin(x)
        return torch.sin(x.reshape(self.shape)).reshape(x.shape)


class INormPool2(nn.Module):
    def forward(self, x, m):
        def mean(v):
            return F.avg_pool1d(F.avg_pool1d(v, 64), v.shape[-1] // 64)
        frac = mean(m)
        mu = mean(x * m) / frac
        xc = (x - mu) * m
        return (x - mu) * torch.rsqrt(mean(xc * xc) / frac + 1e-5)


class SnakeFold(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.rand(1, C, 1) + 0.5)

    def forward(self, x):
        y = torch.sin((self.a * x).reshape(1, C, -1, 64)).reshape(x.shape)
        return x + (1 / self.a) * y ** 2


class SnakeFold4(nn.Module):
    """Whole snake computed in a folded [1, C, n/64, 64] layout."""

    def __init__(self, sq="pow"):
        super().__init__()
        self.a = nn.Parameter(torch.rand(1, C, 1, 1) + 0.5)
        self.sq = sq

    def forward(self, x):
        x4 = x.reshape(1, C, -1, 64)
        y = torch.sin(self.a * x4)
        y = y * y if self.sq == "mul" else y ** 2
        return (x4 + (1 / self.a) * y).reshape(x.shape)


class INormPlain(nn.Module):
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        xc = x - mean
        var = (xc * xc).mean(-1, keepdim=True)
        return xc * torch.rsqrt(var + 1e-5)


class Snake(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.rand(1, C, 1) + 0.5)

    def forward(self, x):
        return x + (1 / self.a) * torch.sin(self.a * x) ** 2


class Leaky(nn.Module):
    def forward(self, x):
        return F.leaky_relu(x, 0.2)


class Up(nn.Module):
    def __init__(self):
        super().__init__()
        self.u = nn.ConvTranspose1d(C, C // 2, 20, 10, padding=5)

    def forward(self, x):
        return self.u(x)


class DwUp(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.rand(1, C, 3))

    def forward(self, x):
        x_next = F.pad(x[:, :, 1:], (0, 1))
        y = torch.stack([self.w[..., 1:2] * x, self.w[..., 2:3] * x + self.w[..., 0:1] * x_next], -1)
        return y.reshape(1, C, -1)


class Reflect(nn.Module):
    def forward(self, x):
        return F.pad(x, (1, 0), mode="reflect")


class AdaFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 2 * C)

    def forward(self, x, s):
        h = self.fc(s)[:, :, None]
        g, b = torch.chunk(h, 2, 1)
        return (1 + g) * x + b


class Nearest(nn.Module):
    def forward(self, x):
        return torch.repeat_interleave(x, 2, dim=-1)


class Exp(nn.Module):
    def forward(self, x):
        return torch.exp(x)


MODELS = {
    "conv3": (Conv(3), lambda: (torch.randn(1, C, T),)),
    "conv7d3": (Conv(7, 3), lambda: (torch.randn(1, C, T),)),
    "convmask": (ConvMask(), lambda: (torch.randn(1, C, T), torch.ones(1, 1, T))),
    "inorm": (INorm(), lambda: (torch.randn(1, C, T), torch.ones(1, 1, T))),
    "inorm_pool": (INormPool(), lambda: (torch.randn(1, C, T), torch.ones(1, 1, T), torch.ones(1, 1, 1))),
    "inorm_pool_long": (INormPool(), lambda: (torch.randn(1, C, 3841), torch.ones(1, 1, 3841), torch.ones(1, 1, 1))),
    "conv3_3841": (Conv(3), lambda: (torch.randn(1, C, 3841),)),
    "conv3_1920": (Conv(3), lambda: (torch.randn(1, C, 1920),)),
    "snake_3841": (Snake(), lambda: (torch.randn(1, C, 3841),)),
    "pool64_3840": (Pool64(), lambda: (torch.randn(1, C, 3840),)),
    "sin_1024": (Sin(), lambda: (torch.randn(1, C, 1024),)),
    "sin_2048": (Sin(), lambda: (torch.randn(1, C, 2048),)),
    "sin_3841": (Sin(), lambda: (torch.randn(1, C, 3841),)),
    "sin_3841_4d": (Sin((1, C, 23, 167)), lambda: (torch.randn(1, C, 3841),)),
    "conv11d5_3904": (Conv(11, 5), lambda: (torch.randn(1, C, 3904),)),
    "conv11d1_3904": (Conv(11, 1), lambda: (torch.randn(1, C, 3904),)),
    "inormpool_3904": (INormPool2(), lambda: (torch.randn(1, C, 3904), torch.ones(1, 1, 3904))),
    "snakefold_3904": (SnakeFold(), lambda: (torch.randn(1, C, 3904),)),
    "sinf_61x64": (Sin((1, C, 61, 64)), lambda: (torch.randn(1, C, 3904),)),
    "sinf_64x61": (Sin((1, C, 64, 61)), lambda: (torch.randn(1, C, 3904),)),
    "sinf_244x16": (Sin((1, C, 244, 16)), lambda: (torch.randn(1, C, 3904),)),
    "sinf_16x244": (Sin((1, C, 16, 244)), lambda: (torch.randn(1, C, 3904),)),
    "sinf_8x488": (Sin((1, C, 8, 488)), lambda: (torch.randn(1, C, 3904),)),
    "sinf_flat": (Sin((1, 1, C * 61, 64)), lambda: (torch.randn(1, C, 3904),)),
    "snakef4_3904": (SnakeFold4(), lambda: (torch.randn(1, C, 3904),)),
    "snakef4mul_3904": (SnakeFold4("mul"), lambda: (torch.randn(1, C, 3904),)),
    "inorm_plain": (INormPlain(), lambda: (torch.randn(1, C, T),)),
    "snake": (Snake(), lambda: (torch.randn(1, C, T),)),
    "leaky": (Leaky(), lambda: (torch.randn(1, C, T),)),
    "up": (Up(), lambda: (torch.randn(1, C, T),)),
    "dwup": (DwUp(), lambda: (torch.randn(1, C, T),)),
    "reflect": (Reflect(), lambda: (torch.randn(1, C, T),)),
    "adafc": (AdaFC(), lambda: (torch.randn(1, C, T), torch.randn(1, 128))),
    "nearest": (Nearest(), lambda: (torch.randn(1, C, T),)),
    "exp": (Exp(), lambda: (torch.randn(1, C, T),)),
}


def main():
    import litert_torch

    out = sys.argv[1]
    only = sys.argv[2:] or list(MODELS)
    os.makedirs(out, exist_ok=True)
    for name in only:
        mod, mk = MODELS[name][:2]
        if mod is None:
            mk_part = _rb_parts_h() if len(MODELS[name]) > 3 else _rb_parts()
            mod = mk_part(MODELS[name][2])
        x = mk()
        edge = litert_torch.convert(mod.eval(), x, enable_x64=False)
        p = os.path.join(out, f"micro_{name}.tflite")
        edge.export(p)
        with open(p[:-7] + ".json", "w") as fh:
            json.dump({"inputs": [{"dtype": "f32", "shape": list(t.shape)} for t in x]}, fh)
        print("wrote", p, flush=True)




def _rb_parts():
    """Progressively larger prefixes of one AdaINResBlock1 step at the generator's last rate."""
    sys.path.insert(0, os.path.dirname(__file__))
    from kokoro_graphs import Rate, adain, snake
    from kokoro.istftnet import AdaINResBlock1

    blk = AdaINResBlock1(128, 3, (1, 3, 5), 128).eval()
    for m in blk.modules():
        if hasattr(m, "weight_g"):
            torch.nn.utils.remove_weight_norm(m)

    class Part(nn.Module):
        def __init__(self, level):
            super().__init__()
            self.blk, self.level = blk, level

        def forward(self, x, m, s):
            r = Rate(m)
            b = self.blk
            y = adain(b.adain1[0], x, s, r)
            if self.level >= 1:
                y = snake(y, b.alpha1[0])
            if self.level >= 2:
                y = b.convs1[0](y * r.m)
            if self.level >= 3:
                y = snake(adain(b.adain2[0], y, s, r), b.alpha2[0])
            if self.level >= 4:
                y = b.convs2[0](y * r.m) + x
            return y

    return Part


def _rb_parts_h():
    """Same as _rb_parts but with time on the H axis: tensors [1, C, T, 1], (k x 1) convs."""
    from kokoro.istftnet import AdaINResBlock1

    blk = AdaINResBlock1(128, 3, (1, 3, 5), 128).eval()
    for m in blk.modules():
        if hasattr(m, "weight_g"):
            torch.nn.utils.remove_weight_norm(m)

    def conv2(c):
        n = nn.Conv2d(c.in_channels, c.out_channels, (c.kernel_size[0], 1), padding=(c.padding[0], 0),
                      dilation=(c.dilation[0], 1))
        n.weight.data = c.weight.data[..., None]
        n.bias.data = c.bias.data
        return n

    def mean(v):  # over H
        for f in (64, v.shape[2] // 64):
            v = F.avg_pool2d(v, (f, 1))
        return v

    def adain(mod, x, s, m):
        h = mod.fc(s)[:, :, None, None]
        g, b = torch.chunk(h, 2, 1)
        frac = mean(m)
        mu = mean(x * m) / frac
        xc = (x - mu) * m
        y = (x - mu) * torch.rsqrt(mean(xc * xc) / frac + 1e-5)
        y = y * mod.norm.weight[None, :, None, None] + mod.norm.bias[None, :, None, None]
        return (1 + g) * y + b

    def snake(x, a):
        a = a[..., None]
        return x + (1 / a) * torch.sin(a * x) ** 2

    class Part(nn.Module):
        def __init__(self, level):
            super().__init__()
            self.level = level
            self.c1, self.c2 = conv2(blk.convs1[0]), conv2(blk.convs2[0])

        def forward(self, x, m, s):
            x, m = x[..., None], m[..., None]
            y = adain(blk.adain1[0], x, s, m)
            if self.level >= 1:
                y = snake(y, blk.alpha1[0])
            if self.level >= 2:
                y = self.c1(y * m)
            if self.level >= 3:
                y = snake(adain(blk.adain2[0], y, s, m), blk.alpha2[0])
            if self.level >= 4:
                y = self.c2(y * m) + x
            return y[..., 0]

    return Part


for _lvl in (0, 4):
    MODELS[f"rbh{_lvl}_3904"] = (None, lambda: (torch.randn(1, C, 3904), torch.ones(1, 1, 3904), torch.randn(1, 128)), _lvl, "h")

for _lvl in range(5):
    for _n in (1984, 3904):
        MODELS[f"rbpart{_lvl}_{_n}"] = (None, (lambda n: lambda: (torch.randn(1, C, n), torch.ones(1, 1, n), torch.randn(1, 128)))(_n), _lvl)


if __name__ == "__main__":
    main()
