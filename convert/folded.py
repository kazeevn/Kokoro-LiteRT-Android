"""Generator ops on a row-folded time axis, for the Pixel EdgeTPU.

The TPU compiler fails on rows (time axis) longer than ~2k samples, so the vocoder's
generator works on [1, C, H, W] tensors where flat time t = h * W + w. Convolutions read
`halo` columns from the neighbouring rows, so results are identical to the 1-D originals.
Row widths per rate: 2x -> 8, 20x -> 80, 120x -> 480 (H rows shared by all rates).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

W2, W20, W120 = 8, 80, 480


def halo(x, left, right):
    """[1,C,H,W] -> [1,C,H,left+W+right] with columns borrowed from adjacent rows (zeros at ends)."""
    parts = []
    if left:
        z = torch.zeros(x.shape[0], x.shape[1], 1, left)
        parts.append(torch.cat([z, x[:, :, :-1, x.shape[3] - left :]], 2))
    parts.append(x)
    if right:
        z = torch.zeros(x.shape[0], x.shape[1], 1, right)
        parts.append(torch.cat([x[:, :, 1:, :right], z], 2))
    return torch.cat(parts, 3) if len(parts) > 1 else x


def fconv(conv: nn.Conv1d, x, max_taps=None):
    """Conv1d (any stride) on folded input. Output row width = W / stride.
    With `max_taps`, wider kernels run as a sum of convs over groups of at most that many taps
    (the Tensor SDK AOT compiler miscompiles the 11-tap convs at 33 rows)."""
    k, s, d, p = conv.kernel_size[0], conv.stride[0], conv.dilation[0], conv.padding[0]
    W = x.shape[3]
    assert W % s == 0
    # output j (row-relative) reads inputs s*j - p + d*kk, kk in [0, k)
    left = p
    right = max(0, s * (W // s - 1) - p + d * (k - 1) - (W - 1))
    xh = halo(x, left, right)
    w = conv.weight[:, :, None, :]
    if not max_taps or k <= max_taps:
        return F.conv2d(xh, w, conv.bias, stride=(1, s), dilation=(1, d))
    span = s * (W // s - 1) + 1  # input columns covered by one tap across the outputs
    y = None
    for t0 in range(0, k, max_taps):
        t1 = min(k, t0 + max_taps)
        part = xh[:, :, :, d * t0 : d * (t1 - 1) + span]
        z = F.conv2d(part, w[..., t0:t1], conv.bias if t0 == 0 else None, stride=(1, s), dilation=(1, d))
        y = z if y is None else y + z
    return y


def fconv_t(conv: nn.ConvTranspose1d, x):
    """ConvTranspose1d on folded input. Output row width = W * stride."""
    k, s, p = conv.kernel_size[0], conv.stride[0], conv.padding[0]
    W = x.shape[3]
    left = -math.ceil((p - k + 1) / s)
    right = (s - 1 + p) // s
    xh = halo(x, left, right)
    y = F.conv_transpose2d(xh, conv.weight[:, :, None, :], conv.bias, stride=(1, s))
    start = p + s * left
    return y[:, :, :, start : start + s * W]


def fconv_t_poly(conv: nn.ConvTranspose1d, x):
    """fconv_t as a polyphase regular convolution (the Tensor SDK AOT compiler miscompiles these
    strided TRANSPOSE_CONVs). With n + p = s*q + r: y[n] = sum_m x[q-m] w[s*m + r] + b, m < k/s;
    one conv yields all s phases as channels c*s + r, which are then interleaved."""
    k, s, p = conv.kernel_size[0], conv.stride[0], conv.padding[0]
    assert k % s == 0 and p < s
    M, W, C = k // s, x.shape[3], conv.out_channels
    q1 = (s * W - 1 + p) // s  # q runs over [0, q1]
    xh = halo(x, M - 1, max(0, q1 - (W - 1)))
    w = conv.weight  # [C_in, C_out, k]
    taps = [w[:, :, s * (M - 1 - j) : s * (M - 1 - j) + s] for j in range(M)]  # [C_in, C_out, s] each
    wc = torch.stack(taps, -1).permute(1, 2, 0, 3).reshape(C * s, -1, 1, M)
    z = F.conv2d(xh, wc, conv.bias.repeat_interleave(s))  # [1, C*s, H, q1+1]
    T = z.shape[3]
    z = z.reshape(1, C, s, z.shape[2], T).permute(0, 1, 3, 4, 2).reshape(1, C, z.shape[2], T * s)
    return z[:, :, :, p : p + s * W]


def shift_reflect1(x):
    """Folded equivalent of cat([x[1:2], x])[:len]: y[t] = x[t-1], y[0] = x[1]."""
    first = torch.cat([x[:, :, :1, 1:2], x[:, :, :-1, -1:]], 2)
    return torch.cat([first, x[:, :, :, :-1]], 3)


class FRate:
    """Folded frame mask [1,1,H,W] and masked means over the whole (folded) time axis."""

    def __init__(self, m):
        from kokoro_graphs import pool_factors

        self.m = m
        self.wf = pool_factors(m.shape[3])
        self.hf = pool_factors(m.shape[2]) if m.shape[2] > 1 else []
        self.frac = self.mean(m)

    def mean(self, x):
        for f in self.wf:
            x = F.avg_pool2d(x, (1, f))
        for f in self.hf:
            x = F.avg_pool2d(x, (f, 1))
        return x

    def masked_mean(self, x):
        return self.mean(x * self.m) / self.frac


def fadain(mod, x, s, r: FRate):
    h = mod.fc(s)[:, :, None, None]
    gamma, beta = torch.chunk(h, chunks=2, dim=1)
    mean = r.masked_mean(x)
    xc = (x - mean) * r.m
    var = r.mean(xc * xc) / r.frac
    y = (x - mean) * torch.rsqrt(var + mod.norm.eps)
    y = y * mod.norm.weight[None, :, None, None] + mod.norm.bias[None, :, None, None]
    return (1 + gamma) * y + beta


def fsnake(x, a):
    a = a[..., None]
    return x + (1 / a) * torch.sin(a * x) ** 2


def fresblock1(blk, x, s, r: FRate, steps=None, max_taps=None):
    """AdaINResBlock1 on folded input; `steps` limits which of its 3 dilation steps run;
    `max_taps` see fconv."""
    for k, (c1, c2, n1, n2, a1, a2) in enumerate(
        zip(blk.convs1, blk.convs2, blk.adain1, blk.adain2, blk.alpha1, blk.alpha2)
    ):
        if steps is not None and k not in steps:
            continue
        xt = fsnake(fadain(n1, x, s, r), a1)
        xt = fconv(c1, xt * r.m, max_taps)
        xt = fsnake(fadain(n2, xt, s, r), a2)
        xt = fconv(c2, xt * r.m, max_taps)
        x = xt + x
    return x


def fold(x, w):
    """[1,C,n] -> [1,C,n/w,w]"""
    return x.reshape(x.shape[0], x.shape[1], x.shape[2] // w, w)


def unfold(x):
    return x.reshape(x.shape[0], x.shape[1], -1)


def rows_for(L):
    """Folded rows H for frame bucket L: H*480 must cover 120L+1 samples."""
    return L // 4 + 1


# ---------------------------------------------------------------------------------------------
# Row-block processing of the generator's last (120x) rate for the EdgeTPU.
#
# The TPU compiler fails on graphs whose activations exceed ~1M elements, which the last rate
# reaches for chunks over ~32 frames. Convolutions there only need one neighbouring row, and
# instance norms only need per-channel statistics, so the stage runs on fixed blocks of BLOCK_ROWS
# rows (+1 halo row each side). Each graph also emits per-block statistics (mean and centered
# second moment) of what it produces; the host combines them (parallel variance formula) into the
# global mean / rstd fed to the next graph.
# ---------------------------------------------------------------------------------------------
BLOCK_ROWS = 7  # 128 x 9 x 480 activations: the largest that compiles reliably


def block_stats(x, m):
    """Per-channel (mean over valid, mean over block of ((x - mean) * m)^2), each [1,C,1,1]."""
    def avg(v):
        for f in (60, 8):
            v = F.avg_pool2d(v, (1, f))
        return F.avg_pool2d(v, (v.shape[2], 1))
    frac = avg(m)
    mean = avg(x * m) / (frac + 1e-12)
    xc = (x - mean) * m
    return mean, avg(xc * xc)


def badain(mod, x, s, mean, rstd):
    """AdaIN1d with externally supplied instance-norm statistics."""
    h = mod.fc(s)[:, :, None, None]
    gamma, beta = torch.chunk(h, chunks=2, dim=1)
    y = (x - mean) * rstd * mod.norm.weight[None, :, None, None] + mod.norm.bias[None, :, None, None]
    return (1 + gamma) * y + beta


def interior(x):
    return x[:, :, 1:-1, :]


class FinalBlock(nn.Module):
    """One last-rate graph on a row block. Kinds (inputs -> outputs):
      src0:      har [1,24,R,480], m [1,1,R,480]                       -> src0, mean, m2
      <B>s<k>a:  x [1,C,R+2,480], m [1,1,R+2,480], mean, rstd, ref_s  -> h, mean, m2
      <B>s<k>b:  h [1,C,R+2,480], m [1,1,R+2,480], mean, rstd, x [1,C,R,480], ref_s -> out, mean, m2
      g1u:       y0, y1, y2 [1,256,R+2,80], m20 [1,1,R+2,80]            -> x_up [1,128,R,480]
      add:       x_up, src [1,128,R,480], m [1,1,R,480]                 -> xP, mean, m2
      post:      z0, z1, z2 [1,128,R+2,480], m [1,1,R+2,480]            -> out [1,24,R,480]
    <B> is nr (noise_res[1]) or r0/r1/r2 (resblocks[3..5]); step k uses that block's k-th
    (convs1, convs2, adain1, adain2, alpha1, alpha2). R = BLOCK_ROWS.
    """

    def __init__(self, gen, kind):
        super().__init__()
        self.g, self.kind = gen, kind

    def _blk(self):
        b = self.kind[:2]
        return self.g.noise_res[1] if b == "nr" else self.g.resblocks[3 + int(b[1])]

    def forward(self, *a):
        g, kind = self.g, self.kind
        if kind == "src0":
            har, m = a
            y = F.conv2d(har, g.noise_convs[1].weight[:, :, None, :], g.noise_convs[1].bias) * m
            return (y,) + block_stats(y, m)
        if kind == "g1u":
            y0, y1, y2, m20 = a
            x = F.leaky_relu((y0 + y1 + y2) / g.num_kernels, negative_slope=0.1) * m20
            xt = fconv_t_poly(g.ups[1], x)  # [1,128,R+2,480]; rows 1..R are exact
            # shift by one sample (cat([x[1:2], x])): the first sample of row r comes from row r-1
            return torch.cat([xt[:, :, :-2, -1:], xt[:, :, 1:-1, :-1]], 3)
        if kind == "add":
            x_up, src, m = a
            y = (x_up + src) * m
            return (y,) + block_stats(y, m)
        if kind == "post":
            z0, z1, z2, m = a
            x = F.leaky_relu((z0 + z1 + z2) / g.num_kernels)
            return interior(fconv(g.conv_post, x * m))
        blk, k, half = self._blk(), int(kind[3]), kind[4]
        if half == "a":
            x, m, mean, rstd, ref_s = a
            s = ref_s[:, :128]
            h = fsnake(badain(blk.adain1[k], x, s, mean, rstd), blk.alpha1[k])
            h = interior(fconv(blk.convs1[k], h * m)) * interior(m)
            return (h,) + block_stats(h, interior(m))
        h, m, mean, rstd, x, ref_s = a
        s = ref_s[:, :128]
        y = fsnake(badain(blk.adain2[k], h, s, mean, rstd), blk.alpha2[k])
        y = (interior(fconv(blk.convs2[k], y * m)) + x) * interior(m)
        return (y,) + block_stats(y, interior(m))


FINAL_BLOCK_KINDS = ["src0", "g1u", "add", "post"] + [
    f"{b}s{k}{h}" for b in ("nr", "r0", "r1", "r2") for k in range(3) for h in "ab"]


def combine_stats(parts, counts, eps):
    """parts: [(mean [C], m2_over_block [C])], counts: valid elements per block (block size
    R*480 for the m2 normalization). Returns (mean, rstd) as [1,C,1,1] (parallel variance)."""
    size = BLOCK_ROWS * W120
    n = sum(counts)
    mean = sum(c * p[0] for p, c in zip(parts, counts)) / n
    m2 = sum(p[1] * size + c * (p[0] - mean) ** 2 for p, c in zip(parts, counts))
    return mean, torch.rsqrt(m2 / n + eps)


def run_final_blocks(gen, ys, har, m20, mP, ref_s, run=None):  # noqa: C901
    """Host driver for the last rate (mirrors KokoroEngine.kt). ys: 3 x [1,256,H,80],
    har [1,24,H,480], m20 [1,1,H,80], mP [1,1,H,480] (folded, H rows). Returns [1,24,H,480]."""
    run = run or (lambda kind, *args: FinalBlock(gen, kind)(*args))
    R = BLOCK_ROWS
    H = mP.shape[2]
    nb = -(-H // R)
    Hp = nb * R

    def pad_rows(t):
        return F.pad(t, (0, 0, 0, Hp - H))

    ys, har, m20, mP = [pad_rows(y) for y in ys], pad_rows(har), pad_rows(m20), pad_rows(mP)

    def rows(t, b, halo):
        lo, hi = b * R - halo, b * R + R + halo
        parts = [t[:, :, max(lo, 0) : min(hi, Hp)]]
        if lo < 0:
            parts.insert(0, torch.zeros(t.shape[0], t.shape[1], -lo, t.shape[3]))
        if hi > Hp:
            parts.append(torch.zeros(t.shape[0], t.shape[1], hi - Hp, t.shape[3]))
        return torch.cat(parts, 2)

    counts = [int(rows(mP, b, 0).sum()) for b in range(nb)]

    def stats(parts, eps=1e-5):
        keep = [(p, c) for p, c in zip(parts, counts) if c > 0]
        return combine_stats([p for p, _ in keep], [c for _, c in keep], eps)

    def full(blocks):
        return torch.cat(blocks, 2)

    # noise branch
    outs = [run("src0", rows(har, b, 0), rows(mP, b, 0)) for b in range(nb)]
    src, st = full([o[0] for o in outs]), stats([(o[1], o[2]) for o in outs])

    def resblock(name, x, st):
        for k in range(3):
            eps = 1e-5
            outs = [run(f"{name}s{k}a", rows(x, b, 1), rows(mP, b, 1), st[0], st[1], ref_s) for b in range(nb)]
            h, sth = full([o[0] for o in outs]), stats([(o[1], o[2]) for o in outs], eps)
            outs = [run(f"{name}s{k}b", rows(h, b, 1), rows(mP, b, 1), sth[0], sth[1], rows(x, b, 0), ref_s)
                    for b in range(nb)]
            x, st = full([o[0] for o in outs]), stats([(o[1], o[2]) for o in outs], eps)
        return x, st

    src, _ = resblock("nr", src, st)
    x_up = full([run("g1u", *[rows(y, b, 1) for y in ys], rows(m20, b, 1)) for b in range(nb)])
    x_up[:, :, 0, 0] = x_up[:, :, 0, 2]  # reflect pad of the very first sample
    outs = [run("add", rows(x_up, b, 0), rows(src, b, 0), rows(mP, b, 0)) for b in range(nb)]
    xP, st = full([o[0] for o in outs]), stats([(o[1], o[2]) for o in outs])
    zs = [resblock(f"r{j}", xP, st)[0] for j in range(3)]
    out = full([run("post", *[rows(z, b, 1) for z in zs], rows(mP, b, 1)) for b in range(nb)])
    return out[:, :, :H]
