"""Export-friendly re-implementation of Kokoro-82M split into three static-shape graphs.

The original KModel.forward_with_tokens has a data-dependent output length, so it is split:

  text graph    (token bucket T): ids, mask, rev_idx, ref_s -> duration logits sum, d, t_en
  host                           : pred_dur = round(dur / speed); stretch last token so sum == L;
                                   en = d^T @ aln, asr = t_en @ aln (column repeats)
  f0n graph     (frame bucket L): en, ref_s -> F0, N           (at 2L)
  host                           : har = STFT(hn-NSF sine source(F0))
  decoder graph (frame bucket L): asr, F0, N, har, ref_s -> conv_post output (22 x (120L+1))
  host                           : spec = exp(x[:11]), phase = sin(x[11:]); iSTFT -> 600L samples

Token graphs are right-padded; masks reproduce pack_padded_sequence exactly: the backward
LSTM direction runs over a per-sequence reversal (rev_idx) so padding never reaches valid
positions. Frame graphs are never padded: the host lengthens the final (silent) boundary token
so that sum(pred_dur) equals the bucket size, and trims the extra audio afterwards.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from kokoro.istftnet import AdaIN1d, AdainResBlk1d, AdaINResBlock1
from kokoro.model import KModel


def strip_weight_norm(model: nn.Module):
    for m in model.modules():
        if hasattr(m, "weight_g") and hasattr(m, "weight_v"):
            torch.nn.utils.remove_weight_norm(m)


class BiLSTM(nn.Module):
    """Unrolled single-layer bidirectional LSTM (batch_first, B=1).

    If rev_idx is given (int [T]: L-1-i for valid i, else i), the backward direction runs over
    the sequence reversed within its valid length, matching pack_padded_sequence semantics.
    """

    def __init__(self, lstm: nn.LSTM):
        super().__init__()
        assert lstm.bidirectional and lstm.num_layers == 1 and lstm.batch_first
        self.H = lstm.hidden_size
        for sfx in ("", "_reverse"):
            w_ih = getattr(lstm, f"weight_ih_l0{sfx}").detach()
            w_hh = getattr(lstm, f"weight_hh_l0{sfx}").detach()
            b = (getattr(lstm, f"bias_ih_l0{sfx}") + getattr(lstm, f"bias_hh_l0{sfx}")).detach()
            name = "b" if sfx else "f"
            self.register_buffer(f"w_ih_{name}", w_ih.t().contiguous())
            self.register_buffer(f"w_hh_{name}", w_hh.t().contiguous())
            self.register_buffer(f"b_{name}", b)

    def _run(self, xw, w_hh):
        # xw: [T, 4H] precomputed input projections
        T = xw.shape[0]
        H = self.H
        h = torch.zeros(1, H, dtype=xw.dtype)
        c = torch.zeros(1, H, dtype=xw.dtype)
        outs = []
        for t in range(T):
            g = xw[t : t + 1] + h @ w_hh
            i = torch.sigmoid(g[:, 0:H])
            f = torch.sigmoid(g[:, H : 2 * H])
            gg = torch.tanh(g[:, 2 * H : 3 * H])
            o = torch.sigmoid(g[:, 3 * H : 4 * H])
            c = f * c + i * gg
            h = o * torch.tanh(c)
            outs.append(h)
        return torch.cat(outs, 0)  # [T, H]

    def forward(self, x, rev_idx=None):
        x = x[0]  # [T, C]
        yf = self._run(x @ self.w_ih_f + self.b_f, self.w_hh_f)
        if rev_idx is None:
            xb = torch.flip(x, [0])
        else:
            xb = x[rev_idx]
        yb = self._run(xb @ self.w_ih_b + self.b_b, self.w_hh_b)
        yb = torch.flip(yb, [0]) if rev_idx is None else yb[rev_idx]
        return torch.cat([yf, yb], -1)[None]  # [1, T, 2H]


class DepthwiseUp2(nn.Module):
    """ConvTranspose1d(C, C, k=3, s=2, p=1, output_padding=1, groups=C) without grouped deconv."""

    def __init__(self, convt: nn.ConvTranspose1d):
        super().__init__()
        w = convt.weight.detach()[:, 0, :]  # [C, 3]
        # contiguous copies: litert-torch serializes strided buffer views incorrectly
        self.register_buffer("w0", w[:, 0:1][None].contiguous().clone())
        self.register_buffer("w1", w[:, 1:2][None].contiguous().clone())
        self.register_buffer("w2", w[:, 2:3][None].contiguous().clone())
        self.register_buffer("bias", convt.bias.detach()[None, :, None].contiguous().clone())

    def forward(self, x):  # [1, C, L] -> [1, C, 2L]
        # y[2i] = w1 x[i];  y[2i+1] = w2 x[i] + w0 x[i+1]. Interleaving is done with a nearest
        # upsample and parity masks: litert-torch mis-converts the stack+reshape interleave.
        x_next = F.pad(x[:, :, 1:], (0, 1))
        up = torch.repeat_interleave(x, 2, dim=-1)
        up_next = torch.repeat_interleave(x_next, 2, dim=-1)
        odd = (torch.arange(up.shape[-1]) % 2).to(x.dtype)[None, None, :]
        even = 1 - odd
        y = up * (self.w1 * even + self.w2 * odd) + up_next * (self.w0 * odd)
        return y + self.bias


def pool_factors(n, max_window=64):
    """Factor n into average-pool windows <= max_window (the EdgeTPU rejects wider pools)."""
    fs, r = [], n
    for f in range(max_window, 1, -1):
        while r % f == 0 and r > 1:
            fs.append(f)
            r //= f
    assert r == 1, f"length {n} is not {max_window}-smooth"
    return fs


def smooth_length(n, multiple=64, max_window=64):
    """Smallest length >= n that is a multiple of `multiple` and pool_factors()-friendly."""
    p = -(-n // multiple) * multiple
    while True:
        try:
            pool_factors(p, max_window)
            return p
        except AssertionError:
            p += multiple


class Rate:
    """Frame mask at one time resolution plus how to reduce/elementwise-map over it."""

    def __init__(self, m):
        self.m = m  # [1, 1, len] 0/1 float
        self.len = m.shape[-1]
        self.factors = pool_factors(self.len)
        self._frac = None

    def mean(self, x):
        """Mean over the time axis of x (already masked), via stacked average pools."""
        for f in self.factors:
            x = F.avg_pool1d(x, f)
        return x

    @property
    def frac(self):  # fraction of valid frames, [1,1,1]
        if self._frac is None:
            self._frac = self.mean(self.m)
        return self._frac

    def masked_mean(self, x):
        return self.mean(x * self.m) / self.frac


def masked_instance_norm(x, r: Rate, norm: nn.InstanceNorm1d):
    """InstanceNorm1d(affine) over valid frames only."""
    mean = r.masked_mean(x)
    xc = (x - mean) * r.m
    var = r.mean(xc * xc) / r.frac
    w = norm.weight[None, :, None]
    b = norm.bias[None, :, None]
    return (x - mean) * torch.rsqrt(var + norm.eps) * w + b


def adain(mod: AdaIN1d, x, s, r: Rate):
    h = mod.fc(s)[:, :, None]
    gamma, beta = torch.chunk(h, chunks=2, dim=1)
    return (1 + gamma) * masked_instance_norm(x, r, mod.norm) + beta


def adain_resblk1d(blk: AdainResBlk1d, pool, x, s, r: Rate, r_up: Rate = None):
    """AdainResBlk1d with frame masking; pads are zeroed before every op that mixes frames."""
    h = adain(blk.norm1, x, s, r)
    h = blk.actv(h)
    if pool is not None:
        h = pool(h * r.m)
        r = r_up
    h = blk.conv1(h * r.m)
    h = adain(blk.norm2, h, s, r)
    h = blk.actv(h)
    h = blk.conv2(h * r.m)
    sc = x
    if pool is not None:
        sc = torch.repeat_interleave(sc, 2, dim=-1)
    if blk.learned_sc:
        sc = blk.conv1x1(sc)
    return (h + sc) * (1.0 / math.sqrt(2.0))


def snake(x, a, fold=True):
    """Snake1D. With `fold`, rows longer than 2048 frames are computed in a folded
    [1, C, n/64, 64] layout: the EdgeTPU compiler rejects SIN on long rows and on fold/unfold
    around a lone SIN."""
    n = x.shape[-1]
    if n <= 2048 or not fold:
        return x + (1 / a) * torch.sin(a * x) ** 2
    assert n % 64 == 0, n
    x4 = x.reshape(x.shape[0], x.shape[1], n // 64, 64)
    a4 = a[..., None]
    return (x4 + (1 / a4) * torch.sin(a4 * x4) ** 2).reshape(x.shape)


def adain_resblock1(blk: AdaINResBlock1, x, s, r: Rate, fold=True):
    for c1, c2, n1, n2, a1, a2 in zip(blk.convs1, blk.convs2, blk.adain1, blk.adain2, blk.alpha1, blk.alpha2):
        xt = snake(adain(n1, x, s, r), a1, fold)
        xt = c1(xt * r.m)
        xt = snake(adain(n2, xt, s, r), a2, fold)
        xt = c2(xt * r.m)
        x = xt + x
    return x


def pad_in_channels(conv: nn.Conv1d, extra: int) -> nn.Conv1d:
    """Copy of conv with `extra` zero-weight input channels appended."""
    c = nn.Conv1d(conv.in_channels + extra, conv.out_channels, conv.kernel_size, conv.stride,
                  conv.padding, conv.dilation, bias=conv.bias is not None)
    with torch.no_grad():
        c.weight.zero_()
        c.weight[:, : conv.in_channels] = conv.weight
        if conv.bias is not None:
            c.bias.copy_(conv.bias)
    return c


def pad_out_channels(conv: nn.Conv1d, extra: int) -> nn.Conv1d:
    c = nn.Conv1d(conv.in_channels, conv.out_channels + extra, conv.kernel_size, conv.stride,
                  conv.padding, conv.dilation, bias=True)
    with torch.no_grad():
        c.weight.zero_()
        c.bias.zero_()
        c.weight[: conv.out_channels] = conv.weight
        c.bias[: conv.out_channels] = conv.bias
    return c


def pad_adain_in(mod: AdaIN1d, extra: int) -> AdaIN1d:
    n = mod.norm.num_features
    out = AdaIN1d(mod.fc.in_features, n + extra)
    with torch.no_grad():
        out.norm.weight.zero_()
        out.norm.bias.zero_()
        out.norm.weight[:n] = mod.norm.weight
        out.norm.bias[:n] = mod.norm.bias
        out.fc.weight.zero_()
        out.fc.bias.zero_()
        out.fc.weight[:n] = mod.fc.weight[:n]  # gamma
        out.fc.bias[:n] = mod.fc.bias[:n]
        out.fc.weight[n + extra : 2 * n + extra] = mod.fc.weight[n:]  # beta
        out.fc.bias[n + extra : 2 * n + extra] = mod.fc.bias[n:]
    out.norm.eps = mod.norm.eps
    return out


def pad_resblk_in(blk: AdainResBlk1d, extra: int):
    """Give an AdainResBlk1d `extra` zero input channels (keeps channel counts multiples of 4)."""
    blk.norm1 = pad_adain_in(blk.norm1, extra)
    blk.conv1 = pad_in_channels(blk.conv1, extra)
    assert blk.learned_sc
    blk.conv1x1 = pad_in_channels(blk.conv1x1, extra)
    if isinstance(blk.pool, nn.ConvTranspose1d):  # depthwise (groups == channels) upsampler
        old = blk.pool
        c = old.in_channels + extra
        new = nn.ConvTranspose1d(c, c, 3, stride=2, groups=c, padding=1, output_padding=1)
        with torch.no_grad():
            new.weight.zero_()
            new.bias.zero_()
            new.weight[: old.in_channels] = old.weight
            new.bias[: old.in_channels] = old.bias
        blk.pool = new


def frame_pools(blocks):
    return nn.ModuleList([DepthwiseUp2(b.pool) if isinstance(b.pool, nn.ConvTranspose1d) else nn.Identity()
                          for b in blocks])


class TextGraph(nn.Module):
    """ids [1,T] int32, mask [1,T] f32 (1=valid), rev_idx [T] int32, ref_s [1,256] ->
    dur [1,T] (sum of sigmoid duration bins, divide by speed on host), d [1,T,640], t_en [1,512,T]."""

    def __init__(self, km: KModel):
        super().__init__()
        self.bert = km.bert
        self.bert_encoder = km.bert_encoder
        p = km.predictor
        de = p.text_encoder
        self.de_lstms = nn.ModuleList([BiLSTM(b) for b in de.lstms if isinstance(b, nn.LSTM)])
        self.de_norms = nn.ModuleList([b for b in de.lstms if not isinstance(b, nn.LSTM)])
        self.dur_lstm = BiLSTM(p.lstm)
        self.duration_proj = p.duration_proj
        te = km.text_encoder
        self.embedding = te.embedding
        self.cnn = te.cnn
        self.te_lstm = BiLSTM(te.lstm)

    def forward(self, ids, mask, rev_idx, ref_s):
        s = ref_s[:, 128:]
        bert_dur = self.bert(ids, attention_mask=mask)
        d_en = self.bert_encoder(bert_dur)  # [1, T, 512]
        m = mask[:, :, None]  # [1, T, 1]
        T = ids.shape[1]
        s_t = s[:, None, :].expand(1, T, s.shape[-1])
        # DurationEncoder
        x = torch.cat([d_en, s_t], -1) * m
        for lstm, norm in zip(self.de_lstms, self.de_norms):
            x = lstm(x, rev_idx)  # [1, T, 512]
            x = norm(x, s)  # AdaLayerNorm on [B, T, C]
            x = torch.cat([x, s_t], -1) * m
        d = x  # [1, T, 640]
        x = self.dur_lstm(d, rev_idx)
        dur = torch.sigmoid(self.duration_proj(x)).sum(-1)  # [1, T]
        # TextEncoder
        e = self.embedding(ids).transpose(1, 2)  # [1, 512, T]
        mt = mask[:, None, :]
        e = e * mt
        for c in self.cnn:
            e = c(e) * mt
        t_en = self.te_lstm(e.transpose(1, 2), rev_idx).transpose(1, 2) * mt
        return dur, d, t_en


class F0NGraph(nn.Module):
    """en [1,640,L] (zero past L0), rev [L] int32, m1 [1,1,L], m2 [1,1,2L], ref_s [1,256]
    -> F0 [1,2L], N [1,2L] (valid for the first 2*L0 entries)."""

    def __init__(self, km: KModel):
        super().__init__()
        p = km.predictor
        self.shared = BiLSTM(p.shared)
        self.F0, self.N = p.F0, p.N
        self.F0_pools, self.N_pools = frame_pools(p.F0), frame_pools(p.N)
        self.F0_proj, self.N_proj = p.F0_proj, p.N_proj

    def _branch(self, blocks, pools, x, s, r1, r2):
        r = r1
        for blk, pool in zip(blocks, pools):
            if isinstance(pool, nn.Identity):
                x = adain_resblk1d(blk, None, x, s, r)
            else:
                x = adain_resblk1d(blk, pool, x, s, r, r2)
                r = r2
        return x

    def forward(self, en, rev, m1, m2, ref_s):
        s = ref_s[:, 128:]
        r1, r2 = Rate(m1), Rate(m2)
        x = self.shared(en.transpose(-1, -2), rev).transpose(-1, -2) * m1
        f0 = self.F0_proj(self._branch(self.F0, self.F0_pools, x, s, r1, r2))
        nn_ = self.N_proj(self._branch(self.N, self.N_pools, x, s, r1, r2))
        return f0[:, 0], nn_[:, 0]


def f0n_downsample(curve, conv: nn.Conv1d):
    """Decoder.F0_conv / N_conv (1->1 channel, k=3, stride 2, pad 1) on the host side, numpy."""
    import numpy as np

    w = conv.weight.detach().numpy()[0, 0]
    b = float(conv.bias.detach()[0])
    x = np.pad(curve, (1, 1))
    return b + w[0] * x[0:-2:2] + w[1] * x[1:-1:2] + w[2] * x[2::2]


class DecoderGraph(nn.Module):
    """Vocoder, laid out for the Pixel EdgeTPU (all channel counts multiples of 4).

    asr [1,512,L], fn [1,4,L] (= F0_conv(F0), N_conv(N), 0, 0 computed on the host),
    har [1,24,P] (22 STFT features + 2 zero channels, zero past 120*L0+1),
    m1 [1,1,L], m2 [1,1,2L], m20 [1,1,20L], mP [1,1,P], ref_s [1,256]
    -> x [1,24,P]: spec = exp(x[:, :11]), phase = sin(x[:, 11:22]); valid for 120*L0+1 frames.
    P = smooth_length(120*L+1) for the TPU experiments; the app's CPU graph (fold_snake=False)
    uses P = (L/4 + 1) * 480, the same padded length as the folded TPU stages.
    """

    def __init__(self, km: KModel, fold_snake=True):
        super().__init__()
        self.fold = fold_snake
        dec = copy.deepcopy(km.decoder)  # padded in place below
        self.dec = dec
        g = self.g = dec.generator
        pad_resblk_in(dec.encode, 2)  # [asr 512, F0, N] -> + 2 zero channels
        for blk in dec.decode:
            pad_resblk_in(blk, 2)  # [x 1024, asr_res 64, F0, N] -> + 2
        g.noise_convs[0] = pad_in_channels(g.noise_convs[0], 2)
        g.noise_convs[1] = pad_in_channels(g.noise_convs[1], 2)
        g.conv_post = pad_out_channels(g.conv_post, 2)
        self.pools = frame_pools(dec.decode)

    def forward(self, asr, fn, har, m1, m2, m20, mP, ref_s):
        r1, r2, r20, rP = Rate(m1), Rate(m2), Rate(m20), Rate(mP)
        x = self.front(asr, fn, r1, r2, ref_s)
        x = self.gen0(x, har, r2, r20, ref_s)
        return self.gen1(x, har, r20, rP, ref_s)

    def front(self, asr, fn, r1, r2, ref_s):
        """Encoder + decode blocks: [1,512,L] -> [1,512,2L]."""
        dec = self.dec
        s = ref_s[:, :128]
        x = torch.cat([asr, fn], 1)
        x = adain_resblk1d(dec.encode, None, x, s, r1)
        asr_res = dec.asr_res(asr)
        res = True
        r = r1
        for block, pool in zip(dec.decode, self.pools):
            if res:
                x = torch.cat([x, asr_res, fn], 1)
            if isinstance(pool, nn.Identity):
                x = adain_resblk1d(block, None, x, s, r)
            else:
                x = adain_resblk1d(block, pool, x, s, r, r2)
                r = r2
                res = False
        return x * r2.m

    def front_block(self, k, x, asr, fn, r1, r2, ref_s):
        """Front split per AdainResBlk1d: k=0 encode ([asr, fn] -> x), k=1..4 decode[k-1]."""
        dec = self.dec
        s = ref_s[:, :128]
        if k == 0:
            return adain_resblk1d(dec.encode, None, torch.cat([asr, fn], 1), s, r1) * r1.m
        block, pool = dec.decode[k - 1], self.pools[k - 1]
        x = torch.cat([x, dec.asr_res(asr), fn], 1)
        if isinstance(pool, nn.Identity):
            return adain_resblk1d(block, None, x, s, r1) * r1.m
        return adain_resblk1d(block, pool, x, s, r1, r2) * r2.m

    def gen0(self, x, har, r2, r20, ref_s):
        """Generator stage 0: [1,512,2L] -> [1,256,20L]."""
        g = self.g
        s = ref_s[:, :128]
        L = r2.len // 2
        x = F.leaky_relu(x, negative_slope=0.1)
        x_source = g.noise_convs[0](har[:, :, : 120 * L + 1])
        x_source = adain_resblock1(g.noise_res[0], x_source, s, r20, self.fold)
        x = g.ups[0](x * r2.m) + x_source
        return self._resblocks(0, x, s, r20) * r20.m

    def gen1(self, x, har, r20, rP, ref_s):
        """Generator stage 1: [1,256,20L] -> conv_post output [1,24,P]."""
        g = self.g
        s = ref_s[:, :128]
        x = F.leaky_relu(x, negative_slope=0.1)
        x_source = g.noise_convs[1](har)
        x_source = adain_resblock1(g.noise_res[1], x_source, s, rP, self.fold)
        x = g.ups[1](x * r20.m)
        x = torch.cat([x[:, :, 1:2], x, torch.zeros(1, x.shape[1], rP.len - x.shape[-1] - 1)], -1)
        x = x + x_source
        x = self._resblocks(1, x, s, rP)
        x = F.leaky_relu(x)
        return g.conv_post(x * rP.m)

    def _resblocks(self, i, x, s, r):
        g = self.g
        xs = None
        for j in range(g.num_kernels):
            y = adain_resblock1(g.resblocks[i * g.num_kernels + j], x, s, r, self.fold)
            xs = y if xs is None else xs + y
        return xs / g.num_kernels


class DecoderStage(nn.Module):
    """DecoderGraph split into EdgeTPU-sized graphs; the generator runs row-folded (folded.py).
    H = L/4 + 1 rows. Stages and their inputs, in order:
      front: asr [1,512,L], fn [1,4,L], m1 [1,1,L], m2 [1,1,2L], ref_s  -> x2 [1,512,2L]
      g0:    x2, har [1,24,H,480], m2f [1,1,H,8], m20 [1,1,H,80], ref_s -> x20 [1,256,H,80]
             (on the EdgeTPU split in two, as one program it times out at 96+ frames:
      g0s:   har, m20, ref_s                                          -> src [1,256,H,80]
      g0u:   x2, src, m20                                             -> x20)
             (g0u derives m2f from m20 and upsamples with a polyphase conv: the AOT compiler
             miscompiles the strided TRANSPOSE_CONV)
      rb0j:  x20, m20, ref_s                                         -> y_j  (j = 0..2)
      g1:    y0, y1, y2, har, m20, mP [1,1,H,480], ref_s             -> xP [1,128,H,480]
      rb1j:  xP, mP, ref_s                                           -> z_j  (j = 0..2)
      post:  z0, z1, z2, mP                                          -> out [1,24,H,480]
    """

    STAGES = ["front", "fr0", "fr1", "fr2", "fr3", "fr4", "g0", "g0s", "g0u", "rb00", "rb01", "rb02", "g1", "rb10", "rb11", "rb12", "post"]

    def __init__(self, dec: DecoderGraph, stage: str):
        super().__init__()
        assert stage in self.STAGES or stage[:4] in self.STAGES, stage
        self.dec, self.stage = dec, stage

    def forward(self, *a):
        import folded as fd

        d, g, st = self.dec, self.dec.g, self.stage
        if st == "front":
            return d.front(a[0], a[1], Rate(a[2]), Rate(a[3]), a[4])
        if st.startswith("fr"):  # fr0: asr, fn, m1, m2, ref_s ; fr1..fr4: x, asr, fn, m1, m2, ref_s
            k = int(st[2])
            if k == 0:
                return d.front_block(0, None, a[0], a[1], Rate(a[2]), Rate(a[3]), a[4])
            return d.front_block(k, a[0], a[1], a[2], Rate(a[3]), Rate(a[4]), a[5])
        if st == "g0":
            x2, har, m2f, r20, ref_s = a[0], a[1], a[2], fd.FRate(a[3]), a[4]
            x = F.leaky_relu(x2, negative_slope=0.1)
            x = F.pad(x, (0, m2f.shape[2] * fd.W2 - x.shape[-1]))
            x = fd.fold(x, fd.W2) * m2f
            src = fd.fresblock1(g.noise_res[0], fd.fconv(g.noise_convs[0], har), ref_s[:, :128], r20)
            return (fd.fconv_t(g.ups[0], x) + src) * r20.m
        if st == "g0s":
            har, r20, ref_s = a[0], fd.FRate(a[1]), a[2]
            return fd.fresblock1(g.noise_res[0], fd.fconv(g.noise_convs[0], har), ref_s[:, :128], r20) * r20.m
        if st == "g0u":
            x2, src, m20 = a
            # prefix masks: 2x-rate sample i is valid iff 20x-rate sample 10*i is
            m2f = m20.reshape(*m20.shape[:3], fd.W2, fd.W20 // fd.W2)[..., 0]
            x = F.leaky_relu(x2, negative_slope=0.1)
            x = F.pad(x, (0, m2f.shape[2] * fd.W2 - x.shape[-1]))
            x = fd.fold(x, fd.W2) * m2f
            return (fd.fconv_t_poly(g.ups[0], x) + src) * m20
        if st.startswith("rb"):  # rb<i><j> or rb<i><j>s<k> (single dilation step k)
            i, j = int(st[2]), int(st[3])
            steps = [int(st[5])] if len(st) > 4 else None
            r = fd.FRate(a[1])
            return fd.fresblock1(g.resblocks[i * g.num_kernels + j], a[0], a[2][:, :128], r, steps, max_taps=7) * r.m
        if st == "g1":
            y0, y1, y2, har, r20, rP, ref_s = a[0], a[1], a[2], a[3], fd.FRate(a[4]), fd.FRate(a[5]), a[6]
            x = F.leaky_relu((y0 + y1 + y2) / g.num_kernels, negative_slope=0.1)
            src = fd.fresblock1(g.noise_res[1], fd.fconv(g.noise_convs[1], har), ref_s[:, :128], rP)
            x = fd.shift_reflect1(fd.fconv_t(g.ups[1], x * r20.m))
            return (x + src) * rP.m
        if st == "post":
            x = F.leaky_relu((a[0] + a[1] + a[2]) / g.num_kernels)
            return fd.fconv(g.conv_post, x * a[3])
        raise ValueError(st)


def run_decoder_stages(dec: DecoderGraph, asr, fn, har_flat, m1, m2, L0, ref_s, run=None):
    """Reference driver for the staged/folded decoder (mirrors the Android host code).
    har_flat: [1,24,>=120*L0+1] zero-padded. Returns conv_post output [1,24,H*480] (flat)."""
    import folded as fd

    run = run or (lambda st, *args: DecoderStage(dec, st)(*args))
    L = asr.shape[-1]
    H = fd.rows_for(L)

    def fmask(valid, w):
        m = torch.zeros(1, 1, H * w)
        m[..., :valid] = 1
        return m.reshape(1, 1, H, w)

    m2f, m20, mP = fmask(2 * L0, fd.W2), fmask(20 * L0, fd.W20), fmask(120 * L0 + 1, fd.W120)
    har = torch.zeros(1, 24, H * fd.W120)
    n = min(har_flat.shape[-1], H * fd.W120)
    har[..., :n] = har_flat[..., :n]
    har = har.reshape(1, 24, H, fd.W120)
    x2 = run("front", asr, fn, m1, m2, ref_s)
    x20 = run("g0", x2, har, m2f, m20, ref_s)
    ys = [run(f"rb0{j}", x20, m20, ref_s) for j in range(3)]
    xP = run("g1", *ys, har, m20, mP, ref_s)
    zs = [run(f"rb1{j}", xP, mP, ref_s) for j in range(3)]
    out = run("post", *zs, mP)
    return out.reshape(1, 24, -1)


def load_kmodel(config_path, weights_path) -> KModel:
    km = KModel(repo_id="hexgrad/Kokoro-82M", config=config_path, model=weights_path).eval()
    strip_weight_norm(km)
    return km
