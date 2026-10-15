"""Host-side (non-neural) steps of the split Kokoro pipeline, in numpy.

This is the executable spec for the Kotlin implementation in the Android app
(KokoroDsp.kt); keep them in sync.
"""
import numpy as np

SR = 24000
UPSAMPLE = 300  # prod(upsample_rates) * istft hop = 10 * 6 * 5
N_FFT = 20
HOP = 5
HARMONICS = 9
SINE_AMP = 0.1
NOISE_STD = 0.003
VOICED_THRESHOLD = 10.0


def hann(n=N_FFT):
    k = np.arange(n)
    return (0.5 - 0.5 * np.cos(2 * np.pi * k / n)).astype(np.float64)  # periodic


def align(pred_dur, L):
    """Frame -> token index map for durations (sum == L)."""
    assert pred_dur.sum() == L, (pred_dur.sum(), L)
    return np.repeat(np.arange(len(pred_dur)), pred_dur)


def frame_inputs(pred_dur, d, t_en, L):
    """Expand token features to frames (zero-padded to bucket L) and build masks.
    d: [T, 640], t_en: [512, T]. Returns float32 arrays with a leading batch dim."""
    L0 = int(pred_dur.sum())
    idx = np.repeat(np.arange(len(pred_dur)), pred_dur)
    en = np.zeros((1, d.shape[1], L), np.float32)
    en[0, :, :L0] = d[idx].T
    asr = np.zeros((1, t_en.shape[0], L), np.float32)
    asr[0, :, :L0] = t_en[:, idx]
    rev = np.arange(L, dtype=np.int32)
    rev[:L0] = np.arange(L0)[::-1]

    def mask(n_valid, n):
        m = np.zeros((1, 1, n), np.float32)
        m[..., :n_valid] = 1
        return m

    P = last_stage_length(L)
    return dict(en=en, asr=asr, rev=rev, m1=mask(L0, L), m2=mask(2 * L0, 2 * L),
                m20=mask(20 * L0, 20 * L), mP=mask(120 * L0 + 1, P), P=P)


def last_stage_length(L, multiple=64, max_window=64):
    """Padded length of the decoder's last stage: smallest multiple of 64 >= 120L+1 whose
    factors are all <= 64 (average-pool windows on the EdgeTPU). Mirrors smooth_length()."""
    def smooth(n):
        for f in range(max_window, 1, -1):
            while n % f == 0 and n > 1:
                n //= f
        return n == 1

    p = -(-(120 * L + 1) // multiple) * multiple
    while not smooth(p):
        p += multiple
    return p


def f0n_features(f0, n, w_f0, b_f0, w_n, b_n):
    """Decoder.F0_conv / N_conv (k=3, stride 2, pad 1) -> fn [1, 4, len/2] (2 zero channels)."""
    def conv(c, w, b):
        x = np.pad(c.astype(np.float64), (1, 1))
        return b + w[0] * x[0:-2:2] + w[1] * x[1:-1:2] + w[2] * x[2::2]

    out = np.zeros((1, 4, len(f0) // 2), np.float32)
    out[0, 0] = conv(f0, w_f0, b_f0)
    out[0, 1] = conv(n, w_n, b_n)
    return out


def durations(dur, n_valid, speed, L_bucket=None):
    """dur: [T] graph output. Returns int durations for the n_valid tokens; when L_bucket is given
    the final (boundary/silence) token absorbs the slack so the total equals L_bucket."""
    d = np.maximum(np.round(dur[:n_valid] / speed), 1).astype(np.int64)
    if L_bucket is not None:
        slack = L_bucket - d.sum()
        assert slack >= 0
        d[-1] += slack
    return d


def harmonic_source(f0, l_linear_w, l_linear_b, noise=None):
    """f0: [2L] F0 curve from the f0n graph. Returns har_source [600L] (float64).

    Mirrors SineGen._f02sine + SourceModuleHnNSF: the random initial phase in SineGen is
    discarded by its linear down-sampling, so the phase is a deterministic function of f0.
    """
    n_frames = f0.shape[0]
    n = n_frames * UPSAMPLE
    h = np.arange(1, HARMONICS + 1, dtype=np.float64)
    rad = np.mod(f0.astype(np.float64)[:, None] * h[None, :] / SR, 1.0)  # [F, 9]
    phase = np.cumsum(rad, 0) * 2 * np.pi * UPSAMPLE  # [F, 9]
    # F.interpolate(linear, align_corners=False, scale_factor=UPSAMPLE)
    src = (np.arange(n) + 0.5) / UPSAMPLE - 0.5
    src = np.maximum(src, 0.0)
    i0 = np.floor(src).astype(np.int64)
    i1 = np.minimum(i0 + 1, n_frames - 1)
    lam = (src - i0)[:, None]
    ph = phase[i0] * (1 - lam) + phase[i1] * lam  # [n, 9]
    sines = np.sin(ph) * SINE_AMP
    f0_up = np.repeat(f0.astype(np.float64), UPSAMPLE)
    uv = (f0_up > VOICED_THRESHOLD).astype(np.float64)[:, None]
    noise_amp = uv * NOISE_STD + (1 - uv) * SINE_AMP / 3
    if noise is None:
        noise = np.random.randn(n, HARMONICS)
    sw = sines * uv + noise_amp * noise
    return np.tanh(sw @ l_linear_w.astype(np.float64).T[:, 0] + l_linear_b.astype(np.float64)[0])


def stft_mag_phase(x):
    """torch.stft(n_fft=20, hop=5, hann, center=True, reflect) -> (mag, phase) [11, n/5+1]."""
    pad = N_FFT // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    n_frames = 1 + (len(xp) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = xp[idx] * hann()[None, :]
    spec = np.fft.rfft(frames, axis=1).T  # [11, F]
    return np.abs(spec), np.angle(spec)


def har_features(f0, l_linear_w, l_linear_b, noise=None):
    mag, ph = stft_mag_phase(harmonic_source(f0, l_linear_w, l_linear_b, noise))
    return np.concatenate([mag, ph], 0).astype(np.float32)  # [22, 120L+1]


def istft(x):
    """x: conv_post output [22, F]. spec=exp(x[:11]), phase=sin(x[11:]); torch.istft(center=True)."""
    x = x.astype(np.float64)
    spec = np.exp(x[: N_FFT // 2 + 1]) * np.exp(1j * np.sin(x[N_FFT // 2 + 1 :]))
    frames = np.fft.irfft(spec.T, n=N_FFT, axis=1) * hann()[None, :]  # [F, 20]
    F_ = frames.shape[0]
    out_len = N_FFT + HOP * (F_ - 1)
    y = np.zeros(out_len)
    env = np.zeros(out_len)
    w2 = hann() ** 2
    for t in range(F_):
        y[t * HOP : t * HOP + N_FFT] += frames[t]
        env[t * HOP : t * HOP + N_FFT] += w2
    pad = N_FFT // 2
    y = y[pad : pad + HOP * (F_ - 1)]
    env = env[pad : pad + HOP * (F_ - 1)]
    return (y / np.where(env > 1e-11, env, 1.0)).astype(np.float32)
