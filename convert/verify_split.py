"""Check the split graphs + numpy host DSP against the original KModel (PyTorch, no LiteRT).

Noise in the hn-NSF source is zeroed in both paths so outputs are comparable.
Usage: python verify_split.py [--tflite DIR]   (with --tflite, runs converted graphs instead)
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import host_dsp  # noqa: E402
from kokoro_graphs import DecoderGraph, F0NGraph, TextGraph, load_kmodel  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
HF = os.path.join(ROOT, "work", "hf")

SENTENCES = [
    "Hello world.",
    "The quick brown fox jumps over the lazy dog, and then it runs away into the forest.",
    "Kokoro is an open-weight TTS model with 82 million parameters. Despite its lightweight "
    "architecture, it delivers comparable quality to larger models while being significantly "
    "faster and more cost-efficient.",
]


def snr_db(ref, x):
    n = min(len(ref), len(x))
    ref, x = ref[:n], x[:n]
    return 10 * np.log10(np.sum(ref**2) / max(np.sum((ref - x) ** 2), 1e-20))


def spec_corr(ref, x):
    n = min(len(ref), len(x))
    S = lambda a: np.abs(np.fft.rfft(a[: n // 480 * 480].reshape(-1, 480) * np.hanning(480), axis=1))
    a, b = np.log1p(S(ref)), np.log1p(S(x))
    return np.corrcoef(a.ravel(), b.ravel())[0, 1]


class TorchRunner:
    def __init__(self, km):
        self.text, self.f0n, self.dec = TextGraph(km).eval(), F0NGraph(km).eval(), DecoderGraph(km).eval()

    def run_text(self, ids, mask, rev, ref_s):
        with torch.no_grad():
            o = self.text(torch.from_numpy(ids), torch.from_numpy(mask), torch.from_numpy(rev), torch.from_numpy(ref_s))
        return [t.numpy() for t in o]

    def run_f0n(self, *args):
        with torch.no_grad():
            return [t.numpy() for t in self.f0n(*[torch.from_numpy(a) for a in args])]

    def run_dec(self, *args):
        with torch.no_grad():
            return self.dec(*[torch.from_numpy(a) for a in args]).numpy()


def split_synth(runner, km, ids_list, ref_s, T, frame_bucket, speed=1.0):
    n_valid = len(ids_list)
    ids = np.zeros((1, T), np.int32)
    ids[0, :n_valid] = ids_list
    mask = np.zeros((1, T), np.float32)
    mask[0, :n_valid] = 1
    rev = np.arange(T, dtype=np.int32)
    rev[:n_valid] = np.arange(n_valid)[::-1]
    dur, d, t_en = runner.run_text(ids, mask, rev, ref_s)
    pd = host_dsp.durations(dur[0], n_valid, speed)
    L0 = int(pd.sum())
    L = frame_bucket(L0)
    fi = host_dsp.frame_inputs(pd, d[0], t_en[0], L)
    f0, n = runner.run_f0n(fi["en"], fi["rev"], fi["m1"], fi["m2"], ref_s)
    f0, n = f0 * fi["m2"][0], n * fi["m2"][0]
    lin = km.decoder.generator.m_source.l_linear
    har = host_dsp.har_features(f0[0, : 2 * L0], lin.weight.detach().numpy(), lin.bias.detach().numpy(),
                                noise=np.zeros((2 * L0 * host_dsp.UPSAMPLE, host_dsp.HARMONICS)))
    har_full = np.zeros((1, 24, fi["P"]), np.float32)
    har_full[0, :22, : har.shape[1]] = har
    dec = km.decoder
    fn = host_dsp.f0n_features(f0[0], n[0], dec.F0_conv.weight.detach().numpy()[0, 0], float(dec.F0_conv.bias[0]),
                               dec.N_conv.weight.detach().numpy()[0, 0], float(dec.N_conv.bias[0]))
    x = runner.run_dec(fi["asr"], fn, har_full, fi["m1"], fi["m2"], fi["m20"], fi["mP"], ref_s)
    audio = host_dsp.istft(x[0][:22, : 120 * L0 + 1])
    return audio[: L0 * 600], L0, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", help="directory with converted graphs (uses ai_edge_litert)")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--bucket", type=int, default=32, help="frame bucket granularity")
    args = ap.parse_args()

    from kokoro import KPipeline

    km = load_kmodel(os.path.join(HF, "config.json"), os.path.join(HF, "kokoro-v1_0.pth"))
    pipe = KPipeline(lang_code="a", model=False)
    pack = torch.load(os.path.join(HF, "voices", f"{args.voice}.pt"), weights_only=True)

    if args.tflite:
        from tflite_runner import TfliteRunner
        runner = TfliteRunner(args.tflite)
    else:
        runner = TorchRunner(km)

    orig_randn = torch.randn_like
    for text in SENTENCES:
        ps = "".join(r.phonemes for r in pipe(text))
        ids_list = [0] + [km.vocab[p] for p in ps if p in km.vocab] + [0]
        ref_s = pack[len(ps) - 1].numpy().astype(np.float32)  # [1, 256]
        torch.randn_like = lambda x, **kw: torch.zeros_like(x)
        try:
            with torch.no_grad():
                ref_audio, ref_dur = km.forward_with_tokens(torch.LongTensor([ids_list]), torch.from_numpy(ref_s), 1.0)
        finally:
            torch.randn_like = orig_randn
        ref_audio = ref_audio.numpy()
        T = 1 << max(5, (len(ids_list) - 1).bit_length())
        bucketed = lambda L0: -(-L0 // args.bucket) * args.bucket
        for name, fb in ((f"bucket{args.bucket}", bucketed),):
            audio, L0, L = split_synth(runner, km, ids_list, ref_s, T, fb)
            print(f"[{name:9s}] T={T} tokens={len(ids_list)} L0={L0} (ref {int(ref_dur.sum())}) L={L} "
                  f"SNR={snr_db(ref_audio, audio):6.2f} dB  logspec_corr={spec_corr(ref_audio, audio):.5f}")


if __name__ == "__main__":
    main()
