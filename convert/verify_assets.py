"""End-to-end check of the exported app assets (multi-signature .tflite) on the host.

Runs text -> f0n -> host DSP -> staged/folded decoder exactly as KokoroEngine.kt sequences them,
and compares against the original PyTorch KModel (hn-NSF noise disabled in both).
  python verify_assets.py [--wav-dir DIR]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from ai_edge_litert.interpreter import Interpreter, OpResolverType

sys.path.insert(0, os.path.dirname(__file__))
import host_dsp  # noqa: E402
from export_all import OUT  # noqa: E402
from kokoro_graphs import load_kmodel  # noqa: E402
from verify_split import HF, SENTENCES, snr_db, spec_corr  # noqa: E402


class Assets:
    def __init__(self, d=OUT):
        self.meta = json.load(open(os.path.join(d, "kokoro.json")))
        self.d = d
        self.its = {}

    def run(self, name, sig, *args):
        if name not in self.its:
            # builtin kernels: the default XNNPACK delegate packs shared weights per op and per
            # signature (unrolled LSTMs, bucketed models) and runs out of memory
            it = Interpreter(model_path=os.path.join(self.d, f"{name}.tflite"), num_threads=8,
                             experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
            self.its[name] = it
        runner = self.its[name].get_signature_runner(sig)
        out = runner(**{f"args_{i}": np.ascontiguousarray(a) for i, a in enumerate(args)})
        return [out[f"output_{i}"] for i in range(len(out))]


def synth(A: Assets, ids_list, ref_s, speed=1.0, cpu=False):
    m = A.meta
    n = len(ids_list)
    T = next(t for t in m["token_buckets"] if t >= n)
    ids = np.zeros((1, T), np.int32); ids[0, :n] = ids_list
    mask = np.zeros((1, T), np.float32); mask[0, :n] = 1
    rev = np.arange(T, dtype=np.int32); rev[:n] = np.arange(n)[::-1]
    dur, d, t_en = A.run("text", f"T{T}", ids, mask, rev, ref_s)
    pd = np.maximum(np.round(dur[0, :n] / speed), 1).astype(np.int64)
    L0 = int(pd.sum())
    L = next(b for b in m["frame_buckets"] if b >= L0)
    H = L // 4 + 1
    fi = host_dsp.frame_inputs(pd, d[0], t_en[0], L)
    f0, nn_ = A.run("f0n", f"L{L}", fi["en"], fi["rev"], fi["m1"], fi["m2"], ref_s)
    f0[0, 2 * L0:] = 0
    nn_[0, 2 * L0:] = 0
    har = host_dsp.har_features(f0[0, : 2 * L0], np.array(m["l_linear_w"])[None, :], np.array([m["l_linear_b"]]),
                                noise=np.zeros((2 * L0 * 300, 9)))
    harf = np.zeros((1, 24, H * 480), np.float32)
    harf[0, :22, : har.shape[1]] = har
    harf = harf.reshape(1, 24, H, 480)
    fn = host_dsp.f0n_features(f0[0], nn_[0], m["f0_conv_w"], m["f0_conv_b"], m["n_conv_w"], m["n_conv_b"])

    def fm(valid, w):
        a = np.zeros((1, 1, H * w), np.float32); a[..., :valid] = 1
        return a.reshape(1, 1, H, w)

    m2f, m20, mP = fm(2 * L0, 8), fm(20 * L0, 80), fm(120 * L0 + 1, 480)
    sig = f"L{L}"
    if cpu:  # single 1-D graph, same buffers flattened
        (out,) = A.run("dec_cpu", sig, fi["asr"], fn, harf.reshape(1, 24, -1), fi["m1"], fi["m2"],
                       m20.reshape(1, 1, -1)[..., : 20 * L], mP.reshape(1, 1, -1), ref_s)
        out = out.reshape(24, -1)
        return host_dsp.istft(out[:22, : 120 * L0 + 1]), L0, L
    (x2,) = A.run("dec_front", sig, fi["asr"], fn, fi["m1"], fi["m2"], ref_s)
    (x20,) = A.run("dec_g0", sig, x2, harf, m2f, m20, ref_s)
    ys = [A.run(f"dec_rb0{j}", sig, x20, m20, ref_s)[0] for j in range(3)]
    (xP,) = A.run("dec_g1", sig, *ys, harf, m20, mP, ref_s)
    zs = []
    for j in range(3):
        z = xP
        for k in range(3):
            (z,) = A.run(f"dec_rb1{j}s{k}", sig, z, mP, ref_s)
        zs.append(z)
    (out,) = A.run("dec_post", sig, *zs, mP)
    out = out.reshape(24, -1)
    return host_dsp.istft(out[:22, : 120 * L0 + 1]), L0, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--wav-dir")
    args = ap.parse_args()
    from kokoro import KPipeline

    km = load_kmodel(os.path.join(HF, "config.json"), os.path.join(HF, "kokoro-v1_0.pth"))
    pipe = KPipeline(lang_code="a", model=False)
    pack = torch.load(os.path.join(HF, "voices", f"{args.voice}.pt"), weights_only=True)
    A = Assets()
    orig = torch.randn_like
    for i, text in enumerate(SENTENCES[:2]):
        ps = "".join(r.phonemes for r in pipe(text))
        ids_list = [0] + [km.vocab[p] for p in ps if p in km.vocab] + [0]
        if len(ids_list) > A.meta["token_buckets"][-1]:
            print("skip (too long for one chunk):", text[:40]); continue
        ref_s = pack[len(ps) - 1].numpy().astype(np.float32)
        torch.randn_like = lambda x, **kw: torch.zeros_like(x)
        try:
            with torch.no_grad():
                ref, _ = km.forward_with_tokens(torch.LongTensor([ids_list]), torch.from_numpy(ref_s), 1.0)
        finally:
            torch.randn_like = orig
        ref = ref.numpy()
        for cpu in (False, True):
            audio, L0, L = synth(A, ids_list, ref_s, cpu=cpu)
            print(f"{'dec_cpu' if cpu else 'stages '} tokens={len(ids_list)} L0={L0} L={L} "
                  f"SNR={snr_db(ref, audio):.2f} dB logspec_corr={spec_corr(ref, audio):.5f}")
        if args.wav_dir:
            import soundfile as sf
            os.makedirs(args.wav_dir, exist_ok=True)
            sf.write(os.path.join(args.wav_dir, f"ref{i}.wav"), ref, 24000)
            sf.write(os.path.join(args.wav_dir, f"tflite{i}.wav"), audio, 24000)


if __name__ == "__main__":
    main()
