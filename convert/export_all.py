"""Export everything the Android app needs into android/app/src/main/assets/kokoro/.

  python export_all.py [--only text f0n dec_front ...]

Graphs are multi-signature .tflite files (one signature per length bucket, weights shared):
  text.tflite        signatures T<n>  (token buckets)
  f0n.tflite         signatures L<n>  (frame buckets)
  dec_cpu.tflite     signatures L<n>  (whole 1-D decoder in one graph; CPU)
EdgeTPU graphs (dec_<stage>_L<n> with the one signature L<n>, blk_<kind> with signature "B") go
to work/npu_src/; convert/aot_compile_g3.py compiles them into assets/kokoro/npu/ for the Tensor G3.
plus voices/<name>.bin (float32 [510, 256]) and kokoro.json (vocab, buckets, DSP constants).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from export import HF, ROOT, sample_inputs  # noqa: E402
from kokoro_graphs import DecoderGraph, DecoderStage, F0NGraph, TextGraph, load_kmodel  # noqa: E402

OUT = os.path.join(ROOT, "android", "app", "src", "main", "assets", "kokoro")
NPU_SRC = os.path.join(ROOT, "work", "npu_src")
TOKEN_BUCKETS = [32, 64, 128]
FRAME_BUCKETS = [32, 64, 96, 128, 192, 256]  # text/f0n/CPU decoder
# EdgeTPU decoder stages. (NNAPI's on-device compiler failed at 192+ frames; the AOT-compiled
# stages are accurate up to 256 frames = 6.4 s.) Each bucket carries its own compiled weights
# (~108 MB for all stages), so there is no 192 bucket: chunks of 129-256 frames pad to 256.
# One file per stage and bucket: the Google Tensor dispatch library loads a compiled file's whole
# bytecode (every signature's program) into DMA-BUF once per signature, so a stage with all five
# buckets in one file took 5x its size (2.6 GB for the decoder instead of ~0.53 GB).
NPU_FRAME_BUCKETS = [32, 64, 96, 128, 256]
# EdgeTPU decoder graphs as shipped (bucketed by frames); the last (120x) rate runs as fixed-size
# row blocks (folded.FinalBlock). The front is split per residual block: as one program the TPU
# compiler loses precision (7% relative error vs ~0.1% per block). Generator stage 0 is split in
# two (source path / upsampling): as one AOT-compiled program it times out at 96+ frames.
DEC_STAGES = ["fr0", "fr1", "fr2", "fr3", "fr4", "g0s", "g0u", "rb00", "rb01", "rb02"]
ENGLISH_VOICES = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova",
    "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
    "am_michael", "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]


def export_graph(name, module, graph_key, buckets, prefix, out_dir=OUT):
    import litert_torch

    t0 = time.time()
    conv = None
    for n in buckets:
        args = sample_inputs(graph_key, n)
        conv = (litert_torch.signature if conv is None else conv.signature)(f"{prefix}{n}", module, args)
    with torch.no_grad():
        model = conv.convert(enable_x64=False)
    path = os.path.join(out_dir, f"{name}.tflite")
    model.export(path)
    print(f"{path}: {os.path.getsize(path) / 1e6:.1f} MB in {time.time() - t0:.0f}s", flush=True)


def export_final_blocks(dec, kinds, out_dir=None):
    """Last-rate row-block graphs (fixed shape, single signature "B"); see folded.FinalBlock."""
    import litert_torch
    import folded as fd

    R, W = fd.BLOCK_ROWS, fd.W120
    f = lambda *s: torch.zeros(*s)
    m = lambda r, w=W: torch.ones(1, 1, r, w)
    st = (f(1, 128, 1, 1), torch.ones(1, 128, 1, 1))
    ref = f(1, 256)
    samples = {
        "src0": (f(1, 24, R, W), m(R)),
        "g1u": (f(1, 256, R + 2, 80), f(1, 256, R + 2, 80), f(1, 256, R + 2, 80), m(R + 2, 80)),
        "add": (f(1, 128, R, W), f(1, 128, R, W), m(R)),
        "post": (f(1, 128, R + 2, W), f(1, 128, R + 2, W), f(1, 128, R + 2, W), m(R + 2)),
    }
    for kind in kinds:
        if kind in samples:
            args = samples[kind]
        elif kind.endswith("a"):
            args = (f(1, 128, R + 2, W), m(R + 2)) + st + (ref,)
        else:
            args = (f(1, 128, R + 2, W), m(R + 2)) + st + (f(1, 128, R, W), ref)
        t0 = time.time()
        conv = litert_torch.signature("B", fd.FinalBlock(dec.g, kind).eval(), args)
        with torch.no_grad():
            model = conv.convert(enable_x64=False)
        path = os.path.join(out_dir or NPU_SRC, f"blk_{kind}.tflite")
        model.export(path)
        print(f"{path}: {os.path.getsize(path) / 1e6:.1f} MB in {time.time() - t0:.0f}s", flush=True)


def export_assets(km):
    os.makedirs(os.path.join(OUT, "voices"), exist_ok=True)
    for v in ENGLISH_VOICES:
        pack = torch.load(os.path.join(HF, "voices", f"{v}.pt"), weights_only=True)
        arr = pack.numpy().astype("<f4").reshape(-1, 256)
        assert arr.shape == (510, 256), arr.shape
        arr.tofile(os.path.join(OUT, "voices", f"{v}.bin"))
    dec = km.decoder
    lin = dec.generator.m_source.l_linear
    meta = {
        "vocab": km.vocab,
        "token_buckets": TOKEN_BUCKETS,
        "frame_buckets": FRAME_BUCKETS,
        "npu_frame_buckets": NPU_FRAME_BUCKETS,
        "decoder_stages": DEC_STAGES,
        "final_blocks": __import__("folded").FINAL_BLOCK_KINDS,
        "block_rows": __import__("folded").BLOCK_ROWS,
        "voices": ENGLISH_VOICES,
        "l_linear_w": lin.weight.detach().numpy().reshape(-1).tolist(),
        "l_linear_b": float(lin.bias.detach()[0]),
        "f0_conv_w": dec.F0_conv.weight.detach().numpy().reshape(-1).tolist(),
        "f0_conv_b": float(dec.F0_conv.bias.detach()[0]),
        "n_conv_w": dec.N_conv.weight.detach().numpy().reshape(-1).tolist(),
        "n_conv_b": float(dec.N_conv.bias.detach()[0]),
    }
    with open(os.path.join(OUT, "kokoro.json"), "w") as fh:
        json.dump(meta, fh)
    print("wrote voices + kokoro.json", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(NPU_SRC, exist_ok=True)
    km = load_kmodel(os.path.join(HF, "config.json"), os.path.join(HF, "kokoro-v1_0.pth"))
    want = lambda n: not args.only or n in args.only
    if want("assets"):
        export_assets(km)
    if want("text"):
        export_graph("text", TextGraph(km).eval(), "text", TOKEN_BUCKETS, "T")
    if want("f0n"):
        export_graph("f0n", F0NGraph(km).eval(), "f0n", FRAME_BUCKETS, "L")
    import folded as fd
    blk = [k for k in fd.FINAL_BLOCK_KINDS if want(f"blk_{k}") or want("blocks")]
    if blk:
        export_final_blocks(DecoderGraph(km).eval(), blk)
    if want("dec_cpu"):
        export_graph("dec_cpu", DecoderGraph(km, fold_snake=False).eval(), "dec_cpu", FRAME_BUCKETS, "L")
    dec = DecoderGraph(km).eval()
    for st in DEC_STAGES:
        if want(f"dec_{st}"):
            for n in NPU_FRAME_BUCKETS:
                export_graph(f"dec_{st}_L{n}", DecoderStage(dec, st).eval(), f"dec_{st}", [n], "L", NPU_SRC)


if __name__ == "__main__":
    main()
