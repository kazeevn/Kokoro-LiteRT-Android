"""Convert the split Kokoro graphs to LiteRT (.tflite), one file per graph/bucket size.

  python export.py --graph text --sizes 32 64 128 256
  python export.py --graph f0n --sizes 64 128 256 512
  python export.py --graph decoder --sizes 64 128 256 512

Each <graph>_<size>.tflite gets a <graph>_<size>.json sidecar describing inputs (used by the
on-device probe harness and as documentation of the I/O contract).
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kokoro_graphs import DecoderGraph, DecoderStage, F0NGraph, TextGraph, load_kmodel, smooth_length  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
HF = os.path.join(ROOT, "work", "hf")


def sample_inputs(graph, n):
    f = lambda *s: torch.zeros(*s, dtype=torch.float32)
    i = lambda *s: torch.zeros(*s, dtype=torch.int32)
    ref_s = f(1, 256)
    if graph == "text":
        return (i(1, n), torch.ones(1, n), torch.arange(n, dtype=torch.int32), ref_s)
    if graph == "f0n":
        return (f(1, 640, n), torch.arange(n, dtype=torch.int32), torch.ones(1, 1, n), torch.ones(1, 1, 2 * n), ref_s)
    if graph == "dec_cpu":  # 1-D decoder for the CPU backend; P matches the folded stages
        P = (n // 4 + 1) * 480
        return (f(1, 512, n), f(1, 4, n), f(1, 24, P), torch.ones(1, 1, n),
                torch.ones(1, 1, 2 * n), torch.ones(1, 1, 20 * n), torch.ones(1, 1, P), ref_s)
    if graph == "decoder":
        P = smooth_length(120 * n + 1)
        return (f(1, 512, n), f(1, 4, n), f(1, 24, P), torch.ones(1, 1, n),
                torch.ones(1, 1, 2 * n), torch.ones(1, 1, 20 * n), torch.ones(1, 1, P), ref_s)
    H = n // 4 + 1  # folded rows (folded.rows_for)
    m = lambda *k: torch.ones(1, 1, *k)
    st = graph[4:] if graph.startswith("dec_") else None
    if st == "front" or st == "fr0":
        return (f(1, 512, n), f(1, 4, n), m(n), m(2 * n), ref_s)
    if st and st.startswith("fr"):
        return (f(1, 1024, n), f(1, 512, n), f(1, 4, n), m(n), m(2 * n), ref_s)
    if st == "g0":
        return (f(1, 512, 2 * n), f(1, 24, H, 480), m(H, 8), m(H, 80), ref_s)
    if st == "g0s":
        return (f(1, 24, H, 480), m(H, 80), ref_s)
    if st == "g0u":
        return (f(1, 512, 2 * n), f(1, 256, H, 80), m(H, 80))
    if st and st.startswith("rb0"):
        return (f(1, 256, H, 80), m(H, 80), ref_s)
    # distinct tensors per input: torch.export merges aliased sample inputs into one placeholder
    if st == "g1":
        return (f(1, 256, H, 80), f(1, 256, H, 80), f(1, 256, H, 80), f(1, 24, H, 480), m(H, 80), m(H, 480), ref_s)
    if st and st.startswith("rb1"):
        return (f(1, 128, H, 480), m(H, 480), ref_s)
    if st == "post":
        return (f(1, 128, H, 480), f(1, 128, H, 480), f(1, 128, H, 480), m(H, 480))
    raise ValueError(graph)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, choices=["text", "f0n", "decoder"] + ["dec_" + st for st in DecoderStage.STAGES]
                    + [f"dec_rb{i}{j}s{k}" for i in range(2) for j in range(3) for k in range(3)])
    ap.add_argument("--sizes", type=int, nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "work", "tflite"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import litert_torch

    km = load_kmodel(os.path.join(HF, "config.json"), os.path.join(HF, "kokoro-v1_0.pth"))
    if args.graph.startswith("dec_"):
        module = DecoderStage(DecoderGraph(km), args.graph[4:]).eval()
    else:
        module = {"text": TextGraph, "f0n": F0NGraph, "decoder": DecoderGraph}[args.graph](km).eval()
    for n in args.sizes:
        x = sample_inputs(args.graph, n)
        t0 = time.time()
        with torch.no_grad():
            edge = litert_torch.convert(module, x, enable_x64=False)
        path = os.path.join(args.out, f"{args.graph}_{n}.tflite")
        edge.export(path)
        spec = {"inputs": [{"dtype": "i32" if t.dtype == torch.int32 else "f32", "shape": list(t.shape)} for t in x]}
        with open(path[:-7] + ".json", "w") as fh:
            json.dump(spec, fh)
        print(f"{path}: {os.path.getsize(path) / 1e6:.1f} MB in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
