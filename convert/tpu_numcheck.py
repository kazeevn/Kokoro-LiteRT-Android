"""Numerical check of single graphs on the Pixel EdgeTPU (via the nnprobe app) vs host CPU.

  python tpu_numcheck.py fr0 fr1 fr2 fr3 fr4      (decoder front sub-blocks, L=32)

Inputs are real activations from one sentence (so value ranges are realistic). For each graph:
export (L=32), push model + inputs, run on the TPU with output dump, compare to the host.
"""
import os
import subprocess
import sys

import numpy as np
import torch
from ai_edge_litert.interpreter import Interpreter, OpResolverType

sys.path.insert(0, os.path.dirname(__file__))
import host_dsp  # noqa: E402
from kokoro_graphs import DecoderGraph, DecoderStage, F0NGraph, Rate, TextGraph, load_kmodel  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "work", "numcheck")
PKG = "io.github.kazeevn.nnprobe"


def adb(*a, check=True):
    return subprocess.run([os.path.join(ROOT, "tools", "adbr"), *a], check=check, capture_output=True, text=True).stdout


def real_front_inputs(km, L=32):
    from kokoro import KPipeline

    pipe = KPipeline(lang_code="a", model=False)
    pack = torch.load(os.path.join(ROOT, "work/hf/voices/af_heart.pt"), weights_only=True)
    ps = "".join(r.phonemes for r in pipe("Mr. Bennet made no answer."))
    ids = [0] + [km.vocab[p] for p in ps if p in km.vocab] + [0]
    ref_s = pack[len(ps) - 1].float()
    n, T = len(ids), 64
    I = torch.zeros(1, T, dtype=torch.int32); I[0, :n] = torch.tensor(ids)
    M = torch.zeros(1, T); M[0, :n] = 1
    R = torch.arange(T, dtype=torch.int32); R[:n] = torch.arange(n - 1, -1, -1, dtype=torch.int32)
    with torch.no_grad():
        dur, d, ten = TextGraph(km).eval()(I, M, R, ref_s)
    pd = np.maximum(np.round(dur[0, :n].numpy()), 1).astype(int)
    while pd.sum() > L:  # shorten for the L=32 bucket
        pd[np.argmax(pd)] -= 1
    L0 = int(pd.sum())
    fi = host_dsp.frame_inputs(pd, d[0].numpy(), ten[0].numpy(), L)
    with torch.no_grad():
        f0, nn_ = F0NGraph(km).eval()(*[torch.from_numpy(fi[k]) for k in ("en", "rev", "m1", "m2")], ref_s)
    dec = km.decoder
    fn = host_dsp.f0n_features(f0[0].numpy() * fi["m2"][0, 0], nn_[0].numpy() * fi["m2"][0, 0],
                               dec.F0_conv.weight.detach().numpy()[0, 0], float(dec.F0_conv.bias[0]),
                               dec.N_conv.weight.detach().numpy()[0, 0], float(dec.N_conv.bias[0]))
    return dict(asr=torch.from_numpy(fi["asr"]), fn=torch.from_numpy(fn), m1=torch.from_numpy(fi["m1"]),
                m2=torch.from_numpy(fi["m2"]), ref_s=ref_s)


def main(names):
    import litert_torch

    os.makedirs(OUT, exist_ok=True)
    km = load_kmodel(os.path.join(ROOT, "work/hf/config.json"), os.path.join(ROOT, "work/hf/kokoro-v1_0.pth"))
    dg = DecoderGraph(km).eval()
    x = real_front_inputs(km)
    # chain of front blocks on torch to get realistic inputs for each
    acts = {}
    with torch.no_grad():
        h = dg.front_block(0, None, x["asr"], x["fn"], Rate(x["m1"]), Rate(x["m2"]), x["ref_s"])
        acts["fr0"] = (x["asr"], x["fn"], x["m1"], x["m2"], x["ref_s"])
        for k in range(1, 5):
            acts[f"fr{k}"] = (h, x["asr"], x["fn"], x["m1"], x["m2"], x["ref_s"])
            h = dg.front_block(k, h, x["asr"], x["fn"], Rate(x["m1"]), Rate(x["m2"]), x["ref_s"])
    for name in names:
        args = acts[name]
        mod = DecoderStage(dg, name).eval()
        path = os.path.join(OUT, f"{name}.tflite")
        litert_torch.convert(mod, args, enable_x64=False).export(path)
        it = Interpreter(model_path=path, experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        it.allocate_tensors()
        ins = sorted(it.get_input_details(), key=lambda d: int(d["name"].rsplit("_", 1)[-1]))
        for i, (d, a) in enumerate(zip(ins, args)):
            arr = np.ascontiguousarray(a.numpy().astype(d["dtype"]))
            it.set_tensor(d["index"], arr)
            arr.tofile(f"{path}.in{i}.bin")
        it.invoke()
        ref = it.get_tensor(it.get_output_details()[0]["index"])
        b = os.path.basename(path)
        for f in [path] + [f"{path}.in{i}.bin" for i in range(len(args))]:
            adb("push", f, "/data/local/tmp/")
            adb("shell", "run-as", PKG, "sh", "-c", f"'mkdir -p files/probe && cp /data/local/tmp/{os.path.basename(f)} files/probe/'")
        res = f"/sdcard/Android/data/{PKG}/files/results/{b}.out0.google-edgetpu.bin"
        adb("shell", "rm", "-f", res, check=False)
        adb("shell", "am", "start", "-n", f"{PKG}/.NnProbeActivity", "--es", "model", b, "--es", "accel",
            "google-edgetpu", "--ei", "iters", "2", "--ez", "dumpout", "true")
        for _ in range(300):
            if "yes" in adb("shell", f"test -f {res} && echo yes", check=False):
                break
            subprocess.run(["sleep", "1"])
        local = os.path.join(OUT, f"{b}.tpu.bin")
        adb("pull", res, local)
        got = np.fromfile(local, dtype=np.float32).reshape(ref.shape)
        err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        print(f"{name}: rel L2 err TPU vs CPU = {err:.4f}  (max |ref| {np.abs(ref).max():.2f})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
