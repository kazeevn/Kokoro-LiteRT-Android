"""Numerical check of one AOT-compiled (Tensor SDK) graph signature on the Pixel EdgeTPU.

  .venv-aot/bin/python convert/aot_numcheck.py <compiled.tflite> <graph> <sig> [dump_dir [src_dir]]

Runs <compiled.tflite> in the :aotprobe app (LiteRT CompiledModel + Google Tensor dispatch) on
real inputs dumped by the app (see aot_verify_dump.py: dump_dir/<graph>.<sig>.in<k>.bin) and
compares every output with the source graph <src_dir>/<graph>.tflite (default work/npu_src) on the host CPU.
Use it to try compiler flags: aot_compile_g3.py <graph> --out DIR --flag google_tensor_...=...
"""
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
from ai_edge_litert.interpreter import Interpreter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PKG = "io.github.kazeevn.aotprobe"
RES = f"/sdcard/Android/data/{PKG}/files/results"
STAGE = "/data/local/tmp/aotstage"


def adb(*a):
    return subprocess.run([os.path.join(ROOT, "tools", "adbr"), *a], check=True, capture_output=True, text=True).stdout


def main():
    model, graph, sig = sys.argv[1:4]
    dump = sys.argv[4] if len(sys.argv) > 4 else os.path.join(ROOT, "work", "dump")
    src = sys.argv[5] if len(sys.argv) > 5 else os.path.join(ROOT, "work", "npu_src")
    runner = Interpreter(os.path.join(src, f"{graph}.tflite")).get_signature_runner(sig)
    det = runner.get_input_details()
    files = [os.path.join(dump, f"{graph}.{sig}.in{k}.bin") for k in range(len(det))]
    args = {f"args_{k}": np.fromfile(f, det[f"args_{k}"]["dtype"]).reshape(det[f"args_{k}"]["shape"])
            for k, f in enumerate(files)}
    ref = runner(**args)

    m = "probe.tflite"
    adb("shell", f"rm -rf {STAGE} {RES}/{m}.* && mkdir -p {STAGE}")
    adb("push", model, f"{STAGE}/{m}")
    for k, f in enumerate(files):
        adb("push", f, f"{STAGE}/{m}.in{k}.bin")
    adb("shell", f"run-as {PKG} sh -c 'rm -rf files/probe; mkdir -p files/probe && cp {STAGE}/* files/probe/'")
    adb("shell", "rm", "-rf", STAGE)
    adb("shell", "am", "start", "-W", "-n", f"{PKG}/.AotProbeActivity", "--es", "model", m, "--es", "sig", sig,
        "--ei", "iters", "10", "--ez", "dumpout", "true")
    rep = ""
    for _ in range(120):
        rep = adb("shell", f"cat {RES}/{m}.npu.txt 2>/dev/null || true")
        if rep.strip():
            break
        time.sleep(1)
    timing = next((l for l in rep.splitlines() if l.startswith("run_us") or l.startswith("ERROR")), rep.strip())
    errs = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(len(ref)):
            if "dumped" not in rep:
                break
            out = os.path.join(tmp, f"out{i}.bin")
            adb("pull", f"{RES}/{m}.out{i}.npu.bin", out)
            r = ref[f"output_{i}"].ravel()
            npu = np.fromfile(out, np.float32)
            ok = np.isfinite(r)
            errs.append(np.linalg.norm((npu - r)[ok]) / max(np.linalg.norm(r[ok]), 1e-30))
    print(f"{graph}/{sig} {os.path.basename(os.path.dirname(os.path.abspath(model)))}: rel L2 "
          + " ".join(f"{e:.5f}" for e in errs) + f"  {timing}", flush=True)


if __name__ == "__main__":
    main()
