"""Checks the app's EdgeTPU outputs against the host CPU, graph by graph, on real activations.

  1. on the phone: BenchService compare mode with --ez check true (Graph.checkOutputs) dumps the
     first call of every graph/signature to files/dump/<graph>.<sig>.in<k>.bin / .out<i>.bin
  2. adb pull /sdcard/Android/data/io.github.kazeevn.kokorotts/files/dump/. work/dump/
  3. .venv-aot/bin/python convert/aot_verify_dump.py [work/dump]

Each dumped NPU call is replayed on the source graph (work/npu_src) on the host. Prints the
relative L2 error per output (over finite reference values; mean/m2 statistics of fully padded
row blocks are 0/0 on both sides).
"""
import glob
import os
import re
import sys

import numpy as np
from ai_edge_litert.interpreter import Interpreter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = os.path.join(ROOT, "work", "npu_src")


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "work", "dump")
    calls = sorted({re.sub(r"\.out\d+\.bin$", "", os.path.basename(f)) for f in glob.glob(os.path.join(d, "*.out0.bin"))})
    worst = 0.0
    for call in calls:
        name, sig = call.rsplit(".", 1)
        path = os.path.join(SRC, f"{name}.tflite")
        if not os.path.exists(path):
            continue  # a CPU graph
        runner = Interpreter(path).get_signature_runner(sig)
        details = runner.get_input_details()
        args = {}
        for k in range(len(details)):
            det = details[f"args_{k}"]
            args[f"args_{k}"] = np.fromfile(os.path.join(d, f"{call}.in{k}.bin"), det["dtype"]).reshape(det["shape"])
        ref = runner(**args)
        errs = []
        for i in range(len(ref)):
            r = ref[f"output_{i}"].ravel()
            f = os.path.join(d, f"{call}.out{i}.bin")
            if not os.path.exists(f):
                continue
            npu = np.fromfile(f, np.float32)
            ok = np.isfinite(r)
            bad = int((~np.isfinite(npu[ok])).sum())
            e = float(np.linalg.norm((npu - r)[ok]) / max(np.linalg.norm(r[ok]), 1e-30)) if not bad else float("inf")
            errs.append(e)
            worst = max(worst, e)
        flag = "  <-- BAD" if max(errs) > 0.02 else ""
        print(f"{call:24s} rel L2 " + " ".join(f"{e:.5f}" for e in errs) + flag, flush=True)
    print(f"worst {worst:.5f}")


if __name__ == "__main__":
    main()
