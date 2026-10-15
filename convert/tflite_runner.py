"""Runs converted .tflite graphs on the host (ai_edge_litert), choosing the bucket per call."""
import glob
import os
import re

import numpy as np
from ai_edge_litert.interpreter import Interpreter


class TfliteRunner:
    def __init__(self, directory, threads=8):
        self.models = {}
        for p in glob.glob(os.path.join(directory, "*.tflite")):
            m = re.match(r"(text|f0n|decoder)_(\d+)\.tflite$", os.path.basename(p))
            if m:
                self.models[(m.group(1), int(m.group(2)))] = p
        self.cache = {}
        self.threads = threads

    def _run(self, graph, size, *args):
        key = (graph, size)
        if key not in self.cache:
            it = Interpreter(model_path=self.models[key], num_threads=self.threads)
            it.allocate_tensors()
            self.cache[key] = it
        it = self.cache[key]
        ins = sorted(it.get_input_details(), key=lambda d: int(d["name"].rsplit("_", 1)[-1]))
        for d, a in zip(ins, args):
            it.set_tensor(d["index"], np.ascontiguousarray(a, dtype=d["dtype"]))
        it.invoke()
        outs = sorted(it.get_output_details(), key=lambda d: d["name"])
        return [it.get_tensor(d["index"]) for d in outs]

    def run_text(self, ids, mask, rev, ref_s):
        return self._run("text", ids.shape[1], ids, mask, rev, ref_s)

    def run_f0n(self, en, rev, m1, m2, ref_s):
        return self._run("f0n", en.shape[2], en, rev, m1, m2, ref_s)

    def run_dec(self, asr, fn, har, m1, m2, m20, mP, ref_s):
        return self._run("decoder", asr.shape[2], asr, fn, har, m1, m2, m20, mP, ref_s)[0]
