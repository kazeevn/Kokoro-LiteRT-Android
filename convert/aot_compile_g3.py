"""AOT-compiles the EdgeTPU decoder graphs for the Tensor G3 with the Google Tensor SDK.

Runs in .venv-aot (ai-edge-litert 2.3.0 + ai-edge-litert-sdk-google-tensor, see README).
Usage: .venv-aot/bin/python convert/aot_compile_g3.py [graph ...] [--out DIR] [--flag k=v ...]
Compiles work/npu_src/<graph>.tflite (from export_all.py) into assets/kokoro/npu/<graph>.tflite;
without graph names, every NPU graph listed in assets/kokoro/kokoro.json.
"""

import argparse
import json
import pathlib
import shutil
import sys
import tempfile
import time

from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.vendors.google_tensor import target as gt_target

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSETS = ROOT / "android/app/src/main/assets/kokoro"
SRC = ROOT / "work/npu_src"


def npu_graphs() -> list[str]:
    meta = json.loads((ASSETS / "kokoro.json").read_text())
    return [f"dec_{s}" for s in meta["decoder_stages"]] + [f"blk_{b}" for b in meta["final_blocks"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graphs", nargs="*")
    ap.add_argument("--out", default=str(ASSETS / "npu"))
    ap.add_argument("--src", default=str(SRC), help="directory of the source graphs")
    ap.add_argument("--flag", action="append", default=[],
                    help="google_tensor_* compiler flag, e.g. google_tensor_truncation_type=half")
    args = ap.parse_args()
    # "auto" truncates to bfloat16 (~6x the error of NNAPI's fp16 on the decoder front);
    # "half" matches it (convert/aot_numcheck.py). "no_truncation" does not compile.
    flags = {"google_tensor_truncation_type": "half"} | dict(f.split("=", 1) for f in args.flag)
    # dec_fr0 compiled with fp16 truncation has 2-5% error on some inputs (bfloat16: <1%)
    graph_flags = {"dec_fr0": {"google_tensor_truncation_type": "bfloat16"}}
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = gt_target.Target(gt_target.SocModel.TENSOR_G3)

    tmp = pathlib.Path(tempfile.mkdtemp())
    report = {}
    for name in args.graphs or npu_graphs():
        t0 = time.time()
        try:
            res = aot_lib.aot_compile(str(pathlib.Path(args.src) / f"{name}.tflite"), output_dir=tmp,
                                      target=[target], keep_going=False,
                                      **(flags | ({} if args.flag else graph_flags.get(name, {}))))
            res.export(out, model_name=name)
            (out / f"{name}_Google_Tensor_G3.tflite").rename(out / f"{name}.tflite")
            stats = [str(m.partition_stats) for m in res.models if m.partition_stats]
            report[name] = {"ok": True, "s": round(time.time() - t0, 1), "partitions": stats}
            print(f"{name}: ok in {time.time() - t0:.0f}s {stats}", flush=True)
        except Exception as e:  # noqa: BLE001 — keep compiling the other graphs
            report[name] = {"ok": False, "s": round(time.time() - t0, 1), "error": repr(e)}
            print(f"{name}: FAILED {e!r}", flush=True)
    shutil.rmtree(tmp)
    (pathlib.Path(args.src) / "aot_report.json").write_text(json.dumps(report, indent=1))
    failed = [n for n, r in report.items() if not r["ok"]]
    print(f"{len(report) - len(failed)}/{len(report)} compiled; failed: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
