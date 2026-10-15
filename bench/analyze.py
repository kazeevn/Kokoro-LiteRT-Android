"""Integrate Pixel ODPM power-rail energy over BenchService phases.

  .venv-bench/bin/python bench/analyze.py bench/results/idle1 bench/results/cpu1 bench/results/npu1 ...

Each argument is a results prefix (<prefix>.json from BenchService, <prefix>.pftrace from
Perfetto). Rails are cumulative energy counters (uWs); energy in [t0, t1] is the difference of
the linearly interpolated counters. Reports SoC energy (all rails except display and radios),
the CPU / TPU / memory split, and energy per second of synthesized audio, both gross and net of
the idle baseline (taken from IDLE runs, else the quiet settle gaps of the same trace).
"""
import json
import sys
from collections import defaultdict

import numpy as np
from perfetto.trace_processor import TraceProcessor

EXCLUDE = ("display", "modem", "radio", "wifi")
GROUPS = {
    "cpu": ("cpu.big", "cpu.mid", "cpu.little"),
    "tpu": ("tpu",),
    "mem": ("ddr", "memory.interface"),
}


def load_rails(path):
    tp = TraceProcessor(trace=path)
    q = tp.query(
        "select t.name as name, c.ts as ts, c.value as v from counter c join counter_track t on c.track_id = t.id "
        "where t.name like 'power.%' order by t.name, c.ts")
    rails = defaultdict(lambda: ([], []))
    for r in q:
        name = r.name.removeprefix("power.rails.").removeprefix("power.")
        rails[name][0].append(r.ts)
        rails[name][1].append(r.v)
    tp.close()
    return {k: (np.array(ts, np.float64), np.array(v, np.float64)) for k, (ts, v) in rails.items()}


def energy_j(rail, t0, t1):
    ts, v = rail
    if t0 < ts[0] or t1 > ts[-1]:
        raise ValueError("phase outside trace")
    return (np.interp(t1, ts, v) - np.interp(t0, ts, v)) / 1e6  # uWs -> J


def soc_rails(rails):
    return [k for k in rails if not any(x in k for x in EXCLUDE)]


def clip(rails, t0, t1, min_cover=0.6):
    """Clip a phase to the span covered by rail samples (ODPM data starts ~20 s into a trace)."""
    first = max(v[0][0] for k, v in rails.items() if k in soc_rails(rails))
    last = min(v[0][-1] for k, v in rails.items() if k in soc_rails(rails))
    c0, c1 = max(t0, first), min(t1, last)
    if (c1 - c0) < min_cover * (t1 - t0):
        raise ValueError("phase mostly outside trace")
    return c0, c1


def phase_energy(rails, t0, t1):
    e = {k: energy_j(rails[k], t0, t1) for k in soc_rails(rails)}
    out = {"soc": sum(e.values())}
    for g, keys in GROUPS.items():
        out[g] = sum(v for k, v in e.items() if any(k.startswith(x) for x in keys))
    out["other"] = out["soc"] - out["cpu"] - out["tpu"] - out["mem"]
    return out


def main(prefixes):
    runs = []
    for p in prefixes:
        meta = json.load(open(p + ".json"))
        rails = load_rails(p + ".pftrace")
        runs.append((p, meta, rails))

    # idle baseline power per group (W)
    idle = defaultdict(list)
    for p, meta, rails in runs:
        for ph in meta["phases"]:
            if ph["name"] == "idle":
                t0, t1 = clip(rails, ph["start_ns"] + 5e9, ph["end_ns"])  # skip screen-off transient
                dt = (t1 - t0) / 1e9
                for k, v in phase_energy(rails, t0, t1).items():
                    idle[k].append(v / dt)
    base = {k: float(np.median(v)) for k, v in idle.items()}
    if base:
        print("idle baseline (W): " + "  ".join(f"{k}={v:.3f}" for k, v in base.items()))

    rows = []
    for p, meta, rails in runs:
        if meta["mode"] == "idle":
            continue
        for ph in meta["phases"]:
            if ph["name"] == "prepare":
                dt = (ph["end_ns"] - ph["start_ns"]) / 1e9
                e = phase_energy(rails, ph["start_ns"], ph["end_ns"])
                print(f"{p.split('/')[-1]}: prepare (load/compile + warm-up pass) {dt:.1f}s, SoC {e['soc']:.1f} J "
                      f"(cpu {e['cpu']:.1f}, tpu {e['tpu']:.1f}, mem {e['mem']:.1f})")
        for ph in meta["phases"]:
            if not ph["name"].startswith("synth"):
                continue
            dt = (ph["end_ns"] - ph["start_ns"]) / 1e9
            e = phase_energy(rails, ph["start_ns"], ph["end_ns"])
            audio = ph["audio_s"]
            net = {k: v - base.get(k, 0.0) * dt for k, v in e.items()}
            rows.append(dict(run=p.split("/")[-1], backend=meta.get("active_backend", meta["backend"]),
                             wall_s=dt, audio_s=audio, rtf=dt / audio,
                             **{f"J_{k}": v for k, v in e.items()},
                             **{f"mJ_per_audio_s_{k}": 1000 * v / audio for k, v in e.items()},
                             **{f"net_mJ_per_audio_s_{k}": 1000 * v / audio for k, v in net.items()},
                             avg_W=e["soc"] / dt, stage_ms=ph.get("stage_ms"),
                             ttfa_median_ms=float(np.median(ph["ttfa_ms"])) if ph.get("ttfa_ms") else None,
                             ttfa_p90_ms=float(np.percentile(ph["ttfa_ms"], 90)) if ph.get("ttfa_ms") else None))
    print(f"{'run':10s} {'backend':7s} {'wall s':>7s} {'audio s':>7s} {'RTF':>6s} {'avg W':>6s} "
          f"{'mJ/aud-s SoC':>12s} {'cpu':>7s} {'tpu':>7s} {'mem':>7s} {'net SoC':>8s} {'TTFA med/p90 ms':>16s}")
    for r in rows:
        print(f"{r['run']:10s} {r['backend']:7s} {r['wall_s']:7.2f} {r['audio_s']:7.2f} {r['rtf']:6.3f} "
              f"{r['avg_W']:6.2f} {r['mJ_per_audio_s_soc']:12.1f} {r['mJ_per_audio_s_cpu']:7.1f} "
              f"{r['mJ_per_audio_s_tpu']:7.1f} {r['mJ_per_audio_s_mem']:7.1f} {r['net_mJ_per_audio_s_soc']:8.1f} "
              f"{(r['ttfa_median_ms'] or 0):7.0f}/{(r['ttfa_p90_ms'] or 0):<7.0f}")
    json.dump(dict(idle_baseline_W=base, runs=rows), open("bench/results/summary.json", "w"), indent=2)


if __name__ == "__main__":
    main(sys.argv[1:])
