# CPU vs NPU (EdgeTPU) power benchmark — Pixel 8a, Android 17

## Method
* **Workload**: the opening of *Pride and Prejudice* (`android/app/src/main/assets/bench_corpus.txt`),
  synthesized sentence by sentence like Readest does (one TTS request per sentence), voice
  `af_heart`, speed 1.0 → 88.4 s of speech (81.5 s with 256-frame CPU chunks, fewer chunk joins).
  Audio is generated but not played. Three timed repeats after an untimed warm-up pass, 15 s
  gaps between repeats.
* **Conditions**: `BenchService` foreground service with a partial wake lock, screen off, phone on
  USB, started only when the hottest CPU cluster is below 42–44 °C.
* **Energy**: Perfetto `android.power` on-device power monitor (ODPM) rails (cumulative energy
  per rail, ~4 Hz), integrated over each repeat. "SoC" = all rails except display and radios
  (CPU clusters, TPU, GPU, DDR/memory interface, fabric, AoC, LDOs …). "net" subtracts the
  idle baseline power over the same duration (screen off, idle: 0.30 W SoC).
* **Configurations** (same text, same G2P/text/F0 graphs on CPU in all cases):
  * **CPU**: vocoder as one XNNPACK graph, 6 threads, fp32 — chunks ≤ 128 frames (3.2 s, same
    chunking as the NPU) and ≤ 256 frames (the app's CPU default).
  * **NPU**: vocoder on the Tensor G3 EdgeTPU via NNAPI (fp16), 9 bucketed graphs + 28
    row-block graphs; chunks ≤ 128 frames. Text and F0 graphs stay on the CPU. Three successive
    versions of the host-side driver were measured (rows below); the last is what ships.
* Accuracy (NPU vs CPU vocoder on identical inputs, `compare` mode): median log-spectrogram
  correlation 0.99988, median SNR 31 dB — i.e. the same audio.

## Results (per second of synthesized speech, median of 3 repeats)

| configuration | SoC energy | net of idle | CPU rails | TPU rail | memory rails | real-time factor | time to first audio (median / p90) |
|---|---|---|---|---|---|---|---|
| CPU, 128-frame chunks | 2.52 J | 2.09 J | 1.89 J | 0 | 0.28 J | 1.37 | 2.9 s / 3.8 s |
| CPU, 256-frame chunks | 2.54 J | 2.03 J | 1.78 J | 0 | 0.38 J | 1.63 (throttling) | 3.5 s / 8.4 s |
| NPU (first version) | 1.85 J | 1.58 J | 0.41 J | 0.65 J | 0.45 J | 0.89 | 1.9 s / 2.5 s |
| NPU (+ reused buffers) | 1.70 J | 1.46 J | 0.36 J | 0.61 J | 0.40 J | 0.79 | 1.7 s / 2.2 s |
| **NPU (final: + bulk block copies)** | **1.67 J** | **1.43 J** | 0.36 J | 0.60 J | 0.40 J | **0.77** | **1.65 s / 2.15 s** |

One-time cost of the NPU path: compiling the 37 TPU graphs on-device (~3.5 min; NNAPI cannot
cache them) plus the warm-up pass ≈ 6 min and ~400 J per engine process start (CPU path:
~105 s and ~200–235 J for loading + the same warm-up pass). The NPU saves ~0.85 J per second of
speech, so the extra start-up energy (~170 J) is recovered after ~3.5 minutes of listening.

## Conclusion
The NPU is more power-efficient: ~34 % less SoC energy per second of speech (~32 % net of idle),
and it is the only configuration that is faster than real time. Readest requests one sentence at a
time and asks for the next only after the previous one finished playing, so the time to first
audio is heard as a pause between sentences: ~1.65 s on the NPU vs ~2.9 s on the CPU.
The gain is not the 5–10× one might expect from the TPU's raw speed:
* the text and F0 graphs (LSTMs, ALBERT) and the DSP still run on the CPU in both configurations;
* the vocoder's last stage must run as ~140 small TPU calls per 3.2 s chunk (TPU compiler size
  limits), and the host-side block copies / statistics cost CPU time and memory bandwidth;
* memory traffic (DDR rails) is similar or higher on the NPU path.

Raw data: `bench/results/*.json` (phase timestamps, per-stage timings, per-sentence latency),
`*.pftrace` (Perfetto traces), `summary*.json` (`bench/analyze.py` output).
