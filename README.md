# Kokoro-82M on LiteRT for the Pixel 8a

An Android system text-to-speech engine running [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
on-device with [LiteRT](https://ai.google.dev/edge/litert), with the vocoder on the Tensor G3
EdgeTPU (NPU) or on the CPU. English voices only (28 American/British voices). Built for and
tested on a Pixel 8a (Android 17); it is not meant to run anywhere else.

Use it from any app that speaks through Android's TTS API (e.g. [Readest](https://github.com/readest/readest)):
install the APK, then *Settings → System → Languages → Text-to-speech output → Preferred engine →
Kokoro TTS*. The app's own screen picks the default voice, the decoder backend and a speed factor.

## How it works

Kokoro's output length depends on its predicted phoneme durations, while LiteRT graphs — and
especially the EdgeTPU — need static shapes. The model is therefore split into fixed-size graphs
(one *signature* per length bucket, weights shared) with the data-dependent glue on the host:

```
text ─ G2P (misaki port + espeak-ng) ─ phoneme ids
  text graph   (token buckets 32/64/128): ALBERT, duration predictor, text encoder
  host         durations → alignment (repeat token features per frame)
  f0n graph    (frame buckets 32…256): pitch (F0) and energy curves
  host         hn-NSF harmonic source + STFT (n_fft 20, hop 5)
  decoder      CPU: one graph · NPU: 16 TPU-sized stage graphs (iSTFTNet generator)
  host         iSTFT → 24 kHz PCM
```

Exactness tricks (padding never changes the result):
* Token graphs are right-padded; masks reproduce `pack_padded_sequence`. The backward LSTM
  direction runs over a host-provided per-sequence reversal index.
* Frame graphs are padded to the bucket; instance-norm statistics are masked and every
  frame-mixing op sees zero padding, so bucketing is bit-for-bit equivalent up to float rounding.

EdgeTPU-specific rewrites of the vocoder (found by bisecting compiler failures on the device):
* rows longer than ~2k samples fail to compile, so the generator works on a row-folded layout
  `[1, C, H, W]` (W = 8/80/480 at the 2×/20×/120× rates) with halo columns from neighbouring rows;
* graphs with more than ~1M activation elements fail to compile or hit the TPU firmware's
  inference timeout. The decoder therefore runs as 9 bucketed graphs (front split per residual
  block, stage 0 at the 20× rate) plus 28 fixed-shape *row-block* graphs for the last (120×)
  rate: blocks of 7 rows (+1 halo row each side); each block graph also returns per-block
  instance-norm statistics that the host combines (parallel variance formula);
* reductions over time are stacked ≤64-wide average pools; SIN on long rows is computed in a
  folded layout; channel counts padded to multiples of 4; 1-channel convs moved to the host;
* the decoder front compiled as *one* TPU program loses precision (7 % relative error, the
  compiler presumably truncates intermediates); split per block it is accurate (0.06 %).

NPU chunks are at most 128 frames (3.2 s of audio); the CPU decoder (one graph) takes up to
256. Validation: convert/verify_assets.py (host, vs PyTorch: log-spectrogram correlation
0.999 — the residual is float32 phase drift of the original's sine source) and the app's
`compare` benchmark mode (NPU vs CPU decoder on identical inputs, on the phone: median
log-spectrogram correlation 0.99988, median SNR 31 dB).

CPU: XNNPACK through our own delegate with a weight cache (cpp/xnnpack_jni.cpp): LiteRT's
default delegate packs shared weights once per op *and* per signature subgraph (unrolled LSTMs,
bucketed models), which took > 3.5 GB. 6 threads (RTF 1.32) beat 4 (RTF 1.57); XNNPACK's
FORCE_FP16 mode aborts inside LiteRT 1.4.2, so the CPU path is fp32.

### Why NNAPI and not LiteRT's NPU accelerator
LiteRT 2.x's `CompiledModel` + Google Tensor plugin does compile the vocoder on-device for the
Tensor G3, but the vendor TPU service only runs models for apps on Google's EdgeTPU allowlist
(`edgetpu_native/allowlist_app*` device flags, not writable without root). The AOT compiler of
the Tensor SDK is gated behind a beta sign-up. The public path to the TPU is NNAPI (deprecated,
still shipped in Android 17): LiteRT's Interpreter API with the NNAPI delegate targeting the
`google-edgetpu` device. The driver does not cache compilations, so the TPU graphs are compiled
in the background each time the engine process starts (a few minutes); the CPU serves meanwhile.

## CPU vs NPU power (bench/RESULTS.md)
Per second of synthesized speech, SoC energy from the on-device power monitor (screen off):
**CPU 2.52 J** (1.37× real time) vs **NPU 1.67 J** (0.77× real time) — the NPU path uses ~34 %
less energy and is the only one faster than real time (median first-audio latency per sentence
1.65 s vs 2.9 s). It costs ~6 min / ~400 J of on-device TPU compilation per engine start, recovered
after ~3.5 min of listening. Default backend: NPU (CPU serves until the TPU graphs are compiled).

## Layout
* `android/` — the app (`:app`); `:nnprobe` is a throwaway NNAPI probe used to map TPU support.
* `convert/` — export pipeline: `kokoro_graphs.py` (graph rewrites), `folded.py` (row-folded
  generator), `host_dsp.py` (host DSP spec), `export_all.py`, verification scripts.
* `bench/` — power benchmark (Perfetto ODPM power rails, `run_all.sh`, `analyze.py`).
* `tools/` — adb helpers for a flaky USB link (`adbr`), cool-down, memory watchdog, probes.
* `third_party/espeak-ng` — submodule (G2P fallback for out-of-dictionary words).

## Building
Requirements: Python 3.12 via `uv`, Android SDK 37 + NDK 29 + CMake 3.31, JDK 21+.
```
git submodule update --init
uv venv -p 3.12 .venv-convert
VIRTUAL_ENV=.venv-convert uv pip install --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu -r convert/requirements.txt
# Kokoro weights + English voices into work/hf (config.json, kokoro-v1_0.pth, voices/*.pt)
.venv-convert/bin/python convert/export_all.py           # ~45 min, writes android/app/src/main/assets/kokoro
.venv-convert/bin/python convert/build_lexicon.py        # G2P dictionaries
cd android && ./gradlew :app:assembleRelease
```
espeak-ng's compiled English data (`android/app/src/main/assets/espeak-ng-data`) comes from a
host build of the submodule (`cmake -S third_party/espeak-ng -B work/espeak-host && ninja -C work/espeak-host`).

## License
Kokoro-82M weights: Apache-2.0. espeak-ng is GPL-3.0, so the APK as a whole is GPL-3.0.
