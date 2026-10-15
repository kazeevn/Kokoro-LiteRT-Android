# Kokoro-82M on LiteRT for the Pixel 8a

An Android system text-to-speech engine running [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
on-device with [LiteRT](https://ai.google.dev/edge/litert) 2.3.0, with the vocoder on the Tensor
G3 EdgeTPU (NPU, ahead-of-time compiled with the Google Tensor SDK) or on the CPU. English voices only (28 American/British voices). Built for and
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
  decoder      CPU: one graph · NPU: 10 TPU-sized stage graphs + 28 row-block graphs (iSTFTNet generator)
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
  inference timeout. The decoder therefore runs as 10 bucketed graphs (front split per residual
  block, stage 0 at the 20× rate split into source path and upsampling, its three resblocks)
  plus 28 fixed-shape *row-block* graphs for the last (120×)
  rate: blocks of 7 rows (+1 halo row each side); each block graph also returns per-block
  instance-norm statistics that the host combines (parallel variance formula). Blocks holding
  only padding are skipped, so a chunk padded to a larger bucket costs little extra;
* reductions over time are stacked ≤64-wide average pools; SIN on long rows is computed in a
  folded layout; channel counts padded to multiples of 4; 1-channel convs moved to the host;
* the decoder front compiled as *one* TPU program loses precision (7 % relative error, the
  compiler presumably truncates intermediates); split per block it is accurate (0.06 %).

The graphs are compiled ahead of time with the Google Tensor SDK (`convert/aot_compile_g3.py`)
and shipped precompiled (`assets/kokoro/npu`); the app runs them through LiteRT's `CompiledModel`
API and the Google Tensor dispatch library (`litert-npu-runtime-google-tensor`), loading the whole
NPU decoder in ~1.5 s. The vendor TPU service only allows on-device *compilation* for apps on
Google's EdgeTPU allowlist; precompiled models run for any app. Workarounds for the AOT compiler,
each found by comparing every graph's TPU output with the host on real activations
(`convert/aot_verify_dump.py`):
* fp16 truncation (`google_tensor_truncation_type=half`): the default `auto` means bfloat16,
  ~6× the error. Exception: the first front block (`dec_fr0`) is 2–5 % off in fp16 on some
  inputs and uses bfloat16 (< 1 %); the cause was not found (not activation range, not eps).
* strided `TRANSPOSE_CONV` (the 10× and 6× upsampling) is miscompiled (garbage output): it is
  written as a polyphase convolution (`folded.fconv_t_poly`);
* the 11-tap convs of the third 20×-rate resblock are wrong at 33 rows (L128): they run as
  sums of ≤7-tap convs (`folded.fconv(max_taps=7)`);
* generator stage 0 as one program times out at 96+ frames: split in two;
* a `[1,1,H,8]` mask input made the dispatch library pad its buffer (uninitialized tail):
  derived in-graph from the 20×-rate mask instead.

App-side, tensors are copied with one memcpy between direct ByteBuffers and the locked LiteRT
tensor buffers (cpp/litert_buffers.cpp): the Kotlin TensorBuffer API only copies Java arrays,
which cost more than the TPU itself for the ~4 MB row-block calls.

Chunks are at most 256 frames (6.4 s of audio) on both decoders. The NPU stages have buckets
32/64/96/128/256 frames (each bucket adds ~100 MB of compiled weights, hence no 192); the CPU
graphs 32…256. Validation: convert/verify_assets.py (host, vs PyTorch: log-spectrogram correlation
0.999 — the residual is float32 phase drift of the original's sine source) and the app's
`compare` benchmark mode (NPU vs CPU decoder on identical inputs, on the phone: 19 chunks of
1.4–6 s, log-spectrogram correlation ≥ 0.9993, SNR 26–32.6 dB, median 31 dB).

CPU: XNNPACK through `CompiledModel` with a weight cache file (`CpuOptions`): without it XNNPACK
packs shared weights once per op *and* per signature subgraph (unrolled LSTMs, bucketed models),
which took > 3.5 GB. 6 threads beat 4. The CPU path is fp32. `text`, `f0n` and `dec_cpu` do not
compile for the TPU (INTERNAL compiler error), so they always run on the CPU.

## Performance and power
On the bench corpus (bench/run_bench.sh, one TTS request per sentence), LiteRT 2.3.0:
NPU decoder **0.53× real time**, first audio per sentence 1.17 s median / 3.0 s p90 (long
sentences are no longer split at 3.2 s, so their first chunk takes longer); CPU 1.05× real
time, 2.6 s. Peak memory (VmHWM) with the NPU decoder ~0.9 GB. Default backend: NPU (the CPU
serves while the NPU graphs load, ~1.5 s).

bench/RESULTS.md has SoC energy from the on-device power monitor, measured with the previous
NNAPI build (LiteRT 1.4.2, on-device TPU compilation): CPU 2.52 J vs NPU 1.67 J per second of
speech (CPU 1.37×, NPU 0.77× real time). The NNAPI build also spent ~6 min / ~400 J compiling the
TPU graphs at every engine start, which the precompiled models remove; energy has not been
re-measured on 2.3.0.

## Layout
* `android/` — the app (`:app`); `:nnprobe` is the old NNAPI probe used to map TPU support,
  `:aotprobe` runs one AOT-compiled graph (used by `convert/aot_numcheck.py`).
* `convert/` — export pipeline: `kokoro_graphs.py` (graph rewrites), `folded.py` (row-folded
  generator), `host_dsp.py` (host DSP spec), `export_all.py`, `aot_compile_g3.py`, verification
  scripts (`aot_verify_dump.py`: every TPU graph vs host on activations dumped by the app).
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
.venv-convert/bin/python convert/export_all.py           # ~45 min: assets/kokoro + work/npu_src (TPU graphs)
.venv-convert/bin/python convert/build_lexicon.py        # G2P dictionaries
# Google Tensor SDK (beta tarball litert_plugin_compiler.tar.gz; Linux x86-64 only)
uv venv -p 3.12 .venv-aot
GOOGLE_TENSOR_SDK_BETA=$PWD/litert_plugin_compiler.tar.gz VIRTUAL_ENV=.venv-aot uv pip install \
    --no-cache ai-edge-litert==2.3.0 ai-edge-litert-sdk-google-tensor==2.3.0
.venv-aot/bin/python convert/aot_compile_g3.py           # ~1.5 min: work/npu_src -> assets/kokoro/npu
cd android && ./gradlew :app:assembleRelease
```
espeak-ng's compiled English data (`android/app/src/main/assets/espeak-ng-data`) comes from a
host build of the submodule (`cmake -S third_party/espeak-ng -B work/espeak-host && ninja -C work/espeak-host`).

## License
Kokoro-82M weights: Apache-2.0. espeak-ng is GPL-3.0, so the APK as a whole is GPL-3.0.
