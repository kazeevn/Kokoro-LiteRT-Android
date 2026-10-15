"""Golden data for the Kotlin Dsp unit test (noise disabled): writes android/app/src/test/resources/dsp/."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import host_dsp  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "android", "app", "src", "test", "resources", "dsp")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
F = 64
f0 = (rng.random(F) * 180 + 80).astype(np.float32)
f0[20:28] = 0
w = (rng.standard_normal(9) * 0.3).astype(np.float32)
b = np.float32(0.05)
src = host_dsp.harmonic_source(f0, w[None, :], np.array([b]), noise=np.zeros((F * 300, 9)))
mag, ph = host_dsp.stft_mag_phase(src)
x = (rng.standard_normal((22, 401)) * 0.5).astype(np.float32)
y = host_dsp.istft(x)
for name, arr in dict(f0=f0, w=w, b=np.array([b]), src=src, mag=mag, ph=ph, x=x, y=y).items():
    np.asarray(arr, "<f4").tofile(os.path.join(OUT, f"{name}.bin"))
print("ok", src.shape, mag.shape, y.shape)
