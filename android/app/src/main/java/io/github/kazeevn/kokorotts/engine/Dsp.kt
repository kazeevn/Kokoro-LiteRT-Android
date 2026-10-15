package io.github.kazeevn.kokorotts.engine

import kotlin.math.PI
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tanh

/**
 * Non-neural signal processing of the split Kokoro pipeline. Executable spec:
 * convert/host_dsp.py (validated against the PyTorch model); keep the two in sync.
 */
class Dsp(
    private val lLinearW: FloatArray, // [9] SourceModuleHnNSF.l_linear weight
    private val lLinearB: Float,
    private val f0ConvW: FloatArray, // [3] Decoder.F0_conv
    private val f0ConvB: Float,
    private val nConvW: FloatArray, // [3] Decoder.N_conv
    private val nConvB: Float,
    private val withNoise: Boolean = true,
) {
    companion object {
        const val SR = 24000
        const val UPSAMPLE = 300
        const val N_FFT = 20
        const val HOP = 5
        const val BINS = N_FFT / 2 + 1
        const val HARMONICS = 9
        const val SINE_AMP = 0.1
        const val NOISE_STD = 0.003
        const val VOICED_THRESHOLD = 10.0

        /** Periodic Hann window. */
        val HANN = DoubleArray(N_FFT) { 0.5 - 0.5 * cos(2 * PI * it / N_FFT) }
        private val COS = Array(BINS) { k -> DoubleArray(N_FFT) { t -> cos(2 * PI * k * t / N_FFT) } }
        private val SIN = Array(BINS) { k -> DoubleArray(N_FFT) { t -> sin(2 * PI * k * t / N_FFT) } }

        /** Gaussian noise table; the hn-NSF noise is tiny, so a reused table is indistinguishable. */
        private val NOISE: FloatArray = java.util.Random(1234).let { r -> FloatArray(1 shl 16) { r.nextGaussian().toFloat() } }
    }

    private var noisePos = 0

    /**
     * hn-NSF excitation for an F0 curve of [nFrames] values (2 per acoustic frame).
     * Returns har_source of nFrames * 300 samples. Mirrors host_dsp.harmonic_source.
     */
    fun harmonicSource(f0: FloatArray, nFrames: Int): FloatArray {
        noisePos = 0 // deterministic per chunk (same input -> same audio on every backend)
        val n = nFrames * UPSAMPLE
        // phase[f][h] = 2*pi*300 * cumsum(frac(f0*h/SR))
        val phase = Array(HARMONICS) { DoubleArray(nFrames) }
        for (h in 0 until HARMONICS) {
            var acc = 0.0
            val p = phase[h]
            for (f in 0 until nFrames) {
                val r = f0[f].toDouble() * (h + 1) / SR
                acc += r - floor(r)
                p[f] = acc * 2 * PI * UPSAMPLE
            }
        }
        val out = FloatArray(n)
        val s = DoubleArray(HARMONICS)
        val c = DoubleArray(HARMONICS)
        val ds = DoubleArray(HARMONICS)
        val dc = DoubleArray(HARMONICS)
        // F.interpolate(linear, align_corners=False, x300): src(i) = (i + 0.5)/300 - 0.5, clamped at 0.
        // Samples [0, 150) sit at phase[0]; then segment k covers [150 + 300k, 450 + 300k) with
        // i0 = k and a phase linear in i, so sin/cos advance by a fixed rotation per sample.
        var start = 0
        var k = -1
        while (start < n) {
            val end: Int
            if (k < 0) {
                end = minOf(UPSAMPLE / 2, n)
                for (h in 0 until HARMONICS) {
                    s[h] = sin(phase[h][0]); c[h] = cos(phase[h][0]); ds[h] = 0.0; dc[h] = 1.0
                }
            } else {
                end = minOf(start + UPSAMPLE, n)
                val i1 = minOf(k + 1, nFrames - 1)
                for (h in 0 until HARMONICS) {
                    val dp = phase[h][i1] - phase[h][k]
                    val ph = phase[h][k] + dp * (0.5 / UPSAMPLE)
                    s[h] = sin(ph); c[h] = cos(ph)
                    ds[h] = sin(dp / UPSAMPLE); dc[h] = cos(dp / UPSAMPLE)
                }
            }
            for (i in start until end) {
                val f0v = f0[i / UPSAMPLE].toDouble()
                val voiced = f0v > VOICED_THRESHOLD
                val noiseAmp = if (!withNoise) 0.0 else if (voiced) NOISE_STD else SINE_AMP / 3
                var acc = lLinearB.toDouble()
                for (h in 0 until HARMONICS) {
                    val sine = if (voiced) SINE_AMP * s[h] else 0.0
                    acc += (sine + noiseAmp * NOISE[noisePos]) * lLinearW[h]
                    noisePos = (noisePos + 1) and (NOISE.size - 1)
                    val ns = s[h] * dc[h] + c[h] * ds[h]
                    c[h] = c[h] * dc[h] - s[h] * ds[h]
                    s[h] = ns
                }
                out[i] = tanh(acc).toFloat()
            }
            start = end
            k++
        }
        return out
    }

    /**
     * torch.stft(n_fft=20, hop=5, hann, center=True, reflect) -> magnitude and phase, written into
     * [dst] laid out as [24 channels][stride] (channels 0..10 mag, 11..21 phase, 22..23 zero).
     * Returns the number of frames (len/5 + 1).
     */
    fun stftInto(x: FloatArray, dst: FloatArray, stride: Int): Int {
        val pad = N_FFT / 2
        val len = x.size
        val nFrames = 1 + len / HOP
        fun at(i: Int): Double {
            var j = i - pad
            if (j < 0) j = -j
            if (j >= len) j = 2 * (len - 1) - j
            return x[j].toDouble()
        }
        val frame = DoubleArray(N_FFT)
        for (f in 0 until nFrames) {
            val base = f * HOP
            for (t in 0 until N_FFT) frame[t] = at(base + t) * HANN[t]
            for (k in 0 until BINS) {
                var re = 0.0
                var im = 0.0
                val ck = COS[k]
                val sk = SIN[k]
                for (t in 0 until N_FFT) {
                    re += frame[t] * ck[t]
                    im -= frame[t] * sk[t]
                }
                dst[k * stride + f] = sqrt(re * re + im * im).toFloat()
                dst[(BINS + k) * stride + f] = atan2(im, re).toFloat()
            }
        }
        return nFrames
    }

    /**
     * iSTFT of the decoder's conv_post output [channels][stride] (spec = exp(x[0..10]),
     * phase = sin(x[11..21])) over [nFrames] frames; returns 5 * (nFrames - 1) samples.
     */
    fun istft(x: FloatArray, stride: Int, nFrames: Int): FloatArray {
        val outLen = N_FFT + HOP * (nFrames - 1)
        val y = DoubleArray(outLen)
        val env = DoubleArray(outLen)
        val re = DoubleArray(BINS)
        val im = DoubleArray(BINS)
        for (f in 0 until nFrames) {
            for (k in 0 until BINS) {
                val mag = exp(x[k * stride + f].toDouble())
                val ph = sin(x[(BINS + k) * stride + f].toDouble())
                re[k] = mag * cos(ph)
                im[k] = mag * sin(ph)
            }
            val base = f * HOP
            for (t in 0 until N_FFT) {
                // numpy irfft: imag parts of DC and Nyquist bins are ignored
                var v = re[0] + re[BINS - 1] * (if (t % 2 == 0) 1.0 else -1.0)
                for (k in 1 until BINS - 1) v += 2 * (re[k] * COS[k][t] - im[k] * SIN[k][t])
                v /= N_FFT
                y[base + t] += v * HANN[t]
                env[base + t] += HANN[t] * HANN[t]
            }
        }
        val pad = N_FFT / 2
        val n = HOP * (nFrames - 1)
        return FloatArray(n) { i ->
            val e = env[pad + i]
            (y[pad + i] / (if (e > 1e-11) e else 1.0)).toFloat()
        }
    }

    /** Decoder F0_conv/N_conv (k=3, stride 2, pad 1) over curves of length 2L -> [4][L] channels. */
    fun f0nFeaturesInto(f0: FloatArray, nCurve: FloatArray, len2: Int, dst: FloatArray, L: Int) {
        fun conv(src: FloatArray, w: FloatArray, b: Float, ch: Int) {
            for (i in 0 until L) {
                val a = 2 * i - 1
                val x0 = if (a >= 0 && a < len2) src[a] else 0f
                val x1 = if (a + 1 < len2) src[a + 1] else 0f
                val x2 = if (a + 2 < len2) src[a + 2] else 0f
                dst[ch * L + i] = b + w[0] * x0 + w[1] * x1 + w[2] * x2
            }
        }
        java.util.Arrays.fill(dst, 0, 4 * L, 0f)
        conv(f0, f0ConvW, f0ConvB, 0)
        conv(nCurve, nConvW, nConvB, 1)
    }
}
