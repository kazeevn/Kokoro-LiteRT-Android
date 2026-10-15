package io.github.kazeevn.kokorotts.engine

import java.nio.ByteBuffer
import kotlin.math.sqrt

/**
 * The generator's last (120x) rate on the EdgeTPU as fixed-size row blocks.
 * Port of convert/folded.py: FinalBlock / run_final_blocks — keep them in sync.
 *
 * Tensors are row-folded [C][rows][W]. Each block graph processes [rows] = R interior rows
 * (+1 halo row each side for convolutions) and returns per-block statistics (mean over valid
 * samples, centered second moment over the block) of its output; the host combines them into the
 * global mean / rstd that the next graph's instance norm uses.
 */
class FinalBlocks(
    private val graphs: Map<String, Graph>,
    private val pool: BufferPool,
    private val R: Int,
) {
    companion object {
        const val W = 480
        const val W20 = 80
        const val C = 128
        const val EPS = 1e-5
        private const val SIG = "B"
    }

    private class Stats(val mean: FloatArray, val rstd: FloatArray)

    /** Time spent inside TPU graph calls vs total (for profiling), in nanoseconds. */
    var graphNs = 0L
    var totalNs = 0L

    private val arrays = HashMap<String, FloatArray>()

    /** Reused full-size work arrays (avoids ~200 MB of garbage per chunk). */
    private fun work(key: String, n: Int): FloatArray {
        val a = arrays[key]
        if (a != null && a.size == n) return a
        return FloatArray(n).also { arrays[key] = it }
    }

    private fun runGraph(name: String, ins: List<ByteBuffer>, outs: List<ByteBuffer>) {
        val t = System.nanoTime()
        graphs.getValue(name).run(SIG, ins, outs)
        graphNs += System.nanoTime() - t
    }

    /**
     * ys: 3 x [256][H][80] (rate-20 resblock outputs), har: [24][H][480], m20: [H][80],
     * mP: [H][480] (validity masks). Writes conv_post output [24][H][480] into [out].
     */
    fun run(ys: List<FloatArray>, har: FloatArray, m20: FloatArray, mP: FloatArray, H: Int,
            refS: ByteBuffer, out: FloatArray) {
        val t0 = System.nanoTime()
        try { runImpl(ys, har, m20, mP, H, refS, out) } finally { totalNs += System.nanoTime() - t0 }
    }

    private fun runImpl(ys: List<FloatArray>, har: FloatArray, m20: FloatArray, mP: FloatArray, H: Int,
                        refS: ByteBuffer, out: FloatArray) {
        val nb = (H + R - 1) / R
        val hp = nb * R
        val counts = IntArray(nb) { b ->
            var n = 0
            for (r in b * R until minOf(b * R + R, H)) for (w in 0 until W) if (mP[r * W + w] > 0.5f) n++
            n
        }
        // validity masks per block, built once per chunk (they are the same for every step)
        val masks = Array(2) { halo -> Array(nb) { b -> rows(mP, 1, H, W, b * R - halo, R + 2 * halo, "mP${halo}_$b") } }
        val mPbuf = { b: Int, halo: Int -> masks[halo][b] }

        // noise branch: 1x1 conv on har, then noise_res[1]
        val src0 = work("src0", C * hp * W)
        var st = blockStats(nb, counts, src0, hp) { b ->
            run1("src0", listOf(rows(har, 24, H, W, b * R, R, "har"), mPbuf(b, 0)))
        }
        val src = resblock("nr", src0, hp, nb, counts, st, mPbuf, refS).first

        // upsampled main branch (+ reflect pad of the very first sample), then add
        val xUp = work("xup", C * hp * W)
        for (b in 0 until nb) {
            val ins = ys.mapIndexed { j, y -> rows(y, 256, H, W20, b * R - 1, R + 2, "y$j") } +
                rows(m20, 1, H, W20, b * R - 1, R + 2, "m20")
            val o = pool.floats("blk_out0", C * R * W)
            runGraph("blk_g1u", ins, listOf(o))
            putRows(xUp, C, hp, W, b * R, o)
        }
        for (c in 0 until C) xUp[c * hp * W] = xUp[c * hp * W + 2]
        val xP = work("xP", C * hp * W)
        st = blockStats(nb, counts, xP, hp) { b ->
            run1("add", listOf(rows(xUp, C, hp, W, b * R, R, "xup"), rows(src, C, hp, W, b * R, R, "src"), mPbuf(b, 0)))
        }

        val zs = (0 until 3).map { j -> resblock("r$j", xP, hp, nb, counts, st, mPbuf, refS).first }
        for (b in 0 until nb) {
            val ins = zs.mapIndexed { j, z -> rows(z, C, hp, W, b * R - 1, R + 2, "z$j") } + mPbuf(b, 1)
            val o = pool.floats("blk_post", 24 * R * W)
            runGraph("blk_post", ins, listOf(o))
            // copy interior rows into out [24][H][480] (dropping padded rows)
            val fb = o.asFloatBuffer()
            for (c in 0 until 24) for (r in 0 until R) {
                val gr = b * R + r
                if (gr >= H) continue
                fb.position((c * R + r) * W)
                fb.get(out, (c * H + gr) * W, W)
            }
        }
    }

    /** Runs a block graph with outputs (tensor, mean, m2); returns the output buffers. */
    private fun run1(kind: String, ins: List<ByteBuffer>): List<ByteBuffer> {
        val outs = listOf(pool.floats("blk_out0", C * R * W), pool.floats("blk_mean", C), pool.floats("blk_m2", C))
        runGraph("blk_$kind", ins, outs)
        return outs
    }

    /**
     * Runs [f] per block, writing each block's output tensor into rows of [dst] ([C][hp][W]) and
     * combining the per-block statistics (parallel variance formula).
     */
    private fun blockStats(nb: Int, counts: IntArray, dst: FloatArray, hp: Int, f: (Int) -> List<ByteBuffer>): Stats {
        val means = Array(nb) { FloatArray(C) }
        val m2s = Array(nb) { FloatArray(C) }
        for (b in 0 until nb) {
            val o = f(b)
            putRows(dst, C, hp, W, b * R, o[0])
            o[1].asFloatBuffer().get(means[b])
            o[2].asFloatBuffer().get(m2s[b])
        }
        val n = counts.sum().toDouble()
        val size = (R * W).toDouble()
        val mean = FloatArray(C)
        val rstd = FloatArray(C)
        for (c in 0 until C) {
            var mu = 0.0
            for (b in 0 until nb) if (counts[b] > 0) mu += counts[b] * means[b][c].toDouble()
            mu /= n
            var m2 = 0.0
            for (b in 0 until nb) if (counts[b] > 0) {
                val d = means[b][c] - mu
                m2 += m2s[b][c] * size + counts[b] * d * d
            }
            mean[c] = mu.toFloat()
            rstd[c] = (1.0 / sqrt(m2 / n + EPS)).toFloat()
        }
        return Stats(mean, rstd)
    }

    private fun resblock(
        name: String, xIn: FloatArray, hp: Int, nb: Int, counts: IntArray, stIn: Stats,
        mPbuf: (Int, Int) -> ByteBuffer, refS: ByteBuffer,
    ): Pair<FloatArray, Stats> {
        var x = xIn
        var st = stIn
        for (k in 0 until 3) {
            val h = work("h", C * hp * W)
            val sth = blockStats(nb, counts, h, hp) { b ->
                run1("${name}s${k}a", listOf(rows(x, C, hp, W, b * R - 1, R + 2, "x"), mPbuf(b, 1),
                    statBuf("mean", st.mean), statBuf("rstd", st.rstd), refS))
            }
            // step outputs alternate between two arrays; the last one is this block's own result
            val y = work(if (k == 2) "out_$name" else "tmp${k % 2}", C * hp * W)
            st = blockStats(nb, counts, y, hp) { b ->
                run1("${name}s${k}b", listOf(rows(h, C, hp, W, b * R - 1, R + 2, "h"), mPbuf(b, 1),
                    statBuf("meanh", sth.mean), statBuf("rstdh", sth.rstd), rows(x, C, hp, W, b * R, R, "xi"), refS))
            }
            x = y
        }
        return x to st
    }

    private fun statBuf(key: String, v: FloatArray): ByteBuffer =
        pool.floats("blk_$key", C).also { it.asFloatBuffer().put(v) }

    /**
     * Copies rows [lo, lo+n) of a [ch][srcRows][w] tensor into a pooled buffer [ch][n][w],
     * zero-filling rows outside [0, srcRows).
     */
    private fun rows(src: FloatArray, ch: Int, srcRows: Int, w: Int, lo: Int, n: Int, key: String): ByteBuffer {
        val buf = pool.floats("rows_$key", ch * n * w)
        val fb = buf.asFloatBuffer()
        // rows of one channel are contiguous: copy the in-range span with one bulk put
        val a = maxOf(lo, 0)
        val b = minOf(lo + n, srcRows)
        val before = (a - lo) * w
        val after = (lo + n - maxOf(b, a)) * w
        val zeros = FloatArray(maxOf(before, after))
        for (c in 0 until ch) {
            if (before > 0) fb.put(zeros, 0, before)
            if (b > a) fb.put(src, (c * srcRows + a) * w, (b - a) * w)
            if (after > 0) fb.put(zeros, 0, after)
        }
        return buf
    }

    /** Writes a block output [ch][R][w] into rows [lo, lo+R) of a [ch][dstRows][w] tensor. */
    private fun putRows(dst: FloatArray, ch: Int, dstRows: Int, w: Int, lo: Int, block: ByteBuffer) {
        val fb = block.duplicate().order(java.nio.ByteOrder.nativeOrder()).also { it.rewind() }.asFloatBuffer()
        for (c in 0 until ch) fb.get(dst, (c * dstRows + lo) * w, R * w)
    }
}
