package io.github.kazeevn.kokorotts.engine

import android.content.Context
import android.util.Log
import io.github.kazeevn.kokorotts.g2p.EnglishG2P
import io.github.kazeevn.kokorotts.g2p.Espeak
import io.github.kazeevn.kokorotts.g2p.EspeakFallback
import io.github.kazeevn.kokorotts.g2p.G2PToken
import io.github.kazeevn.kokorotts.g2p.Lexicon
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean

/** One synthesized chunk: 24 kHz mono samples plus where each word starts (sample offsets). */
class Chunk(val audio: FloatArray, val words: List<WordMark>, val stats: ChunkStats)

class WordMark(val sample: Int, val start: Int, val end: Int)

class ChunkStats(val tokens: Int, val frames: Int, val bucket: Int, val ms: Map<String, Long>)

/**
 * Kokoro-82M on LiteRT, split into static-shape graphs (see convert/kokoro_graphs.py):
 * text (token buckets) -> host alignment -> f0n -> host hn-NSF source + STFT ->
 * decoder stages (frame buckets; CPU via XNNPACK or AOT-compiled for the EdgeTPU) -> host iSTFT.
 */
class KokoroEngine private constructor(private val context: Context) {
    private val assets = context.assets
    private val meta = JSONObject(assets.open("kokoro/kokoro.json").bufferedReader().readText())
    private val vocab: Map<Char, Int> = meta.getJSONObject("vocab").let { v ->
        v.keys().asSequence().associate { it[0] to v.getInt(it) }
    }
    private val tokenBuckets = meta.getJSONArray("token_buckets").let { a -> IntArray(a.length()) { a.getInt(it) } }
    val frameBuckets = meta.getJSONArray("frame_buckets").let { a -> IntArray(a.length()) { a.getInt(it) } }
    /** The EdgeTPU decoder only handles chunks up to the largest of these (see export_all.py). */
    private val npuFrameBuckets = meta.getJSONArray("npu_frame_buckets").let { a -> IntArray(a.length()) { a.getInt(it) } }
    val npuMaxFrames get() = npuFrameBuckets.last()

    /** Optional cap on frames per chunk (e.g. to run the CPU with the NPU's chunking). */
    @Volatile var frameCap: Int = 0

    private fun maxFrames(): Int {
        val m = if (activeDecoderBackend == Backend.NPU) npuFrameBuckets.last() else frameBuckets.last()
        return if (frameCap > 0) minOf(m, frameCap) else m
    }
    private val decStages = meta.getJSONArray("decoder_stages").let { a -> List(a.length()) { a.getString(it) } }
    private val finalBlocks = meta.getJSONArray("final_blocks").let { a -> List(a.length()) { a.getString(it) } }
    private val blockRows = meta.getInt("block_rows")
    val voices: List<String> = meta.getJSONArray("voices").let { a -> List(a.length()) { a.getString(it) } }
    private val dsp = Dsp(
        floats(meta, "l_linear_w"), meta.getDouble("l_linear_b").toFloat(),
        floats(meta, "f0_conv_w"), meta.getDouble("f0_conv_b").toFloat(),
        floats(meta, "n_conv_w"), meta.getDouble("n_conv_b").toFloat(),
    )

    private val lock = Any()
    private val pool = BufferPool()
    private val voiceCache = HashMap<String, FloatArray>()
    private val g2p = HashMap<Boolean, EnglishG2P>()

    private var textGraph: Graph? = null
    private var f0nGraph: Graph? = null
    private val textLoaded get() = textGraph != null
    private val f0nLoaded get() = f0nGraph != null
    private val text: Graph get() = textGraph ?: cpuGraph("text").also { textGraph = it }
    private val f0n: Graph get() = f0nGraph ?: cpuGraph("f0n").also { f0nGraph = it }

    private val xnnCacheDir: File by lazy {
        val version = context.packageManager.getPackageInfo(context.packageName, 0).lastUpdateTime
        File(context.cacheDir, "xnnpack-$version").also { dir ->
            context.cacheDir.listFiles { f -> f.name.startsWith("xnnpack-") && f != dir }?.forEach { it.deleteRecursively() }
            dir.mkdirs()
        }
    }

    /** CPU tuning; changing them reloads the CPU graphs on next use. */
    @Volatile var cpuThreads = THREADS
    @Volatile var cpuFp16 = false

    private fun cpuGraph(name: String): Graph {
        val tag = if (cpuFp16) "fp16" else "fp32"
        return Graph(context, name, Backend.CPU, cpuThreads, File(xnnCacheDir, "$name.$tag.xnn"),
            if (cpuFp16) Graph.XNN_FLAG_FORCE_FP16 else 0)
    }

    /** Applies CPU tuning, dropping CPU graphs built with other settings. */
    fun configureCpu(threads: Int, fp16: Boolean) = synchronized(lock) {
        if (threads == cpuThreads && fp16 == cpuFp16) return@synchronized
        cpuThreads = threads
        cpuFp16 = fp16
        if (textLoaded) { textGraph?.close(); textGraph = null }
        if (f0nLoaded) { f0nGraph?.close(); f0nGraph = null }
        decoderCpu?.close(); decoderCpu = null
    }
    /** CPU decoder: one 1-D graph (one activation workspace); see convert/export_all.py. */
    private var decoderCpu: Graph? = null
    @Volatile private var decoderNpu: Map<String, Graph>? = null
    @Volatile var npuState: String = "not loaded"
        private set

    /** Backend the decoder should use; NPU falls back to CPU until its graphs are loaded. */
    @Volatile var preferredBackend: Backend = Backend.CPU

    val activeDecoderBackend: Backend get() =
        if (preferredBackend == Backend.NPU && decoderNpu != null) Backend.NPU else Backend.CPU

    // ---------------------------------------------------------------- loading

    /**
     * EdgeTPU decoder: TPU-sized, row-folded stage graphs + last-rate row-block graphs, all
     * AOT-compiled for the Tensor G3 (convert/aot_compile_g3.py).
     */
    private fun loadNpuDecoder(): Map<String, Graph> {
        val graphs = LinkedHashMap<String, Graph>()
        try {
            for (st in decStages) graphs[st] = Graph(context, "dec_$st", Backend.NPU, THREADS)
            for (b in finalBlocks) graphs["blk_$b"] = Graph(context, "blk_$b", Backend.NPU, THREADS)
        } catch (e: Throwable) {
            graphs.values.forEach { it.close() }
            throw e
        }
        return graphs
    }

    /** Loads the precompiled decoder onto the EdgeTPU (seconds). Blocking. */
    fun prepareNpu(): Boolean {
        synchronized(npuLock) {
            if (decoderNpu != null) return true
            npuState = "loading"
            return try {
                val t0 = System.nanoTime()
                decoderNpu = loadNpuDecoder()
                npuState = "ready (loaded in ${(System.nanoTime() - t0) / 1_000_000} ms)"
                // The CPU decoder is no longer needed: free its packed weights.
                synchronized(lock) { decoderCpu?.close(); decoderCpu = null }
                true
            } catch (e: Throwable) {
                Log.e(TAG, "NPU load failed", e)
                npuState = "failed: ${e.message}"
                false
            }
        }
    }

    private val npuLock = Any()

    fun prepareNpuAsync() {
        if (decoderNpu != null || npuState == "loading") return
        Thread({ prepareNpu() }, "kokoro-npu-load").apply { priority = Thread.MIN_PRIORITY }.start()
    }

    fun releaseNpu() {
        synchronized(npuLock) {
            synchronized(lock) { decoderNpu?.values?.forEach { it.close() }; decoderNpu = null; finalBlocksRunner = null }
            npuState = "not loaded"
        }
    }

    private fun g2pFor(british: Boolean): EnglishG2P = g2p.getOrPut(british) {
        val dir = File(context.filesDir, "espeak")
        ensureEspeakData(dir)
        Espeak.init(dir.absolutePath)
        val p = if (british) "gb" else "us"
        val lex = assets.open("g2p/${p}_gold.tsv").use { g ->
            assets.open("g2p/${p}_silver.tsv").use { s -> Lexicon.load(g, s, british) }
        }
        val voice = if (british) "en-gb" else "en-us"
        EnglishG2P(lex, british, EspeakFallback(british) { Espeak.phonemize(it, voice) })
    }

    private fun ensureEspeakData(dir: File) {
        val stamp = File(dir, ".version")
        val version = context.packageManager.getPackageInfo(context.packageName, 0).lastUpdateTime.toString()
        if (stamp.exists() && stamp.readText() == version) return
        fun copy(path: String) {
            val children = assets.list(path) ?: emptyArray()
            if (children.isEmpty()) {
                val out = File(dir, path)
                out.parentFile?.mkdirs()
                assets.open(path).use { i -> out.outputStream().use { o -> i.copyTo(o) } }
            } else children.forEach { copy("$path/$it") }
        }
        copy("espeak-ng-data")
        stamp.writeText(version)
    }

    private fun voicePack(name: String): FloatArray = voiceCache.getOrPut(name) {
        val bytes = assets.open("kokoro/voices/$name.bin").use { it.readBytes() }
        val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        FloatArray(fb.remaining()).also { fb.get(it) }
    }

    /** Loads everything needed for CPU synthesis (and the G2P for [british]). */
    fun warmUp(british: Boolean = false) {
        synchronized(lock) {
            g2pFor(british)
            text; f0n
            if (activeDecoderBackend == Backend.CPU && decoderCpu == null) decoderCpu = cpuGraph("dec_cpu")
        }
    }

    // ---------------------------------------------------------------- synthesis

    fun isBritish(voice: String) = voice.startsWith("b")

    /**
     * Synthesizes [text] with [voice] at [speed] (1 = normal), delivering chunks in order to
     * [sink]; stops early when [cancel] is set or [sink] returns false.
     */
    fun synthesize(text: String, voice: String, speed: Float, cancel: AtomicBoolean, sink: (Chunk) -> Boolean) {
        val british = isBritish(voice)
        val tokens = synchronized(lock) { g2pFor(british).phonemize(text).second }
        // ~2.5 frames per phoneme at speed 1: keep typical chunks within the largest bucket
        val limit = (maxFrames() / FRAMES_PER_PHONEME * speed.coerceIn(0.5f, 2f)).toInt()
        for (chunk in chunkTokens(tokens, limit)) {
            if (cancel.get()) return
            for (part in synthesizeChunk(chunk, voice, speed, cancel)) {
                if (cancel.get() || !sink(part)) return
            }
        }
    }

    private fun psOf(tks: List<G2PToken>) = tks.joinToString("") { it.phonemes + if (it.whitespace.isNotEmpty()) " " else "" }.trim()

    /** KPipeline.en_tokenize with a phoneme budget of [limit] per chunk. */
    private fun chunkTokens(tokens: List<G2PToken>, limit: Int): List<List<G2PToken>> {
        val out = ArrayList<List<G2PToken>>()
        var tks = ArrayList<G2PToken>()
        var pcount = 0
        for (t in tokens) {
            var nextPs = t.phonemes + if (t.whitespace.isNotEmpty()) " " else ""
            val nextPcount = pcount + nextPs.trimEnd().length
            if (nextPcount > limit && tks.isNotEmpty()) {
                val z = waterfallLast(tks, nextPcount, limit)
                out.add(tks.subList(0, z).toList())
                tks = ArrayList(tks.subList(z, tks.size))
                pcount = psOf(tks).length
                if (tks.isEmpty()) nextPs = nextPs.trimStart()
            }
            tks.add(t)
            pcount += nextPs.length
        }
        if (tks.isNotEmpty()) out.add(tks)
        return out.filter { psOf(it).isNotEmpty() }
    }

    private fun waterfallLast(tks: List<G2PToken>, nextCount: Int, limit: Int): Int {
        for (w in listOf("!.?…", ":;", ",—")) {
            var z = tks.indexOfLast { it.phonemes.length == 1 && it.phonemes[0] in w }
            if (z < 0) continue
            z += 1
            if (z < tks.size && tks[z].phonemes in setOf(")", "”")) z += 1
            if (nextCount - psOf(tks.subList(0, z)).length <= limit) return z
        }
        return tks.size
    }

    private fun synthesizeChunk(tks: List<G2PToken>, voice: String, speed: Float, cancel: AtomicBoolean): List<Chunk> {
        // phoneme string -> ids, remembering where each token's phonemes start
        val ps = StringBuilder()
        val tokenStartChar = IntArray(tks.size)
        for ((i, t) in tks.withIndex()) {
            tokenStartChar[i] = ps.length
            ps.append(t.phonemes)
            if (t.whitespace.isNotEmpty()) ps.append(' ')
        }
        // Kokoro's pipeline strips the phoneme string before mapping it to ids
        val first = ps.indexOfFirst { !it.isWhitespace() }
        if (first < 0) return emptyList()
        val last = ps.indexOfLast { !it.isWhitespace() }
        val maxIds = tokenBuckets.last() - 2
        val ids = ArrayList<Int>()
        val idAtChar = IntArray(ps.length + 1)
        for ((c, ch) in ps.withIndex()) {
            idAtChar[c] = ids.size
            if (c < first || c > last) continue
            val id = vocab[ch] ?: continue
            if (ids.size < maxIds) ids.add(id)
        }
        idAtChar[ps.length] = ids.size
        if (ids.isEmpty()) return emptyList()
        val psLen = last - first + 1

        return synchronized(lock) {
            val result = runModel(ids, psLen, voice, speed, cancel) ?: return@synchronized emptyList()
            val (audio, durs, stats) = result
            if (durs == null) {
                // Too long for the largest frame bucket: split the chunk and retry each half.
                val mid = splitPoint(tks)
                if (mid <= 0 || mid >= tks.size) return@synchronized emptyList()
                return@synchronized synthesizeChunk(tks.subList(0, mid), voice, speed, cancel) +
                    synthesizeChunk(tks.subList(mid, tks.size), voice, speed, cancel)
            }
            // word marks: frame where each token's first phoneme starts (+1 for the BOS token)
            val cum = IntArray(durs.size + 1)
            for (i in durs.indices) cum[i + 1] = cum[i] + durs[i]
            val marks = tks.indices.filter { tks[it].end > tks[it].start && tks[it].phonemes.any(Char::isLetter) }.map { i ->
                val id = idAtChar[tokenStartChar[i]].coerceAtMost(durs.size - 2)
                WordMark(cum[id + 1] * 600, tks[i].start, tks[i].end)
            }
            listOf(Chunk(audio!!, marks, stats))
        }
    }

    private fun splitPoint(tks: List<G2PToken>): Int {
        val mid = tks.size / 2
        val punct = (1 until tks.size).filter { tks[it - 1].phonemes.let { p -> p.length == 1 && p[0] in ",;:—.!?" } }
        return punct.minByOrNull { kotlin.math.abs(it - mid) } ?: mid
    }

    /**
     * Runs the graphs for one chunk of phoneme ids. Returns (audio, durations incl. BOS/EOS, stats),
     * or (null, null, stats) if the predicted length exceeds the largest frame bucket.
     */
    private fun runModel(idsIn: List<Int>, psLen: Int, voice: String, speed: Float, cancel: AtomicBoolean):
        Triple<FloatArray?, IntArray?, ChunkStats>? {
        val ms = LinkedHashMap<String, Long>()
        var t0 = System.nanoTime()
        fun lap(key: String) { val t = System.nanoTime(); ms[key] = (ms[key] ?: 0) + (t - t0) / 1_000_000; t0 = t }

        val n = idsIn.size + 2
        val T = tokenBuckets.first { it >= n }
        val pack = voicePack(voice)
        val row = (psLen - 1).coerceIn(0, pack.size / 256 - 1)
        val refS = pool.floats("ref_s", 256).also { b -> b.asFloatBuffer().put(pack, row * 256, 256) }

        // ---- text graph
        val ids = pool.get("ids", T * 4).also { b ->
            val ib = b.asIntBuffer(); ib.put(0); idsIn.forEach { ib.put(it) }; ib.put(0); while (ib.hasRemaining()) ib.put(0)
        }
        val mask = pool.floats("tmask", T).also { b -> val f = b.asFloatBuffer(); for (i in 0 until T) f.put(if (i < n) 1f else 0f) }
        val rev = pool.get("trev", T * 4).also { b -> val ib = b.asIntBuffer(); for (i in 0 until T) ib.put(if (i < n) n - 1 - i else i) }
        val durB = pool.floats("dur", T)
        val dB = pool.floats("d", T * 640)
        val tenB = pool.floats("t_en", 512 * T)
        text.run("T$T", listOf(ids, mask, rev, refS), listOf(durB, dB, tenB))
        lap("text")
        Log.d(TAG, "after text T$T: RSS ${Graph.rssMb()} MB")

        val durRaw = FloatArray(T).also { durB.asFloatBuffer().get(it) }
        val durs = IntArray(n) { i -> maxOf(1.0, Math.rint(durRaw[i] / speed.toDouble())).toInt() }
        val L0 = durs.sum()
        val npu = decoderNpu.takeIf { preferredBackend == Backend.NPU }
        val buckets = if (npu != null) npuFrameBuckets else frameBuckets
        val cap = if (frameCap > 0) minOf(buckets.last(), frameCap) else buckets.last()
        if (L0 > cap) return Triple(null, null, ChunkStats(n, L0, 0, ms))
        val L = buckets.first { it >= L0 }
        val H = L / 4 + 1

        // ---- frame expansion
        val d = FloatArray(T * 640).also { dB.asFloatBuffer().get(it) }
        val tEn = FloatArray(512 * T).also { tenB.asFloatBuffer().get(it) }
        val tokOf = IntArray(L0)
        run { var f = 0; for (i in 0 until n) repeat(durs[i]) { tokOf[f++] = i } }
        val en = pool.floats("en", 640 * L).also { b ->
            val fb = b.asFloatBuffer()
            for (c in 0 until 640) for (f in 0 until L) fb.put(if (f < L0) d[tokOf[f] * 640 + c] else 0f)
        }
        val asr = pool.floats("asr", 512 * L).also { b ->
            val fb = b.asFloatBuffer()
            for (c in 0 until 512) for (f in 0 until L) fb.put(if (f < L0) tEn[c * T + tokOf[f]] else 0f)
        }
        val frev = pool.get("frev", L * 4).also { b -> val ib = b.asIntBuffer(); for (i in 0 until L) ib.put(if (i < L0) L0 - 1 - i else i) }
        val m1 = maskBuf("m1", L0, L)
        val m2 = maskBuf("m2", 2 * L0, 2 * L)
        lap("expand")

        // ---- f0n graph
        val f0B = pool.floats("f0", 2 * L)
        val nB = pool.floats("n", 2 * L)
        f0n.run("L$L", listOf(en, frev, m1, m2, refS), listOf(f0B, nB))
        Log.d(TAG, "after f0n L$L: RSS ${Graph.rssMb()} MB")
        val f0 = FloatArray(2 * L).also { f0B.asFloatBuffer().get(it) }
        val nc = FloatArray(2 * L).also { nB.asFloatBuffer().get(it) }
        for (i in 2 * L0 until 2 * L) { f0[i] = 0f; nc[i] = 0f }
        lap("f0n")
        if (cancel.get()) return null

        // ---- host: harmonic source -> STFT (folded [24][H*480]), F0/N features
        val stride = H * 480
        val har = FloatArray(24 * stride)
        dsp.stftInto(dsp.harmonicSource(f0, 2 * L0), har, stride)
        val harB = pool.floats("har", 24 * stride).also { it.asFloatBuffer().put(har) }
        val fnArr = FloatArray(4 * L)
        dsp.f0nFeaturesInto(f0, nc, 2 * L0, fnArr, L)
        val fnB = pool.floats("fn", 4 * L).also { it.asFloatBuffer().put(fnArr) }
        val m20 = maskBuf("m20", 20 * L0, H * 80)
        val mP = maskBuf("mP", 120 * L0 + 1, H * 480)
        lap("dsp")

        // ---- decoder
        val outB = pool.floats("out", 24 * stride)
        if (npu != null) {
            runNpuDecoder(npu, "L$L", H, asr, fnB, harB, m1, m2, m20, mP, refS, outB, cancel)
            ms["dec_blocks"] = (ms["dec_blocks"] ?: 0) + lastBlockMs.first
            ms["dec_blocks_tpu"] = (ms["dec_blocks_tpu"] ?: 0) + lastBlockMs.second
            if (compareWithCpu) {
                // debug: decode the same inputs on the CPU graph and record the agreement
                val cpu = decoderCpu ?: cpuGraph("dec_cpu").also { decoderCpu = it }
                val refB = pool.floats("out_cpu", 24 * stride)
                cpu.run("L$L", listOf(asr, fnB, harB, m1, m2, maskBuf("m20cpu", 20 * L0, 20 * L), mP, refS), listOf(refB))
                val a = dsp.istft(FloatArray(24 * stride).also { outB.asFloatBuffer().get(it) }, stride, 120 * L0 + 1)
                val b = dsp.istft(FloatArray(24 * stride).also { refB.asFloatBuffer().get(it) }, stride, 120 * L0 + 1)
                comparisons.add(compareAudio(b, a))
            }
        } else {
            val dec = decoderCpu ?: cpuGraph("dec_cpu").also { decoderCpu = it }
            // the 1-D graph takes flat [24][P] / [1][P] tensors: same memory as the folded ones
            dec.run("L$L", listOf(asr, fnB, harB, m1, m2, maskBuf("m20cpu", 20 * L0, 20 * L), mP, refS), listOf(outB))
        }
        if (cancel.get()) return null
        lap("decoder")

        val out = FloatArray(24 * stride).also { outB.asFloatBuffer().get(it) }
        val audio = dsp.istft(out, stride, 120 * L0 + 1)
        lap("istft")
        return Triple(audio, durs, ChunkStats(n, L0, L, ms))
    }

    private fun runStage(dec: Map<String, Graph>, name: String, sig: String, ins: List<ByteBuffer>, outs: List<ByteBuffer>) =
        dec.getValue(name).run(sig, ins, outs)

    private fun runNpuDecoder(
        dec: Map<String, Graph>, sig: String, H: Int, asr: ByteBuffer, fnB: ByteBuffer, harB: ByteBuffer,
        m1: ByteBuffer, m2: ByteBuffer, m20: ByteBuffer, mP: ByteBuffer, refS: ByteBuffer,
        outB: ByteBuffer, cancel: AtomicBoolean,
    ) {
        val L = (H - 1) * 4
        // front: encoder + 4 decode blocks, one graph each
        var h = pool.floats("fr_a", 1024 * L)
        runStage(dec, "fr0", sig, listOf(asr, fnB, m1, m2, refS), listOf(h))
        for (k in 1..3) {
            val next = pool.floats(if (k % 2 == 1) "fr_b" else "fr_a", 1024 * L)
            runStage(dec, "fr$k", sig, listOf(h, asr, fnB, m1, m2, refS), listOf(next))
            h = next
        }
        val x2 = pool.floats("x2", 512 * 2 * L)
        runStage(dec, "fr4", sig, listOf(h, asr, fnB, m1, m2, refS), listOf(x2))
        // generator stage 0 (20x rate, folded rows of 80): noise source, then upsampling + add
        val src = pool.floats("src20", 256 * H * 80)
        runStage(dec, "g0s", sig, listOf(harB, m20, refS), listOf(src))
        val x20 = pool.floats("x20", 256 * H * 80)
        runStage(dec, "g0u", sig, listOf(x2, src, m20), listOf(x20))
        val ys = (0 until 3).map { j ->
            pool.floats("y$j", 256 * H * 80).also { runStage(dec, "rb0$j", sig, listOf(x20, m20, refS), listOf(it)) }
        }
        if (cancel.get()) return
        // last (120x) rate in row blocks
        fun arr(b: ByteBuffer, n: Int) = FloatArray(n).also { b.asFloatBuffer().get(it) }
        val out = FloatArray(24 * H * 480)
        val fb = finalBlocksRunner ?: FinalBlocks(dec, pool, blockRows).also { finalBlocksRunner = it }
        val g0 = fb.graphNs; val t0 = fb.totalNs
        val tFront = System.nanoTime()
        fb.run(ys.map { arr(it, 256 * H * 80) }, arr(harB, 24 * H * 480), arr(m20, H * 80), arr(mP, H * 480), H, refS, out)
        lastBlockMs = (fb.totalNs - t0) / 1_000_000 to (fb.graphNs - g0) / 1_000_000
        outB.asFloatBuffer().put(out)
        Log.d(TAG, "final blocks L$L: ${lastBlockMs.first} ms total, ${lastBlockMs.second} ms in TPU calls (${(System.nanoTime() - tFront) / 1_000_000} ms)")
    }

    private var finalBlocksRunner: FinalBlocks? = null
    /** (total, inside TPU graph calls) ms of the last chunk's final-rate block stage. */
    @Volatile var lastBlockMs = 0L to 0L

    /** Debug: when set, NPU chunks are also decoded on the CPU; see [comparisons]. */
    @Volatile var compareWithCpu = false
    val comparisons = java.util.Collections.synchronizedList(ArrayList<DoubleArray>())

    /** [SNR dB, log-magnitude spectrogram correlation] of [x] against reference [ref]. */
    fun compare(ref: FloatArray, x: FloatArray) = compareAudio(ref, x)

    private fun compareAudio(ref: FloatArray, x: FloatArray): DoubleArray {
        val n = minOf(ref.size, x.size)
        var se = 0.0; var sr = 0.0
        for (i in 0 until n) { val d = (ref[i] - x[i]).toDouble(); se += d * d; sr += ref[i].toDouble() * ref[i] }
        val frame = 480
        val nf = n / frame
        val la = ArrayList<Double>(); val lb = ArrayList<Double>()
        for (f in 0 until nf) for (k in 1 until 120) {
            var ra = 0.0; var ia = 0.0; var rb = 0.0; var ib = 0.0
            for (t in 0 until frame) {
                val w = 0.5 - 0.5 * kotlin.math.cos(2 * Math.PI * t / frame)
                val c = kotlin.math.cos(2 * Math.PI * k * t / frame); val sn = kotlin.math.sin(2 * Math.PI * k * t / frame)
                ra += ref[f * frame + t] * w * c; ia += ref[f * frame + t] * w * sn
                rb += x[f * frame + t] * w * c; ib += x[f * frame + t] * w * sn
            }
            la.add(kotlin.math.ln(1 + kotlin.math.sqrt(ra * ra + ia * ia))); lb.add(kotlin.math.ln(1 + kotlin.math.sqrt(rb * rb + ib * ib)))
        }
        val ma = la.average(); val mb = lb.average()
        var cov = 0.0; var va = 0.0; var vb = 0.0
        for (i in la.indices) { cov += (la[i] - ma) * (lb[i] - mb); va += (la[i] - ma) * (la[i] - ma); vb += (lb[i] - mb) * (lb[i] - mb) }
        return doubleArrayOf(10 * kotlin.math.log10(sr / maxOf(se, 1e-20)), cov / kotlin.math.sqrt(va * vb))
    }

    private fun maskBuf(key: String, valid: Int, len: Int): ByteBuffer = pool.floats(key, len).also { b ->
        val fb = b.asFloatBuffer()
        for (i in 0 until len) fb.put(if (i < valid) 1f else 0f)
    }

    companion object {
        private const val TAG = "KokoroEngine"
        const val THREADS = 6 // X3 + 4x A715 + 1x A510: 6 measured fastest (RTF 1.32 vs 1.57 with 4)
        /** Conservative frames per phoneme at speed 1, for sizing chunks to the frame buckets. */
        const val FRAMES_PER_PHONEME = 2.6f

        @Volatile private var instance: KokoroEngine? = null

        fun get(context: Context): KokoroEngine = instance ?: synchronized(this) {
            instance ?: KokoroEngine(context.applicationContext).also { instance = it }
        }

        private fun floats(o: JSONObject, key: String): FloatArray =
            o.getJSONArray(key).let { a -> FloatArray(a.length()) { a.getDouble(it).toFloat() } }
    }
}
