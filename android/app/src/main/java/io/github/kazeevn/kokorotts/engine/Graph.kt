package io.github.kazeevn.kokorotts.engine

import android.content.Context
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import com.google.ai.edge.litert.TensorType
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Where a graph runs. NPU = the Tensor G3 EdgeTPU: graphs AOT-compiled with the Google Tensor SDK
 * (assets/kokoro/npu, see convert/aot_compile_g3.py), run through the Google Tensor dispatch library.
 */
enum class Backend { CPU, NPU }

/**
 * A multi-signature LiteRT model (one signature per length bucket) on one backend, via LiteRT's
 * CompiledModel API. Inputs/outputs are named args_<i> / output_<i> by litert-torch; callers
 * pass them in order.
 */
class Graph(
    context: Context,
    val name: String,
    val backend: Backend,
    threads: Int,
    /**
     * CPU backend: XNNPACK weight-cache file. Without it XNNPACK packs shared weights once per op
     * and per signature subgraph (GBs for the unrolled LSTMs and bucketed models); with it each
     * distinct weight buffer is packed once, and the packed weights are reused on later loads.
     */
    xnnCacheFile: java.io.File? = null,
    /** CPU backend: XNNPACK flags (e.g. [XNN_FLAG_FORCE_FP16]). */
    xnnFlags: Int = 0,
) : AutoCloseable {
    private val model: CompiledModel
    val initMs: Long

    init {
        val t0 = System.nanoTime()
        val opts = if (backend == Backend.CPU) {
            CompiledModel.Options(Accelerator.CPU).apply {
                cpuOptions = CompiledModel.CpuOptions(threads, xnnFlags, xnnCacheFile!!.absolutePath)
            }
        } else CompiledModel.Options(Accelerator.NPU)
        val path = if (backend == Backend.NPU) "kokoro/npu/$name.tflite" else "kokoro/$name.tflite"
        model = CompiledModel.create(context.assets, path, opts, environment(context))
        initMs = (System.nanoTime() - t0) / 1_000_000
        Log.i(TAG, "$name on $backend ready in $initMs ms, process RSS ${rssMb()} MB")
    }

    /** Per-signature tensor buffers and the host arrays staged into / out of them. */
    /**
     * Per-signature tensor buffers. Each buffer is copied directly from/to the caller's direct
     * ByteBuffer when its native handle checks out ([DirectBuffers]); otherwise through a
     * staging array and the TensorBuffer array API.
     */
    private class Io(val ins: Map<String, TensorBuffer>, val outs: Map<String, TensorBuffer>,
                     val inDirect: LongArray, val outDirect: LongArray,
                     val inArrays: List<Any?>, val outArrays: List<FloatArray?>)

    private val io = HashMap<String, Io>()

    private fun io(sig: String, inputs: List<ByteBuffer>, outputs: List<ByteBuffer>): Io = io.getOrPut(sig) {
        val inNames = List(inputs.size) { "args_$it" }
        val outNames = List(outputs.size) { "output_$it" }
        val ins = inNames.associateWith { model.createInputBuffer(it, sig) }
        val outs = outNames.associateWith { model.createOutputBuffer(it, sig) }
        val inDirect = LongArray(inputs.size) { DirectBuffers.handle(ins.getValue(inNames[it]), inputs[it].capacity()) }
        val outDirect = LongArray(outputs.size) { DirectBuffers.handle(outs.getValue(outNames[it]), outputs[it].capacity()) }
        val inArrays = inNames.mapIndexed { i, n ->
            if (inDirect[i] != 0L) return@mapIndexed null
            val words = inputs[i].capacity() / 4
            when (val t = model.getInputTensorType(n, sig).elementType) {
                TensorType.ElementType.FLOAT -> FloatArray(words)
                TensorType.ElementType.INT -> IntArray(words)
                else -> error("$name/$sig/$n: unsupported input type $t")
            }
        }
        val outArrays = outputs.mapIndexed { i, o -> if (outDirect[i] != 0L) null else FloatArray(o.capacity() / 4) }
        if (inDirect.any { it == 0L } || outDirect.any { it == 0L })
            Log.w(TAG, "$name/$sig: direct buffer copies unavailable for some tensors")
        Io(ins, outs, inDirect, outDirect, inArrays, outArrays)
    }

    /** Runs signature [sig]; [inputs] and [outputs] are direct native-order buffers, in order. */
    fun run(sig: String, inputs: List<ByteBuffer>, outputs: List<ByteBuffer>) {
        val b = io(sig, inputs, outputs)
        require(b.inDirect.size == inputs.size && b.outDirect.size == outputs.size) { "$name/$sig: arity mismatch" }
        for (i in inputs.indices) {
            val buf = b.ins.getValue("args_$i")
            if (b.inDirect[i] != 0L) {
                check(DirectBuffers.copy(b.inDirect[i], inputs[i], toTensor = true)) { "$name/$sig: write args_$i failed" }
                continue
            }
            val src = inputs[i].duplicate().order(ByteOrder.nativeOrder()).also { it.rewind() }
            when (val a = b.inArrays[i]) {
                is FloatArray -> { src.asFloatBuffer().get(a); buf.writeFloat(a) }
                is IntArray -> { src.asIntBuffer().get(a); buf.writeInt(a) }
            }
        }
        model.run(b.ins, b.outs, sig)
        for (i in outputs.indices) {
            outputs[i].rewind()
            if (b.outDirect[i] != 0L) {
                check(DirectBuffers.copy(b.outDirect[i], outputs[i], toTensor = false)) { "$name/$sig: read output_$i failed" }
            } else {
                val a = b.outs.getValue("output_$i").readFloat()
                outputs[i].asFloatBuffer().put(a, 0, minOf(a.size, outputs[i].capacity() / 4))
            }
            if (checkOutputs) check(sig, i, inputs, outputs[i])
        }
    }

    /** Debug: logs non-finite outputs; with [dumpDir], dumps the first call of each signature. */
    private fun check(sig: String, i: Int, inputs: List<ByteBuffer>, output: ByteBuffer) {
        fun floats(b: ByteBuffer) = FloatArray(b.capacity() / 4).also {
            b.duplicate().order(ByteOrder.nativeOrder()).also { d -> d.rewind() }.asFloatBuffer().get(it)
        }
        val a = floats(output)
        val bad = a.count { !it.isFinite() }
        val maxAbs = a.maxOfOrNull { kotlin.math.abs(it) } ?: 0f
        if (bad > 0 || reported.add("$name/$sig/$i"))
            Log.w(TAG, "check $name/$sig output_$i: ${a.size} values, $bad non-finite, max |x| $maxAbs")
        val dir = dumpDir ?: return
        if (!dumped.add("$name/$sig/$i")) return
        // raw bytes are native order = little-endian on arm64
        fun bytes(b: ByteBuffer) = ByteArray(b.capacity()).also { b.duplicate().also { d -> d.rewind() }.get(it) }
        if (i == 0) inputs.forEachIndexed { k, x -> java.io.File(dir, "$name.$sig.in$k.bin").writeBytes(bytes(x)) }
        java.io.File(dir, "$name.$sig.out$i.bin").writeBytes(bytes(output))
    }

    override fun close() {
        io.values.forEach { b -> b.ins.values.forEach { it.close() }; b.outs.values.forEach { it.close() } }
        io.clear()
        model.close()
    }

    companion object {
        private const val TAG = "KokoroGraph"

        /** TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16 */
        const val XNN_FLAG_FORCE_FP16 = 0x4

        /** Debug: log non-finite outputs (and each graph/signature/output's range once). */
        @Volatile var checkOutputs = false
        /** Debug: with [checkOutputs], dump the first call of each graph/signature here. */
        @Volatile var dumpDir: java.io.File? = null
        private val dumped = java.util.Collections.synchronizedSet(HashSet<String>())
        private val reported = java.util.Collections.synchronizedSet(HashSet<String>())

        @Volatile private var env: Environment? = null

        /** One LiteRT environment per process; the NPU dispatch library ships in the APK. */
        fun environment(context: Context): Environment = env ?: synchronized(this) {
            env ?: Environment.create(mapOf(
                Environment.Option.DispatchLibraryDir to context.applicationInfo.nativeLibraryDir,
            )).also { env = it }
        }

        fun rssMb(): Long = java.io.File("/proc/self/status").readLines()
            .firstOrNull { it.startsWith("VmRSS:") }?.split(Regex("\\s+"))?.get(1)?.toLong()?.div(1024) ?: -1
    }
}

/**
 * Reusable direct buffers keyed by role, grown on demand. Returns exact-size slices (the graph
 * wrapper sizes its staging arrays from a buffer's capacity).
 */
class BufferPool {
    private val map = HashMap<String, ByteBuffer>()

    fun floats(key: String, n: Int): ByteBuffer = get(key, n * 4)

    fun get(key: String, bytes: Int): ByteBuffer {
        var b = map[key]
        if (b == null || b.capacity() < bytes) {
            b = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder())
            map[key] = b
        }
        val dup = b!!.duplicate()
        dup.clear().limit(bytes)
        return dup.slice().order(ByteOrder.nativeOrder())
    }

    fun clear() = map.clear()
}
