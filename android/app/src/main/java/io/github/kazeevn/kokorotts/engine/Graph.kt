package io.github.kazeevn.kokorotts.engine

import android.content.res.AssetManager
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/** Where a graph runs. NPU = the Tensor G3 EdgeTPU, reached through NNAPI ("google-edgetpu"). */
enum class Backend { CPU, NPU }

/**
 * A multi-signature LiteRT model (one signature per length bucket) on one backend.
 * Inputs/outputs are named args_<i> / output_<i> by litert-torch; callers pass them in order.
 */
class Graph(
    assets: AssetManager,
    val name: String,
    val backend: Backend,
    threads: Int,
    /** CPU backend: XNNPACK weight-cache file (packed weights, reused on later loads). */
    xnnCacheFile: java.io.File? = null,
    /** CPU backend: XNNPACK delegate flags (e.g. [XnnpackDelegate.FLAG_FORCE_FP16]). */
    xnnFlags: Int = 0,
) : AutoCloseable {
    private val model: MappedByteBuffer = mapAsset(assets, "kokoro/$name.tflite")
    private val delegate: org.tensorflow.lite.Delegate?
    private val interpreter: Interpreter
    val initMs: Long

    init {
        val t0 = System.nanoTime()
        val opts = Interpreter.Options().setNumThreads(threads).setUseXNNPACK(false)
        delegate = if (backend == Backend.CPU) {
            // Our XNNPACK delegate with a weight cache: the default one packs every weight once
            // per op and per signature subgraph (GBs for the unrolled LSTMs and bucketed models).
            // (One delegate per interpreter: a finalized file-backed cache cannot be shared.)
            XnnpackDelegate(threads, xnnCacheFile!!.absolutePath, xnnFlags).also { opts.addDelegate(it) }
        } else if (backend == Backend.NPU) {
            NnApiDelegate(
                NnApiDelegate.Options()
                    .setAcceleratorName("google-edgetpu")
                    .setUseNnapiCpu(false)
                    .setAllowFp16(true)
                    .setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED)
            ).also { opts.addDelegate(it) }
        } else null
        interpreter = Interpreter(model, opts)
        initMs = (System.nanoTime() - t0) / 1_000_000
        Log.i(TAG, "$name on $backend ready in $initMs ms, process RSS ${rssMb()} MB")
    }

    private val inputNames = HashMap<String, List<String>>()
    private val outputNames = HashMap<String, List<String>>()

    private fun ordered(names: Array<String>, prefix: String) =
        names.sortedBy { it.removePrefix(prefix).toInt() }

    /** Runs signature [sig]; [inputs] and [outputs] are direct native-order buffers, in order. */
    fun run(sig: String, inputs: List<ByteBuffer>, outputs: List<ByteBuffer>) {
        val ins = inputNames.getOrPut(sig) { ordered(interpreter.getSignatureInputs(sig), "args_") }
        val outs = outputNames.getOrPut(sig) { ordered(interpreter.getSignatureOutputs(sig), "output_") }
        require(ins.size == inputs.size && outs.size == outputs.size) { "$name/$sig: arity mismatch" }
        val inMap = HashMap<String, Any>(ins.size * 2)
        for (i in ins.indices) inMap[ins[i]] = inputs[i].rewind()
        val outMap = HashMap<String, Any>(outs.size * 2)
        for (i in outs.indices) outMap[outs[i]] = outputs[i].rewind()
        interpreter.runSignature(inMap, outMap, sig)
        outputs.forEach { it.rewind() }
    }

    override fun close() {
        interpreter.close()
        delegate?.close()
    }

    companion object {
        private const val TAG = "KokoroGraph"

        fun rssMb(): Long = java.io.File("/proc/self/status").readLines()
            .firstOrNull { it.startsWith("VmRSS:") }?.split(Regex("\\s+"))?.get(1)?.toLong()?.div(1024) ?: -1

        fun mapAsset(assets: AssetManager, path: String): MappedByteBuffer {
            assets.openFd(path).use { fd ->
                FileInputStream(fd.fileDescriptor).channel.use { ch ->
                    return ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
                }
            }
        }
    }
}

/**
 * Reusable direct buffers keyed by role, grown on demand. Returns exact-size slices: LiteRT
 * compares a buffer's capacity (not its limit) with the tensor size.
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
