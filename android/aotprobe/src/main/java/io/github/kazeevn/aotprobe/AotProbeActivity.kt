package io.github.kazeevn.aotprobe

import android.app.Activity
import android.os.Bundle
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.concurrent.thread

/**
 * am start -n io.github.kazeevn.aotprobe/.AotProbeActivity --es model x.tflite [--es accel npu|cpu]
 *   [--ei iters 5] [--es sig name] [--ez dumpout true]
 * Model and optional inputs (<model>.in<i>.bin, float32) live in files/probe; the report goes
 * to external files/results/<model>.<accel>.txt, outputs to <model>.out<i>.<accel>.bin.
 */
class AotProbeActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val model = File(filesDir, "probe/" + (intent.getStringExtra("model") ?: "probe.tflite"))
        val accel = intent.getStringExtra("accel") ?: "npu"
        val iters = intent.getIntExtra("iters", 5)
        val sig = intent.getStringExtra("sig")
        thread {
            val out = StringBuilder()
            fun log(s: String) { Log.i("AotProbe", s); out.appendLine(s) }
            try {
                val env = Environment.create(mapOf(
                    Environment.Option.DispatchLibraryDir to applicationInfo.nativeLibraryDir,
                    Environment.Option.CompilerPluginLibraryDir to applicationInfo.nativeLibraryDir,
                ))
                log("available accelerators: ${env.getAvailableAccelerators()}")
                val acc = if (accel == "cpu") Accelerator.CPU else Accelerator.NPU
                val t0 = System.nanoTime()
                val cm = CompiledModel.create(model.path, CompiledModel.Options(acc), env)
                log("model=${model.name} accel=$accel init_ms=${(System.nanoTime() - t0) / 1_000_000}")
                val ins = if (sig != null) cm.createInputBuffers(sig) else cm.createInputBuffers(0)
                val outs = if (sig != null) cm.createOutputBuffers(sig) else cm.createOutputBuffers(0)
                ins.forEachIndexed { i, b ->
                    val f = File(model.path + ".in$i.bin")
                    if (f.exists()) {
                        val bb = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                        b.writeFloat(FloatArray(bb.remaining()).also { bb.get(it) })
                    }
                }
                val times = LongArray(iters)
                for (k in 0 until iters) {
                    val s = System.nanoTime()
                    if (sig != null) cm.run(ins, outs, sig) else cm.run(ins, outs, 0)
                    times[k] = (System.nanoTime() - s) / 1000
                }
                log("run_us first=${times[0]} median=${times.sorted()[iters / 2]} min=${times.min()}")
                if (intent.getBooleanExtra("dumpout", false)) {
                    outs.forEachIndexed { i, b ->
                        val a = b.readFloat()
                        val bb = ByteBuffer.allocate(a.size * 4).order(ByteOrder.LITTLE_ENDIAN)
                        bb.asFloatBuffer().put(a)
                        File(getExternalFilesDir("results"), "${model.name}.out$i.$accel.bin").writeBytes(bb.array())
                    }
                    log("dumped ${outs.size} outputs")
                }
                ins.forEach { it.close() }; outs.forEach { it.close() }; cm.close(); env.close()
            } catch (e: Throwable) { log("ERROR ${e.javaClass.simpleName}: ${e.message}"); Log.e("AotProbe", "fail", e) }
            File(getExternalFilesDir("results"), "${model.name}.$accel.txt").writeText(out.toString())
            runOnUiThread { finish() }
        }
    }
}
