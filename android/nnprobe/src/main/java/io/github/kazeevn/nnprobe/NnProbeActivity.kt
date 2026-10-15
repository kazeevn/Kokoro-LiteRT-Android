package io.github.kazeevn.nnprobe

import android.app.Activity
import android.os.Bundle
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.concurrent.thread

class NnProbeActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val model = File(filesDir, "probe/" + (intent.getStringExtra("model") ?: "probe.tflite"))
        val accel = intent.getStringExtra("accel") ?: "google-edgetpu"
        val iters = intent.getIntExtra("iters", 5)
        thread {
            val out = StringBuilder()
            fun log(s: String) { Log.i("NnProbe", s); out.appendLine(s) }
            try {
                val opts = Interpreter.Options()
                var nn: NnApiDelegate? = null
                if (accel != "cpu") {
                    nn = NnApiDelegate(NnApiDelegate.Options().setAcceleratorName(accel)
                        .setUseNnapiCpu(false).setAllowFp16(intent.getBooleanExtra("fp16", true))
                        .setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED)
                        .apply {
                            if (intent.getBooleanExtra("cache", false)) {
                                setCacheDir(File(cacheDir, "nnapi").apply { mkdirs() }.absolutePath)
                                setModelToken(model.name + model.length())
                            }
                        })
                    opts.addDelegate(nn)
                } else opts.setNumThreads(4)
                val t0 = System.nanoTime()
                val it = Interpreter(model, opts)
                log("model=${model.name} accel=$accel init_ms=${(System.nanoTime() - t0) / 1_000_000}")
                val sig = intent.getStringExtra("sig")
                val times = LongArray(iters)
                if (sig == null) {
                    // inputs from <model>.in<i>.bin when present (raw native-order tensors)
                    val ins = Array<Any>(it.inputTensorCount) { i ->
                        val b = ByteBuffer.allocateDirect(it.getInputTensor(i).numBytes()).order(ByteOrder.nativeOrder())
                        val f = File(model.path + ".in$i.bin")
                        if (f.exists()) { b.put(f.readBytes()); b.rewind() }
                        b
                    }
                    val outs = HashMap<Int, Any>()
                    for (i in 0 until it.outputTensorCount) outs[i] = ByteBuffer.allocateDirect(it.getOutputTensor(i).numBytes()).order(ByteOrder.nativeOrder())
                    for (k in 0 until iters) {
                        ins.forEach { (it as ByteBuffer).rewind() }
                        outs.values.forEach { (it as ByteBuffer).rewind() }
                        val s = System.nanoTime(); it.runForMultipleInputsOutputs(ins, outs); times[k] = (System.nanoTime() - s) / 1000
                    }
                } else {
                    log("signatures=${it.signatureKeys.toList()}")
                    val ins = it.getSignatureInputs(sig).associateWith { n ->
                        ByteBuffer.allocateDirect(it.getInputTensorFromSignature(n, sig).numBytes()).order(ByteOrder.nativeOrder()) as Any
                    }
                    val outs = it.getSignatureOutputs(sig).associateWith { n ->
                        ByteBuffer.allocateDirect(it.getOutputTensorFromSignature(n, sig).numBytes()).order(ByteOrder.nativeOrder()) as Any
                    }
                    for (k in 0 until iters) {
                        ins.values.forEach { (it as ByteBuffer).rewind() }
                        outs.values.forEach { (it as ByteBuffer).rewind() }
                        val s = System.nanoTime(); it.runSignature(ins, outs, sig); times[k] = (System.nanoTime() - s) / 1000
                    }
                }
                log("run_us first=${times[0]} median=${times.sorted()[iters / 2]} min=${times.min()}")
                if (sig == null && intent.getBooleanExtra("dumpout", false)) {
                    val outs = HashMap<Int, Any>()
                    for (i in 0 until it.outputTensorCount) outs[i] = ByteBuffer.allocateDirect(it.getOutputTensor(i).numBytes()).order(ByteOrder.nativeOrder())
                    val ins = Array<Any>(it.inputTensorCount) { i ->
                        val b = ByteBuffer.allocateDirect(it.getInputTensor(i).numBytes()).order(ByteOrder.nativeOrder())
                        val f = File(model.path + ".in$i.bin"); if (f.exists()) { b.put(f.readBytes()); b.rewind() }; b
                    }
                    it.runForMultipleInputsOutputs(ins, outs)
                    for ((i, o) in outs) File(getExternalFilesDir("results"), "${model.name}.out$i.$accel.bin")
                        .writeBytes(ByteArray((o as ByteBuffer).rewind().remaining()).also { a -> o.get(a) })
                    log("dumped outputs")
                }
                it.close(); nn?.close()
            } catch (e: Throwable) { log("ERROR ${e.javaClass.simpleName}: ${e.message}"); Log.e("NnProbe", "fail", e) }
            File(getExternalFilesDir("results"), "${model.name}.nnapi-$accel.txt").writeText(out.toString())
            runOnUiThread { finish() }
        }
    }
}
