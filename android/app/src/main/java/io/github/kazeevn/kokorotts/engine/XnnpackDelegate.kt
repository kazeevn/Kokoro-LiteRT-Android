package io.github.kazeevn.kokorotts.engine

import org.tensorflow.lite.Delegate

/**
 * LiteRT's XNNPACK delegate with a weight cache file (see cpp/xnnpack_jni.cpp): each distinct
 * weight buffer is packed once instead of once per op and per signature subgraph.
 * The cache file is reused on later loads (keep it per model + app version).
 */
class XnnpackDelegate(threads: Int, cacheFile: String, flags: Int = 0) : Delegate {
    private var handle: Long = nativeCreate(threads, cacheFile, flags).also {
        check(it != 0L) { "failed to create XNNPACK delegate" }
    }

    override fun getNativeHandle(): Long = handle

    override fun close() {
        if (handle != 0L) nativeDelete(handle)
        handle = 0
    }

    companion object {
        init { System.loadLibrary("kokoro_native") }

        /** TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16 */
        const val FLAG_FORCE_FP16 = 0x4

        @JvmStatic private external fun nativeCreate(threads: Int, cacheFile: String?, flags: Int): Long
        @JvmStatic private external fun nativeDelete(handle: Long)
    }
}
