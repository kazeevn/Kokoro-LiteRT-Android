package io.github.kazeevn.kokorotts.engine

import android.util.Log
import com.google.ai.edge.litert.TensorBuffer
import java.nio.ByteBuffer

/**
 * One-memcpy copies between direct ByteBuffers and LiteRT tensor buffers (cpp/litert_buffers.cpp).
 * The Kotlin TensorBuffer API only reads and writes Java arrays.
 */
internal object DirectBuffers {
    init { System.loadLibrary("kokoro_native") }

    /** JniHandle's native handle getter (internal in the LiteRT Kotlin API). */
    private val handleGetter = runCatching {
        com.google.ai.edge.litert.JniHandle::class.java.methods
            .first { it.name.startsWith("getHandle") && it.parameterCount == 0 && it.returnType == Long::class.javaPrimitiveType }
    }.onFailure { Log.w("KokoroBuf", "no TensorBuffer handle getter", it) }.getOrNull()

    /**
     * The LiteRtTensorBuffer behind [b] if it checks out as one holding exactly [bytes] packed
     * bytes, else 0 (callers then fall back to the array API).
     */
    fun handle(b: TensorBuffer, bytes: Int): Long {
        val jni = handleGetter?.let { runCatching { it.invoke(b) as Long }.getOrNull() } ?: return 0
        val h = nativeCHandle(jni)
        val n = nativePackedSize(h)
        if (n != bytes.toLong() && warned++ < 3) Log.w("KokoroBuf", "tensor buffer check failed: packed $n, expected $bytes")
        return if (n == bytes.toLong()) h else 0
    }

    private var warned = 0

    fun copy(handle: Long, buf: ByteBuffer, toTensor: Boolean): Boolean =
        nativeCopy(handle, buf, buf.capacity(), toTensor)

    @JvmStatic private external fun nativeCHandle(jniHandle: Long): Long
    @JvmStatic private external fun nativePackedSize(tb: Long): Long
    @JvmStatic private external fun nativeCopy(tb: Long, buf: ByteBuffer, bytes: Int, toTensor: Boolean): Boolean
}
