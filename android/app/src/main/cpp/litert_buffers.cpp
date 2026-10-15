// Copies between Java direct ByteBuffers and LiteRT tensor buffers with one memcpy, through the
// LiteRT C API (LiteRtLockTensorBuffer). The Kotlin TensorBuffer API only reads/writes Java
// arrays: for the EdgeTPU block graphs (~4 MB per call, thousands of calls per minute of speech)
// the extra array copies and allocations cost more than the TPU itself.
#include <jni.h>
#include <dlfcn.h>
#include <stddef.h>
#include <string.h>
#include <android/log.h>

#define LOG_TAG "KokoroBuf"

// litert/c/litert_tensor_buffer.h, litert_common.h (LiteRT 2.3.0)
typedef int (*LockFn)(void* tensor_buffer, void** host_mem_addr, int lock_mode);
typedef int (*UnlockFn)(void* tensor_buffer);
typedef int (*PackedSizeFn)(void* tensor_buffer, size_t* size);
static const int kLockModeRead = 0;
static const int kLockModeWrite = 1;

static LockFn lockFn;
static UnlockFn unlockFn;
static PackedSizeFn packedSizeFn;

static bool init() {
    if (lockFn) return true;
    void* h = dlopen("libLiteRt.so", RTLD_NOW | RTLD_NOLOAD);
    if (!h) h = dlopen("libLiteRt.so", RTLD_NOW);
    if (!h) { __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "dlopen: %s", dlerror()); return false; }
    packedSizeFn = (PackedSizeFn)dlsym(h, "LiteRtGetTensorBufferPackedSize");
    unlockFn = (UnlockFn)dlsym(h, "LiteRtUnlockTensorBuffer");
    lockFn = (LockFn)dlsym(h, "LiteRtLockTensorBuffer");
    return lockFn && unlockFn && packedSizeFn;
}

/**
 * The LiteRtTensorBuffer behind the Kotlin TensorBuffer's JNI handle: that handle points to the
 * C++ litert::TensorBuffer, whose first member is the C handle (LiteRT 2.3.0; the Kotlin side
 * only uses the result after checking its packed size against the expected tensor size).
 */
extern "C" JNIEXPORT jlong JNICALL
Java_io_github_kazeevn_kokorotts_engine_DirectBuffers_nativeCHandle(JNIEnv*, jclass, jlong jniHandle) {
    return jniHandle ? (jlong)(*(void**)jniHandle) : 0;
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_github_kazeevn_kokorotts_engine_DirectBuffers_nativePackedSize(JNIEnv*, jclass, jlong tb) {
    if (!init() || !tb) return -1;
    size_t n = 0;
    return packedSizeFn((void*)tb, &n) == 0 ? (jlong)n : -1;
}

/** Copies [bytes] between [buf] (direct) and tensor buffer [tb]; returns false on any failure. */
extern "C" JNIEXPORT jboolean JNICALL
Java_io_github_kazeevn_kokorotts_engine_DirectBuffers_nativeCopy(JNIEnv* env, jclass, jlong tb, jobject buf,
                                                                  jint bytes, jboolean toTensor) {
    if (!init()) return JNI_FALSE;
    void* host = env->GetDirectBufferAddress(buf);
    if (!host || env->GetDirectBufferCapacity(buf) < bytes) return JNI_FALSE;
    void* mem = nullptr;
    if (lockFn((void*)tb, &mem, toTensor ? kLockModeWrite : kLockModeRead) != 0 || !mem) return JNI_FALSE;
    if (toTensor) memcpy(mem, host, (size_t)bytes); else memcpy(host, mem, (size_t)bytes);
    return unlockFn((void*)tb) == 0 ? JNI_TRUE : JNI_FALSE;
}
