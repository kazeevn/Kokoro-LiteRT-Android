// Creates LiteRT's XNNPACK delegate with a weight cache. The Java API only offers the default
// delegate, which packs weights once per op and per signature subgraph: with unrolled LSTMs and
// multi-signature (bucketed) models that multiplies memory by hundreds. The weight cache packs
// each distinct weight buffer once and persists the packed weights to a file.
#include <jni.h>
#include <dlfcn.h>
#include <stdint.h>
#include <android/log.h>

#define LOG_TAG "KokoroXnn"

extern "C" {
// Prefix of the layout in LiteRT tflite/delegates/xnnpack/xnnpack_delegate.h. Newer builds append
// fields (e.g. weight-cache memory locking), so keep generous padding: the struct is returned by
// value from TfLiteXNNPackDelegateOptionsDefault() and must fit whatever the library writes.
typedef struct {
    int32_t num_threads;
    uint32_t runtime_flags;
    uint32_t flags;
    void* weights_cache;
    bool handle_variable_ops;
    const char* weight_cache_file_path;
    int weight_cache_file_descriptor;
    void* weight_cache_provider;
    char reserved[512];
} XnnOptions;
}

typedef XnnOptions (*OptionsDefaultFn)();
typedef void* (*CreateFn)(const XnnOptions*);
typedef void (*DeleteFn)(void*);

static void* lib() {
    static void* h = dlopen("libtensorflowlite_jni.so", RTLD_NOW | RTLD_NOLOAD);
    if (!h) h = dlopen("libtensorflowlite_jni.so", RTLD_NOW);
    return h;
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_github_kazeevn_kokorotts_engine_XnnpackDelegate_nativeCreate(JNIEnv* env, jclass, jint threads,
                                                                      jstring jcache, jint flags) {
    void* h = lib();
    if (!h) { __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "dlopen: %s", dlerror()); return 0; }
    auto def = (OptionsDefaultFn)dlsym(h, "TfLiteXNNPackDelegateOptionsDefault");
    auto create = (CreateFn)dlsym(h, "TfLiteXNNPackDelegateCreate");
    if (!def || !create) { __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "dlsym failed"); return 0; }
    XnnOptions o = def();
    o.num_threads = threads;
    o.flags |= (uint32_t)flags;
    const char* path = jcache ? env->GetStringUTFChars(jcache, nullptr) : nullptr;
    o.weight_cache_file_path = path;  // copied by the delegate
    void* d = create(&o);
    if (path) env->ReleaseStringUTFChars(jcache, path);
    return (jlong)d;
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_kazeevn_kokorotts_engine_XnnpackDelegate_nativeDelete(JNIEnv*, jclass, jlong d) {
    auto del = (DeleteFn)dlsym(lib(), "TfLiteXNNPackDelegateDelete");
    if (del && d) del((void*)d);
}
