// JNI bridge exposing espeak-ng's text -> IPA phonemizer (used as misaki's OOV fallback).
#include <jni.h>
#include <mutex>
#include <string>
#include <android/log.h>
#include <espeak-ng/speak_lib.h>

#define LOG_TAG "KokoroEspeak"

static std::mutex g_lock;
static bool g_ready = false;

extern "C" JNIEXPORT jboolean JNICALL
Java_io_github_kazeevn_kokorotts_g2p_Espeak_nativeInit(JNIEnv *env, jclass, jstring jdataDir) {
    std::lock_guard<std::mutex> lk(g_lock);
    if (g_ready) return JNI_TRUE;
    const char *dir = env->GetStringUTFChars(jdataDir, nullptr);
    int rate = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, dir, espeakINITIALIZE_DONT_EXIT);
    env->ReleaseStringUTFChars(jdataDir, dir);
    if (rate <= 0) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "espeak_Initialize failed: %d", rate);
        return JNI_FALSE;
    }
    g_ready = true;
    return JNI_TRUE;
}

// Returns IPA for `text` in `voice` (e.g. "en-us", "en-gb"); clauses are joined with a space.
// `mode` is the espeak_TextToPhonemes phonememode (bit 1 = IPA, bits 8.. = tie character).
extern "C" JNIEXPORT jstring JNICALL
Java_io_github_kazeevn_kokorotts_g2p_Espeak_nativePhonemize(JNIEnv *env, jclass, jstring jtext,
                                                           jstring jvoice, jint mode) {
    std::lock_guard<std::mutex> lk(g_lock);
    if (!g_ready) return nullptr;
    const char *voice = env->GetStringUTFChars(jvoice, nullptr);
    espeak_ERROR st = espeak_SetVoiceByName(voice);
    env->ReleaseStringUTFChars(jvoice, voice);
    if (st != EE_OK) return nullptr;

    const char *text = env->GetStringUTFChars(jtext, nullptr);
    std::string input(text);
    env->ReleaseStringUTFChars(jtext, text);

    std::string out;
    const void *ptr = input.c_str();
    while (ptr != nullptr) {
        const char *ph = espeak_TextToPhonemes(&ptr, espeakCHARS_UTF8, mode);
        if (ph != nullptr && *ph) {
            if (!out.empty()) out.push_back(' ');
            out.append(ph);
        }
    }
    return env->NewStringUTF(out.c_str());
}
