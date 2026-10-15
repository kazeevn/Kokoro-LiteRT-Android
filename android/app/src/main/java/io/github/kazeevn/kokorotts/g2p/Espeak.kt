package io.github.kazeevn.kokorotts.g2p

/** Thin wrapper over the native espeak-ng phonemizer. */
object Espeak {
    init { System.loadLibrary("kokoro_native") }

    /** espeak_TextToPhonemes mode: IPA (0x02), tie multi-letter phonemes (0x80) with '^' (bits 8+),
     *  matching phonemizer's tie='^' as used by misaki's EspeakFallback. */
    private const val MODE_IPA = 0x02 or 0x80 or ('^'.code shl 8)

    @Volatile private var ready = false

    /** [dataDir] must contain an `espeak-ng-data` directory. */
    fun init(dataDir: String): Boolean {
        if (!ready) ready = nativeInit(dataDir)
        return ready
    }

    fun phonemize(text: String, voice: String): String? =
        if (ready) nativePhonemize(text, voice, MODE_IPA) else null

    @JvmStatic private external fun nativeInit(dataDir: String): Boolean
    @JvmStatic private external fun nativePhonemize(text: String, voice: String, mode: Int): String?
}
