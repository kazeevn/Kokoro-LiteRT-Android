package io.github.kazeevn.kokorotts

import android.media.AudioFormat
import android.speech.tts.SynthesisCallback
import android.speech.tts.SynthesisRequest
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeechService
import android.speech.tts.Voice
import android.util.Log
import io.github.kazeevn.kokorotts.engine.KokoroEngine
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean

/** Android system TTS engine backed by Kokoro-82M (English voices only). */
class KokoroTtsService : TextToSpeechService() {
    private lateinit var engine: KokoroEngine
    private val cancel = AtomicBoolean(false)
    @Volatile private var currentVoice: String? = null

    override fun onCreate() {
        // TextToSpeechService.onCreate() calls onLoadLanguage(), which needs the engine already
        engine = KokoroEngine.get(this)
        Settings.apply(this, engine)
        super.onCreate()
    }

    // ------------------------------------------------------------- languages

    override fun onIsLanguageAvailable(lang: String?, country: String?, variant: String?): Int {
        if (!isEnglish(lang)) return TextToSpeech.LANG_NOT_SUPPORTED
        return when (country?.uppercase()) {
            null, "" -> TextToSpeech.LANG_AVAILABLE
            "USA", "US", "GBR", "GB" -> TextToSpeech.LANG_COUNTRY_AVAILABLE
            else -> TextToSpeech.LANG_AVAILABLE
        }
    }

    override fun onGetLanguage(): Array<String> {
        val v = currentVoice ?: Settings.defaultVoice(this)
        return if (engine.isBritish(v)) arrayOf("eng", "GBR", "") else arrayOf("eng", "USA", "")
    }

    override fun onLoadLanguage(lang: String?, country: String?, variant: String?): Int {
        val r = onIsLanguageAvailable(lang, country, variant)
        if (r >= TextToSpeech.LANG_AVAILABLE) currentVoice = onGetDefaultVoiceNameFor(lang, country, variant)
        return r
    }

    // ------------------------------------------------------------- voices

    override fun onGetVoices(): List<Voice> = engine.voices.map { toVoice(it) }

    override fun onIsValidVoiceName(voiceName: String?): Int =
        if (voiceName != null && voiceName in engine.voices) TextToSpeech.SUCCESS else TextToSpeech.ERROR

    override fun onLoadVoice(voiceName: String?): Int {
        if (onIsValidVoiceName(voiceName) != TextToSpeech.SUCCESS) return TextToSpeech.ERROR
        currentVoice = voiceName
        return TextToSpeech.SUCCESS
    }

    override fun onGetDefaultVoiceNameFor(lang: String?, country: String?, variant: String?): String {
        val def = Settings.defaultVoice(this)
        val gb = country?.uppercase() in setOf("GBR", "GB")
        return if (gb == engine.isBritish(def)) def else if (gb) "bf_emma" else "af_heart"
    }

    private fun toVoice(name: String) = Voice(
        name,
        if (engine.isBritish(name)) Locale.UK else Locale.US,
        Voice.QUALITY_VERY_HIGH,
        Voice.LATENCY_NORMAL,
        false,
        emptySet(),
    )

    // ------------------------------------------------------------- synthesis

    override fun onStop() {
        cancel.set(true)
    }

    override fun onSynthesizeText(request: SynthesisRequest, callback: SynthesisCallback) {
        cancel.set(false)
        val text = request.charSequenceText?.toString() ?: request.text ?: ""
        val voice = request.voiceName?.takeIf { it in engine.voices }
            ?: currentVoice
            ?: onGetDefaultVoiceNameFor(request.language, request.country, request.variant)
        val speed = (request.speechRate / 100f).coerceIn(0.25f, 4f) * Settings.speedScale(this)
        if (callback.start(SAMPLE_RATE, AudioFormat.ENCODING_PCM_16BIT, 1) != TextToSpeech.SUCCESS) return
        if (text.isBlank()) { callback.done(); return }
        try {
            var offset = 0 // samples already delivered for this request
            val maxBytes = callback.maxBufferSize
            engine.synthesize(text, voice, speed, cancel) { chunk ->
                for (w in chunk.words) callback.rangeStart(offset + w.sample, w.start, w.end)
                val pcm = toPcm16(chunk.audio)
                var pos = 0
                while (pos < pcm.size) {
                    val len = minOf(maxBytes, pcm.size - pos)
                    if (callback.audioAvailable(pcm, pos, len) != TextToSpeech.SUCCESS) return@synthesize false
                    pos += len
                }
                offset += chunk.audio.size
                Log.d(TAG, "chunk ${chunk.stats.frames} frames (bucket ${chunk.stats.bucket}) ${chunk.stats.ms} on ${engine.activeDecoderBackend}")
                !cancel.get()
            }
            if (cancel.get()) callback.error(TextToSpeech.STOPPED) else callback.done()
        } catch (e: Throwable) {
            Log.e(TAG, "synthesis failed", e)
            callback.error(TextToSpeech.ERROR_SYNTHESIS)
        }
    }

    companion object {
        private const val TAG = "KokoroTts"
        const val SAMPLE_RATE = 24000

        fun isEnglish(lang: String?) = lang != null && lang.lowercase() in setOf("eng", "en")

        fun toPcm16(audio: FloatArray): ByteArray {
            val out = ByteArray(audio.size * 2)
            for (i in audio.indices) {
                val v = (audio[i].coerceIn(-1f, 1f) * 32767f).toInt()
                out[2 * i] = (v and 0xff).toByte()
                out[2 * i + 1] = (v shr 8).toByte()
            }
            return out
        }
    }
}
