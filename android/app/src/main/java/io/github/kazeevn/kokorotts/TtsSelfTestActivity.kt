package io.github.kazeevn.kokorotts

import android.app.Activity
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.io.File

/**
 * Exercises the engine through the public TextToSpeech API the way Readest's native-tts plugin
 * does (engine binding, getVoices, setVoice by name, setSpeechRate, speak with an utterance id,
 * UtteranceProgressListener incl. onRangeStart). Driven over adb:
 *   adb shell am start -n io.github.kazeevn.kokorotts/.TtsSelfTestActivity --es voice bf_emma
 * Writes a log to <external files>/results/tts_selftest.txt.
 */
class TtsSelfTestActivity : Activity() {
    private lateinit var tts: TextToSpeech
    private val log = StringBuilder()
    private val t0 = System.currentTimeMillis()

    private fun log(s: String) {
        val line = "+${System.currentTimeMillis() - t0}ms $s"
        Log.i(TAG, line)
        synchronized(log) { log.appendLine(line) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val voiceName = intent.getStringExtra("voice") ?: "af_heart"
        val text = intent.getStringExtra("text")
            ?: "Chapter one. It was a bright cold day in April, and the clocks were striking thirteen."
        tts = TextToSpeech(this, { status ->
            log("init status=$status engine=${tts.defaultEngine} (bound: $packageName)")
            if (status != TextToSpeech.SUCCESS) return@TextToSpeech finishTest()
            val voices = tts.voices
            log("voices=${voices?.size}: ${voices?.joinToString { "${it.name}/${it.locale.toLanguageTag()}" }}")
            val v = voices?.find { it.name == voiceName }
            log("setVoice($voiceName) -> ${v?.let { tts.setVoice(it) }}")
            tts.setSpeechRate(1.0f)
            tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(id: String?) = log("onStart $id")
                override fun onDone(id: String?) { log("onDone $id"); finishTest() }
                @Deprecated("deprecated") override fun onError(id: String?) { log("onError $id"); finishTest() }
                override fun onError(id: String?, code: Int) { log("onError $id code=$code"); finishTest() }
                override fun onRangeStart(id: String?, start: Int, end: Int, frame: Int) =
                    log("onRangeStart $id [$start,$end) '${text.substring(start, end)}' frame=$frame")
            })
            val r = tts.speak(text, TextToSpeech.QUEUE_FLUSH, Bundle().apply {
                putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, "u1")
            }, "u1")
            log("speak -> $r")
        }, packageName)
    }

    private fun finishTest() {
        File(getExternalFilesDir("results"), "tts_selftest.txt").writeText(synchronized(log) { log.toString() })
        runOnUiThread { tts.shutdown(); finish() }
    }

    companion object { private const val TAG = "KokoroSelfTest" }
}
