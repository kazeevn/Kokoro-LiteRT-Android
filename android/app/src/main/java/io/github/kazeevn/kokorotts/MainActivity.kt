package io.github.kazeevn.kokorotts

import android.app.Activity
import android.content.Intent
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.Spinner
import android.widget.TextView
import io.github.kazeevn.kokorotts.engine.Backend
import io.github.kazeevn.kokorotts.engine.KokoroEngine
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

/** Settings (default voice, backend, speed) and a test bench for the engine. */
class MainActivity : Activity() {
    private lateinit var engine: KokoroEngine
    private lateinit var status: TextView
    private val ui = Handler(Looper.getMainLooper())
    private val cancel = AtomicBoolean(false)
    @Volatile private var speaking = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        engine = KokoroEngine.get(this)
        Settings.apply(this, engine)
        val pad = (16 * resources.displayMetrics.density).toInt()
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(pad, pad, pad, pad) }

        root.addView(TextView(this).apply { text = "Default voice" })
        val voices = engine.voices
        root.addView(Spinner(this).apply {
            adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, voices.map(::label))
            setSelection(voices.indexOf(Settings.defaultVoice(this@MainActivity)).coerceAtLeast(0))
            onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) =
                    Settings.setDefaultVoice(this@MainActivity, voices[pos])
                override fun onNothingSelected(p: AdapterView<*>?) {}
            }
        })

        root.addView(TextView(this).apply { text = "\nNeural decoder runs on" })
        root.addView(RadioGroup(this).apply {
            orientation = RadioGroup.HORIZONTAL
            for (b in Backend.entries) addView(RadioButton(this@MainActivity).apply {
                id = View.generateViewId()
                text = if (b == Backend.NPU) "NPU (EdgeTPU)" else "CPU"
                isChecked = Settings.backend(this@MainActivity) == b
                setOnClickListener {
                    Settings.setBackend(this@MainActivity, b)
                    Settings.apply(this@MainActivity, engine)
                    refresh()
                }
            })
        })

        val speedLabel = TextView(this)
        fun showSpeed() { speedLabel.text = "\nExtra speed: %.2fx".format(Settings.speedScale(this)) }
        showSpeed()
        root.addView(speedLabel)
        root.addView(SeekBar(this).apply {
            max = 30 // 0.5x .. 2.0x in 0.05 steps
            progress = ((Settings.speedScale(this@MainActivity) - 0.5f) / 0.05f).toInt()
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(s: SeekBar?, p: Int, user: Boolean) {
                    if (user) { Settings.setSpeedScale(this@MainActivity, 0.5f + p * 0.05f); showSpeed() }
                }
                override fun onStartTrackingTouch(s: SeekBar?) {}
                override fun onStopTrackingTouch(s: SeekBar?) {}
            })
        })

        val input = EditText(this).apply {
            setText("Kokoro is an open-weight text to speech model with 82 million parameters. " +
                "This build runs it on the phone, with the vocoder on the Tensor G3's edge TPU.")
            minLines = 3
        }
        root.addView(input)
        root.addView(Button(this).apply {
            text = "Speak"
            setOnClickListener { speak(input.text.toString()) }
        })
        root.addView(Button(this).apply {
            text = "Stop"
            setOnClickListener { cancel.set(true) }
        })
        root.addView(Button(this).apply {
            text = "System text-to-speech settings"
            setOnClickListener { startActivity(Intent("com.android.settings.TTS_SETTINGS")) }
        })
        status = TextView(this).apply { setTextIsSelectable(true) }
        root.addView(status)
        setContentView(ScrollView(this).apply { addView(root) })
        refresh()
    }

    private fun label(v: String): String {
        val accent = if (v[0] == 'b') "British" else "American"
        val gender = if (v[1] == 'f') "female" else "male"
        return "${v.substringAfter('_').replaceFirstChar { it.uppercase() }} ($accent $gender)"
    }

    private var lastInfo = ""

    private fun refresh() {
        status.text = "\nNPU: ${engine.npuState}\nDecoder backend in use: ${engine.activeDecoderBackend}\n$lastInfo"
        if (!isFinishing) ui.postDelayed(::refresh, 1000)
    }

    private fun speak(text: String) {
        if (speaking) return
        speaking = true
        cancel.set(false)
        val voice = Settings.defaultVoice(this)
        val speed = Settings.speedScale(this)
        thread(name = "kokoro-test") {
            val minBuf = AudioTrack.getMinBufferSize(24000, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
            val track = AudioTrack.Builder()
                .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                .setAudioFormat(AudioFormat.Builder().setSampleRate(24000).setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
                .setBufferSizeInBytes(maxOf(minBuf, 24000 * 2))
                .build()
            val t0 = System.nanoTime()
            var first = -1L
            var audioS = 0.0
            val log = StringBuilder()
            try {
                engine.warmUp(engine.isBritish(voice))
                track.play()
                engine.synthesize(text, voice, speed, cancel) { chunk ->
                    if (first < 0) first = (System.nanoTime() - t0) / 1_000_000
                    audioS += chunk.audio.size / 24000.0
                    log.appendLine("${chunk.stats.frames} frames -> L${chunk.stats.bucket}: ${chunk.stats.ms}")
                    val pcm = KokoroTtsService.toPcm16(chunk.audio)
                    track.write(pcm, 0, pcm.size)
                    !cancel.get()
                }
                val total = (System.nanoTime() - t0) / 1e9
                lastInfo = "First audio after $first ms; %.1f s of audio synthesized in %.1f s\n%s".format(audioS, total, log)
            } catch (e: Throwable) {
                lastInfo = "Error: $e"
            } finally {
                track.stop(); track.release()
                speaking = false
            }
        }
    }
}

/** ACTION_CHECK_TTS_DATA: everything ships in the APK, so all voices are always installed. */
class CheckVoiceDataActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val data = Intent().apply {
            putStringArrayListExtra(android.speech.tts.TextToSpeech.Engine.EXTRA_AVAILABLE_VOICES, arrayListOf("eng-USA", "eng-GBR"))
            putStringArrayListExtra(android.speech.tts.TextToSpeech.Engine.EXTRA_UNAVAILABLE_VOICES, arrayListOf())
        }
        setResult(android.speech.tts.TextToSpeech.Engine.CHECK_VOICE_DATA_PASS, data)
        finish()
    }
}

/** ACTION_GET_SAMPLE_TEXT for the system "Listen to an example" button. */
class SampleTextActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setResult(android.speech.tts.TextToSpeech.LANG_AVAILABLE, Intent().putExtra("sampleText",
            "This is an example of speech synthesis in English with Kokoro."))
        finish()
    }
}
