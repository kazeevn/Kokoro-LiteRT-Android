package io.github.kazeevn.kokorotts

import android.content.ContentProvider
import android.content.ContentValues
import android.content.res.AssetFileDescriptor
import android.database.Cursor
import android.net.Uri
import android.os.Bundle
import android.os.CancellationSignal
import android.os.ParcelFileDescriptor
import android.util.Log
import io.github.kazeevn.kokorotts.engine.KokoroEngine
import java.io.BufferedOutputStream
import java.io.DataOutputStream
import java.io.IOException
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Direct synthesis for apps that schedule playback themselves (Readest): text in, 24 kHz PCM
 * and word timings out. Unlike the system TTS API, the caller can ask for sentences ahead of
 * playback; the framework only synthesizes the next utterance once the previous one is within
 * ~0.5 s of its end (SynthesisPlaybackQueueItem.MAX_UNCONSUMED_AUDIO_MS).
 *
 * - `call("voices")` → `voices` (String[]), `default` (String), `sampleRate` (Int).
 * - `call("cancel", id)` stops request `id` at the engine's next check.
 * - `openTypedAssetFile(content://AUTHORITY/synthesize, <any MIME type>, {text, voice, id})`
 *   returns a pipe of records (big-endian ints), one W + A pair per synthesized chunk:
 *   - `'W'` n, then n × (sample, start, end): word starts (sample offset from the start of the
 *     utterance) and their char range in `text`;
 *   - `'A'` nbytes, then 16-bit little-endian mono PCM;
 *   - `'E'`: done; `'X'` + UTF message: failed. A stream that ends without either was cut off.
 *   Closing the read end also cancels, at the next chunk.
 * Requests run one at a time, in arrival order: the engine is serial anyway.
 */
class SynthesisProvider : ContentProvider() {
    private val executor = Executors.newSingleThreadExecutor { Thread(it, "kokoro-synthesis") }
    private val cancels = ConcurrentHashMap<String, AtomicBoolean>()
    @Volatile private var settingsApplied = false

    private val engine: KokoroEngine get() = KokoroEngine.get(context!!)

    override fun onCreate() = true

    override fun call(method: String, arg: String?, extras: Bundle?): Bundle? = when (method) {
        // Cheap: the engine's metadata only, no graphs.
        "voices" -> Bundle().apply {
            putStringArray("voices", engine.voices.toTypedArray())
            putString("default", Settings.defaultVoice(context!!))
            putInt("sampleRate", KokoroTtsService.SAMPLE_RATE)
        }
        "cancel" -> { arg?.let { cancels[it]?.set(true) }; null }
        else -> null
    }

    override fun openTypedAssetFile(
        uri: Uri, mimeTypeFilter: String, opts: Bundle?, signal: CancellationSignal?,
    ): AssetFileDescriptor {
        val text = opts?.getString("text") ?: throw IllegalArgumentException("missing text")
        val voice = opts.getString("voice")?.takeIf { it in engine.voices } ?: Settings.defaultVoice(context!!)
        val id = opts.getString("id")
        val cancel = AtomicBoolean(false)
        if (id != null) cancels[id] = cancel
        val (read, write) = ParcelFileDescriptor.createPipe()
        executor.execute {
            try {
                DataOutputStream(BufferedOutputStream(ParcelFileDescriptor.AutoCloseOutputStream(write))).use {
                    stream(it, text, voice, cancel)
                }
            } catch (e: IOException) {
                Log.d(TAG, "reader went away: ${e.message}")
            } finally {
                if (id != null) cancels.remove(id, cancel)
            }
        }
        return AssetFileDescriptor(read, 0, AssetFileDescriptor.UNKNOWN_LENGTH)
    }

    private fun stream(out: DataOutputStream, text: String, voice: String, cancel: AtomicBoolean) {
        if (cancel.get()) return
        val engine = engine
        if (!settingsApplied) {
            Settings.apply(context!!, engine)
            settingsApplied = true
        }
        // Rate 1: the caller time-stretches at playout.
        val speed = Settings.speedScale(context!!)
        var offset = 0
        try {
            engine.synthesize(text, voice, speed, cancel) { chunk ->
                out.writeByte('W'.code)
                out.writeInt(chunk.words.size)
                for (w in chunk.words) {
                    out.writeInt(offset + w.sample)
                    out.writeInt(w.start)
                    out.writeInt(w.end)
                }
                val pcm = KokoroTtsService.toPcm16(chunk.audio)
                out.writeByte('A'.code)
                out.writeInt(pcm.size)
                out.write(pcm)
                out.flush()
                offset += chunk.audio.size
                Log.d(TAG, "chunk ${chunk.stats.frames} frames ${chunk.stats.ms} on ${engine.activeDecoderBackend}")
                !cancel.get()
            }
        } catch (e: IOException) {
            throw e
        } catch (e: Throwable) {
            Log.e(TAG, "synthesis failed", e)
            out.writeByte('X'.code)
            out.writeUTF(e.message ?: e.javaClass.simpleName)
            out.flush()
            return
        }
        if (!cancel.get()) {
            out.writeByte('E'.code)
            out.flush()
        }
    }

    override fun getType(uri: Uri): String? = null
    override fun query(uri: Uri, p: Array<out String>?, s: String?, a: Array<out String>?, o: String?): Cursor? = null
    override fun insert(uri: Uri, values: ContentValues?): Uri? = null
    override fun delete(uri: Uri, s: String?, a: Array<out String>?) = 0
    override fun update(uri: Uri, values: ContentValues?, s: String?, a: Array<out String>?) = 0

    companion object {
        private const val TAG = "KokoroSynthesis"
        const val AUTHORITY = "io.github.kazeevn.kokorotts.synthesis"
    }
}
