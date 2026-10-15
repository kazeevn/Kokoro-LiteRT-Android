package io.github.kazeevn.kokorotts

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.IBinder
import android.os.SystemClock
import android.util.Log
import io.github.kazeevn.kokorotts.engine.Backend
import io.github.kazeevn.kokorotts.engine.KokoroEngine
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

/**
 * Benchmark driver, run as a foreground service so it gets normal CPU scheduling with the screen
 * off or locked (like a TTS engine bound by a reading app). Driven over adb:
 *
 *   adb shell am start-foreground-service -n io.github.kazeevn.kokorotts/.BenchService \
 *       --es backend NPU --ei repeats 3 --ei settle_s 20 --es tag npu1 [--es mode idle --ei idle_s 60]
 *
 * Phases are timestamped with CLOCK_BOOTTIME (= Perfetto's default trace clock) in
 * <external files>/results/bench_<tag>.json, so the host can integrate power-rail energy per phase.
 */
class BenchService : Service() {
    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel("bench", "Benchmark", NotificationManager.IMPORTANCE_LOW))
        val notification = Notification.Builder(this, "bench").setContentTitle("Kokoro benchmark running")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now).build()
        startForeground(1, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)

        val backend = Backend.valueOf(intent?.getStringExtra("backend") ?: "CPU")
        val repeats = intent?.getIntExtra("repeats", 3) ?: 3
        val settle = intent?.getIntExtra("settle_s", 20) ?: 20
        val tag = intent?.getStringExtra("tag") ?: backend.name.lowercase()
        val mode = intent?.getStringExtra("mode") ?: "synth"
        val idleS = intent?.getIntExtra("idle_s", 60) ?: 60
        val voice = intent?.getStringExtra("voice") ?: "af_heart"
        val dumpWav = intent?.getBooleanExtra("dump", false) ?: false
        val threads = intent?.getIntExtra("threads", KokoroEngine.THREADS) ?: KokoroEngine.THREADS
        val fp16 = intent?.getBooleanExtra("fp16", false) ?: false
        val frameCap = intent?.getIntExtra("frame_cap", 0) ?: 0
        val wakeLock = getSystemService(android.os.PowerManager::class.java)
            .newWakeLock(android.os.PowerManager.PARTIAL_WAKE_LOCK, "kokoro:bench")
        wakeLock.acquire(3 * 60 * 60 * 1000L)
        thread(name = "kokoro-bench") {
            val out = JSONObject().put("tag", tag).put("backend", backend.name).put("mode", mode)
            val phases = JSONArray()
            fun now() = SystemClock.elapsedRealtimeNanos()
            try {
                if (mode == "compare") {
                    // NPU vs CPU decoder on identical inputs: two passes with the same chunking
                    // (CPU capped at the NPU's largest bucket), so both decoders are never loaded at once.
                    val engine = KokoroEngine.get(this)
                    val corpus = assets.open("bench_corpus.txt").bufferedReader().readText()
                    fun pass(): List<FloatArray> {
                        val chunks = ArrayList<FloatArray>()
                        engine.synthesize(corpus, voice, 1f, AtomicBoolean(false)) { chunks.add(it.audio); true }
                        return chunks
                    }
                    io.github.kazeevn.kokorotts.engine.Graph.checkOutputs = intent?.getBooleanExtra("check", false) ?: false
                    io.github.kazeevn.kokorotts.engine.Graph.dumpDir = getExternalFilesDir("dump")
                    engine.preferredBackend = Backend.NPU
                    check(engine.prepareNpu()) { engine.npuState }
                    engine.warmUp(false)
                    val npu = pass()
                    engine.releaseNpu()
                    engine.preferredBackend = Backend.CPU
                    engine.frameCap = engine.npuMaxFrames
                    engine.warmUp(false)
                    val cpu = pass()
                    engine.frameCap = 0
                    check(npu.size == cpu.size) { "chunking differs: ${npu.size} vs ${cpu.size}" }
                    val arr = JSONArray()
                    for (i in npu.indices) {
                        val r = engine.compare(cpu[i], npu[i])
                        val bad = npu[i].count { !it.isFinite() } to cpu[i].count { !it.isFinite() }
                        Log.i("KokoroBench", "chunk $i: ${npu[i].size} samples, snr ${r[0]} dB, " +
                            "logspec corr ${r[1]}, non-finite npu/cpu ${bad.first}/${bad.second}")
                        arr.put(JSONArray(r.map { if (it.isFinite()) it else JSONObject.NULL }))
                    }
                    out.put("snr_db_and_logspec_corr", arr)
                } else if (mode == "idle") {
                    val t0 = now(); Thread.sleep(idleS * 1000L)
                    phases.put(JSONObject().put("name", "idle").put("start_ns", t0).put("end_ns", now()))
                } else {
                    val engine = KokoroEngine.get(this)
                    engine.configureCpu(threads, fp16)
                    engine.frameCap = frameCap
                    out.put("frame_cap", frameCap)
                    out.put("cpu_threads", threads).put("cpu_fp16", fp16)
                    engine.preferredBackend = backend
                    val tPrep = now()
                    if (backend == Backend.NPU) check(engine.prepareNpu()) { "NPU load failed: ${engine.npuState}" }
                    else engine.releaseNpu()
                    engine.warmUp(engine.isBritish(voice))
                    val corpus = assets.open("bench_corpus.txt").bufferedReader().readText()
                    // one untimed pass so every bucket is touched (lazy allocations, caches)
                    val dump = java.io.ByteArrayOutputStream()
                    val tWarm = now()
                    engine.synthesize(corpus, voice, 1f, AtomicBoolean(false)) { c ->
                        if (dumpWav) dump.write(KokoroTtsService.toPcm16(c.audio))
                        true
                    }
                    out.put("warm_pass_s", (now() - tWarm) / 1e9)
                    if (dumpWav) writeWav(File(getExternalFilesDir("results"), "bench_$tag.wav"), dump.toByteArray())
                    phases.put(JSONObject().put("name", "prepare").put("start_ns", tPrep).put("end_ns", now()))
                    out.put("active_backend", engine.activeDecoderBackend.name)
                    Thread.sleep(settle * 1000L) // let rails settle before measuring
                    // Like Readest: one TTS request per sentence (it parses SSML marks client-side).
                    val sentences = corpus.split(Regex("(?<=[.!?…\"”])\\s+")).map { it.trim() }.filter { it.isNotEmpty() }
                    out.put("sentences", sentences.size)
                    for (r in 0 until repeats) {
                        var audioSamples = 0L
                        val stageMs = HashMap<String, Long>()
                        var chunks = 0
                        val ttfa = JSONArray()
                        val t0 = now()
                        for (sentence in sentences) {
                            val ts = now()
                            var first = true
                            engine.synthesize(sentence, voice, 1f, AtomicBoolean(false)) { c ->
                                if (first) { ttfa.put((now() - ts) / 1_000_000); first = false }
                                audioSamples += c.audio.size
                                chunks++
                                c.stats.ms.forEach { (k, v) -> stageMs[k] = (stageMs[k] ?: 0) + v }
                                true
                            }
                        }
                        val t1 = now()
                        phases.put(JSONObject().put("name", "synth$r").put("start_ns", t0).put("end_ns", t1)
                            .put("audio_s", audioSamples / 24000.0).put("chunks", chunks)
                            .put("wall_s", (t1 - t0) / 1e9).put("stage_ms", JSONObject(stageMs as Map<*, *>))
                            .put("ttfa_ms", ttfa))
                        Log.i(TAG, "$tag repeat $r: ${audioSamples / 24000.0}s audio in ${(t1 - t0) / 1e9}s $stageMs")
                        Thread.sleep(settle * 1000L)
                    }
                }
                out.put("ok", true)
            } catch (e: Throwable) {
                Log.e(TAG, "benchmark failed", e)
                out.put("ok", false).put("error", e.toString())
            }
            out.put("phases", phases)
            File(getExternalFilesDir("results"), "bench_$tag.json").writeText(out.toString(2))
            Log.i(TAG, "DONE $tag")
            wakeLock.release()
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
        }
        return START_NOT_STICKY
    }

    private fun writeWav(f: File, pcm: ByteArray) {
        val h = java.nio.ByteBuffer.allocate(44).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        h.put("RIFF".toByteArray()).putInt(36 + pcm.size).put("WAVE".toByteArray())
        h.put("fmt ".toByteArray()).putInt(16).putShort(1).putShort(1).putInt(24000).putInt(48000)
            .putShort(2).putShort(16).put("data".toByteArray()).putInt(pcm.size)
        f.outputStream().use { it.write(h.array()); it.write(pcm) }
    }

    companion object { private const val TAG = "KokoroBench" }
}
