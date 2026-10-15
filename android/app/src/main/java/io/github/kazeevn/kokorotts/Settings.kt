package io.github.kazeevn.kokorotts

import android.content.Context
import io.github.kazeevn.kokorotts.engine.Backend
import io.github.kazeevn.kokorotts.engine.KokoroEngine

/** User preferences shared by the settings screen and the TTS service. */
object Settings {
    private const val PREFS = "kokoro"
    private const val KEY_VOICE = "voice"
    private const val KEY_BACKEND = "backend"
    private const val KEY_SPEED = "speed_scale"

    private fun prefs(c: Context) = c.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    fun defaultVoice(c: Context): String = prefs(c).getString(KEY_VOICE, "af_heart")!!
    fun setDefaultVoice(c: Context, v: String) = prefs(c).edit().putString(KEY_VOICE, v).apply()

    fun backend(c: Context): Backend = Backend.valueOf(prefs(c).getString(KEY_BACKEND, Backend.NPU.name)!!)
    fun setBackend(c: Context, b: Backend) = prefs(c).edit().putString(KEY_BACKEND, b.name).apply()

    /** Extra speed multiplier applied on top of the requesting app's speech rate. */
    fun speedScale(c: Context): Float = prefs(c).getFloat(KEY_SPEED, 1f)
    fun setSpeedScale(c: Context, v: Float) = prefs(c).edit().putFloat(KEY_SPEED, v).apply()

    /** Applies the backend preference: NPU graphs load in the background; CPU serves meanwhile. */
    fun apply(c: Context, engine: KokoroEngine) {
        engine.preferredBackend = backend(c)
        if (engine.preferredBackend == Backend.NPU) engine.prepareNpuAsync() else engine.releaseNpu()
    }
}
