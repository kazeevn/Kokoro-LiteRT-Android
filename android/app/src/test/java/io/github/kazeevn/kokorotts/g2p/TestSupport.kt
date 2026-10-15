package io.github.kazeevn.kokorotts.g2p

import java.io.File
import java.util.concurrent.TimeUnit

/** Shared helpers for the G2P JVM tests (run from android/app). */
object TestSupport {
    val repoRoot: File by lazy {
        var d: File? = File("").absoluteFile
        while (d != null && !File(d, "convert/g2p_test").isDirectory) d = d.parentFile
        requireNotNull(d) { "cannot find repo root from ${File("").absolutePath}" }
    }
    val assets: File get() = File(repoRoot, "android/app/src/main/assets/g2p")
    val espeakHost: File get() = File(repoRoot, "work/espeak-host")

    private val lexicons = HashMap<Boolean, Lexicon>()

    @Synchronized
    fun lexicon(british: Boolean): Lexicon = lexicons.getOrPut(british) {
        val p = if (british) "gb" else "us"
        File(assets, "${p}_gold.tsv").inputStream().use { g ->
            File(assets, "${p}_silver.tsv").inputStream().use { s -> Lexicon.load(g, s, british) }
        }
    }

    val espeakAvailable: Boolean get() = File(espeakHost, "src/espeak-ng").canExecute()

    private val cache = HashMap<String, String?>()

    /** Raw espeak-ng IPA (tie '^') from the host build, like phonemizer's backend. */
    @Synchronized
    fun hostEspeak(text: String, british: Boolean, tie: Boolean = true): String? = cache.getOrPut("$british|$tie|$text") {
        // Text as an argument: stdin line mode treats the end of line as a clause end and
        // stresses differently ("St" -> sˈənt vs sənt), unlike espeak_TextToPhonemes.
        val arg = text.replace('\n', ' ').let { if (it.startsWith("-")) " $it" else it }
        val cmd = mutableListOf(File(espeakHost, "src/espeak-ng").path, "-q", "--ipa", "-v", if (british) "en-gb" else "en-us", arg)
        if (tie) cmd.add(3, "--tie=^")
        val pb = ProcessBuilder(cmd)
        pb.environment()["ESPEAK_DATA_PATH"] = espeakHost.path
        pb.redirectErrorStream(false)
        val p = pb.start()
        p.outputStream.close()
        val out = p.inputStream.bufferedReader(Charsets.UTF_8).readText()
        p.waitFor(30, TimeUnit.SECONDS)
        out.lines().map { it.trim() }.filter { it.isNotEmpty() }.joinToString(" ").ifEmpty { null }
    }

    /** Minimal JSON parser (org.json is a stub in Android unit tests). */
    class Json(private val s: String) {
        private var i = 0
        fun parse(): Any? {
            val v = value()
            ws()
            require(i == s.length) { "trailing data at $i" }
            return v
        }
        private fun ws() { while (i < s.length && s[i].isWhitespace()) i++ }
        private fun value(): Any? {
            ws()
            return when (s[i]) {
                '{' -> {
                    i++
                    val m = LinkedHashMap<String, Any?>()
                    ws()
                    if (s[i] == '}') { i++; return m }
                    while (true) {
                        ws()
                        val k = str()
                        ws(); require(s[i] == ':'); i++
                        m[k] = value()
                        ws()
                        if (s[i] == ',') { i++; continue }
                        require(s[i] == '}'); i++
                        return m
                    }
                    @Suppress("UNREACHABLE_CODE") m
                }
                '[' -> {
                    i++
                    val l = ArrayList<Any?>()
                    ws()
                    if (s[i] == ']') { i++; return l }
                    while (true) {
                        l.add(value())
                        ws()
                        if (s[i] == ',') { i++; continue }
                        require(s[i] == ']'); i++
                        return l
                    }
                    @Suppress("UNREACHABLE_CODE") l
                }
                '"' -> str()
                'n' -> { i += 4; null }
                't' -> { i += 4; true }
                'f' -> { i += 5; false }
                else -> {
                    val st = i
                    while (i < s.length && (s[i].isDigit() || s[i] in "+-.eE")) i++
                    s.substring(st, i).toDouble()
                }
            }
        }
        private fun str(): String {
            require(s[i] == '"'); i++
            val sb = StringBuilder()
            while (s[i] != '"') {
                val c = s[i++]
                if (c == '\\') {
                    when (val e = s[i++]) {
                        'n' -> sb.append('\n'); 't' -> sb.append('\t'); 'r' -> sb.append('\r')
                        'b' -> sb.append('\b'); 'f' -> sb.append('\u000c')
                        'u' -> { sb.append(s.substring(i, i + 4).toInt(16).toChar()); i += 4 }
                        else -> sb.append(e)
                    }
                } else {
                    sb.append(c)
                }
            }
            i++
            return sb.toString()
        }
    }
}
