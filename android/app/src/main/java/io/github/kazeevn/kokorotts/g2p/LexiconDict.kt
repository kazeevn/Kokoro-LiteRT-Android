package io.github.kazeevn.kokorotts.g2p

import java.io.InputStream

/**
 * A misaki lexicon dictionary entry: either a plain phoneme string, or a heteronym map
 * (POS tag -> phonemes, null meaning "no pronunciation", always containing "DEFAULT").
 */
internal sealed class LexEntry {
    class Plain(val ps: String) : LexEntry()
    class Tagged(val map: Map<String, String?>) : LexEntry() {
        val default: String? get() = map["DEFAULT"]
    }
}

/**
 * Read-only dictionary over a sorted `key\tvalue\n` TSV produced by convert/build_lexicon.py.
 *
 * The file is kept as raw bytes and binary-searched (keys are printable ASCII and sorted
 * byte-wise), so loading is just reading the bytes and indexing line starts: no per-entry
 * objects are created. Values are decoded on lookup and cached.
 *
 * [get] emulates misaki's `Lexicon.grow_dictionary` (lowercase keys are also reachable in
 * Capitalized form and vice versa), which the preprocessing step deliberately does not
 * materialize.
 */
internal class LexiconDict private constructor(
    private val data: ByteArray,
    private val lineStarts: IntArray,
    private val tabs: IntArray,
) {
    val size: Int get() = lineStarts.size

    private val cache = HashMap<String, Any>()

    /** Exact-key lookup (no grow emulation). */
    private fun getExact(key: String): LexEntry? {
        val idx = find(key)
        if (idx < 0) return null
        val start = tabs[idx] + 1
        val end = (if (idx + 1 < lineStarts.size) lineStarts[idx + 1] else data.size) - 1 // drop '\n'
        val s = String(data, start, end - start, Charsets.UTF_8)
        return if (s.startsWith("{")) {
            val m = LinkedHashMap<String, String?>()
            for (part in s.substring(1, s.length - 1).split('|')) {
                val eq = part.indexOf('=')
                val v = part.substring(eq + 1)
                m[part.substring(0, eq)] = if (v == "~") null else v
            }
            LexEntry.Tagged(m)
        } else {
            LexEntry.Plain(s)
        }
    }

    /** `grow_dictionary(d).get(word)`. */
    @Synchronized
    fun get(word: String): LexEntry? {
        cache[word]?.let { return it as? LexEntry }
        val r = getGrown(word)
        if (cache.size > 50_000) cache.clear()
        cache[word] = r ?: MISSING
        return r
    }

    operator fun contains(word: String): Boolean = get(word) != null

    private fun getGrown(w: String): LexEntry? {
        getExact(w)?.let { return it }
        if (w.length < 2) return null
        val lo = w.lowercase()
        val cap = pyCapitalize(lo)
        if (w == lo && cap != lo) return getExact(cap)
        if (w == cap && w != lo) return getExact(lo)
        return null
    }

    private fun compareAt(idx: Int, key: String): Int {
        var p = lineStarts[idx]
        val end = tabs[idx]
        var i = 0
        while (p < end && i < key.length) {
            val a = data[p].toInt() and 0xff
            val b = key[i].code
            if (a != b) return a - b
            p++
            i++
        }
        return (end - p) - (key.length - i)
    }

    private fun find(key: String): Int {
        for (c in key) if (c.code !in 32..126) return -1
        var lo = 0
        var hi = lineStarts.size - 1
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            val c = compareAt(mid, key)
            when {
                c < 0 -> lo = mid + 1
                c > 0 -> hi = mid - 1
                else -> return mid
            }
        }
        return -1
    }

    companion object {
        private val MISSING = Any()

        fun read(input: InputStream): LexiconDict {
            val data = input.readBytes()
            var n = 0
            for (b in data) if (b == '\n'.code.toByte()) n++
            val starts = IntArray(n)
            val tabs = IntArray(n)
            var line = 0
            var start = 0
            var tab = -1
            for (i in data.indices) {
                val b = data[i]
                if (b == '\t'.code.toByte() && tab < 0) {
                    tab = i
                } else if (b == '\n'.code.toByte()) {
                    require(tab >= 0) { "malformed lexicon line $line" }
                    starts[line] = start
                    tabs[line] = tab
                    line++
                    start = i + 1
                    tab = -1
                }
            }
            return LexiconDict(data, starts, tabs)
        }

        /** Python str.capitalize() for the ASCII keys of the lexicon. */
        fun pyCapitalize(s: String): String =
            if (s.isEmpty()) s else s.substring(0, 1).uppercase() + s.substring(1).lowercase()
    }
}
