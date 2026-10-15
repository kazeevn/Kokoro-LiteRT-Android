package io.github.kazeevn.kokorotts.g2p

/**
 * Port of misaki.espeak.EspeakFallback (misaki 0.9.4): phonemizes out-of-vocabulary words with
 * espeak-ng and maps espeak's IPA to misaki's phoneme set.
 *
 * [espeak] receives a chunk of text and must return espeak-ng's IPA for it with words separated
 * by spaces (or null on failure), using voice "en-us" / "en-gb" according to [british].
 *
 * misaki runs espeak through phonemizer with `tie='^'`, i.e. multi-letter phonemes are joined
 * by '^' ("a^ɪ", "d^ʒ"). For identical output, call `espeak_TextToPhonemes` with
 *     phonememode = 0x02 /* IPA */ or 0x80 /* use tie */ or ('^'.code shl 8)
 * and pass [tied] = true. With [tied] = false (plain IPA, mode 0x02) the ties are re-inserted
 * heuristically for the sequences espeak-ng 1.52 ties in English; this differs from misaki for
 * roughly 2-4% of fallback words (mostly syllabic "əl").
 *
 * Like phonemizer's `preserve_punctuation=True`, the text is split at punctuation marks, the
 * pieces are phonemized separately and the marks are kept (e.g. "10:30" -> "tˈɛn:θˈɜɹTi").
 */
class EspeakFallback(
    private val british: Boolean,
    private val tied: Boolean = true,
    private val espeak: (String) -> String?,
) : (String) -> String? {

    override fun invoke(text: String): String? {
        val raw = phonemizePreservingPunctuation(text) ?: return null
        val ps = if (tied) raw else retie(raw)
        return convert(ps, british)
    }

    private fun phonemizePreservingPunctuation(text: String): String? {
        val sb = StringBuilder()
        var last = 0
        var anyOk = false
        for (m in MARKS_RE.findAll(text)) {
            if (m.range.first > last) {
                val p = espeak(text.substring(last, m.range.first)) ?: return null
                sb.append(p.trim())
                anyOk = true
            }
            sb.append(m.value.replace(WS_RE, " "))
            last = m.range.last + 1
        }
        if (last < text.length) {
            val p = espeak(text.substring(last)) ?: return null
            sb.append(p.trim())
            anyOk = true
        }
        if (!anyOk && sb.isEmpty()) return null
        return sb.toString().trim()
    }

    companion object {
        /** Heuristically re-inserts espeak's '^' ties into untied IPA output. */
        internal fun retie(ps: String): String = RETIE_RE.replace(ps) { m -> RETIE.getValue(m.value) }

        private const val PUNCTUATION_MARKS = ";:,.!?¡¿—…\"«»“”(){}[]"
        private val MARKS_RE = Regex("(?:\\s*[" + PUNCTUATION_MARKS.map { "\\" + it }.joinToString("") + "]+\\s*)+")
        private val WS_RE = Regex("\\s")

        /** Sequences espeak-ng 1.52 (en-us / en-gb) joins with a tie character. */
        private val RETIE: Map<String, String> = listOf(
            "aɪə", "aɪɚ", "aɪ", "aʊ", "eɪ", "ɔɪ", "oʊ", "əʊ", "dʒ", "tʃ", "əl", "iə",
        ).associateWith { s -> s.toList().joinToString("^") }.mapValues { (k, v) ->
            when (k) {
                "aɪə" -> "a^ɪ^ə"
                "aɪɚ" -> "a^ɪ^ɚ"
                else -> v
            }
        }
        private val RETIE_RE = Regex(RETIE.keys.sortedByDescending { it.length }.joinToString("|") { Regex.escape(it) })

        private val E2M: List<Pair<String, String>> = listOf(
            "ʔˌn̩" to "ʔn", "ʔn̩" to "ʔn",
            "a^ɪ" to "I", "a^ʊ" to "W",
            "d^ʒ" to "ʤ",
            "e^ɪ" to "A", "e" to "A",
            "t^ʃ" to "ʧ",
            "ɔ^ɪ" to "Y",
            "ə^l" to "ᵊl",
            "ʲo" to "jo", "ʲə" to "jə", "ʲ" to "",
            "ɚ" to "əɹ",
            "r" to "ɹ",
            "x" to "k", "ç" to "k",
            "ɐ" to "ə",
            "ɬ" to "l",
            "̃" to "",
        ).sortedByDescending { it.first.length } // stable, like Python's sorted(key=-len)

        private val SYLLABIC_RE = Regex("(\\S)̩")

        /** The mapping part of EspeakFallback.__call__ (version != "2.0"). */
        fun convert(espeakIpa: String, british: Boolean): String {
            var ps = espeakIpa.trim()
            for ((old, new) in E2M) ps = ps.replace(old, new)
            ps = SYLLABIC_RE.replace(ps) { m -> "ᵊ" + m.groupValues[1] }.replace("̩", "")
            if (british) {
                ps = ps.replace("e^ə", "ɛː")
                ps = ps.replace("iə", "ɪə")
                ps = ps.replace("ə^ʊ", "Q")
            } else {
                ps = ps.replace("o^ʊ", "O")
                ps = ps.replace("ɜːɹ", "ɜɹ")
                ps = ps.replace("ɜː", "ɜɹ")
                ps = ps.replace("ɪə", "iə")
                ps = ps.replace("ː", "")
            }
            ps = ps.replace("o", "ɔ") // for espeak < 1.52
            ps = ps.replace("ɾ", "T").replace("ʔ", "t")
            return ps.replace("^", "")
        }
    }
}
