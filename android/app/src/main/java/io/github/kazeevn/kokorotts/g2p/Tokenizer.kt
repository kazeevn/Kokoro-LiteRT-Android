package io.github.kazeevn.kokorotts.g2p

/** A word-level token with char offsets [start, end) into the original text. */
internal class RawToken(
    val text: String,
    val start: Int,
    val end: Int,
    var whitespace: String,
    var tag: String = "",
) {
    override fun toString(): String = "$text/$tag" + if (whitespace.isEmpty()) "" else "_"
}

/**
 * Rule-based approximation of spaCy's English tokenizer (spaCy 3.x `English.Defaults`):
 * whitespace split, then the same prefix / suffix / infix regexes and a subset of the
 * tokenizer exceptions (contractions, common abbreviations), plus spaCy's URL matcher.
 *
 * Differences from spaCy: whitespace other than single spaces is not turned into SPACE tokens
 * (any whitespace run just separates tokens and is reported as a single " "), emoticon and
 * some rare exceptions are not included.
 */
internal object EnTokenizer {
    private fun alt(items: List<String>): String = items.joinToString("|")
    private fun esc(chars: String): List<String> = chars.split(' ').filter { it.isNotEmpty() }.map { Regex.escape(it) }

    private const val PUNCT_CHARS = "… …… , : ; ! ? ¿ ؟ ¡ ( ) [ ] { } < > _ # * & 。 ？ ！ ， 、 ； ： ～ · । ، ۔ ؛ ٪"
    private const val QUOTE_CHARS = "' \" ” “ ` ‘ ´ ’ ‚ , „ » « 「 」 『 』 （ ） 〔 〕 【 】 《 》 〈 〉 〈 〉 ⟦ ⟧"
    private const val CURRENCY_CHARS = "$ £ € ¥ ฿ US$ C$ A$ ₽ ﷼ ₴ ₠ ₡ ₢ ₣ ₤ ₥ ₦ ₧ ₨ ₩ ₪ ₫ € ₭ ₮ ₯ ₰ ₱ ₲ ₳ ₴ ₵ ₶ ₷ ₸ ₹ ₺ ₻ ₼ ₽ ₾ ₿"
    private const val UNITS = "km km² km³ m m² m³ dm dm² dm³ cm cm² cm³ mm mm² mm³ ha µm nm yd in ft kg g mg µg t lb oz m/s km/h kmh mph hPa Pa mbar mb MB kb KB gb GB tb TB T G M K % км км² км³ м м² м³ дм дм² дм³ см см² см³ мм мм² мм³ нм кг г мг м/с км/ч кПа Па мбар Кб КБ кб Мб МБ мб Гб ГБ гб Тб ТБ тб"

    private val LIST_PUNCT = esc(PUNCT_CHARS)
    private val LIST_QUOTES = esc(QUOTE_CHARS)
    private val LIST_CURRENCY = esc(CURRENCY_CHARS)
    private val LIST_ELLIPSES = listOf("\\.\\.+", "…")
    private const val ICONS = "[\\p{So}&&[^°]]"

    /** Characters for use inside a [...] class. */
    private fun cls(chars: String): String = chars.split(' ').filter { it.length == 1 }.joinToString("") {
        if (it in "\\[]^-&") "\\$it" else it
    }

    // spaCy's PUNCT is merge_chars(_punct): the punct chars joined with '|'; CONCAT_QUOTES: quote chars.
    private val PUNCT_CLASS = cls(PUNCT_CHARS) + "|"
    private val QUOTE_CLASS = cls(QUOTE_CHARS)

    private val PREFIX_RE = Regex(
        "^(?:" + alt(
            listOf("§", "%", "=", "—", "–", "\\+(?![0-9])") + LIST_PUNCT + LIST_ELLIPSES + LIST_QUOTES +
                LIST_CURRENCY + listOf(ICONS),
        ) + ")",
    )

    private val SUFFIX_RE = Regex(
        "(?:" + alt(
            LIST_PUNCT + LIST_ELLIPSES + LIST_QUOTES + listOf(ICONS) +
                listOf("'s", "'S", "’s", "’S", "—", "–") +
                listOf(
                    "(?<=[0-9])\\+",
                    "(?<=°[FfCcKk])\\.",
                    "(?<=[0-9])(?:" + alt(LIST_CURRENCY) + ")",
                    "(?<=[0-9])(?:" + alt(esc(UNITS)) + ")",
                    "(?<=[0-9\\p{Ll}%²\\-+$QUOTE_CLASS$PUNCT_CLASS])\\.",
                    "(?<=[\\p{Lu}][\\p{Lu}])\\.",
                ),
        ) + ")$",
    )

    private val INFIX_RE = Regex(
        alt(
            LIST_ELLIPSES + listOf(ICONS) + listOf(
                "(?<=[0-9])[+\\-*^](?=[0-9-])",
                "(?<=[\\p{Ll}$QUOTE_CLASS])\\.(?=[\\p{Lu}$QUOTE_CLASS])",
                "(?<=\\p{L}),(?=\\p{L})",
                "(?<=[\\p{L}0-9])(?:---|--|——|-|–|—|~)(?=\\p{L})",
                "(?<=[\\p{L}0-9])[:<>=/](?=\\p{L})",
            ),
        ),
    )

    private const val HOST_CHAR = "A-Za-z0-9\\u00a1-\\uffff"
    private val URL_RE = Regex(
        "^(?:(?:[\\w+\\-.]{2,})://)?" +
            "(?:\\S+(?::\\S*)?@)?" +
            "(?:(?:(?:[$HOST_CHAR][${HOST_CHAR}_-]{0,62})?[$HOST_CHAR]\\.)+(?:\\p{Ll}{2,63}))" +
            "(?::\\d{2,5})?" +
            "(?:[/?#]\\S*)?$",
    )

    /** Tokenizer exceptions: string -> pieces. */
    private val SPECIALS: Map<String, List<String>> = buildSpecials()

    private fun buildSpecials(): Map<String, List<String>> {
        val exc = HashMap<String, List<String>>()
        fun add(vararg pieces: String) {
            exc[pieces.joinToString("")] = pieces.toList()
        }
        fun title(s: String) = En.capitalize(s)

        for (orth in listOf("i", "I")) {
            add(orth, "'m"); add(orth, "'m", "a")
        }
        for (pron in listOf("i", "you", "he", "she", "it", "we", "they")) {
            for (orth in listOf(pron, title(pron)).distinct()) {
                add(orth, "'ll"); add(orth, "'ll", "'ve"); add(orth, "'d"); add(orth, "'d", "'ve")
            }
        }
        for (pron in listOf("i", "you", "we", "they")) for (orth in listOf(pron, title(pron)).distinct()) add(orth, "'ve")
        for (pron in listOf("you", "we", "they")) for (orth in listOf(pron, title(pron))) add(orth, "'re")
        for (pron in listOf("he", "she", "it")) for (orth in listOf(pron, title(pron))) add(orth, "'s")
        for (word in listOf("who", "what", "when", "where", "why", "how", "there", "that")) {
            for (orth in listOf(word, title(word))) {
                add(orth, "'s"); add(orth, "'ll"); add(orth, "'ll", "'ve"); add(orth, "'re")
                add(orth, "'ve"); add(orth, "'d"); add(orth, "'d", "'ve")
            }
        }
        for (verb in listOf("ca", "could", "do", "does", "did", "had", "may", "might", "must", "need", "ought", "sha", "should", "wo", "would")) {
            for (orth in listOf(verb, title(verb))) {
                add(orth, "n't"); add(orth, "n't", "'ve")
            }
        }
        for (verb in listOf("could", "might", "must", "should", "would")) for (orth in listOf(verb, title(verb))) add(orth, "'ve")
        for (verb in listOf("ai", "are", "is", "was", "were", "have", "has", "dare")) for (orth in listOf(verb, title(verb))) add(orth, "n't")
        for (w in listOf("doin", "goin", "nothin", "nuthin", "ol", "somethin")) for (orth in listOf(w, title(w))) add("$orth'")
        for (w in listOf("em", "ll", "nuff")) add("'$w")
        for (h in 1..12) for (p in listOf("a.m.", "am", "p.m.", "pm")) add("$h", p)
        add("y'", "all"); add("how", "'d", "'y"); add("How", "'d", "'y"); add("not", "'ve"); add("Not", "'ve")
        add("can", "not"); add("Can", "not"); add("gon", "na"); add("Gon", "na"); add("got", "ta"); add("Got", "ta")
        add("let", "'s"); add("Let", "'s"); add("c'm", "on"); add("C'm", "on")
        for (s in listOf(
            "'S", "'s", "'re", "'Cause", "'cause", "'cos", "'Cos", "'coz", "'Coz", "'cuz", "'Cuz", "'bout",
            "ma'am", "Ma'am", "o'clock", "O'clock", "lovin'", "Lovin'", "havin'", "Havin'",
            "and/or", "w/o", "''", "C++",
            "Mt.", "Ak.", "Ala.", "Apr.", "Ariz.", "Ark.", "Aug.", "Calif.", "Colo.", "Conn.", "Dec.", "Del.",
            "Feb.", "Fla.", "Ga.", "Ia.", "Id.", "Ill.", "Ind.", "Jan.", "Jul.", "Jun.", "Kan.", "Kans.", "Ky.",
            "La.", "Mar.", "Mass.", "May.", "Mich.", "Minn.", "Miss.", "N.C.", "N.D.", "N.H.", "N.J.", "N.M.",
            "N.Y.", "Neb.", "Nebr.", "Nev.", "Nov.", "Oct.", "Okla.", "Ore.", "Pa.", "S.C.", "Sep.", "Sept.",
            "Tenn.", "Va.", "Wash.", "Wis.",
            "'d", "a.m.", "Adm.", "Bros.", "co.", "Co.", "Corp.", "D.C.", "Dr.", "e.g.", "E.g.", "E.G.", "Gen.",
            "Gov.", "i.e.", "I.e.", "I.E.", "Inc.", "Jr.", "Ltd.", "Md.", "Messrs.", "Mo.", "Mont.", "Mr.",
            "Mrs.", "Ms.", "p.m.", "Ph.D.", "Prof.", "Rep.", "Rev.", "Sen.", "St.", "vs.", "v.s.",
        )) add(s)
        for (c in 'a'..'z') add("$c.")
        for (s in listOf("ä.", "ö.", "ü.")) add(s)
        for (s in listOf("Ill", "ill", "Its", "its", "Hell", "hell", "Shell", "shell", "Shed", "shed", "were", "Were", "Well", "well", "Whore", "whore")) exc.remove(s)
        // Curly-apostrophe variants.
        for ((k, v) in exc.entries.toList()) {
            if ('\'' in k) {
                val k2 = k.replace('\'', '’')
                if (k2 !in exc) exc[k2] = v.map { it.replace('\'', '’') }
            }
        }
        return exc
    }

    private fun findPrefix(s: String): Int = PREFIX_RE.find(s)?.let { it.range.last + 1 } ?: 0
    private fun findSuffix(s: String): Int = SUFFIX_RE.find(s)?.let { s.length - it.range.first } ?: 0

    /** Tokenizes [text]; whitespace between tokens is reported as " " on the preceding token. */
    fun tokenize(text: String): List<RawToken> {
        val out = ArrayList<RawToken>()
        var i = 0
        val n = text.length
        while (i < n) {
            while (i < n && text[i].isWhitespace()) i++
            if (i >= n) break
            var j = i
            while (j < n && !text[j].isWhitespace()) j++
            val first = out.size
            splitChunk(text, i, j, out)
            if (out.size > first && j < n) out[out.size - 1].whitespace = " "
            i = j
        }
        return out
    }

    private fun splitChunk(text: String, from: Int, to: Int, out: MutableList<RawToken>) {
        var s = text.substring(from, to)
        var offset = from
        SPECIALS[s]?.let { emitSpecial(it, offset, out); return }
        val prefixes = ArrayList<String>()
        val suffixes = ArrayList<String>()
        var lastSize = 0
        while (s.isNotEmpty() && s.length != lastSize) {
            if (s in SPECIALS) break
            lastSize = s.length
            val preLen = findPrefix(s)
            if (preLen != 0) {
                val minusPre = s.substring(preLen)
                if (minusPre.isNotEmpty() && minusPre in SPECIALS) {
                    prefixes.add(s.substring(0, preLen))
                    s = minusPre
                    break
                }
            }
            val sufLen = findSuffix(s.substring(preLen))
            if (sufLen != 0) {
                val minusSuf = s.substring(0, s.length - sufLen)
                if (minusSuf.isNotEmpty() && minusSuf in SPECIALS) {
                    suffixes.add(s.substring(s.length - sufLen))
                    s = minusSuf
                    break
                }
            }
            if (preLen != 0 && sufLen != 0 && preLen + sufLen <= s.length) {
                prefixes.add(s.substring(0, preLen))
                suffixes.add(s.substring(s.length - sufLen))
                s = s.substring(preLen, s.length - sufLen)
            } else if (preLen != 0) {
                prefixes.add(s.substring(0, preLen))
                s = s.substring(preLen)
            } else if (sufLen != 0) {
                suffixes.add(s.substring(s.length - sufLen))
                s = s.substring(0, s.length - sufLen)
            }
            if (s.isNotEmpty() && s in SPECIALS) break
        }
        for (p in prefixes) {
            out.add(RawToken(p, offset, offset + p.length, ""))
            offset += p.length
        }
        if (s.isNotEmpty()) {
            val special = SPECIALS[s]
            if (special != null) {
                emitSpecial(special, offset, out)
            } else if (URL_RE.matches(s)) {
                out.add(RawToken(s, offset, offset + s.length, ""))
            } else {
                var start = 0
                for (m in INFIX_RE.findAll(s)) {
                    val a = m.range.first
                    val b = m.range.last + 1
                    if (a == 0) continue
                    if (a != start) out.add(RawToken(s.substring(start, a), offset + start, offset + a, ""))
                    if (a != b) out.add(RawToken(s.substring(a, b), offset + a, offset + b, ""))
                    start = b
                }
                if (start < s.length) out.add(RawToken(s.substring(start), offset + start, offset + s.length, ""))
            }
            offset += s.length
        }
        for (suf in suffixes.asReversed()) {
            out.add(RawToken(suf, offset, offset + suf.length, ""))
            offset += suf.length
        }
    }

    private fun emitSpecial(pieces: List<String>, offset0: Int, out: MutableList<RawToken>) {
        var offset = offset0
        for (p in pieces) {
            out.add(RawToken(p, offset, offset + p.length, ""))
            offset += p.length
        }
    }
}
