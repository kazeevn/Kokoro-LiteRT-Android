package io.github.kazeevn.kokorotts.g2p

import java.io.InputStream
import java.math.BigInteger
import java.text.Normalizer

/** Port of misaki.en.TokenContext. */
internal data class TokenContext(val futureVowel: Boolean? = null, val futureTo: Boolean = false)

/** Port of misaki.token.MToken (+ the `_` underscore fields), with char offsets into the input. */
internal class MToken(
    var text: String,
    var tag: String,
    var whitespace: String,
    var start: Int,
    var end: Int,
    var phonemes: String? = null,
    var isHead: Boolean = true,
    var alias: String? = null,
    var stress: Double? = null,
    var currency: String? = null,
    var numFlags: String = "",
    var prespace: Boolean = false,
) {
    fun copy(): MToken = MToken(text, tag, whitespace, start, end, phonemes, isHead, alias, stress, currency, numFlags, prespace)
    override fun toString(): String = "MToken($text/$tag ws='$whitespace' ps=$phonemes)"
}

/** Constants and helpers shared by the misaki port (misaki/en.py module level). */
internal object En {
    const val PRIMARY_STRESS = 'ˈ'
    const val SECONDARY_STRESS = 'ˌ'
    const val STRESSES = "ˌˈ"
    val VOWELS: Set<Char> = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ".toSet()
    val CONSONANTS: Set<Char> = "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ".toSet()
    val US_TAUS: Set<Char> = "AIOWYiuæɑəɛɪɹʊʌ".toSet()
    val DIPHTHONGS: Set<Char> = "AIOQWYʤʧ".toSet()
    val SUBTOKEN_JUNKS: Set<Char> = "',-._‘’/".toSet()
    val PUNCTS: Set<Char> = ";:,.!?—…\"“”".toSet()
    val NON_QUOTE_PUNCTS: Set<Char> = PUNCTS.filter { it !in "\"“”" }.toSet()
    val PUNCT_TAGS = setOf(".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":", "$", "#", "NFP")
    val PUNCT_TAG_PHONEMES = mapOf("-LRB-" to "(", "-RRB-" to ")", "``" to "“", "\"\"" to "”", "''" to "”")
    val CURRENCIES = mapOf("$" to ("dollar" to "cent"), "£" to ("pound" to "pence"), "€" to ("euro" to "cent"))
    val ORDINALS = setOf("st", "nd", "rd", "th")
    val ADD_SYMBOLS = mapOf("." to "dot", "/" to "slash")
    val SYMBOLS = mapOf("%" to "percent", "&" to "and", "+" to "plus", "@" to "at")

    /** ord(c) in LEXICON_ORDS: apostrophe, hyphen, ASCII letters. */
    fun isLexiconChar(c: Char): Boolean = c == '\'' || c == '-' || c in 'A'..'Z' || c in 'a'..'z'

    /** Python str.isalpha() (non-empty, all letters). Surrogate pairs are checked as code points. */
    fun isAlpha(s: String): Boolean {
        if (s.isEmpty()) return false
        var i = 0
        while (i < s.length) {
            val cp = s.codePointAt(i)
            if (!Character.isLetter(cp)) return false
            i += Character.charCount(cp)
        }
        return true
    }

    /** misaki is_digit: ^[0-9]+$ */
    fun isDigit(s: String): Boolean = s.isNotEmpty() && s.all { it in '0'..'9' }

    fun lower(s: String): String = s.lowercase()
    fun upper(s: String): String = s.uppercase()

    /** Python str.capitalize(). */
    fun capitalize(s: String): String =
        if (s.isEmpty()) s else s.substring(0, 1).uppercase() + s.substring(1).lowercase()

    fun stressWeight(ps: String?): Int {
        if (ps == null) return 0
        var w = 0
        for (c in ps) w += if (c in DIPHTHONGS) 2 else 1
        return w
    }

    private fun restress(ps: String): String {
        // Only ever called with a single stress mark at index 0 and no other marks: move it
        // right before the first vowel (misaki sorts (index, char) pairs to the same effect).
        val sb = StringBuilder()
        val marks = ArrayList<Pair<Double, Char>>()
        ps.forEachIndexed { i, c ->
            if (c in STRESSES) {
                var j = i
                while (j < ps.length && ps[j] !in VOWELS) j++
                marks.add((j - 0.5) to c)
            } else {
                marks.add(i.toDouble() to c)
            }
        }
        marks.sortedWith(compareBy<Pair<Double, Char>> { it.first }.thenBy { it.second }).forEach { sb.append(it.second) }
        return sb.toString()
    }

    fun applyStress(ps: String?, stress: Double?): String? {
        if (stress == null || ps == null) return ps
        val hasPrimary = PRIMARY_STRESS in ps
        val hasAnyStress = ps.any { it in STRESSES }
        return when {
            stress < -1 -> ps.replace(PRIMARY_STRESS.toString(), "").replace(SECONDARY_STRESS.toString(), "")
            stress == -1.0 || ((stress == 0.0 || stress == -0.5) && hasPrimary) ->
                ps.replace(SECONDARY_STRESS.toString(), "").replace(PRIMARY_STRESS, SECONDARY_STRESS)
            (stress == 0.0 || stress == 0.5 || stress == 1.0) && !hasAnyStress -> {
                if (ps.none { it in VOWELS }) ps else restress(SECONDARY_STRESS + ps)
            }
            stress >= 1 && !hasPrimary && SECONDARY_STRESS in ps -> ps.replace(SECONDARY_STRESS, PRIMARY_STRESS)
            stress > 1 && !hasAnyStress -> {
                if (ps.none { it in VOWELS }) ps else restress(PRIMARY_STRESS + ps)
            }
            else -> ps
        }
    }
}

/**
 * Port of misaki.en.Lexicon (misaki 0.9.4). Load with [Lexicon.load].
 *
 * Differences from Python: lookup ratings are not tracked (they never influence the output).
 */
class Lexicon private constructor(
    internal val british: Boolean,
    private val golds: LexiconDict,
    private val silvers: LexiconDict,
) {
    private val capStresses = doubleArrayOf(0.5, 2.0)

    companion object {
        /** Loads `{us,gb}_gold.tsv` + `{us,gb}_silver.tsv` produced by convert/build_lexicon.py. */
        fun load(gold: InputStream, silver: InputStream, british: Boolean): Lexicon =
            Lexicon(british, LexiconDict.read(gold), LexiconDict.read(silver))

        internal fun getParentTag(tag: String?): String? = when {
            tag == null -> null
            tag.startsWith("VB") -> "VERB"
            tag.startsWith("NN") -> "NOUN"
            tag.startsWith("ADV") || tag.startsWith("RB") -> "ADV"
            tag.startsWith("ADJ") || tag.startsWith("JJ") -> "ADJ"
            else -> tag
        }

        internal fun isCurrency(word: String): Boolean {
            if ('.' !in word) return true
            if (word.count { it == '.' } > 1) return false
            val cents = word.split('.')[1]
            return cents.length < 3 // `or set(cents) == {0}` is always False in misaki
        }

        /** Lexicon.numeric_if_needed */
        internal fun numericIfNeeded(c: Char): String {
            if (!Character.isDigit(c)) return c.toString()
            val n = Character.getNumericValue(c)
            return if (n >= 0) n.toString() else c.toString()
        }

        private val NUMBER_SUFFIXES = listOf("ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s")

        internal fun isNumber(word: String, isHead: Boolean): Boolean {
            if (word.none { it in '0'..'9' }) return false
            var w = word
            for (s in NUMBER_SUFFIXES) {
                if (w.endsWith(s)) {
                    w = w.substring(0, w.length - s.length)
                    break
                }
            }
            return w.withIndex().all { (i, c) -> c in '0'..'9' || c == ',' || c == '.' || (isHead && i == 0 && c == '-') }
        }

        private val ING_DOUBLE = Regex("([bcdgklmnprstvxz])\\1ing$|cking$")
        private val NUM_SUFFIX = Regex("[a-z']+$")
        private val NON_LOWER = Regex("[^a-z]+")
        private val VS = Regex("(?i)vs\\.?")
    }

    private fun goldString(key: String): String = (golds.get(key) as LexEntry.Plain).ps
    private fun goldTagged(key: String, tag: String): String? = (golds.get(key) as LexEntry.Tagged).map[tag]

    internal fun getNNP(word: String): String? {
        val sb = StringBuilder()
        for (c in word) {
            if (!Character.isLetter(c)) continue
            val e = golds.get(c.toString().uppercase()) ?: return null
            sb.append((e as? LexEntry.Plain)?.ps ?: return null)
        }
        val ps = En.applyStress(sb.toString(), 0.0)!!
        val i = ps.lastIndexOf(En.SECONDARY_STRESS)
        return if (i < 0) ps else ps.substring(0, i) + En.PRIMARY_STRESS + ps.substring(i + 1)
    }

    private fun getSpecialCase(word: String, tag: String, stress: Double?, ctx: TokenContext): String? {
        if (tag == "ADD" && word in En.ADD_SYMBOLS) return lookup(En.ADD_SYMBOLS.getValue(word), null, -0.5, ctx)
        if (word in En.SYMBOLS) return lookup(En.SYMBOLS.getValue(word), null, null, ctx)
        val stripped = word.trim('.')
        if ('.' in stripped && En.isAlpha(word.replace(".", "")) && word.split('.').maxOf { it.length } < 3) {
            return getNNP(word)
        }
        when (word) {
            "a", "A" -> return if (tag == "DT") "ɐ" else "ˈA"
            "am", "Am", "AM" -> {
                if (tag.startsWith("NN")) return getNNP(word)
                if (ctx.futureVowel == null || word != "am" || (stress != null && stress > 0)) return goldString("am")
                return "ɐm"
            }
            "an", "An", "AN" -> {
                if (word == "AN" && tag.startsWith("NN")) return getNNP(word)
                return "ɐn"
            }
        }
        if (word == "I" && tag == "PRP") return "${En.SECONDARY_STRESS}I"
        if ((word == "by" || word == "By" || word == "BY") && getParentTag(tag) == "ADV") return "bˈI"
        if (word == "to" || word == "To" || (word == "TO" && (tag == "TO" || tag == "IN"))) {
            return when (ctx.futureVowel) {
                null -> goldString("to")
                false -> "tə"
                true -> "tʊ"
            }
        }
        if (word == "in" || word == "In" || (word == "IN" && tag != "NNP")) {
            val s = if (ctx.futureVowel == null || tag != "IN") En.PRIMARY_STRESS.toString() else ""
            return s + "ɪn"
        }
        if (word == "the" || word == "The" || (word == "THE" && tag == "DT")) {
            return if (ctx.futureVowel == true) "ði" else "ðə"
        }
        if (tag == "IN" && VS.matches(word)) return lookup("versus", null, null, ctx)
        if (word == "used" || word == "Used" || word == "USED") {
            if ((tag == "VBD" || tag == "JJ") && ctx.futureTo) return goldTagged("used", "VBD")
            return goldTagged("used", "DEFAULT")
        }
        return null
    }

    internal fun isKnown(word: String, tag: String?): Boolean {
        if (word in golds || word in En.SYMBOLS || word in silvers) return true
        if (!En.isAlpha(word) || !word.all { En.isLexiconChar(it) }) return false
        if (word.length == 1) return true
        if (word == En.upper(word) && En.lower(word) in golds) return true
        val rest = word.substring(1)
        return rest == En.upper(rest)
    }

    internal fun lookup(word0: String, tag0: String?, stress: Double?, ctx: TokenContext?): String? {
        var word = word0
        var tag = tag0
        var isNNP: Boolean? = null
        if (word == En.upper(word) && word !in golds) {
            word = En.lower(word)
            isNNP = tag == "NNP"
        }
        var entry = golds.get(word)
        if (entry == null && isNNP != true) entry = silvers.get(word)
        var ps: String? = when (entry) {
            null -> null
            is LexEntry.Plain -> entry.ps
            is LexEntry.Tagged -> {
                val m = entry.map
                if (ctx != null && ctx.futureVowel == null && "None" in m) {
                    tag = "None"
                } else if (tag == null || tag !in m) {
                    tag = getParentTag(tag)
                }
                if (tag != null && tag in m) m[tag] else entry.default
            }
        }
        if (ps == null || (isNNP == true && En.PRIMARY_STRESS !in ps)) {
            // misaki overwrites ps with get_NNP's result, even when that is None
            ps = getNNP(word)
            if (ps != null) return ps
        }
        return En.applyStress(ps, stress)
    }

    private fun suffixS(stem: String?): String? = when {
        stem.isNullOrEmpty() -> null
        stem.last() in "ptkfθ" -> stem + "s"
        stem.last() in "szʃʒʧʤ" -> stem + (if (british) "ɪ" else "ᵻ") + "z"
        else -> stem + "z"
    }

    internal fun stemS(word: String, tag: String?, stress: Double?, ctx: TokenContext?): String? {
        if (word.length < 3 || !word.endsWith("s")) return null
        val stem = when {
            !word.endsWith("ss") && isKnown(word.dropLast(1), tag) -> word.dropLast(1)
            (word.endsWith("'s") || (word.length > 4 && word.endsWith("es") && !word.endsWith("ies"))) &&
                isKnown(word.dropLast(2), tag) -> word.dropLast(2)
            word.length > 4 && word.endsWith("ies") && isKnown(word.dropLast(3) + "y", tag) -> word.dropLast(3) + "y"
            else -> return null
        }
        return suffixS(lookup(stem, tag, stress, ctx))
    }

    private fun suffixEd(stem: String?): String? = when {
        stem.isNullOrEmpty() -> null
        stem.last() in "pkfθʃsʧ" -> stem + "t"
        stem.last() == 'd' -> stem + (if (british) "ɪ" else "ᵻ") + "d"
        stem.last() != 't' -> stem + "d"
        british || stem.length < 2 -> stem + "ɪd"
        stem[stem.length - 2] in En.US_TAUS -> stem.dropLast(1) + "ɾᵻd"
        else -> stem + "ᵻd"
    }

    internal fun stemEd(word: String, tag: String?, stress: Double?, ctx: TokenContext?): String? {
        if (word.length < 4 || !word.endsWith("d")) return null
        val stem = when {
            !word.endsWith("dd") && isKnown(word.dropLast(1), tag) -> word.dropLast(1)
            word.length > 4 && word.endsWith("ed") && !word.endsWith("eed") && isKnown(word.dropLast(2), tag) -> word.dropLast(2)
            else -> return null
        }
        return suffixEd(lookup(stem, tag, stress, ctx))
    }

    private fun suffixIng(stem: String?): String? = when {
        stem.isNullOrEmpty() -> null
        british -> if (stem.last() in "əː") null else stem + "ɪŋ"
        stem.length > 1 && stem.last() == 't' && stem[stem.length - 2] in En.US_TAUS -> stem.dropLast(1) + "ɾɪŋ"
        else -> stem + "ɪŋ"
    }

    internal fun stemIng(word: String, tag: String?, stress: Double?, ctx: TokenContext?): String? {
        if (word.length < 5 || !word.endsWith("ing")) return null
        val stem = when {
            word.length > 5 && isKnown(word.dropLast(3), tag) -> word.dropLast(3)
            isKnown(word.dropLast(3) + "e", tag) -> word.dropLast(3) + "e"
            word.length > 5 && ING_DOUBLE.containsMatchIn(word) && isKnown(word.dropLast(4), tag) -> word.dropLast(4)
            else -> return null
        }
        return suffixIng(lookup(stem, tag, stress, ctx))
    }

    internal fun getWord(word0: String, tag: String, stress: Double?, ctx: TokenContext): String? {
        getSpecialCase(word0, tag, stress, ctx)?.let { return it }
        var word = word0
        val wl = En.lower(word)
        if (word.length > 1 && En.isAlpha(word.replace("'", "")) && word != wl &&
            (tag != "NNP" || word.length > 7) && word !in golds && word !in silvers &&
            (word == En.upper(word) || word.substring(1) == En.lower(word.substring(1))) &&
            (wl in golds || wl in silvers ||
                stemS(wl, tag, stress, ctx) != null || stemEd(wl, tag, stress, ctx) != null ||
                stemIng(wl, tag, stress, ctx) != null)
        ) {
            word = wl
        }
        if (isKnown(word, tag)) return lookup(word, tag, stress, ctx)
        if (word.endsWith("s'") && isKnown(word.dropLast(2) + "'s", tag)) return lookup(word.dropLast(2) + "'s", tag, stress, ctx)
        if (word.endsWith("'") && isKnown(word.dropLast(1), tag)) return lookup(word.dropLast(1), tag, stress, ctx)
        stemS(word, tag, stress, ctx)?.let { return it }
        stemEd(word, tag, stress, ctx)?.let { return it }
        stemIng(word, tag, stress ?: 0.5, ctx)?.let { return it }
        return null
    }

    internal fun getNumber(word0: String, currency: String?, isHead: Boolean, numFlags: String): String? {
        val suffix = NUM_SUFFIX.find(word0)?.value
        var word = if (suffix != null) word0.substring(0, word0.length - suffix.length) else word0
        val result = ArrayList<String?>()
        if (word.startsWith("-")) {
            result.add(lookup("minus", null, null, null))
            word = word.substring(1)
        }
        fun extendNum(num: String, first: Boolean = true, escape: Boolean = false) {
            val text = if (escape) num else Num2Words.cardinal(BigInteger(num))
            val splits = NON_LOWER.split(text)
            for ((i, w) in splits.withIndex()) {
                if (w != "and" || '&' in numFlags) {
                    if (first && i == 0 && splits.size > 1 && w == "one" && 'a' in numFlags) {
                        result.add("ə")
                    } else {
                        result.add(lookup(w, null, if (w == "point") -2.0 else null, null))
                    }
                } else if (w == "and" && 'n' in numFlags && result.isNotEmpty()) {
                    result[result.size - 1] = result.last() + "ən"
                }
            }
        }
        if (En.isDigit(word) && suffix in En.ORDINALS) {
            extendNum(Num2Words.ordinal(BigInteger(word)), escape = true)
        } else if (result.isEmpty() && word.length == 4 && currency !in En.CURRENCIES && En.isDigit(word)) {
            extendNum(Num2Words.year(BigInteger(word)), escape = true)
        } else if (!isHead && '.' !in word) {
            val num = word.replace(",", "")
            if (num.isEmpty()) return null
            if (num[0] == '0' || num.length > 3) {
                for (n in num) extendNum(n.toString(), first = false)
            } else if (num.length == 3 && !num.endsWith("00")) {
                extendNum(num[0].toString())
                if (num[1] == '0') {
                    result.add(lookup("O", null, -2.0, null))
                    extendNum(num[2].toString(), first = false)
                } else {
                    extendNum(num.substring(1), first = false)
                }
            } else {
                extendNum(num)
            }
        } else if (word.count { it == '.' } > 1 || !isHead) {
            var first = true
            for (num in word.replace(",", "").split('.')) {
                if (num.isEmpty()) {
                    // pass
                } else if (num[0] == '0' || (num.length != 2 && num.substring(1).any { it != '0' })) {
                    for (n in num) extendNum(n.toString(), first = false)
                } else {
                    extendNum(num, first = first)
                }
                first = false
            }
        } else if (currency in En.CURRENCIES && isCurrency(word)) {
            val units = En.CURRENCIES.getValue(currency!!)
            var pairs = word.replace(",", "").split('.').zip(listOf(units.first, units.second))
                .map { (num, unit) -> (if (num.isEmpty()) BigInteger.ZERO else BigInteger(num)) to unit }
            if (pairs.size > 1) {
                if (pairs[1].first.signum() == 0) pairs = pairs.subList(0, 1)
                else if (pairs[0].first.signum() == 0) pairs = pairs.subList(1, pairs.size)
            }
            for ((i, p) in pairs.withIndex()) {
                val (num, unit) = p
                if (i > 0) result.add(lookup("and", null, null, null))
                extendNum(num.toString(), first = i == 0)
                result.add(
                    if (num.abs() != BigInteger.ONE && unit != "pence") stemS(unit + "s", null, null, null)
                    else lookup(unit, null, null, null)
                )
            }
        } else {
            val text: String = if (En.isDigit(word)) {
                Num2Words.cardinal(BigInteger(word))
            } else if ('.' !in word) {
                val n = BigInteger(word.replace(",", ""))
                if (suffix in En.ORDINALS) Num2Words.ordinal(n) else Num2Words.cardinal(n)
            } else {
                val w = word.replace(",", "")
                if (w[0] == '.') {
                    "point " + w.substring(1).map { Num2Words.cardinal((it - '0').toLong()) }.joinToString(" ")
                } else {
                    Num2Words.float(w)
                }
            }
            extendNum(text, escape = true)
        }
        if (result.isEmpty()) return null
        if (result.any { it == null }) return null // would raise in misaki
        val joined = result.joinToString(" ")
        return when (suffix) {
            "s", "'s" -> suffixS(joined)
            "ed", "'d" -> suffixEd(joined)
            "ing" -> suffixIng(joined)
            else -> joined
        }
    }

    private fun appendCurrency(ps: String, currency: String?): String {
        if (currency.isNullOrEmpty()) return ps
        val units = En.CURRENCIES[currency] ?: return ps
        val c = stemS(units.first + "s", null, null, null)
        return if (c != null) "$ps $c" else ps
    }

    /** Lexicon.__call__: returns phonemes or null. */
    internal fun invoke(tk: MToken, ctx: TokenContext): String? {
        var word = (tk.alias ?: tk.text).replace('‘', '\'').replace('’', '\'')
        word = Normalizer.normalize(word, Normalizer.Form.NFKC)
        word = buildString { for (c in word) append(numericIfNeeded(c)) }
        val stress: Double? = if (word == En.lower(word)) null else capStresses[if (word == En.upper(word)) 1 else 0]
        val ps = getWord(word, tk.tag, stress, ctx)
        if (ps != null) return En.applyStress(appendCurrency(ps, tk.currency), tk.stress)
        if (isNumber(word, tk.isHead)) {
            val n = try {
                getNumber(word, tk.currency, tk.isHead, tk.numFlags)
            } catch (e: NumberFormatException) {
                null
            } catch (e: IllegalArgumentException) {
                null
            }
            return En.applyStress(n, tk.stress)
        }
        return null
    }
}
