package io.github.kazeevn.kokorotts.g2p

/**
 * One output token: [start, end) are char offsets into the ORIGINAL input text, [phonemes] is
 * what this token contributes to the phoneme string (possibly empty, e.g. for "#" or a currency
 * sign whose name is read after the number), followed by [whitespace].
 */
data class G2PToken(val start: Int, val end: Int, val phonemes: String, val whitespace: String)

/**
 * Pure-Kotlin port of misaki 0.9.4 `misaki.en.G2P` (English G2P for Kokoro-82M).
 *
 * spaCy is replaced by [EnTokenizer] + [PosTagger] (rule-based). Everything after tokenization
 * (retokenize/subtokenize, the reversed-order lexicon pass with token context, merging, stress
 * resolution, number/currency reading, the espeak fallback hook and the final ɾ→T, ʔ→t mapping)
 * follows misaki.
 *
 * Not ported: misaki's markdown-link feature syntax `[word](/phonemes/)` (text is read
 * verbatim), timestamps, lookup ratings. Whitespace runs (incl. newlines) are normalized to a
 * single space in the phoneme string.
 *
 * @param fallback called for words the lexicon cannot handle, with the word's text; must return
 *   misaki-format phonemes (see [EspeakFallback]) or null. Without a fallback, or when it returns
 *   null, the word is rendered as [unk].
 * @param unk placeholder for unknown words (misaki uses "❓"; default "" drops them).
 */
class EnglishG2P(
    private val lexicon: Lexicon,
    british: Boolean,
    private val fallback: ((String) -> String?)?,
    private val unk: String = "",
) {
    init {
        require(british == lexicon.british) { "lexicon dialect does not match british=$british" }
    }

    /** Returns misaki's phoneme string for [text] and the per-token breakdown with offsets. */
    fun phonemize(text: String): Pair<String, List<G2PToken>> {
        val raw = EnTokenizer.tokenize(text)
        if (raw.isEmpty()) return "" to emptyList()
        PosTagger.tag(raw)
        val tokens = raw.map { MToken(it.text, it.tag, it.whitespace, it.start, it.end) }
        val words = retokenize(tokens)
        var ctx = TokenContext()
        for (wi in words.indices.reversed()) {
            val w = words[wi]
            if (w is MToken) {
                if (w.phonemes == null) w.phonemes = lexicon.invoke(w.copy(), ctx)
                if (w.phonemes == null && fallback != null) w.phonemes = fallback.invoke(w.text)
                ctx = tokenContext(ctx, w.phonemes, w)
                continue
            }
            @Suppress("UNCHECKED_CAST")
            val group = w as MutableList<MToken>
            var left = 0
            var right = group.size
            var shouldFallback = false
            while (left < right) {
                val sub = group.subList(left, right)
                val tk = if (sub.any { it.alias != null || it.phonemes != null }) null else mergeTokens(sub, null)
                val ps = if (tk == null) null else lexicon.invoke(tk, ctx)
                if (ps != null) {
                    group[left].phonemes = ps
                    for (x in group.subList(left + 1, right)) x.phonemes = ""
                    ctx = tokenContext(ctx, ps, tk!!)
                    right = left
                    left = 0
                } else if (left + 1 < right) {
                    left += 1
                } else {
                    right -= 1
                    val t = group[right]
                    if (t.phonemes == null) {
                        if (t.text.all { it in En.SUBTOKEN_JUNKS }) {
                            t.phonemes = ""
                        } else if (fallback != null) {
                            shouldFallback = true
                            break
                        }
                    }
                    left = 0
                }
            }
            if (shouldFallback) {
                val tk = mergeTokens(group, null)
                group[0].phonemes = fallback!!.invoke(tk.text)
                for (j in 1 until group.size) group[j].phonemes = ""
            } else {
                resolveTokens(group)
            }
        }
        val out = ArrayList<G2PToken>(words.size)
        val sb = StringBuilder()
        for (w in words) {
            @Suppress("UNCHECKED_CAST")
            val tk = if (w is MToken) w else mergeTokens(w as List<MToken>, unk)
            var ps = tk.phonemes ?: unk
            ps = ps.replace('ɾ', 'T').replace('ʔ', 't')
            sb.append(ps).append(tk.whitespace)
            out.add(G2PToken(tk.start, tk.end, ps, tk.whitespace))
        }
        return sb.toString() to out
    }

    companion object {
        // misaki's subtokenize regex; \d -> \p{Nd} to match Python's Unicode \d.
        private val SUBTOKEN_RE = Regex(
            "^['‘’]+|\\p{Lu}(?=\\p{Lu}\\p{Ll})|(?:^-)?(?:\\p{Nd}?[,.]?\\p{Nd})+|[-_]+|['‘’]{2,}|" +
                "\\p{L}*?(?:['‘’]\\p{L})*?\\p{Ll}(?=\\p{Lu})|\\p{L}+(?:['‘’]\\p{L})*|[^-_\\p{L}'‘’\\p{Nd}]|['‘’]+$",
        )

        internal fun mergeTokens(tokens: List<MToken>, unk: String?): MToken {
            val stresses = tokens.mapNotNull { it.stress }.toSet()
            val currencies = tokens.mapNotNull { it.currency }
            var phonemes: String? = null
            if (unk != null) {
                val sb = StringBuilder()
                for (tk in tokens) {
                    if (tk.prespace && sb.isNotEmpty() && !sb.last().isWhitespace() && !tk.phonemes.isNullOrEmpty()) sb.append(' ')
                    sb.append(tk.phonemes ?: unk)
                }
                phonemes = sb.toString()
            }
            val text = buildString {
                for (k in 0 until tokens.size - 1) append(tokens[k].text).append(tokens[k].whitespace)
                append(tokens.last().text)
            }
            // max() keeps the first maximal element, like Python.
            var best = tokens[0]
            var bestW = Int.MIN_VALUE
            for (tk in tokens) {
                var wgt = 0
                for (c in tk.text) wgt += if (c.toString() == c.toString().lowercase()) 1 else 2
                if (wgt > bestW) {
                    best = tk
                    bestW = wgt
                }
            }
            return MToken(
                text = text,
                tag = best.tag,
                whitespace = tokens.last().whitespace,
                start = tokens.first().start,
                end = tokens.last().end,
                phonemes = phonemes,
                isHead = tokens[0].isHead,
                alias = null,
                stress = if (stresses.size == 1) stresses.first() else null,
                currency = currencies.maxOrNull(),
                numFlags = tokens.flatMap { it.numFlags.toList() }.toSortedSet().joinToString(""),
                prespace = tokens[0].prespace,
            )
        }

        /**
         * Not in misaki: caps the number of subtokens resolved together. The group resolution is
         * cubic in the group size, so pathological tokens ("a-b-c-…" thousands long) would stall.
         */
        private const val MAX_GROUP_SIZE = 64

        private fun subtokenize(tk: MToken): List<MToken> = try {
            subtokenizeRegex(tk)
        } catch (e: StackOverflowError) {
            listOf(tk.copy().also { it.whitespace = "" }) // absurdly long token; keep it whole
        }

        private fun subtokenizeRegex(tk: MToken): List<MToken> =
            SUBTOKEN_RE.findAll(tk.text).map { m ->
                MToken(
                    text = m.value, tag = tk.tag, whitespace = "",
                    start = tk.start + m.range.first, end = tk.start + m.range.last + 1,
                    phonemes = tk.phonemes, isHead = true, alias = null, stress = tk.stress,
                    currency = null, numFlags = tk.numFlags, prespace = false,
                )
            }.toList()

        /** G2P.retokenize: returns a list whose items are MToken or MutableList<MToken>. */
        internal fun retokenize(tokens: List<MToken>): List<Any> {
            val words = ArrayList<Any>()
            var currency: String? = null
            for ((i, token) in tokens.withIndex()) {
                val tks = if (token.alias == null && token.phonemes == null) subtokenize(token) else listOf(token)
                if (tks.isEmpty()) continue // Python would raise IndexError here
                tks.last().whitespace = token.whitespace
                for ((j, tk) in tks.withIndex()) {
                    if (tk.alias != null || tk.phonemes != null) {
                        // pass
                    } else if (tk.tag == "$" && tk.text in En.CURRENCIES) {
                        currency = tk.text
                        tk.phonemes = ""
                    } else if (tk.tag == ":" && (tk.text == "-" || tk.text == "–")) {
                        tk.phonemes = "—"
                    } else if (tk.tag in En.PUNCT_TAGS && !tk.text.all { it.lowercaseChar() in 'a'..'z' }) {
                        tk.phonemes = En.PUNCT_TAG_PHONEMES[tk.tag] ?: tk.text.filter { it in En.PUNCTS }
                    } else if (currency != null) {
                        if (tk.tag != "CD") {
                            currency = null
                        } else if (j + 1 == tks.size && (i + 1 == tokens.size || tokens[i + 1].tag != "CD")) {
                            tk.currency = currency
                        }
                    } else if (j > 0 && j < tks.size - 1 && tk.text == "2" &&
                        (tks[j - 1].text.last().toString() + tks[j + 1].text.first()).let { En.isAlpha(it) }
                    ) {
                        tk.alias = "to"
                    }
                    val last = words.lastOrNull()
                    if (tk.alias != null || tk.phonemes != null) {
                        words.add(tk)
                    } else if (last is MutableList<*> && (last.last() as MToken).whitespace.isEmpty() && last.size < MAX_GROUP_SIZE) {
                        tk.isHead = false
                        @Suppress("UNCHECKED_CAST")
                        (last as MutableList<MToken>).add(tk)
                    } else {
                        words.add(if (tk.whitespace.isNotEmpty()) tk else mutableListOf(tk))
                    }
                }
            }
            return words.map { if (it is MutableList<*> && it.size == 1) it[0]!! else it }
        }

        internal fun tokenContext(ctx: TokenContext, ps: String?, token: MToken): TokenContext {
            var vowel = ctx.futureVowel
            if (ps != null) {
                for (c in ps) {
                    if (c in En.VOWELS || c in En.CONSONANTS || c in En.NON_QUOTE_PUNCTS) {
                        vowel = if (c in En.NON_QUOTE_PUNCTS) null else c in En.VOWELS
                        break
                    }
                }
            }
            val futureTo = token.text == "to" || token.text == "To" || (token.text == "TO" && (token.tag == "TO" || token.tag == "IN"))
            return TokenContext(vowel, futureTo)
        }

        internal fun resolveTokens(tokens: List<MToken>) {
            val text = buildString {
                for (k in 0 until tokens.size - 1) append(tokens[k].text).append(tokens[k].whitespace)
                append(tokens.last().text)
            }
            val classes = HashSet<Int>()
            for (c in text) {
                if (c in En.SUBTOKEN_JUNKS) continue
                classes.add(if (Character.isLetter(c)) 0 else if (c in '0'..'9') 1 else 2)
            }
            val prespace = ' ' in text || '/' in text || classes.size > 1
            for ((i, tk) in tokens.withIndex()) {
                if (tk.phonemes == null) {
                    if (i == tokens.size - 1 && tk.text.length == 1 && tk.text[0] in En.NON_QUOTE_PUNCTS) {
                        tk.phonemes = tk.text
                    } else if (tk.text.all { it in En.SUBTOKEN_JUNKS }) {
                        tk.phonemes = ""
                    }
                } else if (i > 0) {
                    tk.prespace = prespace
                }
            }
            if (prespace) return
            data class Idx(val primary: Boolean, val weight: Int, val i: Int)
            val indices = tokens.withIndex().filter { !it.value.phonemes.isNullOrEmpty() }
                .map { (i, tk) -> Idx(En.PRIMARY_STRESS in tk.phonemes!!, En.stressWeight(tk.phonemes), i) }
            if (indices.size == 2 && tokens[indices[0].i].text.length == 1) {
                val i = indices[1].i
                tokens[i].phonemes = En.applyStress(tokens[i].phonemes, -0.5)
                return
            } else if (indices.size < 2 || indices.count { it.primary } <= (indices.size + 1) / 2) {
                return
            }
            val sorted = indices.sortedWith(compareBy<Idx>({ it.primary }, { it.weight }, { it.i })).take(indices.size / 2)
            for (x in sorted) tokens[x.i].phonemes = En.applyStress(tokens[x.i].phonemes, -0.5)
        }
    }
}
