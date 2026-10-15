package io.github.kazeevn.kokorotts.g2p

/**
 * Heuristic Penn-Treebank-style tagger replacing spaCy's statistical tagger.
 *
 * misaki only looks at tags in a few places, so the heuristics target exactly those:
 *  - punctuation tags (".", ",", ":", "``", "''", "-LRB-", "-RRB-", "$", "NFP", "HYPH") decide
 *    which tokens become punctuation phonemes and how quotes are rendered;
 *  - "CD" decides currency attachment ("$3 billion" -> "three billion dollars");
 *  - "DT" for a/an/the/that, "PRP" for I, "IN"/"RB" for in/by, "TO", "VBD" for used,
 *    "NN*" for am/AM (spelled out), "NNP" for proper nouns and ALL-CAPS words;
 *  - NOUN vs VERB vs ADJ for heteronyms (record, present, live, read, wound, ...):
 *    a word is a verb after "to"/modals/do-support/subject pronouns or when it starts an
 *    imperative sentence, a noun after determiners/possessives/adjectives/prepositions,
 *    an adjective after forms of "be";
 *  - everything else defaults to NN / NNS / VBG / VBD / RB by suffix.
 */
internal object PosTagger {
    private val DT = setOf(
        "the", "a", "an", "this", "these", "those", "every", "each", "some", "any", "no", "another",
        "either", "neither", "all", "both",
    )
    private val PRP = setOf(
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "us", "them", "myself", "yourself",
        "himself", "herself", "itself", "ourselves", "themselves", "'em", "ya", "y'",
    )
    private val SUBJECT_PRP = setOf("i", "you", "we", "they", "he", "she", "it", "who")
    private val THIRD_SG = setOf("he", "she", "it", "who")
    private val PRP_POSS = setOf("my", "your", "his", "her", "its", "our", "their")
    private val IN = setOf(
        "of", "in", "on", "at", "by", "for", "with", "from", "into", "onto", "upon", "about", "above",
        "across", "after", "against", "along", "among", "around", "before", "behind", "below",
        "beneath", "beside", "besides", "between", "beyond", "during", "except", "inside", "near",
        "off", "outside", "over", "since", "through", "throughout", "till", "toward", "towards",
        "under", "underneath", "until", "unto", "via", "within", "without", "because", "if",
        "whether", "while", "although", "though", "as", "than", "like", "per", "vs", "vs.", "unlike",
        "despite", "whereas", "unless", "out",
    )
    private val CC = setOf("and", "or", "but", "nor", "&", "plus")
    private val MD = setOf(
        "will", "would", "shall", "should", "can", "could", "may", "might", "must", "ca", "wo", "sha",
        "'ll", "'d", "’ll", "’d", "cannot",
    )
    private val WH = mapOf(
        "which" to "WDT", "whose" to "WP$", "who" to "WP", "whom" to "WP", "what" to "WP",
        "where" to "WRB", "when" to "WRB", "why" to "WRB", "how" to "WRB",
    )
    private val RB = setOf(
        "not", "n't", "n’t", "never", "very", "so", "too", "also", "just", "only", "always", "often",
        "already", "still", "even", "now", "then", "here", "again", "ever", "soon", "almost", "quite",
        "rather", "really", "yet", "once", "twice", "away", "back", "maybe", "perhaps", "else",
        "instead", "together", "anyway", "however", "therefore", "thus", "else", "later", "today",
        "tomorrow", "yesterday", "tonight", "nt",
    )
    private val UH = setOf(
        "oh", "ah", "uh", "um", "hmm", "wow", "yes", "yeah", "okay", "ok", "hello", "hi", "hey",
        "nah", "shh", "oops", "ugh", "gosh", "alas", "please", "thanks",
    )
    private val BE = mapOf(
        "be" to "VB", "am" to "VBP", "are" to "VBP", "is" to "VBZ", "was" to "VBD", "were" to "VBD",
        "been" to "VBN", "being" to "VBG", "'m" to "VBP", "'re" to "VBP", "ai" to "VBP",
        "’m" to "VBP", "’re" to "VBP",
    )
    private val AUX = mapOf(
        "have" to "VBP", "has" to "VBZ", "had" to "VBD", "'ve" to "VBP", "’ve" to "VBP",
        "do" to "VBP", "does" to "VBZ", "did" to "VBD", "having" to "VBG", "doing" to "VBG",
    )
    private val DO = setOf("do", "does", "did")
    private val HAVE = setOf("have", "has", "had", "'ve", "’ve", "having")
    private val NUMBER_WORDS = setOf(
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
        "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
        "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
        "thousand", "million", "billion", "trillion",
    )
    private val SAY_VERBS = setOf(
        "said", "say", "says", "think", "thinks", "thought", "know", "knew", "knows", "believe",
        "believed", "hope", "hoped", "feel", "felt", "realize", "realized", "sure", "so", "such",
        "now", "noticed", "saw", "see", "claimed", "told", "tell", "wish", "suppose", "means",
        "meant", "show", "shows", "showed", "fact", "seems", "seemed",
    )
    private val INTENSIFIERS = setOf("very", "so", "too", "more", "most", "quite", "rather", "really", "extremely", "less", "least")
    private val PAST_HETERONYMS = setOf("read", "wound", "reread", "dove")
    /** Abbreviations whose trailing "." spaCy's tagger usually tags as part of the word. */
    private val ABBREVIATIONS = setOf(
        "etc", "approx", "dept", "est", "fig", "figs", "vol", "vols", "ch", "pp", "cf", "al", "no", "nos",
        "sr", "sra", "srta", "mme", "mlle", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
        "oct", "nov", "dec", "mon", "tue", "tues", "wed", "thu", "thurs", "fri", "sat", "sun", "ave", "blvd",
        "rd", "ln", "hwy", "ft", "lt", "col", "capt", "cmdr", "sgt", "cpl", "pvt", "gen", "esp", "incl",
    )
    private val NOT_RB_LY = setOf("family", "only", "early", "daily", "fly", "july", "italy", "reply", "supply", "ally", "rely", "apply", "belly", "bully", "jelly", "lily", "holy", "ugly", "silly", "lovely", "friendly", "likely", "lonely", "elderly", "jolly", "folly", "hilly", "curly", "burly", "comply", "sly", "assembly", "anomaly", "monopoly", "rally", "tally", "sally", "molly", "polly", "kelly", "emily", "billy", "willy")

    private val QUOTE_BRACKET_TAGS = setOf("``", "''", "-LRB-", "-RRB-")
    private val CLAUSE_END_TAGS = setOf(".", ",", ":", "''", "-RRB-")
    private val CONTRACTION_SUFFIXES = setOf("'s", "’s", "'ll", "’ll", "'d", "’d", "'re", "’re")
    private val NOUN_PREMODIFIER_TAGS = setOf("DT", "PRP$", "JJ", "POS", "CD", "IN")
    private val NUMBER_ORDINAL_RE = Regex("[0-9,]*[0-9](st|nd|rd|th)")
    private val NUMBER_PLURAL_RE = Regex("[0-9,.]*[0-9]'?s")

    private fun isWordChar(c: Char) = Character.isLetterOrDigit(c)
    private fun hasLetter(s: String) = s.any { Character.isLetter(it) }
    private fun hasDigit(s: String) = s.any { it in '0'..'9' }
    private fun isAllCaps(s: String) = hasLetter(s) && s.none { Character.isLowerCase(it) } && s.count { Character.isLetter(it) } >= 2
    private fun isCapitalized(s: String) = s.isNotEmpty() && Character.isUpperCase(s[0])

    private fun punctTag(tokens: List<RawToken>, i: Int): String? {
        val t = tokens[i].text
        if (t.any { isWordChar(it) }) return null
        val prevAttached = i > 0 && tokens[i - 1].whitespace.isEmpty()
        val nextAttached = i + 1 < tokens.size && tokens[i].whitespace.isEmpty()
        val prevText = if (i > 0) tokens[i - 1].text else ""
        val nextText = if (i + 1 < tokens.size) tokens[i + 1].text else ""
        return when {
            t.all { it in ".!?" } && !(t.length > 1 && t.all { it == '.' }) -> "."
            t.all { it == '.' } || t == "…" || t.all { it == '…' } -> ":"
            t == "," -> ","
            t == ":" || t == ";" -> ":"
            t == "-" || t == "–" || t == "—" || t == "--" || t == "---" || t == "——" -> {
                val between = prevAttached && nextAttached
                when {
                    between && t == "-" && prevText.lastOrNull()?.isDigit() == true && nextText.firstOrNull()?.isDigit() == true -> "SYM"
                    between && t == "-" && hasLetter(prevText) && hasLetter(nextText) -> "HYPH"
                    else -> ":"
                }
            }
            t == "“" || t == "‘" || t == "``" || t == "`" || t == "«" -> "``"
            t == "”" || t == "''" || t == "´" || t == "»" -> "''"
            t == "\"" || t == "'" || t == "’" -> {
                val opening = (i == 0 || !prevAttached || tokens[i - 1].tag == "``" || tokens[i - 1].tag == "-LRB-") && nextAttached
                if (t != "\"" && prevAttached && !opening && prevText.lastOrNull()?.lowercaseChar() == 's' && !nextAttached) {
                    "POS"
                } else if (opening) "``" else "''"
            }
            t == "(" || t == "[" || t == "{" -> "-LRB-"
            t == ")" || t == "]" || t == "}" -> "-RRB-"
            t in En.CURRENCIES || t == "#" || t.all { Character.getType(it) == Character.CURRENCY_SYMBOL.toInt() } -> "$"
            t == "%" -> "NN"
            t == "&" -> "CC"
            t == "@" -> "IN"
            t.all { it in "*•·~_" } -> "NFP"
            t.all { Character.getType(it) == Character.OTHER_SYMBOL.toInt() && it != '°' } -> "NFP"
            else -> "SYM"
        }
    }

    private fun numberTag(t: String): String? {
        if (!hasDigit(t)) return null
        val lower = t.lowercase()
        return when {
            NUMBER_ORDINAL_RE.matches(lower) -> "JJ"
            NUMBER_PLURAL_RE.matches(lower) -> "NNS"
            else -> "CD"
        }
    }

    /** Tags that depend only on the (lowercased) word itself; null for open-class words. */
    private fun closedTag(lw: String): String? = when {
        lw in DT -> "DT"
        lw == "to" -> "TO"
        lw in PRP -> "PRP"
        lw in PRP_POSS -> "PRP$"
        lw in MD -> "MD"
        lw in BE -> BE[lw]
        lw in AUX -> AUX[lw]
        lw in WH -> WH[lw]
        lw in CC -> "CC"
        lw in IN -> "IN"
        lw in RB -> "RB"
        lw in UH -> "UH"
        lw in NUMBER_WORDS -> "CD"
        lw == "there" -> "EX"
        lw == "na" || lw == "ta" -> "TO"
        lw == "gon" -> "VBG"
        lw == "used" -> "VBD"
        else -> null
    }

    private fun hasLowercaseNeighbor(tokens: List<RawToken>, i: Int): Boolean {
        for (j in listOf(i - 1, i + 1)) {
            if (j < 0 || j >= tokens.size) continue
            val t = tokens[j].text
            if (t.any { Character.isLowerCase(it) }) return true
        }
        return false
    }

    fun tag(tokens: List<RawToken>) {
        val n = tokens.size
        val open = BooleanArray(n)
        val sentStart = BooleanArray(n)
        val preTagged = BooleanArray(n)
        var atStart = true
        // Pass 1: punctuation, numbers, closed-class words.
        for (i in 0 until n) {
            val tk = tokens[i]
            val t = tk.text
            val lw = t.lowercase()
            sentStart[i] = atStart
            var p = punctTag(tokens, i)
            if (p == "." && t == "." && i > 0 && tokens[i - 1].whitespace.isEmpty() &&
                tokens[i - 1].text.lowercase() in ABBREVIATIONS &&
                (i + 1 >= n || tokens[i + 1].text in setOf(",", ";", ":") || tokens[i + 1].text.firstOrNull()?.isLowerCase() == true)
            ) {
                p = tokens[i - 1].tag.ifEmpty { "NN" } // part of the abbreviation, not a sentence end
            }
            if (preTagged[i]) continue
            if ((t == "'" || t == "’") && i + 2 < n && tokens[i + 1].text.lowercase() == "n" &&
                tokens[i + 2].text == t && tk.whitespace.isEmpty() && tokens[i + 1].whitespace.isEmpty()
            ) {
                // rock 'n' roll: spaCy tags all three as CC, so misaki reads them as one word.
                for (k in i..i + 2) {
                    tokens[k].tag = "CC"
                    preTagged[k] = true
                }
                atStart = false
                continue
            }
            if (p != null) {
                tk.tag = p
                if (p == ".") atStart = true
                continue
            }
            atStart = false
            val num = numberTag(t)
            if (num != null) {
                tk.tag = num
                continue
            }
            val isCaps = isAllCaps(t)
            val closed = closedTag(lw)
            tk.tag = when {
                lw == "'s" || lw == "’s" -> {
                    val prev = if (i > 0) tokens[i - 1].text.lowercase() else ""
                    when (prev) {
                        "let" -> "PRP"
                        "he", "she", "it", "that", "what", "who", "where", "there", "here", "how", "this" -> "VBZ"
                        else -> "POS"
                    }
                }
                t == "I" -> {
                    val prevNNP = i > 0 && !sentStart[i] && isCapitalized(tokens[i - 1].text) && tokens[i - 1].tag == "NNP"
                    val nextEnd = i + 1 >= n || punctTag(tokens, i + 1) != null
                    if (prevNNP && nextEnd) "NNP" else "PRP"
                }
                // An all-caps word among lowercase words is an acronym/emphasis ("The THE"), but in
                // an all-caps sentence ("IN THE END") closed-class words keep their tags.
                isCaps && (closed == null || hasLowercaseNeighbor(tokens, i)) -> "NNP"
                closed != null -> closed
                !sentStart[i] && isCapitalized(t) -> "NNP"
                t.any { Character.isUpperCase(it) } && t.drop(1).any { Character.isUpperCase(it) } -> "NNP" // iPhone, McDonald
                else -> {
                    open[i] = true
                    ""
                }
            }
        }
        fun prevWordIdx(i: Int): Int {
            var j = i - 1
            while (j >= 0 && tokens[j].tag in QUOTE_BRACKET_TAGS) j--
            return j
        }
        fun isEnd(i: Int) = i >= n || tokens[i].tag in CLAUSE_END_TAGS
        // Pass 2: open-class words, left to right.
        for (i in 0 until n) {
            if (!open[i]) continue
            val tk = tokens[i]
            val lw = tk.text.lowercase()
            // Look through adverbs: "have already read", "do n't tear", "will never record".
            var pi = prevWordIdx(i)
            while (pi >= 0 && tokens[pi].tag == "RB") pi = prevWordIdx(pi)
            val prev = if (pi >= 0) tokens[pi] else null
            val prevLw = prev?.text?.lowercase() ?: ""
            val prevTag = prev?.tag ?: ""
            val pi2 = if (pi >= 0) prevWordIdx(pi) else -1
            val prev2Lw = if (pi2 >= 0) tokens[pi2].text.lowercase() else ""
            val prev2Tag = if (pi2 >= 0) tokens[pi2].tag else ""
            val next = if (i + 1 < n) tokens[i + 1] else null
            val nextTag = next?.tag ?: ""
            val endsS = lw.length > 2 && lw.endsWith("s") && !lw.endsWith("ss") && !lw.endsWith("us") && !lw.endsWith("is")
            val plural = if (endsS) "NNS" else "NN"
            tk.tag = when {
                // Verb contexts.
                prevTag == "TO" || prevTag == "MD" || prevLw in DO ||
                    ((prevLw == "n't" || prevLw == "n’t" || prevLw == "not" || prevLw == "never") && (prev2Lw in DO || prev2Lw in MD)) ||
                    (prevLw == "please" || (prevLw == "'s" && prev2Lw == "let")) -> if (lw.endsWith("ing")) "VBG" else "VB"
                prevLw in HAVE || (prevLw == "'d" && lw.endsWith("ed")) -> "VBN"
                // Question inversion: "can you read", "did they record".
                prevTag == "PRP" && prevLw in SUBJECT_PRP && (prev2Tag == "MD" || prev2Lw in DO) -> if (lw.endsWith("ing")) "VBG" else "VB"
                prevTag == "PRP" && prevLw in SUBJECT_PRP -> when {
                    lw.endsWith("ing") -> "VBG"
                    lw.endsWith("ed") -> "VBD"
                    prevLw in THIRD_SG -> if (endsS) "VBZ" else "VBD"
                    else -> "VBP"
                }
                prevTag == "WP" && prevLw == "who" -> if (endsS) "VBZ" else "VBD"
                sentStart[i] && (nextTag == "DT" || nextTag == "PRP$" || (nextTag == "PRP" && next?.text?.lowercase() !in SUBJECT_PRP) || nextTag == "CD") ->
                    if (lw.endsWith("ing")) "VBG" else "VB"
                lw in PAST_HETERONYMS && prevTag.startsWith("NN") &&
                    (nextTag == "IN" || nextTag == "DT" || nextTag == "PRP" || nextTag == "RB" || nextTag == "PRP$" || nextTag == "CD") -> "VBD"
                lw in PAST_HETERONYMS && prevLw in BE -> "VBN"
                // Adjective / participle after "be".
                prevTag in BE.values && prevLw in BE -> when {
                    lw.endsWith("ing") -> "VBG"
                    lw.endsWith("ed") -> "VBN"
                    else -> "JJ"
                }
                prevTag == "RB" && prevLw in INTENSIFIERS -> "JJ"
                lw.length > 4 && lw.endsWith("ly") && lw !in NOT_RB_LY -> "RB"
                endsS && (nextTag == "DT" || nextTag == "PRP$") && prevTag !in NOUN_PREMODIFIER_TAGS -> "VBZ"
                // Noun contexts.
                prevTag in NOUN_PREMODIFIER_TAGS -> plural
                lw.endsWith("ing") -> "VBG"
                lw.endsWith("ed") && lw.length > 3 -> if (prevTag.startsWith("NN") || prevTag == "PRP") "VBD" else "VBN"
                prevTag.startsWith("NN") && endsS && (nextTag == "DT" || nextTag == "PRP$" || nextTag == "IN") -> "VBZ"
                (prevTag == "VBD" || prevTag == "VBZ" || prevTag == "VBP") && isEnd(i + 1) -> "RB" // went home, glittered overhead
                else -> plural
            }
        }
        // Pass 3: context-dependent closed-class words.
        for (i in 0 until n) {
            val tk = tokens[i]
            val lw = tk.text.lowercase()
            val pi = prevWordIdx(i)
            val prev = if (pi >= 0) tokens[pi] else null
            val next = if (i + 1 < n) tokens[i + 1] else null
            when (lw) {
                "that" -> {
                    val nextTag = next?.tag ?: "."
                    val nextLw = next?.text?.lowercase() ?: ""
                    tk.tag = when {
                        isEnd(i + 1) -> "DT"
                        nextLw in CONTRACTION_SUFFIXES -> "DT" // that's, that'll
                        prev != null && prev.tag.startsWith("NN") && !sentStart[i] -> if (nextTag.startsWith("VB") || nextTag == "MD") "WDT" else "IN"
                        prev != null && prev.text.lowercase() in SAY_VERBS -> "IN"
                        (nextTag == "PRP" && nextLw in SUBJECT_PRP) || nextTag == "DT" || nextTag == "PRP$" || nextTag == "NNP" || nextTag == "EX" -> "IN"
                        else -> "DT"
                    }
                }
                "in" -> if (isEnd(i + 1)) tk.tag = "RB"
                "by" -> if (isEnd(i + 1)) tk.tag = "RB"
                "am" -> if (prev != null && prev.tag == "CD") tk.tag = "NN"
                "a" -> {
                    if (tk.text == "A" && (prev?.tag == "DT" || (prev != null && !sentStart[i] && prev.tag.startsWith("NN")) || isEnd(i + 1))) tk.tag = "NN"
                    else if (tk.text == "a" && isEnd(i + 1) && prev != null) tk.tag = "NN"
                }
            }
        }
    }
}
