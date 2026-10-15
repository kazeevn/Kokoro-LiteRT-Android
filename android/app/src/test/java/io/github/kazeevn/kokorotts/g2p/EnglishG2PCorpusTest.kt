package io.github.kazeevn.kokorotts.g2p

import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * Compares the Kotlin port against reference misaki output (convert/g2p_reference.py) on
 * convert/g2p_test/corpus.txt, using the host espeak-ng build as fallback. Prints agreement
 * statistics and diffs.
 */
class EnglishG2PCorpusTest {
    private class Ref(val text: String, val ps: String, val tokens: List<List<Any?>>, val spacy: List<List<Any?>>)

    private fun loadRefs(file: String, dialect: String): List<Ref> {
        val f = File(TestSupport.repoRoot, "convert/g2p_test/$file")
        return f.readLines().filter { it.isNotBlank() }.map { line ->
            @Suppress("UNCHECKED_CAST")
            val m = TestSupport.Json(line).parse() as Map<String, Any?>
            @Suppress("UNCHECKED_CAST")
            Ref(
                m["text"] as String, m[dialect] as String,
                m[dialect + "_tokens"] as List<List<Any?>>, m["spacy"] as List<List<Any?>>,
            )
        }
    }

    private fun run(british: Boolean, file: String): Pair<Double, Double> {
        assumeTrue("host espeak-ng not built", TestSupport.espeakAvailable)
        val dialect = if (british) "gb" else "us"
        val g2p = EnglishG2P(
            TestSupport.lexicon(british), british,
            EspeakFallback(british) { TestSupport.hostEspeak(it, british) },
            unk = "❓",
        )
        val refs = loadRefs(file, dialect)
        var sentOk = 0
        var words = 0
        var wordsOk = 0
        var tokMismatch = 0
        val wordDiffs = ArrayList<String>()
        for (ref in refs) {
            val (ps, toks) = g2p.phonemize(ref.text)
            if (ps == ref.ps) sentOk++
            // Expected token spans from misaki's tokens (text + whitespace concatenation).
            var pos = 0
            val mine = toks.associateBy { it.start to it.end }
            val diffs = ArrayList<String>()
            for (t in ref.tokens) {
                val text = t[0] as String
                val ws = t[1] as String
                val tps = t[2] as String?
                val start = ref.text.indexOf(text, pos).let { if (it < 0) pos else it }
                val end = start + text.length
                pos = end + ws.length
                if (text.none { it.isLetterOrDigit() }) continue
                words++
                val m = mine[start to end]
                if (m == null) tokMismatch++
                if (m != null && m.phonemes == (tps ?: "❓")) {
                    wordsOk++
                } else {
                    val got = m?.phonemes ?: toks.filter { it.end > start && it.start < end }.joinToString(" + ") { "${ref.text.substring(it.start, it.end)}=${it.phonemes}" }
                    diffs.add("    '$text' [${t[3]}]: misaki=$tps kotlin=$got")
                }
            }
            if (ps != ref.ps) {
                println("[$dialect] ${ref.text}")
                println("    misaki: ${ref.ps}")
                println("    kotlin: $ps")
                println("    spacy:  " + ref.spacy.joinToString(" ") { "${it[0]}/${it[1]}" })
                println("    mine:   " + EnTokenizer.tokenize(ref.text).also { PosTagger.tag(it) }.joinToString(" ") { "${it.text}/${it.tag}" })
                diffs.forEach { println(it) }
                wordDiffs.addAll(diffs)
            }
        }
        val sentRate = sentOk.toDouble() / refs.size
        val wordRate = wordsOk.toDouble() / words
        println("[$dialect] $file")
        println("[$dialect] sentences: $sentOk/${refs.size} = ${"%.1f".format(100 * sentRate)}%")
        println("[$dialect] words: $wordsOk/$words = ${"%.2f".format(100 * wordRate)}% (token-span mismatches: $tokMismatch)")
        return sentRate to wordRate
    }

    // reference.jsonl (corpus.txt) and holdout_reference.jsonl (holdout.txt) were used while
    // developing the tagger heuristics; holdout2_reference.jsonl (holdout2.txt) was not.
    @Test
    fun usCorpus() {
        val (_, w) = run(british = false, "reference.jsonl")
        assertTrue("US word agreement $w", w >= 0.97)
    }

    @Test
    fun gbCorpus() {
        val (_, w) = run(british = true, "reference.jsonl")
        assertTrue("GB word agreement $w", w >= 0.97)
    }

    @Test
    fun usHoldout() {
        val (_, w) = run(british = false, "holdout_reference.jsonl")
        assertTrue("US holdout word agreement $w", w >= 0.95)
    }

    @Test
    fun usHoldout2() {
        val (_, w) = run(british = false, "holdout2_reference.jsonl")
        assertTrue("US holdout2 word agreement $w", w >= 0.95)
    }

    @Test
    fun gbHoldout2() {
        val (_, w) = run(british = true, "holdout2_reference.jsonl")
        assertTrue("GB holdout2 word agreement $w", w >= 0.95)
    }

    @Test
    fun gbHoldout() {
        val (_, w) = run(british = true, "holdout_reference.jsonl")
        assertTrue("GB holdout word agreement $w", w >= 0.95)
    }
}
