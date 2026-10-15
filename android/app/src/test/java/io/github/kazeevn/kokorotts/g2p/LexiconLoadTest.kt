package io.github.kazeevn.kokorotts.g2p

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.File

class LexiconLoadTest {
    @Test
    fun loadTimeAndBasicLookups() {
        for (british in listOf(false, true)) {
            val p = if (british) "gb" else "us"
            // Read the files into memory first so that only parsing is timed.
            val gold = File(TestSupport.assets, "${p}_gold.tsv").readBytes()
            val silver = File(TestSupport.assets, "${p}_silver.tsv").readBytes()
            var best = Long.MAX_VALUE
            var lex: Lexicon? = null
            repeat(5) {
                val t0 = System.nanoTime()
                lex = Lexicon.load(ByteArrayInputStream(gold), ByteArrayInputStream(silver), british)
                best = minOf(best, System.nanoTime() - t0)
            }
            println("[$p] lexicon load: ${best / 1_000_000.0} ms (best of 5), ${gold.size + silver.size} bytes")
            val g2p = EnglishG2P(lex!!, british, null)
            val (ps, toks) = g2p.phonemize("Hello, world!")
            println("[$p] $ps $toks")
            assertTrue(ps.isNotEmpty())
            assertEquals(4, toks.size)
        }
    }

    @Test
    fun offsetsPointIntoOriginalText() {
        val g2p = EnglishG2P(TestSupport.lexicon(false), false, null)
        val text = "  Mr. Smith paid \$12.99 for \"The well-known book\"... didn't he?"
        val (ps, toks) = g2p.phonemize(text)
        println(ps)
        for (t in toks) println("${t.start}-${t.end} '${text.substring(t.start, t.end)}' -> '${t.phonemes}' ws='${t.whitespace}'")
        assertEquals("Mr.", text.substring(toks[0].start, toks[0].end))
        assertTrue(toks.all { it.start in 0..text.length && it.end in it.start..text.length })
    }
}
