package io.github.kazeevn.kokorotts.g2p

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.random.Random

/** The G2P must never throw and offsets must stay consistent, whatever the input. */
class RobustnessTest {
    private val g2p = EnglishG2P(TestSupport.lexicon(false), false, null)
    private val g2pFallback = EnglishG2P(TestSupport.lexicon(false), false, { "fəlbˈæk" })

    private fun check(text: String) {
        for (g in listOf(g2p, g2pFallback)) {
            val (ps, toks) = g.phonemize(text)
            assertEquals(ps, toks.joinToString("") { it.phonemes + it.whitespace })
            var last = 0
            for (t in toks) {
                assertTrue("bad span $t in '$text'", t.start >= last && t.end >= t.start && t.end <= text.length)
                last = t.end
            }
        }
    }

    @Test
    fun edgeCases() {
        val cases = listOf(
            "", " ", "\n\n", "...", "!!!", "\"\"", "''", "'", "-", "--", "—", "$", "$$", "£.", "€", "#", "%", "&",
            "-5", "--5", "5-", "1e10", "1,2,3", "1.2.3.4", ".5", "5.", "$.50", "$1,000,000,000,000,000,000,000",
            "123456789012345678901234567890", "0.000000000000000000001", "3.141592653589793238462643",
            "1st 2nd 3rd 4th 11th 12th 13th 21st 101st 1000000th", "1990s 80s '90s", "12:30:45", "2020-01-01",
            "a-b-c-d-e-f-g", "x".repeat(5000), "a" + "-b".repeat(500), "a'".repeat(20000), "1-".repeat(2000), "-".repeat(10000),
            "Ab".repeat(3000), "1".repeat(3000), "\$" + "9".repeat(400) + ".99", "😀 hello 👋 world 🎉", "日本語のテキスト",
            "Привет, мир!", "ﬁnancial ﬂow", "éclair", "naïve café", "A.B.C.D.E.F.", "U.S.A.", "mr. MR. Mr",
            "I'm I'M i'm", "don't DON'T Don’t", "rock'n'roll", "''quoted''", "‘single’ “double”", "(((nested)))",
            "[x](/ˈɛks/)", "http://example.com/a?b=c#d", "user@mail.example.org", "C++ C# F#", "50%+", "a&b",
            "TO BE OR NOT TO BE", "The THE", "AM PM am pm", "2 b or not 2 b", "abc2def", "Ⅻ ⅷ ½ ¾ ² ³",
            "\u0000\u0001\u0002", "\t\ttabbed\ttext\t", "line1\nline2\r\nline3",
        )
        for (c in cases) check(c)
    }

    @Test
    fun randomStrings() {
        val rnd = Random(42)
        val alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?'\"-–—()[]{}\$£€%&@#/\\_*+=<>’‘“”…éüñßøЖ中😀\n"
        repeat(3000) {
            val len = rnd.nextInt(1, 60)
            val s = buildString { repeat(len) { append(alphabet[rnd.nextInt(alphabet.length)]) } }
            check(s)
        }
    }

    @Test
    fun throughput() {
        val text = "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age " +
            "of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of " +
            "Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair."
        repeat(50) { g2p.phonemize(text) }
        val n = 500
        val t0 = System.nanoTime()
        repeat(n) { g2p.phonemize(text) }
        val ms = (System.nanoTime() - t0) / 1e6 / n
        println("phonemize: %.3f ms per %d-char sentence".format(ms, text.length))
    }
}
