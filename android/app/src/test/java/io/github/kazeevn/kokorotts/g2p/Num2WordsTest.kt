package io.github.kazeevn.kokorotts.g2p

import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File
import java.math.BigInteger

class Num2WordsTest {
    @Test
    fun matchesPythonNum2words() {
        val f = File(TestSupport.repoRoot, "convert/g2p_test/num2words.tsv")
        val failures = ArrayList<String>()
        var n = 0
        f.forEachLine { line ->
            if (line.isBlank()) return@forEachLine
            val (kind, input, expected) = line.split('\t')
            val actual = when (kind) {
                "cardinal" -> Num2Words.cardinal(BigInteger(input))
                "ordinal" -> Num2Words.ordinal(BigInteger(input))
                "year" -> Num2Words.year(BigInteger(input))
                "float" -> Num2Words.float(input)
                else -> error(kind)
            }
            n++
            if (actual != expected) failures.add("$kind($input): expected '$expected', got '$actual'")
        }
        failures.take(30).forEach { println(it) }
        assertEquals("num2words mismatches out of $n", 0, failures.size)
    }
}
