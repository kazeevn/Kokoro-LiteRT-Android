package io.github.kazeevn.kokorotts.g2p

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.util.concurrent.TimeUnit

class EspeakFallbackTest {
    @Test
    fun convertMatchesMisakiExamples() {
        // Expected values computed with misaki.espeak.EspeakFallback's mapping.
        assertEquals("wˈIldli", EspeakFallback.convert("wˈa^ɪldli", false))
        assertEquals("ʤˈʌʤ", EspeakFallback.convert("d^ʒˈʌd^ʒ", false))
        assertEquals("ʧˈɑklət", EspeakFallback.convert("t^ʃˈɑːklət", false))
        assertEquals("ɡˈO", EspeakFallback.convert("ɡˈo^ʊ", false))
        assertEquals("ɡˈQ", EspeakFallback.convert("ɡˈə^ʊ", true))
        assertEquals("tˈɛn:θˈɜɹTi", EspeakFallback.convert("tˈɛn:θˈɜːɾi", false))
        assertEquals("bˈʌdᵊld", EspeakFallback.convert("bˈʌdə^ld", false))
    }

    private fun espeakLines(words: List<String>, british: Boolean, tie: Boolean): List<String> {
        val f = File.createTempFile("words", ".txt")
        f.writeText(words.joinToString("\n") + "\n")
        val cmd = mutableListOf(File(TestSupport.espeakHost, "src/espeak-ng").path, "-q", "--ipa", "-v", if (british) "en-gb" else "en-us", "-f", f.path)
        if (tie) cmd.add(3, "--tie=^")
        val pb = ProcessBuilder(cmd)
        pb.environment()["ESPEAK_DATA_PATH"] = TestSupport.espeakHost.path
        val p = pb.start()
        val out = p.inputStream.bufferedReader(Charsets.UTF_8).readLines()
        p.waitFor(60, TimeUnit.SECONDS)
        f.delete()
        // espeak joins lines into clauses; every (alphabetic) word yields exactly one IPA word.
        return out.flatMap { it.trim().split(Regex("\\s+")) }.filter { it.isNotEmpty() }
    }

    /** How often the untied-IPA mode (plain mode 0x02) differs from the tied mode misaki uses. */
    @Test
    fun untiedModeAgreement() {
        assumeTrue(TestSupport.espeakAvailable)
        for (british in listOf(false, true)) {
            val p = if (british) "gb" else "us"
            val words = File(TestSupport.assets, "${p}_silver.tsv").readLines()
                .map { it.substringBefore('\t') }.filter { w -> w.all { it in 'a'..'z' } }
                .filterIndexed { i, _ -> i % 40 == 0 }
            val tied = espeakLines(words, british, true)
            val untied = espeakLines(words, british, false)
            assumeTrue(tied.size == words.size && untied.size == words.size)
            var same = 0
            for (i in words.indices) {
                if (EspeakFallback.convert(tied[i], british) == EspeakFallback.convert(EspeakFallback.retie(untied[i]), british)) same++
            }
            val rate = same.toDouble() / words.size
            println("[$p] untied espeak mode agrees with tied mode on $same/${words.size} = ${"%.1f".format(100 * rate)}% of OOV words")
            assertTrue(rate > 0.9)
        }
    }
}
