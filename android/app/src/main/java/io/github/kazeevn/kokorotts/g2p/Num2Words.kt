package io.github.kazeevn.kokorotts.g2p

import java.math.BigDecimal
import java.math.BigInteger

/**
 * Port of the English converter of Python `num2words` 0.5.14 (`Num2Word_EN`), limited to what
 * misaki uses: cardinal, ordinal, year and float. Output strings are identical to Python's,
 * including the "and"/comma/hyphen punctuation (misaki splits on non-letters afterwards).
 */
internal object Num2Words {
    /** (value, word) in the descending order of Python's `self.cards` OrderedDict. */
    private val cards: List<Pair<BigInteger, String>> = buildList {
        val lows = listOf("non", "oct", "sept", "sext", "quint", "quadr", "tr", "b", "m")
        val units = listOf("", "un", "duo", "tre", "quattuor", "quin", "sex", "sept", "octo", "novem")
        val tens = listOf(
            "dec", "vigint", "trigint", "quadragint", "quinquagint",
            "sexagint", "septuagint", "octogint", "nonagint",
        )
        val gen = tens.flatMap { t -> units.map { u -> u + t } }.reversed() + lows
        val high = listOf("cent") + gen
        var n = 3 + 3 * high.size
        for (word in high) {
            if (n <= 3) break
            add(BigInteger.TEN.pow(n) to word + "illion")
            n -= 3
        }
        val mid = listOf(
            1000 to "thousand", 100 to "hundred", 90 to "ninety", 80 to "eighty",
            70 to "seventy", 60 to "sixty", 50 to "fifty", 40 to "forty", 30 to "thirty",
        )
        for ((v, w) in mid) add(BigInteger.valueOf(v.toLong()) to w)
        val low = listOf(
            "twenty", "nineteen", "eighteen", "seventeen", "sixteen", "fifteen", "fourteen",
            "thirteen", "twelve", "eleven", "ten", "nine", "eight", "seven", "six", "five",
            "four", "three", "two", "one", "zero",
        )
        for ((i, w) in low.withIndex()) add(BigInteger.valueOf((low.size - 1 - i).toLong()) to w)
    }
    private val cardMap: Map<BigInteger, String> = cards.toMap()
    private val maxVal: BigInteger = cards.first().first.multiply(BigInteger.valueOf(1000))

    private val ords = mapOf(
        "one" to "first", "two" to "second", "three" to "third", "four" to "fourth",
        "five" to "fifth", "six" to "sixth", "seven" to "seventh", "eight" to "eighth",
        "nine" to "ninth", "ten" to "tenth", "eleven" to "eleventh", "twelve" to "twelfth",
    )

    private data class Pr(val text: String, val num: BigInteger)

    /** Python splitnum: returns a nested list of Pr / List. */
    private fun splitnum(value: BigInteger): List<Any> {
        for ((elem, word) in cards) {
            if (elem > value) continue
            val out = ArrayList<Any>()
            val div: BigInteger
            val mod: BigInteger
            if (value.signum() == 0) {
                div = BigInteger.ONE; mod = BigInteger.ZERO
            } else {
                val dm = value.divideAndRemainder(elem)
                div = dm[0]; mod = dm[1]
            }
            if (div == BigInteger.ONE) {
                out.add(Pr(cardMap.getValue(BigInteger.ONE), BigInteger.ONE))
            } else {
                if (div == value) return listOf(Pr(word.repeat(div.toInt()), div * elem)) // tallies; unreachable
                out.add(splitnum(div))
            }
            out.add(Pr(word, elem))
            if (mod.signum() != 0) out.add(splitnum(mod))
            return out
        }
        return emptyList()
    }

    private fun merge(l: Pr, r: Pr): Pr {
        val hundred = BigInteger.valueOf(100)
        return when {
            l.num == BigInteger.ONE && r.num < hundred -> Pr(r.text, r.num)
            hundred > l.num && l.num > r.num -> Pr("${l.text}-${r.text}", l.num + r.num)
            l.num >= hundred && hundred > r.num -> Pr("${l.text} and ${r.text}", l.num + r.num)
            r.num > l.num -> Pr("${l.text} ${r.text}", l.num * r.num)
            else -> Pr("${l.text}, ${r.text}", l.num + r.num)
        }
    }

    private fun clean(v: List<Any>): Any {
        var value = v
        var out: List<Any> = value
        while (value.size != 1) {
            val o = ArrayList<Any>()
            val left = value[0]
            val right = value[1]
            if (left is Pr && right is Pr) {
                o.add(merge(left, right))
                if (value.size > 2) o.add(value.subList(2, value.size).toList())
            } else {
                for (elem in value) {
                    if (elem is List<*>) {
                        @Suppress("UNCHECKED_CAST")
                        val e = elem as List<Any>
                        if (e.size == 1) o.add(e[0]) else o.add(clean(e))
                    } else {
                        o.add(elem)
                    }
                }
            }
            value = o
            out = o
        }
        return out[0]
    }

    fun cardinal(value: BigInteger): String {
        var v = value
        var prefix = ""
        if (v.signum() < 0) {
            v = v.negate()
            prefix = "minus "
        }
        require(v < maxVal) { "number too large: $value" }
        val r = clean(splitnum(v)) as Pr
        return prefix + r.text
    }

    fun cardinal(value: Long): String = cardinal(BigInteger.valueOf(value))

    fun ordinal(value: BigInteger): String {
        require(value.signum() >= 0)
        val outwords = cardinal(value).split(" ").toMutableList()
        val lastwords = outwords.last().split("-").toMutableList()
        var lastword = lastwords.last().lowercase()
        val o = ords[lastword]
        if (o != null) {
            lastword = o
        } else {
            if (lastword.endsWith("y")) lastword = lastword.dropLast(1) + "ie"
            lastword += "th"
        }
        lastwords[lastwords.size - 1] = lastword
        outwords[outwords.size - 1] = lastwords.joinToString("-")
        return outwords.joinToString(" ")
    }

    fun year(value: BigInteger): String {
        var v = value
        var suffix: String? = null
        if (v.signum() < 0) {
            v = v.negate()
            suffix = "BC"
        }
        val hundred = BigInteger.valueOf(100)
        val high = v.divide(hundred)
        val low = v.mod(hundred).toInt()
        val valText = if (high.signum() == 0 || (high.mod(BigInteger.TEN).signum() == 0 && low < 10) || high >= hundred) {
            cardinal(v)
        } else {
            val highText = cardinal(high)
            val lowText = when {
                low == 0 -> "hundred"
                low < 10 -> "oh-" + cardinal(low.toLong())
                else -> cardinal(low.toLong())
            }
            "$highText $lowText"
        }
        return if (suffix == null) valText else "$valText $suffix"
    }

    /**
     * `num2words(float(s))` for a decimal string `s` such as "3.14" (digits with at most one '.').
     * Python formats the float with repr() to get the precision, so trailing zeros vanish
     * ("0.50" -> "zero point five") and integral values read as integers ("2.0" -> "two").
     */
    fun float(s: String): String {
        var bd = BigDecimal(s)
        if (bd.precision() > 15) {
            // Beyond double precision: emulate Python's float rounding via the shortest repr.
            bd = BigDecimal(bd.toDouble().toString())
        }
        bd = bd.stripTrailingZeros()
        if (bd.scale() <= 0) return cardinal(bd.toBigInteger())
        val pre = bd.toBigInteger()
        val precision = bd.scale()
        val frac = bd.subtract(BigDecimal(pre)).abs().movePointRight(precision).toBigInteger().toString()
        val post = "0".repeat(maxOf(0, precision - frac.length)) + frac
        val out = ArrayList<String>()
        out.add(cardinal(pre))
        out.add("point")
        for (i in 0 until precision) out.add(cardinal((post[i] - '0').toLong()))
        return out.joinToString(" ")
    }
}
