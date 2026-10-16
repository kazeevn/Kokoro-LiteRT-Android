package io.github.kazeevn.kokorotts.engine

import java.io.File

/**
 * DMA-BUF memory this process holds (the EdgeTPU's model and tensor buffers live there). Process
 * RSS/PSS do not include it; `dumpsys meminfo -S` shows only the system-wide total.
 */
object DmaBufStats {
    /** Total size of the distinct DMA-BUFs this process has open (fds) or mapped, in bytes. */
    fun bytes(): Long {
        val sizes = HashMap<Long, Long>() // inode -> size
        File("/proc/self/fd").listFiles()?.forEach { fd ->
            val target = try { fd.canonicalPath } catch (_: Exception) { return@forEach }
            if (!target.contains("dmabuf")) return@forEach
            var ino = -1L
            var size = -1L
            try {
                File("/proc/self/fdinfo/${fd.name}").forEachLine { line ->
                    when {
                        line.startsWith("ino:") -> ino = line.substringAfter(':').trim().toLong()
                        line.startsWith("size:") -> size = line.substringAfter(':').trim().toLong()
                    }
                }
            } catch (_: Exception) {
            }
            if (ino >= 0 && size >= 0) sizes[ino] = size
        }
        // Mappings whose fd was closed: /proc/self/maps lines like "a-b rw-s 0 00:0b 1234 /dmabuf:name".
        File("/proc/self/maps").forEachLine { line ->
            if (!line.contains("dmabuf")) return@forEachLine
            val f = line.split(Regex("\\s+"))
            val ino = f.getOrNull(4)?.toLongOrNull() ?: return@forEachLine
            if (ino in sizes) return@forEachLine
            val (a, b) = f[0].split('-').map { it.toLong(16) }
            sizes[ino] = b - a
        }
        return sizes.values.sum()
    }

    fun mb(): Long = bytes() / (1024 * 1024)
}
