buildscript {
    dependencies {
        // LiteRT 2.3.0 ships Kotlin 2.4 metadata; AGP's built-in Kotlin defaults to 2.2.
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:2.4.21")
    }
}

plugins {
    id("com.android.application") version "9.4.1" apply false
}
