// Throwaway feasibility probe: LiteRT 1.x Interpreter + NNAPI delegate on the Pixel EdgeTPU.
plugins {
    id("com.android.application")
}

android {
    namespace = "io.github.kazeevn.nnprobe"
    compileSdk = 37
    defaultConfig {
        applicationId = "io.github.kazeevn.nnprobe"
        minSdk = 35
        targetSdk = 37
        ndk { abiFilters += "arm64-v8a" }
    }
    androidResources { noCompress += listOf("tflite") }
}

dependencies {
    implementation("com.google.ai.edge.litert:litert:1.4.2")
}
