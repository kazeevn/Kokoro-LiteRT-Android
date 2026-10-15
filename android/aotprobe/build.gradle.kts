// Throwaway probe: Tensor SDK AOT-compiled models via LiteRT 2.x CompiledModel on the Pixel EdgeTPU.
plugins {
    id("com.android.application")
}

android {
    namespace = "io.github.kazeevn.aotprobe"
    compileSdk = 37
    defaultConfig {
        applicationId = "io.github.kazeevn.aotprobe"
        minSdk = 35
        targetSdk = 37
        ndk { abiFilters += "arm64-v8a" }
    }
    // the dispatch library must be a real file in nativeLibraryDir
    packaging { jniLibs { useLegacyPackaging = true } }
}

dependencies {
    implementation("com.google.ai.edge.litert:litert:2.3.0")
    implementation("com.google.ai.edge.litert:litert-npu-runtime-google-tensor:2.3.0")
}
