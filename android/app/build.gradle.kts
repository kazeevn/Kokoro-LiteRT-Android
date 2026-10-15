plugins {
    id("com.android.application")
}

android {
    namespace = "io.github.kazeevn.kokorotts"
    compileSdk = 37
    ndkVersion = "29.0.14206865"

    defaultConfig {
        applicationId = "io.github.kazeevn.kokorotts"
        minSdk = 35
        targetSdk = 37
        versionCode = 1
        versionName = "0.1"
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild {
            cmake { arguments += listOf("-DANDROID_STL=c++_shared") }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.31.6"
        }
    }

    packaging {
        jniLibs { useLegacyPackaging = true }
    }

    androidResources {
        noCompress += listOf("tflite", "bin")
        // multi-hundred-MB assets: keep them out of the compressed resources table
    }
}

dependencies {
    // LiteRT Interpreter API: XNNPACK on CPU, NNAPI delegate for the Tensor G3 EdgeTPU.
    // (LiteRT 2.x's CompiledModel NPU path needs Google's EdgeTPU app allowlist on Pixel.)
    implementation("com.google.ai.edge.litert:litert:1.4.2")
    testImplementation("junit:junit:4.13.2")
}
