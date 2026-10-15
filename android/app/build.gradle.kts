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
    // LiteRT CompiledModel API: XNNPACK on CPU; on the Tensor G3 EdgeTPU, models AOT-compiled with
    // the Google Tensor SDK run through the Google Tensor dispatch library (no on-device compile,
    // so no EdgeTPU app allowlist).
    implementation("com.google.ai.edge.litert:litert:2.3.0")
    implementation("com.google.ai.edge.litert:litert-npu-runtime-google-tensor:2.3.0")
    testImplementation("junit:junit:4.13.2")
}
