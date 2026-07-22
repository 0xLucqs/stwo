plugins {
    id("com.android.application") version "9.2.1"
}

android {
    namespace = "eu.stwo.bench"
    compileSdk = 36
    enableKotlin = false

    defaultConfig {
        applicationId = "eu.stwo.bench"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

dependencies {
    testImplementation("junit:junit:4.13.2")
}
