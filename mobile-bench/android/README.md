# Android Game Loop wrapper

This app exposes the Rust benchmark as Firebase Test Lab Game Loop scenario 1. It only
packages `arm64-v8a`; it does not execute a copied binary from app storage.

## Build

Prerequisites: JDK 17+, Android SDK/API 36 with Build Tools 36.0.0, Android NDK,
`cargo-ndk`, and the Rust `aarch64-linux-android` target. The checked-in wrapper pins
Gradle 9.5.0, and AGP is pinned to 9.2.1.

From the repository root:

```bash
rtk proxy rustup target add aarch64-linux-android
CARGO_PROFILE_RELEASE_LTO=fat \
CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 \
rtk cargo ndk -t arm64-v8a -P 26 \
  -o mobile-bench/android/app/src/main/jniLibs \
  build --release --features "jni,parallel" -p bench-runner
rtk proxy ./mobile-bench/android/gradlew -p mobile-bench/android :app:assembleRelease
rtk proxy unzip -l mobile-bench/android/app/build/outputs/apk/release/app-release.apk
```

The final listing must contain `lib/arm64-v8a/libbench_runner.so` and no x86 ABI.
The APK is signed with the standard debug key because it is a Test Lab-only harness,
not a distributable application. Legacy JNI packaging is intentional: Android extracts
the library to a readable filesystem path so the benchmark can hash the exact loaded
`libbench_runner.so` in its metadata.

The Cargo profile environment values are deliberate: a profile declared in a workspace
member is ignored, while these values apply even if the repository root profile changes.
`thin` is the minimum acceptable value for `CARGO_PROFILE_RELEASE_LTO`; official result
runs use `fat`.

## Run in Firebase Test Lab

List physical models first, choose one model/API pair, and run devices sequentially:

```bash
rtk proxy gcloud firebase test android models list --filter="form=PHYSICAL"
rtk proxy gcloud firebase test android run \
  --type=game-loop \
  --app=mobile-bench/android/app/build/outputs/apk/release/app-release.apk \
  --scenario-numbers=1 \
  --device=model=MODEL_ID,version=API_LEVEL \
  --timeout=30m \
  --results-bucket=GCS_BUCKET \
  --results-dir=UNIQUE_RESULTS_DIR
rtk proxy gcloud storage cp --recursive \
  gs://GCS_BUCKET/UNIQUE_RESULTS_DIR ./mobile-bench-results
```

Use a unique results directory for every invocation. Repeat each device at a different
time of day; if run medians differ by more than 10%, run it a third time and take the
overall median.

## Game Loop contract

The dedicated activity filter uses action `com.google.intent.action.TEST_LOOP`, category
`android.intent.category.DEFAULT`, and MIME type `application/javascript`. Test Lab puts
the scenario number in the integer extra `scenario` and the writable result destination
in `Intent.data`. The JNI API takes a path, so the wrapper writes to app-private cache and
then copies the completed JSONL through `ContentResolver` to that destination. Finishing
the activity tells Test Lab that the loop is complete; missing output or JNI/I/O failures
crash the process so the matrix cannot silently pass without an artifact.

Firebase documents a single JSON object if console visualization is desired. This harness
intentionally emits the benchmark's JSON-lines schema; consume the raw object from the
configured results bucket rather than relying on the console's game-stat visualization.

Sources: [Firebase Android Game Loop guide](https://firebase.google.com/docs/test-lab/android/game-loop),
[gcloud Android test reference](https://cloud.google.com/sdk/gcloud/reference/firebase/test/android/run),
[AGP 9.2 compatibility](https://developer.android.com/build/releases/agp-9-2-0-release-notes).
