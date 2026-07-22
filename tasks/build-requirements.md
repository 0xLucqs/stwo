# stwo mobile build requirements

These settings are part of the benchmark result. A build that does not meet them must not
be compared with, or represented by, the published mobile numbers.

## Required Rust build

| Setting        | Requirement                                                        |
| -------------- | ------------------------------------------------------------------ |
| Cargo profile  | `release` (`opt-level = 3`)                                        |
| LTO            | `fat` for official measurements; `thin` is the integration minimum |
| Codegen units  | `1`                                                                |
| stwo features  | `prover` and `parallel` enabled                                    |
| Android target | `aarch64-linux-android` / APK ABI `arm64-v8a` only                 |
| CPU flags      | Rust target defaults; do not set `target-cpu`                      |
| Panic strategy | `abort` is optional; never unwind a panic across JNI               |

Cargo's release defaults do **not** satisfy this contract: release defaults to LTO off
and 16 codegen units. Configure the workspace root profile:

```toml
[profile.release]
lto = "fat"
codegen-units = 1
# panic = "abort" # optional for an embedding that treats panic as fatal
```

Enable both stwo features in the consuming crate:

```toml
[dependencies]
stwo = { version = "2.2.0", features = ["prover", "parallel"] }
```

For a build that must not depend on the consuming workspace's root manifest, set Cargo's
profile environment overrides when building the wallet's release target:

```bash
CARGO_PROFILE_RELEASE_LTO=fat \
CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 \
rtk cargo build --release --target aarch64-linux-android -p <wallet-native-package>
```

Record the exact command, `RUSTFLAGS`, git commit, thread count, and SHA-256 of the final
binary or shared library alongside every result. Build into a known clean target directory
when comparing configurations; run one benchmark process at a time.

## AArch64 and Android

Neon is baseline on Android AArch64 and the NDK enables it by default, so the portable
`arm64-v8a` build needs no CPU flag. Never set `target-cpu`, publish x86/x86_64 emulator
numbers, or ship x86/x86_64 Android libraries. Optional extensions such as SHA2 must be
detected at runtime. Before release, inspect the APK/AAB and require every native library
to be under `lib/arm64-v8a/` only.

## Five-minute self-check

Use the wallet's shipped combined identity-proof path (`pipeline_e2e`), not a stwo
micro-benchmark. Record one discarded warmup and five runs, then compare the prove-time
median with the matching row below. The whole check should finish within five minutes;
pass only inside the inclusive `reference * [0.70, 1.30]` interval.

These are the filled T0 results from `mobile-bench-spec.md`: Apple M2 Max,
`build/release-lto` at `66c37403`, stwo `8c998390`, fixture `valid_over_18`, measured
2026-07-22. "All" means 12 Rayon threads.

| Tier / configuration | Scheduling                 | Threads | Reference prove ms | Accepted ms (+-30%) |
| -------------------- | -------------------------- | ------: | -----------------: | ------------------: |
| T0 / Mac P-cores     | default                    |     all |                295 |         206.5-383.5 |
| T0 / Mac E-cores     | `taskpolicy -c background` |     all |              1,248 |       873.6-1,622.4 |
| T0 / Mac P-core      | default                    |       1 |                727 |         508.9-945.1 |
| T0 / Mac E-core      | `taskpolicy -c background` |       1 |              3,342 |     2,339.4-4,344.6 |

Android floor, mid-tier, and flagship rows are not filled yet, so they have no valid
performance gate. Do not substitute extrapolated or plain-stwo kernel timings. On every
platform, still require the expected binary SHA-256, git/stwo revisions, feature set,
thread count, successful verification, and stable thermal conditions. Reject and rebuild
after any provenance/configuration mismatch or a filled-tier timing miss.

Sources: [Cargo profiles](https://doc.rust-lang.org/stable/cargo/reference/profiles.html),
[Cargo profile environment variables](https://doc.rust-lang.org/cargo/reference/environment-variables.html),
[Android ABI packaging](https://developer.android.com/ndk/guides/abis), and
[Android Neon support](https://developer.android.com/ndk/guides/cpu-arm-neon).
