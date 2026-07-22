# Mobile proving benchmarks — EU-ID workload — spec (perf campaign)

Self-contained spec for an implementing agent with no prior context. Part of the stwo
perf campaign (companion docs in this directory: `gkr-perf-*`, `byte-identical-*`,
`field-swap-analysis.md`). Results are recorded IN THIS FILE; implementation work
happens in the **eu-id repo at `~/eu-id`** (the wallet AIR stack that consumes stwo).

**The measured workload is ALWAYS the eu-id proofs — the combined identity pipeline
(P-256 ECDSA + SHA-256 + predicates, cross-bound) and its standalone stages — through
the exact FFI code paths the wallet SDK ships. No plain-stwo primitive benchmarks are
in scope anywhere in this spec.** stwo appears only as the pinned dependency whose
rev is recorded with every number.

Deliverable: the filled result tables (bottom) answering, per device tier: identity
prove latency vs a 1-3 s UX budget, verify latency, peak memory vs the 4 GB-device
envelope, proof size vs BLE/NFC presentment, cold-start delta, and the
classical-vs-ML-DSA gap.

## What already exists in ~/eu-id — reuse, do not rebuild

- `crates/eu-id-ffi/src/lib.rs` (branch `build/release-lto`): C-ABI bench entry
  points `eu_id_bench_sha256`, `eu_id_bench_p256`, `eu_id_bench_identity`,
  `eu_id_bench_mdoc(iters)` — each builds its witness from internal fixtures
  (self-contained, no inputs), runs prove+verify `iters` times, and returns
  `{ prove_ms, verify_ms, peak_bytes, proof_bytes, ok }` (medians; phys-footprint
  watermark; panic-safe across FFI). THESE FUNCTIONS ARE THE ENTIRE WORKLOAD.
- `crates/sdk/android`: working Gradle + cargo native build producing jniLibs
  (`make publish-android-local`), incl. build-id symbolization
  (`publish-android-symbols`) for simpleperf later.
- `mobile/EuIdBench`: iOS SwiftUI bench app already wrapping the same FFI fns.
- `crates/eu-id-prover/benches/identity_bench.rs` + `examples/bench_report.rs`:
  laptop criterion benches with per-stage breakdown (sha / p256 / predicates /
  pipeline) sharing stage definitions via `benches/common/stages.rs` — the
  laptop-side interpretation key.
- Root `Makefile`: `bench`, `bench-report`, `bench-breakdown`; `bench-mobile`
  currently delegates to a nonexistent `mobile/Makefile` (fix in W1).

## Branch strategy in ~/eu-id (verified 2026-07-22)

- **Primary target: `build/release-lto`** (latest build/perf line; carries all four
  bench FFI fns — verified). All classical measurements build from it.
- **ML-DSA lives on `feat/quantum-safe`** (active WIP, 103 commits diverged, FFI
  surface differs — only `eu_id_bench_sha256` there, no mldsa bench fn yet). See W5.
- Branch churn is high. Every recorded number carries branch + commit + APK/binary
  sha256 in its metadata record. Numbers without provenance are void.
- The eu-id workspace pins stwo by git rev; record it too. NOTE FOR THE CAMPAIGN:
  the uncommitted stwo perf work in this repo's worktree is NOT in these numbers
  until it is committed and eu-id's pin is bumped — these benchmarks establish the
  PRE-optimization mobile baseline.

## Global constraints

1. Changes in ~/eu-id, additive only, on a NEW branch `feat/android-bench` cut from
   `build/release-lto` (never commit to release-lto directly): `crates/eu-id-ffi`
   (new `jni` feature), new `mobile/EuIdBenchAndroid/` project, `mobile/Makefile`.
   Do NOT modify
   `crates/eu-id-prover`, `crates/stwo-*`, `crates/predicates`, or the measurement
   logic inside the existing FFI fns. No changes in ~/stwo except recording results
   in this file.
2. Hygiene: one benchmark process at a time; `iters >= 5`; medians; never record
   emulator numbers.
3. Gates after any code change (run in ~/eu-id): `make check && make test`, the iOS
   app still builds, the new Android app builds.

## W1 — Android bench app over the existing FFI

1. **JNI surface**: add a `jni` cargo feature to `eu-id-ffi` (off by default; `jni`
   crate gated behind it; ensure crate-type includes `cdylib`). For each existing
   bench fn, one wrapper
   `Java_eu_euid_bench_BenchRunner_<name>(env, _cls, iters: jint) -> jstring`
   returning the result struct as a JSON object string with field names exactly
   `prove_ms`, `verify_ms`, `peak_bytes`, `proof_bytes`, `ok`. No new measurement
   logic — call the existing fn, serialize, return.
2. **App** (`~/eu-id/mobile/EuIdBenchAndroid/`): minimal Gradle project, package
   `eu.euid.bench`, minSdk 26, arm64-v8a only, native lib via `cargo-ndk`
   (release, `lto = "fat"`, `codegen-units = 1`). Add `mobile/Makefile` with
   `bench-android-apk`; point root `bench-mobile` at it.
   Two modes:
   - **Game Loop** (Firebase Test Lab): activity with intent filter
     `com.google.intent.action.TEST_LOOP`; on launch run the suite, write ONE JSON
     document to the intent-designated output URI, `finish()`. (WebFetch the current
     Game Loop contract for the URI handling; it is small.) Invoke Rust via JNI
     only — never exec an extracted binary (Android 10+ W^X denies it).
   - **Manual**: a button + text view showing the same JSON (for the Nord CE and any
     future owned device).
3. **Suite order** (all four fns, `iters = 5`): `identity` first — preceded by one
   extra `iters=1` call recorded as `identity_cold_ms` (cold-start signal; do not
   touch FFI internals) — then `mdoc`, `p256`, `sha256`.
4. **JSON document schema**:
   ```json
   {
     "meta": {"model": "...", "soc_features": "<cpuinfo Features line>", "cores": 8,
              "api": 33, "branch": "build/release-lto", "git": "<commit>",
              "stwo_rev": "<pinned>", "apk_sha256": "...",
              "thermal_before_c": [...], "thermal_after_c": [...],
              "pinning": "unpinned|a55|a77"},
     "identity_cold_ms": 0,
     "benches": {"identity": {...}, "mdoc": {...}, "p256": {...}, "sha256": {...}}
   }
   ```
   Thermal via `/sys/class/thermal/thermal_zone*/temp`, best effort.
5. Plumbing validation on an emulator is allowed; its numbers are never recorded.

## W2 — Owned device: OnePlus Nord CE 5G (RUN FIRST — free, covers two tiers)

Snapdragon 750G = 2x Cortex-A77 (mid-tier reference) + 6x Cortex-A55 (modern floor
core). From `build/release-lto`, W1 app in manual mode (or the same cdylib wrapped in
a tiny `aarch64-linux-android` binary pushed via adb):

1. Identify clusters: `cat /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq`
   (expect A55 = cpu0-5, A77 = cpu6-7; verify, don't assume).
2. Run the suite three ways and record all three: `taskset 3f` (A55 = FLOOR tier),
   `taskset c0` (A77 = MID tier), unpinned (scheduler baseline).
   (APK manual mode: pin via `adb shell taskset -p <mask> <pid>` right after launch,
   or prefer the pushed-binary route where `taskset` wraps the exec.)
3. Record `grep Features /proc/cpuinfo` (the `sha2` flag matters for the future
   hash-channel decision).
4. Five back-to-back unpinned `identity` runs for the thermal/throttle profile.

## W3 — Tier 0: Mac reference (free, anchors ratios; proxies iPhone)

Using the EXISTING laptop path (`make bench-report` / criterion benches in ~/eu-id)
on `build/release-lto`: default run (P-cores) and `taskpolicy -c background`
(E-cores), each also with `RAYON_NUM_THREADS=1`. Four rows into T-RATIO.

## W4 — Firebase Test Lab (fleet diversity, after W2 numbers exist)

1. `gcloud firebase test android models list --filter=form=PHYSICAL`; pick by EU
   relevance: 1 Samsung mid (Galaxy A5x-class), 1 Pixel (Tensor), 1 current flagship
   (Pixel or Galaxy S). The Nord CE already covers generic Snapdragon mid + floor.
2. Per device, sequentially:
   ```bash
   gcloud firebase test android run --type=game-loop \
     --app <apk> --device model=<codename>,version=<api> --timeout 30m \
     --results-bucket <bucket> --results-dir <device>-run<i>
   ```
   Two passes at different times; >10% divergence on `identity.prove_ms` → third
   pass; record overall median. Pull JSONs into `~/eu-id/tasks/bench-results/`
   (committed there); transcribe medians into the tables here.
3. Cost ≈ 3 devices x ~0.5 h x ~$5/h ≈ $8-10 per matrix pass.

## W5 — ML-DSA (post-quantum) coverage — gated

1. Gate: on `feat/quantum-safe`, ML-DSA prove/verify tests must pass. If the
   completeness fix is still in flight, STOP this item and note it — never bench a
   prover that doesn't verify.
2. Additive on that branch: `eu_id_bench_mldsa(iters)` mirroring the existing bench
   fn pattern exactly, + its jni wrapper. Separate APK built FROM that branch (FFI
   surfaces diverged; do not merge). Same devices, `workload=mldsa` labeled meta.
3. This number feeds the stwo engine-batching priorities (see `tasks/todo.md`).

## Results tables (record branch+commit+stwo rev+sha256 above each)

### T-DEVICE (one row per device x pinning x bench; medians)
| device (tier) | pinning | bench | prove ms | verify ms | cold ms | peak MB | proof KB |
|---------------|---------|-------|----------|-----------|---------|---------|----------|

### T-RATIO — identity.prove_ms cross-platform

W3 MEASURED 2026-07-22. Machine: Apple M2 Max (8P+4E). eu-id branch
`build/release-lto` @ 66c37403, stwo pin 8c998390, `--features parallel`,
`bench_report` example (iters=3, fixture valid_over_18). Full per-stage JSONs
archived from scratchpad/w3/. `pipeline_e2e` = the shipped combined identity proof.

| platform | threads | pipeline_e2e prove ms | verify ms | ratio vs Mac-P |
|----------|---------|----------------------|-----------|----------------|
| Mac P-cores (M2 Max) | all | 295 | 85 | 1.00 |
| Mac E-cores (taskpolicy bg) | all | 1,248 | 316 | 4.2 |
| Mac P-core | 1 | 727 | 110 | 2.5 |
| Mac E-core | 1 | 3,342 | 544 | 11.3 |

Per-stage prove_ms (P-cores all-threads / E-1-thread):
p256 334/7,233 · sha 43/783 · age 10/47 · nat 5/12 · pipeline 213/2,587.
Peak memory (pipeline_e2e): 241-307 MiB in every configuration.
Proof size (pipeline_e2e): ~951 KiB; standalone p256 proof ~2.0 MiB.

W3 observations (feed interpretation):
- Parallel scaling at identity sizes is weak: 12 threads buy only 2.5x over 1
  thread on P-cores (2.7x on E) — the small-trace fixed-overhead/serial-fraction
  thesis is CONFIRMED on the real workload; per-core speed will matter more than
  count on phones.
- Thread sweep (same binary, pipeline_e2e): 12 threads 295 ms, 8 threads 325 ms,
  4 threads 359 ms. All-core WINS on M2 Max — Apple E-cores (~2x slower than P)
  are mild enough that work-stealing absorbs them; no straggler tax. This does
  NOT settle Android, where big/little gaps are 3-5x — rerun this exact sweep
  pinned vs unpinned on the Nord CE (W2). Also note 4→12 threads = only 1.22x.
- P256 dominates every configuration (standalone 334ms vs sha 43ms on P-all);
  interpretation q6 is effectively answered: p256 is the stage to optimize.
- Peak memory is a non-issue for the pipeline (~250 MiB) on any 4 GB device.
- proof_bytes ~1 MiB makes the BLE/NFC presentment question (q4) REAL — flag to
  the protocol/product side.
- Rough tier extrapolation (to be replaced by W2 device data): an M2 E-core
  single-thread (3.3 s) is roughly A78-class per-clock; A55 floor cores are
  several times weaker again — floor-device pipeline latency plausibly O(10 s)
  without optimization, mid-tier plausibly 1-3 s. MEASURE, don't trust this.

### T-THERMAL — Nord CE unpinned identity, runs 1-5
| run | prove ms | max zone temp C |
|-----|----------|-----------------|

### Interpretation (answer in writing)
1. Mid-tier (A77) identity prove latency: inside a 1-3 s UX budget?
2. Floor (A55): what latency, and is a floor-specific plan needed (delegation,
   statement redesign) or does it merely need a spinner?
3. peak_bytes vs ~1.5-2 GB practical budget on 4 GB devices: headroom?
4. proof_bytes: BLE/NFC presentment feasibility note.
5. Cold-vs-warm delta: does setup caching deserve a work item?
6. From the laptop per-stage breakdown + device ratios: which stage (sha / p256 /
   predicates / pipeline overhead) most likely dominates on-device? (Directs the
   next optimization spec; per-stage on-device FFI is a possible v2, not in scope.)
7. ML-DSA vs classical gap per tier (if W5 ran).
8. CAMPAIGN LINK: once the uncommitted stwo perf work lands and eu-id bumps its pin,
   re-run W2 unpinned + W3 and record the delta here — that is the mobile-side
   measurement of the campaign's stwo improvements.

## Out of scope

- Any plain-stwo primitive benchmarking (kernels, FFT, hashes).
- Optimization work; per-stage on-device FFI; iOS device-farm runs (Mac + existing
  iOS app cover it); floor-tier device-farm spending; updating the stwo pin.

## Appendix — Honor 10 (Kirin 970 = 4xA73+4xA53), if it surfaces

Same protocol as W2 with masks `0f` (A53) / `f0` (A73). Redundant with the Nord CE
except as a second floor data point (A53 vs A55) — low priority. Aged battery
throttles early: treat as pessimistic bound, the right direction for a go/no-go.
