# stwo build hardening + hash micro-bench — spec

Small self-contained spec (perf campaign; see `campaign-path.md`). Two stwo-side items
that mobile numbers depend on, dropped from `mobile-bench-spec.md` when it was
refocused on eu-id workloads.

## W1 — packed field `#[inline]` hardening

Add `#[inline(always)]` to every arithmetic trait impl on `PackedCM31`
(`crates/stwo/src/prover/backend/simd/cm31.rs`: Add, Sub, Mul, Neg, Mul<PackedM31>,
any *Assign) and `PackedQM31` (`crates/stwo/src/prover/backend/simd/qm31.rs`: Add,
Sub, Mul, Neg, AddAssign, `inverse`, any Mul/Add<PackedCM31>/<PackedM31> lacking it).
Mirror `m31.rs`, which already does this everywhere. Bit-identical output.

Why: eu-id links stwo across crate/FFI boundaries; without these attributes,
integrator builds without fat LTO pay a function call per packed field op in the
hottest loops. The EU builds the wallet APK — we don't control their LTO settings.

Gates: `cargo test --features "prover,parallel" -p stwo --lib`,
`cargo check --features prover -p stwo`, `scripts/clippy.sh`, `scripts/rust_fmt.sh`.
Commit: `perf(simd): inline packed CM31/QM31 field arithmetic`.

## W2 — build-requirements one-pager

Write `tasks/build-requirements.md` for wallet integrators:

- Required profile: `release`, `lto = "fat"` (minimum `"thin"`), `codegen-units = 1`.
- aarch64: NEON is baseline, no target-cpu flag needed; never ship x86 Android.
- Feature flags: `prover`, `parallel`.
- Self-check: expected identity-proof timings per tier (copy from
  `mobile-bench-spec.md` result tables as they fill) +-30%, so an integrator can
  validate their build in five minutes.

## W3 — hash micro-bench (feeds the SHA-256 channel decision)

DO THIS LAST, and only on a quiet machine: before any timed run, verify no other
cargo/gradle/bench processes are active (`pgrep -fl "cargo|gradle|rustc"` must show
nothing but your own invocation). Another worker may be building the Android bench
app in parallel — wait it out or retime later; timing measured against a busy
machine is void.

Standalone, no repo changes needed beyond a bench file (or a scratch crate):
measure, on Apple P-cores AND E-cores (`taskpolicy -c background`):

1. stwo's 16-way SIMD blake2s `compress16` throughput (MB/s over 64-byte messages,
   the Merkle node shape) — reuse `crates/stwo/benches/merkle.rs` machinery or call
   `compress16` in a tight loop.
2. `sha2` crate SHA-256 with hardware acceleration (`asm`/`sha2-asm` feature or the
   crate's runtime detection), same total bytes, both single-stream and N
   interleaved independent streams (N in {4, 8, 16}) — interleaving is how a Merkle
   builder would use it.
   Record MB/s per configuration in this file. Decision input: combined with the P-256
   profile (blake2s = ~37% of prove compute), `speedup_of_merkle_hashing x 0.37` is the
   prove-time upside of the SHA-256 channel on this hardware; the Android-floor upside
   is larger (weak NEON, universal crypto extensions) and the compliance question
   (SOG-IS/NIST) may decide regardless. On-device repeat happens via the Nord CE / W1
   app once available.

### Results

| config                             | blake2s compress16 MB/s | sha256 hw MB/s (best N) | ratio  |
| ---------------------------------- | ----------------------- | ----------------------- | ------ |
| Mac P (default)                    | 1,254.013               | 1,593.200 (N=16)        | 1.270x |
| Mac E (`taskpolicy -c background`) | 529.432                 | 796.964 (N=8)           | 1.505x |

Ratio is best SHA-256 payload throughput divided by Blake2s `compress16` payload
throughput. The complete SHA-256 matrix was:

| config                             |  N=1 MB/s |  N=4 MB/s |  N=8 MB/s | N=16 MB/s |
| ---------------------------------- | --------: | --------: | --------: | --------: |
| Mac P (default)                    | 1,261.529 | 1,510.179 | 1,550.658 | 1,593.200 |
| Mac E (`taskpolicy -c background`) |   535.919 |   667.552 |   796.964 |   590.591 |

Measured 2026-07-22 on an Apple M2 Max (8P+4E), macOS 26.5.2, with Rust
1.94.0-nightly, fat LTO, one codegen unit, and `-C target-cpu=native`. The direct
release binary SHA-256 was
`ca2d127fa9f8350ebf241347e2dba964f0b404a67176bbcdbb959c2dfc3dbab0`.
Each configuration used three 32 MiB warmups and the median of eleven 256 MiB
payload samples. SHA-256 used the `sha2` crate's `asm` path, independently
initialized states, and both the 64-byte data block and mandatory padding block;
the counted throughput is the 64-byte payload. Every low-level SHA state and all
sixteen Blake2s lanes matched the corresponding `Digest` result before timing.
The measured binary contains `sha256h`, `sha256h2`, `sha256su0`, and `sha256su1`.

Immediately before each timed run, `pgrep -fl "cargo|gradle|rustc"` and a separate
benchmark-process scan both returned no processes. P ran first via the direct
binary; E ran second via `taskpolicy -c background`. Raw samples, benchmark source,
quiet-check outputs, and build provenance are in
`/private/tmp/stwo-w3-hash.CjxKKi/`.

Using the spec's first-order estimate `(ratio - 1) * 0.37`, the measured hashing
uplift corresponds to about **10.0%** prove-time upside on Mac P and **18.7%** on
Mac E. Holding the non-hash 63% fixed and applying Amdahl's law gives stricter
whole-proof reductions of 7.9% and 12.4%, respectively. Android still needs the
specified on-device repeat.
