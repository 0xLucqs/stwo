# Perf campaign — the path (2026-07-22)

North star: EU-DI identity proving on all European devices. Product metric:
combined identity proof latency per device tier (budget: 1-3 s mid-tier).
Measured today (Mac M2 Max, pinned stwo 8c998390): 295 ms all-core, P-256 stage
dominant, parallel scaling weak (4→12 threads = 1.22x), memory a non-issue
(~250 MiB), proof size ~1 MiB (flagged for BLE presentment).

## Track A — land what exists (unblocks everything downstream)

1. [DECISION: Lucas] Sign off + commit the stwo worktree (W0): GKR/sumcheck kernel
   parallelization (3.21x), batched Lagrange inversions, safe gen_eq_evals init,
   CPU/SIMD proof-parity tests. Gates green, math-reviewed. `gkr-perf-spec.md`.
2. GKR final experiments: DOWNGRADED (2026-07-22). The e2e verdict is decisive
   without them. Land E1/E2 only if codex already finished them (E1 benefits any
   sumcheck user, incl. MleEval; E2 is trivial glue); do NOT start E3 — its
   soundness-review cost buys nothing on a path that won't ship. Mark
   `gkr-final-experiments-spec.md` accordingly. Note: the W0 commit is justified
   on NON-GKR merits (shared sumcheck/MleEval infra, set_len soundness fix,
   parity-test suite) — it is not a bet on GKR.
3. Bump eu-id's stwo pin to the landed commit → re-run the Mac baseline
   (`mobile-bench-spec.md` W3, ~10 min) → first measured campaign delta on the
   real product workload. This loop (optimize stwo → bump pin → re-measure
   identity proof) is the campaign's heartbeat from now on.

## Track B — mobile measurement (parallel, no owned devices needed)

4. [HAND TO CODEX] `mobile-bench-spec.md` W1: Android bench app over the existing
   eu-id FFI. Emulator plumbing check only.
5. W4: Firebase Test Lab matrix (Samsung mid / Tensor / flagship, ~$10/pass).
   [DECISION: Lucas] GCP project + billing for Test Lab.
6. When hardware appears: Nord CE protocol (W2 — mid + floor tiers + the
   pinned-vs-unpinned thread sweep that the Mac could not settle). ML-DSA bench
   (W5) when feat/quantum-safe tests pass — likely the scariest number in the
   program; feeds stwo engine-batching priorities.

## Track C — optimization, now data-targeted

The Mac numbers reorder the whole audit. Priority order:

7. **P-256 stage profile: DONE (2026-07-22)** — capture archived at
   ~/eu-id/tasks/bench-results/p256-profile-mac-m2max-66c37403.txt.
   Compute split (idle-normalized): **blake2s Merkle ~37%**, FFT ~14%, packed
   field mul ~10%, constraint eval ~8%, quotients+inverse ~5%. AND ~50% of all
   thread-slots are sync/idle (matches the 2.5x/12-thread scaling).
   Config notes: eu-id already runs fold_step=2 (FRI leaf packing engaged — the
   audit's Tier-3.1 item does NOT apply); bridge config uses log_blowup=2 (4x
   LDE) + 59 queries + pow 10 — a prover-time-for-proof-size trade that inflates
   hash+FFT shares; a blowup knob exists but is a security-parameter change.
   CONSEQUENCE — Track C reordered:
   a. Hashing is the #1 compute target → the SHA-256 hw channel (item 10, also
      compliance-aligned) and any committed-bytes reduction lead.
   b. The idle half → serial-structure/phase-overlap work (task-graph overlap of
      independent phases; the 07-06 audit's deep-rework candidate A) + the
      small-trace fixed-overhead program (item 8).
   c. FFT pass-reduction is real but bounded at ~14%.
8. **Small-trace fixed-overhead program** (thesis confirmed twice): cross-proof
   caching of twiddles/preprocessed trees (byte-identical roadmap item 5 — also
   directly the cold-start delta), setup amortization, the <=2^16 single-threaded
   FFT gap, allocation reuse. Byte-identical items 2 (OODS/barycentric) and 4
   (DEEP fuse/tile) as the portable pass-reduction wins.
9. **Prereq for kernel-class changes**: canonicity audit + golden-proof byte gate
   (`byte-identical-prover-optimizations.md` + session amendments).
10. [DECISION: Lucas — ask the eIDAS certification owner NOW] Blake2s vs SHA-256
    channel. MEASURED (2026-07-22, build-hardening W3): hw SHA-256 beats 16-way
    SIMD blake2s 1.27x on Mac P-cores, 1.51x on E-cores → with blake at ~37% of
    P-256 prove compute, the swap is worth ~8% (P) to ~12% (E) of prove time on
    Apple silicon. Decision therefore stays compliance-led; the perf case alone
    is modest HERE but expected larger on the Android floor (weak NEON, same SHA
    silicon) — measure the same ratio on the Nord CE A55 cluster before final
    sizing. Inline hardening landed as cb95b17b; build-requirements doc written.
11. [FLAG to product/protocol] proof_bytes ~1 MiB vs BLE/NFC presentment —
    independent of proving speed; needs an owner.

## Explicitly closed / parked

- GKR lookups for production: closed (no crossover; memory-worse). `gkr-vs-logup-verdict`.
- Field swap (>u32 or M61): closed (`field-swap-analysis.md`).
- Jagged/multilinear protocol migration: parked (strategy: don't copy, stay circle).
- Metal/GPU backend: parked until identity-latency levers are exhausted.
- W1-adaptive-chunk, lambda-hoist, eager layer drop: disproven/superseded.

## Spec coverage map (2026-07-22)

Dumb-agent-ready now: `mobile-bench-spec.md` (W1 Android app / W4 Test Lab / W5
ML-DSA; W3 done), `build-hardening-spec.md` (inline hardening + build-requirements
doc + hash micro-bench). Reference docs: `byte-identical-prover-optimizations.md`
(with session amendments), `perf-redesign-audit.md`, `field-swap-analysis.md`.

Specs still TO WRITE (in order, each gated on its input):
- SHA-256 Merkle channel spec — gated on the eIDAS compliance answer + the
  W3 hash micro-bench numbers (`build-hardening-spec.md`).
- Phase-overlap / serial-structure spec — gated on nothing; input is the P-256
  profile (50% idle) + the 07-06 audit's deep-rework candidate A. Next to write.
- Small-trace fixed-overhead spec (cross-proof caching, sub-2^16 FFT parallelism,
  setup amortization) — sharpened by the cold/warm delta once W1/W2 report it.

## The loop

optimize (Track C) → land (Track A gates) → pin bump → re-measure identity proof
(Track B) → re-rank Track C from the new profile. Every number with provenance.
