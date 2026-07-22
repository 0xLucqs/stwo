# GKR path: make LogUp-GKR beat prefix-sum LogUp

## VERDICT (2026-07-21, final E1–E3 experiments)

**CLOSED — no crossover after E1–E3.** The fastest accepted configuration is E1+E2:
indexed cross-instance parallelism in sumcheck/GKR plus packed, parallel example-side
layer construction and MLE combination. E3's fused fold+sum implementation passed its
correctness and math-review gates and improved isolated kernels by 10.7%–16.5%, but it
regressed end-to-end Path B by 4.9% at L=1 and 7.8% at L=64, so it was reverted.

Across the final `{2^16, 2^20} x L={1,4,16,64}` matrix, Path B remains 1.085x–2.192x
Path A. At 2^20 it remains 1.144x–1.817x slower. Path B uses more peak memory in every
cell; at 2^20/L=64 it uses 5,250.8 MiB versus 3,155.9 MiB. It still wins proof size for
L>=4, but the compute and memory result does not cross. The closest cell is 2^16/L=64
(196.688ms versus 181.353ms); the closest 2^20 cell is L=64 (2,408.856ms versus
2,105.763ms).

Per the final-experiment decision rule, do not start further CPU optimization work in this
workstream. Any reopening should be a new architectural proposal with its own scope and
evidence, not another incremental tuning pass.

Goal: the GKR lookup prover currently loses to the prefix-sum LogUp baseline on
wall-clock. The math says it should win (no committed interaction trace). Close the
implementation gap, measure the crossover, then quantify the end-to-end protocol win.

Existing assets: `benches/lookups.rs` (grand-product + generic/multiplicities/singles
logup, CPU & SIMD), GKR tests in `simd/lookups/gkr.rs:517`, `sumcheck.rs:208`,
MleEval component already landed (recent W2 commits).

Constraint: all Phase 1 changes are bit-identical output, prover-only, no transcript
changes — autonomous per repo rules. Phase 2 changes internal round-poly computation
paths only (protocol unaffected) — gate on full GKR test suite + bench numbers.

## Phase 0 — Baseline (no code changes)
- [x] Run `cargo bench --features "prover,parallel" --bench lookups` — record numbers
- [x] Profile one logup-multiplicities run (macOS `sample`, 5s over the criterion loop)
- [x] Prefix-sum reference point recorded

### Results (2026-07-21, HEAD 4f877db2, parallel feature ON)
Bench means at 2^16 rows:
| bench                        | simd     | cpu       |
|------------------------------|----------|-----------|
| grand product                | 2.93 ms  | 7.75 ms   |
| singles logup                | 4.70 ms  | 14.35 ms  |
| multiplicities logup         | 5.33 ms  | 15.82 ms  |
| generic logup                | 5.95 ms  | 17.72 ms  |
| grand product batch 4x       | 10.98 ms | 29.95 ms  |
Reference: simd prefix_sum at 2^24 = 6.46 ms — 256x the data in the same wall time.
GKR is ~2 orders of magnitude more expensive per element than the raw prefix-sum kernel
(the full baseline also pays interaction-gen + commit + constraint eval, measured in
Phase 3, but this is the gap to attack).

Profile (top-of-stack samples, ~3500 total, single thread doing all work — confirms
the stack ignores the `parallel` feature entirely):
- PackedQM31::mul (inlined sumcheck sum kernels + layer gen): ~65%
- prove_batch residue (round loop glue): ~9%
- scalar QM31::inverse: ~4.5%  ← per-round `correct_sum_as_poly_in_first_variable`
  interpolation does scalar field inversions; batchable, unexpected extra target
- Mle::fix_first_variable (QM31 + M31): ~5%
- next_layer: ~2.5%; gen_eq_evals: ~0.7%
- malloc/free/memmove/memset: ~4%
Conclusion: ALU-bound on packed QM31 mul on ONE core. Parallelization (Phase 1) attacks
the dominant term directly; fused rounds (Phase 2) may cut remaining memory traffic but
must be re-profiled and reviewed first. The proposed λ-hoist saves no packed
multiplications, and the interpolation node derived from `y` is not fixed or cacheable.

## Phase 1 — Bit-identical engineering (expect the bulk of the win)
- [x] Parallelize `simd/lookups/gkr.rs`: `gen_eq_evals`, `next_gen_layer`,
      `next_logup_layer` variants, `eval_grand_product_sum`, `eval_logup_sum` variants —
      `parallel_iter!` chunks + per-thread accumulators (field addition exact →
      deterministic). Implemented in the dirty worktree; acceptance is blocked on the
      unsafe-initialization review. Multi-chunk CPU/SIMD proof parity is now covered.
- [x] Parallelize `simd/lookups/mle.rs::fix_first_variable` (both F variants), dirty worktree
- [x] λ-hoist investigated and rejected: distributing `eq` leaves the same two packed
      multiplications per term.
- [x] Eager layer freeing via `pop` rejected: the reversed `IntoIter` already drops consumed
      layers; a real peak-memory reduction requires checkpointing or recomputation.
- [x] Batch scalar QM31 interpolation inversions (dirty worktree). Equivalence applies to
      distinct nodes; exceptional challenge collisions still panic and are separate protocol work.
- [x] Re-run final accepted parallel kernels at 2^16 and 2^20; means are recorded in
      `tasks/gkr-final-experiments-spec.md`. The earlier isolated serial/parallel W3 matrix
      remains the feature-scaling reference.

### Invalidated 2^20 scaling result (2026-07-21, fixed chunk 2^7)

| bench | serial ms | parallel ms | speedup |
|-------|-----------|-------------|---------|
| grand product | 12.921 | 13.314 | 0.970x |
| generic logup | 23.570 | 24.596 | 0.958x |
| multiplicities logup | 22.207 | 24.604 | 0.903x |
| singles logup | 20.724 | 21.863 | 0.948x |

**Do not use this table.** The serial executable came from the shared `target/` directory
and was dated 2026-07-04, before the current dirty-worktree algorithms. The parallel
executable was current. The apparent 3–11% regression was therefore an artifact mismatch,
not a measurement of feature scaling.

### Corrected isolated A/B and profile (2026-07-21)

Both feature variants were rebuilt from the current sources into distinct target
directories. Their Criterion filters listed the same four SIMD 2^20 targets.

| configuration | estimate ms | 95% interval ms | speedup vs serial |
|---------------|-------------|-----------------|-------------------|
| serial (`prover`) | 72.926 | 72.797–73.070 | 1.00x |
| parallel, 12 Rayon workers | 22.079 | 21.506–22.761 | 3.30x |
| parallel, 8 Rayon workers | 20.721 | 20.651–20.793 | 3.52x |
| parallel, 4 Rayon workers | 24.776 | 24.537–25.181 | 2.94x |

This short diagnostic triggered the full W3 run below. Its apparent 8-worker advantage
for multiplicities did not reproduce under the full protocol.

Full default-Criterion W3 results (3-second warm-up, 100 samples per cell):

| bench | serial ms (95%) | parallel 12 ms (95%) | speedup | parallel 8 ms (95%) |
|-------|-----------------|-----------------------|---------|----------------------|
| grand product | 35.700 (35.661–35.738) | 12.894 (12.816–12.977) | 2.77x | 12.015 (11.987–12.044) |
| generic logup | 83.390 (83.222–83.557) | 23.141 (22.953–23.453) | 3.60x | 22.546 (22.500–22.593) |
| multiplicities logup | 74.910 (74.774–75.056) | 21.964 (21.855–22.095) | 3.41x | 22.187 (21.985–22.406) |
| singles logup | 65.491 (65.416–65.565) | 20.877 (20.687–21.080) | 3.14x | 21.064 (20.553–21.669) |

The 12-worker geometric-mean speedup is 3.21x. Every cell clears the 2x stop gate, though
the three LogUp cells remain below the 4x target. Eight workers lower geometric-mean time
by only 1.92% and the per-cell direction is mixed, so no worker-default change is justified.
W3 is complete; further parallel tuning requires a new measured target rather than a
configuration change based on this noise-sized aggregate difference.

Fresh 10-second macOS `sample` captures:

- Serial: 7,565 snapshots; `PackedQM31::mul` 77.20%, `prove_batch` residue 10.18%,
  secure-field MLE fold 3.85%, fraction add 3.09%, next-layer generation 2.93%.
- Parallel (12 workers): 73,502 thread-slot snapshots across main + 12 workers;
  condition wait 30.04%, scheduler switch 29.96%, `PackedQM31::mul` 26.15%, generic
  LogUp sum closure 2.85%, mutex wait 1.44%, secure-field MLE fold 1.28%.

The parallel wait samples include expected main-thread barriers and idle worker slots, so
they are evidence of remaining utilization headroom, not wall-time regression. The current
parallel path is a measured 3.30x faster. The profiler's outer benchmark symbol is
LTO-folded to `bench_gkr_grand_product`, but the capture contains multiplicities/generic
LogUp kernels and no grand-product kernels; the exact Criterion filter is authoritative.

Raw captures:
`/private/tmp/gkr-profile-wEtDpl/serial-current-10s.sample.txt` and
`/private/tmp/gkr-profile-wEtDpl/parallel-current-10s.sample.txt`.

Full Criterion baselines are retained under `target/criterion/*/gkr-w3-full-{serial,
parallel-12,parallel-8}/`.

## Phase 2 — Round restructuring (same math, different evaluation schedule)
- [x] Fused fold+eval was implemented behind the corrected next-round SIMD cutoff,
      reviewed at 98% mathematical confidence, and passed the full correctness gates.
      It improved isolated kernels but regressed end-to-end `gkr_prove`, so it was rejected
      and reverted. No E3 proof-path code remains in the accepted configuration.
- [ ] **DEFERRED pending human review / out of this spec:** evaluate at t=1/2 instead of
      t=2 (adds instead of double+sub; changes interpolation nodes in
      `correct_sum_as_poly_in_first_variable`)
- [ ] Full GKR/sumcheck/mle_eval test suite green after each step

## Phase 3 — Crossover + end-to-end
- [x] Measured the same lookup statement through both paths at 2^16 and 2^20 for
      `L in {1,4,16,64}` using the final 12-worker E1+E2 configuration.
- [x] Recorded total and phase time, peak RSS, proof size, and GKR proof felts in
      `tasks/gkr-e2e-spec.md` and `tasks/gkr-final-experiments-spec.md`.
- [x] Closed the decision: no production crossover in the measured region; no further
      optimization under this workstream.

## Verification
- Existing GKR correctness tests after every change; field-by-field SIMD-vs-CPU proof parity
  covers all four variants at 2^6, 2^7, and 2^14 plus an unequal-size batch
- `scripts/clippy.sh`, fmt, no-std gate untouched (all changes are prover-feature code)
- Criterion before/after per phase, committed to this file
