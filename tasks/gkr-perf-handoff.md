# GKR prover performance handoff

Status date: 2026-07-21  
Repository: `/Users/lucas/stwo`  
Branch: `dev-copy` tracking `fork/dev-copy`  
HEAD: `9c5bebf1` (`perf(examples): parallelize GKR e2e glue...`)  
State: E1 and E2 committed separately; E3 rejected and reverted; pre-existing W0 and
unrelated changes remain unstaged in the dirty worktree

## Executive conclusion

**CLOSED — no crossover after E1–E3.** The end-to-end comparison has now been built and
measured. The fastest accepted subset is E1+E2. Path B (GKR) remains 1.085x–2.192x Path A
(interaction-trace LogUp) across the complete `{2^16,2^20} x L={1,4,16,64}` matrix and
uses more peak memory in every cell. The closest cell is 2^16/L=64 at 196.688ms versus
181.353ms. At 2^20/L=64, the gap is 2,408.856ms versus 2,105.763ms (1.144x), while peak
RSS is 5,250.8 MiB versus 3,155.9 MiB.

E1 delivered the important proof-path win: at 2^20/L=64 it reduced `gkr_prove` from
1,610.938ms to 882.190ms. E2 reduced the remaining example-side glue at that cell:
`gkr_layers` from 421.472ms to 120.041ms and `mle_combine` from 453.446ms to 114.117ms.
E3's reviewed fused kernels improved isolated 2^20 kernels by 10.7%–16.5%, but regressed
end-to-end Path B by 4.9% at L=1 and 7.8% at L=64. Under the fastest-subset rule, E3 was
rejected and restored byte-for-byte to the pre-E3 implementation.

Do not continue incremental CPU tuning under this workstream. A future reopening should
start from a new architectural proposal and a new decision rule.

## Final experiment record

### Accepted commits

| Experiment | Commit | Result |
|------------|--------|--------|
| E1 | `99d73f3a` | Keep: order-preserving cross-instance parallelism for sumcheck rounds and GKR layer generation |
| E2 | `9c5bebf1` | Keep: packed/parallel GKR e2e layer construction and MLE combination |
| E3 | none | Reject/revert: kernel microbenchmarks won, end-to-end GKR lost |

E1's full proof/artifact golden digests match between serial and parallel builds for both
same-size and unequal-size batches. E1 and E3 each received Math Reviewer post-sign-off at
98% confidence. E3 was not committed because the performance acceptance gate failed.

### Final end-to-end matrix

| log_n | L | A total ms | B total ms | B/A | A peak MiB | B peak MiB | A proof B | B proof B (+GKR felts) |
|-------|---|------------|------------|-----|------------|------------|-----------|-------------------------|
| 16 | 1 | 52.904 | 115.971 | 2.192 | 76.6 | 85.1 | 14,580 | 16,548 (+612) |
| 16 | 4 | 71.437 | 120.378 | 1.685 | 83.3 | 93.2 | 14,712 | 14,136 (+810) |
| 16 | 16 | 83.940 | 174.589 | 2.079 | 101.4 | 122.5 | 18,216 | 16,008 (+1,602) |
| 16 | 64 | 181.353 | 196.688 | 1.085 | 191.3 | 350.4 | 21,720 | 17,816 (+4,770) |
| 20 | 1 | 622.613 | 1,131.489 | 1.817 | 1,190.6 | 1,308.2 | 23,556 | 25,012 (+924) |
| 20 | 4 | 715.958 | 1,185.455 | 1.656 | 1,290.7 | 1,395.1 | 23,160 | 23,816 (+1,170) |
| 20 | 16 | 1,042.914 | 1,480.402 | 1.419 | 1,712.6 | 2,040.0 | 26,152 | 24,152 (+2,154) |
| 20 | 64 | 2,105.763 | 2,408.856 | 1.144 | 3,155.9 | 5,250.8 | 30,184 | 24,792 (+6,090) |

The final executable is
`/private/tmp/gkr-e2-post-target/release/deps/stwo_examples-c57d0ea61395cc20`, SHA-256
`dfe685d11e9ee88ee6eb4f1345c10c8b043aa2b16dcc482da06fa9176a6176db`. It was built
with `parallel` and `-C target-cpu=native` and run with 12 Rayon workers. Every timing is
the median of three recorded samples after one discarded warmup. RSS was measured in a
separate `/usr/bin/time -l` invocation for every path/cell.

### Final 2^20 phase rows

| L | path | base_commit | interaction_gen | interaction_commit | gkr_layers | gkr_prove | mle_combine | mle_trace | mle_trace_commit | stark_prove | total | verify |
|---|------|-------------|-----------------|--------------------|------------|-----------|-------------|-----------|------------------|-------------|-------|--------|
| 1 | A | 53.583 | 17.418 | 56.440 | — | — | — | — | — | 495.108 | 622.613 | 0.272 |
| 1 | B | 34.646 | — | — | 3.416 | 46.815 | 28.611 | 10.698 | 60.056 | 931.553 | 1,131.489 | 0.830 |
| 64 | A | 272.046 | 235.571 | 449.410 | — | — | — | — | — | 1,148.625 | 2,105.763 | 0.435 |
| 64 | B | 260.730 | — | — | 120.041 | 793.635 | 117.181 | 12.639 | 60.145 | 1,037.689 | 2,408.856 | 2.271 |

### Final kernel means (E1+E2)

| SIMD benchmark | 2^16 ms | 2^20 ms |
|----------------|---------|---------|
| Grand product | 2.8709 | 13.047 |
| Generic LogUp | 4.7646 | 24.626 |
| Multiplicities LogUp | 4.6373 | 22.660 |
| Singles LogUp | 4.2370 | 20.718 |

E2 is example-only, so these are also the exact pre-E3 kernel baselines. E3 changed the
2^20 values to 11.645, 20.553, 19.649, and 17.878ms respectively, but those isolated
wins did not survive the end-to-end benchmark.

### Initialization-policy boundary

The local E1 issue in `simd/lookups/gkr.rs::gen_eq_evals` was resolved with safe zero
initialization, so it does not block E1 or E3. The same `set_len`/uninitialized-buffer
contract exists repo-wide through `core/utils.rs::uninit_vec`, FFT/FRI/quotient callers,
and `SecureColumn::uninitialized`. That pre-existing policy question remains deliberately
out of scope: decide repo-wide whether to document and accept the strict initialization
invariants or migrate consistently to `MaybeUninit`/zero-init where it is effectively free.
Do not make piecemeal changes from a performance workstream.

The sections below retain the earlier scoping, W3, and profiling history. Where they say
the end-to-end comparison or E3 was still pending, this final record supersedes them.

## Working-tree inventory and ownership warning

Preserve the dirty worktree. Some GKR performance changes existed before this audit and
must not be overwritten or casually attributed to this pass.

Current tracked modifications:

| File | Current purpose/status |
|------|------------------------|
| `crates/stwo/src/prover/backend/simd/lookups/gkr.rs` | Existing parallel GKR kernels plus newly added CPU/SIMD proof-and-artifact parity tests. Soundness-critical; do not accept or commit without the reviews below. |
| `crates/stwo/src/prover/backend/simd/lookups/mle.rs` | Existing parallel SIMD MLE folding. Soundness-critical. |
| `crates/stwo/src/prover/lookups/utils.rs` | Existing batched Lagrange denominator inversion. Soundness-critical prover utility. Do not modify further under the current spec. |
| `crates/stwo/benches/lookups.rs` | W2 benchmark coverage added for SIMD at 2^16 and 2^20; CPU and batch-4 remain at 2^16. |
| `crates/constraint-framework/src/mle_eval.rs` | Unrelated formatting/import changes; exclude from any GKR performance commit. |

The `tasks/` directory is currently untracked and contains the specification, plan,
measurements, lessons, and this handoff. Several `.DS_Store` files are also untracked and
unrelated. Do not stage them with GKR work.

## Current uncommitted implementation

The proof-path implementation already present in the worktree does the following:

1. Routes four SIMD GKR sum kernels through a chunked `sum_packed_terms` reduction.
2. Uses Rayon for the sum kernels when the `parallel` feature is enabled.
3. Parallelizes `gen_eq_evals` and the four next-layer generation paths.
4. Parallelizes both SIMD `fix_first_variable` implementations.
5. Uses a fixed `PACKED_CHUNK_SIZE` of `1 << 7`.
6. Batches Lagrange denominator inversions in `interpolate_lagrange`, reducing twelve
   secure-field inversions per call to one batch inversion.

Safe additions completed during the scoping pass:

- W2 benchmark harness: all four single-instance SIMD variants now run at both 2^16 and
  2^20. CPU benchmarks and the grand-product batch-4 benchmark remain at 2^16.
- Field-by-field CPU/SIMD parity coverage for all four GKR variants at 2^6, 2^7, and
  multi-chunk 2^14.
- Unequal-size batch parity coverage at 2^6 and 2^14.
- Comparisons cover sumcheck round polynomials, layer masks, output claims, OOD points,
  returned input claims, and per-instance variable counts.
- W3 isolated serial/parallel measurements and profiling.
- Corrections to W4's next-round SIMD cutoff and its required test boundaries in the
  implementation specification.

No W1 or W4 proof-path implementation was added during the audit/profile passes.

## Verification history and final gates

The following gates passed on the current proof/benchmark code state:

| Gate | Result |
|------|--------|
| final E1+E2 `stwo`, `prover,parallel`, library tests | 282 passed |
| final serial `stwo` lookup tests | 30 passed |
| `stwo-constraint-framework` tests | 24 passed |
| release `stwo-examples` GKR e2e tests | 4 passed, measurement test ignored |
| serial benchmark compilation | passed |
| parallel benchmark compilation | passed |
| full clippy script | passed |
| rustfmt script | passed |
| `git diff --check` | passed |

Commands for a fresh verification:

```bash
rtk cargo test --features "prover,parallel" -p stwo --lib
rtk cargo test --features "prover" -p stwo --lib -- lookups
rtk cargo test --features "prover,parallel" -p stwo-constraint-framework
rtk cargo check --features "prover" -p stwo
rtk proxy scripts/clippy.sh
rtk proxy scripts/rust_fmt.sh
rtk git diff --check
```

E1, E2, and the temporary E3 each passed the full gates. The table above is the fresh
post-revert verification of the accepted E1+E2 tree. Run the full set again after any
future proof-path change. Test additions must be retained.

## Earlier 2^16 measurements

These are historical measurements from the current workstream, not an isolated current
serial/parallel comparison. They explain why the fixed chunk size remains provisional.

| SIMD benchmark at 2^16 | Original baseline | Fixed chunk 2^10 | Fixed chunk 2^7 |
|------------------------|-------------------|-------------------|-----------------|
| Grand product | 2.93 ms | 2.64 ms | 3.03 ms |
| Generic LogUp | 5.95 ms | 5.54 ms | 5.04 ms |
| Multiplicities LogUp | 5.33 ms | 5.15 ms | 4.92 ms |
| Singles LogUp | 4.70 ms | 4.57 ms | 4.47 ms |

Original CPU reference times at 2^16 were 7.75 ms, 17.72 ms, 15.82 ms, and 14.35 ms
respectively. The SIMD/CPU grand-product batch-4 references were 10.98 ms and 29.95 ms.

At 2^16, the first sumcheck round has only 2^10 packed terms. A 2^10 chunk creates one
task and starves parallelism, while 2^7 adds noticeable scheduling overhead to the lighter
grand-product kernel. This motivated W1's adaptive-chunk proposal and the 2^20 coverage.

## Benchmark methodology correction

The first W3 comparison is invalid and must never be reused:

| Benchmark | Invalid serial | Invalid parallel | Apparent speedup |
|-----------|----------------|------------------|------------------|
| Grand product | 12.921 ms | 13.314 ms | 0.970x |
| Generic LogUp | 23.570 ms | 24.596 ms | 0.958x |
| Multiplicities LogUp | 22.207 ms | 24.604 ms | 0.903x |
| Singles LogUp | 20.724 ms | 21.863 ms | 0.948x |

The shared `target/` directory supplied a serial executable dated 2026-07-04 and a current
parallel executable. That was an artifact comparison, not a feature comparison.

All future feature A/B benchmarks must:

1. Build serial and parallel variants into distinct target directories.
2. Inspect each Cargo fingerprint before timing.
3. Verify the benchmark executable's exact Criterion filter with `--list`.
4. Run serial and parallel processes sequentially, never concurrently.
5. Compare absolute estimates; Criterion change percentages from separate target
   directories are not comparable.

Verified current artifacts used for the corrected run:

| Variant | Features | SHA-256 |
|---------|----------|---------|
| Serial | `default,prover,std` | `e6e4f610561f827d82334d8c8e718df06dde55320f1b05a882beaab208e7b5fc` |
| Parallel | `default,parallel,prover,rayon,std` | `1697fd00b0231aa95729c8d7bd6fc37e712a43a582088923849aaa342fb2a8a2` |

Those binaries live under `/private/tmp/gkr-current-{serial,parallel}` and are temporary;
rebuild rather than assuming they still exist.

## Corrected W3 results

Machine: Apple Silicon, 12 cores, 8 performance cores.  
Protocol: Criterion default 3-second warm-up and 100 samples per cell.  
Workload: all four single-instance SIMD GKR variants at 2^20 rows.

| Benchmark | Serial ms (95% interval) | Parallel 12 ms (95% interval) | Speedup | Parallel 8 ms (95% interval) |
|-----------|--------------------------|--------------------------------|---------|--------------------------------|
| Grand product | 35.700 (35.661–35.738) | 12.894 (12.816–12.977) | 2.77x | 12.015 (11.987–12.044) |
| Generic LogUp | 83.390 (83.222–83.557) | 23.141 (22.953–23.453) | 3.60x | 22.546 (22.500–22.593) |
| Multiplicities LogUp | 74.910 (74.774–75.056) | 21.964 (21.855–22.095) | 3.41x | 22.187 (21.985–22.406) |
| Singles LogUp | 65.491 (65.416–65.565) | 20.877 (20.687–21.080) | 3.14x | 21.064 (20.553–21.669) |

Decisions:

- Twelve-worker geometric-mean speedup over serial: **3.21x**.
- Every cell clears the W3 `<2x` stop-and-profile gate.
- The three LogUp variants miss the aspirational 4x parallel target.
- Eight workers reduce geometric-mean time by only 1.92% versus twelve workers.
- Eight workers help grand product and generic but are slightly slower for
  multiplicities and singles; no worker-default change is justified.
- The earlier short-run 8-worker multiplicities improvement did not reproduce.

Criterion baselines are retained under:

```text
target/criterion/*/gkr-w3-full-serial/
target/criterion/*/gkr-w3-full-parallel-12/
target/criterion/*/gkr-w3-full-parallel-8/
```

The profile pass first used a shorter 30-sample multiplicities diagnostic:

| Configuration | Estimate | 95% interval | Speedup over diagnostic serial |
|---------------|----------|--------------|---------------------------------|
| Serial | 72.926 ms | 72.797–73.070 | 1.00x |
| Parallel, 12 workers | 22.079 ms | 21.506–22.761 | 3.30x |
| Parallel, 8 workers | 20.721 ms | 20.651–20.793 | 3.52x |
| Parallel, 4 workers | 24.776 ms | 24.537–25.181 | 2.94x |

This diagnostic is valid but superseded by the full W3 matrix for decisions. In
particular, its apparent 8-worker multiplicities advantage did not reproduce.

## Profile results

Fresh ten-second macOS `sample` captures were taken from exact verified multiplicities
executables at 2^20.

Serial profile, 7,565 snapshots:

| Hot path | Samples |
|----------|---------|
| `PackedQM31::mul` | 77.20% |
| `prove_batch` residue | 10.18% |
| secure-field MLE fold | 3.85% |
| fraction addition | 3.09% |
| next-layer generation | 2.93% |

Parallel profile, 73,502 thread-slot snapshots over the main thread and twelve workers:

| Hot path/state | Samples |
|----------------|---------|
| condition wait | 30.04% |
| scheduler switch | 29.96% |
| `PackedQM31::mul` | 26.15% |
| generic LogUp sum closure | 2.85% |
| mutex wait | 1.44% |
| secure-field MLE fold | 1.28% |

The parallel wait/switch percentages include barriers and idle worker slots. They are not
proof wall-time percentages and do not indicate a regression; the corresponding benchmark
is 3.30x faster than serial. They do indicate diminishing parallel utilization as
sumcheck rounds shrink.

Raw temporary captures:

- `/private/tmp/gkr-profile-wEtDpl/serial-current-10s.sample.txt`
- `/private/tmp/gkr-profile-wEtDpl/parallel-current-10s.sample.txt`

The profiler can LTO-fold the outer benchmark symbol to `bench_gkr_grand_product`. The
captured inner symbols and the exact Criterion filter identify the multiplicities path;
do not classify a profile from the outer symbol alone.

## Comparison with production prefix-sum LogUp

Recorded raw reference:

- SIMD prefix sum: 6.46 ms at 2^24 rows.
- Current parallel GKR LogUp: 20.877–23.141 ms at 2^20 rows.

This establishes that the current raw GKR prover kernel is still substantially slower.
The 3.21x W3 result is only parallel GKR versus serial GKR; it is not a speedup over LogUp.

The comparison is not protocol-equivalent. Prefix-sum LogUp also requires interaction
generation, interaction-trace commitment, and constraint evaluation. GKR avoids the
committed interaction trace. The repository does not yet contain the agreed apples-to-
apples end-to-end harness needed to determine whether those savings overcome the slower
GKR kernel.

Do not claim that GKR wins or loses end-to-end until that harness reports total prover
time, peak memory, and proof size for the same lookup workload.

## Scope and blockers

### W0 — accept and commit the current proof-path changes

**The local E1 gate is resolved.** `gen_eq_evals` now uses safe zero initialization, and
E1 was approved and committed. The other pre-existing W0 proof-path changes remain dirty
and user-owned; this workstream neither accepted nor committed them.

The same strict `set_len`/uninitialized-vector contract remains elsewhere in the prover.
Treat that as a separate repo-wide policy decision, as described in the final experiment
record above, rather than a blocker on E1 or a reason for piecemeal edits here.

### W1 — adaptive chunk sizing

**Not implemented and superseded.** E1's cross-instance parallelism changed the scheduling
regime, and the final-experiment spec explicitly removed W1 from scope. Table T1 remains
empty by decision, not as unfinished final work.

The full worker-count experiment shows that scheduler tuning is a small opportunity: eight
workers improved aggregate time by only 1.92%. Do not expect W1 alone to change the
GKR-versus-prefix-sum conclusion.

### W2 — 2^16 and 2^20 benchmark coverage

**Implemented and verified.** Keep the benchmark structure currently in
`crates/stwo/benches/lookups.rs`.

### W3 — isolated serial/parallel A/B

**Complete.** Results and artifact rules are recorded above.

### W4 — fuse fold with the next round's sum

**Implemented as E3, reviewed, measured, rejected, and reverted.** The implementation used
the corrected cutoff below, passed boundary/parity tests, and received 98% Math Reviewer
confidence. It improved isolated kernels by 10.7%–16.5%, but regressed end-to-end Path B
by 4.9%–7.8%, so no W4/E3 code remains in the accepted tree.

The first W4 draft used the wrong SIMD cutoff. It used `2^(V-2)`, but the next round has
`2^(V-3)` packed terms. The wrong cutoff can select the fused SIMD path when no complete
SIMD sum term exists and can cache zero sums. The corrected spec requires:

```text
(1 << layer_variables.saturating_sub(3)) >= N_LANES
```

Workflow completed during E3:

1. A Crypto Specialist implements the corrected design.
2. A Math Reviewer checks the paper/algorithm correspondence, cutoff, fold formula,
   correction factor, reduction order, and transcript identity.
3. Tests cover 2^6 fallback, 2^7 exact cutoff, a larger fused case, and unequal batches.
4. Tests compare each fused folded layer and cached raw sum directly with the ordinary
   two-pass result.
5. Field-by-field proof/artifact parity remains green.
6. The performance acceptance gate rejected the experiment before any commit was made.

### W5 and remaining result tables

**Complete for the final decision.** E1, E2, and E3 before/after tables and the complete
final e2e/kernel tables are in `tasks/gkr-final-experiments-spec.md`. The optimized e2e
matrix is mirrored in `tasks/gkr-e2e-spec.md`. T1 remains intentionally empty because W1
was superseded.

### Deferred or disproven ideas

- Lambda hoisting: disproven; it saves no packed multiplications.
- Eager layer freeing: disproven; reversed `IntoIter` already drops layers as consumed.
- Evaluate at `t = 1/2`: deferred; it changes interpolation nodes and needs human and
  mathematical review.
- CPU fused kernels: intentionally out of scope; the default trait path remains two-pass.
- Changes to `core/`, verifier, channel, proof structs, or proof format: out of scope.
- End-to-end GKR-versus-interaction-trace harness: completed and measured.

## Credible path forward

The bounded path forward was executed: end-to-end harness, E1 cross-instance parallelism,
E2 glue parallelism, and E3 fused rounds. It did not produce a crossover. This workstream
is closed by its own decision rule.

Do not queue W1, re-land E3, or start another local micro-optimization. If GKR lookup is
revisited, require an architectural hypothesis—such as substantially different batching or
amortization, a field-operation redesign, or accelerator work—plus a fresh spec and
end-to-end stop gate. Independently, make a repo-wide decision on the pre-existing
uninitialized-buffer policy.

## Reproduction commands

Build isolated current variants:

```bash
rtk proxy cargo bench --target-dir /private/tmp/gkr-w3-serial \
  --features "prover" -p stwo --bench lookups --no-run
rtk proxy cargo bench --target-dir /private/tmp/gkr-w3-parallel \
  --features "prover,parallel" -p stwo --bench lookups --no-run
```

After locating the single `lookups-*` executable in each target directory, verify its
fingerprint and list the exact cells before benchmarking:

```bash
rtk proxy /private/tmp/gkr-w3-serial/release/deps/lookups-<hash> \
  'simd .* lookup 2\^20$' --list
rtk proxy /private/tmp/gkr-w3-parallel/release/deps/lookups-<hash> \
  'simd .* lookup 2\^20$' --list
```

Run serial and parallel sequentially. Include `--bench`; omitting it only exercises
Criterion's test mode:

```bash
rtk proxy /private/tmp/gkr-w3-serial/release/deps/lookups-<hash> \
  'simd .* lookup 2\^20$' --bench --noplot --save-baseline gkr-w3-serial
rtk proxy env RAYON_NUM_THREADS=12 \
  /private/tmp/gkr-w3-parallel/release/deps/lookups-<hash> \
  'simd .* lookup 2\^20$' --bench --noplot --save-baseline gkr-w3-parallel-12
```

## Source documents

- `tasks/gkr-final-experiments-spec.md`: authoritative E1/E2/E3 experiment record,
  accepted artifact, final matrix, phase rows, kernel means, and close decision.
- `tasks/gkr-e2e-spec.md`: baseline and final optimized end-to-end matrices.
- `tasks/gkr-perf-spec.md`: executable work-item specification and result tables.
- `tasks/gkr-perf-plan.md`: background, prior profile, decisions, and phase status.
- `tasks/todo.md`: audit trail for the scoping, profiling, and full W3 runs.
- `tasks/lessons.md`: stale-artifact and profiling-gate lessons.

If this handoff conflicts with raw code or fresh measurements, stop and reconcile the
difference before modifying any soundness-critical file.
