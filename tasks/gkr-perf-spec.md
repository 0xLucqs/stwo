# GKR prover performance — implementation spec

Self-contained spec. An agent with no prior context should be able to execute every work
item below without making design decisions. Companion doc with background/measurements:
`tasks/gkr-perf-plan.md`.

## Context

- Repo: /Users/lucas/stwo, branch `dev-copy` (push to fork/dev-copy only; base was 4f877db2).
- Goal: make the GKR lookup prover (`prove_batch` in
  `crates/stwo/src/prover/lookups/gkr_prover.rs`) competitive with the prefix-sum LogUp
  baseline so the LogUp-GKR protocol (no committed interaction trace) wins end-to-end.
- Machine used for all recorded numbers: Apple Silicon, 12 cores (8 performance).

## Global constraints (apply to every work item)

1. PROVER-ONLY. Never modify: `crates/stwo/src/core/**` (verifier, channel, transcript,
   fields), `gkr_verifier.rs`, proof structs, or anything the verifier reads. The
   sumcheck round-poly VALUES sent to the channel must be bit-identical before/after
   every change. The existing assert at `sumcheck.rs:95`
   (`assert_eq!(eval_at_0 + eval_at_1, claim)`) and GKR roundtrip tests enforce validity,
   but not bit identity: field-by-field CPU/SIMD proof parity is the regression gate.
2. Files you may modify (whitelist):
   - `crates/stwo/src/prover/backend/simd/lookups/gkr.rs`
   - `crates/stwo/src/prover/backend/simd/lookups/mle.rs`
   - `crates/stwo/src/prover/lookups/gkr_prover.rs` (prover-side only; see W4 for the
     exact allowed change)
   - `crates/stwo/src/prover/lookups/utils.rs` (already changed; do not change further)
   - `crates/stwo/benches/lookups.rs`
   - `tasks/gkr-perf-plan.md`, `tasks/gkr-perf-spec.md` (record results)
3. After EVERY work item, all of these must pass:
   ```bash
   cargo test --features "prover,parallel" -p stwo --lib
   cargo test --features "prover" -p stwo --lib -- lookups
   cargo test --features "prover,parallel" -p stwo-constraint-framework
   cargo check --features "prover" -p stwo        # serial build must compile
   scripts/clippy.sh
   scripts/rust_fmt.sh
   ```
4. Benchmark protocol: `cargo bench --features "prover,parallel" --bench lookups -- simd`.
   Never run two benchmark processes concurrently. Record criterion mean times in the
   results tables at the bottom of this file.
5. Do NOT add a co-author line to commits.

## Already done (uncommitted working-tree changes — do not redo)

- `simd/lookups/gkr.rs`: 4 sum kernels routed through chunked `sum_packed_terms` helper
  (parallel via `parallel_iter!`); `next_*_layer` x4 and `gen_eq_evals` parallelized via
  `par_iter`/`with_min_len`; `PACKED_CHUNK_SIZE` const (currently `1 << 7`).
- `simd/lookups/mle.rs`: both `fix_first_variable` impls parallelized.
- `prover/lookups/utils.rs`: `interpolate_lagrange` batches denominator inversions
  (12 QM31 inversions/call → 1).
- All gates in (3) pass on this state.

Measured so far (criterion means, 2^16 rows, 12-core machine, `parallel` on):

| bench (simd)          | baseline | chunk 2^10 | chunk 2^7 |
|-----------------------|----------|------------|-----------|
| grand product         | 2.93 ms  | 2.64 ms    | 3.03 ms   |
| generic logup         | 5.95 ms  | 5.54 ms    | 5.04 ms   |
| multiplicities logup  | 5.33 ms  | 5.15 ms    | 4.92 ms   |
| singles logup         | 4.70 ms  | 4.57 ms    | 4.47 ms   |

Interpretation: at 2^16 the largest sumcheck round has only 2^10 packed terms, so a fixed
chunk either starves parallelism (2^10 → 1 task) or drowns light kernels in scheduling
overhead (grand product at 2^7). Hence W1 (adaptive chunk) and W2/W3 (measure at 2^20,
where production behavior lives).

### Review prerequisites before further proof-path changes

- `gen_eq_evals` uses an existing unsafe `Vec::set_len` before initialization. Its safety
  invariant must be documented and reviewed before accepting W0/W1, especially now that the
  initialization loop is parallel.
- Field-by-field CPU/SIMD proof parity now covers all four layer variants at 2^6, 2^7, and
  2^14, including round polynomials, masks, output claims, OOD points, returned input claims,
  and an unequal-size batch. Keep this test as the W0/W1 transcript regression gate.

---

## W0 — Commit the current state

**UNBLOCKED (2026-07-21):** the unsafe `gen_eq_evals` initialization was replaced with a
safe zero-initialized `resize` (no `set_len`, no uninit slices); all gates pass. Human
sign-off on the commit itself is still required per the repo workflow.

Single commit of the working-tree changes to `gkr.rs`, `mle.rs`, `utils.rs`:
message `perf(lookups): parallelize GKR/sumcheck kernels + batch Lagrange inversions`.
Do not include `tasks/` files in this commit; commit those separately.

## W1 — Adaptive chunk size

**SUPERSEDED:** the W0 unsafe gate is resolved, but W1 is retired by
`tasks/gkr-final-experiments-spec.md` (E1's cross-instance parallelism changes the
scheduling regime W1 would have tuned). Do not implement unless a human reopens it.

File: `crates/stwo/src/prover/backend/simd/lookups/gkr.rs`.

Replace the fixed-chunk logic in `sum_packed_terms` with an adaptive chunk. Exact code:

```rust
/// Minimum packed terms per parallel task. Set by the W1 experiment (see
/// tasks/gkr-perf-spec.md); tasks smaller than this lose more to rayon scheduling
/// than they gain from parallelism.
pub(crate) const MIN_PACKED_CHUNK_SIZE: usize = 1 << 8; // provisional until W1 table filled

/// Packed terms per task: split the range into ~4 tasks per thread for load balance,
/// but never below MIN_PACKED_CHUNK_SIZE. Chunking never affects results (exact field
/// addition), only scheduling.
fn packed_chunk_size(n_packed_terms: usize) -> usize {
    #[cfg(feature = "parallel")]
    {
        (n_packed_terms / (4 * rayon::current_num_threads())).max(MIN_PACKED_CHUNK_SIZE)
    }
    #[cfg(not(feature = "parallel"))]
    {
        n_packed_terms.max(1)
    }
}
```

In `sum_packed_terms`, replace every use of `PACKED_CHUNK_SIZE` with a
`let chunk_size = packed_chunk_size(n_packed_terms);` computed once at the top
(`n_chunks = n_packed_terms.div_ceil(chunk_size)`, `start = chunk * chunk_size`, etc.).

Rename `PACKED_CHUNK_SIZE` → `MIN_PACKED_CHUNK_SIZE` everywhere it is used as a
`with_min_len(..)` argument (`next_*_layer` x4, `gen_eq_evals` in gkr.rs;
both `fix_first_variable` in mle.rs — that file imports it, keep the
`#[cfg(feature = "parallel")]` on the import).

### Tuning experiment (mechanical)

For MIN in {1<<7, 1<<8, 1<<9, 1<<10}: set the const, run the bench protocol (W2 must be
done first so 2^20 is included), record all simd means in Table T1 below. Selection
rule: for each MIN compute the geometric mean of (time / best-time-for-that-bench)
across all 8 (bench x size) cells; pick the MIN with the smallest geometric mean.
Set the const to the winner, note the choice in Table T1, commit
(`perf(lookups): adaptive GKR chunk sizing`).

## W2 — Bench at 2^16 and 2^20

File: `crates/stwo/benches/lookups.rs`.

Currently `const LOG_N_ROWS: u32 = 16;` and every `bench_*` function uses it. Change:
each `bench_gkr_*` function gains a `log_n_rows: u32` parameter replacing the const
(bench names already interpolate `{LOG_N_ROWS}` — switch to `{log_n_rows}`), and
`gkr_lookup_benches` calls the whole set for `log_n_rows` in `[16, 20]`. Run CPU
benches only at 16 (at 20 they take minutes and add no information):

```rust
fn gkr_lookup_benches(c: &mut Criterion) {
    for log_n_rows in [16, 20] {
        bench_gkr_grand_product::<SimdBackend>(c, "simd", log_n_rows);
        bench_gkr_logup_generic::<SimdBackend>(c, "simd", log_n_rows);
        bench_gkr_logup_multiplicities::<SimdBackend>(c, "simd", log_n_rows);
        bench_gkr_logup_singles::<SimdBackend>(c, "simd", log_n_rows);
    }
    bench_gkr_grand_product::<CpuBackend>(c, "cpu", 16);
    bench_gkr_logup_generic::<CpuBackend>(c, "cpu", 16);
    bench_gkr_logup_multiplicities::<CpuBackend>(c, "cpu", 16);
    bench_gkr_logup_singles::<CpuBackend>(c, "cpu", 16);
    bench_gkr_grand_product_batch::<SimdBackend>(c, "simd", 16);
    bench_gkr_grand_product_batch::<CpuBackend>(c, "cpu", 16);
}
```

Keep `gen_random_mle` as-is. The grand-product batch-4 benchmark remains at 2^16 only;
it is not one of the eight W1 tuning cells. Commit with the W1 tuning result.

## W3 — Serial-vs-parallel A/B at 2^20 (record only, no code)

The serial and parallel builds must use the same current algorithms, so this measures
current feature scaling. It is not a pre-change comparison: both builds include the
batched Lagrange inversion and refactored reduction paths. Build into distinct target
directories; selecting the newest executable from a shared `target/` directory is not a
valid A/B because a stale feature variant can survive there.

```bash
rtk proxy cargo bench --target-dir /private/tmp/gkr-w3-serial \
  --features "prover" -p stwo --bench lookups --no-run
rtk proxy cargo bench --target-dir /private/tmp/gkr-w3-parallel \
  --features "prover,parallel" -p stwo --bench lookups --no-run
```

Verify the serial fingerprint omits `parallel`/`rayon`, the parallel fingerprint includes
them, and each executable's escaped Criterion filter lists the intended targets before
running it. Record absolute estimates from both target directories in Table T2; Criterion
"change" percentages are not comparable across the isolated directories and are not
enough on their own.

Expected: ≥4x on logup kernels at 2^20. If parallel/serial < 2x at 2^20, stop and
re-profile the exact verified executable from `/private/tmp/gkr-w3-parallel` before
proceeding to W4.

Correction, 2026-07-21: the earlier claim that all four parallel results were slower
(0.90–0.97x) is invalid. Its serial executable was dated 2026-07-04 while the parallel
executable was current. The completed isolated W3 matrix measures 2.77–3.60x speedups with
12 workers, so no cell activates the stop condition. The 4x LogUp target was not reached.
An 8-worker confirmation improved geometric-mean time by only 1.92% with mixed per-cell
results, which does not justify a default configuration change. W3 is complete; W1 and W4
remain subject to their independent safety and review blocks.

## W4 — Fused fold + next-round-sum (one pass per sumcheck round instead of two)

**PRE-IMPLEMENTATION MATH APPROVED (2026-07-21):** the corrected cutoff, fixed-chunk
reduction order, cached raw sums, correction factor, and four layer transitions received
Math Reviewer sign-off. A Crypto Specialist must implement it, a Math Reviewer must perform
the post-implementation review, and a human must give final approval for the new fused
kernels. The pre-existing uninitialized-buffer policy does not block W4/E3.

### Why

Per sumcheck round the layer is streamed twice: once by `sum_as_poly_in_first_variable`
(compute f(0), f(2)), once by `fix_first_variable` (fold with the challenge). After the
challenge is drawn, one pass can write the folded layer AND accumulate the raw sums for
the NEXT round. Kernels are memory-bound, so this is ~40% traffic reduction per round.

### Design (exact)

**(a)** `crates/stwo/src/prover/lookups/gkr_prover.rs`:

Add to the `GkrOps` trait (with default impl so `CpuBackend` is untouched):

```rust
    /// Fixes the first variable of `layer` to `challenge` and, when supported, also
    /// returns the raw sums `(f'(0), f'(2))` of the NEXT sumcheck round computed over
    /// the folded layer in the same pass (without the `eq_fixed_var_correction`
    /// factor, exactly as `sum_as_poly_in_first_variable`'s inner kernels return them).
    /// Returning `None` means the caller falls back to the two-pass path.
    fn fix_first_variable_and_sum(
        layer: Layer<Self>,
        challenge: SecureField,
        _eq_evals: &EqEvals<Self>,
        _lambda: SecureField,
    ) -> (Layer<Self>, Option<(SecureField, SecureField)>) {
        (layer.fix_first_variable(challenge), None)
    }
```

Add field to `GkrMultivariatePolyOracle` (after `lambda`):

```rust
    /// Raw next-round sums produced by the fused fold pass (see
    /// `GkrOps::fix_first_variable_and_sum`). `None` when not precomputed.
    pub next_round_raw_sums: Option<(SecureField, SecureField)>,
```

Initialize `next_round_raw_sums: None` in `into_multivariate_poly` (gkr_prover.rs:230)
and in `to_cpu()` (gkr_prover.rs:375).

Change `MultivariatePolyOracle::fix_first_variable` for the oracle (gkr_prover.rs:311):

```rust
    fn fix_first_variable(self, challenge: SecureField) -> Self {
        if self.is_constant() {
            return self;
        }

        let z0 = self.eq_evals.y()[self.eq_evals.y().len() - self.n_variables()];
        let eq_fixed_var_correction = self.eq_fixed_var_correction * eq(&[challenge], &[z0]);

        // Only fuse when there IS a next round for this oracle (n_variables() > 1
        // pre-fix means the folded oracle still has >= 1 variable to sum over).
        let (input_layer, next_round_raw_sums) = if self.n_variables() > 1 {
            B::fix_first_variable_and_sum(self.input_layer, challenge, &self.eq_evals, self.lambda)
        } else {
            (self.input_layer.fix_first_variable(challenge), None)
        };

        Self {
            eq_evals: self.eq_evals,
            eq_fixed_var_correction,
            input_layer,
            lambda: self.lambda,
            next_round_raw_sums,
        }
    }
```

**(b)** `crates/stwo/src/prover/backend/simd/lookups/gkr.rs`:

In `impl GkrOps for SimdBackend`, at the TOP of `sum_as_poly_in_first_variable`
(before the CPU-offload check), consume the cache:

```rust
        if let Some((eval_at_0, eval_at_2)) = h.next_round_raw_sums {
            return correct_sum_as_poly_in_first_variable(
                eval_at_0 * h.eq_fixed_var_correction,
                eval_at_2 * h.eq_fixed_var_correction,
                claim,
                y,
                n_variables,
            );
        }
```

(`y`/`n_variables` are already computed at the top of the function; keep ordering so
they exist. This mirrors lines 174-176 exactly — same correction, same call.)

Implement `fix_first_variable_and_sum` for `SimdBackend`. Gate: run the fused kernel
only when the FOLDED layer's sum kernel would run on SIMD, i.e. when
`(1usize << layer.n_variables().saturating_sub(3)) >= N_LANES`
(next-round `n_terms >= N_LANES`);
otherwise return the default `(layer.fix_first_variable(challenge), None)`.

Access-pattern math for the fused kernel (all indices in PACKED units; derived from the
existing code and verified against `eval_grand_product_sum`'s bounds):

Key relation (from `sum_as_poly_in_first_variable`, gkr.rs: `n_terms =
1 << n_variables.saturating_sub(1)` where the oracle's `n_variables = layer_vars - 1`):
**a sum kernel over a column of packed length L uses `n_packed_terms = L / 4`** and term
`i` reads packed positions `{2i, 2i+1, L/2 + 2i, L/2 + 2i + 1}` (max index `L - 1`).

Let the input layer column have packed length `n_p`. The fold (challenge `r`) writes a
column of packed length `m = n_p / 2`:

```text
folded[p] = old[p] + r * (old[p + m] - old[p])        for p in 0..m
```

The NEXT round's sum kernel over the folded column uses `n_next = m / 4` packed terms;
term `i` reads `folded[{2i, 2i+1, m/2 + 2i, m/2 + 2i + 1}]`.

Fused loop: `for i in 0..m/4`, compute the four folded packed vectors at positions
`{2i, 2i+1, m/2 + 2i, m/2 + 2i + 1}` (each needs 2 old reads → 8 reads total), write
them, and accumulate sum-term `i` from those in-register values. Coverage check:
`{2i, 2i+1 : i < m/4}` covers `[0, m/2)` exactly once and `{m/2+2i, m/2+2i+1}` covers
`[m/2, m)` exactly once — the loop writes the whole folded column, each position once,
so the pass is complete and the writes are disjoint (parallelizable by i-chunks).

Kernel structure per layer variant (4 variants; write LogUpGeneric first, others are
mechanical transcriptions of their existing `eval_*_sum` + `fix` semantics):

```rust
fn fused_fix_and_sum_logup_generic(
    numerators: Mle<SimdBackend, SecureField>,
    denominators: Mle<SimdBackend, SecureField>,
    challenge: SecureField,
    eq_evals: &EqEvals<SimdBackend>,
    lambda: SecureField,
) -> (Layer<SimdBackend>, (SecureField, SecureField)) {
    let r = PackedSecureField::broadcast(challenge);
    let packed_lambda = PackedSecureField::broadcast(lambda);
    let old_n = &numerators.data;   // packed len n_p
    let old_d = &denominators.data;
    let m = old_n.len() / 2;        // folded packed len
    // Allocate the two folded columns with safely initialized storage of len m each.
    // Loop i in 0..m/4 (chunked exactly like sum_packed_terms, same chunk_size fn,
    // with disjoint output regions per chunk so it parallelizes with
    // par_chunks_mut on the two halves of each output column zipped together):
    //   for each of the 4 target positions p in {2i, 2i+1, m/2+2i, m/2+2i+1}:
    //     folded_n[p] = old_n[p] + r * (old_n[p + m] - old_n[p])
    //     folded_d[p] = old_d[p] + r * (old_d[p + m] - old_d[p])
    //   then, using folded values ALREADY IN REGISTERS, replicate the body of
    //   eval_logup_generic_sum for term i (deinterleave pairs, t=2 extrapolation,
    //   Fraction sums, eq_evals.data[i], packed_lambda) accumulating
    //   (acc_at_0, acc_at_2) per chunk; reduce chunks exactly like sum_packed_terms.
    // Return (Layer::LogUpGeneric { folded... }, (sum0.pointwise_sum(), sum2...)).
}
```

Parallel write pattern (no unsafe): split each folded column into
`(lo, hi) = folded.split_at_mut(m / 2)`, then
`lo.par_chunks_mut(2 * C).zip(hi.par_chunks_mut(2 * C)).enumerate()` with
`C = PACKED_CHUNK_SIZE`; chunk index `c` handles `i` in `[c*C, min((c+1)*C, m/4))`
and writes only its own sub-slices; `.map(...)` returns the chunk's partial sums;
`.collect::<Vec<_>>()` then fold-reduce, exactly as in `sum_packed_terms`. Serial
(`not(parallel)`) branch must also produce the same indexed chunk partials before the
serial in-order fold. NOTE: W1's `packed_chunk_size` helper does not exist and W1 is out of
scope for the final experiments; partial-sum reduction order must match current
`sum_packed_terms`' fixed chunk order. Output is then bit-identical to the two-pass path.

Variant mapping:
- `Layer::GrandProduct(col)` → fold 1 column, sum body from `eval_grand_product_sum`.
- `Layer::LogUpGeneric` → 2 columns, body from `eval_logup_generic_sum`.
- `Layer::LogUpMultiplicities` → numerators are `Mle<_, BaseField>`; fold produces
  SecureField numerators (formula `n0 + r*(n1 - n0)` with PackedBaseField inputs and
  PackedSecureField output — same as `fold_packed_mle_evals` in mle.rs); resulting
  layer is `Layer::LogUpGeneric`; sum body from `eval_logup_generic_sum` (the sums are
  over the FOLDED = generic layer).
- `Layer::LogUpSingles` → 1 column, folded layer stays `LogUpSingles`, sum body from
  `eval_logup_singles_sum`.

### Equivalence obligations (why output is bit-identical)

1. Folded values: same formula as `fix_first_variable` (`a + r*(b - a)` via
   `fold_packed_mle_evals`) — use the same helper or identical expression.
2. Raw sums: same per-term arithmetic as `eval_*_sum` on the folded layer, same
   chunk-accumulation order as `sum_packed_terms`.
3. Consumed identically: cache path multiplies by the SAME `eq_fixed_var_correction`
   and calls the SAME `correct_sum_as_poly_in_first_variable` as the kernel path.

### New test (add to gkr.rs tests module)

```rust
    /// Locks in that SIMD (fused path) and CPU (two-pass path) produce identical proofs.
    #[test]
    fn simd_and_cpu_gkr_proofs_match() {
        const N: usize = 1 << 10;  // large enough that the fused SIMD path engages
        let mut rng = SmallRng::seed_from_u64(0);
        // For each of the 4 layer variants: build the same input on both backends,
        // prove_batch with test_channel(), assert_eq! on the two GkrBatchProofs'
        // sumcheck_proofs round_polys, layer masks, and output claims.
    }
```

The final W4 parity suite must cover 2^6 (fallback), 2^7 (exact SIMD cutoff), and 2^10
(fused path), plus a batch with unequal layer sizes. It must also compare each fused folded
layer and cached raw sum directly with the ordinary two-pass result.

(`SumcheckProof`/`UnivariatePoly` implement Deref to slices; compare via the fields.
If `GkrBatchProof` lacks PartialEq, compare field-by-field with `assert_eq!` on the
Deref'd slices — do NOT add derives to proof types, that touches shared code.)

Run gates (constraint-framework tests included — `mle_eval.rs` consumes
`GkrMultivariatePolyOracle`... it consumes `MultivariatePolyOracle` for `Mle`, which
uses the DEFAULT trait path and is unaffected). Then bench; record Table T3. Expected:
20-35% improvement on logup kernels at 2^20 over post-W1 numbers. Commit
(`perf(lookups): fuse sumcheck fold with next-round evaluation`).

## W5 — Final report

Completed by the later `tasks/gkr-final-experiments-spec.md`. Table T4 below records the
accepted E1+E2 kernels; the complete final end-to-end matrix and decision are in that spec.
The end-to-end comparison was subsequently authorized, built, and measured.

## Explicitly OUT of scope (do not implement)

- λ-hoist in sum kernels: disproven, saves zero multiplications.
- Eager GKR layer freeing: disproven, `vec::IntoIter` already frees as consumed.
- t=1/2 evaluation-point change: deferred pending human review (changes interpolation
  nodes in `correct_sum_as_poly_in_first_variable`).
- CPU-backend fused kernels: default trait impl keeps CPU on the two-pass path.
- Any change to `core/`, verifier, channel, or proof formats.
- End-to-end GKR-vs-interaction-trace AIR benchmark was out of scope for this kernel spec;
  it was later completed under `tasks/gkr-final-experiments-spec.md`.

## Result tables

### T1 — MIN_PACKED_CHUNK_SIZE tuning (parallel, means in ms)
| bench \ MIN            | 2^7 | 2^8 | 2^9 | 2^10 |
|------------------------|-----|-----|-----|------|
| grand product 2^16     |     |     |     |      |
| generic 2^16           |     |     |     |      |
| multiplicities 2^16    |     |     |     |      |
| singles 2^16           |     |     |     |      |
| grand product 2^20     |     |     |     |      |
| generic 2^20           |     |     |     |      |
| multiplicities 2^20    |     |     |     |      |
| singles 2^20           |     |     |     |      |
Winner: not selected; W1 was superseded by E1's cross-instance scheduling regime.

### T2 — serial vs parallel A/B at 2^20 (post-W1)

Invalid shared-target measurements (retained only as an audit trail; do not use):

| bench | serial ms | parallel ms | speedup |
|-------|-----------|-------------|---------|
| grand product | 12.921 | 13.314 | 0.970x |
| generic logup | 23.570 | 24.596 | 0.958x |
| multiplicities logup | 22.207 | 24.604 | 0.903x |
| singles logup | 20.724 | 21.863 | 0.948x |

Corrected isolated current-build result (default Criterion: 3-second warm-up and 100
samples per cell):

| bench | serial ms (95%) | parallel 12 ms (95%) | speedup | parallel 8 ms (95%) |
|-------|-----------------|-----------------------|---------|----------------------|
| grand product | 35.700 (35.661–35.738) | 12.894 (12.816–12.977) | 2.77x | 12.015 (11.987–12.044) |
| generic logup | 83.390 (83.222–83.557) | 23.141 (22.953–23.453) | 3.60x | 22.546 (22.500–22.593) |
| multiplicities logup | 74.910 (74.774–75.056) | 21.964 (21.855–22.095) | 3.41x | 22.187 (21.985–22.406) |
| singles logup | 65.491 (65.416–65.565) | 20.877 (20.687–21.080) | 3.14x | 21.064 (20.553–21.669) |

The invalid table mixed a stale 2026-07-04 serial binary with a current parallel binary.
The corrected table uses separate current target directories. The 12-worker
geometric-mean speedup is 3.21x. All cells clear the 2x stop gate, while the LogUp cells
miss the 4x target. Eight workers reduce geometric-mean time by only 1.92%, with faster
grand-product/generic cells but overlapping or slower multiplicities/singles results; do
not change the worker default from this result.

### T3 — W4 fused rounds delta (parallel)
| bench | pre-W4 ms | post-W4 ms | delta |
|-------|-----------|------------|-------|
| grand product 2^20 | 13.047 | 11.645 | -10.7% |
| generic LogUp 2^20 | 24.626 | 20.553 | -16.5% |
| multiplicities LogUp 2^20 | 22.660 | 19.649 | -13.3% |
| singles LogUp 2^20 | 20.718 | 17.878 | -13.7% |

Kernel-only W4 improved every cell, but the isolated interleaved end-to-end experiment
regressed both `gkr_prove` and Path B total at L=1 and L=64. Per the final-experiments
selection rule, W4/E3 is rejected and reverted; it is not part of the final configuration.

### T4 — final vs original baseline
| bench | original 2^16 ms | final 2^16 ms | 2^16 speedup | final 2^20 ms |
|-------|------------------|----------------|--------------|----------------|
| grand product | 2.93 | 2.8709 | 1.02x | 13.047 |
| generic LogUp | 5.95 | 4.7646 | 1.25x | 24.626 |
| multiplicities LogUp | 5.33 | 4.6373 | 1.15x | 22.660 |
| singles LogUp | 4.70 | 4.2370 | 1.11x | 20.718 |

These are the accepted E1+E2 means. E3's faster kernel numbers are retained in T3 only
as the audit trail for a rejected end-to-end experiment.
