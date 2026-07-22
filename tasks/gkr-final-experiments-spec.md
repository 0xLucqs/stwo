# GKR final experiments — closing the verdict

> **DOWNGRADED (2026-07-22).** The e2e verdict (no crossover, `gkr-e2e-spec.md`)
> is accepted as final without these experiments. If E1/E2 are already implemented
> and gated, land them (E1 benefits any sumcheck user incl. MleEval; E2 is trivial
> glue). **Do NOT start E3** — soundness-review cost on a path that won't ship.
> Do not run the final full-matrix measurement. The campaign has moved to
> `campaign-path.md` (EU-DI / P-256 focus).

Self-contained spec for an implementing agent with no prior context. This is the LAST
round of work on the GKR lookup path: three bounded experiments, then a final
measurement that either finds a crossover or closes the workstream with no asterisks.

Prior context (read before starting):
- `tasks/gkr-e2e-spec.md` — the end-to-end harness + current result tables (the
  baseline for everything here). Verdict so far: Path B (GKR) is 1.7-3.0x slower than
  Path A (interaction-trace LogUp) at every measured cell and grows FASTER with L.
- `tasks/gkr-perf-spec.md` — kernel work items; W4 there is Experiment E3 here.
- `tasks/gkr-perf-handoff.md` — benchmark methodology rules (stale artifacts, isolated
  target dirs). Binding for all measurements below.
- `tasks/gkr-perf-plan.md` — VERDICT section this work will finalize.

## Why these three, and the decision rule

Phase breakdown at 2^20, L=64 (median ms): Path B total 4,063 =
base_commit 288 + gkr_layers 433 + gkr_prove 1,664 + mle_combine 452 + mle_trace 12
+ mle_trace_commit 62 + stark_prove 1,134. Path A total 2,217.

- **E1** attacks `gkr_prove` (1,664): the 65 per-round oracle kernels run SERIALLY —
  cross-instance parallelism is unused. Measured within-instance utilization is only
  3.21x on 12 cores.
- **E2** attacks the glue (`gkr_layers` 433 + `mle_combine` 452): serial,
  allocation-heavy example-side code.
- **E3** attacks `gkr_prove` again (fused fold+sum, one stream per sumcheck round
  instead of two): the fully-specced W4 item.

**Decision rule (final):** after all three are evaluated, re-run the full e2e matrix on
the fastest verified cumulative subset. If Path B beats Path A in ANY cell, report the
crossover region. If not, the GKR-lookup verdict is final: append "CLOSED — no crossover
after E1-E3" to the VERDICT section of `tasks/gkr-perf-plan.md` and stop. Either way, do
not start further optimization work.

## Global constraints (all experiments)

1. Transcript identity is absolute: every change must produce bit-identical proofs.
   E1/E3 are order-preserving restructurings of prover-side computation; E2 is
   example-crate code. Never touch `crates/stwo/src/core/**`, `gkr_verifier.rs`,
   channel, or proof structs.
2. Gates after EVERY experiment (all must pass before moving on):
   ```bash
   cargo test --features "prover,parallel" -p stwo --lib
   cargo test --features "prover" -p stwo --lib -- lookups
   cargo test --features "prover,parallel" -p stwo-constraint-framework
   cargo test --release --features "parallel" -p stwo-examples gkr_e2e
   cargo check --features "prover" -p stwo
   scripts/clippy.sh
   scripts/rust_fmt.sh
   ```
3. Measurement protocol: release + `RUSTFLAGS="-C target-cpu=native"`, 12 workers,
   isolated target dirs for any cross-feature A/B (handoff rules), one warmup + 3 runs,
   record medians. Never run two measurement processes concurrently.
4. One commit per experiment after its gates pass (messages given per experiment).
   No co-author lines. `tasks/` files committed separately.
5. If an experiment's measured improvement at L=64/2^20 is under 5%, keep the change
   only if it is non-regressing within measurement noise and gates pass. Revert a measured
   regression; the final verdict uses the fastest verified cumulative subset, not necessarily
   all three experiments.
6. E1 and E3 are soundness-critical because they edit the GKR/sumcheck proof path. Both
   require Crypto Specialist implementation and Math Reviewer post-review. Human approval
   to proceed with E1 was recorded on 2026-07-21. E3 retains its separate human math-review
   gate for the newly fused kernels before commit.

### Soundness escalation — RESOLVED (2026-07-21)

`gen_eq_evals`'s `Vec::set_len`-before-init was replaced with a safe zero-initialized
allocation (`data.resize(packed_len, PackedSecureField::zero())`) and the stale
`#[allow(clippy::uninit_vec)]` removed. Cost is one memset-speed pass on a function
that was 0.7% of the GKR profile — negligible by construction. All gates pass on the
fix (281 parallel lib tests incl. the CPU/SIMD proof-parity suite, serial lookups,
constraint-framework, clippy, fmt). E1 and E3 are UNBLOCKED on this point; E3 retains
its own Math Reviewer + human-approval gate for the fused kernels themselves.

Out of scope here but noted: the same `set_len`/`uninit_vec` idiom exists elsewhere in
the prover (`core/utils.rs::uninit_vec` users in FFT/FRI/quotients, `SecureColumn::
uninitialized`) — a repo-wide policy question for a separate workstream, not this one.

## E1 — Cross-instance parallelism in sumcheck and GKR layer generation

Files: `crates/stwo/src/prover/lookups/sumcheck.rs`,
`crates/stwo/src/prover/lookups/gkr_prover.rs`. Nothing else.

### E1a — parallel round-poly computation and folds in `sumcheck::prove_batch`

Current (sumcheck.rs:84-122): per round, `this_round_polys` is a serial
`zip(&multivariate_polys, &claims).enumerate().map(...)` (each call runs a full
`sum_as_poly_in_first_variable` kernel), and the post-challenge fold is a serial
`multivariate_polys.into_iter().map(...fix_first_variable...)`. Both maps are
independent per oracle and MUST stay index-order-preserving (the round-poly RLC and
the claims vector depend on order).

Change both to rayon indexed parallel maps under `#[cfg(feature = "parallel")]`, with
the serial branch unchanged, following the exact `#[cfg]`-pair idiom used in
`crates/stwo/src/prover/backend/simd/lookups/gkr.rs` (`next_logup_generic_layer`).
`collect` on an indexed parallel iterator preserves order — the transcript is
unchanged. Keep the per-oracle asserts exactly where they are (they run inside the
parallel closure; a failed assert still panics).

Trait bounds: the parallel maps need `O: Send` (fold consumes oracles by value) and
`&O: Sync` for the shared borrow. Add `Send + Sync` to `prove_batch`'s generic bound
unconditionally (`pub fn prove_batch<O: MultivariatePolyOracle + Send + Sync>`).
Before accepting this, enumerate ALL in-repo `MultivariatePolyOracle` impls
(`grep -rn "impl.*MultivariatePolyOracle" crates/`) and confirm each is Send+Sync
(they are Vec-backed; the compiler is the arbiter). If any impl fails, STOP and report
instead of restructuring types.

Nested parallelism note: each oracle's kernel is itself rayon-parallel
(`sum_packed_terms`). Rayon nesting composes via work stealing — do not add scoping or
thread-pool plumbing.

### E1b — parallel `gen_layers` across instances

`gkr_prover.rs:413-416` (`prove_batch`): `input_layer_by_instance.into_iter().map(|l|
gen_layers(l)...)` runs the full layer-pyramid generation serially per instance.
Parallelize across instances with the same `#[cfg]` idiom (`Layer<B>: Send` — verify
via compiler). Order-preserving collect; transcript unchanged.

### E1 verification and measurement

- All gates in Global 2, plus a stable full-proof regression test. Mix every field of
  `GkrBatchProof` and `GkrArtifact` (round-polynomial coefficients, masks, output claims,
  OOD point, returned input claims, and instance sizes) into a deterministic digest. Run the
  same seeded same-size multi-instance and unequal-size batches under `prover` and
  `prover,parallel`; diff the complete digests. Equal proof sizes alone are insufficient.
- Measure: re-run the e2e 2^20 row, both paths, L in {1, 4, 16, 64}. Path A should be
  unchanged (it does not use sumcheck). Build fresh isolated pre-E1 and post-E1 artifacts;
  run one warmup + 3 samples for each and compare medians. If Path A moves by more than 3%,
  investigate the environment before recording. Do not use the historical 1,664ms as the
  sole causal baseline.
- Fill T-E1 below. The number that matters: `gkr_prove` at L=64 (was 1,664ms).
  Expectation: large improvement at L=64 (65 independent kernels per round), little
  change at L=1 (only the two required GKR instances can parallelize across). E1a and E1b
  share the `gkr_prove` timer, so report only their combined effect unless separately ablated.
- Commit: `perf(lookups): parallelize sumcheck and GKR layer generation across instances`.

## E2 — Glue: layer building and MLE combination

Files: `crates/examples/src/gkr_e2e/**` and (if the harness imports rather than
copies it) `crates/examples/src/xor/gkr_lookups/accumulation.rs`. Nothing else.

1. `gkr_layers` (433ms at L=64): the `col - z` SecureField column construction for
   exactly L+1 relation denominators (L lookup columns plus the table column).
   Requirements: packed arithmetic only (`PackedSecureField::broadcast(z)` against the
   column's `PackedM31` data — match the existing combine convention `value - z`
   exactly), parallel over chunks within a column AND over those L+1 columns
   (`par_iter` over columns; reuse the `#[cfg]` idiom). The multiplicities column stays
   BaseField and is never relation-shifted. No per-element scalar loops or intermediate
   `Vec<SecureField>` allocations.
2. `mle_combine` (452ms at L=64): `combine()` (the `acc[i] += alpha * v[i]` loop) is a
   serial zip over full columns, called once per accumulated MLE. Parallelize the zip
   with `par_chunks_mut` + zip (disjoint writes), keeping the exact same arithmetic
   per element. Also check `DynMle::into_secure_mle`'s Base->Secure conversion for a
   serial per-element loop; if present, chunk-parallelize it the same way.
   NOTE: `combine`'s accumulation is per-index (`acc[i] += alpha * v[i]`) — there is
   no cross-index reduction, so parallelism cannot change results.
3. Re-run gates + the e2e 2^20 row; fill T-E2. Targets: `gkr_layers` <= ~100ms,
   `mle_combine` <= ~80ms at L=64.
4. Commit: `perf(examples): parallelize GKR e2e glue (layer build + MLE combine)`.

## E3 — Fused fold + next-round sum (W4)

This is `tasks/gkr-perf-spec.md` W4, with the fixed-chunk correction below (the SIMD
cutoff is `(1 << layer.n_variables().saturating_sub(3)) >= N_LANES` — see that spec's
"Access-pattern math" section for the verified index derivation). Implement exactly as
written there, including:

- New defaulted `GkrOps::fix_first_variable_and_sum` (CPU keeps the two-pass default).
- `next_round_raw_sums: Option<(SecureField, SecureField)>` field on
  `GkrMultivariatePolyOracle`, consumed at the top of the SIMD
  `sum_as_poly_in_first_variable`.
- All four layer-variant fused kernels (write LogUpGeneric first as the template).
- The required tests from that spec: 2^6 fallback, 2^7 exact cutoff boundary, a large
  fused case, unequal batches, fused-vs-two-pass equality of folded layers AND cached
  sums, and the CPU/SIMD proof parity suite staying green.
- W1 is out of scope, so use the current fixed `PACKED_CHUNK_SIZE`; do not call the absent
  `packed_chunk_size` helper. Both serial and parallel fused loops must emit indexed chunk
  partials and then fold them serially in chunk order exactly like current
  `sum_packed_terms`.

Interaction with E1: with E1a, `fix_first_variable` calls run inside a parallel map —
the fused kernel is internally rayon-parallel too; this composes (nesting note above).
With the cache hit, next-round `sum_as_poly` tasks become O(1); that is expected and
fine.

After gates pass, STOP and request human review (math correspondence: fold formula, cutoff,
correction factor, reduction order, transcript identity) before committing. Repeat the full
proof-and-artifact digest gate from E1. The resolved pre-existing initialization issue above
does not block E3. Proposed commit message (post-approval):
`perf(lookups): fuse sumcheck fold with next-round evaluation`.

## Final measurement — full matrix, final verdict

After the fastest verified cumulative subset of E1-E3 (or after E1-E2 if E3 is blocked or
review is pending — say which):

1. Re-run the FULL e2e matrix: {2^16, 2^20} x L in {1, 4, 16, 64}, both paths, per the
   harness protocol (warmup + 3, medians, phase lines, proof sizes, peak RSS).
2. Fill T-FINAL and the phase tables for L=1 and L=64 at 2^20.
3. Write the final crossover statement in `tasks/gkr-e2e-spec.md` AND update the
   VERDICT section in `tasks/gkr-perf-plan.md` per the decision rule at the top.
4. Also refresh the kernel-level picture: `cargo bench --features "prover,parallel"
   --bench lookups -- "simd"` once, record means in T-KERNEL (comparability with the
   tables in `tasks/gkr-perf-spec.md`).

## Out of scope — do not implement

- W1 adaptive chunk sizing (superseded: E1's cross-instance parallelism changes the
  scheduling regime; retune only if a human reopens it).
- t=1/2 evaluation-point change; CPU fused kernels; any `core/` change.
- Any further optimization after the final measurement, regardless of outcome.
- End-to-end harness redesign (measure with it as-is; its glue is E2's scope).

## Results

### T-E1 — e2e 2^20 row after E1 (median ms)
| L | A total | B total | B/A | gkr_prove (was) | gkr_prove (now) |
|---|---------|---------|-----|-----------------|-----------------|
| 1 | 661.251 | 1,141.774 | 1.727 | 49.520 | 47.758 |
| 4 | 724.317 | 1,277.168 | 1.763 | 120.699 | 89.221 |
| 16 | 1,040.498 | 1,765.883 | 1.697 | 406.088 | 231.213 |
| 64 | 2,330.286 | 3,292.604 | 1.413 | 1,610.938 | 882.190 |

E1 result: keep. `gkr_prove` improved by 3.6%, 26.1%, 43.1%, and 45.2% for
L={1,4,16,64}; total Path B improved by 1.9%, 4.6%, 3.7%, and 18.8%. The causal
comparison used an exact pre-E1 dirty-source snapshot and fresh isolated pre/post release
binaries, both built with `-C target-cpu=native`, then interleaved with separate warmups and
three recorded samples per binary at 12 Rayon workers. Pre/post binary SHA-256 values were
`3abe98142a193c8e505368e708f8355da21f3ff3204e805ee70b8907e90f6f4a` and
`ca7d2e29d9c61468ec0bf38556ac154e924b106e6549c888804349cfad50c2c5`.

Path A controls moved -2.0%, +2.1%, and +2.6% at L={4,16,64}. The first combined L=1
control was borderline (-3.2%), so it was investigated with a fresh interleaved Path-A-only
warmup + three samples: 638.910ms pre vs 641.420ms post (+0.4%). E1a and E1b share the
`gkr_prove` timer, so the table attributes only their combined effect.

### T-E2 — e2e 2^20 row after E1+E2 (median ms)
| L | B total | gkr_layers (was/now) | mle_combine (was/now) |
|---|---------|----------------------|------------------------|
| 1 | 1,131.489 | 10.159 / 3.218 | 35.695 / 28.618 |
| 4 | 1,268.416 | 31.674 / 9.956 | 56.253 / 32.167 |
| 16 | 1,506.805 | 118.799 / 32.090 | 135.428 / 48.182 |
| 64 | 2,408.856 | 421.472 / 120.041 | 453.446 / 114.117 |

E2 result: keep. At L=64, `gkr_layers` improved 71.5%, `mle_combine` improved
74.8%, and total Path B improved from the fresh post-E1 control's 3,055.893ms to
2,408.856ms (21.2%). The absolute stretch targets (100ms / 80ms) were not reached, but
the experiment is materially positive. At L=1, total Path B improved 5.8% and the paired
Path A control moved -0.7%. The L=64 paired Path A control moved -2.5%.

The exact accepted post-E2 release artifact SHA-256 is
`dfe685d11e9ee88ee6eb4f1345c10c8b043aa2b16dcc482da06fa9176a6176db`; it includes
the final explicit minimum Rayon chunk size for Base-to-Secure conversion. L={1,64} used
fresh Path-A/Path-B warmup + three-sample medians; L={4,16} Path B used the same protocol.

### T-E3 — e2e 2^20 row after E1+E2+E3 (median ms)
| L | B total | gkr_prove (post-E1 / post-E3) |
|---|---------|-------------------------------|
| 1 | 1,185.360 | 48.205 / 52.262 |
| 64 | 2,579.466 | 787.445 / 867.927 |

E3 result: reject and revert. Although the isolated 2^20 lookup kernels improved by
10.7–16.5%, the interleaved end-to-end control regressed: Path B total increased 4.9% at
L=1 and 7.8% at L=64, while `gkr_prove` increased 8.4% and 10.2%, respectively. Path A
controls moved +0.7% and -2.3%, both within the 3% environment threshold. The fastest
verified cumulative subset is therefore E1+E2; E3 must not be committed or included in the
final matrix.

### T-FINAL — full matrix (mirror the T-A/T-B format from gkr-e2e-spec.md)
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

The final matrix used the exact accepted E1+E2 artifact, SHA-256
`dfe685d11e9ee88ee6eb4f1345c10c8b043aa2b16dcc482da06fa9176a6176db`, built
with `parallel` and `-C target-cpu=native`, with 12 Rayon workers. Each timing is the
median of three recorded runs after one discarded warmup. Peak RSS came from separate
`/usr/bin/time -l` invocations. Proof sizes and GKR felt counts are unchanged from the
baseline because E1 and E2 preserve the proof and transcript.

### T-PHASE-FINAL — 2^20 median-run phase breakdown

#### L = 1

| path | base_commit | interaction_gen | interaction_commit | gkr_layers | gkr_prove | mle_combine | mle_trace | mle_trace_commit | stark_prove | total | verify |
|------|-------------|-----------------|--------------------|------------|-----------|-------------|-----------|------------------|-------------|-------|--------|
| A | 53.583 | 17.418 | 56.440 | — | — | — | — | — | 495.108 | 622.613 | 0.272 |
| B | 34.646 | — | — | 3.416 | 46.815 | 28.611 | 10.698 | 60.056 | 931.553 | 1,131.489 | 0.830 |

#### L = 64

| path | base_commit | interaction_gen | interaction_commit | gkr_layers | gkr_prove | mle_combine | mle_trace | mle_trace_commit | stark_prove | total | verify |
|------|-------------|-----------------|--------------------|------------|-----------|-------------|-----------|------------------|-------------|-------|--------|
| A | 272.046 | 235.571 | 449.410 | — | — | — | — | — | 1,148.625 | 2,105.763 | 0.435 |
| B | 260.730 | — | — | 120.041 | 793.635 | 117.181 | 12.639 | 60.145 | 1,037.689 | 2,408.856 | 2.271 |

### T-KERNEL — lookup bench means after everything
| bench | 2^16 mean ms | 2^20 mean ms |
|-------|--------------|--------------|
| SIMD grand product | 2.8709 | 13.047 |
| SIMD generic LogUp | 4.7646 | 24.626 |
| SIMD multiplicities LogUp | 4.6373 | 22.660 |
| SIMD singles LogUp | 4.2370 | 20.718 |

These are the final E1+E2 kernel means. E2 is example-only, so the kernel comparison is
also the exact pre-E3 baseline. The rejected E3 prototype improved these 2^20 means by
10.7%–16.5%, but regressed the end-to-end `gkr_prove` phase and was reverted under the
fastest-subset decision rule.

### Final crossover statement

After evaluating E1–E3 and selecting E1+E2, there is no crossover at either 2^16 or
2^20 across `L in {1, 4, 16, 64}`. The closest cell is 1.085x at 2^16/L=64; the
closest 2^20 cell is 1.144x at L=64. Path B uses more peak memory in every cell.

**CLOSED — no crossover after E1–E3.**
