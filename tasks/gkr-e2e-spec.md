# GKR vs interaction-trace LogUp — end-to-end harness spec

Self-contained spec for an implementing agent with no prior context. Companion docs:
`tasks/gkr-perf-spec.md` (kernel work items), `tasks/gkr-perf-handoff.md` (state + W3
results), `tasks/gkr-perf-plan.md` (background).

## Goal and decision rule

Build ONE measurement harness that proves the SAME lookup statement two ways and reports
total prover time, per-phase time, proof size, and peak memory:

- **Path A (baseline)**: standard constraint-framework LogUp — interaction trace
  committed, prefix-sum column, logup constraints.
- **Path B (GKR)**: no interaction trace — GKR `prove_batch` over the lookup fractions,
  the resulting MLE claims verified in-STARK by ONE `MleEvalProverComponent`.

The deliverable is the filled result tables at the bottom and the crossover point:
the smallest number of lookup columns `L` (if any) at which Path B's total prover time
beats Path A's, at 2^16 and 2^20 rows. Do not editorialize beyond the tables; the
protocol decision is made by a human from these numbers.

## Global constraints

1. NEW CODE ONLY in `crates/examples/src/gkr_e2e/` (new module) plus one `pub mod
   gkr_e2e;` line in `crates/examples/src/lib.rs`, and result recording in this file.
   Do NOT modify: `crates/stwo/**`, `crates/constraint-framework/**`, existing examples.
   If something seems to require a framework change, STOP and report instead.
2. Both paths MUST verify. A path that proves but does not verify is a wiring bug, not a
   result. Both paths must also FAIL on corrupted input (see Gate G3).
3. Use the promoted framework module `stwo_constraint_framework::mle_eval`
   (`MleEvalProverComponent`, `MleEvalVerifierComponent`, `MleCoeffColumnOracle`,
   `build_trace`) — NOT the legacy copy in `crates/examples/src/xor/gkr_lookups/mle_eval.rs`.
   The canonical wiring template is the framework's own test
   `mle_eval_prover_component` (`crates/constraint-framework/src/mle_eval.rs:931-1003`).
   You may copy `MleCollection` from
   `crates/examples/src/xor/gkr_lookups/accumulation.rs` into the new module if it is
   not importable from where it lives.
4. Verification gates after implementation (all must pass):
   ```bash
   cargo test --features "parallel" -p stwo-examples gkr_e2e 2>&1 | tail -3
   scripts/clippy.sh
   scripts/rust_fmt.sh
   ```
   (Check the examples crate's actual feature names in `crates/examples/Cargo.toml`
   before running; mirror how existing example tests enable the prover.)
5. Benchmarks: never run two measurement processes concurrently; build once, then time.
   Follow the stale-artifact rules in `tasks/gkr-perf-handoff.md` ("Benchmark
   methodology correction") for any A/B involving different features.
6. Commit only after gates pass, message:
   `feat(examples): GKR vs interaction-trace LogUp end-to-end harness`. No co-author line.

## The statement (identical for both paths)

Parameters: `LOG_N` (rows per column), `L` (number of use columns).

Columns (all length `1 << LOG_N`, BaseField, generated with `SmallRng::seed_from_u64(0)`):
- `table[i] = i` (the lookup table: values `0..2^LOG_N`).
- `use_k[i]` for `k in 0..L`: uniform random values in `[0, 2^LOG_N)`.
- `mults[i]` = number of times value `i` appears across ALL `use_k` columns
  (compute by counting; total = `L << LOG_N`).

Claim proven: `sum_{k,i} 1/(use_k[i] - z) == sum_i mults[i]/(table[i] - z)` for a
channel-drawn `z` — the standard LogUp identity, so the base trace is identical for
both paths and the honest inputs satisfy it by construction.

Base trace tree layout (both paths): tree 0 = preprocessed (empty, committed with
`extend_evals(vec![])` exactly like the template test); tree 1 = the `L + 2` base
columns in order `[use_0, ..., use_{L-1}, table, mults]`.

## Path A — exact construction

Template: `crates/examples/src/state_machine/` (a full LogUp prove/verify example —
read its component/gen/prove wiring before writing code).

1. `FrameworkComponent` with a `FrameworkEval` at `log_size = LOG_N`,
   `max_constraint_log_degree_bound = LOG_N + 1`.
2. One relation (`stwo_constraint_framework::relation!` macro, 1 element) drawn from the
   channel AFTER committing tree 1. `z` in the statement above is `relation.z`; for one element,
   `relation.combine([x]) = x - z` (`alpha^0 = 1`).
3. `evaluate` reads the `L + 2` mask columns at offset 0 and writes, via the standard
   logup API (`eval.add_to_relation` / the crate's current equivalent — copy
   state_machine's idiom):
   - for each `k`: fraction `+1 / relation.combine([use_k])`
   - one fraction `-mults / relation.combine([table])`
   then `eval.finalize_logup_in_pairs()` (or the state_machine idiom for batching).
4. Interaction trace: `LogupTraceGenerator` (`crates/constraint-framework/src/prover/logup.rs`),
   same fraction order as `evaluate`, committed as tree 2. `claimed_sum` must be 0
   (assert it — the statement holds by construction).
5. `prove(&[&component], channel, commitment_scheme)` then full `verify(...)`
   mirroring the template test's verifier block.

## Path B — exact construction

1. Commit trees 0 and 1 exactly as Path A (same column order, same channel type).
2. Draw the same one-element relation as Path A after tree 1. Its current API consumes
   `[z, alpha]`; `alpha` is unused for a one-element relation but MUST still be consumed so both
   transcripts use the exact same lookup randomness.
3. Build GKR input layers (SecureField columns computed with packed ops, `col - z`, exactly
   matching `LookupElements<1>::combine`):
   - for each `k in 0..L`: `Layer::LogUpSingles { denominators: Mle::new(use_k_minus_z) }`
   - one `Layer::LogUpMultiplicities { numerators: Mle::new(mults), denominators:
     Mle::new(table_minus_z) }`
   in exactly this order (singles 0..L, then multiplicities last).
4. `let (gkr_proof, artifact) = prove_batch(channel, layers);`
   (`stwo::prover::lookups::gkr_prover::prove_batch`).
5. Statement check (prover side, and mirrored verifier side after
   `partially_verify_batch`): sum the per-instance output claims as fractions —
   `sum_k Fraction::new(out_k[0], out_k[1]) - Fraction::new(out_m[0], out_m[1])` must be
   the zero fraction (numerator == 0). Assert it.
6. Collect the MLE claims to verify from `artifact.claims_to_verify_by_instance`
   (all instances have `n_variables == LOG_N`, so `artifact.ood_point` is shared):
   - singles instance `k`: claims are `[numerator_claim, denominator_claim]`; assert
     `numerator_claim == SecureField::one()`, keep `denominator_claim` (call it `c_k`).
   - multiplicities instance: claims `[num_claim, denom_claim]` — keep both
     (`c_L = num_claim`, `c_{L+1} = denom_claim`).
7. Draw the accumulation coefficient: `let acc_alpha = channel.draw_secure_felt();`
   (immediately after `prove_batch` returns; `prove_batch` has already mixed the layer
   masks and claims into the channel).
8. Prover-side combined MLE: push into an `MleCollection<SimdBackend>` in EXACTLY this
   order: `use_0_minus_z, ..., use_{L-1}_minus_z` (Secure), `mults` (Base),
   `table_minus_z` (Secure); then
   `let [combined_mle] = collection.random_linear_combine_by_n_variables(acc_alpha)`.
   NOTE the RLC convention (`accumulation.rs`): with `n` MLEs, MLE `i` gets coefficient
   `alpha^(n-1-i)` (the LAST pushed gets coefficient 1).
9. Combined claim (both prover and verifier compute this from the artifact):
   `combined_claim = sum_i acc_alpha^(n-1-i) * c_i` with `n = L + 2` and `c_i` ordered
   exactly as pushed in step 8. Sanity-assert prover side:
   `mle_eval_at_point(&combined_mle, &artifact.ood_point) == combined_claim` (helper in
   `crates/examples/src/xor/gkr_lookups/mod.rs` tests — copy it).
10. The oracle: implement the framework's `MleCoeffColumnOracle` trait for a wrapper
    around the base component. Its `evaluate_at_point` (see the trait's signature in
    `stwo_constraint_framework::mle_eval`) must return, from the base component's
    sampled mask values at point `p`:
    `sum_{k<L} acc_alpha^(n-1-k) * (use_k(p) - z) + acc_alpha^1 * mults(p)
     + acc_alpha^0 * (table(p) - z)`
    i.e. the SAME affine combination as steps 8-9, expressed over the mask samples.
    The base component must therefore declare mask offset `[0]` for all `L + 2`
    columns. If the base component has no constraints of its own, it still needs the
    mask declaration — model it on `MleCoeffColumnComponent` in the framework test
    (mle_eval.rs tests define one; adapt it to read L+2 columns and combine as above).
11. MleEval trace: `build_trace(&combined_mle, &artifact.ood_point, combined_claim)`
    committed as tree 2 (this is the eq-evals + prefix-sum trace).
12. Components: `[&base_component, &mle_eval_component]` where
    `mle_eval_component = MleEvalProverComponent::generate(allocator, &oracle,
    &artifact.ood_point, combined_mle, combined_claim, &twiddles, MLE_EVAL_TRACE_IDX)`
    — mirror the template test's argument order and tree indices exactly.
13. `prove(...)`, then verify: fresh channel replaying the SAME transcript order
    (commit tree0, tree1 → draw the same one-element relation (consuming `z` and unused `alpha`)
    → `partially_verify_batch(vec![Gate::LogUp; L+1],
    &gkr_proof, channel)` → check output-claim fraction sum → draw acc_alpha → compute
    combined_claim → commit tree 2 inside the verifier's `CommitmentSchemeVerifier`
    flow → `MleEvalVerifierComponent` + base component → `verify(...)`).
    The GKR proof rides alongside the STARK proof as a separate struct in the harness
    (no combined serialization format — this is a measurement harness).

Twiddles for both paths: precompute once for
`CanonicCoset::new(LOG_N + 1 + config.fri_config.log_blowup_factor)` (mirror the
template test's `LOG_EXPAND = 1` sizing) and reuse.

## Instrumentation (exact)

Wrap each phase in `std::time::Instant` and print ONE line per phase to stdout:

```text
E2E path=<a|b> log_n=<N> l=<L> phase=<name> ms=<f64>
```

Path A phases: `base_commit`, `interaction_gen`, `interaction_commit`, `stark_prove`
(the `prove()` call), `total`, `verify`.
Path B phases: `base_commit`, `gkr_layers` (building z-minus columns + layers),
`gkr_prove` (`prove_batch`), `mle_combine` (steps 8-9), `mle_trace` (build_trace),
`mle_trace_commit`, `stark_prove`, `total`, `verify`.

Proof size, one line per path:
```text
E2E path=<a|b> log_n=<N> l=<L> proof_bytes=<stark> gkr_felts=<count>
```
- `stark` = `proof.size_estimate()` (exists on `StarkProof`, `core/proof.rs:82`).
- `gkr_felts` (Path B only; Path A prints 0) = total SecureField count in
  `GkrBatchProof`: sum over sumcheck proofs of `round_polys[i].len()` (each coeff is 1
  felt via `UnivariatePoly` Deref) + sum over layer masks of their column felts
  (each `GkrMask` column is 2 SecureFields) + output claims count. Count by iterating
  the actual struct fields; do not hardcode formulas.

Peak memory protocol (run per cell, sequentially):
```bash
cargo test --no-run --features "parallel" -p stwo-examples 2>&1 | tail -1
BIN=$(ls -t target/debug/deps/... )   # NO — use release:
cargo test --no-run --release --features "parallel" -p stwo-examples
BIN=$(ls -t target/release/deps/stwo_examples-* | grep -v '\.d$' | head -1)
GKR_E2E_LOG_N=20 GKR_E2E_L=16 GKR_E2E_PATH=b \
  /usr/bin/time -l "$BIN" gkr_e2e::run_one -- --ignored --nocapture 2>&1 \
  | grep -E "E2E |maximum resident set size"
```
The harness reads `GKR_E2E_LOG_N`, `GKR_E2E_L`, `GKR_E2E_PATH` env vars in an
`#[ignore]`d test named `run_one` (defaults: 16, 1, both paths). Record
"maximum resident set size" per cell.

All timing runs: `RUSTFLAGS="-C target-cpu=native"` and `--release`, matching
`poseidon_benchmark.sh` conventions. Timing and memory runs are separate invocations.

## Test gates (in the module, must pass under `cargo test`)

- G1 `gkr_e2e_roundtrip_small`: both paths prove AND verify at `LOG_N=8, L=2`.
- G2 `gkr_e2e_paths_agree`: both paths accept the same honest input at `LOG_N=8` for
  `L in [1, 3]`.
- G3 `gkr_e2e_rejects_bad_multiplicity`: corrupt one `mults` entry (+1) after counting;
  Path A's prove-or-verify pipeline must reject, and Path B must reject (the GKR
  output-claim fraction sum is nonzero — the assert from step 5 must be converted into
  a returned error for this test, not a panic in library code; a panic in the TEST is
  acceptable via `#[should_panic]` or `Result` matching).
- G1-G3 run at small sizes so CI cost is negligible.

## Baseline measurement matrix

`LOG_N in {16, 20}` x `L in {1, 4, 16, 64}` x both paths, `parallel` feature ON,
12 workers (machine default). One warmup run per cell discarded, then 3 timed runs;
record the MEDIAN total and the phase breakdown of the median run.

### T-A / T-B — per-path totals (median ms)
| log_n | L | A total | B total | B/A | A peak MiB | B peak MiB | A proof B | B proof B (+gkr felts) |
|-------|---|---------|---------|-----|------------|------------|-----------|-------------------------|
| 16 | 1 | 64.933 | 112.367 | 1.731 | 73.1 | 82.6 | 14,580 | 16,548 (+612) |
| 16 | 4 | 79.147 | 118.444 | 1.497 | 83.8 | 88.8 | 14,712 | 14,136 (+810) |
| 16 | 16 | 79.659 | 197.580 | 2.480 | 105.3 | 120.5 | 18,216 | 16,008 (+1,602) |
| 16 | 64 | 164.563 | 492.593 | 2.993 | 195.9 | 348.8 | 21,720 | 17,816 (+4,770) |
| 20 | 1 | 645.380 | 1,270.479 | 1.969 | 1,188.7 | 1,306.5 | 23,556 | 25,012 (+924) |
| 20 | 4 | 707.734 | 1,364.911 | 1.929 | 1,288.4 | 1,403.8 | 23,160 | 23,816 (+1,170) |
| 20 | 16 | 1,013.998 | 1,883.332 | 1.857 | 1,713.5 | 2,072.2 | 26,152 | 24,152 (+2,154) |
| 20 | 64 | 2,217.340 | 4,062.866 | 1.832 | 3,154.3 | 5,249.1 | 30,184 | 24,792 (+6,090) |

Measurement artifact:

| Field | Value |
|-------|-------|
| Executable | `/private/tmp/gkr-e2e-release/release/deps/stwo_examples-ea30d8eaf016619b` |
| SHA-256 | `1f227012c866feba26af787c25d480651958d946fdd866d6b756b067a181bc8c` |
| Cargo features | `parallel` |
| Rust flags | `-C target-cpu=native` |
| Rayon workers | 12 |
| Timing samples | one discarded warmup, then three runs; table uses median total and its phase row |
| Peak memory | one separate `/usr/bin/time -l` invocation; bytes converted to MiB |

Actual Path A interaction-tree width:

| L | extension columns | committed BaseField coordinate columns |
|---|-------------------|----------------------------------------|
| 1 | 1 | 4 |
| 4 | 3 | 12 |
| 16 | 9 | 36 |
| 64 | 33 | 132 |

### T-PHASE — phase breakdown at log_n=20 (median ms, one table per L)

Record every phase line for L=1 and L=64 at minimum.

#### L = 1

| path | base_commit | interaction_gen | interaction_commit | gkr_layers | gkr_prove | mle_combine | mle_trace | mle_trace_commit | stark_prove | total | verify |
|------|-------------|-----------------|--------------------|------------|-----------|-------------|-----------|------------------|-------------|-------|--------|
| A | 50.383 | 14.818 | 45.716 | — | — | — | — | — | 534.390 | 645.380 | 0.289 |
| B | 54.587 | — | — | 13.365 | 103.977 | 37.093 | 11.039 | 59.424 | 984.357 | 1,270.479 | 0.840 |

#### L = 64

| path | base_commit | interaction_gen | interaction_commit | gkr_layers | gkr_prove | mle_combine | mle_trace | mle_trace_commit | stark_prove | total | verify |
|------|-------------|-----------------|--------------------|------------|-----------|-------------|-----------|------------------|-------------|-------|--------|
| A | 298.204 | 246.269 | 493.739 | — | — | — | — | — | 1,179.006 | 2,217.340 | 0.427 |
| B | 287.714 | — | — | 432.750 | 1,663.510 | 452.357 | 12.316 | 61.983 | 1,133.674 | 4,062.866 | 2.289 |

### Baseline crossover statement

"At 2^20 rows, Path B does not cross Path A within the measured range
`L in {1, 4, 16, 64}`."

"At 2^16 rows, Path B does not cross Path A within the measured range
`L in {1, 4, 16, 64}`."

## Final optimized E1+E2 matrix (2026-07-21)

E1 parallelizes sumcheck/GKR work across instances while preserving indexed transcript
order. E2 packs and parallelizes the example-side layer construction and MLE combination.
E3's fused fold+sum prototype improved isolated kernels but regressed end-to-end Path B,
so it was rejected and reverted. This table is the final fastest verified subset: E1+E2.

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

Final measurement artifact:

| Field | Value |
|-------|-------|
| Executable | `/private/tmp/gkr-e2-post-target/release/deps/stwo_examples-c57d0ea61395cc20` |
| SHA-256 | `dfe685d11e9ee88ee6eb4f1345c10c8b043aa2b16dcc482da06fa9176a6176db` |
| Cargo features | `parallel` |
| Rust flags | `-C target-cpu=native` |
| Rayon workers | 12 |
| Timing samples | one discarded warmup, then three runs; table uses the median total |
| Peak memory | separate `/usr/bin/time -l` run for every path/cell |

### Final phase breakdown at log_n=20

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

Final crossover: none at either row count. The closest measured cell is 2^16/L=64,
where Path B is 1.085x Path A; at 2^20 the closest cell is L=64 at 1.144x. Path B
uses more peak memory in every cell. **CLOSED — no crossover after E1–E3.**

## Out of scope — do not implement

- Any change under `crates/stwo/` or `crates/constraint-framework/` (report blockers
  instead).
- Proof serialization format for the combined (STARK + GKR) proof.
- The W1/W4 kernel work items (separate spec).
- Multi-size lookup columns (all columns are `LOG_N` here; jagged/mixed sizes are a
  later experiment).
- Any tuning based on these results — measurement only.

## Known risks / reconciliation notes for the implementer

- The framework `mle_eval` module was recently promoted (commits `6621507a`,
  `a46471ae`, `d24324d1`, `4f877db2`) — trust ITS current signatures over anything in
  this spec; if `MleEvalProverComponent::generate`'s signature differs from step 12,
  follow the code and note the difference in this file.
- `crates/constraint-framework/src/mle_eval.rs` currently has unrelated uncommitted
  formatting drift in the worktree — do not include it in any commit.
- The GKR kernels in the worktree are the parallelized versions (uncommitted, see
  `tasks/gkr-perf-handoff.md`) — measure with them as-is; they are the state under
  evaluation.
- If `LogupTraceGenerator`'s batching puts the L+1 fractions into a different number of
  interaction columns than you expect, record the actual interaction tree column count
  in the results — it is part of the answer, not a bug.
- Harness reconciliation (2026-07-21): current `LookupElements<1>::combine` is `value - z`, so
  both paths use that exact sign and both consume the relation's otherwise-unused `alpha` draw.
  After both `prove_batch` and `partially_verify_batch`, validate the complete artifact shape
  (`n_variables_by_instance`, OOD-point length, every two-element MLE claim, and every two-element
  output claim) before indexing; malformed shapes are harness errors, not panics.
- The harness asserts `2^LOG_N < p_M31` and `L * 2^LOG_N < p_M31`, so table values and total
  multiplicities are represented without field wraparound (all measurement-matrix cells satisfy
  both bounds). Before `partially_verify_batch`, it also checks the proof instance counts, each
  instance's `LOG_N` layer-mask depth, and two-element output-claim shape.
