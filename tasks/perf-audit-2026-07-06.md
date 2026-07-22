# Performance audit — 2026-07-06

Pass over the prover hot paths (SIMD backend, FFT, quotients, FRI, PCS opening,
constraint framework, lookups, VCS, channel), plus a second verification pass
covering FRI folding, PoW grinding, Merkle commit, mempool coverage, and the
initially-dropped agent claims. Every finding was verified against the code;
findings that didn't survive are listed at the end so the work isn't repeated.

Not audited (secondary for a Blake2s prover): Poseidon252/Keccak hash+channel
paths, `air-utils` iterators, proof serialization. Ranking assumes a typical
span breakdown — one profiled `./poseidon_benchmark.sh` run would confirm the
percentages.

Context on where prover time goes: trace extension FFTs + Merkle commit,
constraint evaluation, quotient computation + FRI, and OOD evaluation. The
verified wins hit OOD eval, the constraint-eval inner loop, and quotients.

Expected end-to-end gain from items 1–8: **roughly 5–10%** on a typical large
single-component trace (item 1 dominates), more for mask-heavy or very wide
AIRs. None of them touch constraints, field arithmetic, FRI parameters, or the
transcript — except item 7a which edits a SECURITY-CRITICAL file
(transcript-preserving, but per repo rules needs explicit approval).

---

## 1. `barycentric_weights`: scalar-mul + QM31 division per domain element

**Files:** `crates/stwo/src/prover/backend/simd/circle.rs:239-326`
(`SimdBackend::barycentric_weights`); optional parity fix in
`crates/stwo/src/prover/backend/cpu/circle.rs:100-131`.

**Current cost per domain element** (the `vi_p` sweep, lines 265-292):
1. `domain.at(bit_reverse_index(i*N_LANES+j, log_size))` →
   `CirclePointIndex::to_point()` → `M31_CIRCLE_GEN.mul(index)`
   (`core/circle.rs:242`): a ~31-step scalar multiplication, ~100+ M31 muls.
2. `point_vanishing(e, p)` (`core/constraints.rs:87-93`) = `h.y / (1 + h.x)`
   with `h = p - e`: a **full QM31 division** (~40+ M31 muls).

Both are per element of a full domain, once per distinct
`(log_size, sample_point)` pair. This runs on the default path:
`store_polynomials_coefficients` defaults to `false` (`prover/pcs/mod.rs:50`),
so `build_weights_hash_map` → `barycentric_weights` runs every proof.

**Change.** Replace the `vi_p` loop with a sweep over
`CircleDomainBitRevIterator` (`simd/domain.rs:13`), which yields
`CirclePoint<PackedM31>` in exactly the required bit-reversed order at ~1
packed point-add per step, and fold the division into the batch inversion
that already exists at line 294:

- Per packed step, with `p = (prx, pix, pry, piy)` broadcast as `PackedCM31`
  pairs (same decomposition `denominator_inverses` uses,
  `simd/quotients.rs:267-279`):
  - `h.y = (pry - e.y, piy)`, `d = 1 + h.x = (one + prx - e.x, pix)`
    as `(PackedCM31, PackedCM31)` pairs → assemble into `PackedQM31`.
- Collect `hy_vec` and `d_vec`; compute
  `vi_p_inverse[i] = d_vec[i] * batch_inverse(&hy_vec)[i]`.
  Algebra: `vi_p = h.y/(1+h.x)` ⇒ `1/vi_p = (1+h.x)·h.y⁻¹`. The failure set
  is unchanged (both formulations need `h.y ≠ 0` and `1+h.x ≠ 0`; `p` is a
  random secure-field point so probability ~2⁻¹²⁴).
- Keep `si_0`, `vn_p`, `si_i_vn_p` and the final weights loop as is; `vi_p`
  itself is no longer materialized (only its inverse).
- Parallel feature: use `CircleDomainBitRevIterator::par_iter()`
  (`simd/domain.rs:68`, strided `start_at`) mirroring the existing
  `#[cfg(feature = "parallel")]` split.
- The `weights_vec_len == 1` CPU fallback (line 246) stays.

**CPU parity (optional, small-domain only):** `cpu/circle.rs:100-131` also does
per-element `domain.at` + `point_vanishing` division + a final
`vn_p / (si_i[i] * vi_p[i])` division per element; if touched, use one
`batch_inverse` over `si_i[i] * vi_p[i]`. Low value (CPU path = small
domains), do only if convenient.

**Tests.** Existing SIMD-vs-CPU consistency tests in `simd/circle.rs`; if no
weights-specific test exists, add one: random point, log_size ∈ {5, 10, 14},
assert new SIMD weights == CPU weights elementwise. Run
`cargo test --features prover -p stwo`.

**Effort:** ~half a day. **Risk:** low — prover-only file, output identical.
**Expected:** multi-x on the weights sweep → ~3–7% end-to-end.

---

## 2. Hoist loop-invariant `trace_cols` in constraint eval

**File:** `crates/constraint-framework/src/prover/component_prover.rs:178-179`.

`CHUNK_SIZE == 1` (line 27), so
`let trace_cols = trace.as_cols_ref().map_cols(|c| c.as_ref());` allocates a
fresh `TreeVec<Vec<&CircleEvaluation>>` (one Vec per interaction, n_cols
pointers total) for every packed row — 2^15 times for a 2^20 domain, in the
hottest prover loop.

**Change.** Move the binding to just above `iter.for_each` (next to the
existing `self_eval`/`self_claimed_sum` hoists at lines 175-176, which exist
for the same closure-capture reason) and capture `&trace_cols`. `&TreeVec` of
shared refs is `Send + Sync`; `SimdDomainEvaluator::new` already takes it by
reference. Mechanical one-line move; no behavior change.

**Tests.** `cargo test --no-default-features -p stwo-constraint-framework`
plus `cargo test --features prover`.
**Effort:** minutes. **Expected:** 1–3% of constraint eval; <1% end-to-end
(more for very wide AIRs).

---

## 3. Stop regenerating domain points in `denominator_inverses`

**File:** `crates/stwo/src/prover/backend/simd/quotients.rs:255-283`, caller
at `:116-123`.

The function clones `CircleDomainBitRevIterator` once **per sample point**
(line 276-277, comment admits it), re-deriving every packed domain point
n_samples times. The caller `compute_quotients_and_combine` has already
collected the identical points into `subdomain_points` (lines 116-117).

**Change.**
- Signature: `fn denominator_inverses(sample_points: &[CirclePoint<SecureField>],
  domain_points: &[CirclePoint<PackedBaseField>]) -> Vec<Vec<PackedCM31>>`.
- Caller passes `&subdomain_points`.
- Body: outer iteration over sample points unchanged (seq / `par_iter` per
  feature); inner pass becomes `domain_points.iter()` (or `.par_iter()` on the
  slice under `parallel`) — drop the iterator clone and the
  `#[cfg]`-duplicated iterator setup.
- Check whether `CircleDomainBitRevIterator::par_iter`/`start_at` have other
  users before removing anything (they do — leave them).

**Tests.** Existing quotient tests in `simd/quotients.rs` (`cargo test
--features prover -p stwo`).
**Effort:** ~1 hour. **Expected:** removes n_samples redundant full-domain
point generations; ~1% end-to-end.

---

## 4. Batch the offset≠0 mask index computation

**File:** `crates/constraint-framework/src/prover/simd_domain.rs:90-98`;
helper `core/utils.rs:147-163` (`offset_bit_reversed_circle_domain_index`).

Every offset≠0 mask read does 32 independent scalar calls per packed row:
each call = `bit_reverse_index` → branch + `rem_euclid` → `bit_reverse_index`.
Per row, per masked column, in the constraint-eval hot loop.

**Change (step 1 — shared-subexpression batch helper).** Add
`offset_bit_reversed_circle_domain_indices<const N: usize>(base: usize, domain_log_size: u32, eval_log_size: u32, offset: isize) -> [usize; N]`
next to the scalar fn in `core/utils.rs`, exploiting that the N lane indices
are `base | j` with `base` aligned to N:
- `bit_reverse_index(base | j, L) = rev_j[j] << (L - LOG_N) | rev_base`, so
  the first bit-reverse costs one `bit_reverse_index(base, L) >> LOG_N` plus a
  16/32-entry `rev_j` const table.
- `half_size`, `step_size` computed once per call instead of per lane.
- The `rem_euclid` + second `bit_reverse_index` stay per-lane (lane values
  land far apart; no shared structure). Still cuts roughly half the work and
  all the per-lane recomputed constants.
Call it from `next_interaction_mask` and build the gather from the returned
array.

**Step 2 (only if step 1 measures short):** per-component index table
`Vec<u32>` per distinct offset, built once in
`evaluate_constraint_quotients_on_domain` and passed down. Memory: 4B × eval
size per offset (32 MB at 2^23 — why this is step 2, not step 1).

**Tests.** Add a unit test asserting the batch helper equals the scalar fn for
random (base, L, dlog, off) combos. Framework tests as in item 2.
**Effort:** ~half a day for step 1. **Expected:** up to 5–15% of constraint
eval for mask-heavy AIRs; ~0 for offset-0-only AIRs.

---

## 5. Domain-size inversion is a shift, not a `pow`

**File:** `crates/stwo/src/prover/backend/simd/circle.rs:161-163`
(`TODO(alont): Cache this inversion.`).

`BaseField::from(eval.domain.size()).inverse()` runs a full pow-based M31
inversion once per column interpolation (hundreds per proof). In M31,
`2^31 ≡ 1 (mod P)`, so `(2^k)⁻¹ = 2^(31-k)` — no inversion needed:

```rust
let inv = PackedBaseField::broadcast(BaseField::from_u32_unchecked(1 << (31 - log_size)));
```

Valid for `1 ≤ log_size ≤ 30`; here `log_size ≥ MIN_FFT_LOG_SIZE`, and
`1 << (31 - log_size) < P` holds for `log_size ≥ 1`. Grep for an existing
`pow2`-style helper in `core/fields/m31.rs` first; add
`BaseField::inverse_of_pow2(log_size)` there if none, with a debug_assert on
the range, and use it. Check the CPU `interpolate` for the same pattern.

**Tests.** One unit test: `inverse_of_pow2(k) == BaseField::from(1 << k).inverse()`
for k in 1..=30. Existing interpolate/evaluate roundtrip tests cover the rest.
**Effort:** minutes. **Expected:** tiny; it's a free TODO closure.

---

## 6. Memoize the `repeated_double` fold in the opening phase

**File:** `crates/stwo/src/prover/pcs/mod.rs` — `build_weights_hash_map`
(`:158-169`, existing `TODO(Leo)`) and `eval_at_points` (`:203`).

`point.repeated_double(lifting_log_size - log_size)` (QM31 point doublings)
is recomputed for every column, though flat AIRs have hundreds of columns
sharing the same `(log_size, point)`.

**Change.** In `prove_values`, before building the weights map, build one
memo `DashMap<(u32, CirclePoint<SecureField>), CirclePoint<SecureField>>`
(DashMap because both loops are `par_map_cols`/`par_iter` under `parallel`;
a plain `HashMap` built upfront from the distinct pairs also works and is
simpler — distinct pairs = n_log_sizes × n_points, enumerable from
`sampled_points` + the trees' column log-sizes). Use it in both call sites;
delete the TODO comment.

**Tests.** Covered by existing pcs prove/verify roundtrip tests.
**Effort:** ~1 hour. **Expected:** small; removes n_columns × n_points × d
QM31 point doublings.

---

## 7. Channel micro-fixes

### 7a. `mix_u32s` word-at-a-time hashing — ⚠ SECURITY-CRITICAL file
**File:** `crates/stwo/src/core/channel/blake2s.rs:71-79`.

```rust
let bytes: Vec<u8> = data.iter().flat_map(|w| w.to_le_bytes()).collect();
hasher.update(&bytes);
```
Byte stream fed to the hasher is identical → digest identical → transcript
unchanged. Still: `core/channel/` is SECURITY-CRITICAL per CLAUDE.md, so this
needs explicit approval before merging, and the existing channel
regression tests (fixed digests) must pass untouched.

### 7b. Drop the `sampled_values` structural clone
**File:** `crates/stwo/src/prover/pcs/mod.rs:222-225`.

`sampled_values` is moved into the proof later (`:298`), hence the `.clone()`
before `flatten_cols()`. Replace with a borrowing flatten:
```rust
let flat: Vec<SecureField> = sampled_values.iter().flatten().flatten().copied().collect();
channel.mix_felts(&flat);
```
(adjust nesting to `TreeVec<ColumnVec<Vec<SecureField>>>`; same element order
as `flatten_cols` — verify against `TreeVec::flatten_cols` impl so the
transcript ordering is bit-identical).

**Tests.** Channel unit tests + any prove/verify roundtrip (transcript
mismatch would fail verification immediately).
**Effort:** minutes each. **Expected:** negligible perf; hygiene.

---

## 8. `line_ifft` batch inversion — cold path, do only in passing

**File:** `crates/stwo/src/prover/line.rs:88-99`.

Per-element `x.inverse()`, recomputed per chunk within a layer — but the only
caller is FRI last-layer interpolation (`prover/fri.rs:218`), size
2^log_last_layer_degree_bound (tiny). Fix if touching the file: per
`while`-iteration, `let xs_inv = batch_inverse(&domain.iter().take(half).collect_vec());`
hoisted above the chunk loop, reused across chunks.

**Tests.** `line_evaluation_interpolation` test exists in the file.
**Effort:** minutes. **Expected:** ~0 end-to-end.

---

## Suggested order & verification

1. Items 2, 5, 3 (mechanical, low risk) → one PR.
2. Item 1 (the real win) → own PR with SIMD-vs-CPU weight test.
3. Item 4 step 1 → own PR, benchmark a mask-heavy example (e.g. Blake/Poseidon
   examples) before/after.
4. Items 6, 7b, 8 opportunistically; 7a only with approval.
5. Item 9 only after a profile on the target hardware shows bit-reverse as a
   visible span with idle cores.

Benchmark gate for each PR: `./poseidon_benchmark.sh` (or
`cargo bench --features prover` on the affected suite) before/after, plus the
full test matrix from CLAUDE.md (`prover`, `prover,parallel`, verifier-only,
no_std gate).

---

## Areas verified clean (second pass)

Checked directly after the initial report; no action needed:

- **FRI folding** (`simd/fri.rs`): `fold_line`/`fold_circle_into_line` are
  chunked (`FOLD_CHUNK_SIZE=128`), rayon-parallel, use precomputed twiddle
  tables (no per-element inversions), and butterfly in registers. The
  `TODO(andrew) Is this optimized?` can effectively be answered "yes".
- **PoW grinding** (`simd/grind.rs`): rayon `parallel_grind` across workers,
  SIMD 16-way hashing inside.
- **Merkle commit** (`simd/blake2s.rs::commit_on_layer`): 16-way SIMD Blake2s
  + `parallel_iter!` over nodes.
- **Mempool coverage** (`poly/circle/ops.rs::evaluate_polynomials`): eval
  buffers come from `pool.take_or_alloc` before the parallel section, and
  interpolation transforms buffers in place — the reported "coefficient
  buffers not pooled" gap doesn't exist in practice.
- **Constraint `denom_inv`** (`component_prover.rs:103-106`): computed once
  per component over `2^log_expand` (blowup-factor-sized, tiny).

## 9. (Watch list) `bit_reverse_m31` parallel fan-out

**File:** `crates/stwo/src/prover/backend/simd/bit_reverse.rs:52-90`.

Parallelism is `parallel_iter!(0..1 << a_bits)` with
`a_bits = column_log_size − 2·W_BITS − VEC_BITS = column_log_size − 16`:
16 tasks for a 2^20 column, 4 for 2^18. Fine for ≤16 cores on large columns;
under-fans on many-core machines and mid-size columns.

**Gate first.** The operation is memory-bandwidth bound; widening the fan-out
only helps if a profile on the target machine shows bit-reverse as a visible
span with idle cores. Do not implement on spec.

**Change (if gated in).** Fold `w_l` into the parallel range:
`parallel_iter!(0..1 << (a_bits + W_BITS))` with
`let (a, w_l) = (idx >> W_BITS, idx & ((1 << W_BITS) - 1))`, keeping the
`w_h` loop and the `idx > idx_rev` skip unchanged → 16× more tasks.

**Safety argument (required in the PR).** The existing `UnsafeMut` relies on
element-disjointness of writes: elements partition into unordered pairs
`{i, bit_reverse(i)}`, and the `idx > idx_rev` skip assigns each pair to
exactly one `(a, w_l, w_h)` iteration (a pair appears once as
`(a, w_l, w_h)` and once mirrored as `(rev(a), rev(w_h), rev(w_l))`; exactly
one side passes the comparison, palindromes handled separately). This
assignment is per-iteration, not per-`a`-task, so re-partitioning iterations
across tasks — including splitting on `w_l` — preserves disjointness. State
this in the unsafe-block justification per repo rules.

**Tests.** `bit_reverse_m31_works` exists in the file; extend to a size large
enough to exercise multiple parallel tasks under `--features parallel`.
**Effort:** ~1 hour + profiling. **Expected:** 0 on ≤16 cores; up to
core-count scaling on wider machines, capped by memory bandwidth.

## Findings checked and rejected (don't re-chase these)

- **`twiddle_at` "called millions of times"** (`simd/circle.rs:38`): false — in
  `eval_at_point` it runs once per 2^10-chunk and is advanced incrementally via
  `twiddle_steps`/`advance_twiddle`. The `TODO(Ohad): optimize` is about the
  function body, which is O(log n) and called rarely.
- **`COMBINE_CHUNK_SIZE = 16` too small** (`simd/quotients.rs:89`): comment
  says chosen empirically by benchmarking. Only revisit with fresh benchmarks.
- **CPU quotients per-row Vec / domain recompute** (`backend/cpu/quotients.rs`):
  real but the CPU backend only runs for sub-SIMD-size domains. Not worth it.
- **HashMap in lifted decommit** (`vcs_lifted/prover.rs:125-151`) and
  sort/dedup of queries: maps are query-count-sized (tiny); decommit cost is
  noise next to tree hashing.
- **Sumcheck `UnivariatePoly` clones, GKR `vec![zero]` alloc, claim `/2`
  division**: all on tiny data (degree ≤ 3 polys, per-round), negligible.
- **GKR `EqEvals::generate` "per instance redundancy"**: generated once per
  layer for all instances already.
- **Merkle tree per-layer allocation**: total allocation is 2× leaves across
  all layers; hashing dominates.
- **`to_lifted_simd` per element in quotient combine** (`simd/utils.rs:58`):
  it's a compile-time `simd_swizzle!` behind a small match — register-only,
  a few instructions. Nothing to batch.
- **`transpose_vecs` `i >= j` branch** (`fft/mod.rs:43-48`): the loop body is
  load/store dominated (bandwidth-bound); restructuring bounds inside tuned
  unsafe FFT code has poor risk/benefit.
- **`random_coeff_powers` indexed lookup + logup prev-column read**
  (`simd_domain.rs:106`, `logup.rs:197-215`): bounds check / `Vec::last()`
  deref next to a QM31 broadcast-multiply; noise.
- **`mix_felts` byte-buffer allocation, SIMD Blake2s `zeroed` init**: KB-sized
  / compiler-fused respectively.
- **`print_column_size_histogram`** (`pcs/mod.rs:420`): runs unconditionally
  per proof but is a few hundred HashMap ops; noise.

---

## Deep-rework candidates (scoped, not started)

### A. Task-graph parallelism in the quotient/opening phase

**Motivation.** `compute_quotients_and_combine` computes
`denominator_inverses` (a full-domain × n_samples pass) strictly *after* the
numerator accumulation, on the same thread pool. The codebase already flags
this: `simd/quotients.rs:80-81` — *"TODO(Leo): Consider receiving the
denominator inverses from the call site and having them computed in parallel
to other task."*

**Scope.**
- `crates/stwo/src/prover/backend/simd/quotients.rs`: split
  `denominator_inverses(&sample_points, ...)` out of
  `compute_quotients_and_combine`; compute it under `rayon::join` with the
  `accumulate_numerators_on_subdomain` loop in `accumulate_quotients`
  (sample points are known before accumulation starts). Composes with small
  item 3 (slice-based signature).
- Same for the packed `subdomain_points` collection (line 116) — independent
  of the accumulations.
- Optionally overlap `build_weights_hash_map` (`pcs/mod.rs:192`) with the
  twiddle-related setup preceding it; both are pure w.r.t. the channel.

**Constraints.** No Fiat-Shamir reordering: all overlapped tasks are pure
computations that don't touch the channel. Proof output unchanged.

**Expected win.** Hides one full-domain × n_samples pass behind the numerator
pass. Bounded — both passes are rayon-parallel already, so the win comes from
utilization at pass boundaries. Estimate 5-15% of the quotient phase, not of
the whole proof. Profile the quotients span before/after.

**Effort.** ~1-2 days incl. benchmarks. Low risk (no math changes).

### B. Component-level parallelism in composition-polynomial accumulation

**Motivation.** `crates/stwo/src/prover/air/component_prover.rs:118-120`
evaluates components sequentially with `&mut DomainEvaluationAccumulator`.
Per-component row loops are internally rayon-parallel, so the loss is
per-component parallel ramp-up/down and poor scaling for AIRs with many small
components (Cairo-style AIRs with dozens of small tables).

**Why it's a rework, not a fix.** Components with the same composition
log-size share one accumulation column and do read-modify-write
(`chunk.packed_at(i) + row_res`) — a true dependency. Options:

1. **Cheap first step:** group components by target log-size; different
   groups write disjoint columns → `par_iter` over groups, sequential within
   group. Needs only a borrow-splitting API on `DomainEvaluationAccumulator`
   (hand out disjoint `&mut` column slices, like `columns([...])` already
   does). No trait change.
2. **Full version:** per-component private accumulation columns + packed
   tree-reduce. Memory: one extra secure column of composition-domain size per
   concurrent component (boundable by chunking the domain and reducing per
   chunk). Probably wants `&self`-style column handles on the public
   `ComponentProver` trait.

**Expected win.** Zero for single-component AIRs. For N similar-size
components: up to min(N, cores)× on the constraint-eval span *only if*
individual components under-utilize the machine — measure first with a
multi-component benchmark from the examples crate.

**Effort.** Option 1: ~2-3 days. Option 2: ~1-2 weeks incl. memory tuning.
Gate on a profile showing constraint eval is a large, under-utilized span for
the target workload.

### C. Not recommended without profile evidence

- **FFT lazy-reduction / transpose-butterfly fusion:** speculative gains,
  heavy `unsafe`, soundness-adjacent (redundant-representation invariants —
  see `TODO(shahars)` at `simd/circle.rs:130`). Only pursue with a profile
  showing FFT reduction steps are the bottleneck, and with review per the
  CLAUDE.md supervised-change protocol.
- **Prover phase pipelining (commit N+1 vs quotients N):** Fiat-Shamir forces
  commit→draw→compute ordering per phase; the legal overlaps are exactly the
  ones scoped in (A).
