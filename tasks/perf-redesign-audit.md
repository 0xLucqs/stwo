# STWO Performance Redesign Audit (2026-07-21)

Six parallel deep-reads of the prover: SIMD field ops + FFT, FRI/PCS/quotients,
Merkle/hashing, lookups (GKR/LogUp/sumcheck), constraint evaluation, memory/parallelism.
Findings deduplicated and ranked. "Corroborated" = found independently by ≥2 auditors.

Cross-reference: `tasks/perf-audit-2026-07-06.md` independently found and scoped several
of the same items (its #1 = barycentric weights part of 2.1 here; #3 = denominator-inverse
domain regen in 2.9; #4 = offset-mask gathers, subset of 2.3; #6 = repeated_double memo
in 2.9) — all still unimplemented at HEAD 4f877db2, so the two audits converge. Its
"Findings checked and rejected" list stands; notably "Merkle per-layer allocation" was
rejected there on TIME grounds — 2.7 here claims PEAK MEMORY, which that rejection does
not cover. Its deep-rework candidates A (overlap denominator inverses with numerators)
and B (component-level composition parallelism) complement 2.2/2.9 here.

Verdict: yes — there is large headroom, most of it prover-only (no protocol/soundness
surface). The prover is well-designed at the algorithm level (no O(n log n) that should
be O(n), no redundant bit-reversals, batch inversion done right), but leaves 2-10x on
the table through: whole phases running single-threaded, redundant DRAM sweeps, scalar
gathers in the hottest loop, and build/codegen misconfiguration.

---

## Tier 0 — Free wins (hours of work, zero math risk)

### 0.1 `#[inline]` missing on all PackedCM31/PackedQM31 arithmetic
NOTE: commit b2a012c3 already inlined the SCALAR core/fields types; this finding is about
the packed SIMD types, which still lack the attributes (verified at HEAD 4f877db2).
`crates/stwo/src/prover/backend/simd/cm31.rs` (Add:77, Sub:85, Mul:93, Mul<PackedM31>:168,
Neg:177) and `qm31.rs` (Mul:116, Add:100, Sub:108, AddAssign:157, inverse:170) have no
`#[inline]`. These are non-generic fns called cross-crate from constraint-framework's
innermost loops → without LTO every QM31 mul is a function call with SIMD register spills.
`m31.rs` already does `#[inline(always)]` everywhere. Also `mul_twiddle` (fft/mod.rs:110).
Impact: up to 10-30% of constraint eval in non-LTO builds. Add `[profile.release] lto = "thin"` too.

### 0.2 No runtime SIMD dispatch / target-feature defaults
`m31.rs:185-198` selects mul_avx512/avx2/neon at compile time only. Plain
`cargo build --release --features prover` on x86 gets the portable SSE2 multiplier —
2-4x slower field mul. Bench/CI scripts pass `-C target-cpu=native` so this is invisible
in benchmarks. Fix: runtime dispatch at column-op granularity (`is_x86_feature_detected!`
resolved once) or at minimum document + ship target-feature hints.

### 0.3 Loop-invariant hoists in constraint eval
- `logup.rs:51`: `cumsum_shift = claimed_sum / 2^log_size` — a QM31÷M31 (4 M31 inversions,
  ~150 muls) recomputed **per vec-row** inside `LogupAtRow::new`. Hoist to once per component.
- `simd_domain.rs:112-114`: `VeryPackedSecureField::broadcast(random_coeff_powers[i])`
  per constraint per row — pre-broadcast once per component.
- `constraint-framework/src/prover/logup.rs:193-231` `finalize_col`: per-packed-row
  `Option` branch on `trace.last()` is loop-invariant — hoist.

### 0.4 Rayon granularity bugs
- `component_prover.rs:29`: `CHUNK_SIZE = 1` → one rayon task per 64 rows; ~260k tiny
  tasks at 2^24. Raise to 16-64+ (quotients uses 16-64, FRI 128).
- `constraint-framework/src/prover/logup.rs:193`: `chunk_size = min(4, len)` — same
  pathology in interaction-trace gen. Scale to len/threads or `with_min_len(1<<10)`.

### 0.5 `build_leaves` memsets a multi-GB buffer that's never read
`blake2s_lifted.rs:94-99`: `prev_layer_states` (~32B/leaf, ~8GB at 2^28) is fully filled
with `INITIAL_STATE` but only index 0 is ever read before being overwritten. Fill the
prefix only. One full write pass over the largest buffer, per tree, deleted.

### 0.6 GKR lambda hoist
`simd/lookups/gkr.rs:370-371,430-431,481-482` (+ CPU mirror, existing TODO at cpu gkr.rs:92):
keep 4 accumulators (Σeq·num, Σeq·den at t=0,2), apply λ once after the loop instead of
2 packed QM31 muls by λ per iteration. Saves ~2 of ~7 muls per iteration. Bit-identical.

---

## Tier 1 — Parallelism gaps (prover-only, low risk)

### 1.1 The entire GKR/sumcheck/MLE stack is single-threaded  [CORROBORATED x2]
DEPRIORITIZED (2026-07-21): production lookups use the LogUp prefix-sum path, which is
already parallel; GKR is the slower alternative and not the production route. Only worth
doing if/when a GKR- or sumcheck-based path (e.g. MLE-eval components) ships hot.
Detail: zero rayon in `prover/lookups/{gkr_prover,sumcheck,utils}.rs` and
`backend/simd/lookups/{gkr.rs,mle.rs}` — `gen_eq_evals`, `next_*_layer`,
`eval_*_sum`, `fix_first_variable` are all serial packed loops, even with `parallel`.
Fix when relevant: `par_chunks` with per-thread accumulators (bit-identical output).

### 1.2 Other uncovered-by-`parallel` hot spots
- `simd/accumulation.rs:45-78` `lift_and_accumulate`: serial full-domain pass, runs at
  two pipeline points (composition finalize + quotient grouping). Embarrassingly parallel.
- `prefix_sum` finalize (`constraint-framework/prover/logup.rs:124-164`): 4 independent
  coordinate scans run sequentially + an extra shift-subtraction pass that can fuse.
- `blake2s_lifted.rs:284-298` `pack_leaves_input` (FRI first-layer): serial scalar
  transpose (existing TODO(Leo)); rayon + simd_swizzle 4x4 block transpose.
- `simd/column.rs:275-292` `into_secure_column_by_coords`: sequential push-loop unzip
  of full secure columns.
- `simd/circle.rs:161-165`: iFFT 1/N scaling pass is single-threaded (see 2.4 — better fused).
- Structural: `parallel` is ~40 `#[cfg]` fork sites; `parallel_iter!` macro exists but is
  used in ~6 files. Consolidate to prevent drift.

---

## Tier 2 — Algorithmic redesigns, prover-only (phase-level 2x+, medium effort)

### 2.1 OODS eval runs over the full LDE instead of the trace-size prefix  [CORROBORATED x2]
`pcs/mod.rs:135-175, 205-236` + `simd/circle.rs:239-350`: barycentric path builds weights
for the *blown-up* domain and dot-products over the entire LDE per (column, point).
The first N/2^blowup bit-reversed rows are a complete evaluation on the split subdomain
(same prefix trick quotients already exploit). Evaluate there instead.
Impact: ÷blowup on the whole EvaluateOutOfDomain phase (2x default, 16x at blowup=4).
Also: `barycentric_weights` computes domain points via O(log n) scalar ops per element
(`domain.at(bit_reverse_index(..))` + scalar `point_vanishing` per lane) while
`CircleDomainBitRevIterator` exists — 10-50x on weight generation.

### 2.2 Quotient numerator accumulation streams the whole trace once per sample batch
`simd/quotients.rs:67-77, 208-253`: one full pass over every column's subdomain data per
`ColumnSampleBatch`; a column with a 2-point mask + periodicity batch gets streamed 2-3x
from DRAM in a bandwidth-bound phase. Invert the loop: one chunked pass over rows,
per-batch accumulators in-register, each column value loaded once. Also fold the separate
`lift_and_accumulate` merge into per-point max-size accumulation.
Impact: ~2-3x on the FRIQuotients accumulation stage.

### 2.3 Offset mask reads are scalar gathers — hits every LogUp component every row
`simd_domain.rs:94-106` + `column.rs:107-109` (`at()` spills a whole 16-lane vector per
scalar read) + `utils.rs:147-163`: any `offset != 0` mask does 32 scalar index computations
+ 32 `at()` calls per vec-row. The universal LogUp `[-1,0]` cumsum read = 128 scalar
gathers + 256 bit-reverse index computations per vec-row, per component; the 32 indices are
recomputed 4x (once per secure coordinate).
Fix: materialize a rotated column once per (column, offset) before the row loop
(`bit_reverse(rotate(bit_reverse(col)))`) → contiguous packed loads.
Cheaper step: hoist/share index vectors across coordinates. Likely tens of % of
constraint point-wise eval for LogUp AIRs.

### 2.4 Redundant full-column DRAM sweeps in the FFT pipeline  [CORROBORATED x2]
- `simd/circle.rs:161-165`: iFFT 1/N normalization is a separate serial read-modify-write
  pass per interpolated column (stale TODO(alont) on caching the inversion too). Fuse the
  scalar into the last iFFT layer's write-back. ~10-15% of iFFT time.
- `fft/{ifft,rfft}.rs`: radix-8 kernels; AVX-512's 32 registers fit radix-16 (`ifft4`/`fft4`)
  → ~25% fewer full-working-set sweeps in the middle sections. Keep radix-8 on NEON.
- `fft/mod.rs:36-56` `transpose_vecs`: unblocked power-of-2-strided swap transpose — worst
  case for cache sets/TLB. Tile 8x8 vectors. Classic 2-3x on the transpose sweep.
- `fft/mod.rs:14`: `CACHED_FFT_LOG_SIZE = 16` hardcodes an x86-L2 guess; per-arch constant
  (Apple Silicon has 12-16MB shared L2).

### 2.5 Sumcheck: fuse fold with next-round evaluation
`lookups/sumcheck.rs:81-126`: per round the layer is streamed once for f(0)/f(2), then
again by `fix_first_variable`. After the challenge is drawn, a fused
`fix_first_variable_and_sum` writes the folded half and accumulates the next round's
evals in the same pass. ~40% memory-traffic reduction across all sumcheck rounds;
compounds with 1.1. Needs an optional trait method with the current 2-pass default.

### 2.6 Over-lifted Merkle trees hash exponentially many duplicates
`vcs_lifted/prover.rs:71-74` + `blake2s_lifted.rs:192-213`: when global
`lifting_log_size = L` exceeds a small tree's own max column size m (small
preprocessed/interaction trees in multi-tree proofs), every layer between L and m hashes
2^layer nodes of which only ≤2^(m+1) are distinct — redundancy 2^(L-m) in hash work and
memory. Compute distinct hashes only and replicate (or lazy repeated-view layers).
Roots bit-identical, verifier untouched. 4-100x on affected trees.
[SECURITY-CRITICAL file → supervised protocol, but output-identical.]

### 2.7 Peak memory: all Merkle layers + all GKR layers retained  [CORROBORATED x2]
- Merkle: `vcs_lifted/prover.rs:20` keeps every layer of every tree (~64B/row/tree,
  ~16GB at 2^28) until decommit, which reads only ~n_queries×L siblings. Prune: keep every
  k-th layer, recompute 2^k-subtrees at decommit (~n_queries×2^k hashes, trivial).
  2-4x peak-memory; 5-15% commit-time from saved write bandwidth.
- GKR: `gkr_prover.rs:513-518` materializes all layers up front (≈4x input columns,
  all SecureField); drop-as-consumed via `Vec::pop`, or k-th-layer checkpointing.
- Mempool (`mempool.rs`) covers only BaseField commit evaluations. Generalize to pool the
  4 eval-domain columns in `compute_quotients_and_combine`, FRI fold outputs
  (existing TODO fri.rs:60), and secure-column accumulators — these are allocated exactly
  at peak. Also `mempool.rs:28-33` `reserve` zero-inits buffers that get fully overwritten.

### 2.8 Constraint-framework batching path: per-row heap traffic
`simd_domain.rs:60-65` (fresh `LogupAtRow` + Vec alloc per vec-row),
`lib.rs:210-230` (`finalize_logup_batched` builds a second Vec with 512-byte
VeryPackedSecureField clones per row), `component_prover.rs:190` (TreeVec of up to 6 Vecs
allocated per chunk, amplified by CHUNK_SIZE=1). Stream the batching incrementally
(batch size known before the loop) and hoist trace refs out of the rayon closure.
~8 allocs + multiple KB of copies per vec-row deleted.

### 2.9 Smaller contained items
- `compute_quotients_and_combine` (`simd/quotients.rs:188-197`): 4 coordinate
  interpolate+evaluate pipelines run sequentially — rayon::join them; compute denominator
  inverses per chunk instead of materializing full subdomain vectors (TODO(Leo) at :80).
- `pcs/mod.rs:160-171`: `point.repeated_double(...)` recomputed per (column, point) —
  hoist per (log_size, point) (existing TODO(Leo)).
- `eval_at_point` chunk loop (`simd/circle.rs:195-221`): scalar array round-trips for
  twiddle broadcasts; keep packed + lazy u64 accumulation.
- GKR t=2 → t=1/2 evaluation point change: replaces double+sub with adds in the sum
  kernel (~15-25% of round-poly ALU with the λ hoist); touches interpolation nodes in
  `correct_sum_as_poly_in_first_variable` only — protocol unaffected.
- Lazy/deferred reduction in PackedCM31/PackedQM31 mul (`cm31.rs:90-102`, `qm31.rs:116-139`):
  accumulate 62-bit u64 intermediates, reduce once per output coordinate — ~15-30% faster
  QM31 mul → ~3-8% total. Needs exact range analysis; prover-only, randomized tests cover.
- `mle_eval.rs:692-722` `gen_carry_quotient_col`: scalar loop with O(n) distinct values
  filled per-element (explicit TODO(andrew)) — 50-100x on that column gen.
- Keccak256 Merkle + grind are fully scalar while a tested 8-way `keccak_f1600x8` sits
  unused (`keccak256_permutation.rs:69`); 10-50x if the Keccak channel ships.

---

## Tier 3 — Protocol-visible changes (soundness surface — supervised protocol + human approval)

### 3.1 FRI first/inner layers commit one QM31 per leaf in ALL production configs
`prover/fri.rs:295-298, 357-359 (explicit TODO), 368-370`: `pack_leaves` engages only when
`fold_step > 1`, and every config in the repo uses `fold_step = 1` → ≈4N extra Blake
compressions per proof (comparable to committing 30-60 extra base columns) + 2 extra
decommitment levels per tree. Support packed leaves at fold_step=1 or default fold_step=2.
Changes Merkle structure/transcript/verifier → full supervised review.

### 3.2 Uniform composition eval domain wastes work in mixed-degree compositions
`component_prover.rs:88`: every component evaluates on
`trace_log_size + max(composition_log_split over ALL components)`. A degree-excess-1
component alongside a degree-excess-2 one does 2x its needed constraint-eval work.
Per-component domains + per-component verifier lifting = protocol-visible. Up to 2x for
low-degree components; only in mixed-degree compositions.

---

## Defensive fix regardless
`simd/bit_reverse.rs:39-42`: `ColumnOps<SecureField>::bit_reverse_column` is `todo!()` —
a runtime panic landmine. Implement via 4x `bit_reverse_m31` or remove the impl.

## Verified as already good (no action)
Batch inversion (Montgomery trick, chunked, buffer-reused); no redundant bit-reversals in
the FFT pipeline (native bit-reversed convention end-to-end); quotient accumulation is O(N)
with no per-query recomputation; `build_leaves` lifts lazily with progressive chunk sizes;
blake2s is 16-way SIMD + rayon; blake grind is parallel work-stealing; four-step transpose
FFT structure; twiddles precomputed per tree with batch inversion; constraint eval is fully
monomorphized (no JIT opportunity); sumcheck already derives f(1) from the claim; NEON
register-pressure tuning documented.

## Suggested attack order (revised after GKR deprioritization)
1. Tier 0 (a day of work total, several % + protects downstream builds)
2. 2.1 OODS subdomain eval + 2.2 quotient single-pass (phase-level 2-3x each, low risk)
3. 2.3 rotated mask columns (biggest constraint-eval win; LogUp path)
4. 2.4 FFT sweep reductions, 2.6/2.7 memory redesigns
5. Tier 3 only with the supervised soundness workflow.
6. GKR/sumcheck items (1.1, 2.5, GKR parts of 2.7/2.9) only if that path ships hot.
