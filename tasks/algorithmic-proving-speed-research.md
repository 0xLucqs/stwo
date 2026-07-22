# Algorithmic proving-speed research

> Scope note (2026-07-22): this report is retained as the PCS/FRI-focused appendix. It over-centered
> Lifted FRI relative to the user's actual global-algorithm objective. The corrected cross-layer
> analysis and roadmap are in
> [`global-prover-algorithm-research.md`](global-prover-algorithm-research.md).

Date: 2026-07-21  
Repository: `stwo`, branch `dev-copy`, HEAD `9c5bebf1`

## Executive conclusion

Do **not** start by replacing Circle FRI with a protocol called “Lifted FRI”. S-two already has
most of the practical lifting stack: mixed-height columns are virtually lifted in the commitment,
mixed-size DEEP quotients are lifted and accumulated into one largest-domain polynomial, and one
Circle FRI proves that combined polynomial. The full HHM25 Lifted-FRI manuscript cited by the
S-two whitepaper is still unpublished, and no primary benchmark demonstrates a Circle-STARK
prover-speed gain.

The best program is instead:

1. Establish one production-security, end-to-end phase profile. Current Criterion coverage does
   not benchmark the active full SIMD/lifted-VCS/FRI path.
2. Fix the largest measured or structurally repeated work without changing the proof:
   trace-domain OODS evaluation where coefficients are not retained, row-major DEEP quotient
   accumulation, degree-aware composition domains, and repeated lifted-Merkle layers.
3. Benchmark the already-implemented `fold_step = 2` before designing a new FRI. It is currently
   exercised in tests but not used by any non-test repository configuration.
4. Prototype **first-layer-elided FRI with a directly folded LDE**. This is the most promising
   mathematical change: avoid the full quotient LDE and quotient Merkle tree, fold in coefficient
   space, and commit first at size `N / 2^k`.
5. Consider full Lifted FRI only for strongly heterogeneous multi-table workloads or recursion,
   after HHM25 is available and the existing soundness gaps are resolved.

The likely winner is a portfolio, not one protocol swap. A FRI-only redesign cannot accelerate
trace generation, trace LDEs, constraint evaluation, or lookup generation, and therefore has a
strict Amdahl bound unless FRI is shown to dominate the target workload.

All speed ranges below are hypotheses or phase-level bounds, not measured whole-prover promises.

## What the prover actually does today

The active path is [`prove_ex`](../crates/stwo/src/prover/mod.rs), followed by PCS opening in
[`CommitmentSchemeProver::prove_values`](../crates/stwo/src/prover/pcs/mod.rs):

1. Interpolate/extend each committed trace tree and build a lifted Merkle commitment.
2. Evaluate component constraint quotients and commit the split composition polynomial.
3. Sample the OODS point and evaluate all requested columns there.
4. Construct one batched DEEP quotient from all opening claims.
5. Extend that quotient to the full FRI domain and commit it as the first FRI layer.
6. Draw fold challenges, fold, and commit the remaining FRI layers.
7. Grind, sample queries, and decommit all trace and FRI trees.

Important implementation facts:

- Production PCS trees use [`MerkleProverLifted`](../crates/stwo/src/prover/vcs_lifted/prover.rs),
  not the legacy VCS benchmarked by `benches/merkle.rs`.
- [`FriConfig::fold_step`](../crates/stwo/src/core/fri.rs) already supports multi-step folding;
  prover/verifier tests cover steps 2–4.
- Four-row FRI leaf packing is enabled only when `fold_step > 1` in
  [`FriFirstLayerProver`](../crates/stwo/src/prover/fri.rs) and `FriInnerLayerProver`.
- Commitments default to discarding polynomial coefficients, but several examples explicitly call
  `set_store_polynomials_coefficients()`. OODS optimization priorities therefore depend on the
  real production configuration.
- The important “accumulate on a trace-size subdomain, then interpolate and extend” DEEP-quotient
  optimization already landed in commit `bd8c7449`. It should not be proposed again.
- Production lookups use prefix-sum LogUp. GKR is a test/benchmark path and remains slower in every
  measured local cell.

## Lifted FRI: definition, current overlap, and verdict

### What full Lifted FRI means

The [S-two whitepaper](https://www.researchgate.net/publication/407022099_S-two_Whitepaper)
describes Lifted FRI as a multi-domain PCS in which:

- a heterogeneous function ensemble is viewed as lifted to the largest evaluation domain;
- a branching/memoized bit-reversed Merkle strategy avoids paying for physical duplication;
- the FRI cascade is independent of the individual ensemble domains;
- the verifier becomes uniform; and
- cross-domain correlated agreement is obtained by construction, simplifying soundness.

This is not the same thing as the directory name `vcs_lifted`. The current VCS virtually lifts
mixed-height commitment columns, but FRI still receives one materialized, combined DEEP quotient
and runs as ordinary Circle FRI.

### How much is already present

S-two currently implements the practically important decomposition also used by Miden:

1. Virtually lift mixed-height committed matrices.
2. Compute smaller trace quotients at their native sizes.
3. Lift and batch them into one largest-domain DEEP polynomial.
4. Run one regular low-degree test.

Miden's 2026
[lifted-STARK implementation](https://docs.rs/crate/p3-miden-lifted-stark/latest) and
[lifted FRI PCS documentation](https://docs.rs/p3-miden-lifted-fri/latest/p3_miden_lifted_fri/)
corroborate this architecture: LMCS virtual upsampling, DEEP batching, then configurable-arity
FRI. Its design notes also require a **liftable AIR**: wrap-around transitions cannot silently
change semantics when a shorter trace is repeated. The
[Plonky3 lifting design note](https://hackmd.io/HkfET6x1Qh-yNvm4fKc7zA) explicitly favors LMCS,
lifted DEEP, and regular FRI over non-black-box Lifted FRI for practical implementation.

### Expected benefit

Full Lifted FRI is most attractive when there are many materially different trace heights or when
a uniform recursive verifier is the primary goal. It has little automatic single-proof benefit
when one trace height dominates, because S-two already builds one combined quotient and executes
one FRI.

Verdict: **strategically interesting, not a near-term speed recommendation**. Prototype only after
the unpublished construction and its Circle adaptation can be reviewed, and compare it against
the best `fold_step = 2` plus memoized-VCS baseline.

## Ranked opportunities

| Rank | Candidate | Main work removed | Compatibility | Evidence / expected effect |
|---:|---|---|---|---|
| 1 | Trace-domain/algebraic OODS fast path | Full blown-up barycentric-weight construction | Proof-identical | 441 ms weight build versus about 1 ms weighted dot product at `2^20` in local microbench artifacts; conditional on coefficients not being retained |
| 2 | First-layer-elided FRI + directly folded LDE | Full quotient LDE, full quotient Merkle tree, first `k` pointwise folds | Proof/transcript change | Largest structural FRI/PCS saving; must trade against extra source-trace openings |
| 3 | Degree-aware component composition domains | `2^(K-K_i)` excess constraint work for low-degree components | Likely proof-identical | Downstream uniform-`K` measurement was 9.5% slower despite a 17.8% smaller proof |
| 4 | `fold_step = 2` and joint FRI leaf scheduling | Intermediate FRI trees, roots, auth levels, under-filled hash leaves | Already implemented; config/proof change | Lowest-effort protocol experiment; modern FRI implementations commonly tune arity 4 |
| 5 | Row-major DEEP quotient accumulation | Re-reading the same trace columns for each sample batch | Proof-identical | Estimated 2–3x for the numerator-accumulation subphase; current subdomain algorithm remains |
| 6 | Materialized rotated masks and row-major constraints | Scalar gathers and repeated bit-reversed index work | Proof-identical | Likely material for mask-heavy prefix-sum LogUp; must be measured end-to-end |
| 7 | Memoized/checkpointed lifted Merkle layers | Repeated nodes and retained full layer arrays | Root-identical when applied conservatively | Theoretical affected-tree factor approaches `2^(L-m)` for a tree lifted from `2^m` to `2^L`; whole-proof impact depends on height skew |
| 8 | Batch independent STARK instances into one PCS/FRI | Repeated challenge, opening, FRI, query, and PoW fixed costs | New batch API/proof | No single-proof gain; high value for services processing many small proofs |
| 9 | High-radix/fused Circle FFT pipeline | Full-memory sweeps and transpose traffic | Proof-identical | Plausibly 10–30% of the FFT phase; whole-proof effect must be profiled |
| 10 | Flat-sumcheck/hybrid MLE constraints | Running-sum lookup columns or `2^K` high-degree domains | Major protocol work | Promising only for high-degree, sparse, or structured workloads; current generic GKR loses |
| 11 | New PCS family: WHIR/PIPFRI/Blaze | Architectural FFT/hash complexity | System migration | Long-term research, not a Circle-FRI drop-in |

### 1. Trace-domain and algebraic OODS evaluation

When coefficients are discarded, S-two builds a barycentric-weight vector for every distinct
`(log_size, OODS point)` over the **full committed LDE domain**. At `2^20`, local Criterion
artifacts measured:

- SIMD barycentric-weight construction: **441.348 ms**;
- weighted dot product after weights exist: **0.982 ms**; and
- coefficient-form evaluation: **1.387 ms**.

This is a phase microbenchmark, not a whole-proof percentage, and it is irrelevant to trees whose
coefficients are deliberately retained. It nevertheless exposes a severe algorithmic asymmetry.

Two compatible changes should be studied:

1. Evaluate from the trace-size complete subdomain rather than all `B = 2^beta` LDE cosets. The
   arithmetic falls by `B`: 2x at `beta = 1`, 16x at `beta = 4`.
2. In [`point_vanishing`](../crates/stwo/src/core/constraints.rs),
   `v_i(p) = h_y / (1 + h_x)`. Construct its inverse as
   `v_i(p)^-1 = (1 + h_x) / h_y`, generate points with the bit-reversed domain iterator, and use
   one batch inversion rather than a secure-field division per element.

The first change needs a reviewed barycentric formula for the selected non-canonical subcoset.
The result must match current OODS evaluations and proof bytes exactly. A simpler alternative is
to retain coefficients selectively for trees with many OODS openings, trading memory for time.

### 2. Elide the first FRI commitment and fold before the LDE

This is the most interesting protocol-level trick found.

Today the prover:

1. computes the combined DEEP quotient on a trace-size subdomain;
2. performs four coordinate iFFTs and an LDE to the full FRI domain `N`;
3. commits a full `N`-leaf quotient Merkle tree;
4. derives `alpha`; and
5. folds the committed quotient.

The verifier already reconstructs a queried quotient value from the original trace openings and
OOD claims. In the original [FRI IOPP](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ICALP.2018.14),
the input word is an external oracle; auxiliary FRI commitments begin after folding. S-two can
exploit the same separation:

1. Bind the trace roots, composition root, OOD claims, and batching challenges.
2. Draw `k` independent fold challenges.
3. Interpolate the trace-size quotient as today, but decompose/fold its coefficients directly.
4. Evaluate only the folded line polynomial on a domain of size `N / 2^k`.
5. Make that smaller evaluation the first auxiliary FRI commitment.

For `k = 1`, this removes the full-domain quotient LDE, the full quotient Merkle tree, and the
circle-to-line pointwise fold, replacing them with one line-domain evaluation at half size. Larger
`k` removes more early work.

The cost is query amplification. The current quotient tree supplies sibling quotient values
cheaply. Without it, the verifier must reconstruct all `2^k` source quotient values, requiring
extra openings from every participating trace tree. This is attractive for narrow trees and can be
bad for wide AIRs. A selective variant can split the quotient by tree/log-size, leaving a committed
partial quotient for wide groups while reconstructing narrow groups.

Required gate:

- benchmark narrow, medium, and very wide traces;
- record quotient LDE, Merkle, decommit, proof size, and verifier time separately;
- use independent QM31 challenges for skipped rounds or prove a bound for power-derived
  challenges;
- prove the quotient lies in the Circle-FFT space required by Circle FRI; and
- add adversarial tests for every reconstructed sibling and transcript reordering.

### 3. Evaluate every component on its smallest valid composition domain

Current generalized composition splitting chooses one global excess

`K = max_i(max_constraint_log_degree_bound_i - trace_log_size_i)`

and evaluates every component on `2^(n_i + K)` points. A component with local excess `K_i < K`
therefore pays an avoidable factor `2^(K-K_i)`.

The better algorithm is:

1. evaluate component `i` only on `2^(n_i + K_i)` points;
2. interpolate its quotient;
3. extend/lift it into the global `n_max + K` composition accumulator; and
4. retain the existing global composition split and verifier identity.

This reuses the exact interpolate-then-extend idea that already improved DEEP quotient
construction. It can likely preserve the proof format and verifier, although degree declaration,
lifting exponents, and under-declaration behavior are soundness-critical.

The strongest evidence is a downstream experiment already recorded in `tasks/todo.md`: raising
the global split reduced proof size 17.8% but increased proving time 9.5%; the STARK phase gained
712 ms because uniform `K` doubled all components' constraint domains. This proposal targets that
specific regression rather than reverting generalized composition splitting.

An AIR-level companion optimization is to lower the one high-degree outlier with auxiliary
columns when

`extra trace cost < saved constraints * (2^K - 2^K')`.

For sparse or punctuated activity, a smaller component table connected by LogUp may similarly beat
a maximum-height table full of selectors. It must prove identical semantics; arbitrary rows cannot
simply be skipped.

### 4. Tune high-arity Circle FRI before replacing it

`fold_step = k` performs `k` binary folds per commitment round, i.e. arity `2^k`. The current code
already supports and tests this. Approximate inner-layer domain totals, ignoring the final tail,
fall from a binary geometric series to:

- `k = 1`: about `N` values across committed inner layers;
- `k = 2`: about `N/3`;
- `k = 3`: about `N/7`.

The folding arithmetic stays linear as a geometric series, but fewer trees mean fewer roots,
Merkle nodes, authentication levels, allocations, and full-memory passes. `k = 2` also matches the
existing four-row leaf packer and one 64-byte Blake2s message of four QM31 evaluations.

Costs are not monotone:

- each query opens `2^k - 1` siblings for a folded coset;
- larger arity increases local interpolation and proof values;
- a large final polynomial can dominate proof size; and
- current multi-fold challenges use `[alpha, alpha^2, alpha^4, ...]`, which has a different
  bad-challenge polynomial than independent challenges.

Immediate experiment: sweep `fold_step = 1, 2, 3, 4`, last-layer log degree, and packing at fixed
soundness. Start with `k = 2`; only keep `k = 3/4` if proof/verifier costs remain acceptable. Miden's
current FRI exposes arities 2, 4, and 8 and documents the same round-count versus sibling-opening
tradeoff in its
[FRI parameters](https://docs.rs/miden-lifted-stark/latest/src/miden_lifted_stark/pcs/fri/mod.rs.html).

### 5. Make DEEP quotient accumulation row-major

The current SIMD code iterates sample batches and streams participating trace columns once per
batch. Two-point masks and periodic columns can therefore cause 2–3 full DRAM reads of the same
data.

Invert the loop:

- process one packed row/chunk at a time;
- load each trace column once;
- update all relevant sample-batch accumulators in registers;
- share generated subdomain points across all denominator sets; and
- overlap pure denominator generation with numerator accumulation.

The quotient polynomial, transcript, proof, and verifier remain identical. The local redesign
audit estimates a 2–3x improvement for the numerator-accumulation subphase. Four independent
coordinate interpolation/evaluation pipelines can also run concurrently.

### 6. Materialize rotated masks and evaluate constraints row-major

The active prefix-sum LogUp path reads previous-row and other rotated masks. The SIMD domain
evaluator currently derives 32 scalar bit-reversed indices and performs 32 scalar `at()` gathers
for each nonzero offset and packed row. This defeats contiguous SIMD access in a hot, real-proof
path.

Before constraint evaluation, materialize one rotated bit-reversed view per distinct
`(column, offset)` and consume it with packed loads. When memory is too expensive, compute the 32
indices once per offset and share them across all four secure-field coordinates and all constraints
that use the same mask. Then make the outer traversal row/chunk-major so multiple constraints reuse
loaded masks.

This is algorithmic layout work rather than new cryptography: polynomial values and proof bytes are
unchanged. It is especially relevant because every normal LogUp relation reads the previous row.
The current tiny work chunks (`CHUNK_SIZE = 1` in component proving and at most four packed rows in
LogUp finalization) should be retuned in the same experiment, but only after the memory-access change
is profiled.

### 7. Finish the memoized-Merkle part of lifting

The lifted leaf builder already avoids naively re-hashing all short-column prefixes: it hashes at
native heights, lifts intermediate hash states, and continues when larger columns enter. However,
[`MerkleProverLifted::commit`](../crates/stwo/src/prover/vcs_lifted/prover.rs) still builds and stores
every binary layer up to `lifting_log_size`.

When an entire commitment tree has native maximum height `m` but is lifted to global height `L`,
large parts of the upper construction repeat. Represent repeated subtrees symbolically or memoize
their roots. Conservatively implemented, the root and serialized verifier proof can remain
bit-identical.

The affected-tree work can approach a `2^(L-m)` reduction, but this does **not** imply the same
whole-proof speedup. It is valuable only with a material height gap. Add a production lifted-VCS
benchmark; the existing `benches/merkle.rs` covers the legacy tree and cannot answer this question.

Separately, checkpoint every `k`th Merkle layer and recompute queried micro-subtrees after query
sampling. This can reduce peak Merkle storage 2–4x and may improve wall time only when memory
bandwidth or allocator pressure is limiting.

### 8. Batch independent proofs through one opening protocol

S-two already batches tables and columns inside one PCS. A proof service can go further: batch
independent STARK instances so they share quotient aggregation, FRI, query sampling, and PoW.
Plonky3 now ships a
[batch-STARK layer](https://docs.rs/crate/p3-batch-stark/latest) specifically described as reusing
FRI openings across instances.

This gives no single-large-proof improvement, but it can significantly amortize fixed costs across
many small or uneven proofs and improve core utilization. The batch reduction must bind every
instance independently and must not permit challenge cancellation between instances.

For repeated programs/tables, also cache immutable preprocessed-column FFTs, lifted commitment
roots/prover data, barycentric weights, and twiddles across proofs. This is amortization rather than
a new proof algorithm, but unlike ordinary memoization it can remove an entire fixed commitment
tree from every subsequent proof while leaving the verifier statement unchanged.

### 9. Reduce Circle-FFT memory passes rather than replace Circle FFT

The [Circle STARK paper](https://eprint.iacr.org/2024/278) reports the native M31/circle design as
competitive with traditional small-field STARKs, and the S-two whitepaper states that Circle FFT
has fewer arithmetic operations than the closely related transform alternatives it compares.
Replacing it with G-FFT, Chebyshev FFT, or a multiplicative-domain FFT is therefore not an obvious
algorithmic win.

Better targets are:

- fuse inverse-FFT normalization into the final butterfly write;
- use cache-tiled transposes;
- evaluate radix-8 on AVX2/NEON and radix-16 on AVX-512;
- fuse coefficient scaling and adjacent transform stages; and
- use pruned/truncated transforms for known lower-degree columns.

These preserve the proof and asymptotic complexity while reducing expensive DRAM sweeps.

### 10. Flat sumcheck and specialized lookup algorithms

The S-two whitepaper itself lists an “entirely flat constraints” direction: use the univariate flat
sumcheck from [Aurora](https://eprint.iacr.org/2018/828) with the Laurent/trigonometric Circle
representation to prove a column sum without prefix-sum neighbor constraints. This could remove
running-sum LogUp columns and their previous-row masks.

The local warning is strong: optimized GKR still lost to prefix-sum LogUp in all eight measured
cells. At `2^20`, batch 64, LogUp took 2105.763 ms and 3155.9 MiB; GKR took 2408.856 ms and
5250.8 MiB. GKR should be reopened only when it removes a `2^K` high-degree domain, amortizes one
sumcheck across many relations, or targets hardware where its arithmetic is favorable.

Workload-specific alternatives remain interesting:

- [LogUp*](https://eprint.iacr.org/2025/946) avoids additional indexing-length arrays for small
  indexed tables;
- [Twist and Shout](https://eprint.iacr.org/2025/105) reports large memory-checking gains for
  MLE/Sumcheck systems; and
- Lasso-style structured tables are useful when the table is too large to materialize.

These are not drop-in replacements for general S-two LogUp. They imply an MLE adapter or a hybrid
PCS and should be justified by a concrete lookup-dominated workload.

## Sound parameter optimization is an algorithmic project

Each bit removed from `log_blowup_factor` halves major evaluation-domain work. Jointly optimizing

`(blowup, queries, grinding, fold arity, final degree, batching challenges)`

could therefore outperform a low-level kernel change. It cannot use the current
`FriConfig::security_bits() = log_blowup * n_queries` heuristic.

Reasons:

- [Concrete non-interactive FRI analysis](https://eprint.iacr.org/2024/1161) found the provable
  security of most examined deployed parameter sets 21–63 bits below their conjectured security.
- [Small-field hash-based SNARG analysis](https://eprint.iacr.org/2025/2197) gives new attacks whose
  success depends on list sizes of extension codes over small base fields—the exact regime that
  makes M31 fast.
- The S-two whitepaper distinguishes proven Johnson-regime bounds from conjectural list/line
  decoding regimes and does not support the legacy `beta` bits per query estimate.
- Current S-two uses one PoW after FRI commitments, so it directly grinds the query draw, not every
  earlier batching/fold challenge assumed by some parameter tables.

A real optimizer needs a reviewed random-oracle error budget containing batching error, every FRI
round, field/list-size terms, query sampling, grinding location, domain size, and hash collision
security. Only then should lower blowup or fewer queries be claimed as a speedup.

Multilinear/tensor batching is relevant here. Replacing powers of one challenge across a very large
batch with `O(log M)` independent challenges can reduce the bad-challenge degree while retaining
`O(MN)` pointwise work. Its value is not the extra challenges themselves; it may permit safer
parameter reductions.

## Alternative PCS/IOPP families

| Scheme | Primary advantage | Published evidence | STWO fit |
|---|---|---|---|
| [DEEP-FRI](https://eprint.iacr.org/2019/336) | Better soundness by sampling outside the evaluation domain | Linear arithmetic prover retained | S-two already constructs a global OOD/DEEP quotient; per-round DEEP is a soundness/query tool, not an obvious speed win |
| [STIR](https://eprint.iacr.org/2024/390) | Fewer RS queries and smaller arguments | Paper reports materially smaller proofs, with prover time in the same broad range | Multiplicative-domain protocol; primarily proof/verifier work |
| [WHIR](https://eprint.iacr.org/2024/1586) | Very fast verifier and small proofs | Strong published verifier/proof tradeoffs | Constrained-RS/MLE/sumcheck architecture; no published Circle-WHIR construction |
| [BaseFold](https://eprint.iacr.org/2023/1705) / DeepFold | Efficient multilinear/foldable-code PCS | Good MLPCS comparisons | Replaces the univariate Circle PCS and AIR integration |
| [Brakedown](https://eprint.iacr.org/2021/1043) | Linear-time field-agnostic prover | Built system; larger proofs/verifier | R1CS/Spartan and linear-code migration |
| [Blaze](https://eprint.iacr.org/2024/1609) | Fast linear-time binary-field MLPCS at enormous scale | At `2^29`, 30.5 s versus BaseFold 145.7 s on 192 vCPUs; Brakedown can be faster with about 10x larger proofs | Binary extension field, MLE, different workload and hardware; not comparable to M31 Circle STARKs |
| [PIPFRI](https://www.usenix.org/conference/usenixsecurity26/presentation/li-weihan) | Combines linear-code proving with FRI compactness | Reports 10x versus DeepFold and 3.5x versus Orion | Very promising long-term MLPCS/code-switching migration, with no S-two comparison |

These numbers compare each paper against its own baselines, fields, machines, and proof targets.
They must not be presented as expected S-two speedups.

## Ideas to reject or defer

- **“Switch to DEEP-FRI.”** The current PCS already uses a DEEP/OODS quotient. Adding DEEP work to
  every round is primarily a soundness optimization.
- **Replace production LogUp with current GKR.** Local whole-proof measurements reject it.
- **Increase composition degree just to commit fewer columns.** The local uniform-`K` experiment
  made proving 9.5% slower; evaluate low-degree components on smaller domains first.
- **Replace Circle FFT with another FFT family.** No operation-count or end-to-end evidence supports
  it; focus on pass fusion and degree-aware transforms.
- **Lower blowup or queries using `beta * q`.** This is not a defensible non-interactive soundness
  calculation.
- **Adopt STIR/WHIR/Blaze numbers as S-two estimates.** Their domains, fields, arithmetizations,
  commitments, and benchmark targets differ.
- **Promise a Lifted-FRI speedup before HHM25 is public.** The principal documented benefits are
  uniformity and soundness; prover improvement is workload-dependent.

## Measurement plan and decision gates

### Baseline matrix

Build one production harness with Blake2s, SIMD, `parallel`, native CPU features, and a reviewed
security target. Cover:

- sizes `2^16`, `2^20`, and `2^24`;
- narrow, wide, and mask-heavy AIRs;
- homogeneous and heterogeneous trace heights;
- local degree excess `K_i` in `{1,2,3,4}`;
- one large proof and batches of small proofs; and
- production lookup/table distributions.

Record:

- trace generation;
- interpolation/LDE;
- each lifted Merkle tree;
- constraint evaluation and interpolation;
- OODS weight construction and dot products separately;
- DEEP numerator accumulation, denominator generation, interpolation, and extension;
- each FRI fold and commitment;
- grinding, query sampling, and decommitment;
- verification, proof size, and peak RSS.

The current benchmark suite is insufficient: `benches/fri.rs` measures only CPU `fold_line` at
`2^12`; `benches/merkle.rs` measures the legacy VCS; the existing Poseidon artifacts are stale or
use non-production defaults.

### Gates

- Proof-preserving work: require identical roots/proofs for fixed transcript seeds, CPU/SIMD parity,
  and at least 3% end-to-end improvement for an invasive redesign.
- Proof-format work: require at least 10% end-to-end improvement or a major proof-size/recursion
  benefit, with verifier and proof-size budgets stated in advance.
- Stop any optimization that wins only an isolated kernel but regresses the full proof. The recent
  GKR fused-kernel experiment did exactly that and was correctly reverted.
- Every FRI-visible change needs negative proofs for wrong fold order, wrong sibling mapping, wrong
  last-layer degree, altered transcript order, and malformed multi-height openings.

## Recommended implementation/research sequence

1. **Benchmark repair:** add active lifted-VCS and complete SIMD FRI benchmarks plus one end-to-end
   production-security proof trace.
2. **Low-risk algorithms:** OODS fast path or selective coefficient retention; row-major quotient
   accumulation; reuse denominator/domain data; materialize/reuse rotated masks.
3. **Composition algorithm:** per-component `K_i` evaluation followed by interpolation/lifting into
   the global composition.
4. **Existing FRI experiment:** sweep `fold_step = 1..4`, leaf packing, and final degree under a
   reviewed fixed-security model.
5. **Lifted VCS:** memoize only demonstrably repeated upper subtrees and add layer checkpointing.
6. **Mathematical prototype:** first-layer-elided FRI/direct folded LDE, starting with `k = 1` and a
   narrow AIR; then evaluate selective per-tree/log-size variants.
7. **Service-level algorithm:** batch independent proofs if production handles multiple instances.
8. **Long-term:** flat Circle sumcheck for lookup consistency; only then compare full Lifted FRI or
   a Circle-WHIR/linear-code migration.

## Soundness escalation

```text
SOUNDNESS-ESCALATION:
  File: crates/stwo/src/core/fri.rs,
        crates/stwo/src/core/pcs/,
        crates/stwo/src/prover/fri.rs,
        crates/stwo/src/prover/pcs/
  Change: Any FRI commitment-elision, Lifted-FRI, arity, batching, or parameter rewrite
  Invariant at risk: Circle FFT-space membership; dimension-gap handling; cross-domain
                     correlated agreement; fold/domain chain; OODS binding; non-interactive
                     round soundness; group-squaring-consistent query mapping
  Paper reference: Circle_STARKs prot:IOP:proximity;
                   Stwo_Whitepaper §§4.3–4.4, §5.4, §6
  Code location: core/fri.rs:77-88; prover/fri.rs:77-85, 110-195;
                 prover/pcs/mod.rs:177-316
  Confidence: 98%
  Reason: the current security_bits() estimate omits batching, fold, field/list,
          domain-size, and random-oracle round terms; the whitepaper explicitly notes
          that the current non-squaring-consistent query layout lacks a full LDR proof;
          HHM25 Lifted FRI is unpublished; DIVERGENCE-001 remains open.
```

No soundness-critical implementation should begin until the target production configuration,
security regime, and exact paper reduction are fixed. A PCS-family migration additionally requires
human approval under the repository's soundness workflow.
