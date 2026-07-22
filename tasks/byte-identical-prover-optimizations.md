# Faster proving with byte-identical proofs

Date: 2026-07-22

## Session amendments (2026-07-22, reviewed)

1. **Canonicity audit is a prerequisite, not a caveat**: before any kernel-class
   change (item 8 and parts of 2/4), determine which committed buffers can contain
   non-canonical M31 words today (known instance: the FFT redundant-[0,P] TODO in
   `fft/mod.rs`), then either prove they cannot or pin the representative-producing
   behavior into the parity contract. Insert as step 1.5 of the execution order.
2. **Classification relaxation**: byte identity constrains only channel-visible
   words (committed columns, sampled values, FRI layer data, nonce, query
   positions). Internal scratch is unconstrained, and exact field arithmetic makes
   every reordering/chunking/parallelization of sums value-identical automatically.
   This reclassifies several "conditional" entries as trivially exact.
3. **Ordering**: item 3 (derive LogUp inputs from committed evals) moves below 4
   and 5 on effort-adjusted return. Experiment zero is the free
   `store_polynomials_coefficients` flag flip + byte-compare (item 2's cheapest
   datapoint).
4. **EU-DI scope note**: the byte-identity constraint applies to stwo mainline.
   The eu-id product owns both prover and verifier pre-launch — protocol-visible
   improvements are in scope THERE (see `campaign-path.md`), so do not let this
   doc's exclusions leak into eu-id planning.

## Executive answer

Requiring the serialized proof to be byte-for-byte identical removes most protocol-level
optimizations from the global roadmap. Lifted FRI, different FRI folding, mixed-height
commitments, virtual committed columns, new lookup arguments, different security parameters, and
different query or PoW rules all change the proof.

There is still a substantial algorithmic design space. The governing rule is:

> Change how an existing committed polynomial, opening, quotient, tree, or nonce is computed, but
> reproduce the exact object, in the exact existing order, before it becomes transcript-visible.

The highest-upside mathematical candidate is to evaluate each component quotient at its native
degree, then extend it and apply the existing projection lift so that the current global composition
polynomial is reconstructed exactly. This is not yet production-safe: Circle polynomial spaces have
an extra `v_n` direction, and the repository's corresponding dimension-gap invariant is unresolved.
It must start as a parity experiment, not as an assumed identity.

The strongest lower-risk candidates are exact native-domain or coefficient-form OODS evaluation,
removing challenge-delayed lookup-data copies, a fused tiled DEEP quotient, exact analytic generation
and caching of fixed codewords and trees, and implementation-level FFT, Merkle, and constraint-eval
improvements. These preserve every protocol object and therefore can preserve proof bytes.

## 1. What “byte-identical” means

The comparison must fix all of the following:

- statement and witness;
- prover configuration, PCS/FRI parameters, hash, features, and channel implementation;
- baseline backend and its deterministic policies;
- column, sample, query, and decommitment ordering; and
- serialized proof type and serializer version.

For that fixed environment, the complete serialized `ExtendedStarkProof` should be equal as a byte
vector. Equality of proof size, successful verification, or equality of only the Merkle roots is not
enough.

The PCS proof currently serializes the configuration, commitments, sampled values, decommitments,
queried values, proof-of-work nonce, and FRI proof. Its auxiliary proof data also contains query and
decommitment state. Consequently byte identity requires all of this chain to remain equal:

```text
committed column values and layout
    -> every Merkle node and root
    -> transcript state and challenges
    -> sampled values and DEEP quotient
    -> every FRI layer and last polynomial
    -> PoW nonce
    -> query positions and their order
    -> queried values and decommitment witnesses
    -> serialized bytes
```

Field computations are exact rather than floating-point, so algebraic reassociation preserves the
abstract field element. That is not always a sufficient byte-identity argument in this codebase:
packed paths can temporarily carry non-canonical raw M31 words, and hashing/channel code can consume
raw limbs. The strongest boundary invariant is therefore raw-word equality, or an explicitly proved
canonicalization boundary, rather than only field equality. An alternative may not reassign
challenge powers, reorder columns or samples, select a different valid nonce, or serialize logically
equivalent data in a different order.

### Backend scope and PoW caveat

Byte identity should initially be promised relative to one named baseline backend. Current grinding
implementations do not obviously expose one cross-backend nonce-selection contract: the CPU path
searches globally from zero, while the SIMD Blake path partitions a low-bit search region. A faster
grinder is proof-identical only if it returns the exact nonce the baseline would return, not merely any
valid nonce. The safest contract is the minimum valid nonce in the baseline search order, followed by
a separate CPU/SIMD parity project if cross-backend identity is desired.

## 2. Classification rule

There are three useful classes.

| Class | Rule | Examples |
|---|---|---|
| Identical by construction | The old transcript-visible vector/object is reproduced exactly | OODS evaluator substitution, fixed-tree cache, tiled DEEP, deterministic parallel FFT/hash |
| Conditionally identical | An algebraic identity should recreate the old object but needs a Circle-space/order proof | native `K_i` quotient evaluation then exact reconstruction, analytic codeword generation, pruned transforms |
| Protocol-changing | A committed object, transcript message, parameter, query, or serialization field changes | Lifted FRI, mixed-height roots, virtual columns, new lookup protocol, changed blowup/queries |

This distinction is stricter than “same statement and security.” Two valid proofs of the same
statement are normally not byte-identical.

## 3. Ranked byte-identical roadmap

The ranking separates confidence from upside. Whole-prover impact must be measured; the table does
not turn kernel or memory ratios into unsupported end-to-end speedup claims.

| Priority | Candidate | Main deleted work | Identity status | Main risk |
|---:|---|---|---|---|
| 1 | Native `K_i`, then reconstruct today's global composition | surplus component quotient evaluation and extension at global `K` | conditional, highest mathematical upside | Circle `v_n`/`lambda` direction and invalid-trace behavior |
| 2 | Native-domain or coefficient-form OODS | full committed-domain barycentric weights | exact if every sampled value matches | projected point and bit-reversed-domain formula |
| 3 | Derive LogUp inputs from committed trace | challenge-delayed duplicate lookup columns | exact | lifetime/layout refactor and trace/recipe mapping |
| 4 | Fuse and tile the DEEP quotient | `S x N` denominator and numerator scratch objects; repeated traffic | exact | batching/order mistakes and batch-inversion granularity |
| 5 | Reuse immutable coefficients, LDEs, and Merkle trees | repeated fixed preprocessing across proofs | exact | complete cache key and retained opening data |
| 6 | Generate exact periodic/fixed LDEs analytically | interpolate plus generic LDE for structured columns | conditional until vector parity is proved | phase, projection, normalization, and Circle FFT space |
| 7 | Cache shared constraint extensions and common masks | repeated LDEs, rotations, and expression evaluation | exact | cache identity and memory tradeoff |
| 8 | Exact FFT/Merkle/constraint kernel redesign | passes, materialization, hashes, and memory traffic | exact | backend parity and stable ordering |
| 9 | Exact multipoint OODS algorithms | repeated point-evaluation setup across masks | conditional; workload-dependent | Circle-domain derivation complexity |
| 10 | Deterministic exact PoW parallelization | wasted search or scheduling | exact only under baseline nonce rule | returning a different valid nonce |
| 11 | Fuse the already-uncommitted folds inside one configured FRI step | intermediate arrays and passes, not FRI layers | conditional until raw next-layer parity | accidentally crossing a transcript root boundary |
| 12 | Hash repeated lifted subtrees once | duplicate hashes caused by lifted/identical blocks | exact | preserving every authentication path and leaf convention |

The first benchmark sequence should be 2, 3, 4, and 5 because they have simple proof-identity
oracles and do not depend on closing an unresolved Circle-space theorem. Priority 1 deserves a
research prototype in parallel because it is the most global mathematical simplification.

## 4. The main “smart math” candidate: native quotient computation, old composition

Let component `i` have trace log size `n_i`, native quotient expansion `K_i`, and let today's global
expansion be `K = max_i K_i`. The current prover evaluates every component quotient on a domain of
log size `n_i + K` before lifting and accumulating it into the global composition.

The desired internal substitution is:

```text
old:
    evaluate q_i directly on D_(n_i + K)
    -> existing cross-height projection lift
    -> existing global composition Q

candidate:
    evaluate q_i on D_(n_i + K_i)
    -> interpolate in the exact native Circle polynomial space
    -> LDE to D_(n_i + K)
    -> same existing cross-height projection lift
    -> exact same Q
```

Ignoring the added interpolation/LDE work, the pointwise constraint-evaluation term can move from
roughly `sum_i C_i * 2^(n_i + K)` to `sum_i C_i * 2^(n_i + K_i)`. This is measured motivation, not
a speedup estimate: in an existing downstream experiment, moving to uniform global `K` increased
prove time from 4,911 ms to 5,378 ms (+9.5%) and added 712 ms to its STARK phase while changing
other proof geometry too. A native experiment must measure whether the saved rows outweigh its
extra transforms.

If the two paths produce the same `q_i` vector on `D_(n_i + K)`, everything after it can remain
unchanged: the random-coefficient association, global composition coefficients, recursive split
parts, Merkle root, OODS samples, DEEP quotient, FRI proof, and final bytes.

This is importantly different from committing native mixed-height composition parts. The latter
changes the commitment and proof; the former uses native work only as a hidden implementation and
reconstructs the old commitment.

### Why it is conditional in Circle STARKs

The necessary identity is stronger than an informal degree bound:

```text
Eval_(n_i + K)(q_i)
    = LDE_(n_i + K_i -> n_i + K)(Eval_(n_i + K_i)(q_i)).
```

It requires `q_i` to belong to the exact native Circle FFT space used by the implementation. The
Circle space contains an additional vanishing-polynomial direction, often written
`L_N = L'_N + <v_n>`. A total-degree claim alone does not prove membership in the expected
codimension-one subspace. The repository's divergence log already records this dimension gap.

There are two further subtleties:

1. Native-domain LDE and the existing cross-height projection lift are different maps; combining
   their doubling exponents accidentally would change the OODS identity.
2. On an invalid trace, a constraint numerator may have a remainder modulo the trace vanishing
   polynomial. Interpolation of the resulting pointwise quotient on two domains need not commute.
   A production optimization must preserve rejection and fail closed even if unsuccessful proving
   paths do not have meaningful proof bytes to compare.

Required research outcome: either prove the alternating-limit/parity invariant for every component,
handle the `lambda` direction explicitly, or keep the larger-domain path as a checked fallback. The
first prototype should compute both paths and assert vector equality before using the native result.

An even more aggressive version accumulates directly in coefficient space and emits the same global
composition split parts without constructing the full global evaluation first. It is also potentially
proof-identical, but only after proving that this linear map is exactly the current
`lift_and_accumulate` plus interpolation/split map.

## 5. Exact OODS evaluation from less data

OODS samples are transcript-visible field elements, but the method used to calculate them is not.
For a polynomial stored at native height `h` and lifted to global height `L`, the required value is

```text
p_lifted(z) = p(pi^(L-h)(z)),
```

where `pi` is Circle doubling. The current full-domain barycentric calculation may be replaced by:

- barycentric evaluation on the smallest complete native subdomain;
- coefficient-form folding/Horner evaluation when coefficients are retained;
- an exact analytic evaluator for a structured fixed polynomial; or
- direct inverse-vanishing weights with one batch inversion.

For the most direct current-code experiment, split the committed evaluation domain by the configured
blowup, select the same complete native non-canonical subdomain and its bit-reversed prefix, and form
weights and the dot product only over those `N` values rather than all `bN` committed values. This
reduces one secure-field weight vector from roughly `16*bN` bytes to `16*N` bytes and reduces the
corresponding input scan by the blowup. The folded sample point and raw output limbs must equal the
current path.

A previous local `2^20` phase artifact measured 441.348 ms to construct the SIMD weight vector,
versus 0.982 ms for its weighted dot product and 1.387 ms for coefficient-form evaluation. This is
a phase microbenchmark, not a whole-proof speedup, but it strongly identifies setup rather than the
dot product as the target.

The algebraic AIR identity can independently reconstruct the total composition OODS value. It
cannot replace the individual OODS samples of today's committed composition split parts: those
samples enter the current DEEP quotient and transcript separately. They must still be computed and
serialized exactly.

## 6. Remove challenge-delayed lookup copies

Several example trace generators keep full-width copies of values only because lookup denominators
are formed after the verifier challenge arrives. The committed trace evaluations already contain the
same data. A prover can instead retain a compact recipe—column indices, rotations, and expression
metadata—and derive each denominator from the committed trace's trace-size subdomain.

This removes transient objects, not proof objects. If the derived fractions and accumulated
interaction columns are element-for-element equal, the interaction roots and all later bytes remain
equal.

The source audit found large opportunities in the Blake scheduler and round components and in
Poseidon state retention. The rough retained-data counts were about `1,056*N`, `480*N`, and `256*N`
base field elements respectively. These are memory-object counts, not guaranteed RSS reductions:
allocator behavior, simultaneous lifetimes, and recomputation traffic must be measured.

The design choice is workload-specific:

- read the exact values from a retained trace-size view if it already exists;
- retain compact coefficients/recipes if committed storage is only available as an LDE;
- avoid reconstructing values with expensive logic that costs more than the deleted copy; and
- preserve fraction batching, challenge powers, column order, and zero-denominator behavior.

For the first parity implementation, preserve the same packed operation DAG as well as the same
abstract fractions. An algebraically equivalent rearrangement can leave a different non-canonical
representation of zero before the native interaction columns are normalized or hashed.

## 7. Fuse and tile DEEP construction

The SIMD DEEP path currently materializes multiple large intermediates, including per-sample
numerator data and an `S x N` denominator-inverse matrix for `S` samples and quotient domain size
`N`. A tiled exact algorithm can:

1. load or lift a source tile;
2. form its sample numerators in the existing sample/challenge order;
3. batch-invert denominators inside the tile;
4. combine them into the same quotient values; and
5. emit the final quotient tile directly.

The scratch target changes from `Theta(SN)` to roughly `Theta(N + tile*S)`. Tiling is worthwhile only
if extra inversion boundaries and source rereads do not erase the bandwidth benefit. Because the
final quotient vector is the equality oracle, this can be developed without modifying FRI at all.

The safest first implementation preserves the current numerator chunking, denominator
batch-inversion boundaries, left-associative accumulation, and tree/column/sample traversal while
only shortening live ranges. A different association is mathematically sound but should not be
called byte-identical until raw-word parity at the final quotient vector is demonstrated. Changing
which random coefficient is assigned to a tree/column/sample is never allowed. Bit-reversed index
order must also remain unchanged.

## 8. Exact fixed preprocessing

The prover can borrow a prebuilt commitment tree. For repeated proofs of the same program and
configuration, cache all data required to open the existing fixed commitment:

- native coefficients where useful;
- the exact committed LDE vectors;
- leaf layout and full Merkle prover data; and
- metadata sufficient to validate height, hash, column order, and configuration.

The root alone is insufficient because later queries require leaf values and authentication paths.
Cache keys must include every choice that affects bytes, including hash type, blowup/lifting size,
column order, domain conventions, and code version or a content digest.

Structured periodic columns offer an additional exact algorithm. A period-`2^r` polynomial can
often be evaluated on the large domain through the appropriate repeated Circle projection and phase
instead of generic interpolation plus LDE. It must generate the same polynomial, not merely a
Boolean function that agrees on trace rows. Selector normalization, phase offsets, padding, and
bit-reversed leaf order are all part of the identity proof.

## 9. Other implementation algorithms that remain available

Proof identity does not require instruction-by-instruction identity. The following remain valid when
their output vectors match:

- mixed-radix, pass-fused, truncated, or pruned Circle FFTs;
- cached twiddles and shared LDEs for the same `(polynomial, domain)` pair;
- row-major constraint evaluation, common-subexpression elimination, and precomputed rotated masks;
- SIMD/GPU field arithmetic and exact hashing;
- streaming Merkle construction, checkpoint/recompute strategies, and deterministic parallel trees;
- deterministic parallel query materialization with a final restoration of existing order; and
- batch evaluation at multiple OODS points through an exact Circle multipoint algorithm.

Several deserve explicit experiments:

- **Sparse selector evaluation.** If a selector is provably zero off a known support, evaluate an
  expensive gated constraint only on that support and write the same raw zero elsewhere. Preserve
  constraint traversal and random-coefficient exponents. This is especially attractive for boundary
  or rare-row constraints.
- **Exact lookup witness generation.** Block/Blelloch prefix scans, grand-product block summaries,
  deterministic radix/counting sorts, histogram multiplicities, and batch inversion may reproduce
  the existing interaction witness faster. Sorting must retain the exact current tie-break order,
  and the interaction columns need raw parity.
- **Fused configured FRI folds.** If `fold_step = k`, the `k` linear folds performed between two
  existing commitment boundaries may be composed into one tiled `2^k`-input transform that emits
  the exact current next layer. No fold can move across an intervening root because the next
  challenge depends on that root.
- **Repeated lifted subtrees.** When lifted columns create identical leaf blocks or child pairs,
  compute each distinct subtree once and reuse the resulting hash. Complete tree values and
  authentication-path ordering remain unchanged. This wins only when tall columns do not destroy
  the repetition.

Pruned transforms save time only when the caller genuinely needs a strict subset of an unchanged
output. If the old Merkle tree commits every output, no output can be pruned. Likewise, streaming or
recomputation may reduce peak memory without improving latency; measure both.

## 10. What is excluded

The following ideas from the broader global roadmap are not byte-identical to today's proof format:

- Lifted FRI or any different FRI fold schedule, arity, last-layer degree, or leaf packing;
- first-layer/quotient commitment elision or direct-folded openings;
- native mixed-height composition commitments or a different number/order of split parts;
- modular split-and-pack or trace transposition into new committed columns;
- deleting or virtualizing a currently committed selector, table, witness, or derived column;
- event-driven extraction into new components;
- adaptive LogUp fraction batching if it changes interaction columns;
- LogUp-GKR, flat sumcheck, Lasso/TaSSLE, structured-lookup replacements, or hybrid sumcheck islands;
- Appendix-C nested `H subset D` domains;
- STARKPack or other multi-instance aggregation;
- Binius, ring/tower switching, code switching, WHIR, STIR, BaseFold, or a PCS migration;
- changing blowup, number of queries, grinding bits, challenge count, or any security parameter;
- changing transcript mix/draw order, query sampling/deduplication/order, or PoW nonce rule; and
- changing proof fields, serialization order, serializer, or version.

Some of these can inspire hidden internal algorithms. For example, a native representation is fine
if it is converted back into every old oracle and opening before commitment. It cannot retain its
main protocol-level advantage and still produce the old bytes.

In particular, Lifted FRI itself cannot be byte-identical: its advantage comes from changing which
objects are folded/committed and how. Using a lifted representation internally to reconstruct the
old FRI inputs exactly is allowed, but then the proof is still the old FRI proof and most Lifted-FRI
protocol savings disappear.

## 11. Verification gates

### Final byte gate

For fixed deterministic fixtures, serialize the complete baseline and candidate
`ExtendedStarkProof` with the same serializer and assert:

```text
baseline_bytes == candidate_bytes
sha256(baseline_bytes) == recorded_golden_digest
```

The digest makes accidental fixture or serializer changes visible; the byte-vector assertion gives
the useful failure diff.

### Layered localization gate

When equality fails, compare in this order:

1. preprocessing, base-trace, interaction, and composition evaluation vectors;
2. every Merkle layer and root;
3. random-combination coefficient assignment and sampled-value vector;
4. complete DEEP quotient vector;
5. every FRI layer commitment, last polynomial, and auxiliary data;
6. transcript digest after each mix/draw boundary;
7. proof-of-work nonce;
8. unsorted and serialized query locations;
9. queried values and decommitment witnesses; and
10. final proof object and byte vector.

At field-vector boundaries, compare the raw M31 coordinate words in addition to semantic field
equality. Compare the ordinary wire `StarkProof` unconditionally; compare serialized auxiliary data
as well whenever `ExtendedStarkProof` is the persisted artifact.

This turns “proof differs” into a precise first-divergence report.

### Test matrix

- CPU and SIMD implementations where both exist;
- serial and parallel builds under the chosen baseline policy;
- minimum, boundary, and large domain sizes;
- homogeneous and heterogeneous component heights;
- `K_i = K` and `K_i < K` components;
- both constraint evaluation modes;
- all mask rotations and repeated OODS points;
- empty/singleton/maximum-width trees and fixed-column cache hits/misses;
- randomized valid witnesses and deliberately invalid witnesses;
- `v_n`/`lambda`-direction adversarial polynomials for the native-`K_i` experiment; and
- zero/pole precondition and error-path tests for alternative inversion algorithms.

For soundness-critical math, proof verification is necessary but weaker than the intermediate and
byte equality assertions.

### Performance gate

Record wall time, per-phase time, peak RSS, bytes allocated or retained when available, proof size,
and verifier time. Proof size and verifier time should remain exactly equal. An invasive rewrite
should normally clear a predeclared whole-prover threshold (the global study used 3% for
proof-identical work) or deliver a separately valued peak-memory improvement.

## 12. Recommended execution order

1. Add the full-proof byte-parity harness and first-divergence diagnostics.
2. Remove one large challenge-delayed lookup copy and measure memory/time.
3. Implement native/coefficient OODS evaluation behind dual-path equality assertions.
4. Tile DEEP construction and tune tile size without touching FRI.
5. Exercise the existing prebuilt-tree interface in a repeated-program workload.
6. Prototype exact analytic generation for one simple periodic fixed column.
7. Run the native-`K_i` composition experiment with explicit `v_n` adversarial tests; do not ship it
   until the Circle-space invariant is proved or exact equality is enforced with a safe fallback.
8. Only then invest in pruned FFTs, coefficient-space composition, or multipoint Circle evaluation.

This sequence establishes byte identity as an executable invariant before attempting the most subtle
mathematical optimization.

## 13. Source anchors and literature boundary

Current implementation anchors:

- [`CommitmentSchemeProof`](../crates/stwo/src/core/pcs/quotients.rs) defines the transcript-visible
  PCS proof fields.
- [`CommitmentSchemeProver`](../crates/stwo/src/prover/pcs/mod.rs) orders commitments, samples,
  DEEP/FRI work, grinding, queries, and decommitments, and exposes `commit_tree` for borrowed fixed
  trees.
- [`component_prover.rs`](../crates/constraint-framework/src/prover/component_prover.rs) contains the
  current component quotient-domain evaluation modes.
- [`simd/quotients.rs`](../crates/stwo/src/prover/backend/simd/quotients.rs) contains the current SIMD
  DEEP quotient construction and denominator inversion.
- [`global-prover-algorithm-research.md`](global-prover-algorithm-research.md) is the broader roadmap;
  this document is its strict proof-identity filter.

The Circle-space caution and the distinction between native LDE and projection lifting are grounded
in the [Circle STARKs paper](https://eprint.iacr.org/2024/278) and the repository's
paper-implementation divergence log. Other protocol papers cited by the broader study remain useful
for a non-identical future proof format, but cannot justify claiming byte identity with today's one.
