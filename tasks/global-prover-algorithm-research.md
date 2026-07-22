# Global prover-algorithm research: transformations beyond Lifted FRI

Date: 2026-07-22

## Corrected objective

Lifted FRI is one example of the desired pattern: use algebra to remove an entire class of
prover work. It is not the objective. The objective is to redesign the whole proving algorithm so
that it commits, extrapolates, hashes, and checks fewer mathematical objects.

The central conclusion of this review is that S-two's largest plausible algorithmic gains are above
FRI. The most promising program is a proof compiler that jointly transforms trace layout,
componentization, derived/public columns, LogUp arity, and quotient degree before it tunes the PCS.

No proof-system code or benchmarks were changed for this review. Published measurements below are
the cited papers' measurements on their own systems; local measurements are explicitly labelled.

## A cost model for deciding what is actually global

For component `i`, let:

- `N_i` be its rows;
- `W_i` be its committed trace and interaction width in base-field coordinates;
- `A_i` be its active/event rows when the component is sparse;
- `b` be the LDE blowup;
- `K_i` be its logarithmic composition-degree excess/split;
- `F_i` be the number of LogUp fractions per row; and
- `S_i` be the number of distinct OOD mask points.

A useful first-order model is

```text
trace LDE/commit       ~ sum_i b N_i W_i log(b N_i) + Merkle(b N_i, W_i)
constraint quotient   ~ sum_i N_i 2^K_i C_i
interaction LDE       ~ sum_i 4 b N_i ceil(F_i / batch_i)
composition commit    ~ current: 4 b 2^Kmax Nmax
DEEP/OODS              ~ sum_i b N_i S_i plus lifted accumulation
FRI                    ~ geometric work starting at the largest opened domain
```

The most valuable transformations therefore reduce `N_i`, `W_i`, `2^K_i`, `F_i`, or the number
of full-size committed objects. Improving one butterfly or hash compression is useful, but it is
not in the same class.

## Ranked research agenda

| Rank | Transformation | Whole objects deleted | S-two fit | Main price |
|---:|---|---|---|---|
| 1 | Modular split-and-pack / trace transposition | FFT depth, Merkle internal nodes, FRI-domain size, peak per-polynomial memory | High | More opened values, larger proof, slower verifier |
| 2 | Degree-local constraints plus adaptive LogUp and mixed-height composition | Unrelated `2^Kmax` constraint work and many interaction/composition columns | Very high | Soundness-critical heterogeneous composition |
| 3 | Analytic and virtual columns | Preprocessed/selector/derived-column LDEs, commitments, OOD samples and openings | High | Compiler substitution and degree tradeoffs |
| 4 | Event-driven heterogeneous tables | Inactive rows, padding, selectors and gated constraints | High | Dispatcher/link lookups |
| 5 | Nested non-ZK evaluation domain `H subset D` | `1/b` of trace-LDE output work | Medium, conditional | Direct trace leakage; entirely new Circle-domain/quotient path; no benchmark |
| 6 | Structured virtual instruction/table lookups | Wide opcode AIR blocks and materialized giant tables | Medium-high | Lookup/sumcheck adapter and workload dependence |
| 7 | Directly folded quotient / first-layer-elided opening | Full quotient LDE/tree and early FRI layers | Medium | Query amplification over trace trees |
| 8 | Multi-instance packing | Repeated FRI, Merkle authentication and fixed tables across jobs | High for throughput | Does not accelerate witness generation; batching latency |
| 9 | Delete proof-identical transient objects | Duplicate lookup data, full DEEP scratch matrices, repeated preprocessing | Very high | Engineering and memory-layout work, not a new proof |
| 10 | Hybrid sumcheck islands | Quotient FFTs and degree inflation for selected blocks | Medium-low | New Circle/MLE bridge; local GKR currently loses |
| 11 | Bit-width-preserving subproofs | One M31 element per bit/byte | Low-medium | Mixed-field bridge; effectively a second backend |
| 12 | Code switching / constraint-carrying PCS | Circle/RS encoding or a separate quotient layer | Low | New PCS and security proof |
| 13 | Reviewed joint security-parameter optimization | A factor of two in most domains per removed blowup bit | Potentially high | Cannot use the current heuristic as a proof |

## 1. Modular split-and-pack: transpose time into lanes

This is the clearest newly identified example of the kind of mathematical trick requested.

For `k = 2^s`, rewrite one length-`N` column as `k` lane columns of length `N/k`:

```text
lane_j[t] = original[k*t + j].
```

An adjacent-row constraint becomes:

- `lane_j[t] -> lane_(j+1)[t]` for `j < k-1`; and
- `lane_(k-1)[t] -> lane_0[t+1]` at the lane boundary.

More generally, an original offset `Delta` maps lane/row `(s,t)` to
`((s+Delta) mod k, t + floor((s+Delta)/k))`. This gives a mechanical compiler rule for every
transition mask, not only the next-row case.

Over the Circle group this has a subgroup/coset interpretation: every lane advances by `k` times
the original trace generator. The implementation does not need to preserve the original column
polynomial, however. It can re-arithmetize the `k` lane vectors as independent columns on one new
canonical domain of size `N/k`; the AIR-equivalence proof is carried by the intra-lane and
cross-lane transitions above. All lanes can then be packed at the same reduced-domain point in one
Merkle leaf and combined into one opening proof.

The total number of witness cells is approximately unchanged, but the transform cost changes from

```text
W * N * log N  ->  k * W * (N/k) * log(N/k) = W * N * (log N - log k).
```

The FRI starting domain and the number of Merkle leaves also shrink by `k`; leaves become wider.
The construction in [On amortization techniques for FRI-based SNARKs](https://eprint.iacr.org/2024/661)
proves the optimal adjacent-row constraint translation and reports, on a `2^16 x 275` Winterfell
trace with neither copies nor lookups, about 20% less prover time for `k=4` and about 40% for
`k=32`. The corresponding costs were about 3x/11x proof size and 2.2x/6x verifier time. Those are
not S-two estimates, but they justify an S-two prototype.

Why this is more global than a FRI change: the same row permutation shortens trace interpolation,
LDEs, constraint domains, interaction columns, composition domains, commitments, and FRI at once.
That final FRI reduction occurs only if every active oracle on the maximal-height frontier is
transposed or isolated into a separate packed proof. One remaining height-`N` tree keeps the global
opening height at `N` and erases much of the benefit.

LogUp is unusually compatible with the transformation. Its global relation is a multiset sum, so a
row permutation does not change it when all fields of each lookup tuple move together. Each lane can
produce a partial running-sum claim and the lane sums can be aggregated with post-commit randomness.
Every original use/yield must land in exactly one lane. Copy/permutation constraints that depend on
row identity, in contrast, need an explicit translated index map.

Important limitations:

- boundary, periodic, copy and lookup constraints need a Circle-specific translation;
- wide leaves may make leaf hashing closer to byte-linear than the paper's simple hash-count model;
- `k` should remain small without a recursive wrapper; and
- each query opens roughly `k` times as many lane values, so field payload and verifier work grow;
- the multiplicative-RS soundness proof in ePrint 2024/661 does not automatically prove correlated
  agreement for the current Circle/non-squaring query layout; and
- the AIR compiler must bind the public row-to-lane permutation and re-interpolate every lane on a
  fresh canonical reduced domain with the translated offset map above.

First experiment: `k in {2,4,8}` on a wide, lookup-heavy production-shaped AIR, recording trace
FFT, interaction generation/LDE, constraint quotient, every Merkle tree, DEEP, FRI, proof size and
verification separately.

## 2. Make degree and LogUp arity local, then commit a mixed-height composition

The framework already supports `finalize_logup_batched(batch_size)` in
[`constraint-framework/src/lib.rs`](../crates/constraint-framework/src/lib.rs). Batching `F`
fractions `a_j / d_j` into groups of `r` replaces approximately `F` secure running-sum columns by
`ceil(F/r)`, while the group denominator becomes a product of `r` denominators and increases the
constraint degree.

This is currently a bad global bargain when one high-degree component raises the single global
`K`: every unrelated component is evaluated on `2^(K-K_i)` too many points. A local downstream
batch-4 experiment already demonstrated the tension: proof size fell 17.8%, but proving time rose
9.5% because the uniform `K` doubled every component's constraint domain.

The combined algorithm should be:

1. choose a LogUp batch size `r_i` per component;
2. evaluate component `i` only on `2^(Nlog_i + K_i)` points;
3. interpolate and lift its quotient with the verifier-consistent Circle doubling map;
4. group contributions by native `(height, K_i)`;
5. split each group only `K_i` times; and
6. commit all resulting composition parts in the existing mixed-height lifted VCS.

The current implementation deliberately uses one global split in
[`prover/air/accumulation.rs`](../crates/stwo/src/prover/air/accumulation.rs) and creates
`4 * 2^K` base columns of maximal height in
[`prover/mod.rs`](../crates/stwo/src/prover/mod.rs). A proof-preserving first stage can save only
constraint evaluation by evaluating locally and FFT-extending into the current global accumulator.
The full protocol-visible stage changes composition commitment cost from roughly

```text
4 b 2^Kmax Nmax  ->  4 b sum_g 2^Kg Ng.
```

This is especially attractive when a short lookup component has high degree and a long execution
component does not.

A compiler can select `r_i` by minimizing

```text
interaction cost: 4 b N_i ceil(F_i / r_i)
+ native constraint cost: N_i 2^K_i(r_i) C_i
+ mixed-height composition cost: 4 b N_i 2^K_i(r_i).
```

The same optimizer should use the degree/width duality in the other direction. A high-degree
expression can be factored through auxiliary witness columns, for example `u=x*y`, followed by
lower-degree constraints involving `u`. The extra column costs one trace LDE and commitment, but a
one-bit decrease in `K_i` halves the affected constraint and composition domains. Introduce the
auxiliary exactly when

```text
extra column LDE + Merkle + opening
    < saved quotient/composition work from lowering K_i.
```

This decision cannot be made from algebraic degree alone; it depends on component height, number of
constraints sharing the intermediate, and whether the new column can itself remain virtual.

This proposal must preserve the lifting identity repaired by the existing generalized composition
split. A naive return to per-component bounds recreates the old OODS mismatch; the two-stage lift is
not optional. The generalized construction must also resolve the one-dimensional Circle FFT-space
gap `L_N = L'_N + <v_n>`: every native split part needs a proved FFT-space/parity bound, and any
required scalar `lambda` term must be carried rather than silently dropped.

## 3. Stop committing polynomials the verifier can derive

There are three distinct cases.

### 3.1 Public periodic selectors

The [Circle STARK paper](https://eprint.iacr.org/2024/278) constructs periodic/subdomain selector
polynomials from vanishing-polynomial quotients, proves that they remain in the Circle FFT space,
and states that they are succinctly evaluable, so they need not be provided in a preprocessing
commitment.

S-two still models preprocessed columns as ordinary committed columns, and examples contain TODOs
to remove mandatory preprocessed columns. The correct global change is an `AnalyticColumn` class:

- generate its domain value from a compact formula during constraint evaluation;
- evaluate it directly at the OOD point and FRI query locations;
- omit its interpolation, LDE, Merkle leaf data, OOD sample and decommitment.

Boundary selectors, `is_first`/`is_last`, periodic round constants, and punctuated activation
selectors should be the first targets.

### 3.2 Public structured tables

Blake's XOR tables are preprocessed committed columns. For a decomposable table, do not construct
or commit the whole table: compute the small subtables or the table polynomial at the requested
point. [Lasso](https://eprint.iacr.org/2023/1216) introduced the structured-table technique, and
[TaSSLE](https://eprint.iacr.org/2024/1075) shows it can be combined generically with logarithmic-
derivative lookup arguments without committing or constructing the full table.

The low-risk S-two version is analytic fixed tables that retain prefix-sum LogUp. The fuller TaSSLE
version uses GKR/sumcheck and should wait for a workload where table commitment is genuinely
dominant.

### 3.3 Derived witness columns

If `z = a*x + b*y + c`, commit only `x,y`; the verifier can derive every OOD/query evaluation of
`z`. Linear/affine virtual columns do not raise AIR degree. Nonlinear substitution may raise `K`, so
the compiler should choose between committing `z` and inlining it using the same degree/width cost
model as section 2.

This is the transferable insight behind the virtual polynomials in
[Twist and Shout](https://eprint.iacr.org/2025/105), where eliminating derived committed
polynomials gives a reported 4x reduction in commitment cost in SpeedySpartan. That number is not
portable to S-two; the useful idea is the committed-versus-virtual compiler decision.

## 4. Make cost proportional to events, not to the longest table

If an operation is active `A_i` times inside an `N`-row global table, extract it into a dense
`A_i`-row component and link the dispatcher to it with a lookup/permutation relation:

```text
before: Theta(b N W_i)
after:  Theta(b A_i W_i) + link_cost.
```

This removes inactive/padding rows, opcode selector columns, selector multiplications, their
quotient evaluations, LDE values and Merkle payloads. S-two already supports components and
heterogeneous heights, making this much more native than it would be in a monolithic STARK.

The philosophy matches [Ceno](https://eprint.iacr.org/2024/387): non-uniform segments make proving
cost track what was actually executed. Ceno uses a different GKR backend; S-two should borrow the
front-end transformation, not its benchmark claims.

Best candidates are rare instructions, syscalls, builtins, cryptographic accelerators, and
variable-frequency range checks. The win condition is simply that deleted inactive cells and
constraints exceed the dispatcher/link interaction columns.

This combines well with modular split: first extract dense event tables; then transpose only long
tables whose FFT/hash depth remains material.

## 5. Use a nested evaluation domain when zero knowledge is not required

Current S-two interpolates on one canonical Circle coset and evaluates on a larger canonical coset;
canonical cosets at different sizes are disjoint. The Circle STARK paper's Appendix C gives a
different, valid non-ZK construction with `H subset D`, specifically to minimize extrapolation
further. It is a separate Circle-STARK variant, not a configuration switch for the present prover.

If trace evaluations are already present on the `N` points of `H`, a nested-domain encoder outputs
only the new points in `D - H`, rather than a whole disjoint `bN`-point codeword. This deletes output
values, not automatically the same amount of arithmetic: the values on `D-H` still need encoding,
and the quotient is undefined pointwise on `H`. The paper replaces the normal quotient evaluation
with a mixed decomposition. There is no published S-two benchmark, so the idea must not be assigned
an expected whole-prover gain from point counts alone. Precisely, it saves `N` of `bN` trace-LDE
output evaluations per column, or a `1/b` share of that output work. At `b=2` this can roughly halve
the extension-output stage if a partial Circle FFT realizes the point saving, while interpolation,
the full Merkle tree, composition evaluation and FRI remain.

This is a genuine Circle-native mathematical optimization, not a kernel tweak. It requires:

- a group-position evaluation domain and the mixed standard/rotated FFT-space decomposition from
  the paper;
- revised quotient decomposition and query mapping;
- a proof that the implementation's FFT space and dimension-gap invariant match Appendix C; and
- an explicit product decision that direct openings of trace-domain values are acceptable.

It must never be enabled in a zero-knowledge configuration. The current core contains no trace
blinding path found by this review, but deployment/privacy requirements must be checked separately.

## 6. Replace instruction algebra by structured virtual lookups

[Jolt](https://eprint.iacr.org/2023/1217) replaces most instruction semantics with lookups into a
gigantic ISA table. [Lasso](https://eprint.iacr.org/2023/1216) makes that possible because a table of
size `2^w` can be decomposed into `s` much smaller tables and never materialized in full.

For `T` invocations of an operation requiring `G` AIR columns/constraints, the useful comparison is

```text
generic AIR:          Theta(T G)
decomposed lookup:    about Theta(T s + s 2^(w/s)) plus lookup consistency.
```

The decomposition is separable from Lasso's PCS. A first S-two prototype can use existing LogUp on
small materialized subtables, then remove those tables analytically. Good targets are bitwise
operations, instruction decode, byte/range operations and functions with a cheap tensor/decomposable
description. Native M31 additions and multiplications are bad targets.

Specialized alternatives should remain workload-specific:

- [LogUp*](https://eprint.iacr.org/2025/946) avoids index-length auxiliary arrays for small indexed
  tables, but is multilinear/sumcheck-oriented;
- Shout/Twist use one-hot sparsity and increments for read-only/read-write memory, but their best
  commitment economics rely on zeros or bits being cheap; and
- [GKR for Boolean Circuits](https://eprint.iacr.org/2025/717) packs bits into univariate words and
  exploits binary precomputation, but its comparison is not an end-to-end S-two comparison.

Local evidence is a reason for selectivity: after substantial optimization, the existing GKR LogUp
path remained 1.085x-2.192x slower than prefix-sum LogUp in the eight measured whole-proof cells and
used more memory. Structured lookup work should delete a large table/AIR block, not merely replace
the current consistency kernel.

## 7. Do not build the first quotient tree if the verifier can reconstruct it

After the trace roots, composition root, OOD claims and batching challenge are fixed, fold the
trace-size DEEP quotient in coefficient form and evaluate only the folded line polynomial. For one
skipped fold this deletes:

- the full quotient-domain LDE;
- the full quotient Merkle tree; and
- the first pointwise Circle-to-line fold.

Skipping `s` folds requires the verifier to reconstruct `2^s` source quotient values from trace
openings. This is favorable for narrow trees and can lose badly for wide traces. A selective variant
can keep committed partial quotients for wide trees and reconstruct narrow groups.

This is included as one global opening transformation, not as the center of the roadmap. It is
orthogonal to Lifted FRI and composes with modular splitting.

## 8. Pack repeated proofs, not only columns inside one proof

For `B` independent instances, pack values at the same domain location into one leaf, random-combine
their DEEP validity functions, and run one FRI/opening protocol. The amortization paper calls this
STARKPack and shows one Merkle commitment/decommitment and one FRI for all instances.

This helps throughput for queued proofs, shards and repeated programs. It does little for the
latency of one large proof because trace generation and most quotient evaluation remain linear in
`B`. The same-program case should additionally share selector/table columns once.

The paper's wide 275-column case improved prover time only around 5%, while proof size and verifier
time benefited much more. This is why batching should follow, not precede, deletion of trace and
interaction columns.

## 9. Delete large proof-identical transient objects

These are not new cryptography, but they remove whole data structures and are worth doing before a
PCS migration.

### 9.1 Derive interaction data from the committed trace

Current example generators retain full copies of values solely to build post-challenge LogUp data.
The most extreme local example is Blake scheduling, which retains roughly `1,056*N` base felts of
lookup data in addition to the trace. Poseidon duplicates initial/final states, XOR tables clone
multiplicity columns, and Plonk retains circuit/preprocessed copies.

After the base commitment, derive denominators from the committed evaluations' trace-size
subdomain or retain a compact expression/index recipe. This changes transient memory from roughly
`Theta((trace + lookup_copy)N)` to `Theta(trace*bN + recipe)` and leaves proof bytes unchanged.

### 9.2 Evaluate OOD claims from the smallest complete domain

When coefficient retention is disabled, the current OODS path builds a barycentric-weight vector
over the full committed LDE domain for every distinct `(height, point)`. A local `2^20` phase
artifact measured 441.348 ms for SIMD weight construction, versus 0.982 ms for the weighted dot
product and 1.387 ms for coefficient-form evaluation. These are microbenchmarks, not a whole-proof
percentage.

Two proof-identical algorithms should be compared:

- evaluate from the trace-size complete subdomain, reducing the weight vector by the blowup; or
- retain coefficients selectively for trees with many OOD masks and evaluate algebraically.

For the barycentric route, derive inverse point-vanishing values directly and use one base/secure
batch inversion rather than a secure division per point. The non-canonical subdomain formula and
bit-reversed ordering require a mathematical parity test against the present full-domain result.

### 9.3 Fuse DEEP quotient construction end-to-end

The SIMD DEEP path currently materializes per-sample numerator vectors, full lifted per-sample
vectors, domain points, an `S x N` denominator-inverse matrix, and the final quotient. A tiled
algorithm can load/lift source values once, update all sample numerators, batch-invert within a
row/sample tile, and emit the final quotient directly.

This reduces scratch memory from `Theta(SN)` secure/base extension values to `Theta(N + tile*S)`.
The proof is identical. Tile size must remain large enough that extra batch-inversion boundaries do
not erase the bandwidth win.

### 9.4 Reuse immutable preprocessing across proofs

[`CommitmentSchemeProver::commit_tree`](../crates/stwo/src/prover/pcs/mod.rs) accepts a borrowed
prebuilt tree, but no production/example caller was found. Cache fixed-column coefficients, LDE and
Merkle prover data once per program/configuration and borrow it per proof. Twiddles and fixed trees
are reusable; barycentric weights at the random OOD point generally are not.

## 10. Hybrid sumcheck islands, not a wholesale GKR replacement

[HyperPlonk](https://eprint.iacr.org/2022/1355) and
[SuperSpartan/CCS](https://eprint.iacr.org/2023/552) show the global attraction of multilinear
zero-check: linear-time proving without quotient FFTs, and cryptographic work that does not scale
with custom-gate degree.

A plausible S-two hybrid is:

- keep ordinary low-degree AIR components in the Circle quotient;
- send only sparse, lookup-heavy, or degree-inflating components through sumcheck; and
- bind their final evaluation claims into the existing opening protocol.

This is worthwhile only if it removes a large `2^K` composition split or a very wide interaction
trace. If reopened, use modern sumcheck algorithms: [More Optimizations to Sum-Check Proving](https://eprint.iacr.org/2024/1210)
reports removal of roughly `2^(n+1)` equality-factor multiplications and 10-25% end-to-end gains in
its use cases, while [A Time-Space Tradeoff for the Sumcheck Prover](https://eprint.iacr.org/2024/524)
provides concrete memory/time choices.

## 11. Pay for bit width in Boolean-heavy components

M31 commits every Boolean as a 31-bit field element. Binary-tower systems instead keep small values
small through the commitment and lift only when security requires it:

- [Binius](https://eprint.iacr.org/2023/1784) avoids tiny-field embedding overhead;
- [ring switching](https://eprint.iacr.org/2024/504) compiles tiny-field multilinears into a
  large-field PCS with linear prover overhead; and
- [tower sumcheck with basis switching](https://eprint.iacr.org/2025/594) batches base-field witness
  symbols so the large-field commitment is only a fraction of the original symbol count.

The raw representation ceiling relative to M31 is up to 31x for bits, about 3.9x for bytes, and
about 1.9x for 16-bit limbs. Those are storage ratios, not expected end-to-end speedups. Packing
independent Boolean lanes into an M31 element does not preserve lane-wise multiplication, so an
explicit mixed-field bridge is required. Start, if at all, with a Boolean-only hash/lookup component.

## 12. Long-term PCS transformations

[Blaze](https://eprint.iacr.org/2024/1609) demonstrates the code-switching idea: commit with a very
cheap linear-time code, then inherit succinct verification from another proximity protocol.
[PIPFRI](https://www.usenix.org/conference/usenixsecurity26/presentation/li-weihan) similarly
partitions a polynomial into independently processed fragments and bridges to FRI. Their published
speedups are for multilinear/binary-field systems and must not be projected onto S-two.

Transferable research questions are narrower:

- can static or lookup-only auxiliary objects use a linear-time code while the primary trace stays
  on the Circle domain?
- can independent component fragments be encoded locally and code-switched only once?
- can a Circle analogue of a constrained-code LDT carry selected constraint claims and remove a
  separate quotient commitment?

These are separate protocol projects, not near-term optimizations.

## 13. Treat soundness parameters as one global optimization problem

Reducing `log_blowup_factor` by one halves trace/composition LDE domains, most Merkle work, DEEP
work, and the first FRI layer. It may therefore beat many implementation optimizations. The right
design space is joint:

```text
(blowup, query count, fold arity, final degree, grinding position/bits, batch challenges).
```

Current [`FriConfig::security_bits`](../crates/stwo/src/core/fri.rs) estimates only
`log_blowup_factor * n_queries`, with PCS adding PoW bits. That is an accounting heuristic, not a
complete non-interactive soundness bound. [Concrete security of non-interactive FRI](https://eprint.iacr.org/2024/1161)
found that the provable security of most examined deployed parameter sets was 21-63 bits below
their conjectured security, despite the conjectured estimates being close to the desired targets.

The optimizer therefore needs a Math-Reviewer-approved random-oracle error budget for batching,
every fold, list/field-size terms, query sampling, grinding location and hash collisions. Until then,
lowering blowup or queries is not a valid speed proposal.

## A combined global design

The most promising end state is an AIR optimizer with the following sequence:

1. **Horizontal sparsity:** extract event-driven dense components.
2. **Trace geometry:** modularly transpose long components by a small `k`.
3. **Column classification:** mark each value as committed witness, analytic public, linear virtual,
   or nonlinear commit-vs-inline candidate.
4. **Lookup synthesis:** use native AIR for cheap arithmetic, decomposed lookup tables for structured
   word operations, and a special small-table/memory path only where its model wins.
5. **Degree/width optimization:** jointly choose auxiliary columns and LogUp batch arity per component.
6. **Mixed-height composition:** evaluate and commit each degree group at its native height.
7. **Opening optimization:** tile DEEP, then selectively elide early quotient/FRI layers.
8. **Service amortization:** share immutable preprocessing and pack independent jobs when latency
   policy permits.

This is the global analogue of Lifted FRI: lifting should become one capability used by the compiler
to support heterogeneous objects, not the roadmap's destination.

## Measurement and stop gates

### Required analyzer

Before implementing protocol changes, emit per-component:

- `N_i`, active rows `A_i`, committed base-coordinate width `W_i`;
- preprocessed, base, interaction and composition bytes;
- `F_i`, chosen LogUp arity and `K_i`;
- LDE point-count and transform stages;
- leaf bytes, leaf hashes, internal hashes and retained Merkle memory;
- OOD mask count `S_i` and DEEP scratch bytes; and
- FRI values/hashes by layer.

Without this inventory, a transformation can improve its local formula while raising a global
maximum, exactly as batch-4 did.

### Prototype order

1. Modular split `k={2,4,8}` on one wide table.
2. Proof-preserving native `K_i` evaluation, then mixed-height composition plus adaptive LogUp.
3. Remove analytic boundary/periodic selectors and one fixed XOR table commitment.
4. Extract one rare, wide event component.
5. Eliminate duplicate interaction inputs, tile DEEP, and cache fixed trees.
6. Prototype the Appendix-C nested domain behind an explicitly non-ZK configuration.
7. Only then test structured lookup/sumcheck islands and direct-folded quotient opening.

### Gates

- Proof-identical work: identical roots/proofs for fixed transcript seeds and at least 3% whole-proof
  improvement for an invasive rewrite.
- Protocol-visible S-two-native work: at least 10% whole-prover improvement, or a stated proof-size/
  verifier benefit that the product actually values.
- Architecture branches: compare equal statements, security targets, fields, hardware and memory;
  reject paper-to-S-two speedup extrapolation.
- Every optimization must record prover time, peak RSS, proof size and verifier time. A kernel-only
  win is insufficient.

## Soundness boundaries

The following require Math Reviewer sign-off and paper-grounded invariants before implementation:

- modular lane/domain mapping and translated cross-lane transitions;
- omission/virtualization of committed columns;
- heterogeneous composition lifting and OOD reconstruction;
- nested `H subset D` encoding and quotient decomposition;
- structured lookup adapters and hybrid sumcheck claims;
- first-layer quotient commitment elision;
- multi-instance random combination; and
- any PCS or soundness-parameter change.

In particular, the existing global composition lift is the result of a consistency fix. A local-degree
implementation must prove that prover and verifier apply identical Circle doublings at the OOD point.
Mixed-height composition and early folding must additionally account for the Circle FFT-space
dimension gap and its scalar `lambda` term before they can be production candidates.

## Primary literature used

- [Circle STARKs](https://eprint.iacr.org/2024/278)
- [On amortization techniques for FRI-based SNARKs](https://eprint.iacr.org/2024/661)
- [Ceno](https://eprint.iacr.org/2024/387)
- [Jolt](https://eprint.iacr.org/2023/1217) and [Lasso](https://eprint.iacr.org/2023/1216)
- [TaSSLE](https://eprint.iacr.org/2024/1075)
- [Twist and Shout](https://eprint.iacr.org/2025/105) and [LogUp*](https://eprint.iacr.org/2025/946)
- [HyperPlonk](https://eprint.iacr.org/2022/1355) and [CCS/SuperSpartan](https://eprint.iacr.org/2023/552)
- [Binius](https://eprint.iacr.org/2023/1784), [ring switching](https://eprint.iacr.org/2024/504),
  and [tower-field sumcheck/basis switching](https://eprint.iacr.org/2025/594)
- [Blaze](https://eprint.iacr.org/2024/1609) and [PIPFRI](https://www.usenix.org/conference/usenixsecurity26/presentation/li-weihan)
- [Concrete security of non-interactive FRI](https://eprint.iacr.org/2024/1161)
