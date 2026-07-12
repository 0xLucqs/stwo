---
name: paper-implementation-divergence-log
description: >
  Living record of all known divergences between the canonical distilled
  theory references (`Circle_STARKs.llm.md`, `Stwo_Whitepaper.llm.md`) and
  the production codebase. Agents MUST read this before modifying any
  theoretically-grounded component. Agents MUST update this when finding new
  divergences. Never resolve a divergence silently.
---

# Paper-Implementation Divergence Log

Last analyzed: 2026-03-05

## Canonical Theory Sources

- `.agents/papers/llm/INDEX.llm.md` — notation map and source navigation
- `.agents/papers/llm/Circle_STARKs.llm.md` — circle FFT/FRI/AIR theory anchors
- `.agents/papers/llm/Stwo_Whitepaper.llm.md` — STWO protocol/soundness/parameters

## How to Use This Log

- **Before modifying**: Search this log for the component you are modifying.
  If a divergence exists, understand it before changing anything.
- **After discovering**: Add a new entry immediately. Do not proceed without documenting.
- **Resolution**: Only a human with cryptography expertise can close a divergence
  as RESOLVED-SAFE. Agents may propose resolution, never confirm it.

## Active Divergences

### DIVERGENCE-001: Dimension Gap Handling in FRI

Paper: `.agents/papers/llm/Circle_STARKs.llm.md` —
`prot:IOP:proximity` + quotient decomposition anchors (`e:decomposition:q`,
`lem:quotient:decomposition`) require an explicit dimension-gap treatment:
`g = f - lambda * v_n` from `L_N(F)` to `L'_N(F)`.

Code: `crates/stwo/src/core/fri.rs` and `crates/stwo/src/prover/fri.rs` —
The FRI verifier has a `first_layer` / `inner_layers` / `last_layer` structure.
The circle-to-line fold (first layer) handles the J-split. There is no explicit
lambda scalar transmission in the current protocol flow. The quotient decomposition
in `crates/stwo/src/core/pcs/quotients.rs` handles the dimension gap implicitly
through the DEEP quotient construction.

Type: Intentional deviation
Risk: NEUTRAL (the constraint system ensures polynomials are in L'_N by construction
via the quotient identity; the dimension gap parameter lambda = 0 when constraint
degree is odd, which is the common case)
Status: OPEN
Notes: Verify that even-degree constraint systems correctly handle lambda != 0.

### DIVERGENCE-002: FRI Folding Order

Paper: `.agents/papers/llm/Stwo_Whitepaper.llm.md` —
`prot:cFRI:multi` / `e:cFRI:multi:folding` describe first J-split then
subsequent projection folds.

Code: `crates/stwo/src/core/fri.rs` — First layer performs circle-to-line
fold (`fold_circle`), then inner layers perform line folds. The constant
`CIRCLE_TO_LINE_FOLD_STEP` is used for the first fold. Inner layer queries are
derived by folding the original queries.

Type: Optimization (batched structure)
Risk: NEUTRAL
Status: OPEN
Notes: The mathematical operation is equivalent; the batching structure differs
from the sequential presentation in the paper.

### DIVERGENCE-003: QM31 as 4 Base Field Polynomials

Paper: `.agents/papers/llm/Circle_STARKs.llm.md` —
AIR model keeps trace polynomials in `L'_N(F_p)` while composition/challenges
use extension-field structure.

Code: Throughout `crates/stwo/src/core/pcs/` — QM31 polynomials are decomposed
into 4 base field coordinate polynomials for commitment and FRI. The composition
polynomial is split into 2 * SECURE_EXTENSION_DEGREE = 8 coordinate polynomials
(see `verify()` in `crates/stwo/src/core/verifier.rs`).

Type: Intentional deviation (efficiency)
Risk: NEUTRAL (mathematically equivalent; reduces FRI to base field operations)
Status: OPEN
Notes: This is standard practice. The `from_partial_evals` method in
`crates/stwo/src/core/fields/qm31.rs:51-57` reconstructs the combined value.

### DIVERGENCE-004: Composition Polynomial Split

Paper: `.agents/papers/llm/Stwo_Whitepaper.llm.md` —
composition and cross-domain quotient construction (`prot:STARK:IOPP`,
`e:crossdomain:quotient`) requires split handling by degree/domain.

Code (historical): `crates/stwo/src/core/verifier.rs` —
`COMPOSITION_LOG_SPLIT: u32 = 1` was hardcoded. The split produced
`2 * SECURE_EXTENSION_DEGREE` columns.

Type: Intentional deviation (simplified) — now generalized
Risk: PERFORMANCE (limited constraint degree to `trace log size + 1`)
Status: ADDRESSED (2026-07-12) — pending human cryptography review
Notes: The hardcoded single split silently forced every component to declare
`max_constraint_log_degree_bound == log_size + 1`. Any component declaring a
higher bound desynced the prover and verifier by doublings and failed the
OODS sanity check (`ConstraintsNotSatisfied`) while trace-domain
`assert_constraints` still passed. Degree accounting of the generalization
(now in `Components::composition_log_split`):

- Let `n_i` = component trace log size, `b_i` = declared bound,
  `n_max = max n_i`, `K = max(b_i - n_i)` (clamped to >= 1),
  `N = lifting_log_size - log_blowup` (the uniform lifting boundary,
  `= n_max` by default).
- PCS sampling evaluates the LIFT of each column to log degree `N`: the
  sample of column T at point s is `T(δ^{N-n_i}(s))` where δ is point
  doubling. The verifier's per-component quotient (mask step `step_N`,
  denominator `v_N(p)`) therefore computes `q_i(δ^{N-n_i}(p))`, of log
  degree `b_i + N - n_i <= N + K`. This is intrinsic to uniform lifting: no
  choice of mask points or denominator can lower it, because the sampled
  first argument is pinned at `δ^{N-n_i}(p)`.
- The old invariant: the prover evaluated `q_i` on a domain of size
  `2^{b_i}` and the accumulator lifted it by `M - b_i` (`M = max b_i`); with
  one split, `N = M - 1`, so prover and verifier agreed iff `b_i = n_i + 1`
  exactly, for every component.
- The fix (this fork): the composition log degree is `n_max + K`; every
  component evaluates its quotient on a domain of log size `n_i + K` (lift
  exponent `n_max - n_i = N - n_i`, matching the verifier); the composition
  is split `K` times into `2^K` parts of log degree `n_max`, each FRI-bound
  at the committed domain `n_max + log_blowup`.
- Soundness: the OODS identity enforced is unchanged —
  `v_N * F = Σ α^k U_i∘δ^{N-n_i}` (Schwartz-Zippel over the OODS point, then
  α-batching separates components; constraint vanishing is enforced by
  divisibility by `v_N`, not by F's degree). Only F's degree bound rises to
  `2^{n_max+K}`, carried by `2^K` FRI-bound parts. `K = 1` reproduces the
  previous protocol bit-for-bit.
- Caveat (completeness/lint, not soundness): a component under-declaring its
  bound is no longer guaranteed to fail at proving time when another
  component raises `K`, since the shared evaluation domain `n_i + K` may
  cover its true degree. The enforced degree bound was always global.
- Repro/regression: `crates/examples/src/state_machine/mod.rs` —
  `test_state_machine_raised_degree_bound_*`.

### DIVERGENCE-005: Pairwise LogUp Column Grouping

Paper: `.agents/papers/llm/Stwo_Whitepaper.llm.md` —
logUp construction in `prot:STARK:IOPP` and related constraint equations;
implementation applies pairwise grouping optimization.

Code: `crates/constraint-framework/src/logup.rs` and
`crates/constraint-framework/src/prover/logup.rs` — LogUp implementation
groups fractions pairwise. The `Batching` type in `lib.rs:47` controls
which entries are batched together.

Type: Optimization (matches paper)
Risk: NEUTRAL
Status: RESOLVED-SAFE
Notes: Implementation matches the optimization described in the whitepaper.

### DIVERGENCE-006: Non-Transposed Merkle Tree

Paper: `.agents/papers/llm/Stwo_Whitepaper.llm.md` —
cross-domain Merkle commitments (`s:cross:domain:merkle`, `alg:merkle`).

Code: `crates/stwo/src/core/vcs_lifted/` — Uses a "lifted" Merkle tree
variant where multiple polynomials of different sizes are committed in a
single tree by lifting smaller polynomials to the largest domain size.
The `lifting_log_size` parameter in `PcsConfig` controls this.

Type: Intentional deviation (efficiency)
Risk: NEUTRAL (reduces number of Merkle trees and proof size)
Status: OPEN
Notes: The `vcs_lifted` module is a newer addition alongside the original `vcs`.

### DIVERGENCE-007: Security Parameter Defaults

Paper: `.agents/papers/llm/Stwo_Whitepaper.llm.md` —
parameter/soundness targets (Section "6. Parameter Rules", `s:example:params`)
including 100-bit security regimes with ~26 grinding bits.

Code: `PcsConfig::default()` in `crates/stwo/src/core/pcs/mod.rs` — Default uses
`pow_bits: 10`, `log_blowup_factor: 1`, `n_queries: 3`. This yields
`security_bits() = 10 + 1*3 = 13` bits — far below production requirements.

Type: Intentional deviation (test defaults)
Risk: SOUNDNESS (if used in production without override)
Status: OPEN
Notes: The default config is clearly for testing. Production deployments
(e.g., stwo-cairo) must set appropriate parameters. Document this prominently.

### DIVERGENCE-008: Proof of Work Grinding Bit Limit

Paper: `.agents/papers/llm/Stwo_Whitepaper.llm.md` —
parameter guidance in Section "6. Parameter Rules" / `s:example:params`.

Code: `crates/stwo/src/prover/backend/simd/grind.rs` — TODO comment:
"support more than 32 bits." Current implementation limited to 32 PoW bits.

Type: Implementation limitation
Risk: NEUTRAL (26 < 32, so current limit is sufficient for target security)
Status: OPEN
Notes: The 32-bit limit is adequate for the 26-bit target but leaves no room
for future security parameter increases.

### DIVERGENCE-009: Poseidon2 Constants and Round Order

Paper: Poseidon2 paper (https://eprint.iacr.org/2023/323.pdf) Section 5.

Code: `crates/examples/src/poseidon/mod.rs` — Three critical TODOs:
- "Use poseidon's real constants" — placeholder constants in use
- Coefficients unverified against Section 5.3
- Round matrix may be applied in wrong order relative to paper

Type: Bug candidate
Risk: SOUNDNESS (for any system using this Poseidon2 implementation)
Status: OPEN — ESCALATION REQUIRED
Notes: This is in the examples crate, not the core prover. However, any
downstream user of this code (e.g., for a Poseidon2 hash proof) would
inherit these issues. The placeholder constants make this unsuitable for
production use.

### DIVERGENCE-010: LogUp Validation Missing in Blake Example

Code: `crates/examples/src/blake/round/constraints.rs` — TODO: "validate logup"
Code: `crates/examples/src/blake/scheduler/mod.rs` — TODO: "validate logup"

Type: Implementation gap
Risk: SOUNDNESS (for Blake example only — constraints may be under-specified)
Status: OPEN
Notes: Example code only, but users may copy patterns from examples.

## Resolved Divergences

### DIVERGENCE-005 (see above)
Resolved: Pairwise LogUp grouping matches STWO distilled protocol guidance.
