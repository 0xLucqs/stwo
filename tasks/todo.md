# Engine unlock: `bound = log_size + K` for FrameworkComponents (K ≥ 1)

# Build hardening + hash micro-benchmark (2026-07-22)

Spec: `tasks/build-hardening-spec.md`.

## Plan

- [x] W1: audit the existing packed CM31/QM31 inline-only diff, add any omissions, obtain
      independent scalar-semantics review, run every specified gate, and commit only the two
      target files with the requested message.
- [x] W2: reconcile `tasks/build-requirements.md` with the completed mobile benchmark tables and
      verify that release/LTO/codegen, aarch64/ABI, features, and five-minute ±30% self-check
      guidance are complete.
- [x] W3 (last): after all W1/W2 work and checks, create the smallest standalone benchmark,
      verify no other cargo/Gradle/rustc process is active immediately before each timed run,
      measure P/default and E/background Blake2s-compress16 versus SHA-256 N={1,4,8,16}, and record
      reproducible MB/s results in the spec.
- [x] Run final diff/document checks and add a review with commands, results, artifacts, and any
      remaining external limitations.

## Review

W1 added exactly 23 `#[inline(always)]` attributes (8 CM31, 15 QM31) and no body,
signature, representation, or unsafe-path change. Independent Math Reviewer verdict:
APPROVE at 99% confidence. The required stwo library test (282 passed), prover check,
repository clippy, rustfmt, focused CM31/QM31 scalar-equivalence tests, and diff checks
all passed. Commit: `perf(simd): inline packed CM31/QM31 field arithmetic`.

W2 now requires release, fat LTO (thin minimum), one codegen unit, `prover` +
`parallel`, baseline AArch64 Neon with no `target-cpu`, and arm64-only Android. Its
five-minute check uses the filled `pipeline_e2e` identity-proof T0 medians and explicit
±30% intervals; Android tiers remain ungated until actual device results exist. Prettier,
requirement-presence, and independent range calculations passed.

W3 ran last from `/private/tmp/stwo-w3-hash.CjxKKi/`. Both timed launches had empty
Cargo/Gradle/rustc and benchmark-process scans immediately beforehand. Median P/default
throughput was 1,254.013 MB/s Blake2s versus 1,593.200 MB/s SHA-256 at N=16 (1.270x);
E/background was 529.432 versus 796.964 MB/s at N=8 (1.505x). The SHA hardware path,
all digest outputs, binary hash, raw samples, and quiet evidence are recorded in the spec
and scratch provenance. Android repeat remains pending on the Nord CE / W1 app.

---

Spec: ~/eu-id/.claude/worktrees/mldsa-claude-perf/tasks/stwo-engine-batching-spec.md
Branch: dev-copy (fork/dev-copy only). Base rev: 72b638e7.

## Root cause (established by reading, to be confirmed by D1 repro)

Let n_i = component trace log size, b_i = declared max_constraint_log_degree_bound,
e_i = b_i - n_i - 1 (excess over standard), N = max_log_degree_bound = lifting - blowup.

- PCS sampling evaluates the LIFT of each column to log degree N at the requested
  point: sample of T at s = T(δ^{N-n_i}(s)), δ = point doubling.
- Verifier per-component quotient (mask step_N, denominator v_N(p)):
  computes q_i(δ^{N-n_i}(p)).
- Prover accumulates q_i evaluated at domain 2^{b_i}, lifted by (M - b_i) to the
  accumulator size M = max b_i: contributes q_i(δ^{M-b_i}(p)).
- Composition split once (COMPOSITION_LOG_SPLIT = 1) ⇒ N = M - 1.
- Consistency: N - n_i = M - b_i ⟺ b_i = n_i + 1 for EVERY component.
  Any e_i > 0 desyncs prover/verifier by e_i doublings → OODS mismatch (F1),
  while trace-domain assert_constraints still passes (it never uses b_i).

Degree accounting: verifier-side contribution of component i is intrinsically
q_i ∘ δ^{N-n_i}, log degree N + 1 + e_i. With a single split the FRI bound on the
composition halves is N ⇒ forces e_i ≤ 0. The unlock requires splitting the
composition K = 1 + max e_i times into 2^K parts of log degree n_max each.
Today's code is exactly the K=1 special case (the verifier even has
`TODO(Leo): remove this once the composition poly split can be dependant on a
config` on COMPOSITION_LOG_SPLIT).

## Fix design (generalize K=1 → K)

K := max_i(b_i - n_i), n_max := max_i(n_i), M' := n_max + K (composition log degree),
N := M' - K = n_max (unchanged mask/denominator framing — verifier component code
untouched).

1. core/air/components.rs: composition_log_degree_bound() → n_max + K;
   new composition_log_split() → K. (K=1 ⇒ identical to today: max b_i.)
2. prover accumulation: every component evaluates its quotient on domain
   n_i + K (not b_i) and accumulates at slot n_i + K ⇒ lift exponent
   M' - (n_i+K) = n_max - n_i = N - n_i ✓ matches verifier.
   Plumb K through DomainEvaluationAccumulator.
3. EvaluationMode::infer: uniform excess K ⇒ SubDomain{blowup - K} whenever
   K ≤ blowup (mixed b_i-n_i no longer forces ExtendToEvalDomain — fixes the
   F2 practical hit for blowup ≥ K). K > blowup ⇒ ExtendToEvalDomain; clear
   panic message when coefficients not stored.
4. prove_ex: split composition K times (2^K parts, left-major order), extend
   2^K·4 coordinate polys, push 2^K·4 oods sample vecs.
5. extract_composition_oods_eval: fold 2^K secure evals pairwise, level
   ℓ = 1..K multiplier x(δ^{N+ℓ-2}(p)). (K=1: x(δ^{N-1}(p)) = today.)
6. verifier.rs: replace COMPOSITION_LOG_SPLIT const with components-derived K;
   commit composition tree with 2^K·4 columns of size N.

Soundness note: the OODS identity enforced is v_N·F = Σ α^k U_i∘δ^{N-n_i} —
identical to today; only F's degree bound rises to 2^{n_max+K}, carried by 2^K
FRI-bound parts of log degree n_max. Constraint vanishing is enforced by
divisibility, not by F's degree. Per-component declared bounds remain
prover-side sizing hints; the enforced bound is global (as today).
Caveat to document: a component under-declaring its bound is no longer
guaranteed to fail when another component raises K (still sound; the eval
domain covers it).

## Tasks

- [x] D1: repro test committed (2e1aa485) — reproduced exactly per spec at
      72b638e7: assert_constraints passes, prove fails ConstraintsNotSatisfied
      (F1), "coefficients are not stored" panics (F2, incl. blowup 2 via
      non-uniform infer).
- [x] D2: fix committed (96a8c667) + mle_eval custom prover contract fix
      (8c998390). Degree accounting in divergence log (DIVERGENCE-004).
      Adversarial soundness review: no soundness defect; the one finding
      (mle_eval K=1 assumption) fixed.
- [x] Full fork suite green: prover 348, prover+parallel 348, verifier-only
      82, framework no-default 4, slow-tests 350, tracing+prover 349,
      clippy/fmt/doc/no_std all clean. Pushed to fork/dev-copy.
- [x] D3: eu-id downstream complete (branch feat/mldsa-claude-perf,
      commits 1068dbea pin-bump + 7507d8a6 batch-4).

## Review

Engine (fork/dev-copy, pushed): 2e1aa485 repro, 96a8c667 fix, 8c998390
mle_eval contract fix. Root cause: uniform PCS lifting pins the verifier's
per-component quotient at q∘δ^(N-n); the hardcoded single composition split
made the prover lift by (max_bound - b), agreeing only when b = n+1.
Generalized the split to K = max(b-n): quotients evaluate on n+K domains,
composition splits K times into 2^K FRI-bound parts. K=1 bit-identical to
the old protocol. All fork gates green (prover/parallel/verifier-only/
slow/tracing 348-350 each, clippy/fmt/doc/no_std). Adversarial review:
no soundness defect. Degree accounting: divergence log DIVERGENCE-004.

eu-id S6 measured (pq_perf_probe, single-thread, n=4 median):
proof 2,201,000 → 1,810,481 B (−17.8%), verify 18 ms,
prove 4,911 → 5,378 ms (+9.5%). Tree2 down as predicted (1,611→1,337 ms);
stark phase +712 ms because uniform-K doubles every component's
constraint-eval domain (incl. coeffs) — follow-up lever: FFT-extend
low-excess quotient columns engine-side. Proof lands 60 KB above the
~1.75 MB estimate (consumers-only scope vs all-cols estimate).
Full matrix green: mdoc 25+19, keccak 31, mldsa 74, sha256 145,
credential_pipeline 3+1, quantum-only deps clean.

---

# Mobile proving benchmark harness (2026-07-22)

Spec: `tasks/mobile-bench-spec.md`.

## Plan

- [x] Inventory the existing Path A instrumentation, FFT/Merkle/field/hash APIs, workspace
      profiles, Android tooling, and cloud prerequisites while preserving the dirty worktree.
- [x] Add the smallest reusable sink/API adjustment needed in `stwo-examples`, then implement
      `mobile-bench/bench-runner` as a host binary and optional JNI cdylib with deterministic
      JSON-lines output, metadata, warmups, medians, and focused tests.
- [x] Add a minimal arm64 Android Game Loop wrapper and reproducible `cargo-ndk`/Gradle build
      instructions grounded in the current Firebase contract.
- [x] Route the `PackedCM31`/`PackedQM31` inline-only change through Crypto Specialist
      implementation and Math Reviewer sign-off; make no other proof-system changes.
- [x] Document release/aarch64/parallel build requirements, the five-minute self-check, and the
      deferred Honor 10 protocol.
- [x] Run focused host/JNI checks and the binding full test, release e2e, clippy, and rustfmt gates.
- [x] Build fresh isolated Apple artifacts, record binary SHA-256/RUSTFLAGS, and run P-core,
      background/E-core, and one-thread variants sequentially with the required warmups/repeats.
- [x] If already-authorized `gcloud` prerequisites exist, select physical EU-relevant devices and
      run the Firebase matrix sequentially; otherwise record the exact external blocker without
      inventing T1/T2 results.
- [x] Fill available result tables and interpretation, append a review with commands/artifacts,
      and ensure the final diff contains no unrelated user-owned changes.

## Review

Implemented the reusable Path A report API and the `bench-runner` host/JNI crate. The suite now
emits metadata first; uses one discarded warmup plus five e2e runs; uses one discarded warmup plus
seven micro/hash samples; records a distinct cold/warm wall-clock pair before the matrix; captures
the final binary SHA-256, decoded Cargo `RUSTFLAGS`, threads, Android model/CPU/thermal metadata;
and propagates output/measurement failures through CLI panics or Java exceptions. Eight runner
tests cover report ordering, protocol counts/tags, medians, hashes, JSONL, and write failures.

Added the minimal Java Game Loop wrapper with the exact action/category/MIME contract, scenario
metadata, JNI invocation on a background thread, cache-to-`Intent.data` artifact copy, and a
checked-in Gradle 9.5 wrapper. The final fat-LTO arm64 shared library is
`9476192bc505b35fdf9b6189edc1cdbf5e9fbd7b227cdf9eca7f766f8af828a1`; the v2-signed,
arm64-only APK is `cba2151c57e69043764de4b7de2829645ce0881bd4e45cffde229010439b0243`.
JNI symbol and manifest/ABI inspection passed; Gradle unit, lint, and assemble tasks passed.

W3 added exactly 23 `#[inline(always)]` attributes (8 CM31, 15 QM31), with no arithmetic/body
change. Crypto Specialist focused release tests passed; the Math Reviewer independently approved
at 99% confidence and reran the 282-test library gate. Final binding gates: stwo 282 passed;
release GKR e2e 4 passed/1 ignored; bench-runner 8 passed; full clippy, rustfmt, Android cross-build,
Gradle lint/package, signature, JNI-symbol, JSONL, and diff checks passed.

T0 used one fresh M2 Max binary with SHA-256
`17490dc6dc7d8eff18acacdc9168c994ae6fcdb39db9e9da5921305330460ff3`, commit
`9c5bebf1e8b3fde619d6e456e96bbe85b30b0a10`, `RUSTFLAGS=-C target-cpu=native`, fat LTO,
one codegen unit, and `parallel`. Four sequential reports (P/default, E/background, and each at one
thread) contain 171 valid JSON lines apiece under `tasks/mobile-bench-results/`; tables and W5
interpretation are filled in `tasks/mobile-bench-spec.md` and Apple self-check references in
`tasks/build-requirements.md`.

Firebase execution is the only incomplete external result: the configured project/account exist,
but model listing fails token refresh with an interactive `gcloud auth login` requirement. No
device was selected, no Test Lab invocation occurred, and no spend was incurred. T1/T2 and their
Android reference cells are explicitly marked blocked rather than estimated.

---

# Global prover-algorithm research, corrected scope (2026-07-22)

## Plan

- [x] Build a global cost model spanning trace representation, constraints/lookups, quotient
      construction, commitments/LDT, queries, batching, and memory traffic.
- [x] Research mathematical transformations that delete whole columns, domains, commitments,
      FFTs, rounds, or repeated proof work; use Lifted FRI only as one example.
- [x] Separate S-two-native ideas, hybrid Circle/MLE ideas, and full architecture migrations.
- [x] Map each serious candidate to the active source path and quantify its cost-equation change,
      workload assumptions, interactions, proof/verifier consequences, and soundness burden.
- [x] Produce a corrected ranked research agenda and verification plan grounded in primary
      literature, repository evidence, and prior measured experiments.

## Review

Completed a corrected 550+ line cross-layer study in
`tasks/global-prover-algorithm-research.md`. The central recommendation is a global AIR optimizer,
not a Lifted-FRI migration: modularly transpose the max-height trace frontier; make event tables
dense; classify analytic/virtual/committed columns; jointly choose LogUp arity, auxiliary columns,
and local quotient degree; and commit native mixed-height composition parts.

Top concrete candidates are modular split-and-pack, degree-local composition plus adaptive LogUp,
analytic selectors/tables, and event-driven heterogeneous components. A Math Reviewer confirmed
that row-residue splitting is Circle-friendly when re-arithmetized on a fresh canonical `N/k`
domain, and that Appendix-C `H subset D` is sound only as a distinct non-ZK Circle-STARK variant
with its mixed quotient decomposition and `lambda` term.

Repository review added proof-identical object deletion: remove Blake/Poseidon challenge-delayed
lookup copies, tile DEEP construction to eliminate `S*N` scratch matrices, use native-domain OODS,
and reuse immutable trees. Literature claims are linked to primary papers and kept separate from
local phase measurements. No proof-system code or benchmarks changed; untracked documentation only.
Whitespace/diff checks pass. Existing dirty source files were not modified.

---

# Byte-identical prover-algorithm filter (2026-07-22)

## Plan

- [x] Define strict serialized-proof identity and the transcript/PoW invariants it implies.
- [x] Reclassify every global research candidate as proof-identical, conditionally identical, or
      protocol-changing.
- [x] Validate native-degree composition, OODS, DEEP, preprocessing, lookup-data, FFT, Merkle, and
      caching candidates against current source paths.
- [x] Rank the remaining byte-identical work by expected whole-prover impact and define parity tests.

## Review

Completed `tasks/byte-identical-prover-optimizations.md`, a 400+ line strict filter over the global
roadmap. The hard rule is to change internal computation only while recreating every existing
transcript-visible object, order, raw field encoding, PoW nonce, and serialized field. Lifted FRI,
new commitments/columns, altered lookup/FRI structure, batching, and parameter changes are excluded.

The highest-upside mathematical candidate is two-stage native-`K_i` quotient evaluation followed by
exact reconstruction of today's global composition. It remains research-only until the Circle
`L_N = L'_N + <v_n>` dimension gap and invalid-trace behavior are resolved or equality is enforced
with a safe fallback. The lower-risk implementation order is native/coefficient OODS, duplicate
lookup-data removal, tiled DEEP, immutable tree reuse, exact analytic fixed LDEs, then exact
FFT/Merkle/FRI-fold kernels. The report includes full-proof byte gates, raw-M31 parity checks,
first-divergence diagnostics, CPU/SIMD and serial/parallel matrices, and PoW enumeration caveats.
No proof-system source or benchmark was modified; existing dirty source files remain untouched.

---

# GKR performance spec scoping and implementation (2026-07-21)

## Plan

- [x] Audit `tasks/gkr-perf-spec.md` against current code, dirty changes, prior audits, and protocol invariants.
- [x] Classify each work item by ownership/soundness risk and select the smallest safe implementation slice.
- [x] Implement the approved benchmark/test/doc slice without disturbing existing user changes.
- [x] Run correctness gates, serial compilation, formatting, and the W3 2^20 A/B benchmark.
- [x] Record completed work, measured results, deferred scope, and review findings below.

## Review

Implemented W2: four single-instance SIMD benchmarks now run at 2^16 and 2^20;
CPU and batch-4 remain at 2^16. Added field-by-field CPU/SIMD proof and artifact parity
for every GKR variant at 2^6, 2^7, and multi-chunk 2^14, plus an unequal-size batch.

Scoped out / blocked:

- W0: subsequently unblocked by replacing the local `gen_eq_evals` allocation with safe
  zero initialization. W1 remains superseded by the final experiment plan.
- W4: the original SIMD cutoff used `2^(V-2)` instead of the next round's
  `2^(V-3)` terms and could cache zero sums. A later final-experiment pass implemented
  the corrected design as E3, completed Crypto Specialist and Math Reviewer gates, then
  rejected and reverted it on end-to-end performance.
- `t=1/2`: deferred pending human review. The end-to-end comparison was later completed
  under the dedicated final-experiment spec.

The first W3 table was invalid: a stale 2026-07-04 serial executable was compared with
a current parallel executable from the shared target directory. The isolated rebuild and
corrected result are recorded in the profiling review below.

Verification: stwo prover+parallel 281 passed; serial lookups 29 passed;
constraint-framework 24 passed; serial/parallel benchmark checks, full clippy,
rustfmt, and diff checks passed. No commits or staging performed.

---

# GKR 2^20 parallel slowdown profiling (2026-07-21)

## Plan

- [x] Reproduce the serial/parallel gap with distinct, verified feature builds.
- [x] Capture a time profile for the parallel multiplicities benchmark at 2^20.
- [x] Compare serial evidence where needed and attribute compute versus scheduling overhead.
- [x] Record profile artifacts, hot paths, and the smallest evidence-backed next change.

## Review

There is no verified slowdown. Isolated current binaries measured the exact multiplicities
2^20 cell at 72.926 ms serial and 22.079 ms with 12 Rayon workers: 3.30x speedup.
Short diagnostic runs measured 20.721 ms with 8 workers (3.52x) and 24.776 ms with
4 workers (2.94x); the 8-worker result needs full-protocol confirmation.

Fresh serial profile (7,565 snapshots): packed QM31 multiplication 77.20%, `prove_batch`
residue 10.18%, secure MLE folding 3.85%, fraction addition 3.09%, next-layer generation
2.93%. Fresh parallel profile (73,502 main+worker thread-slot snapshots): condition waits
30.04%, scheduler switches 29.96%, packed QM31 multiplication 26.15%, generic sum closure
2.85%, secure MLE folding 1.28%. Wait samples include expected barriers and idle slots.

Raw profiles: `/private/tmp/gkr-profile-wEtDpl/serial-current-10s.sample.txt` and
`/private/tmp/gkr-profile-wEtDpl/parallel-current-10s.sample.txt`. No proof code changed.
Next: run all four W3 cells from isolated target directories and confirm the 8-worker result.

---

# Complete GKR W3 isolated benchmark matrix (2026-07-21)

## Plan

- [x] Re-verify isolated serial and parallel binaries, Cargo feature fingerprints, and exact filters.
- [x] Run the full default-Criterion serial/parallel matrix for all four SIMD 2^20 cells.
- [x] Re-run the fastest candidate with 8 and 12 Rayon workers under the same full protocol.
- [x] Record confidence intervals, speedups, and the W3 decision in the performance documents.
- [x] Run documentation/diff checks and add this pass's review.

## Review

W3 is complete. The isolated full-protocol matrix measured 12-worker speedups of 2.77x
(grand product), 3.60x (generic), 3.41x (multiplicities), and 3.14x (singles), for a 3.21x
geometric mean. Every cell clears the 2x profiling stop gate; the LogUp cells remain below
the 4x target.

Eight workers were 6.82% and 2.57% faster by time for grand product and generic, but 1.02%
and 0.90% slower for multiplicities and singles. The geometric-mean time reduction was
only 1.92%, so the short-run multiplicities signal did not reproduce and no worker-default
change is justified. Criterion baselines are under
`target/criterion/*/gkr-w3-full-{serial,parallel-12,parallel-8}/`. No proof code changed.

---

# GKR performance handoff document (2026-07-21)

## Plan

- [x] Reconcile the scope, implementation, verification, benchmark, and profile records.
- [x] Write one self-contained handoff with decisions, blockers, commands, and next steps.
- [x] Cross-check numbers and paths against the source documents and run diff checks.
- [x] Record the completed handoff review here.

## Review

Created `tasks/gkr-perf-handoff.md`, a self-contained 367+ line handoff covering the dirty
worktree, implemented and pre-existing changes, verification gates, historical 2^16 data,
the invalid and corrected W3 measurements, worker tuning, profiles, the prefix-sum LogUp
comparison, W0/W1/W4 blockers, incomplete result tables, reproduction commands, and the
evidence-backed path forward. Cross-checked measurements and file locations against the
source plan/spec and current code. Documentation-only pass; `git diff --check` passes.

---

# GKR end-to-end harness (2026-07-21)

## Plan

- [x] Audit the current example LogUp and promoted MLE-eval APIs against `gkr-e2e-spec.md`.
- [x] Implement the example-only Path A/Path B harness, instrumentation, and G1-G3 tests.
- [x] Compile and run focused tests, then fix API/wiring mismatches without framework edits.
- [x] Run formatting, clippy, and diff checks; document any spec reconciliation.
- [x] Run the requested measurement matrix if the correctness gates pass.
- [x] Record results and review without committing or staging user-owned changes.

## Review

Implemented the new example-only harness in `crates/examples/src/gkr_e2e/mod.rs` and
exported it from `crates/examples/src/lib.rs`. Both paths use the exact same one-element
relation draw and `column - z` denominators. Path A proves the interaction-trace LogUp;
Path B proves GKR output claims and binds the combined input MLE claim with one promoted
framework MLE-eval component. Added G1-G3, the ignored measurement entry point, phase and
proof-size instrumentation, M31 capacity assertions, and malformed GKR shape preflights.

Math Reviewer sign-off: 97% confidence. Focused tests: 3 passed, 1 ignored. Full clippy,
format, and diff checks passed. The isolated release executable was built with `parallel`
and `-C target-cpu=native`, SHA-256
`1f227012c866feba26af787c25d480651958d946fdd866d6b756b067a181bc8c`.

The full matrix used 12 Rayon workers, one discarded warmup, three timed runs, and separate
peak-RSS invocations. Path B did not cross Path A at either 2^16 or 2^20 for
`L in {1,4,16,64}`. At 2^20 it was 1.83x-1.97x slower and used more peak memory in every
cell. Results, median-run phase breakdowns, proof sizes, GKR felt counts, and interaction
tree widths are recorded in `tasks/gkr-e2e-spec.md`. No commits or staging performed.

---

# GKR final experiments (2026-07-21)

## Plan

- [x] Audit E1-E3 against current implementations, all oracle impls, ownership, and soundness gates.
- [x] Obtain Math Reviewer guidance for E1/E3 before any proof-path edit.
- [x] Implement, verify, measure, and record E1 without E2 contamination.
- [x] Implement, verify, measure, and record example-only E2.
- [x] Implement and verify E3 through Crypto Specialist + Math Reviewer workflow; measure before commit.
- [x] Run the final full matrix and kernel benchmarks on the fastest subset (E1+E2), then close the verdict.
- [x] Keep experiment commits isolated and preserve all unrelated dirty changes.

## Blocker / review record

- E1 and E3 were handled through Crypto Specialist implementation and Math Reviewer
  post-review. E1 was approved and committed. E3 passed the math/correctness gate but
  failed the performance gate, so it was rejected and reverted without a commit.
- The local `simd/lookups/gkr.rs::gen_eq_evals` allocation is now safe zero initialization.
  The broader pre-existing `set_len` / uninitialized-buffer idiom is explicitly out of scope
  here and needs a separate repository-wide policy decision rather than piecemeal perf edits.
- Corrected the experiment spec's determinism, fresh-baseline, regression-selection,
  E2 shifted-column count, and E3 fixed-chunk requirements before implementation.
- E1 Math Reviewer post-sign-off: 98% confidence. Full gates pass; serial and parallel
  full-proof/artifact digests match for same-size and unequal-size batches. Interleaved
  isolated measurement reduced `gkr_prove` from 1,610.938ms to 882.190ms at 2^20/L=64
  (-45.2%) and Path B total from 4,053.760ms to 3,292.604ms (-18.8%).
- E2 full gates pass. At 2^20/L=64, packed/parallel glue reduced `gkr_layers` from
  421.472ms to 120.041ms and `mle_combine` from 453.446ms to 114.117ms; Path B total
  fell from 3,055.893ms to 2,408.856ms (-21.2%).
- E3 Math Reviewer post-sign-off: 98% confidence; all correctness gates pass. Isolated
  kernels improved 10.7–16.5%, but interleaved e2e regressed `gkr_prove` 8.4% at L=1 and
  10.2% at L=64, and Path B total regressed 4.9% / 7.8%. E3 is rejected and reverted; the
  fastest verified subset is E1+E2.
- Final E1+E2 matrix: no crossover in any of eight cells. Path B is 1.085x–2.192x Path A
  and uses more peak memory everywhere. The closest cell is 2^16/L=64 (196.688ms versus
  181.353ms); at 2^20/L=64 it is 2,408.856ms versus 2,105.763ms (1.144x). Proof sizes
  remain bit-identical to baseline. Verdict: **CLOSED — no crossover after E1–E3.**
- Documentation finalized in `gkr-final-experiments-spec.md`, `gkr-e2e-spec.md`,
  `gkr-perf-plan.md`, and `gkr-perf-handoff.md`. E1/E2 commits are `99d73f3a` and
  `9c5bebf1`; pre-existing W0 and unrelated dirty files remain untouched and unstaged.
- Fresh final-tree verification: 282 `stwo` prover+parallel tests, 30 serial lookup tests,
  24 constraint-framework tests, and 4 release GKR e2e tests passed; full clippy and
  rustfmt scripts passed.

# Algorithmic proving-speed research (2026-07-21)

## Plan

- [x] Establish the active production proof paths, benchmark baselines, and dominant costs from source, history, and prior audits.
- [x] Review current Circle FRI/PCS mathematics and compare lifted FRI plus compatible folding/soundness variants from primary sources.
- [x] Survey broader algorithmic prover improvements and separate STWO-compatible upgrades from architectural migrations.
- [x] Rank candidates by realistic prover impact, implementation effort, verifier/proof-size tradeoffs, and soundness risk.
- [x] Define measurement gates and document a sourced recommendation with unresolved questions and escalation points.

## Review

Completed a read-only source/history/literature study and documented the result in
`tasks/algorithmic-proving-speed-research.md`. The main verdict is not to begin with a wholesale
Lifted-FRI migration: the current PCS already has lifted mixed-height commitments, lifted DEEP
batching, one combined quotient, and multi-step Circle FRI, while the cited HHM25 construction is
still unpublished and has no Circle-STARK speed benchmark.

Ranked the trace-domain/algebraic OODS fast path, first-layer-elided FRI with a directly folded
LDE, degree-aware per-component composition domains, an immediate `fold_step = 2` experiment,
row-major DEEP quotient accumulation, and memoized lifted-Merkle layers ahead of a new PCS. The
report also covers batching independent proofs, flat Circle sumcheck, workload-specific lookup
arguments, FFT pass reduction, STIR/WHIR/BaseFold/Brakedown/Blaze/PIPFRI, exact security-parameter
optimization, rejected directions, benchmark gaps, measurement gates, and a formal soundness
escalation.

Cross-checked recommendations against current source, the open divergence log, recent commits,
prior performance audits, the measured uniform-`K` regression, and the final LogUp-vs-GKR matrix.
No proof-system code was modified and no benchmark was rerun; reported timings are existing local
artifacts and are clearly labeled as phase microbenchmarks or prior experiments rather than new
whole-prover measurements.

---
