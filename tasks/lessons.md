# Lessons

## 2026-07-21 — Audit findings must be checked against repo history and prior audits
- **Correction:** Presented "add #[inline] to field ops" as a fresh finding; the user had
  already landed `perf: inline field arithmetic` (b2a012c3) for the scalar types. The
  packed-SIMD finding was still valid, but presenting it without acknowledging the
  existing commit read as stale information.
- **Correction:** Ranked GKR/sumcheck parallelization as the #1 win. In this codebase the
  GKR lookup path is slower than the LogUp prefix-sum path and is not the production
  route — production perf findings must be weighted by which code path actually runs.
- **Rules:**
  1. Before presenting audit findings, run `git log --all --grep=<keyword>` for each
     headline item to check whether it (or a sibling of it) was already done, and say so.
  2. Check `tasks/` for prior audits first (`perf-audit-2026-07-06.md` has a
     "Findings checked and rejected" list — don't re-chase those).
  3. Ask or verify which of multiple alternative code paths (GKR vs LogUp, vcs vs
     vcs_lifted, CPU vs SIMD) is the production one before ranking impact.

## 2026-07-21 — Follow a benchmark stop condition through its diagnostic action
- **Correction:** Reported the GKR `<2x` stop condition and stopped, even though the spec
  explicitly says to re-profile next and the user asked to implement what was possible.
- **Rule:** When a measured gate says "stop and profile/reproduce/audit," stop the risky
  implementation work but immediately perform the named safe diagnostic unless it needs
  new authority or is genuinely blocked.
- **Rule:** For feature A/B benchmarks, build into distinct target directories and verify
  each Cargo fingerprint before timing; a shared target directory can leave a stale or
  overwritten feature artifact that makes the comparison meaningless.

## 2026-07-21 — Do not leave an already-runnable final measurement behind a stale gate

- **Correction:** Treated the final measurement as if it still needed an approval pause
  after E1 had been approved and E3's pre-existing initialization issue had explicitly
  been removed as a blocker.
- **Rule:** Re-evaluate every gate after the user resolves it. If the accepted artifact and
  harness already exist, run the bounded measurement immediately; a rejected experiment
  can be reverted by its performance rule without waiting on an acceptance-only gate.

## 2026-07-22 — Treat examples as a pattern request, not the requested destination

- **Correction:** The user named Lifted FRI as an example of a clever mathematical trick, but the
  research was organized around deciding whether to adopt Lifted FRI. That narrowed a global
  algorithm-search request into a protocol-comparison report.
- **Rules:**
  1. When a user says “for example,” extract the abstraction illustrated by the example before
     choosing the research taxonomy. Here the abstraction is “protocol-level transformations that
     delete asymptotic work or whole commitments,” not “FRI variants.”
  2. For broad optimization research, search every major proving-cost layer—arithmetization,
     constraints/lookups, quotient construction, commitments/LDT, queries, batching, and
     memory/time tradeoffs—and rank cross-layer combinations as well as isolated changes.
  3. Lead with genuinely new global algorithms and explain how each changes the cost equation;
     place implementation-local micro-optimizations and the user's example in supporting roles.

## 2026-07-22 — Encode benchmark hygiene in the runner

- **Correction:** The first mobile-runner draft emitted one matrix sample, omitted discarded
  warmups, and labeled a repeated proof as cold even though its reported total excluded setup.
- **Rule:** Implement warmup counts, recorded-sample counts, medians, and cold/warm ordering as
  executable runner behavior; test invocation counts and tags, and use an outer wall clock when
  the reused phase timings intentionally exclude setup.

## 2026-07-22 — Retire a workstream's tail when its verdict lands
- **Correction:** The campaign path still scheduled GKR experiments (E1-E3) whose
  only purpose was making an already-decisive verdict "asterisk-free," including a
  soundness-review-gated item (E3) on a path that will never ship.
- **Rule:** When a decision-rule measurement lands, re-justify every remaining item
  of that workstream on independent merits (shared infrastructure, correctness,
  reusable tests). Items that only serve the dead conclusion get cut, not finished.
