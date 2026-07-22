# Field swap analysis: primes above 2^32 for one-word-per-FE (2026-07-22)

Question: swap M31 for a prime slightly above 2^32 so one RV32 word fits per field
element (RISC-V zkVM motivation).

## Circle-eligible candidates exist

Constraint: p ≡ 3 (mod 4), p + 1 = c·2^k (small odd c) for the circle group's 2-adic
tower. Verified prime candidates in (2^32, 2^40) (deterministic Miller-Rabin):

| p | size | circle 2-adicity |
|---|------|------------------|
| 5·2^32 − 1 = 21474836479 | ~2^34.3 | **32** (exceeds M31's 31) |
| 3·2^34 − 1 = 51539607551 | ~2^35.6 | 34 |
| 15·2^31 − 1 | ~2^34.9 | 31 |
| 3·2^38 − 1 | ~2^39.6 | 38 |

Best candidate if ever needed: **5·2^32 − 1** (u32 fits with headroom, 2-adicity 32,
reduction = shift + divide-by-5 magic constant).

## Verdict: NO — efficiency case does not close

1. **~3-5x slower field ops on 100% of prover work**: u64 storage halves SIMD lanes;
   34×34 > 64 bits forces two-limb multiplies (~4 widening muls vs M31's ~2);
   degree-4 extension (needed for ~136-bit soundness) inherits all of it.
2. **The word gadgets mostly survive anyway** (the decisive point): field arithmetic is
   mod p, not mod 2^32 — u32 wraparound still needs carry + range check; bitwise
   AND/OR/XOR/shifts (the bulk of RV32 mixes) need limb decomposition or lookups in ANY
   prime field; MUL needs hi/lo split. One-word-per-FE only cleans up representation at
   rest and pure adds — maybe the cheaper half of a gadget overhead that is itself only
   ~20-40% of an RV32 AIR.
3. **Production precedent**: RISC0 launched on Goldilocks (holds u32 natively, cheap
   reduction) and migrated DOWN to BabyBear (31-bit); SP1/OpenVM chose 31-bit from the
   start. Everyone who ran this trade concluded the fast small field wins.
4. Engineering cost is a full fork regardless of prime size: field + SIMD backend +
   extension tower + proof format/channel/Merkle word layout + verifier + full security
   review + every downstream M31-tuned AIR.

## What would reopen the question

- Hardware with free 64-bit multiplies in the vector datapath (some GPU/FPGA targets).
- A measured limb/range-gadget fraction of a representative RV32-over-M31 AIR far above
  expectations (>~60%). Measure that number before ever revisiting this; expected ~25%.

Related: M61 = 2^61 − 1 (fully 2-adic p+1 = 2^61) is the natural big-field circle
candidate; same fork cost, same per-element throughput regression, nobody has shipped it.
