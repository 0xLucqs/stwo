# Lessons

## 2026-03-30

- When a user says "client-side", do not assume browser or wasm. Ask or infer whether they mean native mobile, desktop local, or browser before steering the analysis.
- When a user asks for performance or memory measurements, do not default to debug builds. Use release-mode or the project's intended benchmarking profile unless the user explicitly wants debug behavior.
- When a user asks for a macOS memory report, prefer Xcode Instruments/xctrace over shell-only tools so the result includes category-level memory attribution instead of just peak counters.

## 2026-03-31

- When reporting Instruments memory numbers, do not mix metrics across templates. `Allocations`, `Activity Monitor`, and `/usr/bin/time -l` describe different things, so every comparison must state the tool and metric family explicitly.
- When the user asks to reduce peak memory, optimize against the peak metric first. Do not present improvements in average, tail, or alternate memory metrics as progress if the peak requirement got worse.
- When phase checkpoints and OS-level peak memory disagree, assume there is an uninstrumented transient. Re-profile with Instruments and then add finer-grained checkpoints around the suspected hot loop before changing architecture.
- On Cairo proofs, inspect the retained base-to-interaction bridge directly. A large share of the early peak can live in `InteractionClaimGenerator` payloads that duplicate lookup tuples across phases, even when PCS/FRI low-memory work is already in place.
- When compacting a retained Cairo generator, prefer removing duplicated preprocessed columns first. That is a better ROI than chasing unrelated prover internals, but it still must be validated with the same `Activity Monitor` peak metric because on-demand replay from preprocessed columns can sometimes worsen `Memory` even when it shrinks explicit buffers.
- When the user asks for simple probe logging to identify the last allocation site hit before a crash, do not build generic tracing infrastructure. Use direct pre-allocation `eprintln!` probes at the suspected call sites instead.
