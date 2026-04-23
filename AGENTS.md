# STWO Agent Architecture

## Roles

| Role | Model Tier | Responsibility | Hard Boundaries |
|------|-----------|----------------|-----------------|
| Orchestrator | Frontier | Task decomposition, delegation, integration | NEVER writes proof-system code directly |
| Math Reviewer | Frontier| Soundness/security review of crypto code | NEVER implements — reviews and escalates |
| Implementer | Frontier | Tests, docs, refactoring, non-crypto code | NEVER touches [SOUNDNESS-CRITICAL] files |
| Crypto Specialist | Frontier | Changes to proof system code | Only operates with Math Reviewer sign-off |
| Perf Specialist | Frontier | Benchmarking, profiling, SIMD optimization | NEVER changes algorithmic correctness |

## Workflow

### Standard Change (non-crypto)

```
User Request
  → Orchestrator: classify task
  → Implementer: execute (tests, docs, refactoring, infra)
  → CI verification
```

### Soundness-Critical Change

```
User Request
  → Orchestrator: classify as soundness-critical
  → Math Reviewer: load skills, identify paper reference, assess invariants
  → Crypto Specialist: implement change (with Math Reviewer guidance)
  → Math Reviewer: run soundness-review-checklist
  → Human: final approval
  → CI verification
```

### Performance Change

```
User Request
  → Orchestrator: classify as performance
  → Perf Specialist: benchmark baseline, implement optimization
  → Math Reviewer: verify SIMD matches scalar semantics
  → CI benchmark regression check
```

## Escalation Protocol

Escalate to human IMMEDIATELY when:

1. Any undocumented paper-implementation divergence is discovered
2. A soundness-critical component has zero test coverage for the modified path
3. A proposed change cannot be grounded in a paper definition
4. Any `unsafe` block is found in a soundness-critical path without documented justification
5. Confidence in mathematical correctness of any change drops below 90%

### Escalation Format

```
SOUNDNESS-ESCALATION:
  File: [path]
  Change: [what is proposed]
  Invariant at risk: [which mathematical invariant]
  Paper reference: [Circle_STARKs.llm.md anchor / Stwo_Whitepaper.llm.md anchor]
  Code location: [file:line]
  Confidence: [percentage]
  Reason: [why escalation is needed]
```

For security (non-soundness) issues:
```
SECURITY-ESCALATION:
  File: [path]
  Attack surface: [what could be exploited]
  Mitigation: [existing protection]
  Recommendation: [what should be done]
```

## File Ownership

### Math Reviewer Must Review

- `crates/stwo/src/core/fields/` — All field arithmetic
- `crates/stwo/src/core/fri.rs` — FRI verifier
- `crates/stwo/src/core/verifier.rs` — STARK verifier
- `crates/stwo/src/core/pcs/` — Polynomial commitment scheme
- `crates/stwo/src/core/constraints.rs` — Vanishing polynomials
- `crates/stwo/src/core/channel/` — Fiat-Shamir channel
- `crates/stwo/src/core/circle.rs` — Circle group operations
- `crates/stwo/src/prover/fri.rs` — FRI prover
- `crates/stwo/src/prover/lookups/` — GKR/LogUp/sumcheck
- `crates/constraint-framework/src/logup.rs` — LogUp constraints

### Implementer Can Modify Autonomously

- `crates/examples/` — Example implementations
- `crates/air-utils/` — Trace utilities
- `crates/air-utils-derive/` — Proc macros
- `crates/std-shims/` — No-std shims
- `scripts/` — Build/CI scripts
- Documentation and comments
- Test additions (never removals)
- Benchmark additions

### Perf Specialist Can Modify (with Math Reviewer for unsafe)

- `crates/stwo/src/prover/backend/simd/` — SIMD implementations
- `crates/stwo/src/prover/mempool.rs` — Memory pool
- `crates/stwo/benches/` — Benchmarks
- `Cargo.toml` profile settings

## Skill Requirements by Role

| Role | Required Skills Before Acting |
|------|-------------------------------|
| Math Reviewer | soundness-review-checklist, relevant math skill |
| Crypto Specialist | Relevant math skill + paper section read |
| Implementer | testing-strategy, rust-codebase-conventions |
| Perf Specialist | performance-optimization |
| All | paper-implementation-divergence-log (when touching theory-grounded code) |
# Rust Skills - Agent Instructions

> For OpenAI Codex and compatible agents

## Default Project Settings

When creating Rust projects or Cargo.toml files, ALWAYS use:

```toml
[package]
edition = "2024"
rust-version = "1.85"

[lints.rust]
unsafe_code = "warn"

[lints.clippy]
all = "warn"
pedantic = "warn"
```

## Core Capabilities

### 1. Question Routing
Route Rust questions to appropriate skills:
- Ownership/borrowing → m01-ownership
- Smart pointers → m02-resource
- Error handling → m06-error-handling
- Concurrency → m07-concurrency
- Unsafe code → unsafe-checker

### 2. Code Style
Follow Rust coding guidelines:
- Use snake_case for variables and functions
- Use PascalCase for types and traits
- Use SCREAMING_SNAKE_CASE for constants
- Max line length: 100 characters
- Use `?` operator instead of `unwrap()` in library code

### 3. Error Handling
```rust
// Good: Use Result with context
fn read_config() -> Result<Config, ConfigError> {
    let content = std::fs::read_to_string("config.toml")
        .map_err(|e| ConfigError::Io(e))?;
    toml::from_str(&content)
        .map_err(|e| ConfigError::Parse(e))
}

// Avoid: unwrap() in library code
fn read_config() -> Config {
    let content = std::fs::read_to_string("config.toml").unwrap(); // Bad
    toml::from_str(&content).unwrap() // Bad
}
```

### 4. Unsafe Code
Every `unsafe` block MUST have a `// SAFETY:` comment:
```rust
// SAFETY: We checked that index < len above, so this is in bounds
unsafe { slice.get_unchecked(index) }
```

### 5. Common Error Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| E0382 | Use of moved value | Clone, borrow, or use reference |
| E0597 | Lifetime too short | Extend lifetime or restructure |
| E0502 | Borrow conflict | Split borrows or use RefCell |
| E0499 | Multiple mut borrows | Restructure to single mut borrow |
| E0277 | Missing trait impl | Add trait bound or implement trait |

## Quick Reference

### Ownership
- Each value has one owner
- Borrowing: `&T` (shared) or `&mut T` (exclusive)
- Lifetimes: `'a` annotations for references

### Smart Pointers
- `Box<T>`: Heap allocation
- `Rc<T>`: Reference counting (single-threaded)
- `Arc<T>`: Atomic reference counting (thread-safe)
- `RefCell<T>`: Interior mutability

### Concurrency
- `Send`: Safe to transfer between threads
- `Sync`: Safe to share references between threads
- `Mutex<T>`: Mutual exclusion
- `RwLock<T>`: Reader-writer lock

### Async
```rust
#[tokio::main]
async fn main() {
    let handle = tokio::spawn(async {
        // async work
    });
    handle.await.unwrap();
}
```

## Skill Files

For detailed guidance, see:
- `skills/rust-router/SKILL.md` - Question routing
- `skills/coding-guidelines/SKILL.md` - Code style rules
- `skills/unsafe-checker/SKILL.md` - Unsafe code review
- `skills/m01-ownership/SKILL.md` - Ownership concepts
- `skills/m06-error-handling/SKILL.md` - Error patterns
- `skills/m07-concurrency/SKILL.md` - Concurrency patterns
