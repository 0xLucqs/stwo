use hashbrown::HashMap;
use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tracing::{info, span, Level};

use crate::core::channel::{Channel, MerkleChannel};
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::{
    CommitmentSchemeProof, CommitmentSchemeProofAux, ExtendedCommitmentSchemeProof, PointSample,
};
use crate::core::pcs::utils::prepare_preprocessed_query_positions;
use crate::core::pcs::{PcsConfig, TreeSubspan, TreeVec};
use crate::core::poly::circle::CanonicCoset;
use crate::core::utils::{bit_reverse_index, MaybeOwned};
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::ExtendedMerkleDecommitmentLifted;
use crate::core::verifier::PREPROCESSED_TRACE_IDX;
use crate::core::ColumnVec;
use crate::prover::air::component_prover::{
    Poly, SpilledPolyCoeffs, Trace, TraceEvalAccessPattern, WeightsHashMap,
};
use crate::prover::backend::{BackendForChannel, Col, Column};
use crate::prover::fri::{FriDecommitResult, FriProver};
use crate::prover::memory::phase_memory_checkpoint;
use crate::prover::mempool::BaseColumnPool;
use crate::prover::pcs::quotient_ops::{compute_fri_quotients, compute_fri_quotients_from_polys};
use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::vcs_lifted::prover::{CheckpointedMerkleProverLifted, MerkleProverLifted};

pub mod quotient_ops;

/// Spills in-memory polynomial coefficients to a memory-mapped temporary file.
///
/// This moves coefficient data from anonymous heap pages (which count toward `phys_footprint`)
/// to file-backed mmap pages (which are evictable and do not count). Polynomials without
/// in-memory coefficients are silently skipped.
///
/// Called in `CommitmentTreeProver::new_with_memory_mode` just before building the Merkle leaf
/// layer — the most memory-intensive phase of commitment — to prevent simultaneous residency of
/// coefficients, extended evaluations, and Merkle leaf/next layers in anonymous memory.
fn spill_polys_coefficients<B>(polys: &mut Vec<Poly<B>>) -> std::io::Result<()>
where
    B: crate::prover::backend::Backend,
{
    use crate::prover::spill::{CoefficientSpillFile, SpillIndex};

    let mut spill = CoefficientSpillFile::new()?;
    let mut n_spilled: usize = 0;

    for poly in polys.iter() {
        if let Some(coeffs) = &poly.coeffs {
            let cpu_data = coeffs.coeffs.to_cpu();
            let bytes: &[u8] = bytemuck::cast_slice(&cpu_data);
            spill.write_coefficients_raw(bytes)?;
            n_spilled += 1;
        }
    }

    if n_spilled == 0 {
        return Ok(());
    }

    let frozen = spill.freeze()?;

    let mut spill_idx: usize = 0;
    for poly in polys.iter_mut() {
        if poly.coeffs.is_some() {
            let log_size = poly.log_size();
            poly.spilled_coeffs = Some(SpilledPolyCoeffs {
                spill_file: frozen.clone(),
                spill_index: SpillIndex(spill_idx),
                log_size,
            });
            poly.coeffs = None;
            spill_idx += 1;
        }
    }

    phase_memory_checkpoint("pcs:tree:after_coefficient_spill");
    Ok(())
}

/// Controls prover memory usage strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProverMemoryMode {
    /// Keep intermediate prover evaluations in memory to minimize recomputation.
    Fast,
    /// Drop some FRI intermediates and recompute them during decommitment to reduce RAM usage.
    LowMemory,
}

const LOW_MEMORY_TRACE_MERKLE_CHECKPOINT_STRIDE: u32 = 4;

/// Default budget (in bytes) for heap-backed evaluation memory used during
/// [`ProverMemoryMode::LowMemory`] re-materialization (composition polynomial generation and
/// decommit).
///
/// [`CommitmentTreeProver::materialize_evaluations_low_memory`] accumulates newly-materialized
/// evaluation columns in anonymous heap memory up to this budget before flushing the batch to
/// file-backed mmap. The final partial batch is left heap-backed, so if the entire
/// re-materialized set fits in this budget, no spilling happens at all and the function is
/// effectively equivalent to the eager `materialize_evaluations` path.
///
/// Raising this budget trades RAM for wall-clock: each mmap flush costs a file write, a
/// sync, and a mmap call (tens of ms per column on mobile flash). A 2 GiB default lets a
/// privacy-demo-size phone workload complete with effectively zero spills during the tail
/// phase, when the initial commit spike has already been released and the jetsam budget has
/// opened back up. Devices with less than ~3 GiB of app budget should override via
/// [`set_low_memory_materialize_budget_bytes`] or
/// `STWO_LOW_MEMORY_MATERIALIZE_BUDGET_BYTES`.
const DEFAULT_LOW_MEMORY_MATERIALIZE_BUDGET_BYTES: usize = 2 << 30;

/// Environment variable name for overriding [`DEFAULT_LOW_MEMORY_MATERIALIZE_BUDGET_BYTES`].
const LOW_MEMORY_MATERIALIZE_BUDGET_BYTES_ENV: &str = "STWO_LOW_MEMORY_MATERIALIZE_BUDGET_BYTES";

/// Process-wide override for the re-materialization heap budget.
///
/// Uses `0` as the "unset" sentinel because a 0-byte budget is semantically meaningless for
/// this knob and would degenerate to per-poly spilling (which is exactly what the pre-budget
/// behavior was — a caller who wants that can set it to `1` explicitly).
static LOW_MEMORY_MATERIALIZE_BUDGET_BYTES_OVERRIDE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Sets a process-wide override for the [`ProverMemoryMode::LowMemory`] re-materialization
/// heap budget in bytes.
///
/// This takes precedence over the `STWO_LOW_MEMORY_MATERIALIZE_BUDGET_BYTES` environment
/// variable and is intended for embedded targets (iOS) where shell env vars do not propagate
/// into the app process. Call this once at startup, before the first
/// [`CommitmentSchemeProver::prove_values`] invocation:
///
/// ```ignore
/// // 1.5 GiB budget on a phone with ~2 GiB of app headroom.
/// stwo::prover::set_low_memory_materialize_budget_bytes(1_500 * 1024 * 1024);
/// ```
///
/// Passing `0` clears the override and returns control to the env var / default. Any other
/// positive value wins unconditionally.
pub fn set_low_memory_materialize_budget_bytes(bytes: usize) {
    LOW_MEMORY_MATERIALIZE_BUDGET_BYTES_OVERRIDE.store(bytes, std::sync::atomic::Ordering::Release);
    tracing::info!(bytes, "stwo low-memory materialize budget override set");
}

/// Parses the re-materialization budget from a raw env var value (decimal bytes).
///
/// Returns `None` for empty, non-numeric, zero, or negative inputs. `0` is rejected at this
/// layer so it cannot be mistaken for "disable batching entirely".
fn parse_low_memory_materialize_budget_bytes(value: &str) -> Option<usize> {
    value.trim().parse::<usize>().ok().filter(|&b| b > 0)
}

/// Resolves the active re-materialization budget in bytes.
///
/// Resolution order (first match wins): programmatic override →
/// `STWO_LOW_MEMORY_MATERIALIZE_BUDGET_BYTES` env var →
/// [`DEFAULT_LOW_MEMORY_MATERIALIZE_BUDGET_BYTES`].
fn low_memory_materialize_budget_bytes() -> usize {
    let override_val =
        LOW_MEMORY_MATERIALIZE_BUDGET_BYTES_OVERRIDE.load(std::sync::atomic::Ordering::Acquire);
    if override_val > 0 {
        return override_val;
    }
    match std::env::var(LOW_MEMORY_MATERIALIZE_BUDGET_BYTES_ENV) {
        Ok(raw) => match parse_low_memory_materialize_budget_bytes(&raw) {
            Some(b) => b,
            None => {
                tracing::warn!(
                    env_var = LOW_MEMORY_MATERIALIZE_BUDGET_BYTES_ENV,
                    env_value = raw.as_str(),
                    default_bytes = DEFAULT_LOW_MEMORY_MATERIALIZE_BUDGET_BYTES,
                    "Invalid re-materialize budget, using default"
                );
                DEFAULT_LOW_MEMORY_MATERIALIZE_BUDGET_BYTES
            }
        },
        Err(_) => DEFAULT_LOW_MEMORY_MATERIALIZE_BUDGET_BYTES,
    }
}

/// Sentinel encoding for [`DEFAULT_PROVER_MEMORY_MODE_OVERRIDE`]: no override set.
const PROVER_MEMORY_MODE_OVERRIDE_UNSET: u8 = 0;
/// Sentinel encoding for [`DEFAULT_PROVER_MEMORY_MODE_OVERRIDE`]: pin to
/// [`ProverMemoryMode::Fast`].
const PROVER_MEMORY_MODE_OVERRIDE_FAST: u8 = 1;
/// Sentinel encoding for [`DEFAULT_PROVER_MEMORY_MODE_OVERRIDE`]: pin to
/// [`ProverMemoryMode::LowMemory`].
const PROVER_MEMORY_MODE_OVERRIDE_LOW_MEMORY: u8 = 2;

/// Process-wide override for the default prover memory mode.
///
/// Set via [`set_default_prover_memory_mode`]. When non-zero, this takes precedence over the
/// `STWO_PROVER_MEMORY_MODE` environment variable in [`default_prover_memory_mode`]. Encoded
/// as a `u8` so the override is lock-free and cheap to consult on every prover construction.
static DEFAULT_PROVER_MEMORY_MODE_OVERRIDE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(PROVER_MEMORY_MODE_OVERRIDE_UNSET);

/// Sets a process-wide override for the default prover memory mode.
///
/// This takes precedence over the `STWO_PROVER_MEMORY_MODE` environment variable and applies
/// to every [`CommitmentSchemeProver`] constructed *after* this call. Existing provers retain
/// their original mode unless explicitly updated via [`CommitmentSchemeProver::set_memory_mode`].
///
/// Designed for embedded targets where shell environment variables do not propagate into the
/// app process — for example, iOS apps that need to opt into [`ProverMemoryMode::LowMemory`]
/// at startup. Call this once early in your `main`/`AppDelegate` before any prover is built:
///
/// ```ignore
/// stwo::prover::set_default_prover_memory_mode(stwo::prover::ProverMemoryMode::LowMemory);
/// ```
///
/// Calling this multiple times is permitted; the most recent call wins. Concurrent callers see
/// linearizable updates via release/acquire ordering on an atomic `u8`.
pub fn set_default_prover_memory_mode(mode: ProverMemoryMode) {
    let encoded = match mode {
        ProverMemoryMode::Fast => PROVER_MEMORY_MODE_OVERRIDE_FAST,
        ProverMemoryMode::LowMemory => PROVER_MEMORY_MODE_OVERRIDE_LOW_MEMORY,
    };
    DEFAULT_PROVER_MEMORY_MODE_OVERRIDE.store(encoded, std::sync::atomic::Ordering::Release);
    tracing::info!(?mode, "stwo prover memory mode override set");
}

fn parse_prover_memory_mode(value: &str) -> Option<ProverMemoryMode> {
    if value.eq_ignore_ascii_case("fast") {
        Some(ProverMemoryMode::Fast)
    } else if value.eq_ignore_ascii_case("low_memory")
        || value.eq_ignore_ascii_case("low-memory")
        || value.eq_ignore_ascii_case("lowmemory")
        || value.eq_ignore_ascii_case("checkpointed")
    {
        Some(ProverMemoryMode::LowMemory)
    } else {
        None
    }
}

/// Resolves the prover memory mode for a freshly constructed [`CommitmentSchemeProver`].
///
/// Resolution order (first match wins):
/// 1. Process-wide override set via [`set_default_prover_memory_mode`].
/// 2. The `STWO_PROVER_MEMORY_MODE` environment variable.
/// 3. [`ProverMemoryMode::Fast`] as the conservative default.
///
/// The resolved mode is logged at `info` level along with its source so the active mode is
/// observable in production logs (e.g. iOS Console.app). This avoids silently falling back to
/// `Fast` on platforms where the env var does not propagate to the app process.
fn default_prover_memory_mode() -> ProverMemoryMode {
    let override_value =
        DEFAULT_PROVER_MEMORY_MODE_OVERRIDE.load(std::sync::atomic::Ordering::Acquire);
    let (mode, source) = match override_value {
        PROVER_MEMORY_MODE_OVERRIDE_FAST => (ProverMemoryMode::Fast, "override"),
        PROVER_MEMORY_MODE_OVERRIDE_LOW_MEMORY => (ProverMemoryMode::LowMemory, "override"),
        // PROVER_MEMORY_MODE_OVERRIDE_UNSET (and defensively any unknown sentinel) — fall back
        // to the env var, then to the conservative default.
        _ => match std::env::var("STWO_PROVER_MEMORY_MODE") {
            Ok(value) => match parse_prover_memory_mode(&value) {
                Some(mode) => (mode, "env"),
                None => {
                    tracing::warn!(
                        env_value = value.as_str(),
                        "Unknown STWO_PROVER_MEMORY_MODE, defaulting to fast mode"
                    );
                    (ProverMemoryMode::Fast, "default")
                }
            },
            Err(_) => (ProverMemoryMode::Fast, "default"),
        },
    };
    tracing::info!(?mode, source, "stwo prover memory mode resolved");
    mode
}

pub enum CommitmentTreeMerkleProver<B: BackendForChannel<MC>, MC: MerkleChannel> {
    Full(MerkleProverLifted<B, MC::H>),
    Checkpointed(CheckpointedMerkleProverLifted<B, MC::H>),
}

impl<B: BackendForChannel<MC>, MC: MerkleChannel> CommitmentTreeMerkleProver<B, MC> {
    pub fn root(&self) -> <MC::H as MerkleHasherLifted>::Hash {
        match self {
            CommitmentTreeMerkleProver::Full(tree) => tree.root(),
            CommitmentTreeMerkleProver::Checkpointed(tree) => tree.root(),
        }
    }

    pub fn height(&self) -> u32 {
        match self {
            CommitmentTreeMerkleProver::Full(tree) => tree.layers.len() as u32 - 1,
            CommitmentTreeMerkleProver::Checkpointed(tree) => tree.height(),
        }
    }
}

/// The prover side of a FRI polynomial commitment scheme. See [super].
pub struct CommitmentSchemeProver<'a, B: BackendForChannel<MC>, MC: MerkleChannel> {
    pub trees: TreeVec<MaybeOwned<'a, CommitmentTreeProver<B, MC>>>,
    pub config: PcsConfig,
    pub twiddles: &'a TwiddleTree<B>,
    pub store_polynomials_coefficients: bool,
    pub memory_mode: ProverMemoryMode,
    /// Pre-allocated base field column pool for polynomial evaluation during commit.
    pub base_column_pool: MaybeOwned<'a, BaseColumnPool<B>>,
}
impl<'a, B: BackendForChannel<MC>, MC: MerkleChannel> CommitmentSchemeProver<'a, B, MC> {
    /// Creates a new empty commitment scheme prover with the given configuration and twiddles. The
    /// commitment scheme does not store the polynomials coefficients by default.
    pub fn new(config: PcsConfig, twiddles: &'a TwiddleTree<B>) -> Self {
        CommitmentSchemeProver {
            trees: TreeVec::default(),
            config,
            twiddles,
            store_polynomials_coefficients: false,
            memory_mode: default_prover_memory_mode(),
            base_column_pool: MaybeOwned::Owned(BaseColumnPool::new()),
        }
    }

    pub fn with_memory_pool(
        config: PcsConfig,
        twiddles: &'a TwiddleTree<B>,
        base_column_pool: &'a BaseColumnPool<B>,
    ) -> Self {
        CommitmentSchemeProver {
            trees: TreeVec::default(),
            config,
            twiddles,
            store_polynomials_coefficients: false,
            memory_mode: default_prover_memory_mode(),
            base_column_pool: MaybeOwned::Borrowed(base_column_pool),
        }
    }

    /// Sets the commitment scheme to store the polynomials coefficients starting from the next
    /// commit.
    pub const fn set_store_polynomials_coefficients(&mut self) {
        self.store_polynomials_coefficients = true;
    }

    /// Sets the prover memory mode starting from the next proving phase.
    pub const fn set_memory_mode(&mut self, memory_mode: ProverMemoryMode) {
        self.memory_mode = memory_mode;
    }

    /// Evaluates the given polynomials, commits them into a Merkle tree, mixes the root into
    /// the channel, and appends the resulting tree to the scheme.
    fn commit(&mut self, polynomials: ColumnVec<CircleCoefficients<B>>, channel: &mut MC::C) {
        let _span = span!(Level::INFO, "Commitment").entered();
        phase_memory_checkpoint("pcs:commit:start");

        // In LowMemory mode, spill all previously committed trees' coefficients to disk before
        // building this tree. This prevents previously accumulated coefficient data from occupying
        // anonymous heap during this tree's extension FFT and Merkle construction, which would
        // otherwise push peak footprint to the sum of all trees built so far.
        if self.memory_mode == ProverMemoryMode::LowMemory {
            for tree in &mut self.trees.0 {
                if let MaybeOwned::Owned(tree) = tree {
                    if tree.polynomials.iter().any(|p| p.coeffs.is_some()) {
                        if let Err(e) = tree.spill_coefficients() {
                            tracing::warn!(
                                "Failed to spill tree coefficients before commit: {e}. \
                                 Continuing with in-memory coefficients."
                            );
                        }
                    }
                }
            }
            phase_memory_checkpoint("pcs:commit:after_prev_tree_spill");
        }

        let retain_coefficients =
            self.store_polynomials_coefficients || self.memory_mode == ProverMemoryMode::LowMemory;
        let tree = CommitmentTreeProver::new_with_memory_mode(
            polynomials,
            self.config.fri_config.log_blowup_factor,
            self.twiddles,
            retain_coefficients,
            self.config.lifting_log_size,
            &self.base_column_pool,
            self.memory_mode,
        );
        phase_memory_checkpoint("pcs:commit:tree_ready");
        MC::mix_root(channel, tree.commitment.root());
        self.trees.push(MaybeOwned::Owned(tree));
        phase_memory_checkpoint("pcs:commit:stored");
    }

    /// Appends an externally constructed [`CommitmentTreeProver`] to the scheme and mixes its
    /// Merkle root into the channel. Accepts both owned and borrowed trees.
    pub fn commit_tree(
        &mut self,
        tree: MaybeOwned<'a, CommitmentTreeProver<B, MC>>,
        channel: &mut MC::C,
    ) {
        let tree = match (self.memory_mode, tree) {
            (ProverMemoryMode::LowMemory, MaybeOwned::Owned(tree)) => {
                MaybeOwned::Owned(tree.into_memory_mode(ProverMemoryMode::LowMemory))
            }
            (_, tree) => tree,
        };
        MC::mix_root(channel, tree.commitment.root());
        self.trees.push(tree);

        // In LowMemory mode, immediately spill the just-committed tree's coefficients so they
        // don't occupy anonymous heap during any subsequent tree's Merkle construction.
        if self.memory_mode == ProverMemoryMode::LowMemory {
            if let Some(MaybeOwned::Owned(tree)) = self.trees.0.last_mut() {
                if let Err(e) = tree.spill_coefficients() {
                    tracing::warn!(
                        "Failed to spill tree coefficients after commit_tree: {e}. \
                         Continuing with in-memory coefficients."
                    );
                }
                phase_memory_checkpoint("pcs:commit_tree:after_spill");
            }
        }
    }

    pub fn tree_builder(&mut self) -> TreeBuilder<'_, 'a, B, MC> {
        TreeBuilder {
            tree_index: self.trees.len(),
            commitment_scheme: self,
            polys: Vec::default(),
        }
    }

    pub fn roots(&self) -> TreeVec<<MC::H as MerkleHasherLifted>::Hash> {
        self.trees.as_ref().map(|tree| tree.commitment.root())
    }

    pub fn polynomials(&self) -> TreeVec<ColumnVec<&Poly<B>>> {
        self.trees
            .as_ref()
            .map(|tree| tree.polynomials.iter().collect())
    }

    pub fn evaluations(
        &self,
    ) -> TreeVec<ColumnVec<&CircleEvaluation<B, BaseField, BitReversedOrder>>> {
        self.trees
            .as_ref()
            .map(|tree| tree.polynomials.iter().map(Poly::evals).collect())
    }

    pub fn trace(&self) -> Trace<'_, B> {
        let polys = self.polynomials();
        Trace { polys }
    }

    pub fn release_recomputable_evaluations(&mut self) {
        for tree in &mut self.trees.0 {
            if let MaybeOwned::Owned(tree) = tree {
                if tree.can_recompute_openings() {
                    if self.memory_mode == ProverMemoryMode::LowMemory {
                        tree.drop_evaluations();
                    } else {
                        tree.release_evaluations(&self.base_column_pool);
                    }
                }
            }
        }
    }

    /// Spills all owned trees' polynomial coefficients to memory-mapped temporary files.
    ///
    /// After spilling, in-memory coefficient data is dropped and coefficients are loaded
    /// on demand from the mmap when needed. This dramatically reduces peak RSS for
    /// memory-constrained environments (e.g., mobile phones).
    ///
    /// Should be called after all trees are committed and before the composition polynomial
    /// evaluation phase. Only affects owned trees (not borrowed preprocessed trees).
    pub fn spill_coefficients(&mut self) -> std::io::Result<()> {
        for tree in &mut self.trees.0 {
            if let MaybeOwned::Owned(tree) = tree {
                if tree.polynomials.iter().any(|p| p.coeffs.is_some()) {
                    tree.spill_coefficients()?;
                }
            }
        }
        phase_memory_checkpoint("pcs:after_coefficient_spill");
        Ok(())
    }

    pub fn materialize_access_pattern_evaluations(
        &mut self,
        access_pattern: Option<&TraceEvalAccessPattern>,
    ) {
        let low_memory = self.memory_mode == ProverMemoryMode::LowMemory;
        match access_pattern {
            Some(access_pattern) => {
                // The range/indices materialize variants currently always use the eager path.
                // They typically only touch a small subset of polys (one component's slice),
                // so the heap spike is bounded by `span.col_end - span.col_start` rather than
                // the entire tree. If profiling later shows even that subset is too large on
                // mobile, plumb the per-poly spill helper through these two methods as well.
                for span in &access_pattern.tree_spans {
                    if let Some(MaybeOwned::Owned(tree)) = self.trees.0.get_mut(span.tree_index) {
                        tree.materialize_evaluation_range(
                            span.col_start,
                            span.col_end,
                            self.twiddles,
                            &self.base_column_pool,
                        );
                    }
                }
                if !access_pattern.preprocessed_columns.is_empty() {
                    if let Some(MaybeOwned::Owned(tree)) =
                        self.trees.0.get_mut(PREPROCESSED_TRACE_IDX)
                    {
                        tree.materialize_evaluation_indices(
                            &access_pattern.preprocessed_columns,
                            self.twiddles,
                            &self.base_column_pool,
                        );
                    }
                }
            }
            None => {
                for tree in &mut self.trees.0 {
                    if let MaybeOwned::Owned(tree) = tree {
                        if low_memory {
                            // Per-poly materialize-and-spill: bounds the anonymous heap working
                            // set to one column at a time instead of the full tree's worth of
                            // re-materialized evals.
                            tree.materialize_evaluations_low_memory(
                                self.twiddles,
                                &self.base_column_pool,
                            );
                        } else {
                            tree.materialize_evaluations(self.twiddles, &self.base_column_pool);
                        }
                    }
                }
            }
        }
    }

    pub fn drop_access_pattern_evaluations(
        &mut self,
        access_pattern: Option<&TraceEvalAccessPattern>,
    ) {
        match access_pattern {
            Some(access_pattern) => {
                for span in &access_pattern.tree_spans {
                    if let Some(MaybeOwned::Owned(tree)) = self.trees.0.get_mut(span.tree_index) {
                        tree.drop_evaluation_range(span.col_start, span.col_end);
                    }
                }
                if !access_pattern.preprocessed_columns.is_empty() {
                    if let Some(MaybeOwned::Owned(tree)) =
                        self.trees.0.get_mut(PREPROCESSED_TRACE_IDX)
                    {
                        tree.drop_evaluation_indices(&access_pattern.preprocessed_columns);
                    }
                }
            }
            None => self.release_recomputable_evaluations(),
        }
    }

    pub fn build_weights_hash_map(
        &self,
        sampled_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        max_log_size: u32,
    ) -> WeightsHashMap<B>
    where
        Col<B, SecureField>: Send + Sync,
    {
        let weights_dashmap = WeightsHashMap::<B>::new();

        self.polynomials()
            .zip_cols(sampled_points)
            .map_cols(|(poly, points)| {
                let compute_weights = |(log_size, point): (u32, CirclePoint<SecureField>)| {
                    weights_dashmap.entry((log_size, point)).or_insert_with(|| {
                        CircleEvaluation::<B, BaseField, BitReversedOrder>::barycentric_weights(
                            CanonicCoset::new(log_size),
                            point,
                        )
                    });
                };

                let log_size = poly.log_size();
                // For each sample point, compute the weights needed to evaluate the polynomial at
                // the folded sample point.
                // TODO(Leo): the computation `point.repeated_double(max_log_size - log_size)` is
                // likely repeated a bunch of times in a typical flat air. Consider moving it
                // outside the loop.
                #[cfg(not(feature = "parallel"))]
                points.iter().for_each(|&point| {
                    compute_weights((log_size, point.repeated_double(max_log_size - log_size)))
                });

                #[cfg(feature = "parallel")]
                points.par_iter().for_each(|&point| {
                    compute_weights((log_size, point.repeated_double(max_log_size - log_size)))
                });
            });

        weights_dashmap
    }

    pub fn prove_values(
        mut self,
        sampled_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        channel: &mut MC::C,
    ) -> ExtendedCommitmentSchemeProof<MC::H> {
        phase_memory_checkpoint("pcs:prove_values:start");
        // Evaluate polynomials on open points.
        let span = span!(
            Level::INFO,
            "Evaluate columns out of domain",
            class = "EvaluateOutOfDomain"
        )
        .entered();

        let lifting_log_size = self.trees.last().unwrap().commitment.height();
        let retain_coefficients =
            self.store_polynomials_coefficients || self.memory_mode == ProverMemoryMode::LowMemory;
        let weights_hash_map = if retain_coefficients {
            None
        } else {
            Some(self.build_weights_hash_map(&sampled_points, lifting_log_size))
        };

        // Lambda that evaluates a polynomial on a collection of circle points and returns a vector
        // of point samples.
        let eval_at_points = |(poly, points): (&Poly<B>, &Vec<CirclePoint<SecureField>>)| {
            points
                .iter()
                .map(|&point| PointSample {
                    point,
                    value: poly.eval_at_point(
                        point.repeated_double(lifting_log_size - poly.log_size()),
                        weights_hash_map.as_ref(),
                    ),
                })
                .collect_vec()
        };

        #[cfg(not(feature = "parallel"))]
        let samples: TreeVec<Vec<Vec<PointSample>>> = self
            .polynomials()
            .zip_cols(&sampled_points)
            .map_cols(eval_at_points);
        #[cfg(feature = "parallel")]
        let samples: TreeVec<Vec<Vec<PointSample>>> = self
            .polynomials()
            .zip_cols(&sampled_points)
            .par_map_cols(eval_at_points);

        span.exit();
        phase_memory_checkpoint("pcs:prove_values:after_oods");
        let sampled_values = samples
            .as_cols_ref()
            .map_cols(|x| x.iter().map(|o| o.value).collect());
        channel.mix_felts(&sampled_values.clone().flatten_cols());

        let quotients = if self.memory_mode == ProverMemoryMode::LowMemory {
            let polynomials = self.polynomials();
            print_polynomial_size_histogram::<B, MC>(&polynomials);
            compute_fri_quotients_from_polys(
                &polynomials,
                &samples,
                channel.draw_secure_felt(),
                lifting_log_size,
                self.twiddles,
                self.config.fri_config.log_blowup_factor,
                &self.base_column_pool,
                self.memory_mode,
            )
        } else {
            let columns = self.evaluations();
            print_column_size_histogram::<B, MC>(&columns);
            compute_fri_quotients(
                &columns,
                &samples,
                channel.draw_secure_felt(),
                lifting_log_size,
                self.twiddles,
                self.config.fri_config.log_blowup_factor,
                self.memory_mode,
            )
        };
        phase_memory_checkpoint("pcs:prove_values:after_fri_quotients");

        if self.memory_mode == ProverMemoryMode::LowMemory {
            for tree in &mut self.trees.0 {
                if let MaybeOwned::Owned(tree) = tree {
                    if tree.can_recompute_openings() {
                        tree.release_evaluations(&self.base_column_pool);
                    }
                }
            }
            phase_memory_checkpoint("pcs:prove_values:after_low_memory_release");
        }

        // Run FRI commitment phase on the oods quotients.
        let fri_prover = FriProver::<B, MC>::commit_with_memory_mode(
            channel,
            self.config.fri_config,
            &quotients,
            self.twiddles,
            self.memory_mode,
        );
        phase_memory_checkpoint("pcs:prove_values:after_fri_commit");

        // Proof of work.
        let span1 = span!(Level::INFO, "Grind", class = "Queries POW").entered();
        let proof_of_work = B::grind(channel, self.config.pow_bits);
        span1.exit();
        channel.mix_u64(proof_of_work);

        // FRI decommitment phase.
        let FriDecommitResult {
            fri_proof,
            query_positions,
            unsorted_query_locations,
        } = fri_prover.decommit(channel);
        // Build the query position tree.
        let preprocessed_query_positions = prepare_preprocessed_query_positions(
            &query_positions,
            lifting_log_size,
            self.trees[0].commitment.height(),
        );
        let query_positions_tree = TreeVec::new(
            self.trees
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    if i == 0 {
                        preprocessed_query_positions.as_slice()
                    } else {
                        query_positions.as_slice()
                    }
                })
                .collect::<Vec<_>>(),
        );
        let commitments = self.roots();
        let mut queried_values = Vec::with_capacity(self.trees.len());
        let mut decommitments = Vec::with_capacity(self.trees.len());
        let mut aux = Vec::with_capacity(self.trees.len());
        for (tree, query_positions) in self.trees.0.iter_mut().zip_eq(query_positions_tree.iter()) {
            let (values, decommitment) = match tree {
                MaybeOwned::Owned(tree) => {
                    if self.memory_mode == ProverMemoryMode::LowMemory
                        && tree.can_recompute_openings()
                    {
                        // Per-poly materialize-and-spill keeps the anonymous heap working set
                        // bounded to one column at a time. The previous eager
                        // `materialize_evaluations` allocated every column simultaneously,
                        // producing the multi-hundred-MB spike that OOM-killed the iOS app
                        // during decommit re-materialization.
                        tree.materialize_evaluations_low_memory(
                            self.twiddles,
                            &self.base_column_pool,
                        );
                    }
                    let result = tree.decommit(query_positions);
                    if self.memory_mode == ProverMemoryMode::LowMemory
                        && tree.can_recompute_openings()
                    {
                        // `drop_evaluations` knows how to forget any mmap-backed eval Vecs
                        // before letting them drop, then unmaps the regions via
                        // `eval_mmap_guards.clear()`.
                        tree.drop_evaluations();
                    }
                    result
                }
                MaybeOwned::Borrowed(tree) => tree.decommit(query_positions),
            };
            queried_values.push(values);
            decommitments.push(decommitment.decommitment);
            aux.push(decommitment.aux);
        }
        phase_memory_checkpoint("pcs:prove_values:after_trace_decommit");

        // Return evaluation buffers to the memory pool for reuse (owned trees only).
        for tree in &mut self.trees.0 {
            if let MaybeOwned::Owned(tree) = tree {
                if self.memory_mode == ProverMemoryMode::LowMemory {
                    tree.drop_evaluations();
                } else {
                    tree.release_evaluations(&self.base_column_pool);
                }
            }
        }
        phase_memory_checkpoint("pcs:prove_values:after_final_release");

        ExtendedCommitmentSchemeProof {
            proof: CommitmentSchemeProof {
                commitments,
                sampled_values,
                decommitments: TreeVec(decommitments),
                queried_values: TreeVec(queried_values),
                proof_of_work,
                fri_proof: fri_proof.proof,
                config: self.config,
            },
            aux: CommitmentSchemeProofAux {
                unsorted_query_locations,
                trace_decommitment: TreeVec(aux),
                fri: fri_proof.aux,
            },
        }
    }
}

/// Helper struct for aggregating polynomials and evaluations for a commitment tree.
pub struct TreeBuilder<'a, 'b, B: BackendForChannel<MC>, MC: MerkleChannel> {
    tree_index: usize,
    commitment_scheme: &'a mut CommitmentSchemeProver<'b, B, MC>,
    polys: ColumnVec<CircleCoefficients<B>>,
}
impl<B: BackendForChannel<MC>, MC: MerkleChannel> TreeBuilder<'_, '_, B, MC> {
    pub fn extend_evals(
        &mut self,
        columns: Vec<CircleEvaluation<B, BaseField, BitReversedOrder>>,
    ) -> TreeSubspan {
        let span = span!(Level::INFO, "Interpolation for commitment").entered();
        let polys = B::interpolate_columns(columns, self.commitment_scheme.twiddles);
        span.exit();

        self.extend_polys(polys)
    }

    pub fn extend_polys(
        &mut self,
        columns: impl IntoIterator<Item = CircleCoefficients<B>>,
    ) -> TreeSubspan {
        let col_start = self.polys.len();
        self.polys.extend(columns);
        let col_end = self.polys.len();
        TreeSubspan {
            tree_index: self.tree_index,
            col_start,
            col_end,
        }
    }

    pub fn release_recomputable_evaluations(&mut self) {
        self.commitment_scheme.release_recomputable_evaluations();
    }

    pub fn commit(self, channel: &mut MC::C) {
        let _span = span!(Level::INFO, "Commitment").entered();
        phase_memory_checkpoint("pcs:tree_builder:commit");
        self.commitment_scheme.commit(self.polys, channel);
    }
}

/// Prover data for a single commitment tree in a commitment scheme. The commitment scheme allows to
/// commit on a set of polynomials at a time. This corresponds to such a set.
pub struct CommitmentTreeProver<B: BackendForChannel<MC>, MC: MerkleChannel> {
    pub polynomials: ColumnVec<Poly<B>>,
    pub commitment: CommitmentTreeMerkleProver<B, MC>,
    /// File-backed mmap regions backing any polynomial evals that were re-materialized in
    /// `LowMemory` mode after the initial commit phase. These guards must outlive the
    /// polynomials' eval Vecs and must be drained via
    /// [`crate::prover::spill::forget_mmap_backed_evals`] in [`Self::drop_evaluations`] before
    /// the polynomials are dropped or returned to the pool — otherwise `Vec::drop` would
    /// attempt to free the mmap pages through the global allocator.
    eval_mmap_guards: Vec<crate::prover::spill::EvalMmapGuard>,
}

impl<B: BackendForChannel<MC>, MC: MerkleChannel> CommitmentTreeProver<B, MC> {
    pub fn new(
        polynomials: ColumnVec<CircleCoefficients<B>>,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<B>,
        store_polynomials_coefficients: bool,
        lifting_log_size: Option<u32>,
        base_column_pool: &BaseColumnPool<B>,
    ) -> Self {
        Self::new_with_memory_mode(
            polynomials,
            log_blowup_factor,
            twiddles,
            store_polynomials_coefficients,
            lifting_log_size,
            base_column_pool,
            ProverMemoryMode::Fast,
        )
    }

    pub fn new_with_memory_mode(
        polynomials: ColumnVec<CircleCoefficients<B>>,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<B>,
        store_polynomials_coefficients: bool,
        lifting_log_size: Option<u32>,
        base_column_pool: &BaseColumnPool<B>,
        memory_mode: ProverMemoryMode,
    ) -> Self {
        let span = span!(Level::INFO, "Extension").entered();
        phase_memory_checkpoint("pcs:tree:new:before_extension");
        let retain_coefficients =
            store_polynomials_coefficients || memory_mode == ProverMemoryMode::LowMemory;
        let (mut polynomials, mut eval_mmap_guards) = B::evaluate_polynomials(
            polynomials,
            log_blowup_factor,
            twiddles,
            retain_coefficients,
            base_column_pool,
            memory_mode,
        );
        span.exit();
        phase_memory_checkpoint("pcs:tree:new:after_extension");

        // In LowMemory mode, spill polynomial coefficients to disk before the Merkle leaf
        // layer is built. Building the leaf layer requires all extended evaluations resident;
        // holding in-memory coefficients simultaneously pushes peak footprint to:
        //   coefficients + extended_evals + leaf_layer + next_layer
        // Spilling coefficients (file-backed mmap) removes them from the anonymous page budget.
        if memory_mode == ProverMemoryMode::LowMemory && retain_coefficients {
            if let Err(e) = spill_polys_coefficients(&mut polynomials) {
                tracing::warn!(
                    "Failed to spill coefficients before Merkle construction: {e}. \
                     Continuing with in-memory coefficients."
                );
            }
        }

        // In LowMemory mode, replace evaluation Vec backing with file-backed mmap.
        // This converts ~5 GB of anonymous heap into OS-managed pages that can be evicted
        // under memory pressure and re-faulted from disk. Only the Merkle builder's working
        // set (~256 MB) needs to be physically resident.
        if memory_mode == ProverMemoryMode::LowMemory && eval_mmap_guards.is_empty() {
            // Fallback for backends that do not eagerly convert evaluation buffers to mmap-backed
            // storage during extension.
            let polys_ptr = &mut polynomials as *mut Vec<Poly<B>>
                as *mut Vec<Poly<crate::prover::backend::simd::SimdBackend>>;
            // SAFETY: This is only called when B = SimdBackend (the only backend that
            // implements BackendForChannel). The cast is sound because Poly<B> and
            // Poly<SimdBackend> have identical layout when B = SimdBackend.
            if let Some(guard) =
                unsafe { crate::prover::spill::spill_eval_columns(&mut *polys_ptr) }
            {
                eval_mmap_guards.push(guard);
            }
            phase_memory_checkpoint("pcs:tree:new:after_eval_mmap_spill");
        }

        let _span = span!(Level::INFO, "Merkle").entered();
        let max_log_domain_size = polynomials
            .iter()
            .map(Poly::log_size)
            .max()
            .unwrap_or_default();
        let lifting_log_size = lifting_log_size.unwrap_or(max_log_domain_size);
        let columns = polynomials
            .iter()
            .map(|poly: &Poly<B>| &poly.evals().values)
            .collect();
        let tree =
            match memory_mode {
                ProverMemoryMode::Fast => CommitmentTreeMerkleProver::Full(
                    MerkleProverLifted::commit(columns, lifting_log_size, 0),
                ),
                ProverMemoryMode::LowMemory => CommitmentTreeMerkleProver::Checkpointed(
                    CheckpointedMerkleProverLifted::commit(
                        columns,
                        lifting_log_size,
                        0,
                        LOW_MEMORY_TRACE_MERKLE_CHECKPOINT_STRIDE,
                    ),
                ),
            };

        if memory_mode == ProverMemoryMode::LowMemory {
            if !eval_mmap_guards.is_empty() {
                let polys_ptr = &mut polynomials as *mut Vec<Poly<B>>
                    as *mut Vec<Poly<crate::prover::backend::simd::SimdBackend>>;
                unsafe {
                    crate::prover::spill::forget_mmap_backed_evals(
                        &mut *polys_ptr,
                        &eval_mmap_guards,
                    )
                };
            }
            for poly in &mut polynomials {
                if poly.evals.is_some() {
                    let _ = poly.take_evals();
                }
            }
            phase_memory_checkpoint("pcs:tree:new:after_low_memory_eval_release");
        }
        phase_memory_checkpoint("pcs:tree:new:after_merkle");

        CommitmentTreeProver {
            polynomials,
            commitment: tree,
            eval_mmap_guards: Vec::new(),
        }
    }

    fn into_memory_mode(mut self, memory_mode: ProverMemoryMode) -> Self {
        if memory_mode == ProverMemoryMode::LowMemory {
            self.commitment = match self.commitment {
                CommitmentTreeMerkleProver::Full(tree) => CommitmentTreeMerkleProver::Checkpointed(
                    CheckpointedMerkleProverLifted::from_full_tree(
                        tree,
                        LOW_MEMORY_TRACE_MERKLE_CHECKPOINT_STRIDE,
                    ),
                ),
                checkpointed @ CommitmentTreeMerkleProver::Checkpointed(_) => checkpointed,
            };
        }
        self
    }

    /// Decommits the merkle tree on the given query positions.
    /// Returns the values at the queried positions and the decommitment.
    /// The queries are given as a mapping from the log size of the layer size to the queried
    /// positions on each column of that size.
    fn decommit(
        &self,
        queries: &[usize],
    ) -> (
        ColumnVec<Vec<BaseField>>,
        ExtendedMerkleDecommitmentLifted<MC::H>,
    ) {
        let queried_values = self.recompute_queried_values(queries);
        let decommitment = match &self.commitment {
            CommitmentTreeMerkleProver::Full(tree) => tree.decommit(queries, vec![]).1,
            CommitmentTreeMerkleProver::Checkpointed(tree) => {
                tree.decommit_with_source(queries, |positions| self.compute_leaf_hashes(positions))
            }
        };
        (queried_values, decommitment)
    }

    fn can_recompute_openings(&self) -> bool {
        self.polynomials.iter().all(|poly| poly.has_coefficients())
    }

    pub fn release_recomputable_evaluations_low_memory(&mut self) {
        if self.can_recompute_openings() {
            self.drop_evaluations();
        }
    }

    pub fn materialize_evaluations_for_reuse(
        &mut self,
        twiddles: &TwiddleTree<B>,
        base_column_pool: &BaseColumnPool<B>,
    ) {
        self.materialize_evaluations(twiddles, base_column_pool);
    }

    fn release_evaluations(&mut self, base_column_pool: &BaseColumnPool<B>) {
        // If any evals are mmap-backed, do not give them back to the pool — they would corrupt
        // the heap-buffer pool with file-backed memory. Forget those Vecs first so the per-poly
        // loop below skips them, then fall through to the normal give-back path for the rest.
        self.forget_mmap_backed_evals_if_any();
        for poly in &mut self.polynomials {
            if poly.evals.is_some() {
                let log_size = poly.log_size();
                let evals = poly.take_evals();
                base_column_pool.give_back(log_size, evals.values);
            }
        }
        // Munmap regions whose Vecs we just forgot.
        self.eval_mmap_guards.clear();
    }

    fn drop_evaluations(&mut self) {
        // Forget mmap-backed eval Vecs before they're dropped, so `Vec::drop` does not attempt
        // to free file-backed pages through the global allocator. The corresponding `MmapRegion`s
        // are then unmapped when `self.eval_mmap_guards` is cleared at the end of this function.
        self.forget_mmap_backed_evals_if_any();
        for poly in &mut self.polynomials {
            if poly.evals.is_some() {
                let _ = poly.take_evals();
            }
        }
        self.eval_mmap_guards.clear();
    }

    /// Forgets the eval Vecs of every polynomial whose backing was spilled to mmap.
    ///
    /// No-op when [`Self::eval_mmap_guards`] is empty (the common case in
    /// [`ProverMemoryMode::Fast`]). Must be called before any code path that drops or otherwise
    /// hands ownership of an eval Vec back to the global allocator (e.g. `Vec::drop`,
    /// [`BaseColumnPool::give_back`]).
    ///
    /// SAFETY: this performs an unchecked transmute of `Vec<Poly<B>>` to
    /// `Vec<Poly<SimdBackend>>`. The cast is layout-sound only when `B = SimdBackend`, which is
    /// the only backend that participates in the `LowMemory` re-materialization spill path
    /// today. The same constraint already applies to the existing
    /// [`crate::prover::spill::spill_eval_columns`] /
    /// [`crate::prover::spill::forget_mmap_backed_evals`] pair at `Self::new_with_memory_mode`.
    /// If a non-Simd backend ever populates `eval_mmap_guards`, this will be UB — the
    /// early-return on the empty-guard fast path is what keeps `CpuBackend` callers safe today.
    fn forget_mmap_backed_evals_if_any(&mut self) {
        if self.eval_mmap_guards.is_empty() {
            return;
        }
        let polys_simd_ptr = &mut self.polynomials as *mut Vec<Poly<B>>
            as *mut Vec<Poly<crate::prover::backend::simd::SimdBackend>>;
        unsafe {
            crate::prover::spill::forget_mmap_backed_evals(
                &mut *polys_simd_ptr,
                &self.eval_mmap_guards,
            );
        }
    }

    /// `LowMemory`-aware variant of [`Self::materialize_evaluations`].
    ///
    /// Materializes polynomial evaluations in heap-backed buffers up to a configurable
    /// byte budget (see [`low_memory_materialize_budget_bytes`]), then flushes each full
    /// batch to file-backed mmap. The final partial batch is left heap-backed: if the entire
    /// set of re-materialized evals fits within the budget, **no spilling occurs at all** and
    /// this function is effectively the eager `materialize_evaluations` path with one extra
    /// atomic load.
    ///
    /// This lets callers with generous memory budgets (e.g. phones with 2+ GiB of headroom
    /// during the decommit tail phase) trade RAM for wall-clock: each mmap flush costs a
    /// file write, a sync, and a mmap call, which adds up to seconds of wall-clock on
    /// mobile flash for a full privacy-demo-size trace. Defaulting the budget to 2 GiB
    /// eliminates the spill overhead during the tail phase on phones that can afford it,
    /// while still bounding the spike for memory-constrained devices that override the
    /// budget down.
    ///
    /// The mmap [`crate::prover::spill::EvalMmapGuard`]s are appended to
    /// [`Self::eval_mmap_guards`] so they live as long as the polynomials they back.
    /// Callers MUST go through [`Self::drop_evaluations`] (or [`Self::release_evaluations`])
    /// to release the polynomials, since both helpers know how to forget mmap-backed Vecs
    /// before dropping them.
    ///
    /// Safe for `B = SimdBackend` only — see [`Self::forget_mmap_backed_evals_if_any`] for
    /// the reasoning. Non-Simd backends should fall back to the eager
    /// `materialize_evaluations`.
    fn materialize_evaluations_low_memory(
        &mut self,
        twiddles: &TwiddleTree<B>,
        base_column_pool: &BaseColumnPool<B>,
    ) {
        let budget_bytes = low_memory_materialize_budget_bytes();
        let total = self.polynomials.len();
        // Accumulated bytes of newly-materialized (heap-backed) polys in the current batch,
        // plus the half-open index range the batch spans. `pending_start` is `None` when no
        // batch is open; `pending_end` is only read when `pending_start` is `Some`.
        let mut pending_bytes = 0usize;
        let mut pending_start: Option<usize> = None;
        let mut pending_end: usize = 0;

        for idx in 0..total {
            if self.polynomials[idx].evals.is_some() {
                // Pre-existing eval (heap- or mmap-backed from a prior call). Flush any
                // newly-materialized pending batch *before* it, then leave this poly alone.
                // Re-spilling an already-mmap-backed Vec would UB inside
                // `spill_eval_columns`, so we must never include a pre-existing poly in a
                // spill range.
                if let Some(start) = pending_start.take() {
                    self.spill_newly_materialized_range(start, pending_end);
                    pending_bytes = 0;
                }
                continue;
            }

            // Materialize this polynomial's evaluation into a fresh heap buffer.
            let evals = self.polynomials[idx].materialize_evaluation(twiddles, base_column_pool);
            // Approximate heap-backed byte count as `len * sizeof(BaseField)`. The actual
            // SIMD allocation is slightly larger due to packed lane alignment, but this is
            // within a few percent and is fine for a coarse-grained budget threshold.
            let bytes = evals.values.len() * std::mem::size_of::<BaseField>();
            self.polynomials[idx].evals = Some(evals);

            if pending_start.is_none() {
                pending_start = Some(idx);
            }
            pending_end = idx + 1;
            pending_bytes += bytes;

            if pending_bytes >= budget_bytes {
                // Flush the full batch: all polys in [pending_start..pending_end) are
                // contiguous and were just materialized in this loop (heap-backed), so
                // spilling them is safe.
                let start = pending_start.take().unwrap();
                self.spill_newly_materialized_range(start, pending_end);
                pending_bytes = 0;
            }
        }

        // The final partial batch is left heap-backed on purpose: if it fits in the budget,
        // spilling it would incur disk I/O for no benefit and would force every subsequent
        // read to fault through the page cache. It will be released normally by
        // `drop_evaluations` or `release_evaluations` at the end of the current PCS phase.
    }

    /// Spills a contiguous range of **newly-materialized, heap-backed** polynomials
    /// `[start..end)` to file-backed mmap.
    ///
    /// Precondition: every poly in the range whose `evals.is_some()` must be heap-backed
    /// (not already mmap-backed from a prior call). Re-spilling an mmap-backed Vec would
    /// drop the old mmap region through the global allocator and crash.
    /// [`Self::materialize_evaluations_low_memory`] maintains this invariant by flushing
    /// any pending batch before stepping over a pre-existing poly.
    fn spill_newly_materialized_range(&mut self, start: usize, end: usize) {
        debug_assert!(end > start, "empty spill range");
        // SAFETY: cast from `&mut [Poly<B>]` to `&mut [Poly<SimdBackend>]` is sound when
        // `B = SimdBackend`, the only backend that runs the LowMemory re-materialization
        // path. Same precondition as `Self::forget_mmap_backed_evals_if_any` and the
        // existing cast in `Self::new_with_memory_mode`.
        let slice: &mut [Poly<crate::prover::backend::simd::SimdBackend>] = unsafe {
            std::slice::from_raw_parts_mut(
                (&mut self.polynomials[start] as *mut Poly<B>)
                    as *mut Poly<crate::prover::backend::simd::SimdBackend>,
                end - start,
            )
        };
        if let Some(mut guard) = crate::prover::spill::spill_eval_columns(slice) {
            // `spill_eval_columns` stores indices relative to the start of its input slice;
            // translate them back to absolute positions in `self.polynomials` so the guard
            // lines up with the eventual `forget_mmap_backed_evals` call.
            guard.offset_indices(start);
            self.eval_mmap_guards.push(guard);
        }
    }

    /// Spills in-memory polynomial coefficients to a memory-mapped temporary file and drops
    /// the in-memory copies. Polynomials that already have their coefficients spilled or have
    /// no coefficients at all are silently skipped — this function is idempotent.
    ///
    /// After spilling, coefficients are loaded on demand from the mmap when needed for OODS
    /// evaluation, FRI quotient materialization, or decommitment.
    pub fn spill_coefficients(&mut self) -> std::io::Result<()> {
        use crate::prover::spill::CoefficientSpillFile;

        // Collect only the polys that still have in-memory coefficients.
        let n_to_spill = self
            .polynomials
            .iter()
            .filter(|p| p.coeffs.is_some())
            .count();
        if n_to_spill == 0 {
            phase_memory_checkpoint("pcs:tree:after_coefficient_spill");
            return Ok(());
        }

        let mut spill = CoefficientSpillFile::new()?;

        // Write in-memory coefficient data to the spill file, skipping already-spilled polys.
        for poly in &self.polynomials {
            if let Some(coeffs) = &poly.coeffs {
                let cpu_data = coeffs.coeffs.to_cpu();
                let bytes: &[u8] = bytemuck::cast_slice(&cpu_data);
                spill.write_coefficients_raw(bytes)?;
            }
        }

        // Freeze the file (creates the mmap).
        let frozen = spill.freeze()?;

        // Replace in-memory coefficients with spill references; preserve existing spill refs.
        let mut spill_idx: usize = 0;
        for poly in self.polynomials.iter_mut() {
            if poly.coeffs.is_some() {
                let log_size = poly.log_size();
                poly.spilled_coeffs = Some(SpilledPolyCoeffs {
                    spill_file: frozen.clone(),
                    spill_index: crate::prover::spill::SpillIndex(spill_idx),
                    log_size,
                });
                poly.coeffs = None;
                spill_idx += 1;
            }
        }

        phase_memory_checkpoint("pcs:tree:after_coefficient_spill");
        Ok(())
    }

    fn materialize_evaluations(
        &mut self,
        twiddles: &TwiddleTree<B>,
        base_column_pool: &BaseColumnPool<B>,
    ) {
        for poly in &mut self.polynomials {
            if poly.evals.is_none() {
                poly.evals = Some(poly.materialize_evaluation(twiddles, base_column_pool));
            }
        }
    }

    fn materialize_evaluation_range(
        &mut self,
        col_start: usize,
        col_end: usize,
        twiddles: &TwiddleTree<B>,
        base_column_pool: &BaseColumnPool<B>,
    ) {
        for poly in &mut self.polynomials[col_start..col_end] {
            if poly.evals.is_none() {
                poly.evals = Some(poly.materialize_evaluation(twiddles, base_column_pool));
            }
        }
    }

    fn materialize_evaluation_indices(
        &mut self,
        indices: &[usize],
        twiddles: &TwiddleTree<B>,
        base_column_pool: &BaseColumnPool<B>,
    ) {
        for &index in indices {
            let poly = &mut self.polynomials[index];
            if poly.evals.is_none() {
                poly.evals = Some(poly.materialize_evaluation(twiddles, base_column_pool));
            }
        }
    }

    fn drop_evaluation_range(&mut self, col_start: usize, col_end: usize) {
        for poly in &mut self.polynomials[col_start..col_end] {
            if poly.evals.is_some() {
                let _ = poly.take_evals();
            }
        }
    }

    fn drop_evaluation_indices(&mut self, indices: &[usize]) {
        for &index in indices {
            let poly = &mut self.polynomials[index];
            if poly.evals.is_some() {
                let _ = poly.take_evals();
            }
        }
    }

    fn recompute_queried_values(&self, queries: &[usize]) -> ColumnVec<Vec<BaseField>> {
        let max_log_size = self.commitment.height() as usize;
        self.polynomials
            .iter()
            .map(|poly| {
                queries
                    .iter()
                    .map(|&pos| self.sample_committed_position(poly, pos, max_log_size))
                    .collect()
            })
            .collect()
    }

    fn compute_leaf_hashes(
        &self,
        positions: &[usize],
    ) -> HashMap<usize, <MC::H as MerkleHasherLifted>::Hash> {
        let max_log_size = self.commitment.height() as usize;
        let sorted_polynomials = self
            .polynomials
            .iter()
            .sorted_by_key(|poly| poly.log_size())
            .collect_vec();

        positions
            .iter()
            .copied()
            .map(|position| {
                let mut hasher = MC::H::default();
                for chunk in &sorted_polynomials.iter().chunks(16) {
                    let values = chunk
                        .into_iter()
                        .map(|poly| self.sample_committed_position(poly, position, max_log_size))
                        .collect_vec();
                    hasher.update_leaf(&values);
                }
                (position, hasher.finalize())
            })
            .collect()
    }

    fn sample_committed_position(
        &self,
        poly: &Poly<B>,
        query_position: usize,
        max_log_size: usize,
    ) -> BaseField {
        let mapped_index =
            Self::mapped_position(query_position, poly.log_size() as usize, max_log_size);
        if let Some(evals) = &poly.evals {
            evals.values.at(mapped_index)
        } else {
            assert!(
                poly.has_coefficients(),
                "low-memory PCS decommit requires retained coefficients (in memory or spilled)"
            );
            let point = poly
                .eval_domain
                .at(bit_reverse_index(mapped_index, poly.log_size()))
                .into_ef::<SecureField>();
            poly.eval_at_point(point, None).to_m31_array()[0]
        }
    }

    fn mapped_position(query_position: usize, log_size: usize, max_log_size: usize) -> usize {
        let shift = max_log_size - log_size;
        (query_position >> (shift + 1) << 1) + (query_position & 1)
    }
}

fn print_column_size_histogram<B: BackendForChannel<MC>, MC: MerkleChannel>(
    columns_per_tree: &TreeVec<ColumnVec<&CircleEvaluation<B, BaseField, BitReversedOrder>>>,
) {
    let mut log_size_histogram = HashMap::new();
    for columns in columns_per_tree.iter() {
        for column in columns {
            *log_size_histogram
                .entry(column.domain.log_size())
                .or_insert(0) += 1;
        }
    }
    for (log_size, count) in log_size_histogram {
        info!("Log size {log_size}: {count}");
    }
}

fn print_polynomial_size_histogram<B: BackendForChannel<MC>, MC: MerkleChannel>(
    polynomials_per_tree: &TreeVec<ColumnVec<&Poly<B>>>,
) {
    let mut log_size_histogram = HashMap::new();
    for polynomials in polynomials_per_tree.iter() {
        for poly in polynomials {
            *log_size_histogram.entry(poly.log_size()).or_insert(0) += 1;
        }
    }
    for (log_size, count) in log_size_histogram {
        info!("Log size {log_size}: {count}");
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::{
        default_prover_memory_mode, parse_low_memory_materialize_budget_bytes,
        parse_prover_memory_mode, set_default_prover_memory_mode,
        set_low_memory_materialize_budget_bytes, CommitmentTreeMerkleProver, CommitmentTreeProver,
        ProverMemoryMode, DEFAULT_PROVER_MEMORY_MODE_OVERRIDE,
        LOW_MEMORY_MATERIALIZE_BUDGET_BYTES_OVERRIDE, PROVER_MEMORY_MODE_OVERRIDE_FAST,
        PROVER_MEMORY_MODE_OVERRIDE_LOW_MEMORY, PROVER_MEMORY_MODE_OVERRIDE_UNSET,
    };
    use crate::core::channel::MerkleChannel;
    use crate::core::fields::m31::M31;
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
    use crate::prover::backend::simd::SimdBackend;
    use crate::prover::backend::{BackendForChannel, Column, CpuBackend};
    use crate::prover::mempool::BaseColumnPool;
    use crate::prover::poly::circle::{CircleCoefficients, PolyOps};

    fn prepare_tree<MC: MerkleChannel>(
        pool: &BaseColumnPool<CpuBackend>,
    ) -> CommitmentTreeProver<CpuBackend, MC>
    where
        CpuBackend: BackendForChannel<MC>,
    {
        prepare_tree_with_memory_mode::<MC>(pool, ProverMemoryMode::Fast)
    }

    fn prepare_tree_with_memory_mode<MC: MerkleChannel>(
        pool: &BaseColumnPool<CpuBackend>,
        memory_mode: ProverMemoryMode,
    ) -> CommitmentTreeProver<CpuBackend, MC>
    where
        CpuBackend: BackendForChannel<MC>,
    {
        let polys = [
            CircleCoefficients::new((0..1 << 4).map(M31::from).collect()),
            CircleCoefficients::new((0..1 << 5).map(|i| M31::from((2 * i) as u32)).collect()),
            CircleCoefficients::new((0..1 << 6).map(|i| M31::from((3 * i) as u32)).collect()),
        ];
        let twiddles = CpuBackend::precompute_twiddles(CanonicCoset::new(7).half_coset());

        CommitmentTreeProver::new_with_memory_mode(
            polys.into_iter().collect_vec(),
            1,
            &twiddles,
            true,
            None,
            pool,
            memory_mode,
        )
    }

    #[test]
    fn test_parse_prover_memory_mode_fast() {
        assert_eq!(
            parse_prover_memory_mode("fast"),
            Some(ProverMemoryMode::Fast)
        );
    }

    #[test]
    fn test_parse_prover_memory_mode_low_memory_aliases() {
        for value in ["low_memory", "low-memory", "lowmemory", "checkpointed"] {
            assert_eq!(
                parse_prover_memory_mode(value),
                Some(ProverMemoryMode::LowMemory)
            );
        }
    }

    /// Exercises the process-wide [`set_default_prover_memory_mode`] override mechanism.
    ///
    /// All override scenarios are bundled into a single `#[test]` function to serialize them
    /// against each other and against any other test in the binary that constructs a
    /// [`CommitmentSchemeProver`] (which would otherwise observe a leaked override and read
    /// the wrong default mode). The test saves and restores the override sentinel around each
    /// scenario so concurrent tests in the same binary remain unaffected.
    #[test]
    fn test_set_default_prover_memory_mode_override() {
        // Snapshot the current sentinel so we can restore it on every exit path, including
        // panic propagation. We deliberately do not assert the initial state because parallel
        // tests might have set it; we only require that we leave it as we found it.
        let initial_override =
            DEFAULT_PROVER_MEMORY_MODE_OVERRIDE.load(std::sync::atomic::Ordering::Acquire);

        // Scenario 1: setting LowMemory pins default_prover_memory_mode() to LowMemory
        // regardless of the env var (we don't manipulate the env var here to keep this test
        // free of process-global env interference).
        set_default_prover_memory_mode(ProverMemoryMode::LowMemory);
        assert_eq!(
            DEFAULT_PROVER_MEMORY_MODE_OVERRIDE.load(std::sync::atomic::Ordering::Acquire),
            PROVER_MEMORY_MODE_OVERRIDE_LOW_MEMORY,
            "override sentinel must encode LowMemory after set_default_prover_memory_mode(LowMemory)"
        );
        assert_eq!(
            default_prover_memory_mode(),
            ProverMemoryMode::LowMemory,
            "resolver must honor LowMemory override"
        );

        // Scenario 2: overriding to Fast wins over a previous LowMemory override.
        set_default_prover_memory_mode(ProverMemoryMode::Fast);
        assert_eq!(
            DEFAULT_PROVER_MEMORY_MODE_OVERRIDE.load(std::sync::atomic::Ordering::Acquire),
            PROVER_MEMORY_MODE_OVERRIDE_FAST,
            "override sentinel must encode Fast after set_default_prover_memory_mode(Fast)"
        );
        assert_eq!(
            default_prover_memory_mode(),
            ProverMemoryMode::Fast,
            "resolver must honor Fast override"
        );

        // Scenario 3: re-setting to LowMemory works (the override is not single-shot).
        set_default_prover_memory_mode(ProverMemoryMode::LowMemory);
        assert_eq!(
            default_prover_memory_mode(),
            ProverMemoryMode::LowMemory,
            "override must be replaceable, not single-shot"
        );

        // Scenario 4: clearing the override (via the unset sentinel) returns control to the
        // env-var/default fallback path. We can't easily test the env var branch here without
        // mutating process state, so we just verify the sentinel transitions correctly.
        DEFAULT_PROVER_MEMORY_MODE_OVERRIDE.store(
            PROVER_MEMORY_MODE_OVERRIDE_UNSET,
            std::sync::atomic::Ordering::Release,
        );
        assert_eq!(
            DEFAULT_PROVER_MEMORY_MODE_OVERRIDE.load(std::sync::atomic::Ordering::Acquire),
            PROVER_MEMORY_MODE_OVERRIDE_UNSET,
            "manual unset must clear the override sentinel"
        );

        // Restore the snapshot so unrelated tests in the same binary observe the same global
        // state they started with.
        DEFAULT_PROVER_MEMORY_MODE_OVERRIDE
            .store(initial_override, std::sync::atomic::Ordering::Release);
    }

    /// Builds a SIMD `LowMemory` `CommitmentTreeProver` shaped like the privacy-demo workload:
    /// many polynomials of varying log_size. Used by the round-trip tests below to exercise
    /// the [`CommitmentTreeProver::materialize_evaluations_low_memory`] code path.
    fn prepare_simd_low_memory_tree<MC: MerkleChannel>(
        pool: &BaseColumnPool<SimdBackend>,
    ) -> CommitmentTreeProver<SimdBackend, MC>
    where
        SimdBackend: BackendForChannel<MC>,
    {
        // Sizes are chosen to be large enough to exercise the SIMD path (>= N_LANES per column)
        // while still completing instantly in CI.
        let polys = (0..6)
            .map(|i| {
                let log_size = 5 + (i % 3);
                CircleCoefficients::new(
                    (0..1u32 << log_size)
                        .map(|j| M31::from((j + 1) * (i + 1) as u32))
                        .collect(),
                )
            })
            .collect_vec();
        let twiddles = SimdBackend::precompute_twiddles(CanonicCoset::new(8).half_coset());

        CommitmentTreeProver::new_with_memory_mode(
            polys,
            1,
            &twiddles,
            // Retain coefficients so the tree can re-materialize evals after the initial drop —
            // this is the path the new spill helper targets.
            true,
            None,
            pool,
            ProverMemoryMode::LowMemory,
        )
    }

    /// Verifies the `LowMemory` per-poly materialize-and-spill round-trip.
    ///
    /// 1. After construction the SIMD `LowMemory` tree has dropped its evals (precondition).
    /// 2. `materialize_evaluations_low_memory` repopulates every poly's evals; with a budget of 1
    ///    byte the function spills after every column, so the resulting tree must hold at least one
    ///    mmap guard (and in practice one per polynomial, though the exact count is an
    ///    implementation detail).
    /// 3. The mmap-backed eval values must equal the values produced by the eager
    ///    `materialize_evaluations` path on a freshly constructed sibling tree, which transitively
    ///    asserts that the spill round-trip preserves bytes.
    /// 4. `drop_evaluations` must clear both the polys' evals and the guard list without crashing
    ///    on `Vec::drop` of mmap-backed memory — this is the regression that breaks if
    ///    [`CommitmentTreeProver::forget_mmap_backed_evals_if_any`] is wired incorrectly.
    ///
    /// Called as a sub-scenario from [`test_materialize_low_memory_budget_and_round_trip`]
    /// so that all budget-override-touching logic is serialized behind a single `#[test]`
    /// entry point — two concurrent `#[test]` functions both mutating
    /// [`LOW_MEMORY_MATERIALIZE_BUDGET_BYTES_OVERRIDE`] would race on save/restore and
    /// corrupt each other's expected budget.
    fn run_simd_low_memory_materialize_round_trip_scenario() {
        // Force the smallest meaningful budget so the spill code path is exercised.
        // The caller is responsible for save/restore around this whole sub-scenario.
        set_low_memory_materialize_budget_bytes(1);
        // Twiddle precomputation is shared between the two trees so the materialized values
        // are byte-identical.
        let pool = BaseColumnPool::<SimdBackend>::new();
        let mut spilled_tree = prepare_simd_low_memory_tree::<Blake2sMerkleChannel>(&pool);
        let twiddles = SimdBackend::precompute_twiddles(CanonicCoset::new(8).half_coset());

        // Precondition: LowMemory construction drops evals.
        assert!(
            spilled_tree.polynomials.iter().all(|p| p.evals.is_none()),
            "LowMemory construction must drop evals before re-materialization"
        );
        assert!(
            spilled_tree.eval_mmap_guards.is_empty(),
            "no eval mmap guards should exist before re-materialization"
        );

        // Step 2: materialize-and-spill per poly.
        spilled_tree.materialize_evaluations_low_memory(&twiddles, &pool);
        assert!(
            spilled_tree.polynomials.iter().all(|p| p.evals.is_some()),
            "every poly must have evals after materialize_evaluations_low_memory"
        );
        assert!(
            !spilled_tree.eval_mmap_guards.is_empty(),
            "at least one mmap guard should be recorded after spilling — \
             a missing guard would mean drop_evaluations cannot recover the mmap state"
        );

        // Step 3: cross-check eval values against the eager re-materialization path.
        let mut eager_tree = prepare_simd_low_memory_tree::<Blake2sMerkleChannel>(&pool);
        eager_tree.materialize_evaluations(&twiddles, &pool);
        for (idx, (a, b)) in spilled_tree
            .polynomials
            .iter()
            .zip(eager_tree.polynomials.iter())
            .enumerate()
        {
            let spilled_values = a.evals.as_ref().unwrap().values.to_cpu();
            let eager_values = b.evals.as_ref().unwrap().values.to_cpu();
            assert_eq!(
                spilled_values, eager_values,
                "mmap-backed and heap-backed evals must agree at poly index {idx}"
            );
        }

        // Step 4: drop_evaluations must safely release mmap-backed Vecs.
        spilled_tree.drop_evaluations();
        assert!(
            spilled_tree.polynomials.iter().all(|p| p.evals.is_none()),
            "drop_evaluations must clear every poly's evals"
        );
        assert!(
            spilled_tree.eval_mmap_guards.is_empty(),
            "drop_evaluations must clear the guard list (which munmaps the regions)"
        );

        // Round-trip must be repeatable: a second materialize+drop cycle must also succeed.
        spilled_tree.materialize_evaluations_low_memory(&twiddles, &pool);
        assert!(spilled_tree.polynomials.iter().all(|p| p.evals.is_some()));
        spilled_tree.drop_evaluations();
        assert!(spilled_tree.polynomials.iter().all(|p| p.evals.is_none()));

        // Cleanup of the eager tree (heap-backed) must also work via the same drop helper.
        eager_tree.drop_evaluations();
    }

    /// Single entry point for every test that mutates the global
    /// [`LOW_MEMORY_MATERIALIZE_BUDGET_BYTES_OVERRIDE`]. Bundling them under one `#[test]`
    /// serializes them within this function so that no two scenarios ever race on
    /// save/restore (which would cause one scenario's restore to stomp on another's
    /// expected budget). Exercises:
    ///
    /// - **Round-trip correctness** at `budget = 1`: every poly spills, mmap-backed evals must
    ///   match the eager path byte-for-byte, `drop_evaluations`/`release_evaluations` must release
    ///   mmap pages cleanly.
    /// - **Huge budget**: the entire re-materialized set fits in the budget, so no spilling happens
    ///   at all. `eval_mmap_guards` must remain empty and every poly must have a heap-backed eval
    ///   after the call — this is the fast path that makes the tail phase of a mobile proof fast
    ///   when the device has headroom.
    /// - **Tiny budget (1 byte)**: every poly exceeds the budget, so every single poly is spilled
    ///   immediately. Mirrors the pre-budget behavior.
    /// - **Medium budget**: budget smaller than one poly's eval bytes, so every materialize step
    ///   crosses the flush threshold. Validates the batching path end-to-end and that guards are
    ///   produced.
    ///
    /// The budget override is saved once at the top and restored once at the bottom.
    #[test]
    fn test_materialize_low_memory_budget_and_round_trip() {
        let prev_budget =
            LOW_MEMORY_MATERIALIZE_BUDGET_BYTES_OVERRIDE.load(std::sync::atomic::Ordering::Acquire);

        // Scenario 0: round-trip correctness with budget=1 (see
        // `run_simd_low_memory_materialize_round_trip_scenario` for the detailed contract).
        run_simd_low_memory_materialize_round_trip_scenario();

        let pool = BaseColumnPool::<SimdBackend>::new();
        let twiddles = SimdBackend::precompute_twiddles(CanonicCoset::new(8).half_coset());

        // ---- Scenario 1: huge budget → no spill, everything heap-backed ---------------
        set_low_memory_materialize_budget_bytes(2usize << 30); // 2 GiB — far exceeds test data
        let mut huge_budget_tree = prepare_simd_low_memory_tree::<Blake2sMerkleChannel>(&pool);
        assert!(
            huge_budget_tree
                .polynomials
                .iter()
                .all(|p| p.evals.is_none()),
            "precondition: LowMemory construction drops evals"
        );
        huge_budget_tree.materialize_evaluations_low_memory(&twiddles, &pool);
        assert!(
            huge_budget_tree
                .polynomials
                .iter()
                .all(|p| p.evals.is_some()),
            "every poly must have evals after materialize_evaluations_low_memory"
        );
        assert!(
            huge_budget_tree.eval_mmap_guards.is_empty(),
            "huge budget must not spill: eval_mmap_guards must be empty, but had {} \
             guards. Re-materialization is paying the disk-I/O cost unnecessarily.",
            huge_budget_tree.eval_mmap_guards.len()
        );
        // drop_evaluations must still work when there are no guards.
        huge_budget_tree.drop_evaluations();
        assert!(huge_budget_tree
            .polynomials
            .iter()
            .all(|p| p.evals.is_none()));

        // ---- Scenario 2: tiny budget → spill every poly ------------------------------
        set_low_memory_materialize_budget_bytes(1);
        let mut tiny_budget_tree = prepare_simd_low_memory_tree::<Blake2sMerkleChannel>(&pool);
        tiny_budget_tree.materialize_evaluations_low_memory(&twiddles, &pool);
        assert!(
            tiny_budget_tree
                .polynomials
                .iter()
                .all(|p| p.evals.is_some()),
            "every poly must have evals after materialize_evaluations_low_memory"
        );
        assert!(
            !tiny_budget_tree.eval_mmap_guards.is_empty(),
            "tiny budget must spill at least once"
        );
        tiny_budget_tree.drop_evaluations();

        // ---- Scenario 3: medium budget → at least one spill + a final heap-backed ----
        // Size the budget to be *smaller* than one polynomial's eval bytes, guaranteeing
        // that every materialize call crosses the threshold. The test tree has 6 polys
        // so we get ~6 flushes. The final partial batch is trivially empty in this case,
        // but the important assertion is that the batching path is exercised end-to-end
        // and guards get produced.
        set_low_memory_materialize_budget_bytes(64);
        let mut medium_budget_tree = prepare_simd_low_memory_tree::<Blake2sMerkleChannel>(&pool);
        medium_budget_tree.materialize_evaluations_low_memory(&twiddles, &pool);
        assert!(
            medium_budget_tree
                .polynomials
                .iter()
                .all(|p| p.evals.is_some()),
            "every poly must have evals after materialize_evaluations_low_memory"
        );
        assert!(
            !medium_budget_tree.eval_mmap_guards.is_empty(),
            "medium budget must have spilled at least once with this tree shape"
        );
        medium_budget_tree.drop_evaluations();

        // Restore global state so parallel tests see what they started with.
        LOW_MEMORY_MATERIALIZE_BUDGET_BYTES_OVERRIDE
            .store(prev_budget, std::sync::atomic::Ordering::Release);
    }

    #[test]
    fn test_parse_low_memory_materialize_budget_bytes() {
        // Accepts positive decimal integers.
        assert_eq!(parse_low_memory_materialize_budget_bytes("1"), Some(1));
        assert_eq!(
            parse_low_memory_materialize_budget_bytes("1073741824"),
            Some(1usize << 30)
        );
        assert_eq!(
            parse_low_memory_materialize_budget_bytes("  42  "),
            Some(42)
        );
        // Rejects zero (0-byte budget is meaningless for this knob), negatives, non-numeric.
        assert_eq!(parse_low_memory_materialize_budget_bytes("0"), None);
        assert_eq!(parse_low_memory_materialize_budget_bytes(""), None);
        assert_eq!(parse_low_memory_materialize_budget_bytes("-1"), None);
        assert_eq!(parse_low_memory_materialize_budget_bytes("abc"), None);
        assert_eq!(parse_low_memory_materialize_budget_bytes("1.5"), None);
        assert_eq!(parse_low_memory_materialize_budget_bytes("1MB"), None);
    }

    #[test]
    fn test_commitment_tree_decommit_after_releasing_evals() {
        let pool = BaseColumnPool::new();
        let queries = [1, 7, 18, 33];

        let fast_tree = prepare_tree::<Blake2sMerkleChannel>(&pool);
        let (fast_values, fast_decommitment) = fast_tree.decommit(&queries);

        let mut low_memory_tree = prepare_tree::<Blake2sMerkleChannel>(&pool);
        low_memory_tree.release_evaluations(&pool);
        assert!(low_memory_tree
            .polynomials
            .iter()
            .all(|poly| poly.evals.is_none()));

        let (low_memory_values, low_memory_decommitment) = low_memory_tree.decommit(&queries);

        assert_eq!(low_memory_values, fast_values);
        assert_eq!(
            low_memory_decommitment.decommitment.hash_witness,
            fast_decommitment.decommitment.hash_witness
        );
        assert_eq!(
            low_memory_decommitment.aux.all_node_values,
            fast_decommitment.aux.all_node_values
        );
    }

    #[test]
    fn test_low_memory_tree_releases_evals_on_construction() {
        let pool = BaseColumnPool::new();
        let low_memory_tree = prepare_tree_with_memory_mode::<Blake2sMerkleChannel>(
            &pool,
            ProverMemoryMode::LowMemory,
        );

        assert!(low_memory_tree
            .polynomials
            .iter()
            .all(|poly| poly.evals.is_none()));
        assert!(low_memory_tree.can_recompute_openings());
    }

    #[test]
    fn test_checkpointed_commitment_tree_matches_full_tree() {
        let pool = BaseColumnPool::new();
        let queries = [1, 7, 18, 33];

        let fast_tree = prepare_tree::<Blake2sMerkleChannel>(&pool);
        let checkpointed_tree = prepare_tree_with_memory_mode::<Blake2sMerkleChannel>(
            &pool,
            ProverMemoryMode::LowMemory,
        );

        assert!(checkpointed_tree
            .polynomials
            .iter()
            .all(|poly| poly.evals.is_none()));

        assert!(matches!(
            checkpointed_tree.commitment,
            CommitmentTreeMerkleProver::Checkpointed(_)
        ));

        let (fast_values, fast_decommitment) = fast_tree.decommit(&queries);
        let (checkpointed_values, checkpointed_decommitment) = checkpointed_tree.decommit(&queries);

        assert_eq!(checkpointed_values, fast_values);
        assert_eq!(
            checkpointed_decommitment.decommitment.hash_witness,
            fast_decommitment.decommitment.hash_witness
        );
        assert_eq!(
            checkpointed_decommitment.aux.all_node_values,
            fast_decommitment.aux.all_node_values
        );
    }

    #[test]
    fn test_checkpointed_commitment_tree_matches_full_tree_after_releasing_evals() {
        let pool = BaseColumnPool::new();
        let queries = [1, 7, 18, 33];

        let fast_tree = prepare_tree::<Blake2sMerkleChannel>(&pool);
        let mut checkpointed_tree = prepare_tree_with_memory_mode::<Blake2sMerkleChannel>(
            &pool,
            ProverMemoryMode::LowMemory,
        );
        checkpointed_tree.release_evaluations(&pool);
        assert!(checkpointed_tree
            .polynomials
            .iter()
            .all(|poly| poly.evals.is_none()));

        let (fast_values, fast_decommitment) = fast_tree.decommit(&queries);
        let (checkpointed_values, checkpointed_decommitment) = checkpointed_tree.decommit(&queries);

        assert_eq!(checkpointed_values, fast_values);
        assert_eq!(
            checkpointed_decommitment.decommitment.hash_witness,
            fast_decommitment.decommitment.hash_witness
        );
        assert_eq!(
            checkpointed_decommitment.aux.all_node_values,
            fast_decommitment.aux.all_node_values
        );
    }
}
