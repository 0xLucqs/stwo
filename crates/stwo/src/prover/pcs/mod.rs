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

fn default_prover_memory_mode() -> ProverMemoryMode {
    match std::env::var("STWO_PROVER_MEMORY_MODE") {
        Ok(value) => match parse_prover_memory_mode(&value) {
            Some(mode) => mode,
            None => {
                tracing::warn!(
                    env_value = value.as_str(),
                    "Unknown STWO_PROVER_MEMORY_MODE, defaulting to fast mode"
                );
                ProverMemoryMode::Fast
            }
        },
        Err(_) => ProverMemoryMode::Fast,
    }
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
        match access_pattern {
            Some(access_pattern) => {
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
                        tree.materialize_evaluations(self.twiddles, &self.base_column_pool);
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
                        tree.materialize_evaluations(self.twiddles, &self.base_column_pool);
                    }
                    let result = tree.decommit(query_positions);
                    if self.memory_mode == ProverMemoryMode::LowMemory
                        && tree.can_recompute_openings()
                    {
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
        let mut polynomials = B::evaluate_polynomials(
            polynomials,
            log_blowup_factor,
            twiddles,
            retain_coefficients,
            base_column_pool,
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
        let _eval_mmap_guard = if memory_mode == ProverMemoryMode::LowMemory {
            // The spill function works with concrete SimdBackend types. Check if we can
            // downcast. In practice, B is always SimdBackend in the proving pipeline.
            let polys_ptr = &mut polynomials as *mut Vec<Poly<B>> as *mut Vec<
                Poly<crate::prover::backend::simd::SimdBackend>,
            >;
            // SAFETY: This is only called when B = SimdBackend (the only backend that
            // implements BackendForChannel). The cast is sound because Poly<B> and
            // Poly<SimdBackend> have identical layout when B = SimdBackend.
            let guard = unsafe { crate::prover::spill::spill_eval_columns(&mut *polys_ptr) };
            phase_memory_checkpoint("pcs:tree:new:after_eval_mmap_spill");
            guard
        } else {
            None
        };

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
            if _eval_mmap_guard.is_some() {
                // Evals are mmap-backed: forget them to prevent Vec::drop on mmap memory.
                let polys_ptr = &mut polynomials as *mut Vec<Poly<B>> as *mut Vec<
                    Poly<crate::prover::backend::simd::SimdBackend>,
                >;
                unsafe { crate::prover::spill::forget_mmap_backed_evals(&mut *polys_ptr) };
            } else {
                // Evals are heap-backed: drop normally.
                for poly in &mut polynomials {
                    if poly.evals.is_some() {
                        let _ = poly.take_evals();
                    }
                }
            }
            phase_memory_checkpoint("pcs:tree:new:after_low_memory_eval_release");
        }
        phase_memory_checkpoint("pcs:tree:new:after_merkle");

        CommitmentTreeProver {
            polynomials,
            commitment: tree,
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

    fn release_evaluations(&mut self, base_column_pool: &BaseColumnPool<B>) {
        for poly in &mut self.polynomials {
            if poly.evals.is_some() {
                let log_size = poly.log_size();
                let evals = poly.take_evals();
                base_column_pool.give_back(log_size, evals.values);
            }
        }
    }

    fn drop_evaluations(&mut self) {
        for poly in &mut self.polynomials {
            if poly.evals.is_some() {
                let _ = poly.take_evals();
            }
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
        let n_to_spill = self.polynomials.iter().filter(|p| p.coeffs.is_some()).count();
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
        parse_prover_memory_mode, CommitmentTreeMerkleProver, CommitmentTreeProver,
        ProverMemoryMode,
    };
    use crate::core::channel::MerkleChannel;
    use crate::core::fields::m31::M31;
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
    use crate::prover::backend::{BackendForChannel, CpuBackend};
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
