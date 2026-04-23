use hashbrown::HashMap;
use itertools::Itertools;
use num_traits::Zero;
use tracing::instrument;

use crate::core::channel::{Channel, MerkleChannel};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::{SecureField, QM31, SECURE_EXTENSION_DEGREE};
use crate::core::fri::{
    ExtendedFriLayerProof, ExtendedFriProof, FriConfig, FriLayerProof, FriLayerProofAux, FriProof,
    FriProofAux,
};
use crate::core::poly::line::LinePoly;
use crate::core::queries::{draw_queries, Queries};
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::LOG_PACKED_LEAF_SIZE;
use crate::prover::backend::{Col, ColumnOps};
use crate::prover::line::LineEvaluation;
use crate::prover::pcs::ProverMemoryMode;
use crate::prover::poly::circle::{PolyOps, SecureEvaluation};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;
use crate::prover::vcs_lifted::prover::{CheckpointedMerkleProverLifted, MerkleProverLifted};

pub trait FriOps: ColumnOps<BaseField> + PolyOps + Sized + ColumnOps<SecureField> {
    /// Folds a degree `d` polynomial into a degree `d/2` polynomial.
    ///
    /// Let `eval` be a polynomial evaluated on a [LineDomain] `E`, `alpha` be a random field
    /// element and `pi(x) = 2x^2 - 1` be the circle's x-coordinate doubling map. This function
    /// returns `f' = f0 + alpha * f1` evaluated on `pi(E)` such that `2f(x) = f0(pi(x)) + x *
    /// f1(pi(x))`.
    ///
    /// # Panics
    ///
    /// Panics if there are less than two evaluations.
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
        fold_step: u32,
    ) -> LineEvaluation<Self>;

    /// Folds and accumulates a degree `d` circle polynomial into a degree `d/2` univariate
    /// polynomial.
    ///
    /// Let `src` be the evaluation of a circle polynomial `f` on a
    /// [`CircleDomain`] `E`. This function computes evaluations of `f' = f0
    /// + alpha * f1` on the x-coordinates of `E` such that `2f(p) = f0(px) + py * f1(px)`. The
    /// evaluations of `f'` are accumulated into `dst` by the formula `dst = dst * alpha^2 + f'`.
    ///
    /// # Panics
    ///
    /// Panics if `src` is not double the length of `dst`.
    ///
    /// [`CircleDomain`]: crate::core::poly::circle::CircleDomain
    // TODO(andrew): Make folding factor generic.
    // TODO(andrew): Fold directly into FRI layer to prevent allocation.
    fn fold_circle_into_line(
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self>;

    /// Decomposes a FRI-space polynomial into a polynomial inside the fft-space and the
    /// remainder term.
    /// FRI-space: polynomials of total degree n/2.
    /// Based on lemma #12 from the CircleStark paper: f(P) = g(P)+ lambda * alternating(P),
    /// where lambda is the cosset diff of eval, and g is a polynomial in the fft-space.
    fn decompose(
        eval: &SecureEvaluation<Self, BitReversedOrder>,
    ) -> (SecureEvaluation<Self, BitReversedOrder>, SecureField);
}

pub struct FriDecommitResult<H: MerkleHasherLifted> {
    pub fri_proof: ExtendedFriProof<H>,
    pub query_positions: Vec<usize>,
    pub unsorted_query_locations: Vec<usize>,
}

enum FriFirstLayer<'a, B: FriOps + MerkleOpsLifted<H>, H: MerkleHasherLifted> {
    InMemory(FriFirstLayerProver<'a, B, H>),
    LowMemory(LowMemoryFriFirstLayerProver<'a, B, H>),
}

impl<'a, B: FriOps + MerkleOpsLifted<H>, H: MerkleHasherLifted> FriFirstLayer<'a, B, H> {
    const fn column(&self) -> &'a SecureEvaluation<B, BitReversedOrder> {
        match self {
            FriFirstLayer::InMemory(layer) => layer.column,
            FriFirstLayer::LowMemory(layer) => layer.column,
        }
    }
}

enum FriInnerLayers<'a, B: FriOps + MerkleOpsLifted<H>, H: MerkleHasherLifted> {
    InMemory(Vec<FriInnerLayerProver<B, H>>),
    LowMemory(LowMemoryFriInnerLayers<'a, B, H>),
}

struct LowMemoryFriInnerLayers<'a, B: FriOps + MerkleOpsLifted<H>, H: MerkleHasherLifted> {
    initial_folding_alpha: SecureField,
    twiddles: &'a TwiddleTree<B>,
    layers: Vec<LowMemoryFriInnerLayerProver<B, H>>,
}

const LOW_MEMORY_MERKLE_CHECKPOINT_STRIDE: u32 = 4;

/// A FRI prover that applies the FRI protocol to prove a set of polynomials are of low degree.
pub struct FriProver<'a, B: FriOps + MerkleOpsLifted<MC::H>, MC: MerkleChannel> {
    config: FriConfig,
    first_layer: FriFirstLayer<'a, B, MC::H>,
    inner_layers: FriInnerLayers<'a, B, MC::H>,
    last_layer_poly: LinePoly,
}
impl<'a, B: FriOps + MerkleOpsLifted<MC::H>, MC: MerkleChannel> FriProver<'a, B, MC> {
    /// Runs the commitment phase of FRI on one circle evaluation over a canonic circle domain.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * The evaluation is not from a sufficiently low degree circle polynomial.
    /// * The evaluation domain is not a canonic circle domain.
    #[instrument(skip_all)]
    pub fn commit(
        channel: &mut MC::C,
        config: FriConfig,
        column: &'a SecureEvaluation<B, BitReversedOrder>,
        twiddles: &'a TwiddleTree<B>,
    ) -> Self {
        Self::commit_with_memory_mode(channel, config, column, twiddles, ProverMemoryMode::Fast)
    }

    /// Runs the commitment phase of FRI with the requested prover memory mode.
    pub fn commit_with_memory_mode(
        channel: &mut MC::C,
        config: FriConfig,
        column: &'a SecureEvaluation<B, BitReversedOrder>,
        twiddles: &'a TwiddleTree<B>,
        memory_mode: ProverMemoryMode,
    ) -> Self {
        assert!(column.domain.is_canonic(), "not canonic");

        let first_layer = Self::commit_first_layer(channel, &config, column, memory_mode);
        let (inner_layers, last_layer_evaluation) =
            Self::commit_inner_layers(channel, config, column, twiddles, memory_mode);
        let last_layer_poly = Self::commit_last_layer(channel, config, last_layer_evaluation);

        Self {
            config,
            first_layer,
            inner_layers,
            last_layer_poly,
        }
    }

    fn compute_initial_inner_layer_evaluation(
        column: &SecureEvaluation<B, BitReversedOrder>,
        config: FriConfig,
        twiddles: &TwiddleTree<B>,
        folding_alpha: SecureField,
    ) -> LineEvaluation<B> {
        let mut layer_evaluation = B::fold_circle_into_line(column, folding_alpha, twiddles);

        // Apply any additional line folds requested for the first stage.
        if config.fold_step > 1 {
            let extra_line_folds = config.fold_step - 1;
            let alpha_sq = folding_alpha * folding_alpha;
            layer_evaluation =
                B::fold_line(&layer_evaluation, alpha_sq, twiddles, extra_line_folds);
        }

        layer_evaluation
    }

    /// Commits to the first FRI layer.
    fn commit_first_layer(
        channel: &mut MC::C,
        config: &FriConfig,
        column: &'a SecureEvaluation<B, BitReversedOrder>,
        memory_mode: ProverMemoryMode,
    ) -> FriFirstLayer<'a, B, MC::H> {
        // The circle-to-line fold is always equal to the config.fold_step.
        // TODO(Leo): consider support for smaller steps.
        if memory_mode.uses_checkpointed_merkle() {
            let layer = LowMemoryFriFirstLayerProver::new(column, config.fold_step);
            MC::mix_root(channel, layer.merkle_tree.root());
            FriFirstLayer::LowMemory(layer)
        } else {
            let layer = FriFirstLayerProver::new(column, config.fold_step);
            MC::mix_root(channel, layer.merkle_tree.root());
            FriFirstLayer::InMemory(layer)
        }
    }

    /// Builds and commits to the inner FRI layers (all layers except the first and last).
    ///
    /// Returns all inner layers and the evaluation of the last layer.
    fn commit_inner_layers(
        channel: &mut MC::C,
        config: FriConfig,
        column: &SecureEvaluation<B, BitReversedOrder>,
        twiddles: &'a TwiddleTree<B>,
        memory_mode: ProverMemoryMode,
    ) -> (FriInnerLayers<'a, B, MC::H>, LineEvaluation<B>) {
        let initial_folding_alpha = channel.draw_secure_felt();
        let mut layer_evaluation = Self::compute_initial_inner_layer_evaluation(
            column,
            config,
            twiddles,
            initial_folding_alpha,
        );
        let mut line_log_size = layer_evaluation.domain().log_size();

        let last_layer_log_domain_size = config.last_layer_domain_size().ilog2();
        assert!(
            line_log_size >= last_layer_log_domain_size,
            "The circle-to-line fold results in a smaller line domain than the last layer."
        );
        // If we're already at the last layer, there are no inner layers to compute.
        if line_log_size == last_layer_log_domain_size {
            let layers = if memory_mode.uses_checkpointed_merkle() {
                FriInnerLayers::LowMemory(LowMemoryFriInnerLayers {
                    initial_folding_alpha,
                    twiddles,
                    layers: Vec::new(),
                })
            } else {
                FriInnerLayers::InMemory(Vec::new())
            };
            return (layers, layer_evaluation);
        }

        if memory_mode.uses_checkpointed_merkle() {
            let mut layers = Vec::new();
            while line_log_size > last_layer_log_domain_size + config.fold_step {
                let (merkle_tree, pack_leaves) =
                    commit_checkpointed_line_layer::<B, MC::H>(&layer_evaluation, config.fold_step);
                MC::mix_root(channel, merkle_tree.root());
                let folding_alpha = channel.draw_secure_felt();
                layer_evaluation =
                    B::fold_line(&layer_evaluation, folding_alpha, twiddles, config.fold_step);
                layers.push(LowMemoryFriInnerLayerProver {
                    merkle_tree,
                    fold_step: config.fold_step,
                    pack_leaves,
                    folding_alpha,
                });
                line_log_size -= config.fold_step;
            }

            let last_fold_step = line_log_size - last_layer_log_domain_size;
            let (merkle_tree, pack_leaves) =
                commit_checkpointed_line_layer::<B, MC::H>(&layer_evaluation, last_fold_step);
            MC::mix_root(channel, merkle_tree.root());
            let folding_alpha = channel.draw_secure_felt();
            layer_evaluation =
                B::fold_line(&layer_evaluation, folding_alpha, twiddles, last_fold_step);
            layers.push(LowMemoryFriInnerLayerProver {
                merkle_tree,
                fold_step: last_fold_step,
                pack_leaves,
                folding_alpha,
            });

            (
                FriInnerLayers::LowMemory(LowMemoryFriInnerLayers {
                    initial_folding_alpha,
                    twiddles,
                    layers,
                }),
                layer_evaluation,
            )
        } else {
            let mut layers = Vec::new();
            // While we can, skip `config.fold_step` layers.
            while line_log_size > last_layer_log_domain_size + config.fold_step {
                let layer = FriInnerLayerProver::new(layer_evaluation, config.fold_step);
                MC::mix_root(channel, layer.merkle_tree.root());
                let folding_alpha = channel.draw_secure_felt();
                layer_evaluation =
                    B::fold_line(&layer.evaluation, folding_alpha, twiddles, config.fold_step);
                layers.push(layer);
                line_log_size -= config.fold_step;
            }

            // Do one last fold (of size 0 < k <= config.fold_step) to reach the correct size.
            let last_fold_step = line_log_size - last_layer_log_domain_size;
            let layer = FriInnerLayerProver::new(layer_evaluation, last_fold_step);
            MC::mix_root(channel, layer.merkle_tree.root());
            let folding_alpha = channel.draw_secure_felt();
            layer_evaluation =
                B::fold_line(&layer.evaluation, folding_alpha, twiddles, last_fold_step);
            layers.push(layer);

            (FriInnerLayers::InMemory(layers), layer_evaluation)
        }
    }

    /// Builds and commits to the last layer.
    ///
    /// The layer is committed to by sending the verifier all the coefficients of the remaining
    /// polynomial.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * The evaluation domain size exceeds the maximum last layer domain size.
    /// * The evaluation is not of sufficiently low degree.
    fn commit_last_layer(
        channel: &mut MC::C,
        config: FriConfig,
        evaluation: LineEvaluation<B>,
    ) -> LinePoly {
        assert_eq!(evaluation.len(), config.last_layer_domain_size());

        let evaluation = evaluation.to_cpu();
        let mut coeffs = evaluation.interpolate().into_ordered_coefficients();

        let last_layer_degree_bound = 1 << config.log_last_layer_degree_bound;
        let zeros = coeffs.split_off(last_layer_degree_bound);
        assert!(zeros.iter().all(SecureField::is_zero), "invalid degree");

        let last_layer_poly = LinePoly::from_ordered_coefficients(coeffs);
        channel.mix_felts(&last_layer_poly);

        last_layer_poly
    }

    /// Returns a FRI proof and the query positions.
    pub fn decommit(self, channel: &mut MC::C) -> FriDecommitResult<MC::H> {
        let first_layer_log_size = self.first_layer.column().domain.log_size();
        let unsorted_query_locations =
            draw_queries(channel, first_layer_log_size, self.config.n_queries);
        let queries = Queries::new(&unsorted_query_locations, first_layer_log_size);

        let fri_proof = self.decommit_on_queries(&queries);
        FriDecommitResult {
            fri_proof,
            query_positions: queries.positions,
            unsorted_query_locations,
        }
    }

    /// # Panics
    ///
    /// Panics if the queries were sampled on the wrong domain size.
    pub fn decommit_on_queries(self, queries: &Queries) -> ExtendedFriProof<MC::H> {
        let Self {
            config,
            first_layer,
            inner_layers,
            last_layer_poly,
        } = self;

        let inner_layer_proofs = match inner_layers {
            FriInnerLayers::InMemory(inner_layers) => inner_layers
                .into_iter()
                .scan(queries.fold(config.fold_step), |layer_queries, layer| {
                    let fold_step = layer.fold_step;
                    let layer_proof = layer.decommit(layer_queries);
                    *layer_queries = layer_queries.fold(fold_step);
                    Some(layer_proof)
                })
                .collect_vec(),
            FriInnerLayers::LowMemory(inner_layers) => {
                let mut layer_queries = queries.fold(config.fold_step);
                let mut layer_evaluation = Self::compute_initial_inner_layer_evaluation(
                    first_layer.column(),
                    config,
                    inner_layers.twiddles,
                    inner_layers.initial_folding_alpha,
                );
                let mut proofs = Vec::with_capacity(inner_layers.layers.len());
                for layer in inner_layers.layers {
                    let fold_step = layer.fold_step;
                    let folding_alpha = layer.folding_alpha;
                    proofs.push(layer.decommit(&layer_evaluation, &layer_queries));
                    layer_queries = layer_queries.fold(fold_step);
                    layer_evaluation = B::fold_line(
                        &layer_evaluation,
                        folding_alpha,
                        inner_layers.twiddles,
                        fold_step,
                    );
                }
                proofs
            }
        };

        let first_layer_proof = match first_layer {
            FriFirstLayer::InMemory(layer) => layer.decommit(queries, config.fold_step),
            FriFirstLayer::LowMemory(layer) => layer.decommit(queries, config.fold_step),
        };

        let (inner_proofs, inner_layers_aux): (Vec<_>, Vec<_>) = inner_layer_proofs
            .into_iter()
            .map(|p| (p.proof, p.aux))
            .unzip();

        ExtendedFriProof {
            proof: FriProof {
                first_layer: first_layer_proof.proof,
                inner_layers: inner_proofs,
                last_layer_poly,
            },
            aux: FriProofAux {
                first_layer: first_layer_proof.aux,
                inner_layers: inner_layers_aux,
            },
        }
    }
}

/// Commitment to the first FRI layer.
struct FriFirstLayerProver<'a, B: FriOps + MerkleOpsLifted<H>, H: MerkleHasherLifted> {
    column: &'a SecureEvaluation<B, BitReversedOrder>,
    merkle_tree: MerkleProverLifted<B, H>,
    pack_leaves: bool,
}

impl<'a, B: FriOps + MerkleOpsLifted<H>, H: MerkleHasherLifted> FriFirstLayerProver<'a, B, H> {
    fn new(first_layer_column: &'a SecureEvaluation<B, BitReversedOrder>, fold_step: u32) -> Self {
        let (merkle_tree, pack_leaves) =
            commit_secure_column_layer::<B, H>(&first_layer_column.values, fold_step);

        FriFirstLayerProver {
            column: first_layer_column,
            merkle_tree,
            pack_leaves,
        }
    }

    fn decommit(self, queries: &Queries, fold_step: u32) -> ExtendedFriLayerProof<H> {
        assert_eq!(queries.log_domain_size, self.column.domain.log_size());

        let (decommitment_positions, column_witness, value_map) =
            compute_decommitment_positions_and_witness_evals(
                self.column,
                &queries.positions,
                fold_step,
            );

        let decommitment_positions = if self.pack_leaves {
            decommitment_positions
                .iter()
                .map(|position| position >> LOG_PACKED_LEAF_SIZE)
                .dedup()
                .collect_vec()
        } else {
            decommitment_positions
        };
        // We can pass an empty vector to the merkle decommit because we don't use its returned
        // opened values.
        // TODO(Leo): consider adding a method to merkle prover to decommit only the auth paths.
        let (_, decommitment) = self
            .merkle_tree
            .decommit(&decommitment_positions, Vec::<&Col<B, BaseField>>::new());
        let commitment = self.merkle_tree.root();

        ExtendedFriLayerProof {
            proof: FriLayerProof {
                fri_witness: column_witness,
                decommitment: decommitment.decommitment,
                commitment,
            },
            aux: FriLayerProofAux {
                all_values: vec![value_map],
                decommitment: decommitment.aux,
            },
        }
    }
}

struct LowMemoryFriFirstLayerProver<'a, B: FriOps + MerkleOpsLifted<H>, H: MerkleHasherLifted> {
    column: &'a SecureEvaluation<B, BitReversedOrder>,
    merkle_tree: CheckpointedMerkleProverLifted<B, H>,
    pack_leaves: bool,
}

impl<'a, B: FriOps + MerkleOpsLifted<H>, H: MerkleHasherLifted>
    LowMemoryFriFirstLayerProver<'a, B, H>
{
    fn new(first_layer_column: &'a SecureEvaluation<B, BitReversedOrder>, fold_step: u32) -> Self {
        let (merkle_tree, pack_leaves) =
            commit_checkpointed_secure_column_layer::<B, H>(&first_layer_column.values, fold_step);

        Self {
            column: first_layer_column,
            merkle_tree,
            pack_leaves,
        }
    }

    fn decommit(self, queries: &Queries, fold_step: u32) -> ExtendedFriLayerProof<H> {
        assert_eq!(queries.log_domain_size, self.column.domain.log_size());

        let (decommitment_positions, column_witness, value_map) =
            compute_decommitment_positions_and_witness_evals(
                self.column,
                &queries.positions,
                fold_step,
            );

        let decommitment_positions = if self.pack_leaves {
            decommitment_positions
                .iter()
                .map(|position| position >> LOG_PACKED_LEAF_SIZE)
                .dedup()
                .collect_vec()
        } else {
            decommitment_positions
        };
        let leaves = build_secure_column_leaves::<B, H>(&self.column.values, self.pack_leaves);
        let commitment = self.merkle_tree.root();
        let decommitment = self
            .merkle_tree
            .decommit_from_leaves(&decommitment_positions, leaves);

        ExtendedFriLayerProof {
            proof: FriLayerProof {
                fri_witness: column_witness,
                decommitment: decommitment.decommitment,
                commitment,
            },
            aux: FriLayerProofAux {
                all_values: vec![value_map],
                decommitment: decommitment.aux,
            },
        }
    }
}

/// A FRI layer comprises of a merkle tree that commits to evaluations of a polynomial.
///
/// The polynomial evaluations are viewed as evaluation of a polynomial on multiple distinct cosets
/// of size two. Each leaf of the merkle tree commits to a single coset evaluation.
// TODO(andrew): Support different step sizes and update docs.
// TODO(andrew): The docs are wrong. Each leaf of the merkle tree commits to a single
// QM31 value. This is inefficient and should be changed.
const fn should_pack_leaves(values_len: usize, fold_step: u32) -> bool {
    values_len.ilog2() >= LOG_PACKED_LEAF_SIZE && fold_step > 1
}

fn commit_secure_column_layer<B: FriOps + MerkleOpsLifted<H>, H: MerkleHasherLifted>(
    column: &SecureColumnByCoords<B>,
    fold_step: u32,
) -> (MerkleProverLifted<B, H>, bool) {
    let pack_leaves = should_pack_leaves(column.len(), fold_step);
    let log_rows_per_leaf = if pack_leaves { LOG_PACKED_LEAF_SIZE } else { 0 };
    let merkle_tree = MerkleProverLifted::commit(
        column.columns.iter().collect_vec(),
        column.len().ilog2() - log_rows_per_leaf,
        log_rows_per_leaf,
    );

    (merkle_tree, pack_leaves)
}

fn commit_checkpointed_secure_column_layer<
    B: FriOps + MerkleOpsLifted<H>,
    H: MerkleHasherLifted,
>(
    column: &SecureColumnByCoords<B>,
    fold_step: u32,
) -> (CheckpointedMerkleProverLifted<B, H>, bool) {
    let pack_leaves = should_pack_leaves(column.len(), fold_step);
    let log_rows_per_leaf = if pack_leaves { LOG_PACKED_LEAF_SIZE } else { 0 };
    let merkle_tree = CheckpointedMerkleProverLifted::commit(
        column.columns.iter().collect_vec(),
        column.len().ilog2() - log_rows_per_leaf,
        log_rows_per_leaf,
        LOW_MEMORY_MERKLE_CHECKPOINT_STRIDE,
    );

    (merkle_tree, pack_leaves)
}

fn commit_line_layer<B: FriOps + MerkleOpsLifted<H>, H: MerkleHasherLifted>(
    evaluation: &LineEvaluation<B>,
    fold_step: u32,
) -> (MerkleProverLifted<B, H>, bool) {
    commit_secure_column_layer::<B, H>(&evaluation.values, fold_step)
}

fn commit_checkpointed_line_layer<B: FriOps + MerkleOpsLifted<H>, H: MerkleHasherLifted>(
    evaluation: &LineEvaluation<B>,
    fold_step: u32,
) -> (CheckpointedMerkleProverLifted<B, H>, bool) {
    commit_checkpointed_secure_column_layer::<B, H>(&evaluation.values, fold_step)
}

fn build_secure_column_leaves<B: FriOps + MerkleOpsLifted<H>, H: MerkleHasherLifted>(
    column: &SecureColumnByCoords<B>,
    pack_leaves: bool,
) -> Col<B, H::Hash> {
    if pack_leaves {
        let columns: [&Col<B, BaseField>; SECURE_EXTENSION_DEGREE] =
            column.columns.iter().collect_vec().try_into().unwrap();
        let packed_columns = B::pack_leaves_input(&columns);
        B::build_leaves(
            &packed_columns.iter().collect_vec(),
            column.len().ilog2() - LOG_PACKED_LEAF_SIZE,
        )
    } else {
        B::build_leaves(&column.columns.iter().collect_vec(), column.len().ilog2())
    }
}

fn decommit_line_layer<B: FriOps + MerkleOpsLifted<H>, H: MerkleHasherLifted>(
    evaluation: &LineEvaluation<B>,
    queries: &Queries,
    fold_step: u32,
    pack_leaves: bool,
    merkle_tree: MerkleProverLifted<B, H>,
) -> ExtendedFriLayerProof<H> {
    let (decommitment_positions, fri_witness, value_map) =
        compute_decommitment_positions_and_witness_evals(&evaluation.values, queries, fold_step);

    let decommitment_positions = if pack_leaves {
        decommitment_positions
            .iter()
            .map(|position| position >> LOG_PACKED_LEAF_SIZE)
            .dedup()
            .collect_vec()
    } else {
        decommitment_positions
    };
    let (_, decommitment) =
        merkle_tree.decommit(&decommitment_positions, Vec::<&Col<B, BaseField>>::new());
    let commitment = merkle_tree.root();

    ExtendedFriLayerProof {
        proof: FriLayerProof {
            fri_witness,
            decommitment: decommitment.decommitment,
            commitment,
        },
        aux: FriLayerProofAux {
            all_values: vec![value_map],
            decommitment: decommitment.aux,
        },
    }
}

struct FriInnerLayerProver<B: FriOps + MerkleOpsLifted<H>, H: MerkleHasherLifted> {
    evaluation: LineEvaluation<B>,
    merkle_tree: MerkleProverLifted<B, H>,
    fold_step: u32,
    pack_leaves: bool,
}

impl<B: FriOps + MerkleOpsLifted<H>, H: MerkleHasherLifted> FriInnerLayerProver<B, H> {
    fn new(evaluation: LineEvaluation<B>, fold_step: u32) -> Self {
        let (merkle_tree, pack_leaves) = commit_line_layer::<B, H>(&evaluation, fold_step);

        FriInnerLayerProver {
            evaluation,
            merkle_tree,
            fold_step,
            pack_leaves,
        }
    }

    fn decommit(self, queries: &Queries) -> ExtendedFriLayerProof<H> {
        decommit_line_layer(
            &self.evaluation,
            queries,
            self.fold_step,
            self.pack_leaves,
            self.merkle_tree,
        )
    }
}

struct LowMemoryFriInnerLayerProver<B: FriOps + MerkleOpsLifted<H>, H: MerkleHasherLifted> {
    merkle_tree: CheckpointedMerkleProverLifted<B, H>,
    fold_step: u32,
    pack_leaves: bool,
    /// The alpha used to fold this layer into the next one during commitment.
    folding_alpha: SecureField,
}

impl<B: FriOps + MerkleOpsLifted<H>, H: MerkleHasherLifted> LowMemoryFriInnerLayerProver<B, H> {
    fn decommit(
        self,
        evaluation: &LineEvaluation<B>,
        queries: &Queries,
    ) -> ExtendedFriLayerProof<H> {
        let (decommitment_positions, fri_witness, value_map) =
            compute_decommitment_positions_and_witness_evals(
                &evaluation.values,
                queries,
                self.fold_step,
            );

        let decommitment_positions = if self.pack_leaves {
            decommitment_positions
                .iter()
                .map(|position| position >> LOG_PACKED_LEAF_SIZE)
                .dedup()
                .collect_vec()
        } else {
            decommitment_positions
        };
        let leaves = build_secure_column_leaves::<B, H>(&evaluation.values, self.pack_leaves);
        let commitment = self.merkle_tree.root();
        let decommitment = self
            .merkle_tree
            .decommit_from_leaves(&decommitment_positions, leaves);

        ExtendedFriLayerProof {
            proof: FriLayerProof {
                fri_witness,
                decommitment: decommitment.decommitment,
                commitment,
            },
            aux: FriLayerProofAux {
                all_values: vec![value_map],
                decommitment: decommitment.aux,
            },
        }
    }
}

/// Returns a column's merkle tree decommitment positions and the evals the verifier can't
/// deduce from previous computations but requires for decommitment and folding.
///
/// Returns a map from leaf index to its value.
fn compute_decommitment_positions_and_witness_evals(
    column: &SecureColumnByCoords<impl PolyOps>,
    query_positions: &[usize],
    fold_step: u32,
) -> (Vec<usize>, Vec<QM31>, HashMap<usize, QM31>) {
    let mut decommitment_positions = Vec::new();
    let mut witness_evals = Vec::new();
    let mut value_map = HashMap::new();

    // Group queries by the folding coset they reside in.
    for subset_queries in query_positions.chunk_by(|a, b| a >> fold_step == b >> fold_step) {
        let subset_start = (subset_queries[0] >> fold_step) << fold_step;
        let subset_decommitment_positions = subset_start..subset_start + (1 << fold_step);
        let mut subset_queries_iter = subset_queries.iter().peekable();

        for position in subset_decommitment_positions {
            // Add decommitment position.
            decommitment_positions.push(position);

            let eval = column.at(position);
            value_map.insert(position, eval);

            // Only add evals the verifier can't calculate.
            if subset_queries_iter.next_if_eq(&&position).is_none() {
                witness_evals.push(eval);
            }
        }
    }

    (decommitment_positions, witness_evals, value_map)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use num_traits::One;

    use crate::core::channel::MerkleChannel;
    use crate::core::circle::{CirclePointIndex, Coset};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
    use crate::core::fri::{FriConfig, FriProof};
    use crate::core::poly::circle::CircleDomain;
    use crate::core::queries::Queries;
    use crate::core::test_utils::test_channel;
    use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
    use crate::core::vcs_lifted::verifier::PACKED_LEAF_SIZE;
    use crate::prover::backend::cpu::CpuCirclePoly;
    use crate::prover::backend::simd::SimdBackend;
    use crate::prover::backend::{Col, Column, CpuBackend};
    use crate::prover::pcs::ProverMemoryMode;
    use crate::prover::poly::circle::{PolyOps, SecureEvaluation};
    use crate::prover::poly::BitReversedOrder;
    use crate::prover::secure_column::SecureColumnByCoords;
    use crate::prover::vcs_lifted::ops::PackLeavesOps;

    /// Default blowup factor used for tests.
    const LOG_BLOWUP_FACTOR: u32 = 2;
    const MEMORY_MEASURE_LOG_DEGREE: u32 = 19;
    const MEMORY_MEASURE_N_QUERIES: usize = 32;
    const MEMORY_MEASURE_FOLD_STEP: u32 = 2;

    type FriProver<'a> = super::FriProver<'a, CpuBackend, Blake2sMerkleChannel>;

    fn assert_fri_proofs_match(
        fast: &FriProof<<Blake2sMerkleChannel as MerkleChannel>::H>,
        low_memory: &FriProof<<Blake2sMerkleChannel as MerkleChannel>::H>,
    ) {
        assert_eq!(fast.last_layer_poly, low_memory.last_layer_poly);
        assert_eq!(
            fast.first_layer.fri_witness,
            low_memory.first_layer.fri_witness
        );
        assert_eq!(
            fast.first_layer.decommitment.hash_witness,
            low_memory.first_layer.decommitment.hash_witness
        );
        assert_eq!(
            fast.first_layer.commitment,
            low_memory.first_layer.commitment
        );
        assert_eq!(fast.inner_layers.len(), low_memory.inner_layers.len());

        for (fast_layer, low_memory_layer) in fast.inner_layers.iter().zip(&low_memory.inner_layers)
        {
            assert_eq!(fast_layer.fri_witness, low_memory_layer.fri_witness);
            assert_eq!(
                fast_layer.decommitment.hash_witness,
                low_memory_layer.decommitment.hash_witness
            );
            assert_eq!(fast_layer.commitment, low_memory_layer.commitment);
        }
    }

    #[test]
    #[should_panic = "invalid degree"]
    fn committing_high_degree_polynomial_fails() {
        const LOG_EXPECTED_BLOWUP_FACTOR: u32 = LOG_BLOWUP_FACTOR;
        const LOG_INVALID_BLOWUP_FACTOR: u32 = LOG_BLOWUP_FACTOR - 1;
        let config = FriConfig::new(2, LOG_EXPECTED_BLOWUP_FACTOR, 3, 1);
        let column = polynomial_evaluation(6, LOG_INVALID_BLOWUP_FACTOR);
        let twiddles = CpuBackend::precompute_twiddles(column.domain.half_coset);

        FriProver::commit(&mut test_channel(), config, &column, &twiddles);
    }

    #[test]
    #[should_panic = "not canonic"]
    fn committing_column_from_invalid_domain_fails() {
        let invalid_domain = CircleDomain::new(Coset::new(CirclePointIndex::generator(), 3));
        assert!(!invalid_domain.is_canonic(), "must be an invalid domain");
        let config = FriConfig::new(2, 2, 3, 1);
        let column = SecureEvaluation::new(
            invalid_domain,
            [SecureField::one(); 1 << 4].into_iter().collect(),
        );
        let twiddles = CpuBackend::precompute_twiddles(column.domain.half_coset);

        FriProver::commit(&mut test_channel(), config, &column, &twiddles);
    }

    /// Returns an evaluation of a random polynomial with degree `2^log_degree`.
    ///
    /// The evaluation domain size is `2^(log_degree + log_blowup_factor)`.
    fn polynomial_evaluation(
        log_degree: u32,
        log_blowup_factor: u32,
    ) -> SecureEvaluation<CpuBackend, BitReversedOrder> {
        let poly = CpuCirclePoly::new(vec![BaseField::one(); 1 << log_degree]);
        let coset = Coset::half_odds(log_degree + log_blowup_factor - 1);
        let domain = CircleDomain::new(coset);
        let values = poly.evaluate(domain);
        SecureEvaluation::new(domain, values.into_iter().map(SecureField::from).collect())
    }

    #[test]
    fn test_fri_commit_decommit_with_jumps() {
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR, 3, 2);
        let column = polynomial_evaluation(6, LOG_BLOWUP_FACTOR);
        let twiddles = CpuBackend::precompute_twiddles(column.domain.half_coset);

        let prover = FriProver::commit(&mut test_channel(), config, &column, &twiddles);
        let queries = Queries::from_positions(vec![0, 3], 6 + LOG_BLOWUP_FACTOR);
        prover.decommit_on_queries(&queries);
    }

    #[test]
    fn test_fri_commit_decommit_with_packed_leaves() {
        for fold_step in 2..=4 {
            let config = FriConfig::new(2, LOG_BLOWUP_FACTOR, 3, fold_step);
            let column = polynomial_evaluation(8, LOG_BLOWUP_FACTOR);
            let twiddles = CpuBackend::precompute_twiddles(column.domain.half_coset);

            let prover = FriProver::commit(&mut test_channel(), config, &column, &twiddles);
            let queries = Queries::from_positions(vec![1, 6, 11], 8 + LOG_BLOWUP_FACTOR);
            prover.decommit_on_queries(&queries);
        }
    }

    #[test]
    fn test_fri_commit_decommit_with_packed_leaves_simd() {
        for fold_step in 2..=4 {
            let config = FriConfig::new(2, LOG_BLOWUP_FACTOR, 3, fold_step);
            let cpu_eval = polynomial_evaluation(8, LOG_BLOWUP_FACTOR);
            let column = SecureEvaluation::new(
                cpu_eval.domain,
                cpu_eval.values.to_vec().into_iter().collect(),
            );
            let twiddles = SimdBackend::precompute_twiddles(column.domain.half_coset);
            let prover = super::FriProver::<'_, SimdBackend, Blake2sMerkleChannel>::commit(
                &mut test_channel(),
                config,
                &column,
                &twiddles,
            );
            let queries = Queries::from_positions(vec![1, 6, 11], 8 + LOG_BLOWUP_FACTOR);
            prover.decommit_on_queries(&queries);
        }
    }

    #[test]
    fn test_pack_leaves_input_simd_matches_cpu() {
        for log_size in 2..8 {
            let values = (0..1 << log_size).map(|i| {
                SecureField::from_m31_array(core::array::from_fn(|coord| {
                    BaseField::from_u32_unchecked((i * SECURE_EXTENSION_DEGREE + coord) as u32)
                }))
            });
            let values = SecureColumnByCoords::<SimdBackend>::from_iter(values);

            let cpu_values = values.to_cpu();
            let cpu_cols: [&Col<CpuBackend, BaseField>; SECURE_EXTENSION_DEGREE] =
                core::array::from_fn(|i| &cpu_values.columns[i]);
            let generic: [Col<CpuBackend, BaseField>; SECURE_EXTENSION_DEGREE * PACKED_LEAF_SIZE] =
                <CpuBackend as PackLeavesOps>::pack_leaves_input(&cpu_cols);
            let simd_cols: [&Col<SimdBackend, BaseField>; SECURE_EXTENSION_DEGREE] =
                core::array::from_fn(|i| &values.columns[i]);
            let simd: [Col<SimdBackend, BaseField>; SECURE_EXTENSION_DEGREE * PACKED_LEAF_SIZE] =
                <SimdBackend as PackLeavesOps>::pack_leaves_input(&simd_cols);

            for col in 0..SECURE_EXTENSION_DEGREE * PACKED_LEAF_SIZE {
                assert_eq!(
                    generic[col].to_cpu(),
                    simd[col].to_cpu(),
                    "mismatch at log_size={log_size}, col={col}"
                );
            }
        }
    }

    #[test]
    fn test_low_memory_fri_matches_fast_cpu() {
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR, 3, 2);
        let column = polynomial_evaluation(8, LOG_BLOWUP_FACTOR);
        let twiddles = CpuBackend::precompute_twiddles(column.domain.half_coset);
        let queries = Queries::from_positions(vec![1, 6, 11], 8 + LOG_BLOWUP_FACTOR);

        let fast = FriProver::commit_with_memory_mode(
            &mut test_channel(),
            config,
            &column,
            &twiddles,
            ProverMemoryMode::Fast,
        )
        .decommit_on_queries(&queries)
        .proof;
        let low_memory = FriProver::commit_with_memory_mode(
            &mut test_channel(),
            config,
            &column,
            &twiddles,
            ProverMemoryMode::LowMemory,
        )
        .decommit_on_queries(&queries)
        .proof;

        assert_fri_proofs_match(&fast, &low_memory);
    }

    #[test]
    fn test_low_memory_fri_matches_fast_simd() {
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR, 3, 2);
        let cpu_eval = polynomial_evaluation(8, LOG_BLOWUP_FACTOR);
        let column = SecureEvaluation::new(
            cpu_eval.domain,
            SecureColumnByCoords::<SimdBackend>::from_cpu(cpu_eval.values),
        );
        let twiddles = SimdBackend::precompute_twiddles(column.domain.half_coset);
        let queries = Queries::from_positions(vec![1, 6, 11], 8 + LOG_BLOWUP_FACTOR);

        let fast =
            super::FriProver::<'_, SimdBackend, Blake2sMerkleChannel>::commit_with_memory_mode(
                &mut test_channel(),
                config,
                &column,
                &twiddles,
                ProverMemoryMode::Fast,
            )
            .decommit_on_queries(&queries)
            .proof;
        let low_memory =
            super::FriProver::<'_, SimdBackend, Blake2sMerkleChannel>::commit_with_memory_mode(
                &mut test_channel(),
                config,
                &column,
                &twiddles,
                ProverMemoryMode::LowMemory,
            )
            .decommit_on_queries(&queries)
            .proof;

        assert_fri_proofs_match(&fast, &low_memory);
    }

    fn run_manual_fri_memory_measurement(memory_mode: ProverMemoryMode) {
        let config = FriConfig::new(
            2,
            LOG_BLOWUP_FACTOR,
            MEMORY_MEASURE_N_QUERIES,
            MEMORY_MEASURE_FOLD_STEP,
        );
        let column = polynomial_evaluation(MEMORY_MEASURE_LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let twiddles = CpuBackend::precompute_twiddles(column.domain.half_coset);
        let queries = Queries::from_positions(
            (0..MEMORY_MEASURE_N_QUERIES).collect(),
            MEMORY_MEASURE_LOG_DEGREE + LOG_BLOWUP_FACTOR,
        );
        let mut channel = test_channel();

        let start = Instant::now();
        let proof = FriProver::commit_with_memory_mode(
            &mut channel,
            config,
            &column,
            &twiddles,
            memory_mode,
        )
        .decommit_on_queries(&queries)
        .proof;
        let elapsed = start.elapsed();

        eprintln!(
            "manual_fri_memory_measurement mode={memory_mode:?} log_degree={} log_blowup={} queries={} fold_step={} elapsed_ms={} inner_layers={}",
            MEMORY_MEASURE_LOG_DEGREE,
            LOG_BLOWUP_FACTOR,
            MEMORY_MEASURE_N_QUERIES,
            MEMORY_MEASURE_FOLD_STEP,
            elapsed.as_millis(),
            proof.inner_layers.len(),
        );
        std::hint::black_box(proof);
    }

    #[test]
    #[ignore = "manual memory measurement"]
    fn measure_fri_memory_fast() {
        run_manual_fri_memory_measurement(ProverMemoryMode::Fast);
    }

    #[test]
    #[ignore = "manual memory measurement"]
    fn measure_fri_memory_low_memory() {
        run_manual_fri_memory_measurement(ProverMemoryMode::LowMemory);
    }
}
