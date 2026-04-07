use hashbrown::HashMap;
use itertools::Itertools;
use tracing::{span, Level};

use super::ops::MerkleOpsLifted;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SECURE_EXTENSION_DEGREE;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::{
    ExtendedMerkleDecommitmentLifted, MerkleDecommitmentLifted, MerkleDecommitmentLiftedAux,
};
use crate::core::ColumnVec;
use crate::prover::backend::{Col, Column};

/// Represents the prover side of a Merkle commitment scheme.
#[derive(Debug)]
pub struct MerkleProverLifted<B: MerkleOpsLifted<H>, H: MerkleHasherLifted> {
    /// Layers of the Merkle tree, sorted by increasing length.
    /// The first layer is a column of length 1, containing the root commitment.
    pub layers: Vec<Col<B, H::Hash>>,
}

#[derive(Debug)]
struct StoredMerkleCheckpoint<B: MerkleOpsLifted<H>, H: MerkleHasherLifted> {
    log_size: u32,
    hashes: Col<B, H::Hash>,
}

/// A lifted Merkle prover that keeps only sparse checkpoint layers in memory and reconstructs the
/// missing layers during decommitment.
#[derive(Debug)]
pub struct CheckpointedMerkleProverLifted<B: MerkleOpsLifted<H>, H: MerkleHasherLifted> {
    root: H::Hash,
    height: u32,
    checkpoints: Vec<StoredMerkleCheckpoint<B, H>>,
}

impl<B: MerkleOpsLifted<H>, H: MerkleHasherLifted> MerkleProverLifted<B, H> {
    /// Commits to columns.
    /// Columns must be of power of 2 sizes, not necessarily sorted by length.
    ///
    /// # Arguments
    ///
    /// * `columns` - A vector of references to columns.
    ///
    /// # Returns
    ///
    /// A new instance of `MerkleProverLifted` with the committed layers.
    pub fn commit(
        columns: Vec<&Col<B, BaseField>>,
        lifting_log_size: u32,
        log_rows_per_leaf: u32,
    ) -> Self {
        let _span = span!(Level::TRACE, "Merkle", class = "MerkleCommitment").entered();
        let mut layers: Vec<Col<B, H::Hash>> = vec![build_leaf_layer::<B, H>(
            columns,
            lifting_log_size,
            log_rows_per_leaf,
        )];

        (0..lifting_log_size).for_each(|_| {
            layers.push(B::build_next_layer(layers.last().unwrap()));
        });
        layers.reverse();

        Self { layers }
    }

    /// Decommits to columns on the given queries.
    /// Queries are given as indices to the largest column.
    ///
    /// # Arguments
    ///
    /// * `queries_position` - Vector containing the positions of the queries, in increasing order.
    /// * `columns` - A vector of references to columns.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * A vector of queried values. For each query position, the queried values are column values
    ///   corresponding to the query position, sorted increasingly by column length.
    /// * A `MerkleDecommitment` containing the hash witness.
    pub fn decommit(
        &self,
        query_positions: &[usize],
        columns: Vec<&Col<B, BaseField>>,
    ) -> (
        ColumnVec<Vec<BaseField>>,
        ExtendedMerkleDecommitmentLifted<H>,
    ) {
        // Prepare output buffers.
        let mut queried_values: ColumnVec<Vec<BaseField>> = vec![];
        let mut decommitment = MerkleDecommitmentLifted::<H>::default();
        let mut all_node_values: Vec<HashMap<usize, <H as MerkleHasherLifted>::Hash>> = vec![];

        // Compute the queried values.
        let max_log_size = self.layers.len() - 1;
        for col in columns.iter() {
            let log_size = col.len().ilog2() as usize;
            let shift = max_log_size - log_size;
            let res: Vec<_> = query_positions
                .iter()
                .map(|pos| col.at((pos >> (shift + 1) << 1) + (pos & 1)))
                .collect();
            queried_values.push(res);
        }

        let mut prev_layer_queries = query_positions.to_vec();
        prev_layer_queries.dedup();
        // The largest log size of a layer is equal to `self.layers.len() - 1`. We start iterating
        // from the layer of log size `self.layers.len() - 2` so that we always have a previous
        // layer available for the computation.
        for layer_log_size in (0..self.layers.len() - 1).rev() {
            let mut all_node_values_for_layer =
                HashMap::<usize, <H as MerkleHasherLifted>::Hash>::new();
            // Prepare write buffer for queries to the current layer. This will propagate to the
            // next layer.
            let mut curr_layer_queries: Vec<usize> = vec![];

            // Each layer node is a hash of column values as previous layer hashes.
            // Prepare the previous layer hashes to read from.
            let prev_layer_hashes = self.layers.get(layer_log_size + 1).unwrap();
            // All chunks have either length 1 (only one child is present) or 2 (both children are
            // present).
            for queries_chunk in prev_layer_queries.as_slice().chunk_by(|a, b| a ^ 1 == *b) {
                let first = queries_chunk[0];
                // If the brother of `first` was not queried before, add its hash to the witness.
                if queries_chunk.len() == 1 {
                    decommitment
                        .hash_witness
                        .push(prev_layer_hashes.at(first ^ 1))
                }
                let curr_index = first >> 1;
                curr_layer_queries.push(curr_index);

                // Add the previous layer hashes to all_node_values.
                all_node_values_for_layer
                    .insert(2 * curr_index, prev_layer_hashes.at(2 * curr_index));
                all_node_values_for_layer
                    .insert(2 * curr_index + 1, prev_layer_hashes.at(2 * curr_index + 1));
            }
            // Propagate queries to the next layer.
            prev_layer_queries = curr_layer_queries;

            all_node_values.push(all_node_values_for_layer);
        }
        (
            queried_values,
            ExtendedMerkleDecommitmentLifted {
                decommitment,
                aux: MerkleDecommitmentLiftedAux { all_node_values },
            },
        )
    }

    pub fn root(&self) -> H::Hash {
        self.layers.first().unwrap().at(0)
    }
}

impl<B: MerkleOpsLifted<H>, H: MerkleHasherLifted> CheckpointedMerkleProverLifted<B, H> {
    pub fn commit(
        columns: Vec<&Col<B, BaseField>>,
        lifting_log_size: u32,
        log_rows_per_leaf: u32,
        checkpoint_stride: u32,
    ) -> Self {
        let _span = span!(Level::TRACE, "Merkle", class = "MerkleCommitment").entered();
        assert!(checkpoint_stride > 0, "checkpoint stride must be positive");

        let height = lifting_log_size;
        let mut checkpoints = Vec::new();

        // For the standard trace-tree path (log_rows_per_leaf == 0, non-empty columns, height > 0)
        // use `build_first_layer_above_leaves` which avoids materialising the full leaf hash
        // layer.  This can save up to 6 GiB of peak anonymous memory for lifting_log_size = 27.
        let (mut current_layer, mut current_log_size) =
            if log_rows_per_leaf == 0 && lifting_log_size > 0 && !columns.is_empty() {
                let sorted_columns = columns.into_iter().sorted_by_key(|c| c.len()).collect_vec();
                (
                    B::build_first_layer_above_leaves(&sorted_columns, lifting_log_size),
                    lifting_log_size - 1,
                )
            } else {
                (
                    build_leaf_layer::<B, H>(columns, lifting_log_size, log_rows_per_leaf),
                    lifting_log_size,
                )
            };

        while current_log_size > 0 {
            let next_layer = B::build_next_layer(&current_layer);
            current_log_size -= 1;
            if should_store_checkpoint(current_log_size, height, checkpoint_stride) {
                checkpoints.push(StoredMerkleCheckpoint {
                    log_size: current_log_size,
                    hashes: next_layer.clone(),
                });
            }
            current_layer = next_layer;
        }
        checkpoints.reverse();

        Self {
            root: current_layer.at(0),
            height,
            checkpoints,
        }
    }

    pub fn from_full_tree(tree: MerkleProverLifted<B, H>, checkpoint_stride: u32) -> Self {
        assert!(checkpoint_stride > 0, "checkpoint stride must be positive");

        let root = tree.root();
        let height = tree.layers.len().saturating_sub(1) as u32;
        let checkpoints = tree
            .layers
            .into_iter()
            .enumerate()
            .filter_map(|(log_size, hashes)| {
                let log_size = log_size as u32;
                should_store_checkpoint(log_size, height, checkpoint_stride)
                    .then_some(StoredMerkleCheckpoint { log_size, hashes })
            })
            .collect();

        Self {
            root,
            height,
            checkpoints,
        }
    }

    pub fn decommit_from_leaves(
        self,
        query_positions: &[usize],
        leaves: Col<B, H::Hash>,
    ) -> ExtendedMerkleDecommitmentLifted<H> {
        assert_eq!(leaves.len().ilog2(), self.height);

        let mut decommitment = MerkleDecommitmentLifted::<H>::default();
        let mut all_node_values: Vec<HashMap<usize, H::Hash>> = vec![];
        let mut prev_layer_queries = query_positions.to_vec();
        prev_layer_queries.dedup();

        let mut current_source_log_size = self.height;
        let mut current_source_hashes = leaves;

        for checkpoint in self.checkpoints.into_iter().rev() {
            Self::decommit_segment(
                current_source_hashes,
                current_source_log_size,
                checkpoint.log_size,
                &mut prev_layer_queries,
                &mut decommitment,
                &mut all_node_values,
            );
            current_source_hashes = checkpoint.hashes;
            current_source_log_size = checkpoint.log_size;
        }

        if current_source_log_size > 0 {
            Self::decommit_segment(
                current_source_hashes,
                current_source_log_size,
                0,
                &mut prev_layer_queries,
                &mut decommitment,
                &mut all_node_values,
            );
        }

        ExtendedMerkleDecommitmentLifted {
            decommitment,
            aux: MerkleDecommitmentLiftedAux { all_node_values },
        }
    }

    pub fn root(&self) -> H::Hash {
        self.root
    }

    pub const fn height(&self) -> u32 {
        self.height
    }

    pub fn decommit_with_source<F>(
        &self,
        query_positions: &[usize],
        mut source_hashes: F,
    ) -> ExtendedMerkleDecommitmentLifted<H>
    where
        F: FnMut(&[usize]) -> HashMap<usize, H::Hash>,
    {
        let mut decommitment = MerkleDecommitmentLifted::<H>::default();
        let mut all_node_values: Vec<HashMap<usize, H::Hash>> = vec![];
        let mut prev_layer_queries = query_positions.to_vec();
        prev_layer_queries.sort_unstable();
        prev_layer_queries.dedup();

        let mut checkpoints = self.checkpoints.iter().rev();
        if let Some(checkpoint) = checkpoints.next() {
            Self::decommit_sparse_segment(
                self.height,
                checkpoint.log_size,
                &mut prev_layer_queries,
                &mut decommitment,
                &mut all_node_values,
                &mut source_hashes,
            );

            let mut current_source_log_size = checkpoint.log_size;
            let mut current_source_hashes = &checkpoint.hashes;
            for checkpoint in checkpoints {
                Self::decommit_sparse_segment_from_layer(
                    current_source_hashes,
                    current_source_log_size,
                    checkpoint.log_size,
                    &mut prev_layer_queries,
                    &mut decommitment,
                    &mut all_node_values,
                );
                current_source_log_size = checkpoint.log_size;
                current_source_hashes = &checkpoint.hashes;
            }

            if current_source_log_size > 0 {
                Self::decommit_sparse_segment_from_layer(
                    current_source_hashes,
                    current_source_log_size,
                    0,
                    &mut prev_layer_queries,
                    &mut decommitment,
                    &mut all_node_values,
                );
            }
        } else if self.height > 0 {
            Self::decommit_sparse_segment(
                self.height,
                0,
                &mut prev_layer_queries,
                &mut decommitment,
                &mut all_node_values,
                &mut source_hashes,
            );
        }

        ExtendedMerkleDecommitmentLifted {
            decommitment,
            aux: MerkleDecommitmentLiftedAux { all_node_values },
        }
    }

    #[cfg(test)]
    fn retained_hash_count(&self) -> usize {
        1 + self
            .checkpoints
            .iter()
            .map(|checkpoint| checkpoint.hashes.len())
            .sum::<usize>()
    }

    fn decommit_segment(
        mut prev_layer_hashes: Col<B, H::Hash>,
        source_log_size: u32,
        target_log_size: u32,
        prev_layer_queries: &mut Vec<usize>,
        decommitment: &mut MerkleDecommitmentLifted<H>,
        all_node_values: &mut Vec<HashMap<usize, H::Hash>>,
    ) {
        assert!(
            source_log_size > target_log_size,
            "invalid checkpoint segment ordering"
        );

        let mut current_source_log_size = source_log_size;
        while current_source_log_size > target_log_size {
            let mut all_node_values_for_layer = HashMap::<usize, H::Hash>::new();
            let mut curr_layer_queries: Vec<usize> = vec![];

            for queries_chunk in prev_layer_queries.as_slice().chunk_by(|a, b| a ^ 1 == *b) {
                let first = queries_chunk[0];
                if queries_chunk.len() == 1 {
                    decommitment
                        .hash_witness
                        .push(prev_layer_hashes.at(first ^ 1));
                }

                let curr_index = first >> 1;
                curr_layer_queries.push(curr_index);
                all_node_values_for_layer
                    .insert(2 * curr_index, prev_layer_hashes.at(2 * curr_index));
                all_node_values_for_layer
                    .insert(2 * curr_index + 1, prev_layer_hashes.at(2 * curr_index + 1));
            }

            all_node_values.push(all_node_values_for_layer);
            *prev_layer_queries = curr_layer_queries;
            current_source_log_size -= 1;
            if current_source_log_size > target_log_size {
                prev_layer_hashes = B::build_next_layer(&prev_layer_hashes);
            }
        }
    }

    fn decommit_sparse_segment<F>(
        source_log_size: u32,
        target_log_size: u32,
        prev_layer_queries: &mut Vec<usize>,
        decommitment: &mut MerkleDecommitmentLifted<H>,
        all_node_values: &mut Vec<HashMap<usize, H::Hash>>,
        source_hashes: &mut F,
    ) where
        F: FnMut(&[usize]) -> HashMap<usize, H::Hash>,
    {
        let segment_layers = Self::build_sparse_segment_layers(
            source_log_size,
            target_log_size,
            prev_layer_queries,
            source_hashes,
        );
        Self::append_sparse_segment_witness(
            prev_layer_queries,
            decommitment,
            all_node_values,
            segment_layers,
        );
    }

    fn decommit_sparse_segment_from_layer(
        source_hashes: &Col<B, H::Hash>,
        source_log_size: u32,
        target_log_size: u32,
        prev_layer_queries: &mut Vec<usize>,
        decommitment: &mut MerkleDecommitmentLifted<H>,
        all_node_values: &mut Vec<HashMap<usize, H::Hash>>,
    ) {
        let segment_layers = Self::build_sparse_segment_layers(
            source_log_size,
            target_log_size,
            prev_layer_queries,
            |positions| {
                positions
                    .iter()
                    .copied()
                    .map(|position| (position, source_hashes.at(position)))
                    .collect()
            },
        );
        Self::append_sparse_segment_witness(
            prev_layer_queries,
            decommitment,
            all_node_values,
            segment_layers,
        );
    }

    fn build_sparse_segment_layers<F>(
        source_log_size: u32,
        target_log_size: u32,
        query_positions: &[usize],
        mut source_hashes: F,
    ) -> Vec<HashMap<usize, H::Hash>>
    where
        F: FnMut(&[usize]) -> HashMap<usize, H::Hash>,
    {
        assert!(
            source_log_size > target_log_size,
            "invalid checkpoint segment ordering"
        );

        let segment_height = source_log_size - target_log_size;
        let block_size = 1usize << segment_height;
        let source_positions = query_positions
            .iter()
            .copied()
            .map(|position| position & !(block_size - 1))
            .sorted()
            .dedup()
            .flat_map(|block_start| block_start..(block_start + block_size))
            .collect_vec();

        let mut layers = vec![source_hashes(&source_positions)];
        let mut current_log_size = source_log_size;
        while current_log_size > target_log_size {
            let prev_layer = layers.last().unwrap();
            let next_layer = prev_layer
                .keys()
                .copied()
                .map(|position| position >> 1)
                .sorted()
                .dedup()
                .map(|position| {
                    (
                        position,
                        H::hash_children((
                            prev_layer[&(2 * position)],
                            prev_layer[&(2 * position + 1)],
                        )),
                    )
                })
                .collect();
            layers.push(next_layer);
            current_log_size -= 1;
        }

        layers
    }

    fn append_sparse_segment_witness(
        prev_layer_queries: &mut Vec<usize>,
        decommitment: &mut MerkleDecommitmentLifted<H>,
        all_node_values: &mut Vec<HashMap<usize, H::Hash>>,
        segment_layers: Vec<HashMap<usize, H::Hash>>,
    ) {
        let mut segment_queries = prev_layer_queries.clone();
        for prev_layer_hashes in segment_layers.iter().take(segment_layers.len() - 1) {
            let mut all_node_values_for_layer = HashMap::<usize, H::Hash>::new();
            let mut curr_layer_queries = Vec::with_capacity(segment_queries.len());

            for queries_chunk in segment_queries.as_slice().chunk_by(|a, b| a ^ 1 == *b) {
                let first = queries_chunk[0];
                if queries_chunk.len() == 1 {
                    decommitment
                        .hash_witness
                        .push(prev_layer_hashes[&(first ^ 1)]);
                }

                let curr_index = first >> 1;
                curr_layer_queries.push(curr_index);
                all_node_values_for_layer
                    .insert(2 * curr_index, prev_layer_hashes[&(2 * curr_index)]);
                all_node_values_for_layer
                    .insert(2 * curr_index + 1, prev_layer_hashes[&(2 * curr_index + 1)]);
            }

            all_node_values.push(all_node_values_for_layer);
            segment_queries = curr_layer_queries;
        }

        *prev_layer_queries = segment_queries;
    }
}

const fn should_store_checkpoint(log_size: u32, height: u32, checkpoint_stride: u32) -> bool {
    log_size > 0 && log_size < height && (height - log_size).is_multiple_of(checkpoint_stride)
}

fn build_leaf_layer<B: MerkleOpsLifted<H>, H: MerkleHasherLifted>(
    columns: Vec<&Col<B, BaseField>>,
    lifting_log_size: u32,
    log_rows_per_leaf: u32,
) -> Col<B, H::Hash> {
    if columns.is_empty() {
        return B::build_leaves(&[], lifting_log_size);
    }

    // We enter this branch only during FRI commit phase, in which we commit 4 columns of the
    // same size. In particular, we don't need to sort the columns by size.
    if log_rows_per_leaf > 0 {
        // TODO(Leo): add support for higher log_rows_per_leaf sizes.
        assert_eq!(
            log_rows_per_leaf, 2,
            "Leaf packing is only supported for log_rows_per_leaf = 2."
        );
        let columns: [&Col<B, BaseField>; SECURE_EXTENSION_DEGREE] = columns.try_into().unwrap();
        let packed_columns = B::pack_leaves_input(&columns);
        let max_log_size = packed_columns[0].len().ilog2();
        assert!(lifting_log_size >= max_log_size);
        B::build_leaves(&packed_columns.iter().collect_vec(), lifting_log_size)
    } else {
        let sorted_columns = columns.into_iter().sorted_by_key(|c| c.len()).collect_vec();
        let max_log_size = sorted_columns.last().unwrap().len().ilog2();
        assert!(lifting_log_size >= max_log_size);
        B::build_leaves(&sorted_columns, lifting_log_size)
    }
}

#[cfg(test)]
mod test {
    use num_traits::Zero;

    use super::*;
    use crate::core::fields::m31::M31;
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::vcs::blake2_hash::{Blake2sHash, Blake2sHasher};
    use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher as Blake2sMerkleHasherCurrent;
    use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;
    use crate::core::vcs_lifted::test_utils::lift_poly;
    use crate::prover::backend::cpu::CpuCirclePoly;
    use crate::prover::backend::CpuBackend;
    use crate::prover::vcs::prover::MerkleProver;

    #[test]
    fn test_empty_cols() {
        // Check Merkle commitment on empty columns.
        let mixed_degree_merkle_prover =
            MerkleProver::<CpuBackend, Blake2sMerkleHasherCurrent>::commit(vec![]);
        let lifted_merkle_prover =
            MerkleProverLifted::<CpuBackend, Blake2sMerkleHasher>::commit(vec![], 0, 0);
        assert_eq!(
            mixed_degree_merkle_prover.layers,
            lifted_merkle_prover.layers
        );
    }

    fn prepare_merkle() -> (
        Vec<Vec<BaseField>>,
        MerkleProverLifted<CpuBackend, Blake2sHasher>,
    ) {
        let max_log_size = 4;
        let columns: Vec<Vec<BaseField>> = (2..=max_log_size)
            .map(|i| (0..1 << i).map(M31::from_u32_unchecked).collect())
            .collect();
        let merkle_prover = MerkleProverLifted::<CpuBackend, Blake2sHasher>::commit(
            columns.iter().collect(),
            max_log_size,
            0,
        );
        (columns, merkle_prover)
    }

    #[test]
    fn test_lifted_merkle_leaves() {
        let (_, merkle_prover) = prepare_merkle();
        let leaves = &merkle_prover.layers.last().unwrap();

        // Compute the expected first leaf.
        let mut hasher = Blake2sHasher::default();
        let data = [0u8; 12];
        hasher.update(&data);
        assert_eq!(hasher.finalize(), leaves[0]);

        // Compute the expected fifth leaf.
        let mut hasher = Blake2sHasher::default();
        let mut data = Vec::new();
        data.extend(0_u32.to_le_bytes());
        data.extend(2_u32.to_le_bytes());
        data.extend(4_u32.to_le_bytes());
        hasher.update(&data);
        assert_eq!(hasher.finalize(), leaves[4]);

        // Compute the expected last leaf.
        let mut hasher = Blake2sHasher::default();
        let mut data = Vec::new();
        data.extend(3_u32.to_le_bytes());
        data.extend(7_u32.to_le_bytes());
        data.extend(15_u32.to_le_bytes());
        hasher.update(&data);

        assert_eq!(hasher.finalize(), *leaves.last().unwrap());
    }

    #[test]
    fn test_lifted_decommitted_values() {
        let (cols, merkle_prover) = prepare_merkle();
        // Test decommits at position 0.
        let queried_values = merkle_prover.decommit(&[0], cols.iter().collect_vec()).0;

        let expected_values = vec![vec![BaseField::zero()]; 3];
        assert_eq!(expected_values, queried_values);

        // Test decommits at position 4.
        let queried_values = merkle_prover.decommit(&[4], cols.iter().collect_vec()).0;
        let expected_values = vec![
            vec![BaseField::from_u32_unchecked(0)],
            vec![BaseField::from_u32_unchecked(2)],
            vec![BaseField::from_u32_unchecked(4)],
        ];
        assert_eq!(expected_values, queried_values);

        // Test decommits at position 15.
        let queried_values = merkle_prover.decommit(&[15], cols.iter().collect_vec()).0;
        let expected_values = vec![
            vec![BaseField::from_u32_unchecked(3)],
            vec![BaseField::from_u32_unchecked(7)],
            vec![BaseField::from_u32_unchecked(15)],
        ];
        assert_eq!(expected_values, queried_values);
    }

    /// See the docs of `[crate::prover::backend::cpu::blake2s_lifted::build_leaves]`.
    #[test]
    fn test_bit_reverse_lifted_merkle_cpu() {
        const LOG_SIZE: u32 = 3;
        const LIFTED_LOG_SIZE: u32 = 9;
        let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
        let poly = CpuCirclePoly::new((0..1 << LOG_SIZE).map(BaseField::from).collect());
        let lifted_evaluation = lift_poly(&poly, LIFTED_LOG_SIZE);

        let last_column: Col<CpuBackend, BaseField> =
            (0..1 << LIFTED_LOG_SIZE).map(|_| M31::zero()).collect_vec();

        let mixed_degree_merkle_prover =
            MerkleProver::<CpuBackend, Blake2sMerkleHasherCurrent>::commit(vec![
                &lifted_evaluation.values,
                &last_column,
            ]);
        let lifted_merkle_prover_1 = MerkleProverLifted::<CpuBackend, Blake2sMerkleHasher>::commit(
            vec![&lifted_evaluation.values, &last_column],
            LIFTED_LOG_SIZE,
            0,
        );
        let lifted_merkle_prover_2 = MerkleProverLifted::<CpuBackend, Blake2sMerkleHasher>::commit(
            vec![&poly.evaluate(domain), &last_column],
            LIFTED_LOG_SIZE,
            0,
        );

        assert_eq!(lifted_merkle_prover_1.root(), lifted_merkle_prover_2.root());
        assert_eq!(
            mixed_degree_merkle_prover.root(),
            lifted_merkle_prover_1.root()
        );
    }

    #[test]
    fn test_decommitment_aux() {
        let (columns, merkle_prover) = prepare_merkle();
        let (
            _,
            ExtendedMerkleDecommitmentLifted {
                decommitment: _,
                aux,
            },
        ) = merkle_prover.decommit(&[1], columns.iter().collect_vec());

        let mut expected: Vec<HashMap<usize, Blake2sHash>> = vec![];
        merkle_prover
            .layers
            .iter()
            .skip(1)
            .rev()
            .for_each(|layer| expected.push(HashMap::from_iter([(0, layer[0]), (1, layer[1])])));
        assert_eq!(expected, aux.all_node_values);
    }

    #[test]
    fn test_checkpointed_decommitment_matches_full_tree() {
        let (columns, merkle_prover) = prepare_merkle();
        let queries = [1, 4, 5, 9, 15];
        let leaves = merkle_prover.layers.last().unwrap().clone();
        let retained_full_hashes: usize = merkle_prover.layers.iter().map(Column::len).sum();
        let full_root = merkle_prover.root();

        let (
            _,
            ExtendedMerkleDecommitmentLifted {
                decommitment: full_decommitment,
                aux: full_aux,
            },
        ) = merkle_prover.decommit(&queries, columns.iter().collect_vec());
        let checkpointed = CheckpointedMerkleProverLifted::<CpuBackend, Blake2sHasher>::commit(
            columns.iter().collect_vec(),
            4,
            0,
            2,
        );
        let checkpointed_root = checkpointed.root();
        let retained_checkpointed_hashes = checkpointed.retained_hash_count();
        let ExtendedMerkleDecommitmentLifted {
            decommitment: checkpointed_decommitment,
            aux: checkpointed_aux,
        } = checkpointed.decommit_from_leaves(&queries, leaves);

        assert!(retained_checkpointed_hashes < retained_full_hashes);
        assert_eq!(checkpointed_root, full_root);
        assert_eq!(
            checkpointed_decommitment.hash_witness,
            full_decommitment.hash_witness
        );
        assert_eq!(checkpointed_aux.all_node_values, full_aux.all_node_values);
    }
}
