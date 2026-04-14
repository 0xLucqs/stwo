use serde::{Deserialize, Serialize};

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SECURE_EXTENSION_DEGREE;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::PACKED_LEAF_SIZE;
use crate::prover::backend::{Col, ColumnOps};
use crate::prover::spill::HashLayerMmapGuard;

/// Trait for performing Merkle operations on a commitment scheme.
pub trait MerkleOpsLifted<H: MerkleHasherLifted>:
    ColumnOps<BaseField> + ColumnOps<H::Hash> + PackLeavesOps + for<'de> Deserialize<'de> + Serialize
{
    /// Computes the leaves of the lifted Merkle commitment.
    fn build_leaves(columns: &[&Col<Self, BaseField>], lifting_log_size: u32)
        -> Col<Self, H::Hash>;

    /// Given a layer of hashes as input, computes a new layer by hashing pairs
    /// of adjacent elements of the input, as in a standard Merkle tree.
    fn build_next_layer(prev_layer: &Col<Self, H::Hash>) -> Col<Self, H::Hash>;

    /// Low-memory variant of [`Self::build_next_layer`].
    ///
    /// Backends may override this to place the resulting layer on file-backed mmap storage rather
    /// than anonymous heap. The default implementation preserves the existing in-memory behavior.
    fn build_next_layer_with_guard(
        prev_layer: &Col<Self, H::Hash>,
    ) -> (Col<Self, H::Hash>, Option<HashLayerMmapGuard>) {
        (Self::build_next_layer(prev_layer), None)
    }

    /// Builds the first internal Merkle layer directly from columns without materializing the full
    /// leaf hash array. Returns a layer of `2^(lifting_log_size - 1)` hashes.
    ///
    /// `columns` must be non-empty and sorted in increasing order by length.
    /// `lifting_log_size` must be greater than zero.
    ///
    /// The default implementation materializes the full leaf layer (2^lifting_log_size hashes)
    /// then builds the next layer. Backends may override with a streaming implementation that
    /// avoids the leaf layer allocation, reducing peak memory by up to 4 GiB for large trees.
    fn build_first_layer_above_leaves(
        columns: &[&Col<Self, BaseField>],
        lifting_log_size: u32,
    ) -> Col<Self, H::Hash> {
        let leaves = Self::build_leaves(columns, lifting_log_size);
        Self::build_next_layer(&leaves)
    }

    /// Low-memory variant of [`Self::build_first_layer_above_leaves`].
    ///
    /// Backends may override this to store the returned hash layer on file-backed mmap storage.
    fn build_first_layer_above_leaves_with_guard(
        columns: &[&Col<Self, BaseField>],
        lifting_log_size: u32,
    ) -> (Col<Self, H::Hash>, Option<HashLayerMmapGuard>) {
        (
            Self::build_first_layer_above_leaves(columns, lifting_log_size),
            None,
        )
    }
}

pub trait PackLeavesOps: ColumnOps<BaseField> {
    /// Given a column of QM31s (represented as 4 columns of M31s), reshapes it into 4 columns of
    /// QM31s (represented as 16 columns of M31s). Denoting the input column as [v₀, v₁, v₂, v₃,
    /// ...] where vᵢ ∈ QM31, the output is [[v₀, v₄, v₈, ...], [v₁, v₅, v₉, ...], [v₂, v₆, v₁₀,
    /// ...], [v₃, v₇, v₁₁, ...]].
    fn pack_leaves_input(
        values: &[&Col<Self, BaseField>; SECURE_EXTENSION_DEGREE],
    ) -> [Col<Self, BaseField>; SECURE_EXTENSION_DEGREE * PACKED_LEAF_SIZE];
}
