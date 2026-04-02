use super::circle::PolyOps;
use super::BitReversedOrder;
use crate::core::circle::Coset;

/// Precomputed twiddles for a specific coset tower.
///
/// A coset tower is every repeated doubling of a `root_coset`.
/// The largest CircleDomain that can be ffted using these twiddles is one with `root_coset` as
/// its `half_coset`.
pub struct TwiddleTree<B: PolyOps> {
    pub root_coset: Coset,
    // TODO(shahars): Represent a slice, and grabbing, in a generic way
    pub twiddles: B::Twiddles,
    pub itwiddles: B::Twiddles,
}

unsafe impl<B: PolyOps> Sync for TwiddleTree<B> {}

/// Holds file-backed mmap twiddle data. Must be kept alive as long as the TwiddleTree
/// whose Vecs were spilled.
///
/// On drop, the mmap regions are unmapped. The corresponding TwiddleTree's Vecs will have
/// been leaked (mem::forget) so they won't try to dealloc the mmap memory.
pub struct TwiddleMmapGuard {
    _fwd: crate::prover::spill::MmapVec<u32>,
    _inv: crate::prover::spill::MmapVec<u32>,
}

/// Converts a TwiddleTree's backing Vecs to file-backed mmap storage.
///
/// The original Vec memory is freed. New Vecs are created pointing to the mmap data. These
/// new Vecs are intentionally leaked (not deallocated) — the MmapVec in the returned guard
/// owns the memory and will munmap it on drop.
///
/// The guard MUST outlive the TwiddleTree.
pub fn spill_twiddles_to_mmap(
    tree: &mut TwiddleTree<crate::prover::backend::simd::SimdBackend>,
) -> Option<TwiddleMmapGuard> {
    use crate::prover::spill::MmapVec;

    let fwd_data = std::mem::take(&mut tree.twiddles);
    let inv_data = std::mem::take(&mut tree.itwiddles);

    match (MmapVec::from_vec(fwd_data), MmapVec::from_vec(inv_data)) {
        (Ok(fwd_mmap), Ok(inv_mmap)) => {
            // Create Vecs pointing to the mmap data. These Vecs must be leaked before
            // the TwiddleTree is dropped (via unspill_twiddles) because Vec::drop would
            // try to dealloc mmap memory (UB). The MmapVec handles munmap on guard drop.
            // SAFETY: MmapVec data is valid for reads. The guard outlives the tree.
            tree.twiddles = unsafe {
                Vec::from_raw_parts(fwd_mmap.as_ptr() as *mut u32, fwd_mmap.len(), fwd_mmap.len())
            };
            tree.itwiddles = unsafe {
                Vec::from_raw_parts(inv_mmap.as_ptr() as *mut u32, inv_mmap.len(), inv_mmap.len())
            };
            Some(TwiddleMmapGuard {
                _fwd: fwd_mmap,
                _inv: inv_mmap,
            })
        }
        _ => {
            tracing::warn!("Failed to spill twiddles to mmap. Continuing with heap.");
            None
        }
    }
}

/// Must be called before dropping a TwiddleTree whose twiddles were spilled.
/// Replaces the mmap-backed Vecs with empty Vecs so their Drop doesn't dealloc mmap memory.
pub fn unspill_twiddles(
    tree: &mut TwiddleTree<crate::prover::backend::simd::SimdBackend>,
) {
    // Replace with empty Vecs. The old Vecs point to mmap memory — forgetting them
    // prevents Vec::drop from calling dealloc on the mmap addresses.
    let old_fwd = std::mem::take(&mut tree.twiddles);
    let old_inv = std::mem::take(&mut tree.itwiddles);
    std::mem::forget(old_fwd);
    std::mem::forget(old_inv);
}

/// Trait for twiddle buffers that support subdomain extraction.
pub trait TwiddleBuffer<Order> {
    /// Returns an empty twiddle buffer.
    ///
    /// Can be used as a placeholder when the TwiddleBuffer is not needed.
    fn empty() -> Self;

    /// Extracts twiddles for the subdomain `G_{n+1} * <G_{subdomain_log_size}>` of the
    /// canonic coset `G_{n+1} * <G_n>` (where `n = domain_log_size`).
    ///
    /// The buffer may contain twiddles for a domain larger than `domain_log_size`.
    fn extract_subdomain_twiddles(&self, domain_log_size: u32, subdomain_log_size: u32) -> Self;
}

/// In bit-reversed order the subdomain is a prefix, so at each FFT layer we take the
/// first portion of the corresponding layer.
impl<T: Copy> TwiddleBuffer<BitReversedOrder> for Vec<T> {
    fn empty() -> Self {
        Vec::new()
    }

    fn extract_subdomain_twiddles(&self, domain_log_size: u32, subdomain_log_size: u32) -> Self {
        let domain_half_log_size = domain_log_size - 1;
        let subdomain_half_log_size = subdomain_log_size - 1;
        let buf_half_log_size = self.len().ilog2();
        assert!(
            subdomain_half_log_size <= domain_half_log_size
                && domain_half_log_size <= buf_half_log_size,
            "Invalid sizes: subdomain_half={subdomain_half_log_size}, \
             domain_half={domain_half_log_size}, buf_half={buf_half_log_size}"
        );

        let skip_layers = buf_half_log_size - domain_half_log_size;
        let buf_size = 1usize << buf_half_log_size;
        let out_size = 1usize << subdomain_half_log_size;
        let mut result = Vec::with_capacity(out_size);

        for layer in 0..subdomain_half_log_size as usize {
            let root_layer = skip_layers as usize + layer;
            let layer_start = buf_size - (buf_size >> root_layer);
            let subdomain_layer_size = 1usize << (subdomain_half_log_size as usize - 1 - layer);
            result.extend_from_slice(&self[layer_start..layer_start + subdomain_layer_size]);
        }
        // Padding to round the output buffer to a power of two.
        result.push(self[self.len() - 1]);
        debug_assert_eq!(result.len(), out_size);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::poly::circle::CanonicCoset;
    use crate::prover::backend::cpu::circle::slow_precompute_twiddles;

    #[test]
    fn test_extract_subdomain_twiddles() {
        let committed_log_size = 6;
        let subdomain_log_size = 4;

        let committed_domain = CanonicCoset::new(committed_log_size).circle_domain();
        let subdomain = committed_domain
            .split(committed_log_size - subdomain_log_size)
            .0;

        let committed_twiddles = slow_precompute_twiddles(committed_domain.half_coset);
        let extracted: Vec<_> = TwiddleBuffer::<BitReversedOrder>::extract_subdomain_twiddles(
            &committed_twiddles,
            committed_log_size,
            subdomain_log_size,
        );

        let expected = slow_precompute_twiddles(subdomain.half_coset);

        assert_eq!(extracted.len(), expected.len());
        assert_eq!(
            extracted[..extracted.len() - 1],
            expected[..expected.len() - 1]
        );
    }

    #[test]
    fn test_extract_subdomain_twiddles_from_larger_buffer() {
        let root_log_size = 8;
        let committed_log_size = 6;
        let subdomain_log_size = 4;

        let root_domain = CanonicCoset::new(root_log_size).circle_domain();
        let committed_domain = CanonicCoset::new(committed_log_size).circle_domain();
        let subdomain = committed_domain
            .split(committed_log_size - subdomain_log_size)
            .0;

        let root_twiddles = slow_precompute_twiddles(root_domain.half_coset);
        let extracted: Vec<_> = TwiddleBuffer::<BitReversedOrder>::extract_subdomain_twiddles(
            &root_twiddles,
            committed_log_size,
            subdomain_log_size,
        );

        let expected = slow_precompute_twiddles(subdomain.half_coset);

        assert_eq!(extracted.len(), expected.len());
        assert_eq!(
            extracted[..extracted.len() - 1],
            expected[..expected.len() - 1]
        );
    }
}
