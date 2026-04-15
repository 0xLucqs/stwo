//! Disk-backed storage for low-memory proving.
//!
//! When proving on memory-constrained devices (e.g., mobile phones), polynomial coefficients
//! and evaluations can be spilled to temporary files and memory-mapped. The OS then manages
//! which pages are resident in physical memory, evicting unused pages under memory pressure.
//!
//! Two spilling strategies are supported:
//!
//! 1. **Spill-and-reload**: Write data to file, drop the in-memory copy, reload later via mmap.
//!    Used for polynomial coefficients that aren't needed during Merkle tree building.
//!
//! 2. **Page replacement (MAP_FIXED)**: Write data to file, then use `mmap(MAP_FIXED)` to replace
//!    the anonymous heap pages backing a Vec with file-backed pages *in place*. The Vec
//!    pointer/length/capacity are unchanged, but the OS can now evict pages under memory pressure
//!    and re-fault them from the file. Used for evaluation data that must remain addressable during
//!    Merkle tree building.

use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use bytemuck::Pod;
use memmap2::Mmap;
use tempfile::NamedTempFile;

static ACTIVE_MMAPS: AtomicUsize = AtomicUsize::new(0);
static ACTIVE_MMAP_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Log and return the current mmap stats.
pub fn log_mmap_stats(label: &str) {
    let count = ACTIVE_MMAPS.load(Ordering::Relaxed);
    let bytes = ACTIVE_MMAP_BYTES.load(Ordering::Relaxed);
    eprintln!(
        "MMAP_STATS [{label}] active_mmaps={count} active_bytes={:.1} MB",
        bytes as f64 / (1024.0 * 1024.0),
    );
}

fn track_mmap(bytes: usize) {
    ACTIVE_MMAPS.fetch_add(1, Ordering::Relaxed);
    ACTIVE_MMAP_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

fn track_munmap(bytes: usize) {
    ACTIVE_MMAPS.fetch_sub(1, Ordering::Relaxed);
    ACTIVE_MMAP_BYTES.fetch_sub(bytes, Ordering::Relaxed);
}

/// A file that stores spilled polynomial coefficient data with an mmap for read access.
///
/// Coefficient vectors are written contiguously. After all writes are complete, the file is
/// frozen (via `freeze()`) and an mmap is created for efficient read-only access.
pub struct CoefficientSpillFile {
    /// Offsets and lengths of each spilled coefficient vector, in bytes.
    entries: Vec<SpillEntry>,
    /// The temporary file holding raw coefficient bytes.
    file: NamedTempFile,
    /// Current write offset in bytes.
    offset: u64,
}

struct SpillEntry {
    byte_offset: u64,
    byte_len: usize,
}

/// A frozen, memory-mapped view of spilled coefficient data.
///
/// Multiple `SpilledCoefficients` references can be cloned from this cheaply.
/// The OS manages physical memory residency: pages not recently accessed are evicted
/// under memory pressure, and re-read from the backing file on next access.
pub struct FrozenSpillFile {
    mmap: Mmap,
    mmap_len: usize,
    entries: Vec<SpillEntry>,
    /// Keep the file alive so the mmap remains valid.
    _file: NamedTempFile,
}

impl Drop for FrozenSpillFile {
    fn drop(&mut self) {
        track_munmap(self.mmap_len);
    }
}

/// A handle to a frozen spill file that can be shared across polynomial structs.
pub type SharedSpillFile = Arc<FrozenSpillFile>;

/// Index into a `FrozenSpillFile` identifying a specific coefficient vector.
#[derive(Clone, Copy, Debug)]
pub struct SpillIndex(pub usize);

impl CoefficientSpillFile {
    /// Creates a new spill file in the system's temporary directory.
    pub fn new() -> std::io::Result<Self> {
        let file = NamedTempFile::new()?;
        Ok(Self {
            entries: Vec::new(),
            file,
            offset: 0,
        })
    }

    /// Writes a coefficient vector to the spill file and returns its index.
    ///
    /// The data is written as raw bytes. The element type `T` must be `Pod` (plain old data)
    /// to ensure the byte representation is stable and safe to reinterpret.
    pub fn write_coefficients<T: Pod>(&mut self, data: &[T]) -> std::io::Result<SpillIndex> {
        self.write_coefficients_raw(bytemuck::cast_slice(data))
    }

    /// Writes raw bytes to the spill file and returns its index.
    pub fn write_coefficients_raw(&mut self, bytes: &[u8]) -> std::io::Result<SpillIndex> {
        self.file.write_all(bytes)?;
        let entry = SpillEntry {
            byte_offset: self.offset,
            byte_len: bytes.len(),
        };
        let index = SpillIndex(self.entries.len());
        self.entries.push(entry);
        self.offset += bytes.len() as u64;
        Ok(index)
    }

    /// Flushes writes and creates a read-only memory map over the spill file.
    ///
    /// After this call, no more writes are possible. The returned `SharedSpillFile` provides
    /// efficient random-access reads via the OS page cache.
    pub fn freeze(self) -> std::io::Result<SharedSpillFile> {
        let file = self.file;
        file.as_file().sync_all()?;
        // SAFETY: The file is fully written and synced. We hold an exclusive reference.
        // The mmap is read-only, and the file is kept alive by the FrozenSpillFile.
        let mmap = unsafe { Mmap::map(file.as_file())? };
        let mmap_len = mmap.len();
        track_mmap(mmap_len);
        Ok(Arc::new(FrozenSpillFile {
            mmap,
            mmap_len,
            entries: self.entries,
            _file: file,
        }))
    }
}

impl FrozenSpillFile {
    /// Returns the raw bytes for the coefficient vector at the given index.
    pub fn get_bytes(&self, index: SpillIndex) -> &[u8] {
        let entry = &self.entries[index.0];
        &self.mmap[entry.byte_offset as usize..entry.byte_offset as usize + entry.byte_len]
    }

    /// Returns the coefficient data at the given index as a slice of `T`.
    ///
    /// # Panics
    ///
    /// Panics if the stored bytes are not properly aligned or sized for `T`.
    pub fn get_slice<T: Pod>(&self, index: SpillIndex) -> &[T] {
        bytemuck::cast_slice(self.get_bytes(index))
    }

    /// Creates a `Vec<T>` by copying data from the mmap. Use this when you need an owned
    /// copy (e.g., to create a `BaseColumn` for SIMD operations that require mutable access
    /// or specific alignment guarantees beyond what mmap provides).
    #[track_caller]
    pub fn load_vec<T: Pod + Clone>(&self, index: SpillIndex) -> Vec<T> {
        let slice = self.get_slice::<T>(index);
        let allocation_bytes = std::mem::size_of_val(slice);
        if allocation_bytes >= (128 << 20) {
            let caller = std::panic::Location::caller();
            eprintln!(
                "ALLOC probe caller={}:{} callee=FrozenSpillFile::load_vec bytes={} logical_len={} element_type={} backing=heap",
                caller.file(),
                caller.line(),
                allocation_bytes,
                slice.len(),
                std::any::type_name::<T>(),
            );
        }
        slice.to_vec()
    }

    /// Returns the number of entries in the spill file.
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the spill file has no entries.
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// A Vec-like container whose backing memory is a file-backed mmap instead of anonymous heap.
///
/// When the OS is under memory pressure, file-backed pages can be evicted and re-faulted from
/// disk. Anonymous heap pages (regular Vec) cannot be evicted on mobile (no swap), causing OOM
/// kills. This type converts a Vec's data to file-backed storage.
///
/// On drop, the mmap is properly unmapped (not free'd via the allocator).
pub struct MmapVec<T: Pod> {
    ptr: *mut T,
    len: usize,
    byte_len: usize,
    _file: NamedTempFile,
}

unsafe impl<T: Pod> Send for MmapVec<T> {}
unsafe impl<T: Pod> Sync for MmapVec<T> {}

impl<T: Pod> MmapVec<T> {
    /// Creates a file-backed mmap of uninitialized storage for `len` elements.
    #[cfg(unix)]
    pub fn uninitialized(len: usize) -> std::io::Result<Self> {
        use std::os::unix::io::AsRawFd;

        let byte_len = len * std::mem::size_of::<T>();

        if byte_len == 0 {
            return Ok(Self {
                ptr: std::ptr::NonNull::dangling().as_ptr(),
                len: 0,
                byte_len: 0,
                _file: NamedTempFile::new()?,
            });
        }

        let file = NamedTempFile::new()?;
        file.as_file().set_len(byte_len as u64)?;

        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                byte_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_file().as_raw_fd(),
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }

        track_mmap(byte_len);
        Ok(Self {
            ptr: ptr as *mut T,
            len,
            byte_len,
            _file: file,
        })
    }

    /// Allocates uninitialized backing storage on non-Unix platforms.
    #[cfg(not(unix))]
    pub fn uninitialized(len: usize) -> std::io::Result<Self> {
        #[allow(clippy::uninit_vec)]
        let data = unsafe {
            let mut data = Vec::with_capacity(len);
            data.set_len(len);
            data
        };
        Self::from_vec(data)
    }

    /// Converts a Vec to file-backed mmap storage.
    ///
    /// Writes the Vec's data to a temp file, mmaps it, and returns the mmap-backed view.
    /// The original Vec's memory is freed.
    #[cfg(unix)]
    pub fn from_vec(data: Vec<T>) -> std::io::Result<Self> {
        use std::os::unix::io::AsRawFd;

        let len = data.len();
        let byte_len = len * std::mem::size_of::<T>();

        if byte_len == 0 {
            return Ok(Self {
                ptr: std::ptr::NonNull::dangling().as_ptr(),
                len: 0,
                byte_len: 0,
                _file: NamedTempFile::new()?,
            });
        }

        // Write data to temp file.
        let mut file = NamedTempFile::new()?;
        file.write_all(bytemuck::cast_slice(&data))?;
        file.as_file().sync_all()?;

        // Drop the original Vec to free its anonymous heap pages.
        drop(data);

        // Mmap the file.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                byte_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_file().as_raw_fd(),
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }

        track_mmap(byte_len);
        Ok(Self {
            ptr: ptr as *mut T,
            len,
            byte_len,
            _file: file,
        })
    }

    /// No-op on non-Unix: just wraps the Vec data.
    #[cfg(not(unix))]
    pub fn from_vec(data: Vec<T>) -> std::io::Result<Self> {
        let len = data.len();
        let byte_len = len * std::mem::size_of::<T>();
        let ptr = Box::into_raw(data.into_boxed_slice()) as *mut T;
        Ok(Self {
            ptr,
            len,
            byte_len,
            _file: NamedTempFile::new()?,
        })
    }
}

impl<T: Pod> std::ops::Deref for MmapVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T: Pod> std::ops::DerefMut for MmapVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T: Pod> Drop for MmapVec<T> {
    fn drop(&mut self) {
        if self.byte_len > 0 {
            track_munmap(self.byte_len);
            #[cfg(unix)]
            unsafe {
                libc::munmap(self.ptr as *mut libc::c_void, self.byte_len);
            }
            #[cfg(not(unix))]
            unsafe {
                // Reconstruct the Box to free the allocation.
                let _ = Box::from_raw(std::slice::from_raw_parts_mut(self.ptr, self.len));
            }
        }
    }
}

/// A raw mmap region that needs to be munmapped on drop.
struct MmapRegion {
    ptr: *mut libc::c_void,
    byte_len: usize,
    _file: NamedTempFile,
}

unsafe impl Send for MmapRegion {}
unsafe impl Sync for MmapRegion {}

impl Drop for MmapRegion {
    fn drop(&mut self) {
        track_munmap(self.byte_len);
        #[cfg(unix)]
        unsafe {
            libc::munmap(self.ptr, self.byte_len);
        }
    }
}

/// Guard holding file-backed mmap data for evaluation columns.
///
/// While this guard is alive, Poly evals may contain Vecs pointing into the mmap data.
/// Those Vecs must be forgotten (not dropped) before this guard is dropped. Call
/// `forget_mmap_backed_evals()` on the polynomials before dropping this guard.
pub struct EvalMmapGuard {
    _regions: Vec<MmapRegion>,
    spilled_indices: Vec<usize>,
}

pub struct BaseColumnMmapGuard {
    _mmap: MmapVec<crate::prover::backend::simd::m31::PackedBaseField>,
}

pub struct SecureEvaluationMmapGuard {
    _columns: [BaseColumnMmapGuard; crate::core::fields::qm31::SECURE_EXTENSION_DEGREE],
}

pub struct HashLayerMmapGuard {
    _mmap: MmapVec<crate::core::vcs::blake2_hash::Blake2sHash>,
}

impl std::fmt::Debug for HashLayerMmapGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HashLayerMmapGuard(..)")
    }
}

impl EvalMmapGuard {
    pub fn offset_indices(&mut self, offset: usize) {
        self.spilled_indices
            .iter_mut()
            .for_each(|index| *index += offset);
    }
}

pub fn mmap_base_column(
    length: usize,
) -> std::io::Result<(
    crate::prover::backend::simd::column::BaseColumn,
    BaseColumnMmapGuard,
)> {
    let packed_len = length.div_ceil(crate::prover::backend::simd::m31::N_LANES);
    let allocation_bytes =
        packed_len * std::mem::size_of::<crate::prover::backend::simd::m31::PackedBaseField>();
    if allocation_bytes >= (128 << 20) {
        eprintln!(
            "ALLOC probe {}:{} fn=mmap_base_column bytes={} logical_len={} packed_len={} element_type={} backing=mmap",
            file!(),
            line!(),
            allocation_bytes,
            length,
            packed_len,
            std::any::type_name::<crate::prover::backend::simd::m31::PackedBaseField>(),
        );
    }
    let mmap = MmapVec::uninitialized(packed_len)?;
    let data = unsafe {
        Vec::from_raw_parts(
            mmap.as_ptr() as *mut crate::prover::backend::simd::m31::PackedBaseField,
            packed_len,
            packed_len,
        )
    };
    Ok((
        crate::prover::backend::simd::column::BaseColumn { data, length },
        BaseColumnMmapGuard { _mmap: mmap },
    ))
}

pub fn mmap_secure_column_by_coords(
    length: usize,
) -> std::io::Result<(
    crate::prover::secure_column::SecureColumnByCoords<crate::prover::backend::simd::SimdBackend>,
    SecureEvaluationMmapGuard,
)> {
    let packed_len = length.div_ceil(crate::prover::backend::simd::m31::N_LANES);
    let coordinate_bytes =
        packed_len * std::mem::size_of::<crate::prover::backend::simd::m31::PackedBaseField>();
    let allocation_bytes = coordinate_bytes * crate::core::fields::qm31::SECURE_EXTENSION_DEGREE;
    if allocation_bytes >= (128 << 20) {
        eprintln!(
            "ALLOC probe {}:{} fn=mmap_secure_column_by_coords bytes={} logical_len={} packed_len={} element_type={} backing=mmap",
            file!(),
            line!(),
            allocation_bytes,
            length,
            packed_len * crate::core::fields::qm31::SECURE_EXTENSION_DEGREE,
            std::any::type_name::<crate::prover::backend::simd::m31::PackedBaseField>(),
        );
    }
    let mut columns = Vec::with_capacity(crate::core::fields::qm31::SECURE_EXTENSION_DEGREE);
    let mut guards = Vec::with_capacity(crate::core::fields::qm31::SECURE_EXTENSION_DEGREE);
    for _ in 0..crate::core::fields::qm31::SECURE_EXTENSION_DEGREE {
        let (column, guard) = mmap_base_column(length)?;
        columns.push(column);
        guards.push(guard);
    }

    let columns = match columns.try_into() {
        Ok(columns) => columns,
        Err(_) => unreachable!("secure evaluation coordinate count is fixed"),
    };
    let guards = match guards.try_into() {
        Ok(guards) => guards,
        Err(_) => unreachable!("secure evaluation coordinate count is fixed"),
    };

    Ok((
        crate::prover::secure_column::SecureColumnByCoords { columns },
        SecureEvaluationMmapGuard { _columns: guards },
    ))
}

pub fn mmap_blake2s_hash_layer(
    length: usize,
) -> std::io::Result<(
    Vec<crate::core::vcs::blake2_hash::Blake2sHash>,
    HashLayerMmapGuard,
)> {
    let allocation_bytes = length * std::mem::size_of::<crate::core::vcs::blake2_hash::Blake2sHash>();
    if allocation_bytes >= (128 << 20) {
        eprintln!(
            "ALLOC probe {}:{} fn=mmap_blake2s_hash_layer bytes={} logical_len={} element_type={} backing=mmap",
            file!(),
            line!(),
            allocation_bytes,
            length,
            std::any::type_name::<crate::core::vcs::blake2_hash::Blake2sHash>(),
        );
    }
    let mmap = MmapVec::uninitialized(length)?;
    let data = unsafe {
        Vec::from_raw_parts(
            mmap.as_ptr() as *mut crate::core::vcs::blake2_hash::Blake2sHash,
            length,
            length,
        )
    };
    Ok((data, HashLayerMmapGuard { _mmap: mmap }))
}

/// Replaces evaluation column Vecs with file-backed mmap Vecs for the given polynomials.
///
/// All columns are written contiguously to a **single** temp file and mapped with one `mmap`
/// call. Each column's `Vec` is then pointed at its offset within the mapping. This keeps the
/// kernel vm_map entry count low (1 instead of N) which is critical on iOS where the per-process
/// mapping limit is tight.
///
/// After using the evaluations (e.g., for Merkle tree building), call
/// `forget_mmap_backed_evals()` to prevent the Vecs from deallocating mmap memory on drop.
pub fn spill_eval_columns(
    polynomials: &mut [crate::prover::air::component_prover::Poly<
        crate::prover::backend::simd::SimdBackend,
    >],
) -> Option<EvalMmapGuard> {
    use crate::prover::backend::simd::m31::PackedBaseField;

    // Phase 1: write all column data to a single temp file without dropping heap copies.
    struct ColEntry {
        idx: usize,
        byte_offset: usize,
        byte_len: usize,
        packed_len: usize,
    }
    let mut entries = Vec::new();
    let mut file = match NamedTempFile::new() {
        Ok(f) => f,
        Err(e) => {
            tracing::warn!("Eval mmap spill failed (create file): {e}");
            return None;
        }
    };
    let mut total_bytes: usize = 0;

    for (idx, poly) in polynomials.iter().enumerate() {
        let evals = match &poly.evals {
            Some(e) => e,
            None => continue,
        };

        let col = &evals.values;
        let packed_len = col.data.len();
        let byte_len = packed_len * std::mem::size_of::<PackedBaseField>();
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(col.data.as_ptr() as *const u8, byte_len) };

        if let Err(e) = file.write_all(bytes) {
            tracing::warn!("Eval mmap spill failed (write column {idx}): {e}");
            // Abort the entire batch — partial writes can't be mapped coherently.
            return None;
        }

        entries.push(ColEntry {
            idx,
            byte_offset: total_bytes,
            byte_len,
            packed_len,
        });
        total_bytes += byte_len;
    }

    if entries.is_empty() || total_bytes == 0 {
        return None;
    }

    if let Err(e) = file.as_file().sync_all() {
        tracing::warn!("Eval mmap spill failed (sync): {e}");
        return None;
    }

    // Phase 2: single mmap over the entire file.
    #[cfg(unix)]
    let mmap_ptr = {
        use std::os::unix::io::AsRawFd;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                total_bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_file().as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            tracing::warn!(
                "Eval mmap failed: {}",
                std::io::Error::last_os_error()
            );
            return None;
        }
        ptr
    };

    #[cfg(not(unix))]
    {
        tracing::warn!("Eval mmap spill not supported on this platform");
        return None;
    }

    track_mmap(total_bytes);

    // Phase 3: mmap succeeded — now swap each column's Vec to point at the mmap and
    // drop the heap copies.
    let mut spilled_indices = Vec::with_capacity(entries.len());
    for entry in &entries {
        let col = &mut polynomials[entry.idx]
            .evals
            .as_mut()
            .expect("eval was present during write phase")
            .values;
        // Drop the heap-backed Vec.
        let _old = std::mem::replace(
            &mut col.data,
            unsafe {
                Vec::from_raw_parts(
                    (mmap_ptr as *mut u8).add(entry.byte_offset) as *mut PackedBaseField,
                    entry.packed_len,
                    entry.packed_len,
                )
            },
        );
        // _old is dropped here, freeing the heap allocation.
        spilled_indices.push(entry.idx);
    }

    tracing::info!(
        "Spilled {} evaluation columns to file-backed mmap ({:.1} MB, 1 mapping)",
        entries.len(),
        total_bytes as f64 / (1024.0 * 1024.0),
    );
    log_mmap_stats("after spill_eval_columns");

    Some(EvalMmapGuard {
        _regions: vec![MmapRegion {
            ptr: mmap_ptr,
            byte_len: total_bytes,
            _file: file,
        }],
        spilled_indices,
    })
}

/// Forgets evaluation Vecs that point to mmap memory, preventing dealloc of mmap pages.
///
/// Must be called before the EvalMmapGuard is dropped if any polynomial evals were spilled.
pub fn forget_mmap_backed_evals(
    polynomials: &mut [crate::prover::air::component_prover::Poly<
        crate::prover::backend::simd::SimdBackend,
    >],
    guards: &[EvalMmapGuard],
) {
    for &index in guards.iter().flat_map(|guard| guard.spilled_indices.iter()) {
        if let Some(evals) = polynomials[index].evals.take() {
            // Forget the CircleEvaluation to prevent Vec::drop on mmap memory.
            // The EvalMmapGuard will handle munmap.
            std::mem::forget(evals);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spill_and_reload() {
        let mut spill = CoefficientSpillFile::new().unwrap();
        let data1: Vec<u32> = (0..1024).collect();
        let data2: Vec<u32> = (1024..2048).collect();

        let idx1 = spill.write_coefficients(&data1).unwrap();
        let idx2 = spill.write_coefficients(&data2).unwrap();

        let frozen = spill.freeze().unwrap();

        let loaded1: Vec<u32> = frozen.load_vec(idx1);
        let loaded2: Vec<u32> = frozen.load_vec(idx2);

        assert_eq!(data1, loaded1);
        assert_eq!(data2, loaded2);
    }

    #[test]
    fn test_spill_get_slice() {
        let mut spill = CoefficientSpillFile::new().unwrap();
        let data: Vec<f32> = (0..256).map(|i| i as f32 * 0.5).collect();

        let idx = spill.write_coefficients(&data).unwrap();
        let frozen = spill.freeze().unwrap();

        let slice: &[f32] = frozen.get_slice(idx);
        assert_eq!(data.as_slice(), slice);
    }

    #[test]
    fn test_empty_spill() {
        let spill = CoefficientSpillFile::new().unwrap();
        let frozen = spill.freeze().unwrap();
        assert!(frozen.is_empty());
        assert_eq!(frozen.len(), 0);
    }
}
