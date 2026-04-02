//! Disk-backed coefficient storage for low-memory proving.
//!
//! When proving on memory-constrained devices (e.g., mobile phones), polynomial coefficients
//! can be spilled to a temporary file and memory-mapped. The OS then manages which pages are
//! resident in physical memory, evicting unused pages under memory pressure.
//!
//! This is particularly effective because coefficient access patterns during proving are
//! sequential (OODS evaluation, FFT for materialization), which the kernel readahead handles
//! well.

use std::io::Write;
use std::sync::Arc;

use bytemuck::Pod;
use memmap2::Mmap;
use tempfile::NamedTempFile;

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
    entries: Vec<SpillEntry>,
    /// Keep the file alive so the mmap remains valid.
    _file: NamedTempFile,
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
        Ok(Arc::new(FrozenSpillFile {
            mmap,
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
    pub fn load_vec<T: Pod + Clone>(&self, index: SpillIndex) -> Vec<T> {
        self.get_slice::<T>(index).to_vec()
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
