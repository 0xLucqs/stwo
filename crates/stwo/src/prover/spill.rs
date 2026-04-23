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
use tempfile::NamedTempFile;

static ACTIVE_MMAPS: AtomicUsize = AtomicUsize::new(0);
static ACTIVE_MMAP_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Emit a probe line to the platform's log sink.
///
/// Stderr is the primary sink (captured by Xcode/oslog on Apple platforms and
/// by most desktop shells on Linux). Android zero-out stderr for app processes
/// by default, so we additionally route to logcat there.
fn emit_probe(line: &str) {
    eprintln!("{line}");
    #[cfg(target_os = "android")]
    android_log::write_info(line);
}

#[cfg(target_os = "android")]
mod android_log {
    use std::ffi::CString;

    use libc::{c_char, c_int};

    const ANDROID_LOG_INFO: c_int = 4;
    const TAG: &str = "stwo_probe";

    #[link(name = "log")]
    extern "C" {
        fn __android_log_write(
            priority: c_int,
            tag: *const c_char,
            text: *const c_char,
        ) -> c_int;
    }

    pub fn write_info(line: &str) {
        let Ok(tag) = CString::new(TAG) else {
            return;
        };
        let Ok(text) = CString::new(line) else {
            return;
        };
        unsafe {
            __android_log_write(ANDROID_LOG_INFO, tag.as_ptr(), text.as_ptr());
        }
    }
}

/// Log and return the current mmap stats.
pub fn log_mmap_stats(label: &str) {
    let count = ACTIVE_MMAPS.load(Ordering::Relaxed);
    let bytes = ACTIVE_MMAP_BYTES.load(Ordering::Relaxed);
    emit_probe(&format!(
        "MMAP_STATS [{label}] active_mmaps={count} active_bytes={:.1} MB",
        bytes as f64 / (1024.0 * 1024.0),
    ));
}

/// Re-entrancy guard for the alloc-error hook so that diagnostic emission
/// inside the hook (which itself may allocate) cannot recurse infinitely.
static ALLOC_HOOK_REENTRY: AtomicUsize = AtomicUsize::new(0);

/// Custom alloc-error hook that, when the global allocator fails, dumps a
/// VM_WALK + MMAP_STATS snapshot before the runtime aborts. This is critical
/// on mobile where `memory allocation of N bytes failed` (iOS) or a silent
/// SIGKILL from LMKd (Android) is often the only signal we get from an
/// ENOMEM event during proving.
fn alloc_error_hook(layout: std::alloc::Layout) {
    let depth = ALLOC_HOOK_REENTRY.fetch_add(1, Ordering::Relaxed);
    if depth == 0 {
        emit_probe(&format!(
            "ALLOC_FAILURE size={} align={} active_mmaps={} active_mmap_bytes={}",
            layout.size(),
            layout.align(),
            ACTIVE_MMAPS.load(Ordering::Relaxed),
            ACTIVE_MMAP_BYTES.load(Ordering::Relaxed),
        ));
        log_mmap_stats("alloc_error");
        log_vm_walk("alloc_error");
    }
    ALLOC_HOOK_REENTRY.fetch_sub(1, Ordering::Relaxed);
    // Returning falls through to the runtime's default abort path, which is
    // what we want — we just wanted to attach diagnostics first.
}

/// Installs the custom alloc-error hook exactly once per process. Safe to
/// call from multiple sites; subsequent calls are no-ops.
pub fn ensure_alloc_error_hook_installed() {
    use std::sync::OnceLock;
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        std::alloc::set_alloc_error_hook(alloc_error_hook);
        emit_probe("ALLOC_HOOK installed");
    });
}

/// Snapshot task-level VM accounting and the largest contiguous free VA gap.
///
/// This is meant to distinguish allocator/phys_footprint retention (shows up as elevated
/// `internal`/`phys_footprint` across proofs) from VA fragmentation (shows up as
/// `largest_free_gap_mb` shrinking below the request size while phys_footprint is fine).
///
/// Emits a single grep-friendly line prefixed with `VM_WALK [label]`.
/// No-op on non-Darwin.
///
/// As a side effect, also ensures the alloc-error hook is installed so that any
/// allocation failure later in the same process dumps a VM_WALK snapshot before
/// abort. The very first VM_WALK call from the FFI (`ffi:before_proof`) thus
/// covers cases where the arena is disabled or the first arena mmap never runs.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn log_vm_walk(label: &str) {
    ensure_alloc_error_hook_installed();
    vm_walk_impl::log_vm_walk(label);
}

#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn log_vm_walk(label: &str) {
    ensure_alloc_error_hook_installed();
    vm_walk_impl_linux::log_vm_walk(label);
}

#[cfg(not(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "linux",
    target_os = "android"
)))]
pub fn log_vm_walk(_label: &str) {
    ensure_alloc_error_hook_installed();
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod vm_walk_impl {
    use std::ffi::c_int;
    use std::time::Instant;

    type MachPortT = u32;
    type KernReturnT = c_int;
    type IntegerT = i32;
    type NaturalT = u32;
    type MachMsgTypeNumberT = NaturalT;
    type MachVmAddressT = u64;
    type MachVmSizeT = u64;

    const KERN_SUCCESS: KernReturnT = 0;
    const TASK_VM_INFO: c_int = 22;
    // REV1: through `max_address`; stable since iOS 10 / macOS 10.12.
    // 20 u64 + 2 i32 + 2 u64 = 168 bytes = 42 natural_t.
    const TASK_VM_INFO_REV1_COUNT: MachMsgTypeNumberT = 42;
    // sizeof(vm_region_submap_info_data_64_t) / sizeof(natural_t) = 19 on current SDKs.
    const VM_REGION_SUBMAP_INFO_COUNT_64: MachMsgTypeNumberT = 19;

    #[repr(C)]
    #[derive(Default, Copy, Clone)]
    struct TaskVmInfoRev1 {
        virtual_size: MachVmSizeT,
        region_count: IntegerT,
        page_size: IntegerT,
        resident_size: MachVmSizeT,
        resident_size_peak: MachVmSizeT,
        device: MachVmSizeT,
        device_peak: MachVmSizeT,
        internal: MachVmSizeT,
        internal_peak: MachVmSizeT,
        external: MachVmSizeT,
        external_peak: MachVmSizeT,
        reusable: MachVmSizeT,
        reusable_peak: MachVmSizeT,
        purgeable_volatile_pmap: MachVmSizeT,
        purgeable_volatile_resident: MachVmSizeT,
        purgeable_volatile_virtual: MachVmSizeT,
        compressed: MachVmSizeT,
        compressed_peak: MachVmSizeT,
        compressed_lifetime: MachVmSizeT,
        phys_footprint: MachVmSizeT,
        min_address: MachVmAddressT,
        max_address: MachVmAddressT,
    }

    extern "C" {
        fn mach_task_self() -> MachPortT;
        fn task_info(
            target_task: MachPortT,
            flavor: c_int,
            info: *mut IntegerT,
            count: *mut MachMsgTypeNumberT,
        ) -> KernReturnT;
        fn mach_vm_region_recurse(
            target_task: MachPortT,
            address: *mut MachVmAddressT,
            size: *mut MachVmSizeT,
            nesting_depth: *mut NaturalT,
            info: *mut IntegerT,
            info_count: *mut MachMsgTypeNumberT,
        ) -> KernReturnT;
    }

    pub fn log_vm_walk(label: &str) {
        const MB: f64 = 1024.0 * 1024.0;
        // Hard caps so the probe never dominates proving time.
        const MAX_REGIONS: u64 = 50_000;
        const WALK_DEADLINE_MS: u128 = 500;
        // Gap size thresholds for the histogram (bytes).
        const MB_256: u64 = 256 << 20;
        const MB_128: u64 = 128 << 20;
        const MB_64: u64 = 64 << 20;
        const MB_16: u64 = 16 << 20;

        let task = unsafe { mach_task_self() };

        let mut info = TaskVmInfoRev1::default();
        let mut count = TASK_VM_INFO_REV1_COUNT;
        let kr = unsafe {
            task_info(
                task,
                TASK_VM_INFO,
                (&mut info as *mut TaskVmInfoRev1).cast::<IntegerT>(),
                &mut count,
            )
        };
        if kr != KERN_SUCCESS {
            eprintln!("VM_WALK [{label}] task_info_failed kr={kr}");
            return;
        }

        let user_min = info.min_address;
        let user_max = info.max_address;

        let started = Instant::now();
        let mut addr: MachVmAddressT = 0;
        let mut prev_end: MachVmAddressT = 0;
        let mut walked: u64 = 0;

        // Whole-walk metrics: every gap mach_vm_region_recurse exposes, including the
        // potentially huge hole between the app's user range and iOS shared cache.
        let mut gaps_all: u64 = 0;
        let mut largest_gap_all: u64 = 0;
        let mut total_free_all: u64 = 0;

        // User-range metrics: clipped to [user_min, user_max].  This is the window the
        // user-mmap allocator actually competes for, and therefore the right number for
        // answering "does a contiguous 128/256 MiB hole exist?"
        let mut gaps_user: u64 = 0;
        let mut largest_gap_user: u64 = 0;
        let mut total_free_user: u64 = 0;
        let mut gaps_ge_256mb: u64 = 0;
        let mut gaps_ge_128mb: u64 = 0;
        let mut gaps_ge_64mb: u64 = 0;
        let mut gaps_ge_16mb: u64 = 0;

        // Region-size distribution within user range (tells us how fragmented the
        // map is: many small regions in a tight user VA is the failure signature).
        let mut regions_in_user: u64 = 0;
        let mut regions_ge_128mb: u64 = 0;
        let mut regions_ge_16mb: u64 = 0;

        loop {
            if walked >= MAX_REGIONS || started.elapsed().as_millis() > WALK_DEADLINE_MS {
                break;
            }
            let mut size: MachVmSizeT = 0;
            let mut depth: NaturalT = 1;
            let mut sub_info = [0 as IntegerT; 32];
            let mut sub_count = VM_REGION_SUBMAP_INFO_COUNT_64;
            let kr = unsafe {
                mach_vm_region_recurse(
                    task,
                    &mut addr,
                    &mut size,
                    &mut depth,
                    sub_info.as_mut_ptr(),
                    &mut sub_count,
                )
            };
            if kr != KERN_SUCCESS {
                // KERN_INVALID_ADDRESS (1) signals end-of-map; anything else is a real error.
                break;
            }

            // Whole-walk gap bookkeeping.
            if addr > prev_end {
                let gap = addr - prev_end;
                gaps_all += 1;
                total_free_all += gap;
                if gap > largest_gap_all {
                    largest_gap_all = gap;
                }
            }

            // User-range gap bookkeeping.  Clip the gap [prev_end, addr) to
            // [user_min, user_max).
            if user_max > user_min {
                let gap_start = prev_end.max(user_min);
                let gap_end = addr.min(user_max);
                if gap_end > gap_start {
                    let gap = gap_end - gap_start;
                    gaps_user += 1;
                    total_free_user += gap;
                    if gap > largest_gap_user {
                        largest_gap_user = gap;
                    }
                    if gap >= MB_256 {
                        gaps_ge_256mb += 1;
                    }
                    if gap >= MB_128 {
                        gaps_ge_128mb += 1;
                    }
                    if gap >= MB_64 {
                        gaps_ge_64mb += 1;
                    }
                    if gap >= MB_16 {
                        gaps_ge_16mb += 1;
                    }
                }
            }

            // Region-size bookkeeping (only for regions whose start lies in user range).
            if addr >= user_min && addr < user_max {
                regions_in_user += 1;
                if size >= MB_128 {
                    regions_ge_128mb += 1;
                }
                if size >= MB_16 {
                    regions_ge_16mb += 1;
                }
            }

            walked += 1;
            prev_end = addr.saturating_add(size);
            addr = prev_end;
        }

        // Tail gap: from the last visited region end up to user_max, if it lies within
        // user range.
        if user_max > user_min {
            let gap_start = prev_end.max(user_min);
            if user_max > gap_start {
                let gap = user_max - gap_start;
                gaps_user += 1;
                total_free_user += gap;
                if gap > largest_gap_user {
                    largest_gap_user = gap;
                }
                if gap >= MB_256 {
                    gaps_ge_256mb += 1;
                }
                if gap >= MB_128 {
                    gaps_ge_128mb += 1;
                }
                if gap >= MB_64 {
                    gaps_ge_64mb += 1;
                }
                if gap >= MB_16 {
                    gaps_ge_16mb += 1;
                }
            }
        }

        let user_va_bytes = user_max.saturating_sub(user_min);
        let user_va_mb = user_va_bytes as f64 / MB;
        let user_used_mb = user_va_bytes.saturating_sub(total_free_user) as f64 / MB;

        eprintln!(
            "VM_WALK [{label}] virt={:.1} MB resident={:.1} MB phys={:.1} MB \
             internal={:.1} MB external={:.1} MB reusable={:.1} MB \
             task_regions={} min=0x{:x} max=0x{:x} \
             user_va_mb={:.1} user_used_mb={:.1} \
             walked={} gaps_all={} largest_all_mb={:.1} total_free_all_mb={:.1} \
             gaps_user={} largest_user_mb={:.1} total_free_user_mb={:.1} \
             gaps_ge_256mb={} gaps_ge_128mb={} gaps_ge_64mb={} gaps_ge_16mb={} \
             regions_in_user={} regions_ge_128mb={} regions_ge_16mb={} walk_ms={}",
            info.virtual_size as f64 / MB,
            info.resident_size as f64 / MB,
            info.phys_footprint as f64 / MB,
            info.internal as f64 / MB,
            info.external as f64 / MB,
            info.reusable as f64 / MB,
            info.region_count,
            info.min_address,
            info.max_address,
            user_va_mb,
            user_used_mb,
            walked,
            gaps_all,
            largest_gap_all as f64 / MB,
            total_free_all as f64 / MB,
            gaps_user,
            largest_gap_user as f64 / MB,
            total_free_user as f64 / MB,
            gaps_ge_256mb,
            gaps_ge_128mb,
            gaps_ge_64mb,
            gaps_ge_16mb,
            regions_in_user,
            regions_ge_128mb,
            regions_ge_16mb,
            started.elapsed().as_millis(),
        );
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
mod vm_walk_impl_linux {
    use std::fs;
    use std::sync::atomic::Ordering;
    use std::time::Instant;

    use super::{emit_probe, ACTIVE_MMAPS, ACTIVE_MMAP_BYTES};

    pub fn log_vm_walk(label: &str) {
        const MB: f64 = 1024.0 * 1024.0;
        let started = Instant::now();

        let status = read_proc_status();
        let rollup = read_smaps_rollup();
        let maps = walk_maps();
        let sys = read_meminfo();

        let active_mmaps = ACTIVE_MMAPS.load(Ordering::Relaxed);
        let active_mmap_bytes = ACTIVE_MMAP_BYTES.load(Ordering::Relaxed);

        emit_probe(&format!(
            "VM_WALK [{label}] virt_mb={:.1} rss_mb={:.1} vm_peak_mb={:.1} \
             vm_hwm_mb={:.1} swap_mb={:.1} rss_anon_mb={:.1} rss_file_mb={:.1} \
             rss_shmem_mb={:.1} pss_mb={:.1} pss_anon_mb={:.1} pss_file_mb={:.1} \
             swap_pss_mb={:.1} task_regions={} anon_regions={} file_regions={} \
             shared_regions={} maps_anon_mb={:.1} maps_file_mb={:.1} \
             stwo_mmaps={} stwo_mmap_mb={:.1} sys_mem_avail_mb={:.1} \
             sys_swap_free_mb={:.1} walk_ms={}",
            status.vm_size_kb as f64 / 1024.0,
            status.vm_rss_kb as f64 / 1024.0,
            status.vm_peak_kb as f64 / 1024.0,
            status.vm_hwm_kb as f64 / 1024.0,
            status.vm_swap_kb as f64 / 1024.0,
            status.rss_anon_kb as f64 / 1024.0,
            status.rss_file_kb as f64 / 1024.0,
            status.rss_shmem_kb as f64 / 1024.0,
            rollup.pss_kb as f64 / 1024.0,
            rollup.pss_anon_kb as f64 / 1024.0,
            rollup.pss_file_kb as f64 / 1024.0,
            rollup.swap_pss_kb as f64 / 1024.0,
            maps.total_regions,
            maps.anon_regions,
            maps.file_regions,
            maps.shared_regions,
            maps.anon_bytes as f64 / MB,
            maps.file_bytes as f64 / MB,
            active_mmaps,
            active_mmap_bytes as f64 / MB,
            sys.mem_available_kb as f64 / 1024.0,
            sys.swap_free_kb as f64 / 1024.0,
            started.elapsed().as_millis(),
        ));
    }

    #[derive(Default)]
    struct ProcStatus {
        vm_size_kb: u64,
        vm_rss_kb: u64,
        vm_peak_kb: u64,
        vm_hwm_kb: u64,
        vm_swap_kb: u64,
        rss_anon_kb: u64,
        rss_file_kb: u64,
        rss_shmem_kb: u64,
    }

    fn read_proc_status() -> ProcStatus {
        let mut out = ProcStatus::default();
        let Ok(text) = fs::read_to_string("/proc/self/status") else {
            return out;
        };
        parse_kb_table(&text, |key, val| match key {
            "VmSize" => out.vm_size_kb = val,
            "VmRSS" => out.vm_rss_kb = val,
            "VmPeak" => out.vm_peak_kb = val,
            "VmHWM" => out.vm_hwm_kb = val,
            "VmSwap" => out.vm_swap_kb = val,
            "RssAnon" => out.rss_anon_kb = val,
            "RssFile" => out.rss_file_kb = val,
            "RssShmem" => out.rss_shmem_kb = val,
            _ => {}
        });
        out
    }

    #[derive(Default)]
    struct SmapsRollup {
        pss_kb: u64,
        pss_anon_kb: u64,
        pss_file_kb: u64,
        swap_pss_kb: u64,
    }

    fn read_smaps_rollup() -> SmapsRollup {
        let mut out = SmapsRollup::default();
        let Ok(text) = fs::read_to_string("/proc/self/smaps_rollup") else {
            return out;
        };
        parse_kb_table(&text, |key, val| match key {
            "Pss" => out.pss_kb = val,
            "Pss_Anon" => out.pss_anon_kb = val,
            "Pss_File" => out.pss_file_kb = val,
            "SwapPss" => out.swap_pss_kb = val,
            _ => {}
        });
        out
    }

    #[derive(Default)]
    struct MemInfo {
        mem_available_kb: u64,
        swap_free_kb: u64,
    }

    fn read_meminfo() -> MemInfo {
        let mut out = MemInfo::default();
        let Ok(text) = fs::read_to_string("/proc/meminfo") else {
            return out;
        };
        parse_kb_table(&text, |key, val| match key {
            "MemAvailable" => out.mem_available_kb = val,
            "SwapFree" => out.swap_free_kb = val,
            _ => {}
        });
        out
    }

    fn parse_kb_table<F: FnMut(&str, u64)>(text: &str, mut f: F) {
        for line in text.lines() {
            let Some((key, rest)) = line.split_once(':') else {
                continue;
            };
            let Some(num) = rest.split_whitespace().next() else {
                continue;
            };
            let Ok(val) = num.parse::<u64>() else {
                continue;
            };
            f(key.trim(), val);
        }
    }

    #[derive(Default)]
    struct MapsSummary {
        total_regions: u64,
        anon_regions: u64,
        file_regions: u64,
        shared_regions: u64,
        anon_bytes: u64,
        file_bytes: u64,
    }

    fn walk_maps() -> MapsSummary {
        let mut out = MapsSummary::default();
        let Ok(text) = fs::read_to_string("/proc/self/maps") else {
            return out;
        };
        for line in text.lines() {
            // Format: START-END PERMS OFFSET DEV INODE [PATHNAME]
            let mut parts = line.split_whitespace();
            let Some(range) = parts.next() else { continue };
            let Some(perms) = parts.next() else { continue };
            let _offset = parts.next();
            let _dev = parts.next();
            let Some(inode) = parts.next() else { continue };
            let pathname = parts.next();

            let Some((start_s, end_s)) = range.split_once('-') else {
                continue;
            };
            let Ok(start) = u64::from_str_radix(start_s, 16) else {
                continue;
            };
            let Ok(end) = u64::from_str_radix(end_s, 16) else {
                continue;
            };
            let size = end.saturating_sub(start);

            out.total_regions += 1;

            // Anonymous: inode=0 and either no path or a kernel pseudo-name
            // ([heap], [stack], [anon:...], [vvar], ...). Shared file-backed
            // mappings (e.g. stwo's MAP_SHARED tempfiles) carry 's' in perms.
            let is_anon = inode == "0"
                && pathname.map_or(true, |p| p.is_empty() || p.starts_with('['));
            if is_anon {
                out.anon_regions += 1;
                out.anon_bytes += size;
            } else if perms.contains('s') {
                out.shared_regions += 1;
                out.file_bytes += size;
            } else {
                out.file_regions += 1;
                out.file_bytes += size;
            }
        }
        out
    }
}

fn track_mmap(bytes: usize) {
    ACTIVE_MMAPS.fetch_add(1, Ordering::Relaxed);
    ACTIVE_MMAP_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

fn track_munmap(bytes: usize) {
    ACTIVE_MMAPS.fetch_sub(1, Ordering::Relaxed);
    ACTIVE_MMAP_BYTES.fetch_sub(bytes, Ordering::Relaxed);
}

enum MappingRelease {
    System,
    #[cfg(all(unix, any(target_os = "macos", target_os = "ios")))]
    MerkleSlot {
        _lease: merkle_slot_pool::SlotLease,
    },
}

struct FileBackedMapping {
    ptr: *mut libc::c_void,
    byte_len: usize,
    _file: NamedTempFile,
    release: MappingRelease,
}

unsafe impl Send for FileBackedMapping {}
unsafe impl Sync for FileBackedMapping {}

#[derive(Clone, Copy, Debug)]
pub enum ContiguousMappingPurpose {
    LiftedMerkleState,
    LiftedMerkleHash,
}

#[cfg(all(unix, any(target_os = "macos", target_os = "ios")))]
mod merkle_slot_pool {
    use std::os::unix::io::AsRawFd;
    use std::sync::{Mutex, OnceLock};

    use super::{ContiguousMappingPurpose, NamedTempFile};

    const STATE_SLOT_BYTES: usize = 256 << 20;
    const HASH_SLOT_BYTES: usize = 128 << 20;
    const DEFAULT_STATE_SLOT_COUNT: usize = 2;
    const DEFAULT_HASH_SLOT_COUNT: usize = 2;
    const STATE_SLOT_COUNT_ENV: &str = "STWO_MERKLE_STATE_SLOT_COUNT";
    const HASH_SLOT_COUNT_ENV: &str = "STWO_MERKLE_HASH_SLOT_COUNT";

    #[derive(Debug)]
    struct SlotPoolState {
        base_addr: usize,
        slot_size: usize,
        free_slots: Vec<bool>,
        label: &'static str,
    }

    impl SlotPoolState {
        fn reserve_slot(&mut self, requested_len: usize) -> Option<SlotLease> {
            if requested_len > self.slot_size {
                return None;
            }
            let slot_idx = self.free_slots.iter().position(|is_free| *is_free)?;
            self.free_slots[slot_idx] = false;
            Some(SlotLease {
                kind: None,
                slot_idx,
                slot_size: self.slot_size,
            })
        }

        fn release_slot(&mut self, slot_idx: usize) {
            self.free_slots[slot_idx] = true;
        }

        fn used_slots(&self) -> usize {
            self.free_slots.iter().filter(|&&free| !free).count()
        }
    }

    pub struct SlotLease {
        kind: Option<ContiguousMappingPurpose>,
        slot_idx: usize,
        slot_size: usize,
    }

    pub fn try_map_file(
        kind: ContiguousMappingPurpose,
        file: &NamedTempFile,
        byte_len: usize,
        prot: i32,
    ) -> std::io::Result<Option<(*mut libc::c_void, SlotLease)>> {
        if byte_len == 0 {
            return Ok(None);
        }
        let Some(pool) = pool_state(kind) else {
            return Ok(None);
        };

        let mut lease = {
            let mut state = pool.lock().expect("merkle slot pool mutex poisoned");
            match state.reserve_slot(byte_len) {
                Some(lease) => lease,
                None => {
                    eprintln!(
                        "MERKLE_SLOT_RESERVE_FAIL kind={} size={} slot_size={} used_slots={}/{}",
                        state.label,
                        byte_len,
                        state.slot_size,
                        state.used_slots(),
                        state.free_slots.len(),
                    );
                    return Ok(None);
                }
            }
        };
        lease.kind = Some(kind);

        let addr = slot_addr(kind, lease.slot_idx);
        let ptr = unsafe {
            libc::mmap(
                addr,
                byte_len,
                prot,
                libc::MAP_FIXED | libc::MAP_SHARED,
                file.as_file().as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            let err = std::io::Error::last_os_error();
            eprintln!(
                "MERKLE_SLOT_MAP_FAIL kind={} slot_idx={} size={} prot={} err={err}",
                pool_label(kind),
                lease.slot_idx,
                byte_len,
                prot,
            );
            if let Some(pool) = pool_state(kind) {
                let mut state = pool.lock().expect("merkle slot pool mutex poisoned");
                state.release_slot(lease.slot_idx);
            }
            return Err(err);
        }

        if let Some(pool) = pool_state(kind) {
            let state = pool.lock().expect("merkle slot pool mutex poisoned");
            eprintln!(
                "MERKLE_SLOT_ALLOC kind={} slot_idx={} size={} used_slots={}/{}",
                state.label,
                lease.slot_idx,
                byte_len,
                state.used_slots(),
                state.free_slots.len(),
            );
        }

        Ok(Some((ptr, lease)))
    }

    impl Drop for SlotLease {
        fn drop(&mut self) {
            let Some(kind) = self.kind else {
                return;
            };
            let Some(pool) = pool_state(kind) else {
                return;
            };
            let addr = slot_addr(kind, self.slot_idx);
            let restore = unsafe {
                libc::mmap(
                    addr,
                    self.slot_size,
                    libc::PROT_NONE,
                    libc::MAP_FIXED | libc::MAP_PRIVATE | libc::MAP_ANON,
                    -1,
                    0,
                )
            };
            if restore == libc::MAP_FAILED {
                eprintln!(
                    "MERKLE_SLOT_RESTORE_FAIL kind={} slot_idx={} bytes={} err={}",
                    pool_label(kind),
                    self.slot_idx,
                    self.slot_size,
                    std::io::Error::last_os_error(),
                );
                return;
            }

            let mut state = pool.lock().expect("merkle slot pool mutex poisoned");
            state.release_slot(self.slot_idx);
            eprintln!(
                "MERKLE_SLOT_DEALLOC kind={} slot_idx={} used_slots={}/{}",
                state.label,
                self.slot_idx,
                state.used_slots(),
                state.free_slots.len(),
            );
        }
    }

    fn pool_state(kind: ContiguousMappingPurpose) -> Option<&'static Mutex<SlotPoolState>> {
        match kind {
            ContiguousMappingPurpose::LiftedMerkleState => {
                static STATE_POOL: OnceLock<Option<Mutex<SlotPoolState>>> = OnceLock::new();
                STATE_POOL
                    .get_or_init(|| {
                        init_pool(
                            "state",
                            STATE_SLOT_BYTES,
                            DEFAULT_STATE_SLOT_COUNT,
                            STATE_SLOT_COUNT_ENV,
                        )
                    })
                    .as_ref()
            }
            ContiguousMappingPurpose::LiftedMerkleHash => {
                static HASH_POOL: OnceLock<Option<Mutex<SlotPoolState>>> = OnceLock::new();
                HASH_POOL
                    .get_or_init(|| {
                        init_pool(
                            "hash",
                            HASH_SLOT_BYTES,
                            DEFAULT_HASH_SLOT_COUNT,
                            HASH_SLOT_COUNT_ENV,
                        )
                    })
                    .as_ref()
            }
        }
    }

    fn slot_addr(kind: ContiguousMappingPurpose, slot_idx: usize) -> *mut libc::c_void {
        let pool = pool_state(kind).expect("merkle slot address requested without pool");
        let state = pool.lock().expect("merkle slot pool mutex poisoned");
        (state.base_addr + slot_idx * state.slot_size) as *mut libc::c_void
    }

    fn init_pool(
        label: &'static str,
        slot_size: usize,
        default_slot_count: usize,
        env_var: &str,
    ) -> Option<Mutex<SlotPoolState>> {
        let page_size = page_size()?;
        let slot_count = std::env::var(env_var)
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .unwrap_or(default_slot_count);
        if slot_count == 0 {
            eprintln!("MERKLE_SLOT_POOL disabled label={label} via {env_var}=0");
            return None;
        }

        let slot_size = align_up(slot_size, page_size);
        let total_len = slot_size.checked_mul(slot_count)?;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                total_len,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            eprintln!(
                "MERKLE_SLOT_POOL init_failed label={label} slot_count={} slot_mb={} err={}",
                slot_count,
                slot_size / (1024 * 1024),
                std::io::Error::last_os_error(),
            );
            return None;
        }

        eprintln!(
            "MERKLE_SLOT_POOL reserved label={label} slot_count={} slot_mb={} base={ptr:p}",
            slot_count,
            slot_size / (1024 * 1024),
        );
        Some(Mutex::new(SlotPoolState {
            base_addr: ptr as usize,
            slot_size,
            free_slots: vec![true; slot_count],
            label,
        }))
    }

    fn page_size() -> Option<usize> {
        let raw = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if raw <= 0 {
            eprintln!("MERKLE_SLOT_POOL failed_to_read_page_size");
            None
        } else {
            Some(raw as usize)
        }
    }

    const fn align_up(value: usize, alignment: usize) -> usize {
        value.div_ceil(alignment) * alignment
    }

    const fn pool_label(kind: ContiguousMappingPurpose) -> &'static str {
        match kind {
            ContiguousMappingPurpose::LiftedMerkleState => "state",
            ContiguousMappingPurpose::LiftedMerkleHash => "hash",
        }
    }

    #[cfg(test)]
    mod tests {
        use super::SlotPoolState;

        #[test]
        fn reserve_release_reuses_slot() {
            let mut state = SlotPoolState {
                base_addr: 0,
                slot_size: 256 << 20,
                free_slots: vec![true, true],
                label: "state",
            };

            let first = state.reserve_slot(256 << 20).unwrap();
            let second = state.reserve_slot(128 << 20).unwrap();
            assert_eq!(first.slot_idx, 0);
            assert_eq!(second.slot_idx, 1);
            assert!(state.reserve_slot(1).is_none());

            state.release_slot(first.slot_idx);
            let third = state.reserve_slot(64 << 20).unwrap();
            assert_eq!(third.slot_idx, 0);
        }

        #[test]
        fn reserve_rejects_oversized_request() {
            let mut state = SlotPoolState {
                base_addr: 0,
                slot_size: 128 << 20,
                free_slots: vec![true],
                label: "hash",
            };

            assert!(state.reserve_slot(129 << 20).is_none());
        }
    }
}

impl FileBackedMapping {
    /// Zero-byte placeholder that still owns the temp file so it survives as long as the
    /// mapping does. Callers must not dereference `ptr` (guaranteed by the `byte_len == 0`
    /// early-returns in `as_bytes` and `Drop`).
    const fn empty(file: NamedTempFile) -> Self {
        Self {
            // Non-null, well-aligned sentinel so `slice::from_raw_parts(ptr, 0)` is defined.
            ptr: std::ptr::NonNull::<u8>::dangling().as_ptr().cast(),
            byte_len: 0,
            _file: file,
            release: MappingRelease::System,
        }
    }

    #[track_caller]
    #[cfg(unix)]
    fn map_named_temp_file_with_reservation(
        file: NamedTempFile,
        byte_len: usize,
        prot: i32,
        reservation: Option<ContiguousMappingPurpose>,
    ) -> std::io::Result<Self> {
        use std::os::unix::io::AsRawFd;

        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let _ = reservation;

        #[cfg(all(unix, any(target_os = "macos", target_os = "ios")))]
        if let Some(reservation) = reservation {
            match merkle_slot_pool::try_map_file(reservation, &file, byte_len, prot) {
                Ok(Some((ptr, lease))) => {
                    track_mmap(byte_len);
                    return Ok(Self {
                        ptr,
                        byte_len,
                        _file: file,
                        release: MappingRelease::MerkleSlot { _lease: lease },
                    });
                }
                Ok(None) => {
                    eprintln!(
                        "MERKLE_SLOT_FALLBACK kind={reservation:?} size={byte_len} -> system_mmap",
                    );
                }
                Err(err) => {
                    eprintln!(
                        "MERKLE_SLOT_FALLBACK kind={reservation:?} size={byte_len} err={err} -> system_mmap",
                    );
                }
            }
        }

        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                byte_len,
                prot,
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
            ptr,
            byte_len,
            _file: file,
            release: MappingRelease::System,
        })
    }

    // `#[track_caller]` so the Location captured inside `try_map_file` skips
    // this frame and identifies the actual stwo consumer (MmapVec::uninitialized
    // caller, spill_eval_columns, mmap_blake2s_hash_layer, etc.).
    #[track_caller]
    #[cfg(unix)]
    fn map_named_temp_file(
        file: NamedTempFile,
        byte_len: usize,
        prot: i32,
    ) -> std::io::Result<Self> {
        Self::map_named_temp_file_with_reservation(file, byte_len, prot, None)
    }

    #[track_caller]
    #[cfg(not(unix))]
    fn map_named_temp_file(
        _file: NamedTempFile,
        _byte_len: usize,
        _prot: i32,
    ) -> std::io::Result<Self> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "file-backed mmap is unsupported on this platform",
        ))
    }

    const fn as_ptr(&self) -> *mut libc::c_void {
        self.ptr
    }

    const fn as_bytes(&self) -> &[u8] {
        // SAFETY: `ptr` and `byte_len` cover the full live mapping owned by `self`.
        unsafe { std::slice::from_raw_parts(self.ptr.cast::<u8>(), self.byte_len) }
    }
}

impl Drop for FileBackedMapping {
    fn drop(&mut self) {
        if self.byte_len == 0 {
            return;
        }

        track_munmap(self.byte_len);
        #[cfg(unix)]
        if matches!(self.release, MappingRelease::System) {
            unsafe {
                libc::munmap(self.ptr, self.byte_len);
            }
        }
    }
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
    mapping: FileBackedMapping,
    entries: Vec<SpillEntry>,
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
        // POSIX `mmap` rejects zero-length mappings with `EINVAL`, so short-circuit when no
        // coefficients were spilled. The resulting `FrozenSpillFile` exposes `len() == 0`
        // and no reads can be issued because `entries` is empty.
        let mapping = if self.offset == 0 {
            FileBackedMapping::empty(file)
        } else {
            FileBackedMapping::map_named_temp_file(file, self.offset as usize, libc::PROT_READ)?
        };
        Ok(Arc::new(FrozenSpillFile {
            mapping,
            entries: self.entries,
        }))
    }
}

impl FrozenSpillFile {
    /// Returns the raw bytes for the coefficient vector at the given index.
    pub fn get_bytes(&self, index: SpillIndex) -> &[u8] {
        let entry = &self.entries[index.0];
        &self.mapping.as_bytes()
            [entry.byte_offset as usize..entry.byte_offset as usize + entry.byte_len]
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
        if allocation_bytes >= (4 << 20) {
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
    _mapping: Option<FileBackedMapping>,
}

unsafe impl<T: Pod> Send for MmapVec<T> {}
unsafe impl<T: Pod> Sync for MmapVec<T> {}

impl<T: Pod> MmapVec<T> {
    #[cfg(unix)]
    fn into_mapping(self) -> Option<FileBackedMapping> {
        let this = std::mem::ManuallyDrop::new(self);
        unsafe { std::ptr::read(&this._mapping) }
    }

    #[cfg(unix)]
    fn uninitialized_with_reservation(
        len: usize,
        reservation: Option<ContiguousMappingPurpose>,
    ) -> std::io::Result<Self> {
        let byte_len = len * std::mem::size_of::<T>();

        if byte_len == 0 {
            return Ok(Self {
                ptr: std::ptr::NonNull::dangling().as_ptr(),
                len: 0,
                byte_len: 0,
                _mapping: None,
            });
        }

        let file = NamedTempFile::new()?;
        file.as_file().set_len(byte_len as u64)?;
        let mapping = FileBackedMapping::map_named_temp_file_with_reservation(
            file,
            byte_len,
            libc::PROT_READ | libc::PROT_WRITE,
            reservation,
        )?;
        Ok(Self {
            ptr: mapping.as_ptr() as *mut T,
            len,
            byte_len,
            _mapping: Some(mapping),
        })
    }

    /// Creates a file-backed mmap of uninitialized storage for `len` elements.
    // `#[track_caller]` so the caller captured inside `try_map_file` points to
    // the actual stwo site (e.g. `allocate_state_layer` in blake2s_lifted.rs),
    // not to this helper.
    #[track_caller]
    #[cfg(unix)]
    pub fn uninitialized(len: usize) -> std::io::Result<Self> {
        Self::uninitialized_with_reservation(len, None)
    }

    #[track_caller]
    #[cfg(unix)]
    pub fn uninitialized_for_contiguous_mapping(
        len: usize,
        reservation: ContiguousMappingPurpose,
    ) -> std::io::Result<Self> {
        Self::uninitialized_with_reservation(len, Some(reservation))
    }

    /// Allocates uninitialized backing storage on non-Unix platforms.
    #[track_caller]
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
    #[track_caller]
    #[cfg(unix)]
    pub fn from_vec(data: Vec<T>) -> std::io::Result<Self> {
        let len = data.len();
        let byte_len = len * std::mem::size_of::<T>();

        if byte_len == 0 {
            return Ok(Self {
                ptr: std::ptr::NonNull::dangling().as_ptr(),
                len: 0,
                byte_len: 0,
                _mapping: None,
            });
        }

        // Write data to temp file.
        let mut file = NamedTempFile::new()?;
        file.write_all(bytemuck::cast_slice(&data))?;
        file.as_file().sync_all()?;

        // Drop the original Vec to free its anonymous heap pages.
        drop(data);

        let mapping = FileBackedMapping::map_named_temp_file(
            file,
            byte_len,
            libc::PROT_READ | libc::PROT_WRITE,
        )?;
        Ok(Self {
            ptr: mapping.as_ptr() as *mut T,
            len,
            byte_len,
            _mapping: Some(mapping),
        })
    }

    /// No-op on non-Unix: just wraps the Vec data.
    #[track_caller]
    #[cfg(not(unix))]
    pub fn from_vec(data: Vec<T>) -> std::io::Result<Self> {
        let len = data.len();
        let byte_len = len * std::mem::size_of::<T>();
        let ptr = Box::into_raw(data.into_boxed_slice()) as *mut T;
        Ok(Self {
            ptr,
            len,
            byte_len,
            _mapping: None,
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
        // Zero-length placeholders never allocated or mmapped anything, so nothing to release.
        // On unix builds the backing `_mapping` (when present) does its own `munmap` on drop.
        if self.byte_len != 0 {
            #[cfg(not(unix))]
            if self._mapping.is_none() {
                // SAFETY: on non-Unix builds `from_vec` wrapped the original allocation with
                // `Box::into_raw(data.into_boxed_slice())`, so reconstructing the Box with the
                // same ptr/len releases it via the global allocator.
                unsafe {
                    let _ = Box::from_raw(std::slice::from_raw_parts_mut(self.ptr, self.len));
                }
            }
        }
    }
}

/// A raw mmap region that needs to be munmapped on drop.
struct MmapRegion {
    _mapping: FileBackedMapping,
}

unsafe impl Send for MmapRegion {}
unsafe impl Sync for MmapRegion {}

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

// Keep eval spill chunks large enough to consolidate many columns into a handful of vm regions
// instead of one mapping per column, while staying small enough to succeed on tight devices.
pub const EVAL_SPILL_CHUNK_BYTES: usize = 128 << 20;

impl EvalMmapGuard {
    pub fn offset_indices(&mut self, offset: usize) {
        self.spilled_indices
            .iter_mut()
            .for_each(|index| *index += offset);
    }
}

#[track_caller]
pub fn mmap_base_column(
    length: usize,
) -> std::io::Result<(
    crate::prover::backend::simd::column::BaseColumn,
    BaseColumnMmapGuard,
)> {
    let packed_len = length.div_ceil(crate::prover::backend::simd::m31::N_LANES);
    let allocation_bytes =
        packed_len * std::mem::size_of::<crate::prover::backend::simd::m31::PackedBaseField>();
    if allocation_bytes >= (4 << 20) {
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

#[track_caller]
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
    if allocation_bytes >= (4 << 20) {
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

#[track_caller]
pub fn mmap_blake2s_hash_layer(
    length: usize,
) -> std::io::Result<(
    Vec<crate::core::vcs::blake2_hash::Blake2sHash>,
    HashLayerMmapGuard,
)> {
    let allocation_bytes =
        length * std::mem::size_of::<crate::core::vcs::blake2_hash::Blake2sHash>();
    if allocation_bytes >= (4 << 20) {
        eprintln!(
            "ALLOC probe {}:{} fn=mmap_blake2s_hash_layer bytes={} logical_len={} element_type={} backing=mmap",
            file!(),
            line!(),
            allocation_bytes,
            length,
            std::any::type_name::<crate::core::vcs::blake2_hash::Blake2sHash>(),
        );
    }
    let mmap = if allocation_bytes >= (128 << 20) {
        MmapVec::uninitialized_for_contiguous_mapping(
            length,
            ContiguousMappingPurpose::LiftedMerkleHash,
        )?
    } else {
        MmapVec::uninitialized(length)?
    };
    let data = unsafe {
        Vec::from_raw_parts(
            mmap.as_ptr() as *mut crate::core::vcs::blake2_hash::Blake2sHash,
            length,
            length,
        )
    };
    Ok((data, HashLayerMmapGuard { _mmap: mmap }))
}

#[derive(Clone, Copy, Debug)]
pub struct EvalColumnLayout {
    pub idx: usize,
    pub logical_len: usize,
}

#[track_caller]
pub fn mmap_eval_column_chunk(
    layouts: &[EvalColumnLayout],
) -> std::io::Result<(
    Vec<crate::prover::backend::simd::column::BaseColumn>,
    EvalMmapGuard,
)> {
    use crate::prover::backend::simd::m31::{PackedBaseField, N_LANES};

    let total_packed_len: usize = layouts
        .iter()
        .map(|layout| layout.logical_len.div_ceil(N_LANES))
        .sum();
    let allocation_bytes = total_packed_len * std::mem::size_of::<PackedBaseField>();
    if allocation_bytes >= (4 << 20) {
        eprintln!(
            "ALLOC probe {}:{} fn=mmap_eval_column_chunk bytes={} columns={} element_type={} backing=mmap",
            file!(),
            line!(),
            allocation_bytes,
            layouts.len(),
            std::any::type_name::<PackedBaseField>(),
        );
    }

    let mmap = MmapVec::<PackedBaseField>::uninitialized(total_packed_len)?;
    let packed_ptr = mmap.as_ptr() as *mut PackedBaseField;
    let mut packed_offset = 0usize;
    let mut columns = Vec::with_capacity(layouts.len());
    let mut spilled_indices = Vec::with_capacity(layouts.len());

    for layout in layouts {
        let packed_len = layout.logical_len.div_ceil(N_LANES);
        let data =
            unsafe { Vec::from_raw_parts(packed_ptr.add(packed_offset), packed_len, packed_len) };
        columns.push(crate::prover::backend::simd::column::BaseColumn {
            data,
            length: layout.logical_len,
        });
        spilled_indices.push(layout.idx);
        packed_offset += packed_len;
    }

    let mapping = mmap
        .into_mapping()
        .expect("non-empty eval chunk mmap must carry mapping");

    Ok((
        columns,
        EvalMmapGuard {
            _regions: vec![MmapRegion { _mapping: mapping }],
            spilled_indices,
        },
    ))
}

/// Replaces evaluation column Vecs with file-backed mmap Vecs for the given polynomials.
///
/// Columns are grouped into consolidated chunks (up to `CHUNK_BYTES` each). Each chunk is
/// written to a single temp file and mapped with one `mmap` call. This keeps the kernel
/// vm_map entry count low (tens instead of thousands) while keeping individual files small
/// enough to succeed under disk/memory pressure.
///
/// After using the evaluations (e.g., for Merkle tree building), call
/// `forget_mmap_backed_evals()` to prevent the Vecs from deallocating mmap memory on drop.
pub fn spill_eval_columns(
    polynomials: &mut [crate::prover::air::component_prover::Poly<
        crate::prover::backend::simd::SimdBackend,
    >],
) -> Option<EvalMmapGuard> {
    use crate::prover::backend::simd::m31::PackedBaseField;

    struct ColEntry {
        idx: usize,
        byte_offset: usize,
        packed_len: usize,
    }

    let mut all_regions = Vec::new();
    let mut all_spilled_indices = Vec::new();

    // Collect indices of polynomials that have evaluations.
    let eval_indices: Vec<usize> = polynomials
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.evals.as_ref().map(|_| i))
        .collect();

    let mut pos = 0;
    while pos < eval_indices.len() {
        // Build one chunk: accumulate columns until we hit CHUNK_BYTES.
        let mut entries = Vec::new();
        let mut file = match NamedTempFile::new() {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!("Eval mmap spill failed (create file): {e}");
                break;
            }
        };
        let mut chunk_bytes: usize = 0;
        while pos < eval_indices.len() {
            let idx = eval_indices[pos];
            let col = &polynomials[idx].evals.as_ref().unwrap().values;
            let packed_len = col.data.len();
            let byte_len = packed_len * std::mem::size_of::<PackedBaseField>();
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(col.data.as_ptr() as *const u8, byte_len) };

            if let Err(e) = file.write_all(bytes) {
                tracing::warn!("Eval mmap spill failed (write): {e}");
                // Stop filling this chunk but try to mmap what we have so far.
                break;
            }

            entries.push(ColEntry {
                idx,
                byte_offset: chunk_bytes,
                packed_len,
            });
            chunk_bytes += byte_len;
            pos += 1;

            if chunk_bytes >= EVAL_SPILL_CHUNK_BYTES {
                break;
            }
        }

        if entries.is_empty() || chunk_bytes == 0 {
            break;
        }

        if let Err(e) = file.as_file().sync_all() {
            tracing::warn!("Eval mmap spill failed (sync): {e}");
            // Skip this chunk, columns stay heap-backed. Try next chunk.
            continue;
        }

        // PROT_READ only: eval columns are read-only after being written to the
        // file. On iOS (no swap), PROT_WRITE forces the kernel to reserve
        // physical pages for potential dirty COW copies, which fails with ENOMEM
        // when total mapped bytes exceed available RAM.
        let mapping =
            match FileBackedMapping::map_named_temp_file(file, chunk_bytes, libc::PROT_READ) {
                Ok(mapping) => mapping,
                Err(e) => {
                    tracing::warn!(
                        "Eval mmap failed ({} cols, {:.1} MB): {e}",
                        entries.len(),
                        chunk_bytes as f64 / (1024.0 * 1024.0)
                    );
                    break;
                }
            };
        let mmap_ptr = mapping.as_ptr();

        // Mmap succeeded — swap each column's Vec to the mmap and drop heap copies.
        for entry in &entries {
            let col = &mut polynomials[entry.idx].evals.as_mut().unwrap().values;
            let _old = std::mem::replace(&mut col.data, unsafe {
                Vec::from_raw_parts(
                    (mmap_ptr as *mut u8).add(entry.byte_offset) as *mut PackedBaseField,
                    entry.packed_len,
                    entry.packed_len,
                )
            });
            all_spilled_indices.push(entry.idx);
        }

        all_regions.push(MmapRegion { _mapping: mapping });
    }

    if all_regions.is_empty() {
        return None;
    }

    tracing::info!(
        "Spilled {} evaluation columns to file-backed mmap ({} mappings)",
        all_spilled_indices.len(),
        all_regions.len(),
    );
    log_mmap_stats("after spill_eval_columns");

    Some(EvalMmapGuard {
        _regions: all_regions,
        spilled_indices: all_spilled_indices,
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

    #[test]
    fn test_mmap_eval_column_chunk_groups_multiple_columns_under_one_guard() {
        let layouts = [
            EvalColumnLayout {
                idx: 3,
                logical_len: 1 << 20,
            },
            EvalColumnLayout {
                idx: 7,
                logical_len: 1 << 19,
            },
            EvalColumnLayout {
                idx: 11,
                logical_len: 1 << 18,
            },
        ];

        let (columns, guard) = mmap_eval_column_chunk(&layouts).unwrap();

        assert_eq!(columns.len(), layouts.len());
        assert_eq!(guard._regions.len(), 1);
        assert_eq!(guard.spilled_indices, vec![3, 7, 11]);
        assert_eq!(columns[0].length, layouts[0].logical_len);
        assert_eq!(columns[1].length, layouts[1].logical_len);
        assert_eq!(columns[2].length, layouts[2].logical_len);

        for column in columns {
            std::mem::forget(column);
        }
        drop(guard);
    }
}
