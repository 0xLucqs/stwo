use std::fmt::Write;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use tracing::info;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProcessMemorySnapshot {
    pub footprint_bytes: u64,
    pub resident_bytes: u64,
    pub virtual_bytes: u64,
    pub wired_bytes: u64,
    pub disk_read_bytes: u64,
    pub disk_written_bytes: u64,
}

#[derive(Clone, Debug)]
struct PhaseMemoryEntry {
    label: String,
    elapsed: Duration,
    snapshot: ProcessMemorySnapshot,
}

#[derive(Debug)]
struct PhaseMemoryLedger {
    session_label: String,
    started_at: Instant,
    entries: Vec<PhaseMemoryEntry>,
}

impl PhaseMemoryLedger {
    fn new(session_label: String) -> Self {
        Self {
            session_label,
            started_at: Instant::now(),
            entries: Vec::new(),
        }
    }

    fn checkpoint(&mut self, label: impl Into<String>) {
        let Some(snapshot) = capture_process_memory() else {
            return;
        };
        let label = label.into();
        let elapsed = self.started_at.elapsed();
        info!(
            session = self.session_label.as_str(),
            checkpoint = label.as_str(),
            elapsed_ms = elapsed.as_millis() as u64,
            footprint_bytes = snapshot.footprint_bytes,
            resident_bytes = snapshot.resident_bytes,
            virtual_bytes = snapshot.virtual_bytes,
            wired_bytes = snapshot.wired_bytes,
            disk_read_bytes = snapshot.disk_read_bytes,
            disk_written_bytes = snapshot.disk_written_bytes,
            "phase_memory_checkpoint"
        );
        self.entries.push(PhaseMemoryEntry {
            label,
            elapsed,
            snapshot,
        });
    }

    fn emit_summary(&self) {
        if self.entries.is_empty() {
            return;
        }

        let peak_footprint = self
            .entries
            .iter()
            .max_by_key(|entry| entry.snapshot.footprint_bytes)
            .unwrap();
        let peak_resident = self
            .entries
            .iter()
            .max_by_key(|entry| entry.snapshot.resident_bytes)
            .unwrap();

        info!(
            session = self.session_label.as_str(),
            checkpoints = self.entries.len() as u64,
            peak_footprint_checkpoint = peak_footprint.label.as_str(),
            peak_footprint_bytes = peak_footprint.snapshot.footprint_bytes,
            peak_footprint_elapsed_ms = peak_footprint.elapsed.as_millis() as u64,
            peak_resident_checkpoint = peak_resident.label.as_str(),
            peak_resident_bytes = peak_resident.snapshot.resident_bytes,
            peak_resident_elapsed_ms = peak_resident.elapsed.as_millis() as u64,
            "phase_memory_summary"
        );

        let mut timeline = String::new();
        let _ = writeln!(
            &mut timeline,
            "elapsed_ms,checkpoint,footprint_bytes,resident_bytes,virtual_bytes,wired_bytes,disk_read_bytes,disk_written_bytes"
        );
        for entry in &self.entries {
            let _ = writeln!(
                &mut timeline,
                "{},{},{},{},{},{},{},{}",
                entry.elapsed.as_millis(),
                entry.label,
                entry.snapshot.footprint_bytes,
                entry.snapshot.resident_bytes,
                entry.snapshot.virtual_bytes,
                entry.snapshot.wired_bytes,
                entry.snapshot.disk_read_bytes,
                entry.snapshot.disk_written_bytes
            );
        }
        info!(
            session = self.session_label.as_str(),
            timeline = %timeline,
            "phase_memory_timeline"
        );
    }
}

fn active_ledger_slot() -> &'static Mutex<Option<Arc<Mutex<PhaseMemoryLedger>>>> {
    static ACTIVE: OnceLock<Mutex<Option<Arc<Mutex<PhaseMemoryLedger>>>>> = OnceLock::new();
    ACTIVE.get_or_init(|| Mutex::new(None))
}

pub fn phase_memory_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("STWO_PHASE_MEMORY_REPORT")
            .map(|value| {
                let value = value.trim().to_ascii_lowercase();
                !(value.is_empty() || value == "0" || value == "false" || value == "off")
            })
            .unwrap_or(false)
    })
}

pub struct PhaseMemorySession {
    ledger: Arc<Mutex<PhaseMemoryLedger>>,
}

impl PhaseMemorySession {
    pub fn start(session_label: impl Into<String>) -> Option<Self> {
        if !phase_memory_enabled() {
            return None;
        }

        let ledger = Arc::new(Mutex::new(PhaseMemoryLedger::new(session_label.into())));
        {
            let mut active = active_ledger_slot().lock().unwrap();
            *active = Some(Arc::clone(&ledger));
        }

        ledger.lock().unwrap().checkpoint("session:start");
        Some(Self { ledger })
    }

    pub fn checkpoint(&self, label: impl Into<String>) {
        self.ledger.lock().unwrap().checkpoint(label);
    }
}

impl Drop for PhaseMemorySession {
    fn drop(&mut self) {
        {
            let mut ledger = self.ledger.lock().unwrap();
            ledger.checkpoint("session:end");
            ledger.emit_summary();
        }

        let mut active = active_ledger_slot().lock().unwrap();
        if active
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, &self.ledger))
        {
            *active = None;
        }
    }
}

pub fn phase_memory_checkpoint(label: impl Into<String>) {
    if !phase_memory_enabled() {
        return;
    }

    let Some(ledger) = active_ledger_slot().lock().unwrap().as_ref().cloned() else {
        return;
    };
    ledger.lock().unwrap().checkpoint(label);
}

#[cfg(target_os = "macos")]
fn capture_process_memory() -> Option<ProcessMemorySnapshot> {
    #[allow(deprecated)]
    let task = unsafe { libc::mach_task_self_ };

    let mut basic_info = std::mem::MaybeUninit::<libc::mach_task_basic_info_data_t>::zeroed();
    let mut basic_count = libc::MACH_TASK_BASIC_INFO_COUNT;
    let basic_result = unsafe {
        libc::task_info(
            task,
            libc::MACH_TASK_BASIC_INFO,
            basic_info.as_mut_ptr().cast(),
            &mut basic_count,
        )
    };
    if basic_result != libc::KERN_SUCCESS {
        return None;
    }

    let mut usage = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    let usage_result = unsafe {
        libc::proc_pid_rusage(
            std::process::id() as libc::c_int,
            libc::RUSAGE_INFO_V4,
            usage.as_mut_ptr().cast(),
        )
    };
    if usage_result != 0 {
        return None;
    }

    let basic_info = unsafe { basic_info.assume_init() };
    let usage = unsafe { usage.assume_init() };

    Some(ProcessMemorySnapshot {
        footprint_bytes: usage.ri_phys_footprint,
        resident_bytes: usage.ri_resident_size,
        virtual_bytes: basic_info.virtual_size,
        wired_bytes: usage.ri_wired_size,
        disk_read_bytes: usage.ri_diskio_bytesread,
        disk_written_bytes: usage.ri_diskio_byteswritten,
    })
}

#[cfg(not(target_os = "macos"))]
fn capture_process_memory() -> Option<ProcessMemorySnapshot> {
    None
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{PhaseMemoryEntry, PhaseMemoryLedger, ProcessMemorySnapshot};

    #[test]
    fn test_phase_memory_summary_prefers_peak_values() {
        let mut ledger = PhaseMemoryLedger::new("test".to_string());
        ledger.entries = vec![
            PhaseMemoryEntry {
                label: "start".to_string(),
                elapsed: Duration::from_millis(0),
                snapshot: ProcessMemorySnapshot {
                    footprint_bytes: 10,
                    resident_bytes: 20,
                    ..Default::default()
                },
            },
            PhaseMemoryEntry {
                label: "mid".to_string(),
                elapsed: Duration::from_millis(5),
                snapshot: ProcessMemorySnapshot {
                    footprint_bytes: 30,
                    resident_bytes: 15,
                    ..Default::default()
                },
            },
            PhaseMemoryEntry {
                label: "end".to_string(),
                elapsed: Duration::from_millis(10),
                snapshot: ProcessMemorySnapshot {
                    footprint_bytes: 25,
                    resident_bytes: 40,
                    ..Default::default()
                },
            },
        ];

        let peak_footprint = ledger
            .entries
            .iter()
            .max_by_key(|entry| entry.snapshot.footprint_bytes)
            .unwrap();
        let peak_resident = ledger
            .entries
            .iter()
            .max_by_key(|entry| entry.snapshot.resident_bytes)
            .unwrap();

        assert_eq!(peak_footprint.label, "mid");
        assert_eq!(peak_resident.label, "end");
    }
}
