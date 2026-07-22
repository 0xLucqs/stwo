#![feature(portable_simd)]

use std::array;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::hint::black_box;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::simd::u32x16;
use std::time::{Duration, Instant};

use num_traits::{One, Zero};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::FieldExpOps;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;
use stwo::prover::backend::simd::blake2s::{compress16, INITIAL_STATE, ZEROS};
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::backend::simd::qm31::PackedQM31;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::Column;
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::vcs_lifted::prover::MerkleProverLifted;
use stwo_examples::gkr_e2e::{benchmark_path_a, PathAReport};

const SUITE_VERSION: &str = "v1";
const E2E_CELLS: [(u32, usize); 4] = [(14, 4), (16, 4), (16, 16), (18, 4)];
const KERNEL_LOG_NS: [u32; 3] = [14, 16, 18];
const COLD_WARM_CELL: (u32, usize) = (16, 4);
const E2E_RECORDED_RUNS: usize = 5;
const INNER_ITERATIONS: usize = 7;
const MERKLE_COLUMNS: usize = 16;
const QM31_MUL_ITERATIONS: usize = 32;
const HASH_BATCHES: usize = 1 << 12;
const HASH_MESSAGE_BYTES: usize = 64;
const HASH_LANES: usize = 16;

#[derive(Clone, Debug)]
pub struct SuiteConfig {
    pub threads: usize,
}

impl Default for SuiteConfig {
    fn default() -> Self {
        let threads = std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|&threads| threads > 0)
            .or_else(|| std::thread::available_parallelism().ok().map(usize::from))
            .unwrap_or(1);
        Self { threads }
    }
}

struct SuitePlan<'a> {
    e2e_cells: &'a [(u32, usize)],
    kernel_log_ns: &'a [u32],
    cold_warm_cell: Option<(u32, usize)>,
    e2e_recorded_runs: usize,
    inner_iterations: usize,
    qm31_mul_iterations: usize,
    hash_batches: usize,
}

const DEFAULT_PLAN: SuitePlan<'static> = SuitePlan {
    e2e_cells: &E2E_CELLS,
    kernel_log_ns: &KERNEL_LOG_NS,
    cold_warm_cell: Some(COLD_WARM_CELL),
    e2e_recorded_runs: E2E_RECORDED_RUNS,
    inner_iterations: INNER_ITERATIONS,
    qm31_mul_iterations: QM31_MUL_ITERATIONS,
    hash_batches: HASH_BATCHES,
};

#[derive(Debug)]
enum SuiteError {
    Measurement(String),
    #[cfg(feature = "parallel")]
    ThreadPool(String),
    Output(io::Error),
}

impl Display for SuiteError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Measurement(error) => write!(f, "measurement failed: {error}"),
            #[cfg(feature = "parallel")]
            Self::ThreadPool(error) => write!(f, "thread pool failed: {error}"),
            Self::Output(error) => write!(f, "report output failed: {error}"),
        }
    }
}

impl std::error::Error for SuiteError {}

struct CapturedPathA {
    report: PathAReport,
    wall: Duration,
}

/// Runs the full suite, appends JSON lines to `out`, returns line count.
pub fn run_suite(out: &mut impl Write, config: &SuiteConfig) -> usize {
    try_run_suite(out, config, &DEFAULT_PLAN)
        .unwrap_or_else(|error| panic!("benchmark suite failed: {error}"))
}

#[cfg(test)]
fn run_suite_with_plan(out: &mut impl Write, config: &SuiteConfig, plan: &SuitePlan<'_>) -> usize {
    try_run_suite(out, config, plan)
        .unwrap_or_else(|error| panic!("benchmark suite failed: {error}"))
}

fn try_run_suite(
    out: &mut impl Write,
    config: &SuiteConfig,
    plan: &SuitePlan<'_>,
) -> Result<usize, SuiteError> {
    let thermal_before = thermal_zones();
    let mut records = Vec::new();
    #[cfg(feature = "parallel")]
    let measurement_result = rayon::ThreadPoolBuilder::new()
        .num_threads(config.threads.max(1))
        .build()
        .map_err(|error| SuiteError::ThreadPool(error.to_string()))
        .and_then(|pool| pool.install(|| collect_measurements(&mut records, plan)));
    #[cfg(not(feature = "parallel"))]
    let measurement_result = collect_measurements(&mut records, plan);
    measurement_result?;
    let thermal_after = thermal_zones();

    records.insert(
        0,
        metadata_record(config, thermal_before.as_deref(), thermal_after.as_deref()),
    );
    write_json_lines(out, &records).map_err(SuiteError::Output)
}

fn collect_measurements(records: &mut Vec<Value>, plan: &SuitePlan<'_>) -> Result<(), SuiteError> {
    let cold_warm = collect_e2e_matrix(records, plan, timed_path_a)?;

    for &log_n in plan.kernel_log_ns {
        records.push(fft_record(log_n, plan.inner_iterations));
        records.push(merkle_record(log_n, plan.inner_iterations));
        records.push(batch_inverse_record(log_n, plan.inner_iterations));
        records.push(qm31_mul_record(
            log_n,
            plan.inner_iterations,
            plan.qm31_mul_iterations,
        ));
    }
    records.extend(hash_records(plan.hash_batches, plan.inner_iterations));
    if let Some(cold_warm) = cold_warm {
        append_cold_warm_records(records, cold_warm);
    }
    Ok(())
}

fn collect_e2e_matrix(
    records: &mut Vec<Value>,
    plan: &SuitePlan<'_>,
    mut runner: impl FnMut(u32, usize) -> Result<CapturedPathA, SuiteError>,
) -> Result<Option<[CapturedPathA; 2]>, SuiteError> {
    let cold_warm = plan
        .cold_warm_cell
        .map(|(log_n, n_use_columns)| {
            Ok([runner(log_n, n_use_columns)?, runner(log_n, n_use_columns)?])
        })
        .transpose()?;

    for &(log_n, n_use_columns) in plan.e2e_cells {
        runner(log_n, n_use_columns)?;
        for run in 1..=plan.e2e_recorded_runs {
            let captured = runner(log_n, n_use_columns)?;
            append_path_a_records(records, &captured.report, run, false, "matrix", None);
        }
    }
    Ok(cold_warm)
}

fn append_cold_warm_records(records: &mut Vec<Value>, [cold, warm]: [CapturedPathA; 2]) {
    append_path_a_records(records, &cold.report, 1, true, "cold_warm", Some(cold.wall));
    append_path_a_records(
        records,
        &warm.report,
        2,
        false,
        "cold_warm",
        Some(warm.wall),
    );
}

fn timed_path_a(log_n: u32, n_use_columns: usize) -> Result<CapturedPathA, SuiteError> {
    let started = Instant::now();
    let report = benchmark_path_a(log_n, n_use_columns).map_err(|error| {
        SuiteError::Measurement(format!(
            "Path A failed at log_n={log_n}, l={n_use_columns}: {error}"
        ))
    })?;
    Ok(CapturedPathA {
        report,
        wall: started.elapsed(),
    })
}

fn append_path_a_records(
    records: &mut Vec<Value>,
    report: &PathAReport,
    run: usize,
    cold: bool,
    series: &'static str,
    wall: Option<Duration>,
) {
    records.extend(report.phases.iter().map(|measurement| {
        json!({
            "suite": SUITE_VERSION,
            "bench": "e2e_path_a",
            "log_n": report.log_n,
            "l": report.n_use_columns,
            "phase": measurement.phase,
            "ms": milliseconds(measurement.elapsed),
            "run": run,
            "cold": cold,
            "series": series,
        })
    }));
    records.push(json!({
        "suite": SUITE_VERSION,
        "bench": "e2e_path_a_proof",
        "log_n": report.log_n,
        "l": report.n_use_columns,
        "proof_bytes": report.proof_bytes,
        "run": run,
        "cold": cold,
        "series": series,
    }));
    if let Some(wall) = wall {
        records.push(json!({
            "suite": SUITE_VERSION,
            "bench": "e2e_path_a",
            "log_n": report.log_n,
            "l": report.n_use_columns,
            "phase": "wall",
            "ms": milliseconds(wall),
            "run": run,
            "cold": cold,
            "series": series,
        }));
    }
}

fn fft_record(log_n: u32, iterations: usize) -> Value {
    let mut rng = SmallRng::seed_from_u64(u64::from(log_n));
    let domain = CanonicCoset::new(log_n).circle_domain();
    let twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
    let column = (0..1usize << log_n)
        .map(|_| rng.gen::<BaseField>())
        .collect::<BaseColumn>();
    let elapsed = median_elapsed(iterations, || {
        let evaluation = CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(
            domain,
            column.clone(),
        );
        let started = Instant::now();
        let polynomial = evaluation.interpolate_with_twiddles(&twiddles);
        let result = polynomial.evaluate_with_twiddles(domain, &twiddles);
        black_box(result.values.at(0));
        started.elapsed()
    });
    json!({
        "suite": SUITE_VERSION,
        "bench": "fft_roundtrip",
        "log_n": log_n,
        "phase": "interpolate_evaluate",
        "ms": milliseconds(elapsed),
        "inner_iterations": iterations,
    })
}

fn merkle_record(log_n: u32, iterations: usize) -> Value {
    let mut rng = SmallRng::seed_from_u64(0x4d45_524b_4c45 + u64::from(log_n));
    let columns = (0..MERKLE_COLUMNS)
        .map(|_| {
            (0..1usize << log_n)
                .map(|_| rng.gen::<BaseField>())
                .collect::<BaseColumn>()
        })
        .collect::<Vec<_>>();
    let column_refs = columns.iter().collect::<Vec<_>>();
    let elapsed = median_elapsed(iterations, || {
        let columns = column_refs.clone();
        let started = Instant::now();
        let commitment =
            MerkleProverLifted::<SimdBackend, Blake2sMerkleHasher>::commit(columns, log_n, 0);
        black_box(commitment.root());
        started.elapsed()
    });
    json!({
        "suite": SUITE_VERSION,
        "bench": "merkle_commit",
        "log_n": log_n,
        "columns": MERKLE_COLUMNS,
        "ms": milliseconds(elapsed),
        "inner_iterations": iterations,
    })
}

fn batch_inverse_record(log_n: u32, iterations: usize) -> Value {
    let mut rng = SmallRng::seed_from_u64(0x0042_4154_4348 + u64::from(log_n));
    let values = (0..1usize << log_n)
        .map(|_| {
            let value = rng.gen::<SecureField>();
            if value.is_zero() {
                SecureField::one()
            } else {
                value
            }
        })
        .collect::<Vec<_>>();
    let elapsed = median_elapsed(iterations, || {
        let started = Instant::now();
        let inverses = SecureField::batch_inverse(black_box(&values));
        black_box(inverses[0]);
        started.elapsed()
    });
    json!({
        "suite": SUITE_VERSION,
        "bench": "batch_inverse",
        "field": "qm31",
        "log_n": log_n,
        "ms": milliseconds(elapsed),
        "inner_iterations": iterations,
    })
}

fn qm31_mul_record(log_n: u32, inner_iterations: usize, mul_iterations: usize) -> Value {
    let mut rng = SmallRng::seed_from_u64(0x514d_3331 + u64::from(log_n));
    let packed_len = (1usize << log_n) / 16;
    let lhs = (0..packed_len)
        .map(|_| rng.gen::<PackedQM31>())
        .collect::<Vec<_>>();
    let rhs = (0..packed_len)
        .map(|_| rng.gen::<PackedQM31>())
        .collect::<Vec<_>>();
    let elapsed = median_elapsed(inner_iterations, || {
        let mut values = lhs.clone();
        let start = Instant::now();
        for _ in 0..mul_iterations {
            for (value, rhs) in values.iter_mut().zip(&rhs) {
                *value = black_box(*value * *rhs);
            }
        }
        black_box(values[0]);
        start.elapsed()
    });
    let multiplications = (1usize << log_n) * mul_iterations;
    json!({
        "suite": SUITE_VERSION,
        "bench": "qm31_mul",
        "log_n": log_n,
        "iterations": mul_iterations,
        "ms": milliseconds(elapsed),
        "million_mul_per_s": per_second(multiplications, elapsed) / 1_000_000.0,
        "inner_iterations": inner_iterations,
    })
}

fn hash_records(batches: usize, iterations: usize) -> [Value; 2] {
    let messages: [u32x16; 16] =
        array::from_fn(|word| u32x16::from_array(array::from_fn(|lane| (word + lane) as u32)));
    let blake_elapsed = median_elapsed(iterations, || {
        let started = Instant::now();
        let mut state = INITIAL_STATE;
        for _ in 0..batches {
            state = compress16(
                state,
                messages,
                u32x16::splat(HASH_MESSAGE_BYTES as u32),
                ZEROS,
                u32x16::splat(u32::MAX),
                ZEROS,
            );
        }
        black_box(state);
        started.elapsed()
    });

    let messages = array::from_fn::<_, HASH_LANES, _>(|lane| {
        array::from_fn::<_, HASH_MESSAGE_BYTES, _>(|offset| (lane + offset) as u8)
    });
    let sha_elapsed = median_elapsed(iterations, || {
        let started = Instant::now();
        for _ in 0..batches {
            for message in &messages {
                black_box(Sha256::digest(black_box(message)));
            }
        }
        started.elapsed()
    });
    let bytes = batches * HASH_LANES * HASH_MESSAGE_BYTES;

    [
        json!({
            "suite": SUITE_VERSION,
            "bench": "hash",
            "algo": "blake2s_compress16",
            "mb_per_s": mb_per_second(bytes, blake_elapsed),
            "bytes": bytes,
            "inner_iterations": iterations,
        }),
        json!({
            "suite": SUITE_VERSION,
            "bench": "hash",
            "algo": "sha256_hw",
            "mb_per_s": mb_per_second(bytes, sha_elapsed),
            "bytes": bytes,
            "inner_iterations": iterations,
        }),
    ]
}

fn median_elapsed(iterations: usize, mut operation: impl FnMut() -> Duration) -> Duration {
    assert!(iterations > 0, "benchmark iterations must be nonzero");
    black_box(operation());
    let mut samples = std::iter::repeat_with(operation)
        .take(iterations)
        .collect::<Vec<_>>();
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn milliseconds(elapsed: Duration) -> f64 {
    elapsed.as_secs_f64() * 1000.0
}

fn per_second(count: usize, elapsed: Duration) -> f64 {
    count as f64 / elapsed.as_secs_f64()
}

fn mb_per_second(bytes: usize, elapsed: Duration) -> f64 {
    per_second(bytes, elapsed) / 1_000_000.0
}

fn metadata_record(
    config: &SuiteConfig,
    thermal_before: Option<&str>,
    thermal_after: Option<&str>,
) -> Value {
    let metadata = Map::from_iter([
        ("suite".to_owned(), Value::from(SUITE_VERSION)),
        ("record".to_owned(), Value::from("metadata")),
        (
            "git_commit".to_owned(),
            Value::from(env!("BENCH_RUNNER_GIT_COMMIT")),
        ),
        (
            "binary_sha256".to_owned(),
            Value::from(binary_sha256().unwrap_or_else(|error| format!("unavailable: {error}"))),
        ),
        (
            "rustflags".to_owned(),
            Value::from(env!("BENCH_RUNNER_RUSTFLAGS")),
        ),
        (
            "thread_count".to_owned(),
            Value::from(config.threads.max(1)),
        ),
        (
            "core_count".to_owned(),
            Value::from(
                std::thread::available_parallelism()
                    .map(usize::from)
                    .unwrap_or(1),
            ),
        ),
        (
            "target_arch".to_owned(),
            Value::from(std::env::consts::ARCH),
        ),
        ("target_os".to_owned(), Value::from(std::env::consts::OS)),
        (
            "parallel".to_owned(),
            Value::from(cfg!(feature = "parallel")),
        ),
    ]);

    #[cfg(target_os = "android")]
    {
        let mut metadata = metadata;
        metadata.insert(
            "android_model".to_owned(),
            Value::from(android_model().unwrap_or_else(|| "unknown".to_owned())),
        );
        metadata.insert(
            "cpuinfo".to_owned(),
            Value::from(
                std::fs::read_to_string("/proc/cpuinfo")
                    .unwrap_or_else(|error| format!("unavailable: {error}")),
            ),
        );
        metadata.insert(
            "thermal_before".to_owned(),
            Value::from(thermal_before.unwrap_or("unavailable")),
        );
        metadata.insert(
            "thermal_after".to_owned(),
            Value::from(thermal_after.unwrap_or("unavailable")),
        );
        Value::Object(metadata)
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = (thermal_before, thermal_after);
        Value::Object(metadata)
    }
}

fn binary_sha256() -> io::Result<String> {
    let path = benchmark_binary_path()?;
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(to_hex(&hasher.finalize()))
}

fn benchmark_binary_path() -> io::Result<PathBuf> {
    if let Some(path) = std::env::var_os("BENCH_RUNNER_BINARY_PATH") {
        return Ok(path.into());
    }
    #[cfg(target_os = "android")]
    if let Some(path) = android_library_path() {
        return Ok(path);
    }
    std::env::current_exe()
}

#[cfg(target_os = "android")]
fn android_library_path() -> Option<PathBuf> {
    std::fs::read_to_string("/proc/self/maps")
        .ok()?
        .lines()
        .find(|line| line.contains("libbench_runner.so"))
        .and_then(|line| line.split_whitespace().last())
        .map(PathBuf::from)
}

fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        result.push(HEX[(byte >> 4) as usize] as char);
        result.push(HEX[(byte & 0x0f) as usize] as char);
    }
    result
}

#[cfg(target_os = "android")]
fn android_model() -> Option<String> {
    std::process::Command::new("getprop")
        .arg("ro.product.model")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|model| model.trim().to_owned())
}

fn thermal_zones() -> Option<String> {
    #[cfg(not(target_os = "android"))]
    {
        None
    }
    #[cfg(target_os = "android")]
    {
        let mut zones = std::fs::read_dir("/sys/class/thermal")
            .ok()?
            .flatten()
            .collect::<Vec<_>>();
        zones.sort_by_key(|entry| entry.file_name());
        let readings = zones
            .into_iter()
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("thermal_zone")
            })
            .filter_map(|entry| {
                let path = entry.path();
                let zone_type = std::fs::read_to_string(path.join("type")).ok()?;
                let temperature = std::fs::read_to_string(path.join("temp")).ok()?;
                Some(format!("{}={}", zone_type.trim(), temperature.trim()))
            })
            .collect::<Vec<_>>();
        (!readings.is_empty()).then(|| readings.join(";"))
    }
}

fn write_json_lines(out: &mut impl Write, records: &[Value]) -> io::Result<usize> {
    for record in records {
        let mut line = serde_json::to_vec(record).expect("JSON values are always serializable");
        line.push(b'\n');
        out.write_all(&line)?;
    }
    Ok(records.len())
}

#[cfg(feature = "jni")]
#[no_mangle]
#[allow(non_snake_case)]
pub extern "system" fn Java_eu_stwo_bench_BenchRunner_runSuite(
    mut env: jni::JNIEnv<'_>,
    _class: jni::objects::JClass<'_>,
    out_path: jni::objects::JString<'_>,
) -> jni::sys::jint {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let path = env
            .get_string(&out_path)
            .map_err(|error| error.to_string())?;
        let mut file =
            File::create(path.to_string_lossy().as_ref()).map_err(|error| error.to_string())?;
        Ok::<_, String>(run_suite(&mut file, &SuiteConfig::default()) as jni::sys::jint)
    }));

    match result {
        Ok(Ok(line_count)) => line_count,
        Ok(Err(error)) => {
            let _ = env.throw_new("java/io/IOException", error);
            -1
        }
        Err(_) => {
            let _ = env.throw_new("java/lang/RuntimeException", "bench-runner panicked");
            -1
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{self, Write};
    use std::time::Duration;

    use serde_json::Value;
    use stwo_examples::gkr_e2e::{PathAReport, PhaseTiming};

    use super::{
        append_cold_warm_records, append_path_a_records, collect_e2e_matrix, hash_records,
        median_elapsed, run_suite_with_plan, to_hex, CapturedPathA, SuiteConfig, SuitePlan,
    };

    #[test]
    fn reusable_path_a_report_has_original_phase_order() {
        let report = stwo_examples::gkr_e2e::benchmark_path_a(8, 2).unwrap();
        let phases = report
            .phases
            .iter()
            .map(|measurement| measurement.phase)
            .collect::<Vec<_>>();

        assert_eq!(
            phases,
            [
                "base_commit",
                "interaction_gen",
                "interaction_commit",
                "stark_prove",
                "total",
                "verify",
            ]
        );
        assert!(report.proof_bytes > 0);
    }

    #[test]
    fn median_elapsed_discards_warmup_and_selects_middle_sample() {
        let mut samples = [99, 5, 1, 9, 3, 7].into_iter();
        let median = median_elapsed(5, || Duration::from_millis(samples.next().unwrap()));
        assert_eq!(median, Duration::from_millis(5));
        assert!(samples.next().is_none());
    }

    #[test]
    fn path_a_records_preserve_phase_order_and_tags() {
        let report = PathAReport {
            log_n: 8,
            n_use_columns: 2,
            phases: vec![
                PhaseTiming {
                    phase: "base_commit",
                    elapsed: Duration::from_millis(3),
                },
                PhaseTiming {
                    phase: "verify",
                    elapsed: Duration::from_millis(1),
                },
            ],
            proof_bytes: 123,
        };
        let mut records = Vec::new();
        append_path_a_records(
            &mut records,
            &report,
            2,
            true,
            "cold_warm",
            Some(Duration::from_millis(5)),
        );

        assert_eq!(records.len(), 4);
        assert_eq!(records[0]["phase"], "base_commit");
        assert_eq!(records[1]["phase"], "verify");
        assert_eq!(records[2]["proof_bytes"], 123);
        assert_eq!(records[3]["phase"], "wall");
        assert_eq!(records[3]["ms"], 5.0);
        assert_eq!(records[0]["run"], 2);
        assert_eq!(records[0]["cold"], true);
        assert_eq!(records[0]["series"], "cold_warm");
    }

    #[test]
    fn hash_benches_process_equal_bytes() {
        let [blake, sha] = hash_records(1, 7);
        assert_eq!(blake["bytes"], sha["bytes"]);
        assert!(blake["mb_per_s"].as_f64().unwrap() > 0.0);
        assert!(sha["mb_per_s"].as_f64().unwrap() > 0.0);
        assert_eq!(blake["inner_iterations"], 7);
        assert_eq!(sha["inner_iterations"], 7);
    }

    #[test]
    fn e2e_protocol_discards_warmup_and_keeps_cold_warm_distinct() {
        let plan = SuitePlan {
            e2e_cells: &[(8, 1)],
            kernel_log_ns: &[],
            cold_warm_cell: Some((8, 1)),
            e2e_recorded_runs: 5,
            inner_iterations: 7,
            qm31_mul_iterations: 1,
            hash_batches: 1,
        };
        let mut records = Vec::new();
        let mut calls = 0;
        let cold_warm = collect_e2e_matrix(&mut records, &plan, |log_n, n_use_columns| {
            calls += 1;
            Ok(CapturedPathA {
                report: PathAReport {
                    log_n,
                    n_use_columns,
                    phases: vec![PhaseTiming {
                        phase: "total",
                        elapsed: Duration::from_millis(calls as u64),
                    }],
                    proof_bytes: calls,
                },
                wall: Duration::from_millis(calls as u64),
            })
        })
        .unwrap()
        .unwrap();

        assert_eq!(calls, 8, "2 tagged calls + 1 warmup + 5 recorded calls");
        assert_eq!(records.len(), 10, "five phase and five proof records");
        assert!(records.iter().all(|record| record["series"] == "matrix"));
        assert!(records.iter().all(|record| record["cold"] == false));
        let runs = records
            .iter()
            .filter(|record| record.get("phase").is_some())
            .map(|record| record["run"].as_u64().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(runs, [1, 2, 3, 4, 5]);

        append_cold_warm_records(&mut records, cold_warm);
        let tagged = &records[10..];
        assert_eq!(tagged.len(), 6, "two phase/proof/wall record groups");
        assert_eq!(tagged[0]["cold"], true);
        assert_eq!(tagged[2]["phase"], "wall");
        assert_eq!(tagged[3]["cold"], false);
        assert_eq!(tagged[5]["phase"], "wall");
        assert!(tagged.iter().all(|record| record["series"] == "cold_warm"));
    }

    #[test]
    fn small_suite_is_json_lines_with_metadata_first() {
        let plan = SuitePlan {
            e2e_cells: &[],
            kernel_log_ns: &[4],
            cold_warm_cell: None,
            e2e_recorded_runs: 5,
            inner_iterations: 1,
            qm31_mul_iterations: 1,
            hash_batches: 1,
        };
        let mut output = Vec::new();
        let count = run_suite_with_plan(&mut output, &SuiteConfig { threads: 1 }, &plan);
        let lines = String::from_utf8(output).unwrap();
        let records = lines
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(count, 7);
        assert_eq!(records.len(), count);
        assert_eq!(records[0]["record"], "metadata");
        assert_eq!(records[1]["bench"], "fft_roundtrip");
        assert_eq!(records[4]["bench"], "qm31_mul");
        assert_eq!(records[5]["algo"], "blake2s_compress16");
        assert_eq!(records[6]["algo"], "sha256_hw");
    }

    #[test]
    #[should_panic(expected = "benchmark suite failed: report output failed")]
    fn report_write_failure_is_not_silent() {
        struct FailingWriter;

        impl Write for FailingWriter {
            fn write(&mut self, _buffer: &[u8]) -> io::Result<usize> {
                Err(io::Error::other("injected write failure"))
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let plan = SuitePlan {
            e2e_cells: &[],
            kernel_log_ns: &[],
            cold_warm_cell: None,
            e2e_recorded_runs: 5,
            inner_iterations: 1,
            qm31_mul_iterations: 1,
            hash_batches: 1,
        };
        run_suite_with_plan(&mut FailingWriter, &SuiteConfig { threads: 1 }, &plan);
    }

    #[test]
    fn hex_encoding_is_lowercase_and_complete() {
        assert_eq!(to_hex(&[0x00, 0xab, 0xff]), "00abff");
    }
}
