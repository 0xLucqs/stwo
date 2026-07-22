use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rand::distributions::{Distribution, Standard};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::Field;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::CpuBackend;
use stwo::prover::lookups::gkr_prover::{prove_batch, GkrOps, Layer};
use stwo::prover::lookups::mle::{Mle, MleOps};

const SMALL_LOG_N_ROWS: u32 = 16;
const LARGE_LOG_N_ROWS: u32 = 20;

fn bench_gkr_grand_product<B: GkrOps>(c: &mut Criterion, id: &str, log_n_rows: u32) {
    let mut rng = SmallRng::seed_from_u64(0);
    let layer = Layer::<B>::GrandProduct(gen_random_mle(&mut rng, log_n_rows));
    c.bench_function(&format!("{id} grand product lookup 2^{log_n_rows}"), |b| {
        b.iter_batched(
            || layer.clone(),
            |layer| prove_batch(&mut Blake2sChannel::default(), vec![layer]),
            BatchSize::LargeInput,
        )
    });
}

fn bench_gkr_grand_product_batch<B: GkrOps>(c: &mut Criterion, id: &str, log_n_rows: u32) {
    let mut rng = SmallRng::seed_from_u64(0);
    let layer = Layer::<B>::GrandProduct(gen_random_mle(&mut rng, log_n_rows));
    c.bench_function(
        &format!("{id} grand product lookup batch 4x 2^{log_n_rows}"),
        |b| {
            b.iter_batched(
                || vec![layer.clone(), layer.clone(), layer.clone(), layer.clone()],
                |layers| prove_batch(&mut Blake2sChannel::default(), layers),
                BatchSize::LargeInput,
            )
        },
    );
}

fn bench_gkr_logup_generic<B: GkrOps>(c: &mut Criterion, id: &str, log_n_rows: u32) {
    let mut rng = SmallRng::seed_from_u64(0);
    let generic_layer = Layer::<B>::LogUpGeneric {
        numerators: gen_random_mle(&mut rng, log_n_rows),
        denominators: gen_random_mle(&mut rng, log_n_rows),
    };
    c.bench_function(&format!("{id} generic logup lookup 2^{log_n_rows}"), |b| {
        b.iter_batched(
            || generic_layer.clone(),
            |layer| prove_batch(&mut Blake2sChannel::default(), vec![layer]),
            BatchSize::LargeInput,
        )
    });
}

fn bench_gkr_logup_multiplicities<B: GkrOps>(c: &mut Criterion, id: &str, log_n_rows: u32) {
    let mut rng = SmallRng::seed_from_u64(0);
    let multiplicities_layer = Layer::<B>::LogUpMultiplicities {
        numerators: gen_random_mle(&mut rng, log_n_rows),
        denominators: gen_random_mle(&mut rng, log_n_rows),
    };
    c.bench_function(
        &format!("{id} multiplicities logup lookup 2^{log_n_rows}"),
        |b| {
            b.iter_batched(
                || multiplicities_layer.clone(),
                |layer| prove_batch(&mut Blake2sChannel::default(), vec![layer]),
                BatchSize::LargeInput,
            )
        },
    );
}

fn bench_gkr_logup_singles<B: GkrOps>(c: &mut Criterion, id: &str, log_n_rows: u32) {
    let mut rng = SmallRng::seed_from_u64(0);
    let singles_layer = Layer::<B>::LogUpSingles {
        denominators: gen_random_mle(&mut rng, log_n_rows),
    };
    c.bench_function(&format!("{id} singles logup lookup 2^{log_n_rows}"), |b| {
        b.iter_batched(
            || singles_layer.clone(),
            |layer| prove_batch(&mut Blake2sChannel::default(), vec![layer]),
            BatchSize::LargeInput,
        )
    });
}

/// Generates a random multilinear polynomial.
fn gen_random_mle<B: MleOps<F>, F: Field>(rng: &mut impl Rng, n_variables: u32) -> Mle<B, F>
where
    Standard: Distribution<F>,
{
    Mle::new((0..1 << n_variables).map(|_| rng.gen()).collect())
}

fn gkr_lookup_benches(c: &mut Criterion) {
    for log_n_rows in [SMALL_LOG_N_ROWS, LARGE_LOG_N_ROWS] {
        bench_gkr_grand_product::<SimdBackend>(c, "simd", log_n_rows);
        bench_gkr_logup_generic::<SimdBackend>(c, "simd", log_n_rows);
        bench_gkr_logup_multiplicities::<SimdBackend>(c, "simd", log_n_rows);
        bench_gkr_logup_singles::<SimdBackend>(c, "simd", log_n_rows);
    }
    bench_gkr_grand_product_batch::<SimdBackend>(c, "simd", SMALL_LOG_N_ROWS);

    bench_gkr_grand_product::<CpuBackend>(c, "cpu", SMALL_LOG_N_ROWS);
    bench_gkr_logup_generic::<CpuBackend>(c, "cpu", SMALL_LOG_N_ROWS);
    bench_gkr_logup_multiplicities::<CpuBackend>(c, "cpu", SMALL_LOG_N_ROWS);
    bench_gkr_logup_singles::<CpuBackend>(c, "cpu", SMALL_LOG_N_ROWS);
    bench_gkr_grand_product_batch::<CpuBackend>(c, "cpu", SMALL_LOG_N_ROWS);
}

criterion_group!(benches, gkr_lookup_benches);
criterion_main!(benches);
