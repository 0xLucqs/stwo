use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rand::distributions::{Distribution, Standard};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::QM31;
use stwo::core::fields::{batch_inverse, batch_inverse_chunked, batch_inverse_in_place, Field};

const LOG_SIZES: [u32; 3] = [12, 16, 20];
const CHUNK_SIZE: usize = 1 << 12;

fn bench_field<F>(c: &mut Criterion, name: &str)
where
    F: Field + Send + Sync,
    Standard: Distribution<F>,
{
    let mut rng = SmallRng::seed_from_u64(0);
    for log_size in LOG_SIZES {
        let values = (0..1 << log_size)
            .map(|_| rng.gen::<F>())
            .collect::<Vec<_>>();

        c.bench_function(&format!("{name} batch_inverse 2^{log_size}"), |b| {
            b.iter(|| batch_inverse(black_box(&values)));
        });

        c.bench_function(
            &format!("{name} batch_inverse_in_place 2^{log_size}"),
            |b| {
                b.iter_batched(
                    || vec![F::zero(); values.len()],
                    |mut dst| batch_inverse_in_place(black_box(&values), black_box(&mut dst)),
                    BatchSize::LargeInput,
                );
            },
        );

        c.bench_function(&format!("{name} batch_inverse_chunked 2^{log_size}"), |b| {
            b.iter_batched(
                || vec![F::zero(); values.len()],
                |mut dst| {
                    batch_inverse_chunked(black_box(&values), black_box(&mut dst), CHUNK_SIZE)
                },
                BatchSize::LargeInput,
            );
        });
    }
}

fn batch_inverse_benches(c: &mut Criterion) {
    bench_field::<M31>(c, "M31");
    bench_field::<QM31>(c, "QM31");
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = batch_inverse_benches);
criterion_main!(benches);
