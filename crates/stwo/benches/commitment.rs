use std::iter;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::BackendForChannel;
use stwo::prover::mempool::BaseColumnPool;
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::CommitmentTreeProver;

const LOG_BLOWUP_FACTOR: u32 = 1;
const N_COLUMNS: usize = 32;

fn commit_tree<B: BackendForChannel<Blake2sMerkleChannel>>(
    evals: Vec<CircleEvaluation<B, BaseField, BitReversedOrder>>,
    twiddles: &TwiddleTree<B>,
) {
    let polys = evals
        .into_iter()
        .map(|eval| eval.interpolate_with_twiddles(twiddles))
        .collect();

    CommitmentTreeProver::<B, Blake2sMerkleChannel>::new(
        polys,
        LOG_BLOWUP_FACTOR,
        twiddles,
        false,
        None,
        &BaseColumnPool::new(),
    );
}

fn bench_simd_commitment(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(0);
    for log_size in 16..=18 {
        let small_domain = CanonicCoset::new(log_size);
        let big_domain = CanonicCoset::new(log_size + LOG_BLOWUP_FACTOR);
        let twiddles = SimdBackend::precompute_twiddles(big_domain.half_coset());
        let evals = iter::repeat_with(|| {
            CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(
                small_domain.circle_domain(),
                (0..1 << log_size).map(|_| rng.gen()).collect(),
            )
        })
        .take(N_COLUMNS)
        .collect::<Vec<_>>();

        c.bench_function(
            &format!("simd commitment tree {N_COLUMNS} columns 2^{log_size}"),
            |b| {
                b.iter_batched(
                    || evals.clone(),
                    |evals| commit_tree::<SimdBackend>(black_box(evals), black_box(&twiddles)),
                    BatchSize::LargeInput,
                );
            },
        );
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_simd_commitment);
criterion_main!(benches);
