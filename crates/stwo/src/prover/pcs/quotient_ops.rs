use std::borrow::Cow;
use std::iter::zip;

use itertools::Itertools;
use tracing::{span, Level};

use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::{
    build_samples_with_randomness_and_periodicity, ColumnSampleBatch, PointSample,
};
use crate::core::pcs::TreeVec;
use crate::prover::air::component_prover::Poly;
use crate::prover::backend::{Backend, ColumnOps};
use crate::prover::mempool::BaseColumnPool;
use crate::prover::poly::circle::{CircleEvaluation, PolyOps, SecureEvaluation};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::prover::AccumulationOps;

pub trait QuotientOps: PolyOps {
    /// Receives a non-empty set of columns of the *same* size, and populates the vector
    /// `accumulated_numerators_vec` with their FRI numerators accumulations, across
    /// `sample_batches`.
    ///
    /// For each sample batch in `sample_batches`, accumulates the numerators of the columns
    /// involved in this batch over the evaluation subdomain (the first
    /// `column_size >> log_blowup_factor` rows in bit-reversed order) and pushes an
    /// `AccumulatedNumerators` object into `accumulated_numerators_vec`.
    fn accumulate_numerators(
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        sample_batches: &[ColumnSampleBatch],
        accumulated_numerators_vec: &mut Vec<AccumulatedNumerators<Self>>,
        log_blowup_factor: u32,
    );

    /// Given a vector of `AccumulatedNumerators` (computed on evaluation subdomains), computes
    /// the full quotient on the subdomain of size `2^(lifting_log_size - log_blowup_factor)`,
    /// then interpolates and evaluates on the full domain of size `2^lifting_log_size`.
    ///
    /// For each subdomain point:
    /// * Computes the denominator inverse for each sample point.
    /// * Multiplies it by the accumulated numerator for that (subdomain point, sample point).
    /// * Sums across sample points.
    ///
    /// The result is then extended to the full evaluation domain via interpolation on the
    /// subdomain followed by evaluation on the full domain, using the provided `twiddles`.
    fn compute_quotients_and_combine(
        accs: Vec<AccumulatedNumerators<Self>>,
        lifting_log_size: u32,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<Self>,
    ) -> SecureEvaluation<Self, BitReversedOrder>;
}

/// Helper struct that keeps track of the accumulation of the numerators involved in the FRI
/// quotients.
///
/// Note: `Clone` is derived only for benchmarking purposes.
#[derive(Clone)]
pub struct AccumulatedNumerators<B: ColumnOps<BaseField>> {
    /// One of the sample points received by the pcs.
    pub sample_point: CirclePoint<SecureField>,
    /// Stores a circle evaluation of the form:
    ///     p -> ∑ α^{k_i} * (cᵢ * f̃ᵢ(p) - bᵢ)
    /// where
    /// * p ∈ canonic coset of log size = l (where l is the log size of the column).
    /// * i runs over some column indices of the trace.
    /// * α is the random coefficient for the accumulation.
    /// * k_i is the randomness exponent for column i and sample point `sample_point`.
    /// * f̃ᵢ is the lift of the trace poly fᵢ to log size l.
    /// * (bᵢ, cᵢ) are the `b` and `c` line coefficients for column i and sample point
    ///   `sample_point`.
    pub partial_numerators_acc: SecureColumnByCoords<B>,
    /// Stores an accumulation of the form
    ///      ∑ α^{k_i} * aᵢ
    /// where
    /// * i runs over some column indices of the trace.
    /// * α and k_i are as in the previous docstring for `partial_numerators_acc`.
    /// * aᵢ is the `a` line coefficient for column i and sample point `sample_point`.
    ///
    /// The index set of the summation is equal to the index set of the summation referred in the
    /// previous docstring for `partial_numerators_acc`.
    pub first_linear_term_acc: SecureField,
}

pub fn compute_fri_quotients<B: QuotientOps + AccumulationOps>(
    columns: &TreeVec<Vec<&CircleEvaluation<B, BaseField, BitReversedOrder>>>,
    samples: &TreeVec<Vec<Vec<PointSample>>>,
    random_coeff: SecureField,
    lifting_log_size: u32,
    twiddles: &TwiddleTree<B>,
    log_blowup_factor: u32,
) -> SecureEvaluation<B, BitReversedOrder> {
    let _span = span!(Level::INFO, "Compute FRI quotients", class = "FRIQuotients").entered();
    let mut accumulated_numerators_vec: Vec<AccumulatedNumerators<B>> = vec![];
    let samples_with_randomness = build_samples_with_randomness_and_periodicity(
        samples,
        columns
            .0
            .iter()
            .map(|x| x.iter().map(|c| c.domain.log_size()))
            .collect(),
        lifting_log_size,
        random_coeff,
    );

    // Populate `accumulated_numerators_vec`, per (log_size, sample_point). After this iteration,
    // `accumulated_numerators_vec` will have length equal to
    //
    //   ∑_k (# of distinct sample points per log size k).
    //
    zip(
        columns.iter().flatten(),
        samples_with_randomness.iter().flatten(),
    )
    .sorted_by_key(|(c, _)| c.domain.log_size())
    .group_by(|(c, _)| c.domain.log_size())
    .into_iter()
    .for_each(|(_, tuples)| {
        let (columns, samples_with_randomness): (Vec<_>, Vec<_>) = tuples.unzip();
        // TODO: slice.
        let sample_batches = ColumnSampleBatch::new_vec(&samples_with_randomness);
        B::accumulate_numerators(
            &columns,
            &sample_batches,
            &mut accumulated_numerators_vec,
            log_blowup_factor,
        )
    });

    // Group and accumulate the numerators per sample point: the accumulations (of different
    // lengths) get lifted and accumulated to a single vector. After this step, there is a single
    // accumulation per sample point.
    let accumulations_per_sample_point = accumulated_numerators_vec
        .into_iter()
        .sorted_by_key(|c| (c.sample_point.x, c.sample_point.y))
        .group_by(|c| c.sample_point)
        .into_iter()
        .map(|(sample_point, accumulations_per_log_size)| {
            let accumulations_per_log_size = accumulations_per_log_size.collect_vec();
            // Accumulate the `a` coefficients.
            let first_linear_term_acc: SecureField = accumulations_per_log_size
                .iter()
                .map(|x| x.first_linear_term_acc)
                .sum();
            // Lift and accumulate the partial numerators vectors.
            // `partial_numerators_acc` is already sorted increasingly by size as required by
            // `B::lift_and_accumulate`.
            let partial_numerators_acc = accumulations_per_log_size
                .into_iter()
                .map(|x| x.partial_numerators_acc)
                .collect_vec();
            let res = B::lift_and_accumulate(partial_numerators_acc).unwrap();

            AccumulatedNumerators {
                sample_point,
                partial_numerators_acc: res,
                first_linear_term_acc,
            }
        })
        .collect_vec();

    B::compute_quotients_and_combine(
        accumulations_per_sample_point,
        lifting_log_size,
        log_blowup_factor,
        twiddles,
    )
}

/// Maximum number of columns to materialize simultaneously during low-memory FRI quotient
/// computation. Limits transient memory to `batch_size * eval_size` instead of
/// `all_columns_at_log_size * eval_size`.
const LOW_MEMORY_QUOTIENT_BATCH_SIZE: usize = 8;

pub fn compute_fri_quotients_from_polys<B: QuotientOps + AccumulationOps + Backend>(
    polynomials: &TreeVec<Vec<&Poly<B>>>,
    samples: &TreeVec<Vec<Vec<PointSample>>>,
    random_coeff: SecureField,
    lifting_log_size: u32,
    twiddles: &TwiddleTree<B>,
    log_blowup_factor: u32,
    base_column_pool: &BaseColumnPool<B>,
) -> SecureEvaluation<B, BitReversedOrder> {
    let _span = span!(
        Level::INFO,
        "Compute FRI quotients (low memory)",
        class = "FRIQuotientsLowMemory"
    )
    .entered();
    let mut accumulated_numerators_vec: Vec<AccumulatedNumerators<B>> = vec![];
    let samples_with_randomness = build_samples_with_randomness_and_periodicity(
        samples,
        polynomials
            .0
            .iter()
            .map(|x| x.iter().map(|poly| poly.log_size()))
            .collect(),
        lifting_log_size,
        random_coeff,
    );

    // Process each tree's polynomials separately to avoid cross-tree materialization overlap.
    // Within each tree, polynomials are grouped by log_size and sub-batched.
    for (tree_polys, tree_samples) in polynomials.iter().zip(samples_with_randomness.iter()) {
        zip(tree_polys.iter(), tree_samples.iter())
            .sorted_by_key(|(poly, _)| poly.log_size())
            .group_by(|(poly, _)| poly.log_size())
            .into_iter()
            .for_each(|(_, tuples)| {
                let (polys, samples_with_randomness): (Vec<_>, Vec<_>) = tuples.unzip();

                // Sub-batch materialization to limit transient memory.
                for chunk_start in (0..polys.len()).step_by(LOW_MEMORY_QUOTIENT_BATCH_SIZE) {
                    let chunk_end = (chunk_start + LOW_MEMORY_QUOTIENT_BATCH_SIZE).min(polys.len());
                    let chunk_polys = &polys[chunk_start..chunk_end];
                    let chunk_samples = &samples_with_randomness[chunk_start..chunk_end];

                    let sample_batches = ColumnSampleBatch::new_vec(chunk_samples);

                    let evals: Vec<Cow<'_, CircleEvaluation<B, BaseField, BitReversedOrder>>> =
                        chunk_polys
                            .iter()
                            .map(|poly: &&Poly<B>| {
                                poly.evals.as_ref().map(Cow::Borrowed).unwrap_or_else(|| {
                                    Cow::Owned(
                                        poly.materialize_evaluation(twiddles, base_column_pool),
                                    )
                                })
                            })
                            .collect();
                    let columns = evals.iter().map(|eval| eval.as_ref()).collect_vec();
                    B::accumulate_numerators(
                        &columns,
                        &sample_batches,
                        &mut accumulated_numerators_vec,
                        log_blowup_factor,
                    );
                    // Evaluations are dropped here, freeing memory before the next sub-batch.
                }
            });
    }

    // Sort by (sample_point, size) so that within each sample_point group the entries are in
    // strictly ascending size order, as required by `lift_and_accumulate`. When processing
    // multiple trees sequentially, each tree's entries are inserted in ascending log_size order,
    // but after all of tree 0's entries (log_sizes L1..Lmax) come tree 1's entries (starting
    // again from L1). Without the size sort, `lift_and_accumulate` would see non-monotonic sizes
    // and underflow on the u32 log_ratio calculation.
    let accumulations_per_sample_point = accumulated_numerators_vec
        .into_iter()
        .sorted_by_key(|c| {
            (
                c.sample_point.x,
                c.sample_point.y,
                c.partial_numerators_acc.len(),
            )
        })
        .group_by(|c| c.sample_point)
        .into_iter()
        .map(|(sample_point, accumulations_per_log_size)| {
            let accumulations_per_log_size = accumulations_per_log_size.collect_vec();
            let first_linear_term_acc: SecureField = accumulations_per_log_size
                .iter()
                .map(|x| x.first_linear_term_acc)
                .sum();
            let partial_numerators_acc = accumulations_per_log_size
                .into_iter()
                .map(|x| x.partial_numerators_acc)
                .collect_vec();
            let res = B::lift_and_accumulate(partial_numerators_acc).unwrap();

            AccumulatedNumerators {
                sample_point,
                partial_numerators_acc: res,
                first_linear_term_acc,
            }
        })
        .collect_vec();

    B::compute_quotients_and_combine(
        accumulations_per_sample_point,
        lifting_log_size,
        log_blowup_factor,
        twiddles,
    )
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use itertools::Itertools;
    use num_traits::Zero;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::channel::Blake2sChannel;
    use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use crate::core::fields::m31::M31;
    use crate::core::pcs::quotients::PointSample;
    use crate::core::pcs::{CommitmentSchemeVerifier, PcsConfig, TreeVec};
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
    use crate::core::verifier::VerificationError;
    use crate::prover::backend::cpu::{CpuCircleEvaluation, CpuCirclePoly};
    use crate::prover::backend::simd::column::BaseColumn;
    use crate::prover::backend::simd::SimdBackend;
    use crate::prover::backend::{Backend, BackendForChannel, Column, CpuBackend};
    use crate::prover::pcs::quotient_ops::compute_fri_quotients;
    use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation, PolyOps};
    use crate::prover::{CommitmentSchemeProver, ProverMemoryMode, SecureField};

    #[test]
    fn test_quotients_are_low_degree() {
        let mut rng = SmallRng::seed_from_u64(0);
        const LOG_SIZE: u32 = 3;
        const LOG_BLOWUP_FACTOR: u32 = 4;

        let polynomial = CpuCirclePoly::new((0..1 << LOG_SIZE).map(M31::from).collect());
        let eval_domain = CanonicCoset::new(LOG_SIZE + LOG_BLOWUP_FACTOR).circle_domain();
        let eval = polynomial.evaluate(eval_domain);

        let sample_points = [
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
        ];
        let samples = sample_points
            .into_iter()
            .map(|x| PointSample {
                point: x,
                value: polynomial.eval_at_point(x),
            })
            .collect_vec();
        let rand_coeff =
            SecureField::from_m31_array(std::array::from_fn(|_| M31::from(rng.gen::<u32>())));
        let quot_eval = compute_fri_quotients(
            &TreeVec(vec![vec![&eval]]),
            &TreeVec(vec![vec![samples]]),
            rand_coeff,
            LOG_SIZE + LOG_BLOWUP_FACTOR,
            &CpuBackend::precompute_twiddles(eval_domain.half_coset),
            LOG_BLOWUP_FACTOR,
        );
        let mut coeffs = quot_eval
            .values
            .columns
            .iter()
            .map(|c| CpuCircleEvaluation::new(eval_domain, c.clone()).interpolate())
            .collect_vec();
        let zeros = coeffs[0].coeffs.split_off((1 << LOG_SIZE) - 1);

        assert!(zeros.iter().all(|c| c.is_zero()));
    }

    /// Generates a vector of random polynomials, such that the last one is of degree
    /// `LIFTING_LOG_SIZE - 1` and all the previous ones are of degree < `LIFTING_LOG_SIZE - 1`.
    fn prepare_polys<B: Backend, const N_COLS: usize, const LIFTING_LOG_SIZE: u32>(
    ) -> Vec<CircleCoefficients<B>> {
        let mut rng = SmallRng::seed_from_u64(0);
        let mut polys: Vec<CircleCoefficients<B>> = (0..N_COLS - 1)
            .map(|_| {
                CircleCoefficients::new(
                    (0..1 << rng.gen_range(4..LIFTING_LOG_SIZE - 1))
                        .map(M31::from)
                        .collect(),
                )
            })
            .collect();
        polys.push(CircleCoefficients::new(
            (0..1 << LIFTING_LOG_SIZE).map(M31::from).collect(),
        ));
        polys
    }

    fn prove_and_verify_pcs_with_params<
        B: BackendForChannel<Blake2sMerkleChannel>,
        const STORE_COEFFS: bool,
        const N_COLS: usize,
        const LIFTING_LOG_SIZE: u32,
    >(
        memory_mode: ProverMemoryMode,
    ) -> Result<(), VerificationError> {
        // Setup the prover side of the pcs.
        let mut channel = Blake2sChannel::default();
        let config = PcsConfig::default();
        let twiddles = B::precompute_twiddles(
            CanonicCoset::new(LIFTING_LOG_SIZE + config.fri_config.log_blowup_factor).half_coset(),
        );
        let mut commitment_scheme =
            CommitmentSchemeProver::<B, Blake2sMerkleChannel>::new(config, &twiddles);
        if STORE_COEFFS {
            commitment_scheme.set_store_polynomials_coefficients();
        }
        commitment_scheme.set_memory_mode(memory_mode);
        let polys = prepare_polys::<B, N_COLS, LIFTING_LOG_SIZE>();
        let sizes = polys.iter().map(|poly| poly.log_size()).collect_vec();

        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_polys(polys);
        tree_builder.commit(&mut channel);

        let mut rng = SmallRng::seed_from_u64(0);
        let mask_structure = (0..N_COLS).map(|_| rng.gen_range(1..=2)).collect_vec();
        let samples = [
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
        ];
        let sampled_points = vec![(0..N_COLS)
            .zip(mask_structure.iter())
            .map(|(_, i)| samples.into_iter().take(*i).collect_vec())
            .collect_vec()];

        let proof = commitment_scheme.prove_values(TreeVec(sampled_points.clone()), &mut channel);

        // Verifier side of the pcs.
        let mut channel = Blake2sChannel::default();
        let mut verifier = CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
        verifier.commit(proof.proof.commitments[0], &sizes, &mut channel);
        verifier.verify_values(TreeVec(sampled_points), proof.proof, &mut channel)
    }

    fn prove_and_verify_pcs<
        B: BackendForChannel<Blake2sMerkleChannel>,
        const STORE_COEFFS: bool,
    >(
        memory_mode: ProverMemoryMode,
    ) -> Result<(), VerificationError> {
        prove_and_verify_pcs_with_params::<B, STORE_COEFFS, 10, 8>(memory_mode)
    }

    #[test]
    fn test_pcs_prove_and_verify_cpu() {
        assert!(prove_and_verify_pcs::<CpuBackend, true>(ProverMemoryMode::Fast).is_ok());
    }
    #[test]
    fn test_pcs_prove_and_verify_simd() {
        assert!(prove_and_verify_pcs::<SimdBackend, true>(ProverMemoryMode::Fast).is_ok());
    }
    #[test]
    fn test_pcs_prove_and_verify_simd_with_barycentric() {
        assert!(prove_and_verify_pcs::<SimdBackend, false>(ProverMemoryMode::Fast).is_ok());
    }

    #[test]
    fn test_pcs_prove_and_verify_cpu_low_memory() {
        assert!(prove_and_verify_pcs::<CpuBackend, true>(ProverMemoryMode::LowMemory).is_ok());
    }

    #[test]
    fn test_pcs_prove_and_verify_simd_low_memory() {
        assert!(prove_and_verify_pcs::<SimdBackend, true>(ProverMemoryMode::LowMemory).is_ok());
    }

    #[test]
    fn test_pcs_prove_and_verify_cpu_low_memory_without_explicit_coeffs() {
        assert!(prove_and_verify_pcs::<CpuBackend, false>(ProverMemoryMode::LowMemory).is_ok());
    }

    /// Tests that SIMD quotient computation produces low-degree quotients even when the trace
    /// polynomial is very small (subdomain size < N_LANES).
    #[test]
    fn test_simd_quotients_are_low_degree_small_trace() {
        let mut rng = SmallRng::seed_from_u64(0);
        const LOG_SIZE: u32 = 3;
        const LOG_BLOWUP_FACTOR: u32 = 4;

        let polynomial = CpuCirclePoly::new((0..1 << LOG_SIZE).map(M31::from).collect());
        let eval_domain = CanonicCoset::new(LOG_SIZE + LOG_BLOWUP_FACTOR).circle_domain();
        let cpu_eval = polynomial.evaluate(eval_domain);
        let simd_eval = CircleEvaluation::new(eval_domain, BaseColumn::from_cpu(&cpu_eval.values));

        let sample_points = [
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
        ];
        let samples = sample_points
            .into_iter()
            .map(|x| PointSample {
                point: x,
                value: polynomial.eval_at_point(x),
            })
            .collect_vec();
        let rand_coeff =
            SecureField::from_m31_array(std::array::from_fn(|_| M31::from(rng.gen::<u32>())));
        let twiddles = SimdBackend::precompute_twiddles(eval_domain.half_coset);
        let quot_eval = compute_fri_quotients(
            &TreeVec(vec![vec![&simd_eval]]),
            &TreeVec(vec![vec![samples]]),
            rand_coeff,
            LOG_SIZE + LOG_BLOWUP_FACTOR,
            &twiddles,
            LOG_BLOWUP_FACTOR,
        );
        let mut coeffs = quot_eval
            .values
            .columns
            .iter()
            .map(|c| CpuCircleEvaluation::new(eval_domain, c.to_cpu()).interpolate())
            .collect_vec();
        let zeros = coeffs[0].coeffs.split_off((1 << LOG_SIZE) - 1);

        assert!(zeros.iter().all(|c| c.is_zero()));
    }

    fn run_manual_pcs_memory_measurement(memory_mode: ProverMemoryMode) {
        const N_COLS: usize = 128;
        const LIFTING_LOG_SIZE: u32 = 16;

        let start = Instant::now();
        let result = prove_and_verify_pcs_with_params::<CpuBackend, true, N_COLS, LIFTING_LOG_SIZE>(
            memory_mode,
        );
        let elapsed_ms = start.elapsed().as_millis();
        assert!(result.is_ok());

        println!(
            "manual_pcs_memory_measurement mode={memory_mode:?} n_cols={} lifting_log_size={} elapsed_ms={elapsed_ms}",
            N_COLS,
            LIFTING_LOG_SIZE,
        );
    }

    #[test]
    #[ignore = "manual memory measurement"]
    fn measure_pcs_memory_fast() {
        run_manual_pcs_memory_measurement(ProverMemoryMode::Fast);
    }

    #[test]
    #[ignore = "manual memory measurement"]
    fn measure_pcs_memory_low_memory() {
        run_manual_pcs_memory_measurement(ProverMemoryMode::LowMemory);
    }

    /// Tests that low-memory FRI quotient computation produces the same result as fast mode
    /// when there are multiple commitment trees with many columns at the same log_size.
    ///
    /// This is a regression test for a bug where `lift_and_accumulate` received entries in
    /// non-monotonic size order across trees, causing u32 underflow in the log_ratio calculation
    /// and silently incorrect results in release mode.
    #[test]
    fn test_pcs_prove_and_verify_simd_low_memory_multi_tree_many_columns() {
        // 20 columns per tree forces multiple batches of 8 (batch_size=8, ceil(20/8)=3 batches).
        // With 2 trees at the same log_size, the cross-tree ordering bug is triggered.
        const N_COLS_PER_TREE: usize = 20;
        const LIFTING_LOG_SIZE: u32 = 8;

        let mut channel_fast = Blake2sChannel::default();
        let mut channel_low = Blake2sChannel::default();
        let config = PcsConfig::default();
        let twiddles = SimdBackend::precompute_twiddles(
            CanonicCoset::new(LIFTING_LOG_SIZE + config.fri_config.log_blowup_factor).half_coset(),
        );

        let polys_tree0: Vec<CircleCoefficients<SimdBackend>> = (0..N_COLS_PER_TREE)
            .map(|i| {
                CircleCoefficients::new(
                    (0..1 << LIFTING_LOG_SIZE)
                        .map(|j| M31::from((i * 1000 + j) as u32))
                        .collect(),
                )
            })
            .collect();
        // Second tree: same number of columns at the same log_size
        let polys_tree1: Vec<CircleCoefficients<SimdBackend>> = (0..N_COLS_PER_TREE)
            .map(|i| {
                CircleCoefficients::new(
                    (0..1 << LIFTING_LOG_SIZE)
                        .map(|j| M31::from((i * 2000 + j + 1) as u32))
                        .collect(),
                )
            })
            .collect();

        let sample_point = SECURE_FIELD_CIRCLE_GEN.mul(12345678u128);
        let sampled_points = TreeVec(vec![
            (0..N_COLS_PER_TREE)
                .map(|_| vec![sample_point])
                .collect_vec(),
            (0..N_COLS_PER_TREE)
                .map(|_| vec![sample_point])
                .collect_vec(),
        ]);

        // Fast mode.
        let mut cs_fast =
            CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);
        cs_fast.set_store_polynomials_coefficients();
        let mut tree_builder = cs_fast.tree_builder();
        tree_builder.extend_polys(polys_tree0.clone());
        tree_builder.commit(&mut channel_fast);
        let mut tree_builder = cs_fast.tree_builder();
        tree_builder.extend_polys(polys_tree1.clone());
        tree_builder.commit(&mut channel_fast);
        let fast_proof = cs_fast.prove_values(sampled_points.clone(), &mut channel_fast);

        // Low-memory mode.
        let mut cs_low =
            CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);
        cs_low.set_store_polynomials_coefficients();
        cs_low.set_memory_mode(ProverMemoryMode::LowMemory);
        let mut tree_builder = cs_low.tree_builder();
        tree_builder.extend_polys(polys_tree0);
        tree_builder.commit(&mut channel_low);
        let mut tree_builder = cs_low.tree_builder();
        tree_builder.extend_polys(polys_tree1);
        tree_builder.commit(&mut channel_low);
        let low_proof = cs_low.prove_values(sampled_points.clone(), &mut channel_low);

        // Both should verify successfully.
        let mut verifier_channel_fast = Blake2sChannel::default();
        let mut verifier_channel_low = Blake2sChannel::default();
        let mut verifier_fast = CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
        let mut verifier_low = CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
        let sizes: Vec<u32> = (0..N_COLS_PER_TREE).map(|_| LIFTING_LOG_SIZE).collect_vec();
        verifier_fast.commit(
            fast_proof.proof.commitments[0],
            &sizes,
            &mut verifier_channel_fast,
        );
        verifier_fast.commit(
            fast_proof.proof.commitments[1],
            &sizes,
            &mut verifier_channel_fast,
        );
        verifier_low.commit(
            low_proof.proof.commitments[0],
            &sizes,
            &mut verifier_channel_low,
        );
        verifier_low.commit(
            low_proof.proof.commitments[1],
            &sizes,
            &mut verifier_channel_low,
        );
        assert!(
            verifier_fast
                .verify_values(
                    sampled_points.clone(),
                    fast_proof.proof,
                    &mut verifier_channel_fast
                )
                .is_ok(),
            "fast mode verification failed"
        );
        assert!(
            verifier_low
                .verify_values(sampled_points, low_proof.proof, &mut verifier_channel_low)
                .is_ok(),
            "low-memory mode verification failed (multi-tree regression)"
        );
    }
}
