#[cfg(not(feature = "parallel"))]
use std::iter::zip;

use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::utils::SliceExt;
use crate::core::Fraction;
use crate::parallel_iter;
use crate::prover::backend::cpu::lookups::gkr::gen_eq_evals as cpu_gen_eq_evals;
use crate::prover::backend::simd::column::SecureColumn;
use crate::prover::backend::simd::m31::{LOG_N_LANES, N_LANES};
use crate::prover::backend::simd::qm31::PackedSecureField;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::{Column, CpuBackend};
use crate::prover::lookups::gkr_prover::{
    correct_sum_as_poly_in_first_variable, EqEvals, GkrMultivariatePolyOracle, GkrOps, Layer,
};
use crate::prover::lookups::mle::Mle;
use crate::prover::lookups::sumcheck::MultivariatePolyOracle;
use crate::prover::lookups::utils::{Reciprocal, UnivariatePoly};

/// Number of packed terms each parallel task processes in the GKR/MLE kernels.
///
/// Small enough that the first sumcheck rounds of a 2^16-variable instance still fan out
/// across all cores (~15us of work per task, well above rayon's scheduling overhead).
pub(crate) const PACKED_CHUNK_SIZE: usize = 1 << 7;

/// Computes `(sum_i term(i).0, sum_i term(i).1)` over `i` in `0..n_packed_terms`, reduced
/// to scalars.
///
/// Chunked so the `parallel` feature can fan the accumulation out across threads. Field
/// addition is exact, so the result is bit-identical to sequential accumulation for any
/// chunking.
fn sum_packed_terms(
    n_packed_terms: usize,
    term: impl Fn(usize) -> (PackedSecureField, PackedSecureField) + Send + Sync,
) -> (SecureField, SecureField) {
    let n_chunks = n_packed_terms.div_ceil(PACKED_CHUNK_SIZE);
    let (eval_at_0, eval_at_2) = parallel_iter!(0..n_chunks)
        .map(|chunk| {
            let mut acc_at_0 = PackedSecureField::zero();
            let mut acc_at_2 = PackedSecureField::zero();
            let start = chunk * PACKED_CHUNK_SIZE;
            let end = (start + PACKED_CHUNK_SIZE).min(n_packed_terms);
            for i in start..end {
                let (term_at_0, term_at_2) = term(i);
                acc_at_0 += term_at_0;
                acc_at_2 += term_at_2;
            }
            (acc_at_0, acc_at_2)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .fold(
            (PackedSecureField::zero(), PackedSecureField::zero()),
            |(acc_at_0, acc_at_2), (chunk_at_0, chunk_at_2)| {
                (acc_at_0 + chunk_at_0, acc_at_2 + chunk_at_2)
            },
        );
    (eval_at_0.pointwise_sum(), eval_at_2.pointwise_sum())
}

impl GkrOps for SimdBackend {
    fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Mle<Self, SecureField> {
        if y.len() < LOG_N_LANES as usize {
            return Mle::new(cpu_gen_eq_evals(y, v).into_iter().collect());
        }

        // Start DP with CPU backend to avoid dealing with instances smaller than a SIMD vector.
        let (y_rem, y_last_chunk) = y.split_last_chunk::<{ LOG_N_LANES as usize }>().unwrap();
        let initial = SecureColumn::from_iter(cpu_gen_eq_evals(y_last_chunk, v));
        assert_eq!(initial.len(), N_LANES);

        let packed_len = 1 << y_rem.len();
        let mut data = initial.data;

        // Zero-init the doubling buffer. The loop below overwrites every element before
        // reading it, but a safe initialized allocation avoids `set_len`-before-init
        // (which violates its safety contract and forms slices over uninitialized
        // memory). Cost: one memset-speed pass, negligible vs the multiplications below.
        data.resize(packed_len, PackedSecureField::zero());

        for (i, &y_j) in y_rem.iter().rev().enumerate() {
            let packed_y_j = PackedSecureField::broadcast(y_j);

            let (lhs_evals, rest) = data.split_at_mut(1 << i);
            let rhs_evals = &mut rest[..1 << i];

            #[cfg(not(feature = "parallel"))]
            let iter = zip(lhs_evals, rhs_evals);
            #[cfg(feature = "parallel")]
            let iter = lhs_evals
                .par_iter_mut()
                .zip(rhs_evals)
                .with_min_len(PACKED_CHUNK_SIZE);

            iter.for_each(|(lhs, rhs)| {
                // Equivalent to:
                // `rhs = eq(1, y_j) * lhs`,
                // `lhs = eq(0, y_j) * lhs`
                *rhs = *lhs * packed_y_j;
                *lhs -= *rhs;
            });
        }

        let length = packed_len * N_LANES;
        Mle::new(SecureColumn { data, length })
    }

    fn next_layer(layer: &Layer<Self>) -> Layer<Self> {
        // Offload to CPU backend to avoid dealing with instances smaller than a SIMD vector.
        if layer.n_variables() as u32 <= LOG_N_LANES {
            return into_simd_layer(layer.to_cpu().next_layer().unwrap());
        }

        match layer {
            Layer::GrandProduct(col) => next_grand_product_layer(col),
            Layer::LogUpGeneric {
                numerators,
                denominators,
            } => next_logup_generic_layer(numerators, denominators),
            Layer::LogUpMultiplicities {
                numerators,
                denominators,
            } => next_logup_multiplicities_layer(numerators, denominators),
            Layer::LogUpSingles { denominators } => next_logup_singles_layer(denominators),
        }
    }

    fn sum_as_poly_in_first_variable(
        h: &GkrMultivariatePolyOracle<'_, Self>,
        claim: SecureField,
    ) -> UnivariatePoly<SecureField> {
        let n_variables = h.n_variables();
        let n_terms = 1 << n_variables.saturating_sub(1);
        let eq_evals = h.eq_evals.as_ref();
        // Vector used to generate evaluations of `eq(x, y)` for `x` in the boolean hypercube.
        let y = eq_evals.y();

        // Offload to CPU backend to avoid dealing with instances smaller than a SIMD vector.
        if n_terms < N_LANES {
            return h.to_cpu().sum_as_poly_in_first_variable(claim);
        }

        let n_packed_terms = n_terms / N_LANES;
        let packed_lambda = PackedSecureField::broadcast(h.lambda);

        let (mut eval_at_0, mut eval_at_2) = match &h.input_layer {
            Layer::GrandProduct(col) => eval_grand_product_sum(eq_evals, col, n_packed_terms),
            Layer::LogUpGeneric {
                numerators,
                denominators,
            } => eval_logup_generic_sum(
                eq_evals,
                numerators,
                denominators,
                n_packed_terms,
                packed_lambda,
            ),
            Layer::LogUpMultiplicities {
                numerators,
                denominators,
            } => eval_logup_multiplicities_sum(
                eq_evals,
                numerators,
                denominators,
                n_packed_terms,
                packed_lambda,
            ),
            Layer::LogUpSingles { denominators } => {
                eval_logup_singles_sum(eq_evals, denominators, n_packed_terms, packed_lambda)
            }
        };

        eval_at_0 *= h.eq_fixed_var_correction;
        eval_at_2 *= h.eq_fixed_var_correction;
        correct_sum_as_poly_in_first_variable(eval_at_0, eval_at_2, claim, y, n_variables)
    }
}

/// Generates the next GKR layer for Grand Product.
///
/// Assumption: `len(layer) > N_LANES`.
fn next_grand_product_layer(layer: &Mle<SimdBackend, SecureField>) -> Layer<SimdBackend> {
    assert!(layer.len() > N_LANES);
    let next_layer_len = layer.len() / 2;

    let chunks = layer.data.checked_as_chunks();

    #[cfg(not(feature = "parallel"))]
    let iter = chunks.iter();
    #[cfg(feature = "parallel")]
    let iter = chunks.par_iter().with_min_len(PACKED_CHUNK_SIZE);

    let data = iter
        .map(|&[a, b]| {
            let (evens, odds) = a.deinterleave(b);
            evens * odds
        })
        .collect();

    Layer::GrandProduct(Mle::new(SecureColumn {
        data,
        length: next_layer_len,
    }))
}

/// Generates the next GKR layer for LogUp.
///
/// Assumption: `len(denominators) > N_LANES`.
fn next_logup_generic_layer(
    numerators: &Mle<SimdBackend, SecureField>,
    denominators: &Mle<SimdBackend, SecureField>,
) -> Layer<SimdBackend> {
    assert!(denominators.len() > N_LANES);
    assert_eq!(numerators.len(), denominators.len());

    let next_layer_len = denominators.len() / 2;
    let next_layer_packed_len = next_layer_len / N_LANES;

    #[cfg(not(feature = "parallel"))]
    let iter = 0..next_layer_packed_len;
    #[cfg(feature = "parallel")]
    let iter = (0..next_layer_packed_len)
        .into_par_iter()
        .with_min_len(PACKED_CHUNK_SIZE);

    let (next_numerators, next_denominators): (Vec<_>, Vec<_>) = iter
        .map(|i| {
            let (n_even, n_odd) = numerators.data[i * 2].deinterleave(numerators.data[i * 2 + 1]);
            let (d_even, d_odd) =
                denominators.data[i * 2].deinterleave(denominators.data[i * 2 + 1]);

            let Fraction {
                numerator,
                denominator,
            } = Fraction::new(n_even, d_even) + Fraction::new(n_odd, d_odd);

            (numerator, denominator)
        })
        .unzip();

    let next_numerators = SecureColumn {
        data: next_numerators,
        length: next_layer_len,
    };

    let next_denominators = SecureColumn {
        data: next_denominators,
        length: next_layer_len,
    };

    Layer::LogUpGeneric {
        numerators: Mle::new(next_numerators),
        denominators: Mle::new(next_denominators),
    }
}

/// Generates the next GKR layer for LogUp.
///
/// Assumption: `len(denominators) > N_LANES`.
// TODO(andrew): Code duplication of `next_logup_generic_layer`. Consider unifying these.
fn next_logup_multiplicities_layer(
    numerators: &Mle<SimdBackend, BaseField>,
    denominators: &Mle<SimdBackend, SecureField>,
) -> Layer<SimdBackend> {
    assert!(denominators.len() > N_LANES);
    assert_eq!(numerators.len(), denominators.len());

    let next_layer_len = denominators.len() / 2;
    let next_layer_packed_len = next_layer_len / N_LANES;

    #[cfg(not(feature = "parallel"))]
    let iter = 0..next_layer_packed_len;
    #[cfg(feature = "parallel")]
    let iter = (0..next_layer_packed_len)
        .into_par_iter()
        .with_min_len(PACKED_CHUNK_SIZE);

    let (next_numerators, next_denominators): (Vec<_>, Vec<_>) = iter
        .map(|i| {
            let (n_even, n_odd) = numerators.data[i * 2].deinterleave(numerators.data[i * 2 + 1]);
            let (d_even, d_odd) =
                denominators.data[i * 2].deinterleave(denominators.data[i * 2 + 1]);

            let Fraction {
                numerator,
                denominator,
            } = Fraction::new(n_even, d_even) + Fraction::new(n_odd, d_odd);

            (numerator, denominator)
        })
        .unzip();

    let next_numerators = SecureColumn {
        data: next_numerators,
        length: next_layer_len,
    };

    let next_denominators = SecureColumn {
        data: next_denominators,
        length: next_layer_len,
    };

    Layer::LogUpGeneric {
        numerators: Mle::new(next_numerators),
        denominators: Mle::new(next_denominators),
    }
}

/// Generates the next GKR layer for LogUp.
///
/// Assumption: `len(denominators) > N_LANES`.
fn next_logup_singles_layer(denominators: &Mle<SimdBackend, SecureField>) -> Layer<SimdBackend> {
    assert!(denominators.len() > N_LANES);

    let next_layer_len = denominators.len() / 2;
    let next_layer_packed_len = next_layer_len / N_LANES;

    #[cfg(not(feature = "parallel"))]
    let iter = 0..next_layer_packed_len;
    #[cfg(feature = "parallel")]
    let iter = (0..next_layer_packed_len)
        .into_par_iter()
        .with_min_len(PACKED_CHUNK_SIZE);

    let (next_numerators, next_denominators): (Vec<_>, Vec<_>) = iter
        .map(|i| {
            let (d_even, d_odd) =
                denominators.data[i * 2].deinterleave(denominators.data[i * 2 + 1]);

            let Fraction {
                numerator,
                denominator,
            } = Reciprocal::new(d_even) + Reciprocal::new(d_odd);

            (numerator, denominator)
        })
        .unzip();

    let next_numerators = SecureColumn {
        data: next_numerators,
        length: next_layer_len,
    };

    let next_denominators = SecureColumn {
        data: next_denominators,
        length: next_layer_len,
    };

    Layer::LogUpGeneric {
        numerators: Mle::new(next_numerators),
        denominators: Mle::new(next_denominators),
    }
}

/// Evaluates `sum_x eq(({0}^|r|, 0, x), y) * inp(r, t, x, 0) * inp(r, t, x, 1)` at `t=0` and `t=2`.
///
/// Output of the form: `(eval_at_0, eval_at_2)`.
fn eval_grand_product_sum(
    eq_evals: &EqEvals<SimdBackend>,
    col: &Mle<SimdBackend, SecureField>,
    n_packed_terms: usize,
) -> (SecureField, SecureField) {
    sum_packed_terms(n_packed_terms, |i| {
        // Input polynomial at points `(r, {0, 1, 2}, bits(i), v, {0, 1})`
        // for all `v` in `{0, 1}^LOG_N_SIMD_LANES`.
        let (inp_at_r0iv0, inp_at_r0iv1) = col.data[i * 2].deinterleave(col.data[i * 2 + 1]);
        let (inp_at_r1iv0, inp_at_r1iv1) =
            col.data[(n_packed_terms + i) * 2].deinterleave(col.data[(n_packed_terms + i) * 2 + 1]);
        // Note `inp(r, t, x) = eq(t, 0) * inp(r, 0, x) + eq(t, 1) * inp(r, 1, x)`
        //   => `inp(r, 2, x) = 2 * inp(r, 1, x) - inp(r, 0, x)`
        let inp_at_r2iv0 = inp_at_r1iv0.double() - inp_at_r0iv0;
        let inp_at_r2iv1 = inp_at_r1iv1.double() - inp_at_r0iv1;

        // Product polynomial `prod(x) = inp(x, 0) * inp(x, 1)` at points `(r, {0, 2}, bits(i), v)`.
        // for all `v` in `{0, 1}^LOG_N_SIMD_LANES`.
        let prod_at_r2iv = inp_at_r2iv0 * inp_at_r2iv1;
        let prod_at_r0iv = inp_at_r0iv0 * inp_at_r0iv1;

        let eq_eval_at_0iv = eq_evals.data[i];
        (eq_eval_at_0iv * prod_at_r0iv, eq_eval_at_0iv * prod_at_r2iv)
    })
}

fn eval_logup_generic_sum(
    eq_evals: &EqEvals<SimdBackend>,
    numerators: &Mle<SimdBackend, SecureField>,
    denominators: &Mle<SimdBackend, SecureField>,
    n_packed_terms: usize,
    packed_lambda: PackedSecureField,
) -> (SecureField, SecureField) {
    let inp_numerator = &numerators.data;
    let inp_denom = &denominators.data;

    sum_packed_terms(n_packed_terms, |i| {
        // Input polynomials at points `(r, {0, 1, 2}, bits(i), v, {0, 1})`
        // for all `v` in `{0, 1}^LOG_N_SIMD_LANES`.
        let (inp_numerator_at_r0iv0, inp_numerator_at_r0iv1) =
            inp_numerator[i * 2].deinterleave(inp_numerator[i * 2 + 1]);
        let (inp_denom_at_r0iv0, inp_denom_at_r0iv1) =
            inp_denom[i * 2].deinterleave(inp_denom[i * 2 + 1]);
        let (inp_numerator_at_r1iv0, inp_numerator_at_r1iv1) = inp_numerator
            [(n_packed_terms + i) * 2]
            .deinterleave(inp_numerator[(n_packed_terms + i) * 2 + 1]);
        let (inp_denom_at_r1iv0, inp_denom_at_r1iv1) = inp_denom[(n_packed_terms + i) * 2]
            .deinterleave(inp_denom[(n_packed_terms + i) * 2 + 1]);
        // Note `inp_denom(r, t, x) = eq(t, 0) * inp_denom(r, 0, x) + eq(t, 1) * inp_denom(r, 1, x)`
        //   => `inp_denom(r, 2, x) = 2 * inp_denom(r, 1, x) - inp_denom(r, 0, x)`
        let inp_numerator_at_r2iv0 = inp_numerator_at_r1iv0.double() - inp_numerator_at_r0iv0;
        let inp_numerator_at_r2iv1 = inp_numerator_at_r1iv1.double() - inp_numerator_at_r0iv1;
        let inp_denom_at_r2iv0 = inp_denom_at_r1iv0.double() - inp_denom_at_r0iv0;
        let inp_denom_at_r2iv1 = inp_denom_at_r1iv1.double() - inp_denom_at_r0iv1;

        // Fraction addition polynomials:
        // - `numerator(x) = inp_numerator(x, 0) * inp_denom(x, 1) + inp_numerator(x, 1) *
        //   inp_denom(x, 0)`
        // - `denom(x) = inp_denom(x, 0) * inp_denom(x, 1)`.
        // at points `(r, {0, 2}, bits(i), v)` for all `v` in `{0, 1}^LOG_N_SIMD_LANES`.
        let Fraction {
            numerator: numerator_at_r0iv,
            denominator: denom_at_r0iv,
        } = Fraction::new(inp_numerator_at_r0iv0, inp_denom_at_r0iv0)
            + Fraction::new(inp_numerator_at_r0iv1, inp_denom_at_r0iv1);
        let Fraction {
            numerator: numerator_at_r2iv,
            denominator: denom_at_r2iv,
        } = Fraction::new(inp_numerator_at_r2iv0, inp_denom_at_r2iv0)
            + Fraction::new(inp_numerator_at_r2iv1, inp_denom_at_r2iv1);

        let eq_eval_at_0iv = eq_evals.data[i];
        (
            eq_eval_at_0iv * (numerator_at_r0iv + packed_lambda * denom_at_r0iv),
            eq_eval_at_0iv * (numerator_at_r2iv + packed_lambda * denom_at_r2iv),
        )
    })
}

// TODO(andrew): Code duplication of `eval_logup_generic_sum`. Consider unifying these.
fn eval_logup_multiplicities_sum(
    eq_evals: &EqEvals<SimdBackend>,
    numerators: &Mle<SimdBackend, BaseField>,
    denominators: &Mle<SimdBackend, SecureField>,
    n_packed_terms: usize,
    packed_lambda: PackedSecureField,
) -> (SecureField, SecureField) {
    let inp_numerator = &numerators.data;
    let inp_denom = &denominators.data;

    sum_packed_terms(n_packed_terms, |i| {
        // Input polynomials at points `(r, {0, 1, 2}, bits(i), v, {0, 1})`
        // for all `v` in `{0, 1}^LOG_N_SIMD_LANES`.
        let (inp_numerator_at_r0iv0, inp_numerator_at_r0iv1) =
            inp_numerator[i * 2].deinterleave(inp_numerator[i * 2 + 1]);
        let (inp_denom_at_r0iv0, inp_denom_at_r0iv1) =
            inp_denom[i * 2].deinterleave(inp_denom[i * 2 + 1]);
        let (inp_numerator_at_r1iv0, inp_numerator_at_r1iv1) = inp_numerator
            [(n_packed_terms + i) * 2]
            .deinterleave(inp_numerator[(n_packed_terms + i) * 2 + 1]);
        let (inp_denom_at_r1iv0, inp_denom_at_r1iv1) = inp_denom[(n_packed_terms + i) * 2]
            .deinterleave(inp_denom[(n_packed_terms + i) * 2 + 1]);
        // Note `inp_denom(r, t, x) = eq(t, 0) * inp_denom(r, 0, x) + eq(t, 1) * inp_denom(r, 1, x)`
        //   => `inp_denom(r, 2, x) = 2 * inp_denom(r, 1, x) - inp_denom(r, 0, x)`
        let inp_numerator_at_r2iv0 = inp_numerator_at_r1iv0.double() - inp_numerator_at_r0iv0;
        let inp_numerator_at_r2iv1 = inp_numerator_at_r1iv1.double() - inp_numerator_at_r0iv1;
        let inp_denom_at_r2iv0 = inp_denom_at_r1iv0.double() - inp_denom_at_r0iv0;
        let inp_denom_at_r2iv1 = inp_denom_at_r1iv1.double() - inp_denom_at_r0iv1;

        // Fraction addition polynomials:
        // - `numerator(x) = inp_numerator(x, 0) * inp_denom(x, 1) + inp_numerator(x, 1) *
        //   inp_denom(x, 0)`
        // - `denom(x) = inp_denom(x, 0) * inp_denom(x, 1)`.
        // at points `(r, {0, 2}, bits(i), v)` for all `v` in `{0, 1}^LOG_N_SIMD_LANES`.
        let Fraction {
            numerator: numerator_at_r0iv,
            denominator: denom_at_r0iv,
        } = Fraction::new(inp_numerator_at_r0iv0, inp_denom_at_r0iv0)
            + Fraction::new(inp_numerator_at_r0iv1, inp_denom_at_r0iv1);
        let Fraction {
            numerator: numerator_at_r2iv,
            denominator: denom_at_r2iv,
        } = Fraction::new(inp_numerator_at_r2iv0, inp_denom_at_r2iv0)
            + Fraction::new(inp_numerator_at_r2iv1, inp_denom_at_r2iv1);

        let eq_eval_at_0iv = eq_evals.data[i];
        (
            eq_eval_at_0iv * (numerator_at_r0iv + packed_lambda * denom_at_r0iv),
            eq_eval_at_0iv * (numerator_at_r2iv + packed_lambda * denom_at_r2iv),
        )
    })
}

/// Evaluates `sum_x eq(({0}^|r|, 0, x), y) * (inp_denom(r, t, x, 1) + inp_denom(r, t, x, 0) +
/// lambda * inp_denom(r, t, x, 0) * inp_denom(r, t, x, 1))` at `t=0` and `t=2`.
///
/// Output of the form: `(eval_at_0, eval_at_2)`.
fn eval_logup_singles_sum(
    eq_evals: &EqEvals<SimdBackend>,
    denominators: &Mle<SimdBackend, SecureField>,
    n_packed_terms: usize,
    packed_lambda: PackedSecureField,
) -> (SecureField, SecureField) {
    let inp_denom = &denominators.data;

    sum_packed_terms(n_packed_terms, |i| {
        // Input polynomial at points `(r, {0, 1, 2}, bits(i), v, {0, 1})`
        // for all `v` in `{0, 1}^LOG_N_SIMD_LANES`.
        let (inp_denom_at_r0iv0, inp_denom_at_r0iv1) =
            inp_denom[i * 2].deinterleave(inp_denom[i * 2 + 1]);
        let (inp_denom_at_r1iv0, inp_denom_at_r1iv1) = inp_denom[(n_packed_terms + i) * 2]
            .deinterleave(inp_denom[(n_packed_terms + i) * 2 + 1]);
        // Note `inp_denom(r, t, x) = eq(t, 0) * inp_denom(r, 0, x) + eq(t, 1) * inp_denom(r, 1, x)`
        //   => `inp_denom(r, 2, x) = 2 * inp_denom(r, 1, x) - inp_denom(r, 0, x)`
        let inp_denom_at_r2iv0 = inp_denom_at_r1iv0.double() - inp_denom_at_r0iv0;
        let inp_denom_at_r2iv1 = inp_denom_at_r1iv1.double() - inp_denom_at_r0iv1;

        // Fraction addition polynomials:
        // - `numerator(x) = inp_denom(x, 1) + inp_denom(x, 0)`
        // - `denom(x) = inp_denom(x, 0) * inp_denom(x, 1)`.
        // at points `(r, {0, 2}, bits(i), v)` for all `v` in `{0, 1}^LOG_N_SIMD_LANES`.
        let Fraction {
            numerator: numerator_at_r0iv,
            denominator: denom_at_r0iv,
        } = Reciprocal::new(inp_denom_at_r0iv0) + Reciprocal::new(inp_denom_at_r0iv1);
        let Fraction {
            numerator: numerator_at_r2iv,
            denominator: denom_at_r2iv,
        } = Reciprocal::new(inp_denom_at_r2iv0) + Reciprocal::new(inp_denom_at_r2iv1);

        let eq_eval_at_0iv = eq_evals.data[i];
        (
            eq_eval_at_0iv * (numerator_at_r0iv + packed_lambda * denom_at_r0iv),
            eq_eval_at_0iv * (numerator_at_r2iv + packed_lambda * denom_at_r2iv),
        )
    })
}

fn into_simd_layer(cpu_layer: Layer<CpuBackend>) -> Layer<SimdBackend> {
    match cpu_layer {
        Layer::GrandProduct(mle) => {
            Layer::GrandProduct(Mle::new(mle.into_evals().into_iter().collect()))
        }
        Layer::LogUpGeneric {
            numerators,
            denominators,
        } => Layer::LogUpGeneric {
            numerators: Mle::new(numerators.into_evals().into_iter().collect()),
            denominators: Mle::new(denominators.into_evals().into_iter().collect()),
        },
        Layer::LogUpMultiplicities {
            numerators,
            denominators,
        } => Layer::LogUpMultiplicities {
            numerators: Mle::new(numerators.into_evals().into_iter().collect()),
            denominators: Mle::new(denominators.into_evals().into_iter().collect()),
        },
        Layer::LogUpSingles { denominators } => Layer::LogUpSingles {
            denominators: Mle::new(denominators.into_evals().into_iter().collect()),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use num_traits::One;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::channel::Channel;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::test_utils::test_channel;
    use crate::core::Fraction;
    use crate::prover::backend::simd::SimdBackend;
    use crate::prover::backend::{Column, CpuBackend};
    use crate::prover::lookups::gkr_prover::{prove_batch, GkrOps, Layer};
    use crate::prover::lookups::gkr_verifier::{
        partially_verify_batch, Gate, GkrArtifact, GkrBatchProof, GkrError,
    };
    use crate::prover::lookups::mle::Mle;

    fn assert_gkr_results_match(
        case: &str,
        simd_proof: &GkrBatchProof,
        simd_artifact: &GkrArtifact,
        cpu_proof: &GkrBatchProof,
        cpu_artifact: &GkrArtifact,
    ) {
        assert_eq!(
            simd_proof.sumcheck_proofs.len(),
            cpu_proof.sumcheck_proofs.len(),
            "sumcheck proof count differs for {case}"
        );
        for (layer, (simd_sumcheck, cpu_sumcheck)) in
            zip(&simd_proof.sumcheck_proofs, &cpu_proof.sumcheck_proofs).enumerate()
        {
            assert_eq!(
                simd_sumcheck.round_polys.len(),
                cpu_sumcheck.round_polys.len(),
                "round polynomial count differs for {case}, layer {layer}"
            );
            for (round, (simd_poly, cpu_poly)) in
                zip(&simd_sumcheck.round_polys, &cpu_sumcheck.round_polys).enumerate()
            {
                assert_eq!(
                    &**simd_poly, &**cpu_poly,
                    "round polynomial differs for {case}, layer {layer}, round {round}"
                );
            }
        }

        assert_eq!(
            simd_proof.layer_masks_by_instance.len(),
            cpu_proof.layer_masks_by_instance.len(),
            "instance mask count differs for {case}"
        );
        for (instance, (simd_masks, cpu_masks)) in zip(
            &simd_proof.layer_masks_by_instance,
            &cpu_proof.layer_masks_by_instance,
        )
        .enumerate()
        {
            assert_eq!(
                simd_masks.len(),
                cpu_masks.len(),
                "layer mask count differs for {case}, instance {instance}"
            );
            for (layer, (simd_mask, cpu_mask)) in zip(simd_masks, cpu_masks).enumerate() {
                assert_eq!(
                    simd_mask.columns(),
                    cpu_mask.columns(),
                    "layer mask differs for {case}, instance {instance}, layer {layer}"
                );
            }
        }

        assert_eq!(
            simd_proof.output_claims_by_instance, cpu_proof.output_claims_by_instance,
            "output claims differ for {case}"
        );
        assert_eq!(
            simd_artifact.ood_point, cpu_artifact.ood_point,
            "OOD point differs for {case}"
        );
        assert_eq!(
            simd_artifact.claims_to_verify_by_instance, cpu_artifact.claims_to_verify_by_instance,
            "input claims differ for {case}"
        );
        assert_eq!(
            simd_artifact.n_variables_by_instance, cpu_artifact.n_variables_by_instance,
            "variable counts differ for {case}"
        );
    }

    #[test]
    fn gen_eq_evals_matches_cpu() {
        let two = BaseField::from(2).into();
        let y = [7, 3, 5, 6, 1, 1, 9].map(|v| BaseField::from(v).into());
        let eq_evals_cpu = CpuBackend::gen_eq_evals(&y, two);

        let eq_evals_simd = SimdBackend::gen_eq_evals(&y, two);

        assert_eq!(eq_evals_simd.to_cpu(), *eq_evals_cpu);
    }

    #[test]
    fn gen_eq_evals_with_small_assignment_matches_cpu() {
        let two = BaseField::from(2).into();
        let y = [7, 3, 5].map(|v| BaseField::from(v).into());
        let eq_evals_cpu = CpuBackend::gen_eq_evals(&y, two);

        let eq_evals_simd = SimdBackend::gen_eq_evals(&y, two);

        assert_eq!(eq_evals_simd.to_cpu(), *eq_evals_cpu);
    }

    /// Locks in that the SIMD and CPU paths produce identical GKR proofs and artifacts.
    #[test]
    fn simd_and_cpu_gkr_proofs_match() {
        const LOG_SIZES: [u32; 3] = [6, 7, 14];

        for log_size in LOG_SIZES {
            let n = 1usize << log_size;
            let mut rng = SmallRng::seed_from_u64(log_size.into());

            let grand_product = Layer::GrandProduct(Mle::<SimdBackend, SecureField>::new(
                (0..n).map(|_| rng.gen::<SecureField>()).collect(),
            ));
            let logup_generic = Layer::LogUpGeneric {
                numerators: Mle::<SimdBackend, SecureField>::new(
                    (0..n).map(|_| rng.gen::<SecureField>()).collect(),
                ),
                denominators: Mle::<SimdBackend, SecureField>::new(
                    (0..n).map(|_| rng.gen::<SecureField>()).collect(),
                ),
            };
            let logup_multiplicities = Layer::LogUpMultiplicities {
                numerators: Mle::<SimdBackend, BaseField>::new(
                    (0..n).map(|_| rng.gen::<BaseField>()).collect(),
                ),
                denominators: Mle::<SimdBackend, SecureField>::new(
                    (0..n).map(|_| rng.gen::<SecureField>()).collect(),
                ),
            };
            let logup_singles = Layer::LogUpSingles {
                denominators: Mle::<SimdBackend, SecureField>::new(
                    (0..n).map(|_| rng.gen::<SecureField>()).collect(),
                ),
            };

            for (variant, simd_layer) in [
                ("grand product", grand_product),
                ("generic LogUp", logup_generic),
                ("multiplicities LogUp", logup_multiplicities),
                ("singles LogUp", logup_singles),
            ] {
                let case = format!("{variant} at 2^{log_size}");
                let cpu_layer = simd_layer.to_cpu();
                let (simd_proof, simd_artifact) =
                    prove_batch(&mut test_channel(), vec![simd_layer]);
                let (cpu_proof, cpu_artifact) = prove_batch(&mut test_channel(), vec![cpu_layer]);

                assert_gkr_results_match(
                    &case,
                    &simd_proof,
                    &simd_artifact,
                    &cpu_proof,
                    &cpu_artifact,
                );
            }
        }
    }

    #[test]
    fn simd_and_cpu_unequal_gkr_batch_proofs_match() {
        const LOG_SIZES: [u32; 2] = [6, 14];

        let mut channel = test_channel();
        let simd_layers = LOG_SIZES.map(|log_size| {
            let values = channel.draw_secure_felts(1usize << log_size);
            Layer::GrandProduct(Mle::<SimdBackend, SecureField>::new(
                values.into_iter().collect(),
            ))
        });
        let cpu_layers = simd_layers.iter().map(Layer::to_cpu).collect();
        let (simd_proof, simd_artifact) =
            prove_batch(&mut test_channel(), simd_layers.into_iter().collect());
        let (cpu_proof, cpu_artifact) = prove_batch(&mut test_channel(), cpu_layers);

        assert_gkr_results_match(
            "unequal grand-product batch at 2^6 and 2^14",
            &simd_proof,
            &simd_artifact,
            &cpu_proof,
            &cpu_artifact,
        );
    }

    #[test]
    fn grand_product_works() -> Result<(), GkrError> {
        const N: usize = 1 << 8;
        let values = test_channel().draw_secure_felts(N);
        let product = values.iter().product();
        let col = Mle::<SimdBackend, SecureField>::new(values.into_iter().collect());
        let input_layer = Layer::GrandProduct(col.clone());
        let (proof, _) = prove_batch(&mut test_channel(), vec![input_layer]);

        let GkrArtifact {
            ood_point,
            claims_to_verify_by_instance,
            n_variables_by_instance: _,
        } = partially_verify_batch(vec![Gate::GrandProduct], &proof, &mut test_channel())?;

        assert_eq!(proof.output_claims_by_instance, [vec![product]]);
        assert_eq!(
            claims_to_verify_by_instance,
            [vec![col.eval_at_point(&ood_point)]]
        );
        Ok(())
    }

    #[test]
    fn logup_with_generic_trace_works() -> Result<(), GkrError> {
        const N: usize = 1 << 8;
        let mut rng = SmallRng::seed_from_u64(0);
        let numerators = (0..N).map(|_| rng.gen()).collect::<Vec<SecureField>>();
        let denominators = (0..N).map(|_| rng.gen()).collect::<Vec<SecureField>>();
        let sum = zip(&numerators, &denominators)
            .map(|(&n, &d)| Fraction::new(n, d))
            .sum::<Fraction<SecureField, SecureField>>();
        let numerators = Mle::<SimdBackend, SecureField>::new(numerators.into_iter().collect());
        let denominators = Mle::<SimdBackend, SecureField>::new(denominators.into_iter().collect());
        let input_layer = Layer::LogUpGeneric {
            numerators: numerators.clone(),
            denominators: denominators.clone(),
        };
        let (proof, _) = prove_batch(&mut test_channel(), vec![input_layer]);

        let GkrArtifact {
            ood_point,
            claims_to_verify_by_instance,
            n_variables_by_instance: _,
        } = partially_verify_batch(vec![Gate::LogUp], &proof, &mut test_channel())?;

        assert_eq!(claims_to_verify_by_instance.len(), 1);
        assert_eq!(proof.output_claims_by_instance.len(), 1);
        assert_eq!(
            claims_to_verify_by_instance[0],
            [
                numerators.eval_at_point(&ood_point),
                denominators.eval_at_point(&ood_point)
            ]
        );
        assert_eq!(
            proof.output_claims_by_instance[0],
            [sum.numerator, sum.denominator]
        );
        Ok(())
    }

    #[test]
    fn logup_with_multiplicities_trace_works() -> Result<(), GkrError> {
        const N: usize = 1 << 8;
        let mut rng = SmallRng::seed_from_u64(0);
        let numerators = (0..N).map(|_| rng.gen()).collect::<Vec<BaseField>>();
        let denominators = (0..N).map(|_| rng.gen()).collect::<Vec<SecureField>>();
        let sum = zip(&numerators, &denominators)
            .map(|(&n, &d)| Fraction::new(n.into(), d))
            .sum::<Fraction<SecureField, SecureField>>();
        let numerators = Mle::<SimdBackend, BaseField>::new(numerators.into_iter().collect());
        let denominators = Mle::<SimdBackend, SecureField>::new(denominators.into_iter().collect());
        let input_layer = Layer::LogUpMultiplicities {
            numerators: numerators.clone(),
            denominators: denominators.clone(),
        };
        let (proof, _) = prove_batch(&mut test_channel(), vec![input_layer]);

        let GkrArtifact {
            ood_point,
            claims_to_verify_by_instance,
            n_variables_by_instance: _,
        } = partially_verify_batch(vec![Gate::LogUp], &proof, &mut test_channel())?;

        assert_eq!(claims_to_verify_by_instance.len(), 1);
        assert_eq!(proof.output_claims_by_instance.len(), 1);
        assert_eq!(
            claims_to_verify_by_instance[0],
            [
                numerators.eval_at_point(&ood_point),
                denominators.eval_at_point(&ood_point)
            ]
        );
        assert_eq!(
            proof.output_claims_by_instance[0],
            [sum.numerator, sum.denominator]
        );
        Ok(())
    }

    #[test]
    fn logup_with_singles_trace_works() -> Result<(), GkrError> {
        const N: usize = 1 << 8;
        let mut rng = SmallRng::seed_from_u64(0);
        let denominators = (0..N).map(|_| rng.gen()).collect::<Vec<SecureField>>();
        let sum = denominators
            .iter()
            .map(|&d| Fraction::new(SecureField::one(), d))
            .sum::<Fraction<SecureField, SecureField>>();
        let denominators = Mle::<SimdBackend, SecureField>::new(denominators.into_iter().collect());
        let input_layer = Layer::LogUpSingles {
            denominators: denominators.clone(),
        };
        let (proof, _) = prove_batch(&mut test_channel(), vec![input_layer]);

        let GkrArtifact {
            ood_point,
            claims_to_verify_by_instance,
            n_variables_by_instance: _,
        } = partially_verify_batch(vec![Gate::LogUp], &proof, &mut test_channel())?;

        assert_eq!(claims_to_verify_by_instance.len(), 1);
        assert_eq!(proof.output_claims_by_instance.len(), 1);
        assert_eq!(
            claims_to_verify_by_instance[0],
            [SecureField::one(), denominators.eval_at_point(&ood_point)]
        );
        assert_eq!(
            proof.output_claims_by_instance[0],
            [sum.numerator, sum.denominator]
        );
        Ok(())
    }
}
