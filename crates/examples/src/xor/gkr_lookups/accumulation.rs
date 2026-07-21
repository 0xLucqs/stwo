use std::iter::zip;
use std::ops::{AddAssign, Mul};

use educe::Educe;
use num_traits::One;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use stwo::core::circle::M31_CIRCLE_LOG_ORDER;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
#[cfg(feature = "parallel")]
use stwo::prover::backend::simd::column::SecureColumn;
#[cfg(feature = "parallel")]
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::Backend;
#[cfg(feature = "parallel")]
use stwo::prover::backend::Column;
use stwo::prover::lookups::mle::Mle;
use stwo::prover::AccumulationOps;

pub const MIN_LOG_BLOWUP_FACTOR: u32 = 1;

#[cfg(feature = "parallel")]
const COMBINE_CHUNK_SIZE: usize = 1 << 10;

/// Max number of variables for multilinear polynomials that get compiled into a univariate
/// IOP for multilinear eval at point.
pub const MAX_MLE_N_VARIABLES: u32 = M31_CIRCLE_LOG_ORDER - MIN_LOG_BLOWUP_FACTOR;

/// Collection of [`Mle`]s grouped by their number of variables.
pub struct MleCollection<B: Backend> {
    mles_by_n_variables: Vec<Option<Vec<DynMle<B>>>>,
}

impl<B: Backend> MleCollection<B> {
    /// Appends an [`Mle`] to the back of the collection.
    pub fn push(&mut self, mle: impl Into<DynMle<B>>) {
        let mle = mle.into();
        let mles = self.mles_by_n_variables[mle.n_variables()].get_or_insert(Vec::new());
        mles.push(mle);
    }
}

impl MleCollection<SimdBackend> {
    /// Performs a random linear combination of all MLEs, grouped by their number of variables.
    ///
    /// For `n` accumulated MLEs in a group, the `i`th MLE is multiplied by `alpha^(n-1-i)`.
    /// MLEs are returned in ascending order by number of variables.
    pub fn random_linear_combine_by_n_variables(
        self,
        alpha: SecureField,
    ) -> Vec<Mle<SimdBackend, SecureField>> {
        self.mles_by_n_variables
            .into_iter()
            .flatten()
            .map(|mles| mle_random_linear_combination(mles, alpha))
            .collect()
    }
}

/// # Panics
///
/// Panics if `mles` is empty or all MLEs don't have the same number of variables.
fn mle_random_linear_combination(
    mles: Vec<DynMle<SimdBackend>>,
    random_coeff: SecureField,
) -> Mle<SimdBackend, SecureField> {
    assert!(!mles.is_empty());
    let n_variables = mles[0].n_variables();
    assert!(mles.iter().all(|mle| mle.n_variables() == n_variables));
    let coeff_powers =
        <SimdBackend as AccumulationOps>::generate_secure_powers(random_coeff, mles.len());
    let mut mle_and_coeff = zip(mles, coeff_powers.into_iter().rev());

    // The last value can initialize the accumulator.
    let (mle, coeff) = mle_and_coeff.next_back().unwrap();
    assert!(coeff.is_one());
    let mut acc_mle = mle.into_secure_mle();

    for (mle, coeff) in mle_and_coeff {
        match mle {
            DynMle::Base(mle) => combine(&mut acc_mle.data, &mle.data, coeff.into()),
            DynMle::Secure(mle) => combine(&mut acc_mle.data, &mle.data, coeff.into()),
        }
    }

    acc_mle
}

/// Computes all `acc[i] += alpha * v[i]`.
#[cfg(not(feature = "parallel"))]
pub fn combine<EF: AddAssign + Mul<F, Output = EF> + Copy, F: Copy>(
    acc: &mut [EF],
    v: &[F],
    alpha: EF,
) {
    assert_eq!(acc.len(), v.len());
    zip(acc, v).for_each(|(acc, &v)| *acc += alpha * v);
}

/// Computes all `acc[i] += alpha * v[i]` in disjoint chunks.
#[cfg(feature = "parallel")]
pub fn combine<EF, F>(acc: &mut [EF], v: &[F], alpha: EF)
where
    EF: AddAssign + Mul<F, Output = EF> + Copy + Send + Sync,
    F: Copy + Sync,
{
    assert_eq!(acc.len(), v.len());
    acc.par_chunks_mut(COMBINE_CHUNK_SIZE)
        .zip(v.par_chunks(COMBINE_CHUNK_SIZE))
        .for_each(|(acc, v)| {
            zip(acc, v).for_each(|(acc, &v)| *acc += alpha * v);
        });
}

impl<B: Backend> Default for MleCollection<B> {
    fn default() -> Self {
        Self {
            mles_by_n_variables: vec![None; MAX_MLE_N_VARIABLES as usize + 1],
        }
    }
}

/// Dynamic dispatch for [`Mle`] types.
#[derive(Educe)]
#[educe(Debug, Clone)]
pub enum DynMle<B: Backend> {
    Base(Mle<B, BaseField>),
    Secure(Mle<B, SecureField>),
}

impl<B: Backend> DynMle<B> {
    fn n_variables(&self) -> usize {
        match self {
            DynMle::Base(mle) => mle.n_variables(),
            DynMle::Secure(mle) => mle.n_variables(),
        }
    }
}

impl<B: Backend> From<Mle<B, SecureField>> for DynMle<B> {
    fn from(mle: Mle<B, SecureField>) -> Self {
        DynMle::Secure(mle)
    }
}

impl<B: Backend> From<Mle<B, BaseField>> for DynMle<B> {
    fn from(mle: Mle<B, BaseField>) -> Self {
        DynMle::Base(mle)
    }
}

impl DynMle<SimdBackend> {
    fn into_secure_mle(self) -> Mle<SimdBackend, SecureField> {
        match self {
            Self::Base(mle) => {
                let evals = mle.into_evals();
                #[cfg(not(feature = "parallel"))]
                let evals = evals.into_secure_column();
                #[cfg(feature = "parallel")]
                let evals = {
                    let length = evals.len();
                    let data = evals
                        .data
                        .into_par_iter()
                        .with_min_len(COMBINE_CHUNK_SIZE)
                        .map(PackedSecureField::from)
                        .collect();
                    SecureColumn { data, length }
                };
                Mle::new(evals)
            }
            Self::Secure(mle) => mle,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter::repeat_n;

    use num_traits::Zero;
    use stwo::core::fields::m31::BaseField;
    use stwo::core::fields::qm31::SecureField;
    use stwo::core::fields::Field;
    use stwo::prover::backend::simd::m31::PackedBaseField;
    use stwo::prover::backend::simd::qm31::PackedSecureField;
    use stwo::prover::backend::simd::SimdBackend;
    use stwo::prover::backend::Column;
    use stwo::prover::lookups::mle::{Mle, MleOps};

    use crate::xor::gkr_lookups::accumulation::{combine, MleCollection};
    use crate::xor::gkr_lookups::test::mle_eval_at_point;

    #[test]
    fn random_linear_combine_by_n_variables() {
        const SMALL_N_VARS: usize = 4;
        const LARGE_N_VARS: usize = 6;
        let alpha = SecureField::from(10);
        let mut mle_collection = MleCollection::<SimdBackend>::default();
        mle_collection.push(const_mle(SMALL_N_VARS, BaseField::from(1)));
        mle_collection.push(const_mle(SMALL_N_VARS, SecureField::from(2)));
        mle_collection.push(const_mle(LARGE_N_VARS, BaseField::from(3)));
        mle_collection.push(const_mle(LARGE_N_VARS, SecureField::from(4)));
        mle_collection.push(const_mle(LARGE_N_VARS, SecureField::from(5)));
        let small_eval_point = [SecureField::zero(); SMALL_N_VARS];
        let large_eval_point = [SecureField::zero(); LARGE_N_VARS];

        let [small_mle, large_mle] = mle_collection
            .random_linear_combine_by_n_variables(alpha)
            .try_into()
            .unwrap();

        assert_eq!(small_mle.n_variables(), SMALL_N_VARS);
        assert_eq!(large_mle.n_variables(), LARGE_N_VARS);
        assert_eq!(
            mle_eval_at_point(&small_mle, &small_eval_point),
            SecureField::from(1) * alpha + SecureField::from(2)
        );
        assert_eq!(
            mle_eval_at_point(&large_mle, &large_eval_point),
            (SecureField::from(3) * alpha + SecureField::from(4)) * alpha + SecureField::from(5)
        );
    }

    #[test]
    fn combine_matches_packed_elementwise_formula() {
        const N_PACKED_VALUES: usize = (1 << 10) + 1;
        let alpha = PackedSecureField::broadcast(SecureField::from(7));
        let values = (0..N_PACKED_VALUES)
            .map(|i| PackedBaseField::broadcast(BaseField::from((i % 17) as u32)))
            .collect::<Vec<_>>();
        let mut actual = vec![PackedSecureField::broadcast(SecureField::from(3)); values.len()];
        let expected = actual
            .iter()
            .zip(&values)
            .map(|(&acc, &value)| acc + alpha * value)
            .collect::<Vec<_>>();

        combine(&mut actual, &values, alpha);

        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(actual.to_array(), expected.to_array());
        }
    }

    #[test]
    fn random_linear_combine_converts_base_initializer() {
        const N_VARIABLES: usize = 15;
        let alpha = SecureField::from(10);
        let mut collection = MleCollection::<SimdBackend>::default();
        collection.push(const_mle(N_VARIABLES, SecureField::from(2)));
        collection.push(const_mle(N_VARIABLES, BaseField::from(3)));

        let [combined] = collection
            .random_linear_combine_by_n_variables(alpha)
            .try_into()
            .unwrap();

        let expected = SecureField::from(2) * alpha + SecureField::from(3);
        assert!(combined
            .into_evals()
            .to_cpu()
            .into_iter()
            .all(|value| value == expected));
    }

    fn const_mle<B, F>(n_variables: usize, v: F) -> Mle<B, F>
    where
        B: MleOps<F>,
        F: Field,
    {
        Mle::new(repeat_n(v, 1 << n_variables).collect())
    }
}
