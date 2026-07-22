use core::ops::Sub;
#[cfg(not(feature = "parallel"))]
use std::iter::zip;
use std::ops::{Add, Mul};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
use super::gkr::PACKED_CHUNK_SIZE;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::prover::backend::simd::column::SecureColumn;
use crate::prover::backend::simd::m31::N_LANES;
use crate::prover::backend::simd::qm31::PackedSecureField;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::{Column, CpuBackend};
use crate::prover::lookups::mle::{Mle, MleOps};

impl MleOps<BaseField> for SimdBackend {
    fn fix_first_variable(
        mle: Mle<Self, BaseField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField> {
        let midpoint = mle.len() / 2;

        // Use CPU backend to avoid dealing with instances smaller than `PackedSecureField`.
        if midpoint < N_LANES {
            let cpu_mle = Mle::<CpuBackend, BaseField>::new(mle.to_cpu());
            let cpu_res = cpu_mle.fix_first_variable(assignment);
            return Mle::new(cpu_res.into_evals().into_iter().collect());
        }

        let packed_assignment = PackedSecureField::broadcast(assignment);
        let packed_midpoint = midpoint / N_LANES;
        let (evals_at_0x, evals_at_1x) = mle.data.split_at(packed_midpoint);

        #[cfg(not(feature = "parallel"))]
        let iter = zip(evals_at_0x, evals_at_1x);
        #[cfg(feature = "parallel")]
        let iter = evals_at_0x
            .par_iter()
            .zip(evals_at_1x)
            .with_min_len(PACKED_CHUNK_SIZE);

        let data = iter
            // MLE at points `({0, 1}, rev(bits(i)), v)` for all `v` in `{0, 1}^LOG_N_SIMD_LANES`.
            .map(|(&packed_eval_at_0iv, &packed_eval_at_1iv)| {
                fold_packed_mle_evals(packed_assignment, packed_eval_at_0iv, packed_eval_at_1iv)
            })
            .collect::<Vec<_>>();

        Mle::new(SecureColumn {
            data,
            length: midpoint,
        })
    }
}

impl MleOps<SecureField> for SimdBackend {
    fn fix_first_variable(
        mle: Mle<Self, SecureField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField> {
        let midpoint = mle.len() / 2;

        // Use CPU backend to avoid dealing with instances smaller than `PackedSecureField`.
        if midpoint < N_LANES {
            let cpu_mle = Mle::<CpuBackend, SecureField>::new(mle.to_cpu());
            let cpu_res = cpu_mle.fix_first_variable(assignment);
            return Mle::new(cpu_res.into_evals().into_iter().collect());
        }

        let packed_midpoint = midpoint / N_LANES;
        let packed_assignment = PackedSecureField::broadcast(assignment);
        let mut packed_evals = mle.into_evals().data;

        let (evals_at_0x, evals_at_1x) = packed_evals.split_at_mut(packed_midpoint);

        #[cfg(not(feature = "parallel"))]
        let iter = zip(evals_at_0x, &*evals_at_1x);
        #[cfg(feature = "parallel")]
        let iter = evals_at_0x
            .par_iter_mut()
            .zip(&*evals_at_1x)
            .with_min_len(PACKED_CHUNK_SIZE);

        iter.for_each(|(packed_eval_at_0iv, &packed_eval_at_1iv)| {
            // MLE at points `({0, 1}, rev(bits(i)), v)` for all `v` in `{0, 1}^LOG_N_SIMD_LANES`.
            *packed_eval_at_0iv =
                fold_packed_mle_evals(packed_assignment, *packed_eval_at_0iv, packed_eval_at_1iv);
        });

        packed_evals.truncate(packed_midpoint);

        let length = packed_evals.len() * N_LANES;
        let data = packed_evals;

        Mle::new(SecureColumn { data, length })
    }
}

/// Computes all `eq(0, assignment_i) * eval0_i + eq(1, assignment_i) * eval1_i`.
// TODO(andrew): Remove complex trait bounds once we have something like
// AbstractField/AbstractExtensionField traits.
fn fold_packed_mle_evals<
    PackedF: Sub<Output = PackedF> + Copy,
    PackedEF: Mul<PackedF, Output = PackedEF> + Add<PackedF, Output = PackedEF>,
>(
    assignment: PackedEF,
    eval0: PackedF,
    eval1: PackedF,
) -> PackedEF {
    assignment * (eval1 - eval0) + eval0
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::core::channel::Channel;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::test_utils::test_channel;
    use crate::prover::backend::simd::SimdBackend;
    use crate::prover::backend::{Column, CpuBackend};
    use crate::prover::lookups::mle::Mle;

    #[test]
    fn fix_first_variable_with_secure_field_mle_matches_cpu() {
        const N_VARIABLES: u32 = 8;
        let values = test_channel().draw_secure_felts(1 << N_VARIABLES);
        let mle_simd = Mle::<SimdBackend, SecureField>::new(values.iter().copied().collect());
        let mle_cpu = Mle::<CpuBackend, SecureField>::new(values);
        let random_assignment = SecureField::from_u32_unchecked(7, 12, 3, 2);
        let mle_fixed_cpu = mle_cpu.fix_first_variable(random_assignment);

        let mle_fixed_simd = mle_simd.fix_first_variable(random_assignment);

        assert_eq!(mle_fixed_simd.into_evals().to_cpu(), *mle_fixed_cpu)
    }

    #[test]
    fn fix_first_variable_with_base_field_mle_matches_cpu() {
        const N_VARIABLES: u32 = 8;
        let values = (0..1 << N_VARIABLES).map(BaseField::from).collect_vec();
        let mle_simd = Mle::<SimdBackend, BaseField>::new(values.iter().copied().collect());
        let mle_cpu = Mle::<CpuBackend, BaseField>::new(values);
        let random_assignment = SecureField::from_u32_unchecked(7, 12, 3, 2);
        let mle_fixed_cpu = mle_cpu.fix_first_variable(random_assignment);

        let mle_fixed_simd = mle_simd.fix_first_variable(random_assignment);

        assert_eq!(mle_fixed_simd.into_evals().to_cpu(), *mle_fixed_cpu)
    }
}
