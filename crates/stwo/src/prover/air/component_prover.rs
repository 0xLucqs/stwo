use std::any::type_name;

use dashmap::DashMap;
use itertools::Itertools;

use crate::core::air::{Component, Components};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::{TreeSubspan, TreeVec};
use crate::core::poly::circle::CircleDomain;
use crate::core::ColumnVec;
use crate::prover::air::accumulation::{DomainEvaluationAccumulator, EvaluationMode};
use crate::prover::backend::{Backend, Col, ColumnOps};
use crate::prover::memory::phase_memory_checkpoint;
use crate::prover::mempool::BaseColumnPool;
use crate::prover::poly::circle::{
    CircleCoefficients, CircleEvaluation, PolyOps, SecureCirclePoly,
};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::spill::{SharedSpillFile, SpillIndex};
use crate::prover::CirclePoint;

/// Type alias for the weights hash map used in barycentric eval_at_point.
pub type WeightsHashMap<B> = DashMap<(u32, CirclePoint<SecureField>), Col<B, SecureField>>;

#[derive(Debug, Clone, Default)]
pub struct TraceEvalAccessPattern {
    pub tree_spans: Vec<TreeSubspan>,
    pub preprocessed_columns: Vec<usize>,
}

pub trait ComponentProver<B: Backend>: Component {
    /// Evaluates the constraint quotients of the component on the evaluation domain.
    /// Accumulates quotients in `evaluation_accumulator`.
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &Trace<'_, B>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<B>,
    );

    /// Returns the trace columns this component needs as resident evaluations during
    /// point-wise quotient accumulation. `None` means the component does not expose a
    /// narrower access pattern and may require all trace evaluations.
    fn trace_eval_access_pattern(&self) -> Option<TraceEvalAccessPattern> {
        None
    }

    fn prover_name(&self) -> &'static str {
        type_name::<Self>()
    }
}

/// The set of polynomials that make up the trace.
pub struct Trace<'a, B: Backend> {
    /// Polynomials for each column.
    pub polys: TreeVec<ColumnVec<&'a Poly<B>>>,
}

/// A struct for representing a polynomial corresponding to a trace column.
/// A polynomial is defined by it's evaluations on a circle domain of size at least it's degree,
/// and optionally its coefficients in the FFT basis.
///
/// Coefficients may be stored in memory (`coeffs`), on disk (`spilled_coeffs`), or not at all.
/// When spilled, coefficients are loaded from a memory-mapped tempfile into a temporary
/// allocation on demand, then freed after use. This reduces peak RSS by ensuring only a few
/// polynomials' coefficients are resident at any time.
pub struct Poly<B: Backend> {
    pub coeffs: Option<CircleCoefficients<B>>,
    /// Disk-backed coefficient storage. When set, coefficients can be loaded on demand
    /// even if `coeffs` is `None`.
    pub spilled_coeffs: Option<SpilledPolyCoeffs>,
    pub eval_domain: CircleDomain,
    pub evals: Option<CircleEvaluation<B, BaseField, BitReversedOrder>>,
}

/// Reference to coefficient data stored in a memory-mapped spill file.
pub struct SpilledPolyCoeffs {
    pub spill_file: SharedSpillFile,
    pub spill_index: SpillIndex,
    pub log_size: u32,
}

impl<B: Backend> Poly<B> {
    pub fn new(
        coeffs: Option<CircleCoefficients<B>>,
        evals: CircleEvaluation<B, BaseField, BitReversedOrder>,
    ) -> Self {
        Self {
            coeffs,
            spilled_coeffs: None,
            eval_domain: evals.domain,
            evals: Some(evals),
        }
    }

    pub fn evals(&self) -> &CircleEvaluation<B, BaseField, BitReversedOrder> {
        self.evals
            .as_ref()
            .expect("evaluation buffer is not retained for this polynomial")
    }

    pub const fn log_size(&self) -> u32 {
        self.eval_domain.log_size()
    }

    pub fn take_evals(&mut self) -> CircleEvaluation<B, BaseField, BitReversedOrder> {
        self.evals
            .take()
            .expect("evaluation buffer is not retained for this polynomial")
    }

    /// Returns true if this polynomial has coefficients available (in memory or on disk).
    pub const fn has_coefficients(&self) -> bool {
        self.coeffs.is_some() || self.spilled_coeffs.is_some()
    }

    pub fn eval_at_point(
        &self,
        point: CirclePoint<SecureField>,
        weights_hash_map: Option<&WeightsHashMap<B>>,
    ) -> SecureField {
        if let Some(coeffs) = &self.coeffs {
            coeffs.eval_at_point(point)
        } else if let Some(spilled) = &self.spilled_coeffs {
            let coeffs = Self::load_spilled_coefficients(spilled);
            coeffs.eval_at_point(point)
        } else {
            self.evals().barycentric_eval_at_point(
                &weights_hash_map
                    .unwrap()
                    .get(&(self.log_size(), point))
                    .expect("weights should exist for all sampled points"),
            )
        }
    }

    pub fn get_evaluation_on_domain(
        &self,
        domain: CircleDomain,
        twiddles: &TwiddleTree<B>,
    ) -> CircleEvaluation<B, BaseField, BitReversedOrder> {
        if let Some(coeffs) = &self.coeffs {
            coeffs.evaluate_with_twiddles(domain, twiddles)
        } else if let Some(spilled) = &self.spilled_coeffs {
            let coeffs = Self::load_spilled_coefficients(spilled);
            coeffs.evaluate_with_twiddles(domain, twiddles)
        } else {
            panic!("The polynomial's coefficients are not stored");
        }
    }

    pub fn materialize_evaluation(
        &self,
        twiddles: &TwiddleTree<B>,
        base_column_pool: &BaseColumnPool<B>,
    ) -> CircleEvaluation<B, BaseField, BitReversedOrder>
    where
        B: PolyOps + ColumnOps<BaseField>,
    {
        let allocation_bytes = self
            .eval_domain
            .size()
            .saturating_mul(std::mem::size_of::<BaseField>());
        if allocation_bytes >= (4 << 20) {
            eprintln!(
                "ALLOC probe {}:{} fn=Poly::materialize_evaluation bytes={} logical_len={} element_type={} backing=heap_or_pool",
                file!(),
                line!(),
                allocation_bytes,
                self.eval_domain.size(),
                std::any::type_name::<BaseField>(),
            );
        }
        if let Some(coeffs) = &self.coeffs {
            let buffer = base_column_pool.take_or_alloc(self.log_size());
            self.materialize_evaluation_into_buffer(twiddles, buffer, coeffs)
        } else if let Some(spilled) = &self.spilled_coeffs {
            let coeffs = Self::load_spilled_coefficients(spilled);
            let buffer = base_column_pool.take_or_alloc(self.log_size());
            self.materialize_evaluation_into_buffer(twiddles, buffer, &coeffs)
        } else {
            panic!("low-memory polynomial recomputation requires retained coefficients");
        }
    }

    pub fn materialize_evaluation_with_buffer(
        &self,
        twiddles: &TwiddleTree<B>,
        buffer: Col<B, BaseField>,
    ) -> CircleEvaluation<B, BaseField, BitReversedOrder>
    where
        B: PolyOps + ColumnOps<BaseField>,
    {
        if let Some(coeffs) = &self.coeffs {
            self.materialize_evaluation_into_buffer(twiddles, buffer, coeffs)
        } else if let Some(spilled) = &self.spilled_coeffs {
            let coeffs = Self::load_spilled_coefficients(spilled);
            self.materialize_evaluation_into_buffer(twiddles, buffer, &coeffs)
        } else {
            panic!("low-memory polynomial recomputation requires retained coefficients");
        }
    }

    fn materialize_evaluation_into_buffer(
        &self,
        twiddles: &TwiddleTree<B>,
        buffer: Col<B, BaseField>,
        coeffs: &CircleCoefficients<B>,
    ) -> CircleEvaluation<B, BaseField, BitReversedOrder>
    where
        B: PolyOps + ColumnOps<BaseField>,
    {
        B::evaluate_into(coeffs, self.eval_domain, twiddles, buffer)
    }

    /// Loads coefficient data from the spill file into a fresh in-memory `CircleCoefficients`.
    ///
    /// The loaded data is a temporary allocation that should be used and then dropped
    /// to keep memory usage low.
    fn load_spilled_coefficients(spilled: &SpilledPolyCoeffs) -> CircleCoefficients<B>
    where
        B: ColumnOps<BaseField>,
    {
        let coeff_len = 1usize << spilled.log_size;
        if std::any::type_name::<B>()
            == std::any::type_name::<crate::prover::backend::simd::SimdBackend>()
        {
            let packed = spilled
                .spill_file
                .load_vec::<crate::prover::backend::simd::m31::PackedBaseField>(
                    spilled.spill_index,
                );
            let coeffs = CircleCoefficients::<crate::prover::backend::simd::SimdBackend>::new(
                crate::prover::backend::simd::column::BaseColumn {
                    data: packed,
                    length: coeff_len,
                },
            );
            let coeffs_ptr = &coeffs
                as *const CircleCoefficients<crate::prover::backend::simd::SimdBackend>
                as *const CircleCoefficients<B>;
            let result = unsafe { coeffs_ptr.read() };
            std::mem::forget(coeffs);
            result
        } else {
            let bytes = spilled.spill_file.get_bytes(spilled.spill_index);
            let base_fields: &[BaseField] = &bytemuck::cast_slice(bytes)[..coeff_len];
            let col = Col::<B, BaseField>::from_iter(base_fields.iter().copied());
            CircleCoefficients::new(col)
        }
    }
}

pub struct ComponentProvers<'a, B: Backend> {
    pub components: Vec<&'a dyn ComponentProver<B>>,
    pub n_preprocessed_columns: usize,
}

impl<B: Backend> ComponentProvers<'_, B> {
    pub fn components(&self) -> Components<'_> {
        Components {
            components: self
                .components
                .iter()
                .map(|c| *c as &dyn Component)
                .collect_vec(),
            n_preprocessed_columns: self.n_preprocessed_columns,
        }
    }
    pub fn compute_composition_polynomial(
        &self,
        random_coeff: SecureField,
        trace: &Trace<'_, B>,
        twiddles: &TwiddleTree<B>,
        log_blowup_factor: u32,
    ) -> SecureCirclePoly<B> {
        let total_constraints: usize = self.components.iter().map(|c| c.n_constraints()).sum();
        let components: Vec<&dyn Component> = self
            .components
            .iter()
            .map(|c| *c as &dyn Component)
            .collect();
        let evaluation_mode = EvaluationMode::infer(&components, log_blowup_factor);
        let mut accumulator = DomainEvaluationAccumulator::new(
            random_coeff,
            self.components().composition_log_degree_bound(),
            total_constraints,
            evaluation_mode,
        );
        for component in &self.components {
            component.evaluate_constraint_quotients_on_domain(trace, &mut accumulator)
        }
        phase_memory_checkpoint("composition:after_constraint_accumulation");
        accumulator.finalize(twiddles)
    }
}
