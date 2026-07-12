use core::iter::zip;

use std_shims::{vec, Vec};

use super::accumulation::PointEvaluationAccumulator;
use super::Component;
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::TreeVec;
use crate::core::verifier::PREPROCESSED_TRACE_IDX;
use crate::core::ColumnVec;

pub struct Components<'a> {
    pub components: Vec<&'a dyn Component>,
    pub n_preprocessed_columns: usize,
}

impl Components<'_> {
    /// Log degree bound of the composition polynomial.
    ///
    /// Each component's quotient enters the composition lifted to the maximal trace log size
    /// `n_max`, so the composition log degree is `n_max` plus the maximal constraint log degree
    /// excess over trace log size across components (see [`Self::composition_log_split`]).
    /// When every component declares `bound = log_size + 1` this equals the maximal declared
    /// bound.
    pub fn composition_log_degree_bound(&self) -> u32 {
        let max_trace_log_size = self
            .components
            .iter()
            .map(|component| component_trace_log_size(*component))
            .max()
            .unwrap();
        max_trace_log_size + self.composition_log_split()
    }

    /// The number of times the composition polynomial is split in half before commitment, so
    /// that each of the `2^split` parts has log degree at most the maximal trace log size (the
    /// lifting boundary all sampled columns are lifted to).
    ///
    /// Equals the maximal `max_constraint_log_degree_bound() - trace_log_size` over components,
    /// and at least 1 (the composition always has one more log degree than the trace).
    pub fn composition_log_split(&self) -> u32 {
        self.components
            .iter()
            .map(|component| {
                component
                    .max_constraint_log_degree_bound()
                    .saturating_sub(component_trace_log_size(*component))
            })
            .max()
            .unwrap()
            .max(1)
    }

    pub fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
        max_log_degree_bound: u32,
        include_all_preprocessed_columns: bool,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        let mut mask_points = TreeVec::concat_cols(
            self.components
                .iter()
                .map(|component| component.mask_points(point, max_log_degree_bound)),
        );

        let preprocessed_mask_points = &mut mask_points[PREPROCESSED_TRACE_IDX];
        if include_all_preprocessed_columns {
            *preprocessed_mask_points = vec![vec![point]; self.n_preprocessed_columns];
        } else {
            *preprocessed_mask_points = vec![vec![]; self.n_preprocessed_columns];
            for component in &self.components {
                for idx in component.preprocessed_column_indices() {
                    preprocessed_mask_points[idx] = vec![point];
                }
            }
        }

        mask_points
    }

    pub fn eval_composition_polynomial_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask_values: &TreeVec<Vec<Vec<SecureField>>>,
        random_coeff: SecureField,
        max_log_degree_bound: u32,
    ) -> SecureField {
        let mut evaluation_accumulator = PointEvaluationAccumulator::new(random_coeff);
        for component in &self.components {
            component.evaluate_constraint_quotients_at_point(
                point,
                mask_values,
                &mut evaluation_accumulator,
                max_log_degree_bound,
            )
        }
        evaluation_accumulator.finalize()
    }

    pub fn n_composition_parts(&self) -> usize {
        1 << self.composition_log_split()
    }

    pub fn column_log_sizes(&self) -> TreeVec<ColumnVec<u32>> {
        let mut preprocessed_columns_trace_log_sizes = vec![0; self.n_preprocessed_columns];
        let mut visited_columns = vec![false; self.n_preprocessed_columns];

        let mut column_log_sizes = TreeVec::concat_cols(self.components.iter().map(|component| {
            let component_trace_log_sizes = component.trace_log_degree_bounds();

            for (column_index, &log_size) in zip(
                component.preprocessed_column_indices(),
                &component_trace_log_sizes[PREPROCESSED_TRACE_IDX],
            ) {
                let column_log_size = &mut preprocessed_columns_trace_log_sizes[column_index];
                if visited_columns[column_index] {
                    assert!(
                        *column_log_size == log_size,
                        "Preprocessed column size mismatch for column {column_index}"
                    );
                } else {
                    *column_log_size = log_size;
                    visited_columns[column_index] = true;
                }
            }

            component_trace_log_sizes
        }));

        assert!(
            visited_columns.iter().all(|&updated| updated),
            "Column size not set for all reprocessed columns"
        );

        column_log_sizes[PREPROCESSED_TRACE_IDX] = preprocessed_columns_trace_log_sizes;

        column_log_sizes
    }
}

/// The maximal log size over a component's trace columns.
fn component_trace_log_size(component: &dyn Component) -> u32 {
    component
        .trace_log_degree_bounds()
        .iter()
        .flatten()
        .copied()
        .max()
        .unwrap_or(0)
}
