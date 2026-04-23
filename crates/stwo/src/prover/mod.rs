use thiserror::Error;
use tracing::{info, instrument, span, Level};

use crate::core::channel::{Channel, MerkleChannel};
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use crate::core::pcs::utils::get_lifting_log_size;
use crate::core::proof::{ExtendedStarkProof, StarkProof};
use crate::core::verifier::PREPROCESSED_TRACE_IDX;
use crate::prover::backend::BackendForChannel;
use crate::prover::memory::{phase_memory_checkpoint, PhaseMemorySession};

mod air;
pub use air::component_prover::{
    ComponentProver, ComponentProvers, Poly, Trace, TraceEvalAccessPattern,
};
pub use air::{AccumulationOps, ColumnAccumulator, DomainEvaluationAccumulator, EvaluationMode};
pub mod pcs;
pub use pcs::quotient_ops::QuotientOps;
pub use pcs::{CommitmentSchemeProver, CommitmentTreeProver, ProverMemoryMode, TreeBuilder};
pub mod backend;
pub mod channel;
pub mod fri;
pub mod line;
pub mod lookups;
pub mod memory;
pub mod mempool;
pub mod poly;
pub mod secure_column;
pub mod spill;
pub mod vcs;
pub mod vcs_lifted;

fn short_component_name<B: BackendForChannel<MC>, MC: MerkleChannel>(
    component: &dyn ComponentProver<B>,
) -> String {
    let type_name = component.prover_name();
    if let Some(start) = type_name.find("cairo_air::components::") {
        let tail = &type_name[start + "cairo_air::components::".len()..];
        let end = tail
            .find("::Eval")
            .or_else(|| tail.find("::FrameworkEval"))
            .unwrap_or(tail.len());
        return tail[..end]
            .rsplit("::")
            .next()
            .unwrap_or("component")
            .to_string();
    }

    let without_generics = type_name.split('<').next().unwrap_or(type_name);
    without_generics
        .rsplit("::")
        .next()
        .unwrap_or("component")
        .to_string()
}

pub fn prove<B: BackendForChannel<MC>, MC: MerkleChannel>(
    components: &[&dyn ComponentProver<B>],
    channel: &mut MC::C,
    commitment_scheme: CommitmentSchemeProver<'_, B, MC>,
) -> Result<StarkProof<MC::H>, ProvingError>
where
    // Propagated from `prove_ex`. All MCs used with the prover satisfy this.
    crate::prover::backend::simd::SimdBackend: BackendForChannel<MC>,
{
    Ok(prove_ex(components, channel, commitment_scheme, false)?.proof)
}

#[instrument(skip_all)]
pub fn prove_ex<B: BackendForChannel<MC>, MC: MerkleChannel>(
    components: &[&dyn ComponentProver<B>],
    channel: &mut MC::C,
    mut commitment_scheme: CommitmentSchemeProver<'_, B, MC>,
    include_all_preprocessed_columns: bool,
) -> Result<ExtendedStarkProof<MC::H>, ProvingError>
where
    // Propagated from `materialize_access_pattern_evaluations` and
    // `prove_values`, which call into the iOS-specialised SimdBackend
    // materialize path. In practice all MCs used with the prover satisfy
    // this bound.
    crate::prover::backend::simd::SimdBackend: BackendForChannel<MC>,
{
    let _memory_session = PhaseMemorySession::start("prove_ex");
    phase_memory_checkpoint("prove_ex:start");
    let n_preprocessed_columns = commitment_scheme.trees[PREPROCESSED_TRACE_IDX]
        .polynomials
        .len();
    let component_provers = ComponentProvers {
        components: components.to_vec(),
        n_preprocessed_columns,
    };

    // Evaluate and commit on composition polynomial.
    let random_coeff = channel.draw_secure_felt();

    let span = span!(Level::INFO, "Composition", class = "Composition").entered();
    let span1 = span!(
        Level::INFO,
        "Generation",
        class = "CompositionPolynomialGeneration"
    )
    .entered();

    let composition_poly = if commitment_scheme.memory_mode.rematerializes_evaluations() {
        commitment_scheme.release_recomputable_evaluations();
        phase_memory_checkpoint("prove_ex:after_initial_low_memory_trace_release");

        // Spill trace coefficients to disk-backed mmap before composition generation.
        // The composition loop materializes evaluations one component at a time from these
        // coefficients; keeping them as anonymous heap while the composition accumulator also
        // grows sets the peak footprint unnecessarily high.
        if let Err(e) = commitment_scheme.spill_coefficients() {
            tracing::warn!("Failed to spill coefficients to disk: {e}. Continuing with in-memory coefficients.");
        }
        phase_memory_checkpoint("prove_ex:after_early_coefficient_spill");

        let total_constraints: usize = component_provers
            .components
            .iter()
            .map(|component| component.n_constraints())
            .sum();
        let components = component_provers
            .components
            .iter()
            .map(|component| *component as &dyn crate::core::air::Component)
            .collect::<Vec<_>>();
        let evaluation_mode = EvaluationMode::infer(
            &components,
            commitment_scheme.config.fri_config.log_blowup_factor,
        );
        let mut accumulator = DomainEvaluationAccumulator::new(
            random_coeff,
            component_provers
                .components()
                .composition_log_degree_bound(),
            total_constraints,
            evaluation_mode,
        );

        for (component_index, component) in component_provers.components.iter().enumerate() {
            let component_name = short_component_name::<B, MC>(*component);
            let access_pattern = component.trace_eval_access_pattern();
            phase_memory_checkpoint(format!(
                "composition:component:{component_index}:{component_name}:before_materialize"
            ));
            commitment_scheme.materialize_access_pattern_evaluations(access_pattern.as_ref());
            phase_memory_checkpoint(format!(
                "composition:component:{component_index}:{component_name}:after_materialize"
            ));
            let trace = commitment_scheme.trace();
            component.evaluate_constraint_quotients_on_domain(&trace, &mut accumulator);
            phase_memory_checkpoint(format!(
                "composition:component:{component_index}:{component_name}:after_evaluate"
            ));
            commitment_scheme.drop_access_pattern_evaluations(access_pattern.as_ref());
            phase_memory_checkpoint(format!(
                "composition:component:{component_index}:{component_name}:after_drop"
            ));
        }
        phase_memory_checkpoint("composition:after_constraint_accumulation");
        accumulator.finalize(commitment_scheme.twiddles)
    } else {
        let trace = commitment_scheme.trace();
        component_provers.compute_composition_polynomial(
            random_coeff,
            &trace,
            commitment_scheme.twiddles,
            commitment_scheme.config.fri_config.log_blowup_factor,
        )
    };
    span1.exit();
    phase_memory_checkpoint("prove_ex:after_composition_generation");

    if commitment_scheme.memory_mode.rematerializes_evaluations() {
        commitment_scheme.release_recomputable_evaluations();
        phase_memory_checkpoint("prove_ex:after_low_memory_trace_release");
    }

    // Commit on the Composition Polynomial by splitting its coeffs to two polynomialsof degree
    // half the size of the original polynomial, and commit on each half separately.
    let mut tree_builder = commitment_scheme.tree_builder();
    let (left_comp_poly_half, right_comp_poly_half) = composition_poly.split_at_mid();

    tree_builder.extend_polys(left_comp_poly_half.into_coordinate_polys());
    tree_builder.extend_polys(right_comp_poly_half.into_coordinate_polys());
    tree_builder.commit(channel);
    span.exit();
    phase_memory_checkpoint("prove_ex:after_composition_commit");

    // Draw OODS point.
    let oods_point = CirclePoint::<SecureField>::get_random_point(channel);

    let split_composition_log_size = commitment_scheme.trees.last().unwrap().commitment.height();

    // If `self.config.lifting_log_size` is None, the lifting size is the length of the split
    // composition polynomials' domain.
    let lifting_log_size =
        get_lifting_log_size(&commitment_scheme.config, split_composition_log_size);
    if include_all_preprocessed_columns {
        // If all the preprocessed columns are included, the lifting log size must be greater than
        // or equal to the preprocessed log size.
        let preprocessed_log_size = commitment_scheme.trees[PREPROCESSED_TRACE_IDX]
            .commitment
            .height();
        assert!(lifting_log_size >= preprocessed_log_size);
    }
    let max_log_degree_bound =
        lifting_log_size - commitment_scheme.config.fri_config.log_blowup_factor;

    // Get mask sample points relative to oods point.
    let mut sample_points = component_provers.components().mask_points(
        oods_point,
        max_log_degree_bound,
        include_all_preprocessed_columns,
    );

    // Add the composition polynomial mask points.
    sample_points.push(vec![vec![oods_point]; 2 * SECURE_EXTENSION_DEGREE]);

    // Prove the trace and composition OODS values, and retrieve them.
    let commitment_scheme_proof = commitment_scheme.prove_values(sample_points, channel);
    phase_memory_checkpoint("prove_ex:after_pcs_prove_values");
    let proof = StarkProof(commitment_scheme_proof.proof);
    info!(proof_size_estimate = proof.size_estimate());

    // Evaluate composition polynomial at OODS point and check that it matches the trace OODS
    // values. This is a sanity check.
    if proof
        .extract_composition_oods_eval(oods_point, max_log_degree_bound)
        .unwrap()
        != component_provers
            .components()
            .eval_composition_polynomial_at_point(
                oods_point,
                &proof.sampled_values,
                random_coeff,
                max_log_degree_bound,
            )
    {
        return Err(ProvingError::ConstraintsNotSatisfied);
    }

    Ok(ExtendedStarkProof {
        proof,
        aux: commitment_scheme_proof.aux,
    })
}

#[derive(Clone, Copy, Debug, Error)]
pub enum ProvingError {
    #[error("Constraints not satisfied.")]
    ConstraintsNotSatisfied,
}
