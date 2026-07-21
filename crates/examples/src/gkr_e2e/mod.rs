#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::fmt::{Display, Formatter};
    use std::time::{Duration, Instant};

    use num_traits::{One, Zero};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    #[cfg(feature = "parallel")]
    use rayon::prelude::*;
    use stwo::core::air::accumulation::PointEvaluationAccumulator;
    use stwo::core::air::Components;
    use stwo::core::channel::{Blake2sChannel, Channel};
    use stwo::core::circle::CirclePoint;
    use stwo::core::fields::m31::{BaseField, P as M31_MODULUS};
    use stwo::core::fields::qm31::SecureField;
    use stwo::core::fields::{ExtensionOf, Field};
    use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig, TreeVec};
    use stwo::core::poly::circle::CanonicCoset;
    use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
    use stwo::core::verifier::verify;
    use stwo::core::{ColumnVec, Fraction};
    use stwo::prover::backend::simd::column::{BaseColumn, SecureColumn};
    use stwo::prover::backend::simd::qm31::PackedSecureField;
    use stwo::prover::backend::simd::SimdBackend;
    use stwo::prover::backend::Column;
    use stwo::prover::lookups::gkr_prover::{prove_batch as prove_gkr_batch, Layer};
    use stwo::prover::lookups::gkr_verifier::{
        partially_verify_batch, Gate, GkrArtifact, GkrBatchProof,
    };
    use stwo::prover::lookups::mle::{Mle, MleOps};
    use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
    use stwo::prover::poly::twiddles::TwiddleTree;
    use stwo::prover::poly::BitReversedOrder;
    use stwo::prover::{prove, CommitmentSchemeProver, ComponentProver};
    use stwo_constraint_framework::mle_eval::{
        build_trace as build_mle_eval_trace, MleCoeffColumnOracle, MleEvalProverComponent,
        MleEvalVerifierComponent,
    };
    use stwo_constraint_framework::{
        relation, EvalAtRow, FrameworkComponent, FrameworkEval, LogupTraceGenerator,
        PointEvaluator, Relation, RelationEntry, TraceLocationAllocator,
    };

    use crate::xor::gkr_lookups::accumulation::MleCollection;

    const AUX_TRACE_IDX: usize = 2;
    const LOG_EXPAND: u32 = 1;
    #[cfg(feature = "parallel")]
    const RELATION_COLUMN_CHUNK_SIZE: usize = 1 << 10;

    relation!(LookupRelation, 1);

    type HarnessResult<T> = Result<T, Box<dyn Error>>;
    type BaseComponent = FrameworkComponent<BaseColumnsEval>;

    #[derive(Debug)]
    struct HarnessError(&'static str);

    impl Display for HarnessError {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            f.write_str(self.0)
        }
    }

    impl Error for HarnessError {}

    struct LookupInput {
        log_n: u32,
        use_columns: Vec<BaseColumn>,
        table: BaseColumn,
        multiplicities: BaseColumn,
    }

    impl LookupInput {
        fn generate(log_n: u32, n_use_columns: usize, corrupt_multiplicity: bool) -> Self {
            assert!(n_use_columns > 0);
            assert!(log_n < usize::BITS);
            let n_rows = 1usize << log_n;
            assert!((n_rows as u128) < M31_MODULUS as u128);
            assert!((n_use_columns as u128) * (n_rows as u128) < M31_MODULUS as u128);
            let mut rng = SmallRng::seed_from_u64(0);
            let mut counts = vec![0u32; n_rows];
            let use_columns = (0..n_use_columns)
                .map(|_| {
                    (0..n_rows)
                        .map(|_| {
                            let value = rng.gen_range(0..n_rows);
                            counts[value] += 1;
                            BaseField::from(value as u32)
                        })
                        .collect()
                })
                .collect();
            if corrupt_multiplicity {
                counts[0] += 1;
            }

            Self {
                log_n,
                use_columns,
                table: (0..n_rows)
                    .map(|value| BaseField::from(value as u32))
                    .collect(),
                multiplicities: counts.into_iter().map(BaseField::from).collect(),
            }
        }

        fn n_use_columns(&self) -> usize {
            self.use_columns.len()
        }

        fn n_base_columns(&self) -> usize {
            self.n_use_columns() + 2
        }

        fn base_trace(
            &self,
        ) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
            let domain = CanonicCoset::new(self.log_n).circle_domain();
            self.use_columns
                .iter()
                .chain([&self.table, &self.multiplicities])
                .cloned()
                .map(|column| CircleEvaluation::new(domain, column))
                .collect()
        }
    }

    #[derive(Clone)]
    struct LookupEval {
        log_n: u32,
        n_use_columns: usize,
        relation: LookupRelation,
    }

    impl FrameworkEval for LookupEval {
        fn log_size(&self) -> u32 {
            self.log_n
        }

        fn max_constraint_log_degree_bound(&self) -> u32 {
            self.log_n + 1
        }

        fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
            let uses = (0..self.n_use_columns)
                .map(|_| eval.next_trace_mask())
                .collect::<Vec<_>>();
            let table = eval.next_trace_mask();
            let multiplicities = eval.next_trace_mask();

            for value in &uses {
                eval.add_to_relation(RelationEntry::unit(
                    &self.relation,
                    std::slice::from_ref(value),
                ));
            }
            eval.add_to_relation(RelationEntry::base(
                &self.relation,
                -multiplicities,
                &[table],
            ));
            eval.finalize_logup_in_pairs();
            eval
        }
    }

    #[derive(Clone)]
    struct BaseColumnsEval {
        log_n: u32,
        n_columns: usize,
    }

    impl FrameworkEval for BaseColumnsEval {
        fn log_size(&self) -> u32 {
            self.log_n
        }

        fn max_constraint_log_degree_bound(&self) -> u32 {
            self.log_n
        }

        fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
            for _ in 0..self.n_columns {
                let _ = eval.next_trace_mask();
            }
            eval
        }
    }

    struct CombinedMleOracle<'a> {
        component: &'a BaseComponent,
        relation_shift: SecureField,
        alpha: SecureField,
        n_use_columns: usize,
    }

    impl MleCoeffColumnOracle for CombinedMleOracle<'_> {
        fn evaluate_at_point(
            &self,
            _point: CirclePoint<SecureField>,
            mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        ) -> SecureField {
            let mut accumulator = PointEvaluationAccumulator::new(SecureField::one());
            let mut eval = PointEvaluator::new(
                mask.sub_tree(self.component.trace_locations()),
                &mut accumulator,
                SecureField::one(),
                self.component.log_size(),
                SecureField::zero(),
            );

            let mut result = SecureField::zero();
            for _ in 0..self.n_use_columns {
                result = result * self.alpha + eval.next_trace_mask() + self.relation_shift;
            }
            let table = eval.next_trace_mask();
            let multiplicities = eval.next_trace_mask();
            result = result * self.alpha + multiplicities;
            result * self.alpha + table + self.relation_shift
        }
    }

    fn precompute_twiddles(log_n: u32, config: PcsConfig) -> TwiddleTree<SimdBackend> {
        SimdBackend::precompute_twiddles(
            CanonicCoset::new(log_n + LOG_EXPAND + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        )
    }

    fn print_phase(path: char, input: &LookupInput, phase: &str, elapsed: Duration, emit: bool) {
        if emit {
            println!(
                "E2E path={path} log_n={} l={} phase={phase} ms={:.3}",
                input.log_n,
                input.n_use_columns(),
                elapsed.as_secs_f64() * 1000.0
            );
        }
    }

    fn print_size(
        path: char,
        input: &LookupInput,
        proof_bytes: usize,
        gkr_felts: usize,
        emit: bool,
    ) {
        if emit {
            println!(
                "E2E path={path} log_n={} l={} proof_bytes={proof_bytes} gkr_felts={gkr_felts}",
                input.log_n,
                input.n_use_columns()
            );
        }
    }

    fn interaction_fraction(
        input: &LookupInput,
        relation: &LookupRelation,
        fraction: usize,
        vec_row: usize,
    ) -> (PackedSecureField, PackedSecureField) {
        if fraction < input.n_use_columns() {
            let denominator = relation.combine(&[input.use_columns[fraction].data[vec_row]]);
            (PackedSecureField::one(), denominator)
        } else {
            let denominator = relation.combine(&[input.table.data[vec_row]]);
            (
                PackedSecureField::from(-input.multiplicities.data[vec_row]),
                denominator,
            )
        }
    }

    fn generate_interaction_trace(
        input: &LookupInput,
        relation: &LookupRelation,
    ) -> (
        ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
        SecureField,
    ) {
        let n_fractions = input.n_use_columns() + 1;
        let mut generator = LogupTraceGenerator::new(input.log_n);
        for first in (0..n_fractions).step_by(2) {
            generator.col_from_fn(|vec_row| {
                let (first_num, first_den) = interaction_fraction(input, relation, first, vec_row);
                if first + 1 == n_fractions {
                    return (first_num, first_den);
                }
                let (second_num, second_den) =
                    interaction_fraction(input, relation, first + 1, vec_row);
                (
                    first_num * second_den + second_num * first_den,
                    first_den * second_den,
                )
            });
        }
        generator.finalize_last()
    }

    fn ensure_zero_lookup_sum(
        path: &'static str,
        outputs: &[Vec<SecureField>],
    ) -> HarnessResult<()> {
        let Some((multiplicities, uses)) = outputs.split_last() else {
            return Err(HarnessError("GKR returned no output claims").into());
        };
        if uses.iter().any(|claim| claim.len() != 2) || multiplicities.len() != 2 {
            return Err(HarnessError("GKR returned malformed output claims").into());
        }
        let use_sum = uses
            .iter()
            .map(|claim| Fraction::new(claim[0], claim[1]))
            .sum::<Fraction<SecureField, SecureField>>();
        let table_sum = Fraction::new(multiplicities[0], multiplicities[1]);
        let difference_numerator =
            use_sum.numerator * table_sum.denominator - table_sum.numerator * use_sum.denominator;
        if !difference_numerator.is_zero() {
            return Err(HarnessError(path).into());
        }
        Ok(())
    }

    fn collect_mle_claims(
        artifact: &GkrArtifact,
        n_use_columns: usize,
        log_n: u32,
    ) -> HarnessResult<Vec<SecureField>> {
        if artifact.n_variables_by_instance != vec![log_n as usize; n_use_columns + 1] {
            return Err(HarnessError("GKR artifact shape does not match the lookup batch").into());
        }
        if artifact.ood_point.len() != log_n as usize {
            return Err(HarnessError("GKR OOD point has the wrong dimension").into());
        }
        if artifact.claims_to_verify_by_instance.len() != n_use_columns + 1
            || artifact
                .claims_to_verify_by_instance
                .iter()
                .any(|claim| claim.len() != 2)
        {
            return Err(HarnessError("GKR MLE claims have the wrong shape").into());
        }

        let mut claims = Vec::with_capacity(n_use_columns + 2);
        for claim in &artifact.claims_to_verify_by_instance[..n_use_columns] {
            if claim.len() != 2 || claim[0] != SecureField::one() {
                return Err(HarnessError("GKR singles claim is malformed").into());
            }
            claims.push(claim[1]);
        }
        let multiplicities = &artifact.claims_to_verify_by_instance[n_use_columns];
        if multiplicities.len() != 2 {
            return Err(HarnessError("GKR multiplicities claim is malformed").into());
        }
        claims.extend_from_slice(multiplicities);
        Ok(claims)
    }

    fn preflight_gkr_proof(
        proof: &GkrBatchProof,
        n_use_columns: usize,
        log_n: u32,
    ) -> HarnessResult<()> {
        let n_instances = n_use_columns + 1;
        if proof.layer_masks_by_instance.len() != n_instances
            || proof.output_claims_by_instance.len() != n_instances
        {
            return Err(HarnessError("GKR proof has the wrong instance count").into());
        }
        if proof
            .layer_masks_by_instance
            .iter()
            .any(|masks| masks.len() != log_n as usize)
        {
            return Err(HarnessError("GKR proof has the wrong layer-mask depth").into());
        }
        if proof
            .output_claims_by_instance
            .iter()
            .any(|claim| claim.len() != 2)
        {
            return Err(HarnessError("GKR proof output claims have the wrong shape").into());
        }
        Ok(())
    }

    fn combine_claims(claims: &[SecureField], alpha: SecureField) -> SecureField {
        claims
            .iter()
            .fold(SecureField::zero(), |acc, claim| acc * alpha + *claim)
    }

    fn relation_column(z: SecureField, column: &BaseColumn) -> Mle<SimdBackend, SecureField> {
        let packed_z = PackedSecureField::broadcast(z);
        #[cfg(not(feature = "parallel"))]
        let values = column
            .data
            .iter()
            .map(|&value| PackedSecureField::from(value) - packed_z)
            .collect::<SecureColumn>();
        #[cfg(feature = "parallel")]
        let values = {
            let mut data = vec![PackedSecureField::zero(); column.data.len()];
            data.par_chunks_mut(RELATION_COLUMN_CHUNK_SIZE)
                .zip(column.data.par_chunks(RELATION_COLUMN_CHUNK_SIZE))
                .for_each(|(dst, src)| {
                    std::iter::zip(dst, src).for_each(|(dst, &value)| {
                        *dst = PackedSecureField::from(value) - packed_z;
                    });
                });
            SecureColumn {
                data,
                length: column.len(),
            }
        };
        Mle::new(values)
    }

    fn relation_columns(z: SecureField, input: &LookupInput) -> Vec<Mle<SimdBackend, SecureField>> {
        #[cfg(not(feature = "parallel"))]
        let columns = input
            .use_columns
            .iter()
            .chain(std::iter::once(&input.table));
        #[cfg(feature = "parallel")]
        let columns = input
            .use_columns
            .par_iter()
            .chain(rayon::iter::once(&input.table));
        columns.map(|column| relation_column(z, column)).collect()
    }

    fn mle_eval_at_point<B, F>(mle: &Mle<B, F>, point: &[SecureField]) -> SecureField
    where
        F: Field,
        SecureField: ExtensionOf<F>,
        B: MleOps<F>,
    {
        fn evaluate(evals: &[SecureField], point: &[SecureField]) -> SecureField {
            match point {
                [] => evals[0],
                &[coordinate, ref rest @ ..] => {
                    let (lhs, rhs) = evals.split_at(evals.len() / 2);
                    let lhs = evaluate(lhs, rest);
                    let rhs = evaluate(rhs, rest);
                    coordinate * (rhs - lhs) + lhs
                }
            }
        }

        let evals = mle
            .clone()
            .into_evals()
            .to_cpu()
            .into_iter()
            .map(Into::into)
            .collect::<Vec<_>>();
        evaluate(&evals, point)
    }

    fn gkr_felt_count(proof: &GkrBatchProof) -> usize {
        let round_polys = proof
            .sumcheck_proofs
            .iter()
            .flat_map(|proof| &proof.round_polys)
            .map(|poly| poly.len())
            .sum::<usize>();
        let masks = proof
            .layer_masks_by_instance
            .iter()
            .flatten()
            .map(|mask| mask.columns().len() * 2)
            .sum::<usize>();
        let outputs = proof
            .output_claims_by_instance
            .iter()
            .map(Vec::len)
            .sum::<usize>();
        round_polys + masks + outputs
    }

    fn run_path_a(
        input: &LookupInput,
        config: PcsConfig,
        twiddles: &TwiddleTree<SimdBackend>,
        emit: bool,
    ) -> HarnessResult<()> {
        let channel = &mut Blake2sChannel::default();
        config.mix_into(channel);
        let mut commitment_scheme =
            CommitmentSchemeProver::<_, Blake2sMerkleChannel>::new(config, twiddles);
        let total = Instant::now();

        let phase = Instant::now();
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(vec![]);
        tree_builder.commit(channel);
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(input.base_trace());
        tree_builder.commit(channel);
        print_phase('a', input, "base_commit", phase.elapsed(), emit);

        let relation = LookupRelation::draw(channel);
        let phase = Instant::now();
        let (interaction_trace, claimed_sum) = generate_interaction_trace(input, &relation);
        print_phase('a', input, "interaction_gen", phase.elapsed(), emit);
        if !claimed_sum.is_zero() {
            return Err(HarnessError("Path A lookup sum is nonzero").into());
        }

        let phase = Instant::now();
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(interaction_trace);
        tree_builder.commit(channel);
        print_phase('a', input, "interaction_commit", phase.elapsed(), emit);

        let allocator = &mut TraceLocationAllocator::default();
        let component = FrameworkComponent::new(
            allocator,
            LookupEval {
                log_n: input.log_n,
                n_use_columns: input.n_use_columns(),
                relation,
            },
            claimed_sum,
        );
        let phase = Instant::now();
        let proof = prove(
            &[&component as &dyn ComponentProver<SimdBackend>],
            channel,
            commitment_scheme,
        )?;
        print_phase('a', input, "stark_prove", phase.elapsed(), emit);
        print_phase('a', input, "total", total.elapsed(), emit);
        print_size('a', input, proof.size_estimate(), 0, emit);

        let phase = Instant::now();
        let verifier_channel = &mut Blake2sChannel::default();
        config.mix_into(verifier_channel);
        let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
        commitment_scheme.commit(proof.commitments[0], &[], verifier_channel);
        commitment_scheme.commit(
            proof.commitments[1],
            &vec![input.log_n; input.n_base_columns()],
            verifier_channel,
        );
        let relation = LookupRelation::draw(verifier_channel);
        let allocator = &mut TraceLocationAllocator::default();
        let component = FrameworkComponent::new(
            allocator,
            LookupEval {
                log_n: input.log_n,
                n_use_columns: input.n_use_columns(),
                relation,
            },
            SecureField::zero(),
        );
        let components = Components {
            components: vec![&component],
            n_preprocessed_columns: 0,
        };
        let log_sizes = components.column_log_sizes();
        commitment_scheme.commit(
            proof.commitments[2],
            &log_sizes[AUX_TRACE_IDX],
            verifier_channel,
        );
        verify(
            &components.components,
            verifier_channel,
            commitment_scheme,
            proof,
        )?;
        print_phase('a', input, "verify", phase.elapsed(), emit);
        Ok(())
    }

    fn run_path_b(
        input: &LookupInput,
        config: PcsConfig,
        twiddles: &TwiddleTree<SimdBackend>,
        emit: bool,
    ) -> HarnessResult<()> {
        let channel = &mut Blake2sChannel::default();
        config.mix_into(channel);
        let mut commitment_scheme =
            CommitmentSchemeProver::<_, Blake2sMerkleChannel>::new(config, twiddles);
        let total = Instant::now();

        let phase = Instant::now();
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(vec![]);
        tree_builder.commit(channel);
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(input.base_trace());
        tree_builder.commit(channel);
        print_phase('b', input, "base_commit", phase.elapsed(), emit);

        let relation = LookupRelation::draw(channel);
        let relation_shift: SecureField = relation.combine(&[BaseField::zero()]);
        let phase = Instant::now();
        let mut denominator_mles = relation_columns(-relation_shift, input);
        let table_mle = denominator_mles
            .pop()
            .expect("lookup input always has a table denominator");
        let use_mles = denominator_mles;
        let multiplicities_mle = Mle::<SimdBackend, BaseField>::new(input.multiplicities.clone());
        let mut layers = use_mles
            .iter()
            .cloned()
            .map(|denominators| Layer::LogUpSingles { denominators })
            .collect::<Vec<_>>();
        layers.push(Layer::LogUpMultiplicities {
            numerators: multiplicities_mle.clone(),
            denominators: table_mle.clone(),
        });
        print_phase('b', input, "gkr_layers", phase.elapsed(), emit);

        let phase = Instant::now();
        let (gkr_proof, artifact) = prove_gkr_batch(channel, layers);
        print_phase('b', input, "gkr_prove", phase.elapsed(), emit);
        ensure_zero_lookup_sum(
            "Path B lookup sum is nonzero",
            &gkr_proof.output_claims_by_instance,
        )?;

        let claims = collect_mle_claims(&artifact, input.n_use_columns(), input.log_n)?;
        let acc_alpha = channel.draw_secure_felt();
        let phase = Instant::now();
        let mut collection = MleCollection::<SimdBackend>::default();
        for mle in &use_mles {
            collection.push(mle.clone());
        }
        collection.push(multiplicities_mle);
        collection.push(table_mle);
        let [combined_mle] = collection
            .random_linear_combine_by_n_variables(acc_alpha)
            .try_into()
            .map_err(|_| HarnessError("MLE combination produced an unexpected number of groups"))?;
        let combined_claim = combine_claims(&claims, acc_alpha);
        if mle_eval_at_point(&combined_mle, &artifact.ood_point) != combined_claim {
            return Err(HarnessError("combined MLE claim does not match the GKR artifact").into());
        }
        print_phase('b', input, "mle_combine", phase.elapsed(), emit);

        let phase = Instant::now();
        let mle_trace = build_mle_eval_trace(&combined_mle, &artifact.ood_point, combined_claim);
        print_phase('b', input, "mle_trace", phase.elapsed(), emit);
        let phase = Instant::now();
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(mle_trace);
        tree_builder.commit(channel);
        print_phase('b', input, "mle_trace_commit", phase.elapsed(), emit);

        let allocator = &mut TraceLocationAllocator::default();
        let base_component = FrameworkComponent::new(
            allocator,
            BaseColumnsEval {
                log_n: input.log_n,
                n_columns: input.n_base_columns(),
            },
            SecureField::zero(),
        );
        let oracle = CombinedMleOracle {
            component: &base_component,
            relation_shift,
            alpha: acc_alpha,
            n_use_columns: input.n_use_columns(),
        };
        let mle_eval_component = MleEvalProverComponent::generate(
            allocator,
            oracle,
            &artifact.ood_point,
            combined_mle,
            combined_claim,
            twiddles,
            AUX_TRACE_IDX,
        );
        let phase = Instant::now();
        let proof = prove(
            &[
                &base_component as &dyn ComponentProver<SimdBackend>,
                &mle_eval_component,
            ],
            channel,
            commitment_scheme,
        )?;
        print_phase('b', input, "stark_prove", phase.elapsed(), emit);
        print_phase('b', input, "total", total.elapsed(), emit);
        print_size(
            'b',
            input,
            proof.size_estimate(),
            gkr_felt_count(&gkr_proof),
            emit,
        );

        let phase = Instant::now();
        verify_path_b(input, config, proof, &gkr_proof)?;
        print_phase('b', input, "verify", phase.elapsed(), emit);
        Ok(())
    }

    fn verify_path_b(
        input: &LookupInput,
        config: PcsConfig,
        proof: stwo::core::proof::StarkProof<
            stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher,
        >,
        gkr_proof: &GkrBatchProof,
    ) -> HarnessResult<()> {
        let channel = &mut Blake2sChannel::default();
        config.mix_into(channel);
        let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
        commitment_scheme.commit(proof.commitments[0], &[], channel);
        commitment_scheme.commit(
            proof.commitments[1],
            &vec![input.log_n; input.n_base_columns()],
            channel,
        );
        let relation = LookupRelation::draw(channel);
        let relation_shift: SecureField = relation.combine(&[BaseField::zero()]);
        preflight_gkr_proof(gkr_proof, input.n_use_columns(), input.log_n)?;
        let artifact = partially_verify_batch(
            vec![Gate::LogUp; input.n_use_columns() + 1],
            gkr_proof,
            channel,
        )?;
        ensure_zero_lookup_sum(
            "Path B verifier lookup sum is nonzero",
            &gkr_proof.output_claims_by_instance,
        )?;
        let claims = collect_mle_claims(&artifact, input.n_use_columns(), input.log_n)?;
        let acc_alpha = channel.draw_secure_felt();
        let combined_claim = combine_claims(&claims, acc_alpha);

        let allocator = &mut TraceLocationAllocator::default();
        let base_component = FrameworkComponent::new(
            allocator,
            BaseColumnsEval {
                log_n: input.log_n,
                n_columns: input.n_base_columns(),
            },
            SecureField::zero(),
        );
        let oracle = CombinedMleOracle {
            component: &base_component,
            relation_shift,
            alpha: acc_alpha,
            n_use_columns: input.n_use_columns(),
        };
        let mle_eval_component = MleEvalVerifierComponent::new(
            allocator,
            oracle,
            &artifact.ood_point,
            combined_claim,
            AUX_TRACE_IDX,
        );
        let components = Components {
            components: vec![&base_component, &mle_eval_component],
            n_preprocessed_columns: 0,
        };
        let log_sizes = components.column_log_sizes();
        commitment_scheme.commit(proof.commitments[2], &log_sizes[AUX_TRACE_IDX], channel);
        verify(&components.components, channel, commitment_scheme, proof)?;
        Ok(())
    }

    #[test]
    fn gkr_e2e_roundtrip_small() {
        let input = LookupInput::generate(8, 2, false);
        let config = PcsConfig::default();
        let twiddles = precompute_twiddles(input.log_n, config);
        run_path_a(&input, config, &twiddles, false).unwrap();
        run_path_b(&input, config, &twiddles, false).unwrap();
    }

    #[test]
    fn relation_columns_match_lookup_relation() {
        let input = LookupInput::generate(8, 3, false);
        let relation = LookupRelation::draw(&mut Blake2sChannel::default());
        let relation_shift: SecureField = relation.combine(&[BaseField::zero()]);

        let columns = relation_columns(-relation_shift, &input);

        assert_eq!(columns.len(), input.n_use_columns() + 1);
        for (mle, source) in columns.iter().zip(
            input
                .use_columns
                .iter()
                .chain(std::iter::once(&input.table)),
        ) {
            let values = mle.clone().into_evals();
            assert_eq!(values.data.len(), source.data.len());
            for (&actual, &value) in values.data.iter().zip(&source.data) {
                let expected: PackedSecureField = relation.combine(&[value]);
                assert_eq!(actual.to_array(), expected.to_array());
            }
        }
    }

    #[test]
    fn gkr_e2e_paths_agree() {
        for n_use_columns in [1, 3] {
            let input = LookupInput::generate(8, n_use_columns, false);
            let config = PcsConfig::default();
            let twiddles = precompute_twiddles(input.log_n, config);
            run_path_a(&input, config, &twiddles, false).unwrap();
            run_path_b(&input, config, &twiddles, false).unwrap();
        }
    }

    #[test]
    fn gkr_e2e_rejects_bad_multiplicity() {
        let input = LookupInput::generate(8, 2, true);
        let config = PcsConfig::default();
        let twiddles = precompute_twiddles(input.log_n, config);
        assert!(run_path_a(&input, config, &twiddles, false).is_err());
        assert!(run_path_b(&input, config, &twiddles, false).is_err());
    }

    pub(super) fn run_one() {
        let log_n = std::env::var("GKR_E2E_LOG_N")
            .map(|value| value.parse().expect("GKR_E2E_LOG_N must be a u32"))
            .unwrap_or(16);
        let n_use_columns = std::env::var("GKR_E2E_L")
            .map(|value| value.parse().expect("GKR_E2E_L must be a usize"))
            .unwrap_or(1);
        let path = std::env::var("GKR_E2E_PATH").unwrap_or_else(|_| "both".into());
        let input = LookupInput::generate(log_n, n_use_columns, false);
        let config = PcsConfig::default();
        let twiddles = precompute_twiddles(input.log_n, config);

        match path.as_str() {
            "a" => run_path_a(&input, config, &twiddles, true).unwrap(),
            "b" => run_path_b(&input, config, &twiddles, true).unwrap(),
            "both" => {
                run_path_a(&input, config, &twiddles, true).unwrap();
                run_path_b(&input, config, &twiddles, true).unwrap();
            }
            _ => panic!("GKR_E2E_PATH must be a, b, or both"),
        }
    }
}

#[cfg(test)]
#[test]
#[ignore = "measurement harness; run explicitly with GKR_E2E_* env vars"]
fn run_one() {
    tests::run_one();
}
