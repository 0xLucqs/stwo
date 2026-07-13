#![cfg_attr(feature = "prover", feature(portable_simd))]
#![cfg_attr(not(feature = "std"), no_std)]

/// ! This module contains helpers to express and use constraints for components.
mod component;

#[cfg(feature = "prover")]
pub mod expr;
mod info;
pub mod logup;
#[cfg(all(feature = "prover", feature = "std"))]
pub mod mle_eval;
mod point;
pub mod preprocessed_columns;
#[cfg(all(feature = "prover", feature = "std"))]
mod prover;

use core::array;
use core::fmt::Debug;
use core::ops::{Add, AddAssign, Mul, Neg, Sub};

pub use component::{FrameworkComponent, FrameworkEval, TraceLocationAllocator};
pub use info::InfoEvaluator;
use num_traits::{One, Zero};
pub use point::PointEvaluator;
use preprocessed_columns::PreProcessedColumnId;
#[cfg(all(feature = "prover", feature = "std"))]
pub use prover::{
    assert_constraints_on_polys, assert_constraints_on_trace, relation_tracker, AssertEvaluator,
    CpuDomainEvaluator, FractionWriter, LogupColGenerator, LogupTraceGenerator,
    SimdDomainEvaluator,
};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use stwo::core::fields::FieldExpOps;
use stwo::core::Fraction;

#[rustfmt::skip]
pub use stwo::core::verifier::PREPROCESSED_TRACE_IDX;
pub const ORIGINAL_TRACE_IDX: usize = 1;
pub const INTERACTION_TRACE_IDX: usize = 2;
/// Maximum number of trace interactions (i.e. trace polynomials committed before the composition
/// polynomial is computed). Currently all components need at most 3 interactions, except
/// [`mle_eval::MleEvalProverComponent`]: hosted in a post-interaction (tree-3) slot it spans 4
/// committed trees plus its internal aux tree, i.e. 5 interactions.
pub const MAX_N_INTERACTIONS: usize = 5;

/// A trait for evaluating expressions at some point or row.
pub trait EvalAtRow {
    // TODO(Ohad): Use a better trait for these, like 'Algebra' or something.
    /// The field type holding values of columns for the component. These are the inputs to the
    /// constraints. It might be [BaseField] packed types, or even [SecureField], when evaluating
    /// the columns out of domain.
    type F: FieldExpOps
        + Clone
        + Debug
        + Zero
        + Neg<Output = Self::F>
        + AddAssign
        + AddAssign<BaseField>
        + Add<Self::F, Output = Self::F>
        + Sub<Self::F, Output = Self::F>
        + Mul<BaseField, Output = Self::F>
        + Add<SecureField, Output = Self::EF>
        + Mul<SecureField, Output = Self::EF>
        + Neg<Output = Self::F>
        + From<BaseField>;

    /// A field type representing the closure of `F` with multiplying by [SecureField]. Constraints
    /// usually get multiplied by [SecureField] values for security.
    type EF: One
        + Clone
        + Debug
        + Zero
        + Neg<Output = Self::EF>
        + AddAssign
        + Add<BaseField, Output = Self::EF>
        + Mul<BaseField, Output = Self::EF>
        + Add<SecureField, Output = Self::EF>
        + Sub<SecureField, Output = Self::EF>
        + Mul<SecureField, Output = Self::EF>
        + Add<Self::F, Output = Self::EF>
        + Mul<Self::F, Output = Self::EF>
        + Sub<Self::EF, Output = Self::EF>
        + Mul<Self::EF, Output = Self::EF>
        + From<SecureField>
        + From<Self::F>;

    /// Returns the next mask value for the first interaction at offset 0.
    fn next_trace_mask(&mut self) -> Self::F {
        let [mask_item] = self.next_interaction_mask(ORIGINAL_TRACE_IDX, [0]);
        mask_item
    }

    fn get_preprocessed_column(&mut self, _column: PreProcessedColumnId) -> Self::F {
        let [mask_item] = self.next_interaction_mask(PREPROCESSED_TRACE_IDX, [0]);
        mask_item
    }

    /// Returns the mask values of the given offsets for the next column in the interaction.
    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N];

    /// Returns the extension mask values of the given offsets for the next extension degree many
    /// columns in the interaction.
    fn next_extension_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::EF; N] {
        let mut res_col_major =
            array::from_fn(|_| self.next_interaction_mask(interaction, offsets).into_iter());
        array::from_fn(|_| {
            Self::combine_ef(res_col_major.each_mut().map(|iter| iter.next().unwrap()))
        })
    }

    /// Adds a constraint to the component.
    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: Mul<G, Output = Self::EF> + From<G>;

    /// Adds an intermediate value in the base field to the component and returns its value.
    /// Does nothing by default.
    fn add_intermediate(&mut self, val: Self::F) -> Self::F {
        val
    }

    /// Adds an intermediate value in the extension field to the component and returns its value.
    /// Does nothing by default.
    fn add_extension_intermediate(&mut self, val: Self::EF) -> Self::EF {
        val
    }

    /// Combines 4 base field values into a single extension field value.
    fn combine_ef(values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF;

    /// Adds `entry.values` to `entry.relation` with `entry.multiplicity` for all 'entry' in
    /// 'entries', batched together.
    /// Constraint degree increases with number of batched constraints as the denominators are
    /// multiplied.
    fn add_to_relation<R: Relation<Self::F, Self::EF>>(
        &mut self,
        entry: RelationEntry<'_, Self::F, Self::EF, R>,
    ) {
        let denominator = entry.relation.combine(entry.values);
        self.write_logup_frac_typed(entry.multiplicity.clone(), denominator);
    }

    // TODO(alont): Remove these once LogupAtRow is no longer used.
    fn write_logup_frac(&mut self, fraction: Fraction<Self::EF, Self::EF>) {
        self.write_logup_frac_typed(Multiplicity::Ext(fraction.numerator), fraction.denominator);
    }

    /// Writes a logup fraction whose numerator keeps its multiplicity representation, letting
    /// the finalize step use cheaper formulas for constant (±1) and base-field numerators.
    fn write_logup_frac_typed(
        &mut self,
        _numerator: Multiplicity<Self::F, Self::EF>,
        _denominator: Self::EF,
    ) {
        unimplemented!()
    }
    fn finalize_logup_batched(&mut self, _batch_size: usize) {
        unimplemented!()
    }

    fn finalize_logup(&mut self) {
        unimplemented!();
    }

    fn finalize_logup_in_pairs(&mut self) {
        unimplemented!();
    }
}

/// Default implementation for evaluators that have an element called "logup" that works like a
/// LogupAtRow, where the logup functionality can be proxied.
/// TODO(alont): Remove once LogupAtRow is no longer used.
macro_rules! logup_proxy {
    () => {
        fn write_logup_frac_typed(
            &mut self,
            numerator: $crate::Multiplicity<Self::F, Self::EF>,
            denominator: Self::EF,
        ) {
            if self.logup.fracs.is_empty() {
                self.logup.is_finalized = false;
            }
            self.logup.fracs.push((numerator, denominator));
        }

        /// Finalize the logup by adding the constraints for the fractions, batched into
        /// consecutive groups of `batch_size`. If the number of fractions is not a multiple
        /// of `batch_size`, the last group is smaller.
        // Not all `EF` types implement `MulAssign`, so `den = den * d` cannot be `den *= d`.
        #[allow(clippy::assign_op_pattern)]
        fn finalize_logup_batched(&mut self, batch_size: usize) {
            assert!(!self.logup.is_finalized, "LogupAtRow was already finalized");
            assert!(batch_size > 0, "Batch size must be positive");

            // Sum each chunk of fractions. Numerators keep their multiplicity representation so
            // that constant (±1) and base-field numerators use cheaper multiplication formulas.
            // The resulting values are identical to the generic `Fraction` sum: multiplying by
            // one is the identity and base-field multiplication agrees with the lifted
            // extension-field multiplication, in exact field arithmetic.
            let mut batched: std_shims::Vec<Fraction<Self::EF, Self::EF>> = self
                .logup
                .fracs
                .chunks(batch_size)
                .map(|chunk| {
                    let (first_num, first_den) = chunk[0].clone();
                    let mut rest = chunk[1..].iter().cloned();
                    let Some((second_num, second_den)) = rest.next() else {
                        return Fraction::new(first_num.to_ef(), first_den);
                    };
                    // (n1/d1) + (n2/d2) = (n1*d2 + n2*d1) / (d1*d2), with n1, n2 still typed.
                    let mut num =
                        first_num.mul_by(second_den.clone()) + second_num.mul_by(first_den.clone());
                    let mut den = first_den * second_den;
                    for (n, d) in rest {
                        num = d.clone() * num + n.mul_by(den.clone());
                        den = den * d;
                    }
                    Fraction::new(num, den)
                })
                .collect();

            let last_frac = batched.pop().expect("No fractions to finalize");

            let mut prev_col_cumsum = <Self::EF as num_traits::Zero>::zero();

            // All batches except the last are cumulatively summed in new interaction columns.
            for cur_frac in batched {
                let [cur_cumsum] =
                    self.next_extension_interaction_mask(self.logup.interaction, [0]);
                let diff = cur_cumsum.clone() - prev_col_cumsum.clone();
                prev_col_cumsum = cur_cumsum;
                self.add_constraint(diff * cur_frac.denominator - cur_frac.numerator);
            }

            let [prev_row_cumsum, cur_cumsum] =
                self.next_extension_interaction_mask(self.logup.interaction, [-1, 0]);

            let diff = cur_cumsum - prev_row_cumsum - prev_col_cumsum.clone();
            // Instead of checking diff = num / denom, check diff = num / denom - cumsum_shift.
            // This makes (num / denom - cumsum_shift) have sum zero, which makes the constraint
            // uniform - apply on all rows.
            let shifted_diff = diff + self.logup.cumsum_shift.clone();

            self.add_constraint(shifted_diff * last_frac.denominator - last_frac.numerator);

            self.logup.is_finalized = true;
        }

        /// Finalizes the row's logup in the default way. Currently, this means no batching.
        fn finalize_logup(&mut self) {
            self.finalize_logup_batched(1)
        }

        /// Finalizes the row's logup, batched in pairs.
        /// TODO(alont) Remove this once a better batching mechanism is implemented.
        fn finalize_logup_in_pairs(&mut self) {
            self.finalize_logup_batched(2)
        }
    };
}
pub(crate) use logup_proxy;

pub trait RelationEFTraitBound<F: Clone>:
    Clone + Zero + From<F> + From<SecureField> + Mul<F, Output = Self> + Sub<Self, Output = Self>
{
}

impl<F, EF> RelationEFTraitBound<F> for EF
where
    F: Clone,
    EF: Clone + Zero + From<F> + From<SecureField> + Mul<F, Output = EF> + Sub<EF, Output = EF>,
{
}

/// A trait for defining a logup relation type.
pub trait Relation<F: Clone, EF: RelationEFTraitBound<F>>: Sized {
    fn combine(&self, values: &[F]) -> EF;

    fn get_name(&self) -> &str;
    fn get_size(&self) -> usize;
}

/// A logup multiplicity, kept in its cheapest available representation.
///
/// Multiplying an extension-field denominator by a multiplicity costs a full EF×EF
/// multiplication only in the general [`Multiplicity::Ext`] case. Constant ±1 multiplicities
/// (plain uses/yields) and base-field multiplicities (e.g. table multiplicity columns) admit
/// strictly cheaper formulas that produce bit-identical field values.
#[derive(Clone, Debug)]
pub enum Multiplicity<F, EF> {
    /// Multiplicity is exactly one (a plain "use").
    One,
    /// Multiplicity is exactly minus one (a plain "yield").
    NegOne,
    /// Multiplicity in the base field (e.g. a multiplicity trace column).
    Base(F),
    /// General extension-field multiplicity.
    Ext(EF),
}

impl<F, EF> Multiplicity<F, EF> {
    /// Multiplies `rhs` by this multiplicity. Identical in value to `self.to_ef() * rhs`.
    pub fn mul_by(self, rhs: EF) -> EF
    where
        EF: Neg<Output = EF> + Mul<F, Output = EF> + Mul<EF, Output = EF>,
    {
        match self {
            Self::One => rhs,
            Self::NegOne => -rhs,
            Self::Base(f) => rhs * f,
            // `rhs * e` (not `e * rhs`) matches the operand order of the previous generic
            // `Fraction` sum exactly, keeping formal expression trees byte-identical.
            Self::Ext(e) => rhs * e,
        }
    }

    /// Lifts the multiplicity to the extension field.
    pub fn to_ef(self) -> EF
    where
        EF: One + Neg<Output = EF> + From<F>,
    {
        match self {
            Self::One => EF::one(),
            Self::NegOne => -EF::one(),
            Self::Base(f) => EF::from(f),
            Self::Ext(e) => e,
        }
    }
}

/// A struct representing a relation entry.
/// `relation` is the relation into which elements are entered.
/// `multiplicity` is the multiplicity of the elements.
///     A positive multiplicity is used to signify a "use", while a negative multiplicity
///     signifies a "yield".
/// `values` are elements in the base field that are entered into the relation.
pub struct RelationEntry<'a, F: Clone, EF: RelationEFTraitBound<F>, R: Relation<F, EF>> {
    relation: &'a R,
    multiplicity: Multiplicity<F, EF>,
    values: &'a [F],
}
impl<'a, F: Clone, EF: RelationEFTraitBound<F>, R: Relation<F, EF>> RelationEntry<'a, F, EF, R> {
    /// General extension-field multiplicity. Prefer [`Self::unit`], [`Self::neg_unit`] or
    /// [`Self::base`] when the multiplicity is a constant ±1 or a base-field value — they
    /// produce identical constraint values with strictly fewer field multiplications.
    pub const fn new(relation: &'a R, multiplicity: EF, values: &'a [F]) -> Self {
        Self {
            relation,
            multiplicity: Multiplicity::Ext(multiplicity),
            values,
        }
    }

    /// Entry with multiplicity exactly one (a plain "use").
    pub const fn unit(relation: &'a R, values: &'a [F]) -> Self {
        Self {
            relation,
            multiplicity: Multiplicity::One,
            values,
        }
    }

    /// Entry with multiplicity exactly minus one (a plain "yield").
    pub const fn neg_unit(relation: &'a R, values: &'a [F]) -> Self {
        Self {
            relation,
            multiplicity: Multiplicity::NegOne,
            values,
        }
    }

    /// Entry whose multiplicity is a base-field value (e.g. a multiplicity trace column).
    pub const fn base(relation: &'a R, multiplicity: F, values: &'a [F]) -> Self {
        Self {
            relation,
            multiplicity: Multiplicity::Base(multiplicity),
            values,
        }
    }
}

#[macro_export]
macro_rules! relation {
    ($name:tt, $size:tt) => {
        #[derive(Clone, Debug, PartialEq)]
        pub struct $name($crate::logup::LookupElements<$size>);

        #[allow(dead_code)]
        impl $name {
            pub fn dummy() -> Self {
                Self($crate::logup::LookupElements::dummy())
            }
            pub fn draw(channel: &mut impl stwo::core::channel::Channel) -> Self {
                Self($crate::logup::LookupElements::draw(channel))
            }
        }

        impl<F: Clone, EF: $crate::RelationEFTraitBound<F>> $crate::Relation<F, EF> for $name {
            fn combine(&self, values: &[F]) -> EF {
                self.0.combine(values)
            }

            fn get_name(&self) -> &str {
                stringify!($name)
            }

            fn get_size(&self) -> usize {
                $size
            }
        }
    };
}

#[cfg(test)]
#[macro_export]
macro_rules! m31 {
    ($m:expr) => {
        stwo::core::fields::m31::M31::from_u32_unchecked($m)
    };
}

#[cfg(test)]
#[macro_export]
macro_rules! qm31 {
    ($m0:expr, $m1:expr, $m2:expr, $m3:expr) => {{
        stwo::core::fields::qm31::QM31::from_u32_unchecked($m0, $m1, $m2, $m3)
    }};
}
