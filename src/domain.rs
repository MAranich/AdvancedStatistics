//! A Domain represents the set of points where a function is defined.
//!
//! In this library we use it mainly for the pdf or pmf of Distributons (see
//! [crate::distribution_trait]). It has 2 variants:
//!  - [DiscreteDomain]
//!  - [ContinuousDomain]
//!
//! If you intend to have a domain more complex than that (for a piecewise
//! function, for example), you should take a look at [crate::mixed_distribution].
//!

use core::f64;

use crate::euclid::DEFAULT_EMPTY_DOMAIN_BOUNDS;

/// A [domain](https://en.wikipedia.org/wiki/Domain_of_a_function) composed of
/// finitely many elements.
///
/// [DiscreteDomain] assumes that most discrete domains only include integers.
/// If your domain does not fit this description, here are some possible solutions:
///  - If your domain is a constant factor from the integers (pdf(x) can be evaluated
///     at every `x = k/2` for some integer `k`, you can compute a new pfd_2(x) {pdf(x * 2)}).
///  - In a more general way, you can generate a function `map_domain()` that maps from
///     the original domain to the integers. (so you can call `pdf(map_domain(x))`)
///  - Otherwise you may be interested on the [DiscreteDomain::Custom] variant, wich allows
///     you to maually indicate the values you want to include on your domain.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum DiscreteDomain {
    /// All integers
    #[default]
    Integers,
    /// All the integers in the range [.0, .1] (**both** inclusive).
    /// The first number is the minimum, and the last is the maximum.
    ///
    /// Has the **invariant** that `min <= max`.
    Range(i64, i64),
    /// All the integers from the given value onwards. The value **is** included.
    From(i64),
    /// All the integers until the given value. The value **is** included.
    To(i64),
    /// Detemine manually at wich points can this function be evaluated.
    ///
    /// This variant has the following **invariants**:
    ///  - No infinities (either positive or negative)
    ///  - No NaNs
    ///  - No repeated elements
    ///  - The values in the vector must be sorted
    ///
    /// Use [DiscreteDomain::new_discrete_custom] when creating this variant to ensure
    /// all the invariants are fullfilled.
    Custom(Vec<f64>),
}

/// A [domain](https://en.wikipedia.org/wiki/Domain_of_a_function) of a region
/// of the real numbers.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ContinuousDomain {
    /// All real numbers
    #[default]
    Reals,
    /// The values contained in the range.
    ///
    /// The first number is the minimum, and the last is the maximum.
    ///
    /// Has the **invariant** that `min <= max`.
    Range(f64, f64),
    /// All the numbers from the given value onwards.
    From(f64),
    /// All the numbers until the given value.
    To(f64),
}

impl DiscreteDomain {
    /// Create a domain composed only by the given `valiues` ([DiscreteDomain::Custom]).
    /// This method makes sure to fullfill the necessary invariants:
    ///  - No infinities (either positive or negative)
    ///  - No NaNs
    ///  - No repeated elements
    ///  - The values in the vector must be sorted
    #[must_use]
    pub fn new_discrete_custom(values: &[f64]) -> Self {
        let mut points: Vec<f64> = values
            .iter()
            .copied()
            .filter(|&x| x.is_finite())
            .collect::<Vec<f64>>();
        points.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // remove duplicate elements. (will remove all because `points` is sorted).
        points.dedup();

        return DiscreteDomain::Custom(points);
    }

    #[must_use]
    pub fn contains(&self, x: f64) -> bool {
        if let DiscreteDomain::Custom(vec) = self {
            return vec
                .binary_search_by(|other| x.partial_cmp(other).unwrap())
                .is_ok();
        }

        if x.fract() != 0.0 {
            // the value is fractional, but all other variants only include integers
            return false;
        }

        let x_int: i64 = x as i64;

        match self {
            DiscreteDomain::Integers => x.fract() == 0.0,
            DiscreteDomain::Range(min, max) => (*min <= x_int) && (x_int <= *max),
            DiscreteDomain::From(min) => *min <= x_int,
            DiscreteDomain::To(max) => x_int <= *max,
            _ => {
                // DiscreteDomain::Custom(vec) => todo!(),
                // ^already done
                unreachable!();
            }
        }
    }

    /// Returns the upper and lower bounds of the domain.
    ///
    /// Take into account that the values can also include positive and negative infinity.
    /// It is guaranteed that return.0 <= return.1. If the bounds are finite, the values
    /// themselves are included.
    ///
    /// If the domain is empty, [DEFAULT_EMPTY_DOMAIN_BOUNDS] = `(-0.0, 0.0)` is returned.
    #[must_use]
    pub fn get_bounds(&self) -> (f64, f64) {
        match &self {
            DiscreteDomain::Integers => (f64::NEG_INFINITY, f64::INFINITY),
            DiscreteDomain::Range(min, max) => (*min as f64, *max as f64),
            DiscreteDomain::From(min) => (*min as f64, f64::INFINITY),
            DiscreteDomain::To(max) => (f64::NEG_INFINITY, *max as f64),
            DiscreteDomain::Custom(vec) => match vec.first() {
                Some(first) => (*first, *vec.last().unwrap()),
                None => DEFAULT_EMPTY_DOMAIN_BOUNDS,
            },
        }
    }

    /// Returns an iteratior that iterates trough all the elements in the domain.
    ///
    /// Warning: the iterator may be infinite.
    #[must_use]
    pub fn iter(&self) -> DiscreteDomainIterator {
        // current_value being a NaN sybmolyzes that no values have been given yet
        DiscreteDomainIterator {
            domain: self,
            current_value: f64::NAN,
            custom_domain_index: 0,
        }
    }

    /// Returns true if the domain contains a finite number of elements.
    #[must_use]
    pub fn contains_finite_elements(&self) -> bool {
        match self {
            DiscreteDomain::Range(_, _) | DiscreteDomain::Custom(_) => true,
            DiscreteDomain::Integers | DiscreteDomain::From(_) | DiscreteDomain::To(_) => false,
        }
    }
}

impl ContinuousDomain {
    #[must_use]
    pub fn contains(&self, x: f64) -> bool {
        match self {
            ContinuousDomain::Reals => true,
            ContinuousDomain::Range(min, max) => (*min <= x) && (x <= *max),
            ContinuousDomain::From(min) => *min <= x,
            ContinuousDomain::To(max) => x <= *max,
        }
    }

    /// Returns the upper and lower bounds of the domain.
    ///
    /// Take into account that the values can also include positive and negative infinity.
    /// It is guaranteed that return.0 <= return.1. If the bounds are finite, the values
    /// themselves are included.
    ///
    /// If the domain is empty, [DEFAULT_EMPTY_DOMAIN_BOUNDS] = `(-0.0, 0.0)` is returned.
    #[must_use]
    pub fn get_bounds(&self) -> (f64, f64) {
        match &self {
            ContinuousDomain::Reals => (f64::NEG_INFINITY, f64::INFINITY),
            ContinuousDomain::Range(min, max) => (*min, *max),
            ContinuousDomain::From(min) => (*min, f64::INFINITY),
            ContinuousDomain::To(max) => (f64::NEG_INFINITY, *max),
        }
    }
}

pub struct DiscreteDomainIterator<'a> {
    domain: &'a DiscreteDomain,

    current_value: f64,
    /// auxiliar value if the domain is [DiscreteDomain::Custom]
    custom_domain_index: usize,
}

impl Iterator for DiscreteDomainIterator<'_> {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        match self.domain {
            DiscreteDomain::Integers => {
                if self.current_value.is_nan() {
                    self.current_value = 0.0;
                    return Some(self.current_value);
                }

                if self.current_value.is_sign_positive() {
                    self.current_value = -(self.current_value + 1.0);
                } else {
                    self.current_value = -self.current_value;
                }

                return Some(self.current_value);
            }
            DiscreteDomain::Range(min, max) => {
                if self.current_value.is_nan() {
                    self.current_value = *min as f64;
                    return Some(self.current_value);
                }
                self.current_value = self.current_value + 1.0;
                if (*max as f64) < self.current_value {
                    return None;
                }
                return Some(self.current_value);
            }
            DiscreteDomain::From(min) => {
                if self.current_value.is_nan() {
                    self.current_value = *min as f64;
                    return Some(self.current_value);
                }

                self.current_value = self.current_value + 1.0;
                return Some(self.current_value);
            }
            DiscreteDomain::To(max) => {
                if self.current_value.is_nan() {
                    self.current_value = *max as f64;
                    return Some(self.current_value);
                }

                self.current_value = self.current_value - 1.0;
                return Some(self.current_value);
            }
            DiscreteDomain::Custom(vec) => {
                self.custom_domain_index += 1;

                if self.current_value.is_nan() {
                    self.current_value = 0.0;
                    // ^flag that we have already given the first value

                    self.custom_domain_index = 0;

                    return vec.first().copied();
                }

                return vec.get(self.custom_domain_index).copied();
            }
        }
    }
}
