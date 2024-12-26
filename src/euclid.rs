//! Euclid contains uscefull math functions

use rand::Rng;

use crate::{
    DEFAULT_INTEGRATION_MAXIMUM_STEPS, DEFAULT_INTEGRATION_MAXIMUM_STEPS_F64,
    DEFAULT_INTEGRATION_MINIMUM_STEPS, DEFAULT_INTEGRATION_MINIMUM_STEPS_F64,
    DEFAULT_INTEGRATION_PRECISION, SMALL_INTEGRATION_NUM_STEPS, SMALL_INTEGRATION_PRECISION,
};

/// The [moments](https://en.wikipedia.org/wiki/Moment_(mathematics)) of a function
/// are some values that provide information about the shape of the function.
/// If the function is a valid pdf, the mean, variance, skewness and other values
/// of interest can be expressed as moments.
///
/// The moments can have be of any of the 3 variants in this enum:
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum Moments {
    #[default]
    Raw,
    Central,
    Standarized,
}

/// Domain represents a [domain](https://en.wikipedia.org/wiki/Domain_of_a_function)
/// of a function (the points where it can be evaluated). The Domain struct provides
/// enough functionality to easly create most of the most common domains.
#[derive(Debug, Clone, PartialEq)]
pub struct Domain {
    domain: DomainType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DomainType {
    Discrete(DiscreteDomain),
    Continuous(ContinuousDomain),
    Mixed(MixedDomain),
}

/// A domain composed of finitely many elements.
///
/// DiscreteDomain assumes that most discrete domains only include integers.
/// If your domain does not fit this description, here are some possible solutions:
///  - If your domain is a constant factor from the integers (pdf(x) can be evaluated
/// at every `x = k/2` for some integer `k`, you can compute a new pfd_2(x) {pdf(x * 2)}).
///  - In a more general way, you can generate a function `map_domain()` that maps from
/// the original domain to the integers. (so you can call `pdf(map_domain(x))`)
///  - Otherwise you may be interested on the [DiscreteDomain::Custom] variant, wich allows
/// you to maually indicate the values you want to include on your domain.
///  - If this still does not fit you, another option is to use a [DomainType::Mixed],
/// wich can join multiple simpler domains.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum DiscreteDomain {
    /// All integers
    #[default]
    Integers,
    /// Positive integers. The bool indicates if 0 is included.
    Positive(bool),
    /// Negative integers. The bool indicates if 0 is included.
    Negative(bool),
    /// All the numbers in the range [.0, .1] (**both** inclusive).
    Range(i64, i64),
    /// All the numbers from the given value onwards. The value **is** included.
    From(i64),
    /// All the numbers until the given value. The value **is** included.
    To(i64),
    /// Detemine manually at wich points can this function be evaluated. Has an invariant of
    /// being finite values and **sorted** (needs to be sorted in order to work properly).
    Custom(Vec<f64>),
}

/// A domain a section of th real numbers.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ContinuousDomain {
    /// All real numbers
    #[default]
    Reals,
    /// All values in `[0, 1]`
    ZeroOne,
    /// Only the positive numbers. The bool determines if 0 is included or not.
    Positive(bool),
    /// Only the negative numbers. The bool determines if 0 is included or not.
    Negative(bool),
    /// Only the numbers in the range
    Range(f64, f64),
}

/// Mixed Domain allows the creation of more comlpex domains by mixing multiple
/// of the basic types.
#[derive(Debug, Clone, PartialEq)]
pub enum MixedDomain {
    /// The points contained in **any** of the domains.
    Union(Box<Vec<Domain>>),
    /// The points contained in **all** of the domains
    Disjunction(Box<Vec<Domain>>),
    /// The points that do **not** belong to the domain
    Not(Box<DomainType>),
}
/// Todo implement Debug for MixedDomain, Domain and DomainType

/// Determine the normalitzation constant of a pdf.
///
/// You need to divide the value given by `pdf` by the returned value in order to have
/// a valid probability distribution.
///
/// This function assumes that `pdf` contains a finite area in it's `domain`.
pub fn determine_normalitzation_constant_continuous(
    pdf: impl Fn(f64) -> f64,
    domain: &Domain,
) -> f64 {
    let bounds: (f64, f64) = domain.get_bounds();

    let pdf_checked = |x: f64| {
        if domain.contains(x) {
            pdf(x)
        } else {
            0.0
        }
    };

    let (step_length, _): (f64, usize) = choose_integration_precision_and_steps(bounds);

    // To simpligy things + readability
    enum IntegrationType {
        // closed interval [a, b]
        Finite,
        // [-inf, a]
        InfiniteToPositive,
        // [b, inf]
        InfiniteToNegative,
        // [-inf, inf]
        FullInfinite,
    }

    let integration_type: IntegrationType = match (bounds.0.is_finite(), bounds.1.is_finite()) {
        (true, true) => IntegrationType::Finite,
        (true, false) => IntegrationType::InfiniteToPositive,
        (false, true) => IntegrationType::InfiniteToNegative,
        (false, false) => IntegrationType::FullInfinite,
    };

    let double_step_length: f64 = 2.0 * step_length;
    let step_len_over_3: f64 = step_length / 3.0;

    // let mut last_pdf_evaluation: f64 = pdf_checked(bounds.0);
    let mut last_pdf_evaluation: f64 = match integration_type {
        IntegrationType::Finite => pdf_checked(bounds.0),
        IntegrationType::InfiniteToPositive => {
            // t = 0, it would be a singularity. Skip point
            0.0
        }
        IntegrationType::InfiniteToNegative => {
            // t = 0;     f(a + t/(t - 1))  /  (1 - t)^2
            pdf_checked(bounds.0)
        }
        IntegrationType::FullInfinite => {
            // t = -1;    f(t/(1 - t^2))  *  (1 + t^2) / (1 - t^2)^2
            // would be singularity, skip
            0.0
        }
    };

    let mut accumulator: f64 = 0.0;
    let mut num_step: f64 = 0.0;

    //'integration_loop: loop {
    loop {
        let current_position: f64 = bounds.0 + double_step_length * num_step;

        let (middle, end): (f64, f64) = match integration_type {
            IntegrationType::Finite => {
                let fn_input: f64 = current_position + step_length;
                let middle_: f64 = pdf_checked(fn_input);

                let fn_input: f64 = current_position + double_step_length;
                let end_: f64 = pdf_checked(fn_input);

                (middle_, end_)
            }
            IntegrationType::InfiniteToPositive => {
                //      For -infinite to const:
                // integral {-inf -> a} f(x) dx = integral {0 -> 1} f(a - (1 - t)/t)  /  t^2  dt

                let middle_: f64 = {
                    let t: f64 = current_position + step_length;
                    if t.abs() < f64::EPSILON {
                        // too near singularity, skip
                        0.0
                    } else {
                        let fn_input: f64 = bounds.1 - (1.0 - t) / t;
                        pdf_checked(fn_input) / (t * t)
                    }
                };
                let end_: f64 = {
                    let t: f64 = current_position + double_step_length;
                    if t.abs() < f64::EPSILON {
                        // too near singularity, skip
                        0.0
                    } else {
                        let fn_input: f64 = bounds.1 - (1.0 - t) / t;
                        pdf_checked(fn_input) / (t * t)
                    }
                };
                (middle_, end_)
            }
            IntegrationType::InfiniteToNegative => {
                //For const to infinite:
                // integral {a -> inf} f(x) dx  = integral {0 -> 1} f(a + t/(t - 1))  /  (1 - t)^2  dt

                let middle_: f64 = {
                    let t: f64 = current_position + step_length;
                    let t_minus: f64 = t - 1.0;
                    if t_minus.abs() < f64::EPSILON {
                        // too near singularity, skip
                        0.0
                    } else {
                        let fn_input: f64 = bounds.0 + t / t_minus;
                        pdf_checked(fn_input) / (t_minus * t_minus)
                    }
                };
                let end_: f64 = {
                    let t: f64 = current_position + double_step_length;
                    let t_minus: f64 = t - 1.0;
                    if t_minus.abs() < f64::EPSILON {
                        // too near singularity, skip
                        0.0
                    } else {
                        let fn_input: f64 = bounds.0 + t / t_minus;
                        pdf_checked(fn_input) / (t_minus * t_minus)
                    }
                };
                (middle_, end_)
            }
            IntegrationType::FullInfinite => {
                // For -infinite to infinite:
                // integral {-inf -> inf} f(x) dx  = integral {-1 -> 1} f(t/(1 - t^2))  *  (1 + t^2) / (1 - t^2)^2  dt

                let middle_: f64 = {
                    let t: f64 = current_position + step_length;
                    let u: f64 = 1.0 - t * t;
                    if u.abs() < f64::EPSILON {
                        // too near singularity, skip
                        0.0
                    } else {
                        let fn_input: f64 = t / u;
                        pdf_checked(fn_input) * (1.0 + t * t) / (u * u)
                    }
                };
                let end_: f64 = {
                    let t: f64 = current_position + double_step_length;
                    let u: f64 = 1.0 - t * t;
                    if u.abs() < f64::EPSILON {
                        // too near singularity, skip
                        0.0
                    } else {
                        let fn_input: f64 = t / u;
                        pdf_checked(fn_input) * (1.0 + t * t) / (u * u)
                    }
                };
                (middle_, end_)
            }
        };

        accumulator += step_len_over_3 * (last_pdf_evaluation + 4.0 * middle + end);

        match integration_type {
            IntegrationType::Finite => {
                if bounds.1 <= current_position {
                    break;
                }
            }
            _ => {
                if 1.0 <= current_position {
                    break;
                }
            }
        }

        last_pdf_evaluation = end;
        num_step += 1.0;
        // we do 2 steps each iteration but at `current_position` we are mult. by `double_step_length`
    }

    return accumulator;
}

/// Randomly permute a slice.
///
/// This is an implementation of [Fisher–Yates shuffle](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle),
/// an efficient algorithm to randomly permute a list in `O(n)`.
pub fn random_permutation<T>(arr: &mut [T]) {
    // [Fisher–Yates shuffle](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle)

    let len: usize = arr.len();
    let mut rng: rand::prelude::ThreadRng = rand::thread_rng();

    for i in (1..=(len - 1)).rev() {
        let mut j: f64 = rng.gen();
        j = j * ((i + 1) as f64);
        let j: usize = j as usize;
        // k belongs to  [0, i - 1]

        arr.swap(i, j);
    }
}

/// Returns (step_length, num_steps) depending on the bounds of integration.
///
/// This function is internal to the library and wraps up everything
/// needed to choose the appropiate step_length (precision), and by consequence
/// the number of steps.
///
/// If the bounds are non-finite a change of variables for the integration is assumed.
///
/// bounds.0 < bounds.1
pub fn choose_integration_precision_and_steps(bounds: (f64, f64)) -> (f64, usize) {
    /*
        To select the appropiate step_length (and total_num_steps, indirecly),
        we need to adapt between the possible cases.
        - For standard integration: Use DEFAULT_INTEGRATION_PRECISION unless it's
            too small (if we would do more than DEFAULT_INTEGRATION_MAXIMUM_STEPS)


        num_steps needs to be odd
    */

    match (bounds.0.is_finite(), bounds.0.is_finite()) {
        (true, true) => {
            let range: f64 = bounds.1 - bounds.0;
            let step_length: f64;
            let num_steps: usize;

            if range <= 1.0 {
                // small range (less than an unit)
                num_steps = SMALL_INTEGRATION_NUM_STEPS;
                // ^already odd
                step_length = range / num_steps as f64;
            } else if DEFAULT_INTEGRATION_PRECISION * DEFAULT_INTEGRATION_MAXIMUM_STEPS_F64 < range
            {
                // interval is very big, we will increase the step_lenght.

                step_length = range / DEFAULT_INTEGRATION_MAXIMUM_STEPS_F64;
                num_steps = DEFAULT_INTEGRATION_MAXIMUM_STEPS;
                // ^already odd
            } else {
                // *normal* interval

                let first_num: f64 = range / DEFAULT_INTEGRATION_PRECISION;
                if first_num < DEFAULT_INTEGRATION_MINIMUM_STEPS_F64 {
                    // too litle steps
                    num_steps = DEFAULT_INTEGRATION_MINIMUM_STEPS;
                } else {
                    num_steps = (first_num as usize) | 1;
                }
                step_length = range / (num_steps as f64);
            }

            (step_length, num_steps)
        }
        (false, false) => {
            // if the interval [0, 1] uses SMALL_INTEGRATION_NUM_STEPS,
            // then [-1, 1] will use the doule. Ajust precision accordingly.
            (
                SMALL_INTEGRATION_PRECISION * 0.5,
                (SMALL_INTEGRATION_NUM_STEPS * 2) as usize,
            )
        }
        _ => {
            /* Cases:
                (true, false) => {},
                (false, true) => {},
            */

            // The range in this case is [0, 1], so we can return the same values.

            (SMALL_INTEGRATION_PRECISION, SMALL_INTEGRATION_NUM_STEPS)
        }
    }
}

pub const DEFAULT_EMPTY_DOMAIN_BOUNDS: (f64, f64) = (-0.0, 0.0);

impl Domain {
    /// A [Domain] composed of all the real numbers in the given range. Note both of the
    /// bounds **are includive**.
    ///
    /// If this domain is for the [crate::distributions::distribution_trait::Distribution::pdf], 
    /// we recomend using this over the real domain and just select the interval 
    /// where the function is non-zero.
    ///
    /// For example, for the standard normal distribution, using a range from
    /// `-8.0` to `8.0` is more than enough because the area left out of that range is
    /// `1 - erf(8/sqrt(2))` wich is less than `1.25 * 10^-15`.
    pub const fn new_continuous_range(min_inclusive: f64, max_inclusive: f64) -> Self {
        let domain_type: DomainType =
            DomainType::Continuous(ContinuousDomain::Range(min_inclusive, max_inclusive));
        Domain {
            domain: domain_type,
        }
    }

    /// A [Domain] containing a continuous range of all the values in `[min_inclusive, max_inclusive]`.
    pub const fn new_real() -> Self {
        Domain {
            domain: DomainType::Continuous(ContinuousDomain::Reals),
        }
    }

    /// A [Domain] containing a continuous range of all the values in `[0, inf)`.
    pub const fn new_continuous_positives(include_zero: bool) -> Self {
        Domain {
            domain: DomainType::Continuous(ContinuousDomain::Positive(include_zero)),
        }
    }

    /// A [Domain] containing a continuous range of all the values in `(-inf, 0]`.
    pub const fn new_continuous_negatives(include_zero: bool) -> Self {
        Domain {
            domain: DomainType::Continuous(ContinuousDomain::Negative(include_zero)),
        }
    }

    /// A [Domain] containing a discrete range of all the values in `[-inf, inf]`.
    pub const fn new_discrete_integers() -> Self {
        let domain_type: DomainType = DomainType::Discrete(DiscreteDomain::Integers);
        Domain {
            domain: domain_type,
        }
    }

    /// A [Domain] containing a discrete range of all the values in `[0, inf)`.
    pub const fn new_discrete_positives(include_zero: bool) -> Self {
        let domain_type: DomainType = DomainType::Discrete(DiscreteDomain::Positive(include_zero));
        Domain {
            domain: domain_type,
        }
    }

    /// A [Domain] containing a discrete range of all the values in `(-inf, 0]`.
    pub const fn new_discrete_negatives(include_zero: bool) -> Self {
        let domain_type: DomainType = DomainType::Discrete(DiscreteDomain::Negative(include_zero));
        Domain {
            domain: domain_type,
        }
    }

    /// A [Domain] containing a discrete range of all the values in `[from_inclusive, inf)`.
    pub const fn new_discrete_from(from_inclusive: i64) -> Self {
        let domain_type: DomainType = DomainType::Discrete(DiscreteDomain::From(from_inclusive));
        Domain {
            domain: domain_type,
        }
    }

    /// A [Domain] containing a discrete range of all the values in `(-inf, to_inclusive]`.
    pub const fn new_discrete_to(to_inclusive: i64) -> Self {
        let domain_type: DomainType = DomainType::Discrete(DiscreteDomain::To(to_inclusive));
        Domain {
            domain: domain_type,
        }
    }

    /// A [Domain] containing a discrete range of all the values in `[min_inclusive, max_inclusive]`.
    pub const fn new_discrete_range(min_inclusive: i64, max_inclusive: i64) -> Self {
        let domain_type: DomainType =
            DomainType::Discrete(DiscreteDomain::Range(min_inclusive, max_inclusive));
        Domain {
            domain: domain_type,
        }
    }

    /// A [Domain] containing a continuous range of all the values in `[0, 1]`.
    pub const fn new_zero_one() -> Self {
        let domain_type: DomainType = DomainType::Continuous(ContinuousDomain::ZeroOne);
        Domain {
            domain: domain_type,
        }
    }

    /// Create a domain composed only by the given `valid_points`.
    ///
    /// All invalid values will be ingored (+-inf, NaNs)
    pub fn new_discrete_custom(valid_values: &[f64]) -> Self {
        let mut points: Vec<f64> = valid_values
            .iter()
            .map(|x| *x)
            .filter(|&x| x.is_finite())
            .collect::<Vec<f64>>();
        points.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let domain_type: DomainType = DomainType::Discrete(DiscreteDomain::Custom(points));
        Domain {
            domain: domain_type,
        }
    }

    /// Joins the 2 given domains into a new one.
    pub fn join_domains(mut self, mut other: Self) -> Self {
        // If some of the 2 domains is already a mixed domain of the Union variant,
        // we can simplify things

        let mut self_is_union: bool = false;
        let mut other_is_union: bool = false;

        match (&self.domain, &other.domain) {
            (DomainType::Mixed(mixed_domain_1), DomainType::Mixed(mixed_domain_2)) => {
                if let MixedDomain::Union(_) = mixed_domain_1 {
                    self_is_union = true;
                }

                if let MixedDomain::Union(_) = mixed_domain_2 {
                    other_is_union = true;
                }
            }
            (_, DomainType::Mixed(mixed_domain)) => {
                if let MixedDomain::Union(_) = mixed_domain {
                    other_is_union = true;
                }
            }
            (DomainType::Mixed(mixed_domain), _) => {
                if let MixedDomain::Union(_) = mixed_domain {
                    self_is_union = true;
                }
            }

            _ => {}
        }

        match (self_is_union, other_is_union) {
            (true, true) => {
                if let DomainType::Mixed(mixed_domain) = &mut self.domain {
                    if let MixedDomain::Union(vec) = mixed_domain {
                        if let DomainType::Mixed(other_mixed_domain) = &mut other.domain {
                            if let MixedDomain::Union(other_vec) = other_mixed_domain {
                                vec.append(other_vec);
                            } else {
                                unreachable!(
                                    "if other_is_union is true, this should be unreachable"
                                )
                            }
                        } else {
                            unreachable!("if other_is_union is true, this should be unreachable")
                        }
                    } else {
                        unreachable!("If self_is_union is true, this should be unreachable")
                    }
                } else {
                    unreachable!("If self_is_union is true, this should be unreachable")
                }

                return self;
            }
            (true, false) => {
                if let DomainType::Mixed(mixed_domain) = &mut self.domain {
                    if let MixedDomain::Union(vec) = mixed_domain {
                        vec.push(other);
                    } else {
                        unreachable!("if self_is_union is true, this should be unreachable")
                    }
                } else {
                    unreachable!("if self_is_union is true, this should be unreachable")
                }

                return self;
            }
            (false, true) => {
                if let DomainType::Mixed(mixed_domain) = &mut other.domain {
                    if let MixedDomain::Union(vec) = mixed_domain {
                        vec.push(self);
                    } else {
                        unreachable!("if other_is_union is true, this should be unreachable")
                    }
                } else {
                    unreachable!("if other_is_union is true, this should be unreachable")
                }

                return other;
            }
            (false, false) => {}
        }

        // If neither of the 2 domains is a mixed domain of the join variant,
        // just create a new muxed union domain.

        let domain_type: DomainType =
            DomainType::Mixed(MixedDomain::Union(Box::new(vec![self, other])));
        Domain {
            domain: domain_type,
        }
    }

    /// Creates a new domain with just the intersection of the 2 given domains.
    ///
    /// The counterpart of [Domain::join_domains]
    pub fn intersection_of_domains(mut self, mut other: Self) -> Self {
        // If some of the 2 domains is already a mixed domain of the Disjunction variant,
        // we can simplify things

        let mut self_is_intersection: bool = false;
        let mut other_is_intersection: bool = false;

        match (&self.domain, &other.domain) {
            (DomainType::Mixed(mixed_domain_1), DomainType::Mixed(mixed_domain_2)) => {
                if let MixedDomain::Disjunction(_) = mixed_domain_1 {
                    self_is_intersection = true;
                }

                if let MixedDomain::Disjunction(_) = mixed_domain_2 {
                    other_is_intersection = true;
                }
            }
            (_, DomainType::Mixed(mixed_domain)) => {
                if let MixedDomain::Disjunction(_) = mixed_domain {
                    other_is_intersection = true;
                }
            }
            (DomainType::Mixed(mixed_domain), _) => {
                if let MixedDomain::Disjunction(_) = mixed_domain {
                    self_is_intersection = true;
                }
            }
            _ => {}
        }

        match (self_is_intersection, other_is_intersection) {
            (true, true) => {
                if let DomainType::Mixed(mixed_domain) = &mut self.domain {
                    if let MixedDomain::Disjunction(vec) = mixed_domain {
                        if let DomainType::Mixed(other_mixed_domain) = &mut other.domain {
                            if let MixedDomain::Disjunction(other_vec) = other_mixed_domain {
                                vec.append(other_vec);
                            } else {
                                unreachable!(
                                    "if other_is_intersection is true, this should be unreachable"
                                )
                            }
                        } else {
                            unreachable!(
                                "if other_is_intersection is true, this should be unreachable"
                            )
                        }
                    } else {
                        unreachable!("If self_is_intersection is true, this should be unreachable")
                    }
                } else {
                    unreachable!("If self_is_intersection is true, this should be unreachable")
                }

                return self;
            }
            (true, false) => {
                if let DomainType::Mixed(mixed_domain) = &mut self.domain {
                    if let MixedDomain::Disjunction(vec) = mixed_domain {
                        vec.push(other);
                    } else {
                        unreachable!("if self_is_intersection is true, this should be unreachable")
                    }
                } else {
                    unreachable!("if self_is_intersection is true, this should be unreachable")
                }

                return self;
            }
            (false, true) => {
                if let DomainType::Mixed(mixed_domain) = &mut other.domain {
                    if let MixedDomain::Disjunction(vec) = mixed_domain {
                        vec.push(self);
                    } else {
                        unreachable!("if other_is_intersection is true, this should be unreachable")
                    }
                } else {
                    unreachable!("if other_is_intersection is true, this should be unreachable")
                }

                return other;
            }
            (false, false) => {}
        }

        // If neither of the 2 domains is a mixed domain of the Disjunction variant,
        // just create a new muxed union domain.

        let domain_type: DomainType =
            DomainType::Mixed(MixedDomain::Disjunction(Box::new(vec![self, other])));
        Domain {
            domain: domain_type,
        }
    }

    /// Transforms the [Domain] to it's complement. All previously accepted
    /// values are now rejected and all previously rejected values are now
    /// accepted.
    pub fn inverse_domain(self) -> Self {
        // First we will check if it is a double negation
        let double_negation: bool = if let DomainType::Mixed(mixed_domain) = &self.domain {
            if let MixedDomain::Not(_) = mixed_domain {
                true
            } else {
                false
            }
        } else {
            false
        };

        if double_negation {
            //Here we do it again but taking the value
            if let DomainType::Mixed(mixed_domain) = self.domain {
                if let MixedDomain::Not(inner_domain) = mixed_domain {
                    return Domain {
                        domain: *inner_domain,
                    };
                } else {
                    unreachable!();
                }
            } else {
                unreachable!();
            }
        }

        // not double negation

        let negated_domain: DomainType = DomainType::Mixed(MixedDomain::Not(Box::new(self.domain)));

        return Domain {
            domain: negated_domain,
        };
    }

    /// Determines if a [Domain] contains a value or not.
    pub fn contains(&self, x: f64) -> bool {
        if x.is_infinite() || x.is_nan() {
            return false;
        }

        match &self.domain {
            DomainType::Discrete(discrete_domain) => match discrete_domain {
                DiscreteDomain::Integers => x.fract() == 0.0,
                DiscreteDomain::Positive(include_zero) => {
                    x.fract() == 0.0 && (0.0 < x || (*include_zero && 0.0 <= x))
                }
                DiscreteDomain::Negative(include_zero) => {
                    x.fract() == 0.0 && (x < 0.0 || (x <= 0.0 && *include_zero))
                }
                DiscreteDomain::Range(min, max) => {
                    x.fract() == 0.0 && *min as f64 <= x && x <= *max as f64
                }
                DiscreteDomain::From(min) => x.fract() == 0.0 && *min as f64 <= x,
                DiscreteDomain::To(max) => x.fract() == 0.0 && x <= *max as f64,
                DiscreteDomain::Custom(vec) => {
                    vec.binary_search_by(|a| a.partial_cmp(&x).unwrap()).is_ok()
                }
            },
            DomainType::Continuous(continuous_domain) => match continuous_domain {
                ContinuousDomain::Reals => true,
                ContinuousDomain::ZeroOne => 0.0 <= x && x <= 1.0,
                ContinuousDomain::Positive(include_zero) => 0.0 < x || (*include_zero && 0.0 <= x),
                ContinuousDomain::Negative(include_zero) => x < 0.0 || (*include_zero && x <= 0.0),
                ContinuousDomain::Range(min, max) => *min <= x && x <= *max,
            },
            DomainType::Mixed(mixed_domain) => match mixed_domain {
                MixedDomain::Union(vec) => vec.iter().any(|domain| domain.contains(x)),
                MixedDomain::Disjunction(vec) => vec.iter().all(|domain| domain.contains(x)),
                MixedDomain::Not(domain_type) => {
                    let wrapper: Domain = Domain {
                        domain: *domain_type.clone(),
                    };
                    !wrapper.contains(x)
                }
            },
        }
    }

    /// Determines if a point is contained inside the [Domain] in a continuous region.
    ///
    /// The reason to use this instead of [Domain::contains] is that if you are integrating
    /// in a domain you do not want to count any discrete point the [Domain] may have.
    pub fn contains_continuous(&self, x: f64) -> bool {
        if x.is_infinite() || x.is_nan() {
            return false;
        }

        match &self.domain {
            DomainType::Discrete(_) => false,
            DomainType::Continuous(continuous_domain) => match continuous_domain {
                ContinuousDomain::Reals => true,
                ContinuousDomain::ZeroOne => 0.0 <= x && x <= 1.0,
                ContinuousDomain::Positive(include_zero) => 0.0 < x || (*include_zero && 0.0 <= x),
                ContinuousDomain::Negative(include_zero) => x < 0.0 || (*include_zero && x <= 0.0),
                ContinuousDomain::Range(min, max) => *min <= x && x <= *max,
            },
            DomainType::Mixed(mixed_domain) => match mixed_domain {
                MixedDomain::Union(vec) => vec.iter().any(|domain| domain.contains(x)),
                MixedDomain::Disjunction(vec) => vec.iter().all(|domain| domain.contains(x)),
                MixedDomain::Not(domain_type) => !domain_type.clone().to_domain().contains(x),
            },
        }
    }

    /// Returns true if x is stricly lower than any point inside the domain.
    ///
    /// **Warning**: implementation for [DomainType::Mixed] not correct, may provide incorrect results.
    pub fn is_before_domain(&self, x: f64) -> bool {
        match &self.domain {
            DomainType::Discrete(discrete_domain) => match discrete_domain {
                DiscreteDomain::Integers => false,
                DiscreteDomain::Positive(include_zero) => x < 0.0 || (*include_zero && x <= 0.0),
                DiscreteDomain::Negative(_) => false,
                DiscreteDomain::Range(min, _) => x < *min as f64,
                DiscreteDomain::From(min) => x < *min as f64,
                DiscreteDomain::To(_) => false,
                DiscreteDomain::Custom(vec) => vec
                    .iter()
                    .reduce(|a, b| if *a < *b { a } else { b })
                    .is_some_and(|min| x < *min),
            },
            DomainType::Continuous(continuous_domain) => match continuous_domain {
                ContinuousDomain::Reals => false,
                ContinuousDomain::ZeroOne => x < 0.0,
                ContinuousDomain::Positive(inclue_zero) => x < 0.0 || (*inclue_zero && x <= 0.0),
                ContinuousDomain::Negative(_) => false,
                ContinuousDomain::Range(min, _) => x < *min as f64,
            },
            DomainType::Mixed(mixed_domain) => match mixed_domain {
                MixedDomain::Union(vec) => vec.iter().all(|domain| domain.is_before_domain(x)),
                // Very rough upper bound but correct
                MixedDomain::Disjunction(vec) => {
                    vec.iter().all(|domain| domain.is_before_domain(x))
                }
                MixedDomain::Not(domain_type) => {
                    //todo!("Not correct, fix");
                    !domain_type.clone().to_domain().is_before_domain(x)
                }
            },
        }
    }

    /// Returns true if x is stricly higher than any point inside the domain.
    ///
    /// **Warning**: implementation for [DomainType::Mixed] not correct, may provide incorrect results.
    pub fn is_after_domain(&self, x: f64) -> bool {
        match &self.domain {
            DomainType::Discrete(discrete_domain) => match discrete_domain {
                DiscreteDomain::Integers => false,
                DiscreteDomain::Positive(_) => false,
                DiscreteDomain::Negative(include_zero) => 0.0 < x || (*include_zero && 0.0 <= x),
                DiscreteDomain::Range(_, max) => (*max as f64) < x,
                DiscreteDomain::From(_) => false,
                DiscreteDomain::To(max) => (*max as f64) < x,
                DiscreteDomain::Custom(vec) => vec
                    .iter()
                    .reduce(|a, b| if *b < *a { a } else { b })
                    .is_some_and(|max| *max < x),
            },
            DomainType::Continuous(continuous_domain) => match continuous_domain {
                ContinuousDomain::Reals => false,
                ContinuousDomain::ZeroOne => 1.0 < x,
                ContinuousDomain::Positive(_) => false,
                ContinuousDomain::Negative(inclue_zero) => 0.0 < x || (*inclue_zero && 0.0 <= x),
                ContinuousDomain::Range(_, max) => (*max as f64) < x,
            },
            DomainType::Mixed(mixed_domain) => match mixed_domain {
                MixedDomain::Union(vec) => vec.iter().all(|domain| domain.is_after_domain(x)),
                // Very rough upper bound but correct
                MixedDomain::Disjunction(vec) => vec.iter().all(|domain| domain.is_after_domain(x)),
                MixedDomain::Not(domain_type) => {
                    //todo!("Not correct, fix");
                    !domain_type.clone().to_domain().is_after_domain(x)
                }
            },
        }
    }

    /// Returns the upper and lower bounds of the domain.
    ///
    /// Take into account that the values can also include positive and negative infinity.
    /// It is guaranteed that return.0 <= return.1. If the bounds are finite, the values
    /// themselves are included.
    ///
    /// If the domain is empty, [DEFAULT_EMPTY_DOMAIN_BOUNDS] `(-0.0, 0.0)` is returned.
    pub fn get_bounds(&self) -> (f64, f64) {
        match &self.domain {
            DomainType::Discrete(discrete_domain) => match discrete_domain {
                DiscreteDomain::Integers => (f64::NEG_INFINITY, f64::INFINITY),
                DiscreteDomain::Positive(include_zero) => {
                    let low: f64 = if *include_zero { 0.0 } else { 1.0 };
                    (low, f64::INFINITY)
                }
                DiscreteDomain::Negative(include_zero) => {
                    let high: f64 = if *include_zero { 0.0 } else { -1.0 };
                    (f64::NEG_INFINITY, high)
                }
                DiscreteDomain::Range(min, max) => (*min as f64, *max as f64),
                DiscreteDomain::From(min) => (*min as f64, f64::INFINITY),
                DiscreteDomain::To(max) => (f64::NEG_INFINITY, *max as f64),
                DiscreteDomain::Custom(vec) => {
                    if vec.is_empty() {
                        DEFAULT_EMPTY_DOMAIN_BOUNDS
                    } else {
                        (*vec.first().unwrap(), *vec.last().unwrap())
                    }
                }
            },
            DomainType::Continuous(continuous_domain) => match continuous_domain {
                ContinuousDomain::Reals => (f64::NEG_INFINITY, f64::INFINITY),
                ContinuousDomain::ZeroOne => (0.0, 1.0),
                ContinuousDomain::Positive(include_zero) => (
                    if *include_zero {
                        0.0
                    } else {
                        0.0 + f64::EPSILON
                    },
                    f64::INFINITY,
                ),
                ContinuousDomain::Negative(include_zero) => (
                    f64::NEG_INFINITY,
                    if *include_zero {
                        0.0
                    } else {
                        0.0 - f64::EPSILON
                    },
                ),
                ContinuousDomain::Range(min, max) => (*min, *max),
            },
            DomainType::Mixed(mixed_domain) => match mixed_domain {
                MixedDomain::Union(vec) => vec
                    .iter()
                    .map(|domain| domain.get_bounds())
                    .reduce(|acc, d| {
                        let low: f64 = if acc.0 < d.0 { acc.0 } else { d.0 };
                        let high: f64 = if d.0 < acc.0 { acc.0 } else { d.0 };
                        (low, high)
                    })
                    .unwrap_or(DEFAULT_EMPTY_DOMAIN_BOUNDS),
                MixedDomain::Disjunction(vec) => vec
                    .iter()
                    .map(|domain| domain.get_bounds())
                    .reduce(|acc, d| {
                        let low: f64 = if d.0 < acc.0 { acc.0 } else { d.0 };
                        let high: f64 = if acc.0 < d.0 { acc.0 } else { d.0 };
                        (low, high)
                    })
                    .unwrap_or(DEFAULT_EMPTY_DOMAIN_BOUNDS),
                MixedDomain::Not(domain_type) => {
                    let reverse_bounds: (f64, f64) = domain_type.clone().to_domain().get_bounds();

                    match (reverse_bounds.0.is_finite(), reverse_bounds.1.is_finite()) {
                        (true, true) => (f64::NEG_INFINITY, f64::INFINITY),
                        (true, false) => (reverse_bounds.1, f64::INFINITY),
                        (false, true) => (f64::NEG_INFINITY, reverse_bounds.0),
                        (false, false) => DEFAULT_EMPTY_DOMAIN_BOUNDS,
                    }
                }
            },
        }
    }
}

impl DomainType {
    fn to_domain(self) -> Domain {
        Domain { domain: self }
    }
}
