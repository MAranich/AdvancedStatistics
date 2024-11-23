use std::default;

///! Euclid contains uscefull math functions
use crate::distributions::distribution_interface::Distribution;

/// The [moments](https://en.wikipedia.org/wiki/Moment_(mathematics)) of a function
/// are some values that provide information about the shape of the function.
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
#[derive(Clone, PartialEq)]
pub struct Domain {
    domain: DomainType,
}

#[derive(Clone, PartialEq)]
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
    /// Detemine manually at wich points can this function be evaluated
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
#[derive(Clone, PartialEq)]
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
pub fn determine_normalitzation_constant_continuous(
    pdf: impl Fn(&dyn Distribution, f64) -> f64,
    domain: &Domain,
) -> f64 {
    todo!();
}

impl Domain {
    /// A [Domain] composed of all the real numbers in the given range. Note both of the
    /// bounds **are includive**.
    ///
    /// If this domain is for the [Distribution::pdf], we recomend using this over the
    /// real domain and just select the interval where the function is non-zero.
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

    pub const fn new_real() -> Self {
        Domain {
            domain: DomainType::Continuous(ContinuousDomain::Reals),
        }
    }

    pub const fn new_continuous_positives(include_zero: bool) -> Self {
        Domain {
            domain: DomainType::Continuous(ContinuousDomain::Positive(include_zero)),
        }
    }

    pub const fn new_continuous_negatives(include_zero: bool) -> Self {
        Domain {
            domain: DomainType::Continuous(ContinuousDomain::Negative(include_zero)),
        }
    }

    pub const fn new_discrete_integers() -> Self {
        let domain_type: DomainType = DomainType::Discrete(DiscreteDomain::Integers);
        Domain {
            domain: domain_type,
        }
    }

    pub const fn new_discrete_positives(include_zero: bool) -> Self {
        let domain_type: DomainType = DomainType::Discrete(DiscreteDomain::Positive(include_zero));
        Domain {
            domain: domain_type,
        }
    }

    pub const fn new_discrete_negatives(include_zero: bool) -> Self {
        let domain_type: DomainType = DomainType::Discrete(DiscreteDomain::Negative(include_zero));
        Domain {
            domain: domain_type,
        }
    }

    pub const fn new_discrete_from(from_inclusive: i64) -> Self {
        let domain_type: DomainType = DomainType::Discrete(DiscreteDomain::From(from_inclusive));
        Domain {
            domain: domain_type,
        }
    }

    pub const fn new_discrete_to(to_inclusive: i64) -> Self {
        let domain_type: DomainType = DomainType::Discrete(DiscreteDomain::To(to_inclusive));
        Domain {
            domain: domain_type,
        }
    }

    pub const fn new_discrete_range(min_inclusive: i64, max_inclusive: i64) -> Self {
        let domain_type: DomainType =
            DomainType::Discrete(DiscreteDomain::Range(min_inclusive, max_inclusive));
        Domain {
            domain: domain_type,
        }
    }

    pub fn new_discrete_custom(valid_values: &[f64]) -> Self {
        let domain_type: DomainType =
            DomainType::Discrete(DiscreteDomain::Custom(Vec::from(valid_values)));
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
                            unreachable!("if other_is_intersection is true, this should be unreachable")
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
                    x.fract() == 0.0 && ((0.0 < x && !*include_zero) || (0.0 <= x && *include_zero))
                }
                DiscreteDomain::Negative(include_zero) => {
                    x.fract() == 0.0 && ((x < 0.0 && !*include_zero) || (x <= 0.0 && *include_zero))
                }
                DiscreteDomain::Range(min, max) => {
                    x.fract() == 0.0 && *min as f64 <= x && x <= *max as f64
                }
                DiscreteDomain::From(min) => x.fract() == 0.0 && *min as f64 <= x,
                DiscreteDomain::To(max) => x.fract() == 0.0 && x <= *max as f64,
                DiscreteDomain::Custom(vec) => vec.iter().find(|&y| x == *y).is_some(),
            },
            DomainType::Continuous(continuous_domain) => match continuous_domain {
                ContinuousDomain::Reals => true,
                ContinuousDomain::ZeroOne => 0.0 <= x && x <= 1.0,
                ContinuousDomain::Positive(include_zero) => {
                    (*include_zero && 0.0 <= x) || (!*include_zero && 0.0 < x)
                }
                ContinuousDomain::Negative(include_zero) => {
                    (*include_zero && x <= 0.0) || (!*include_zero && x < 0.0)
                }
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
            DomainType::Discrete(discrete_domain) => match discrete_domain {
                DiscreteDomain::Integers => x.fract() == 0.0,
                DiscreteDomain::Positive(include_zero) => {
                    x.fract() == 0.0 && ((0.0 < x && !*include_zero) || (0.0 <= x && *include_zero))
                }
                DiscreteDomain::Negative(include_zero) => {
                    x.fract() == 0.0 && ((x < 0.0 && !*include_zero) || (x <= 0.0 && *include_zero))
                }
                DiscreteDomain::Range(min, max) => {
                    x.fract() == 0.0 && *min as f64 <= x && x <= *max as f64
                }
                DiscreteDomain::From(min) => x.fract() == 0.0 && *min as f64 <= x,
                DiscreteDomain::To(max) => x.fract() == 0.0 && x <= *max as f64,
                DiscreteDomain::Custom(vec) => vec.iter().find(|&y| x == *y).is_some(),
            },
            DomainType::Continuous(_) => false,
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
}
