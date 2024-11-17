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
    Standarized
}


/// Domain represents a [domain](https://en.wikipedia.org/wiki/Domain_of_a_function) 
/// of a function (the points where it can be evaluated). The Domain struct provides 
/// enough functionality to easly create most of the most common domains. 
#[derive(Clone, PartialEq)]
pub struct Domain {
    domain: DomainType
}

#[derive(Clone, PartialEq)]
pub enum DomainType {
    Discrete(DiscreteDomain), 
    Continuous(ContinuousDomain), 
    Mixed(MixedDomain)
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
    Integers, 
    /// Positive integers
    PositiveIntegers, 
    /// Positive integers **and** 0
    #[default]
    Naturals, 
    /// All the negative numbers (**without** 0)
    NegativeIntegers, 
    /// All negative numbers **including** 0
    NonPositiveIntegers, 
    /// All the numbers in the range [.0, .1] (**both** inclusive). 
    Range(i64, i64), 
    /// All the numbers from the given value onwards. The value **is** included. 
    From(i64), 
    /// All the numbers until the given value. The value **is** included. 
    To(i64), 
    /// Detemine manually at wich points can this function be evaluated
    Custom(Vec<f64>)
}


/// A domain a section of th real numbers. 
#[derive(Debug, Clone, PartialEq, Default)]

pub enum ContinuousDomain {
    /// All real numbers
    #[default]
    Reals, 
    /// Only the positive numbers. The bool determines if 0 is included or not. 
    Positive(bool), 
    /// Only the negative numbers. The bool determines if 0 is included or not. 
    Negative(bool), 
    /// Only the numbers in the range
    Range(f64, f64)
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
    Not(Box<Domain>), 
}
/// Todo implement Debug for MixedDomain, Domain and DomainType

/// Determine the normalitzation constant of a pdf. 
pub fn determine_normalitzation_constant_continuous(pdf: impl Fn(&dyn Distribution, f64) -> f64, domain: &Domain) -> f64 {
    todo!();
}
