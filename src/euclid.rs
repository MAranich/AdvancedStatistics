//! Euclid contains uscefull math functions

use rand::Rng;
use std::usize;

use crate::{
    configuration::integration::{
        DEFAULT_INTEGRATION_MAXIMUM_STEPS, DEFAULT_INTEGRATION_MAXIMUM_STEPS_F64,
        DEFAULT_INTEGRATION_MINIMUM_STEPS, DEFAULT_INTEGRATION_MINIMUM_STEPS_F64,
        DEFAULT_INTEGRATION_PRECISION, SMALL_INTEGRATION_NUM_STEPS, SMALL_INTEGRATION_PRECISION,
    },
    domain::{ContinuousDomain, DiscreteDomain},
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

/// Determine the normalitzation constant of a pdf.
///
/// You need to divide the value given by `pdf` by the returned value in order to have
/// a valid probability distribution.
///
/// This function assumes that `pdf` contains a finite area in it's `domain`.
pub fn get_normalitzation_constant_continuous(
    pdf: impl Fn(f64) -> f64,
    domain: &ContinuousDomain,
) -> f64 {

    todo!("redo"); 
}


/// Determine the total probability of a pmf.
///
/// If you want to have a pmf wich represents a valid probability distribution, 
/// the result of this function shouls be one. If it is not, you can divise the 
/// result of the pmf by the returned value in order to nomalize it. 
///
/// This function assumes that `pmf` contains a finite area in it's `domain`.
pub fn get_normalitzation_constant_discrete(
    pmf: impl Fn(f64) -> f64,
    domain: &DiscreteDomain,
    max_steps: Option<usize>,
) -> f64 {

    let mut ret: f64 = 0.0;

    if let Some(max) = max_steps {
        let mut i: usize = 0; 
        for point in domain.iter() {
            ret += pmf(point);
            i += 1; 
            if max <= i {
                break;
            }
        }
    } else {
        for point in domain.iter() {
            ret += pmf(point);
        }
    }

    return ret;
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

/// Deafult return value if a domain is empty
pub const DEFAULT_EMPTY_DOMAIN_BOUNDS: (f64, f64) = (-0.0, 0.0);

/// Indicates how big is the range to integrate. 
/// 
/// Mainly ised for readability
pub enum IntegrationType {
    /// closed interval: `[a, b]`
    Finite,
    /// interval: `[-inf, a]`
    InfiniteToConst,
    /// interval: `[b, inf]`
    ConstToInfinite,
    /// interval: `[-inf, inf]`
    FullInfinite,
}

impl IntegrationType {
    pub fn from_bounds(bounds: (f64, f64)) -> IntegrationType {
        match (bounds.0.is_finite(), bounds.1.is_finite()) {
            (true, true) => IntegrationType::Finite,
            (true, false) => IntegrationType::InfiniteToConst,
            (false, true) => IntegrationType::ConstToInfinite,
            (false, false) => IntegrationType::FullInfinite,
        }
    }
}
