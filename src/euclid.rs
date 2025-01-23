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

/// Deafult return value if a domain is empty
pub const DEFAULT_EMPTY_DOMAIN_BOUNDS: (f64, f64) = (-0.0, 0.0);

/// Integrate a function it it's whole domain.
/// Can be used to determine the normalitzation constant of a pdf.
///
/// This function assumes that `pdf` contains a finite area in it's `domain`.
///
/// You need to divide the value given by `pdf` by the returned value in order to have
/// a valid probability distribution.
///
/// If you already created a distribution `d`, and want to make sure it integrates
/// to `1.0`, you can use:
/// ```
/// let c: f64 = euclid::numerical_integration(|x| d.pdf(x), d.get_domain());
/// assert!((c - 1.0).abs() < 0.0001)
/// ```
pub fn numerical_integration(pdf: impl Fn(f64) -> f64, domain: &ContinuousDomain) -> f64 {
    /*
           Plan:

       To compute integrals over an infinite range, we will perform a special
       [numerial integration](https://en.wikipedia.org/wiki/Numerical_integration#Integrals_over_infinite_intervals).
       (change of variable)

           For a (const) to infinite:
        integral {a -> inf} f(x) dx =
                    integral {0 -> 1} f(a + t/(1 - t))  /  (1 - t)^2  dt
        let u = 1/(1-t);
                    integral {0 -> 1} f(a + t * u) * u * u   dt

           For -infinite to a (const):
        integral {-inf -> a} f(x) dx =
                    integral {0 -> 1} f(a - (1 - t)/t)  /  t^2  dt
        let u = 1/t;
                    integral {0 -> 1} f(a - (1 - t) * u) * u * u  dt

           For -infinite to infinite:
        integral {-inf -> inf} f(x) dx =
                   integral {-1 -> 1} f( t / (1-t^2) ) * (1 + t^2) / (1 - t^2)^2  dt
        let u = 1/(1-t^2);
                   integral {-1 -> 1} f( t * u ) ) * (1 + t^2) * u * u  dt

    */
    let mut ret: f64 = -0.0;

    let bounds: (f64, f64) = domain.get_bounds();
    let integration_type: IntegrationType = IntegrationType::from_bounds(bounds);
    let (step_length, max_iters): (f64, usize) = choose_integration_precision_and_steps(bounds);
    let half_step_length: f64 = 0.5 * step_length;
    let step_len_over_6: f64 = step_length / 6.0;

    let mut num_step: f64 = 0.0;

    // estimate the bound value with the next 2 values
    let mut last_pdf_evaluation: f64 = match integration_type {
        IntegrationType::Finite | IntegrationType::ConstToInfinite => {
            let middle: f64 = pdf(bounds.0 + half_step_length);
            let end: f64 = pdf(bounds.0 + step_length);
            2.0 * middle - end
        }
        IntegrationType::InfiniteToConst => {
            let middle: f64 = pdf(bounds.1 - half_step_length);
            let end: f64 = pdf(bounds.1 - step_length);
            2.0 * middle - end
        }
        IntegrationType::FullInfinite => 0.0,
    };

    for _i in 0..max_iters {
        let current_position: f64;

        let (middle, end): (f64, f64) = match integration_type {
            IntegrationType::Finite => {
                current_position = bounds.0 + step_length * num_step;
                let _middle: f64 = pdf(current_position + half_step_length);
                let _end: f64 = pdf(current_position + step_length);
                (_middle, _end)
            }
            IntegrationType::ConstToInfinite => {
                // integral {a -> inf} f(x) dx = integral {0 -> 1} f(a + t/(1 - t))  /  (1 - t)^2  dt

                current_position = step_length * num_step;

                let _middle: f64 = 'mid: {
                    let t: f64 = current_position + half_step_length;
                    let e: f64 = 1.0 - t;
                    if e.abs() < f64::EPSILON {
                        break 'mid 0.0;
                        // todo: implement something better here
                    }
                    let u: f64 = 1.0 / e; // 1/(1-t)
                    pdf(bounds.0 + t * u) * u * u
                };
                let _end: f64 = 'end: {
                    let t: f64 = current_position + step_length;
                    let e: f64 = 1.0 - t;
                    if e.abs() < f64::EPSILON {
                        break 'end 0.0;
                        // todo: implement something better here
                    }
                    let u: f64 = 1.0 / e; // = 1/(1 - t)
                    pdf(bounds.0 + t * u) * u * u
                };
                (_middle, _end)
            }
            IntegrationType::InfiniteToConst => {
                // integral {-inf -> a} f(x) dx = integral {0 -> 1} f(a - (1 - t)/t)  /  t^2  dt
                // this integral is done in "reverse". Form 1 to 0.

                current_position = 1.0 - step_length * num_step;

                let _middle: f64 = 'mid: {
                    let t: f64 = current_position - half_step_length;
                    let u: f64 = 1.0 / t;
                    if u.is_infinite() {
                        break 'mid 0.0;
                        // todo: implement something better here
                    }
                    pdf(bounds.1 - (1.0 - t) * u) * u * u
                };
                let _end: f64 = 'end: {
                    let t: f64 = current_position - step_length;
                    let u: f64 = 1.0 / t;
                    if u.is_infinite() {
                        break 'end 0.0;
                        // todo: implement something better here
                    }
                    pdf(bounds.1 - (1.0 - t) * u) * u * u
                };
                (_middle, _end)
            }
            IntegrationType::FullInfinite => {
                // integral {-inf -> inf} f(x) dx = integral {-1 -> 1} f( t / (1-t^2) ) * (1 + t^2) / (1 - t^2)^2  dt

                current_position = -1.0 + step_length * num_step;

                let _middle: f64 = {
                    let t: f64 = current_position + half_step_length;
                    let u: f64 = 1.0 / (1.0 - t * t);
                    let v: f64 = 1.0 + t * t;
                    pdf(t * u) * v * u * u
                };
                let _end: f64 = {
                    let t: f64 = current_position + step_length;
                    let e: f64 = 1.0 - t * t;
                    if e.abs() < f64::EPSILON {
                        0.0
                    } else {
                        let u: f64 = 1.0 / e; // =1/(1-t^2)
                        let v: f64 = 1.0 + t * t;
                        pdf(t * u) * v * u * u
                    }
                };
                (_middle, _end)
            }
        };

        ret += step_len_over_6 * (last_pdf_evaluation + 4.0 * middle + end);

        last_pdf_evaluation = end;
        num_step += 1.0;
    }

    return ret;
}

/// Sum all the discrete values in a distribution.
/// Can be used to determine the total probability of a pmf.
///
/// If you want to have a pmf wich represents a valid probability distribution,
/// the result of this function shouls be one. If it is not, you can divise the
/// result of the pmf by the returned value in order to nomalize it.
///
/// This function assumes that `pmf` contains a finite area in it's `domain`.
///
/// **Warning:** if `max_steps` is set to None and the domain contains infinitely
/// many values, the function will not terminate (infinite loop).
pub fn discrete_integration(
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

/// Numerical integration but for a finite range.
///
/// Numerical integration for a function `func` within a finite range. The
/// integration is performed with
/// [Simpson's rule](https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_1/3_rule).
pub fn numerical_integration_finite(
    func: impl Fn(f64) -> f64,
    integration_range: (f64, f64),
    num_steps: u64,
) -> f64 {
    let mut ret: f64 = -0.0;

    let bounds: (f64, f64) = integration_range;
    let step_length: f64 = (bounds.1 - bounds.0) / num_steps as f64;
    let half_step_length: f64 = 0.5 * step_length;
    let step_len_over_6: f64 = step_length / 6.0;

    let mut num_step: f64 = 0.0;

    let mut last_pdf_evaluation: f64 = func(bounds.0 + f64::EPSILON);

    for _ in 0..num_steps {
        let current_position: f64 = bounds.0 + step_length * num_step;

        let middle: f64 = func(current_position + half_step_length);
        let end: f64 = func(current_position + step_length);

        ret += step_len_over_6 * (last_pdf_evaluation + 4.0 * middle + end);

        last_pdf_evaluation = end;
        num_step += 1.0;
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
/// Guarantees:
///  - `bounds.0 < bounds.1`
///  -
pub fn choose_integration_precision_and_steps(bounds: (f64, f64)) -> (f64, usize) {
    /*
        To select the appropiate step_length (and total_num_steps, indirecly),
        we need to adapt between the possible cases.
        - For standard integration: Use DEFAULT_INTEGRATION_PRECISION unless it's
            too small (if we would do more than DEFAULT_INTEGRATION_MAXIMUM_STEPS)


        num_steps needs to be odd
    */

    match (bounds.0.is_finite(), bounds.1.is_finite()) {
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
            // then [-1, 1] will use the doule.
            (
                SMALL_INTEGRATION_PRECISION,
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

/// Indicates how big is the range to integrate.
///
/// Mainly ised for readability
pub enum IntegrationType {
    /// closed interval: `[a, b]`
    Finite,
    /// interval: `(-inf, a]`
    InfiniteToConst,
    /// interval: `[b, inf)`
    ConstToInfinite,
    /// interval: `(-inf, inf)`
    FullInfinite,
}

impl IntegrationType {
    pub fn from_bounds(bounds: (f64, f64)) -> IntegrationType {
        match (bounds.0.is_finite(), bounds.1.is_finite()) {
            (true, true) => IntegrationType::Finite,
            (true, false) => IntegrationType::ConstToInfinite,
            (false, true) => IntegrationType::InfiniteToConst,
            (false, false) => IntegrationType::FullInfinite,
        }
    }
}
