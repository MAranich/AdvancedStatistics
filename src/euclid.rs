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
pub fn determine_normalitzation_constant_continuous(
    pdf: impl Fn(f64) -> f64,
    domain: &ContinuousDomain,
) -> f64 {
    todo!("redo");
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

pub fn determine_normalitzation_constant_discrete(
    pdf: impl Fn(f64) -> f64,
    domain: &DiscreteDomain,
    max_steps: Option<usize>,
) -> f64 {
    todo!("redo");

    let mut ret: f64 = 0.0;

    for point in domain.iter() {
        ret += pdf(point);
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
