use std::usize;

use rand::Rng;

///! This script contains the interfaces used to comunicate with the distributions.
use crate::errors::*;
use crate::euclid::*;
use crate::RNG;
use crate::SMALL_INTEGRATION_NUM_STEPS;
use crate::SMALL_INTEGRATION_PRECISION;
use crate::{DEFAULT_INTEGRATION_MAXIMUM_STEPS, DEFAULT_INTEGRATION_PRECISION};

/// The trait for any continuous distribution.
pub trait Distribution {
    //Requiered method:

    /// Evaluates the [PDF](https://en.wikipedia.org/wiki/Probability_density_function)
    /// (Probability Density function) of the distribution at point x.
    /// If the function is evaluated outside the domain of the pdf,
    /// it will return `0.0`.
    ///
    /// The PDF is assumed to be a valid probability distribution. If you are not sure
    /// if the PDF is normalized to have a 1 unit of area under the curve of the pdf, you
    /// can use [crate::euclid::determine_normalitzation_constant_continuous].
    fn pdf(&self, x: f64) -> f64;

    /// Returns a reference to the pdf domain, wich indicates at wich points the pdf can 
    /// be evaluated. 
    fn get_pdf_domain(&self) -> &Domain;

    // Provided methods:
    // Manual implementation for a specific distribution is recommended.

    /// Evaluates the [CDF](https://en.wikipedia.org/wiki/Cumulative_distribution_function)
    /// (Cumulative distribution function).
    /// If the function is evaluated outside the domain of the pdf, it will either 
    /// return either `0.0` or  `1.0`.
    /// 
    /// Note that the deafult implemetation requieres numerical integration and 
    /// may be expensive. 
    fn cdf(&self, x: f64) -> f64 {

        if x.is_nan() {
            return 0.0;
        }

        let domain: &Domain = self.get_pdf_domain();
        let bounds: (f64, f64) = domain.get_bounds();
        let pdf_checked = |x: f64, domain: &Domain| {
            if domain.contains(x) {
                self.pdf(x)
            } else {
                0.0
            }
        };

        // easy cases
        if x <= bounds.0 {
            return 0.0;
        }

        if bounds.1 <= x {
            return 1.0;
        }

        let cumulative_probability: f64;

        // time to integrate. Note how we only care if the lower bound is finite or not.
        // If the upper bound is infinite, it does not matter because `x` is always a
        // smaller number and we can perform numerical integration.
        // However, if the lower bound is -infinite, we need to perform special
        // numerical integration.

        if bounds.0.is_finite() {
            // We will use [Simpson's rule](https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_1/3_rule) for integration.

            let (step_length, total_num_steps): (f64, usize) = {
                let integration_range: f64 = x - bounds.0;
                let alternative_step_length: f64 =
                    integration_range / DEFAULT_INTEGRATION_MAXIMUM_STEPS as f64;
                if DEFAULT_INTEGRATION_PRECISION < alternative_step_length {
                    (alternative_step_length, DEFAULT_INTEGRATION_MAXIMUM_STEPS)
                    // DEFAULT_INTEGRATION_MAXIMUM_STEPS is even
                } else {
                    let number_steps: usize =
                        ((integration_range / DEFAULT_INTEGRATION_PRECISION) as usize) | 1;
                    // x | 1 makes sure number_steps is even
                    let corrected_step_length: f64 = integration_range / (number_steps as f64);
                    (corrected_step_length, number_steps)
                }
            };

            let mut odd_acc: f64 = 0.0;
            let mut even_acc: f64 = 0.0;
            let mut num_step: usize = 1;

            while num_step < total_num_steps {
                // do the remaining even terms.
                let current_positon: f64 = bounds.0 + step_length * num_step as f64;
                let pdf_value: f64 = pdf_checked(current_positon, &domain);
                if (num_step & 1) == 0 {
                    //even
                    even_acc += pdf_value;
                } else {
                    // odd
                    odd_acc += pdf_value;
                }
                num_step += 1;
            }
            /*

            // separate loops version: todo: check wich is faster or better, paralelize execution
            let mut num_step: usize = 1;
            while num_step < total_num_steps {
                // do all the odd terms
                let current_positon: f64 = bounds.0 + step_length * num_step as f64;
                let pdf_value: f64 = pdf_checked(current_positon, &domain);
                odd_acc += pdf_value;
                num_step += 2;
            }

            let mut even_acc: f64 = 0.0;
            num_step = 2;
            while num_step < total_num_steps {
                // do the remaining even terms.
                let current_positon: f64 = bounds.0 + step_length * num_step as f64;
                let pdf_value: f64 = pdf_checked(current_positon, &domain);
                even_acc += pdf_value;
                num_step += 2;
            }
            */

            // Get the final bound values that have not been included
            let bound_values: f64 = pdf_checked(bounds.0, &domain) + pdf_checked(x, &domain);
            // final result
            cumulative_probability =
                step_length / 3.0 * (4.0 * odd_acc + 2.0 * even_acc + bound_values);
        } else {
            // the range is infinite therefore we will perform a special
            // [numerial integration](https://en.wikipedia.org/wiki/Numerical_integration#Integrals_over_infinite_intervals).

            // integral {-inf -> a} f(x) dx = integral {0 -> 1} f(a - (1 - t) / t) / t^2 dt

            // We will use [Simpson's rule](https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_1/3_rule) for integration.
            // to integrate this second integral.

            let mut odd_acc: f64 = 0.0;
            let mut even_acc: f64 = 0.0;

            // we do not include 0 because it createa a singularity and 
            // 0 and SMALL_INTEGRATION_NUM_STEPS are the boundary values
            for num_step in 2..SMALL_INTEGRATION_NUM_STEPS {
                let current_positon: f64 = SMALL_INTEGRATION_PRECISION * num_step as f64;
                // current_position is `t`
                let input: f64 = x - (1.0 - current_positon) / current_positon;
                let pdf_value: f64 = pdf_checked(input, &domain);
                let integrand_value: f64 = pdf_value / (current_positon * current_positon);
                if (num_step & 1) == 0 {
                    //even
                    even_acc += integrand_value;
                } else {
                    // odd
                    odd_acc += integrand_value;
                }
            }

            let first_value: f64 = {
                let current_positon: f64 = SMALL_INTEGRATION_PRECISION;
                let input: f64 = x - (1.0 - current_positon) / current_positon;
                let pdf_value: f64 = pdf_checked(input, &domain);
                pdf_value / (current_positon * current_positon)
            };

            let last_value: f64 = {
                // this last computation simplifies a lot
                /*
                let current_positon: f64 = SMALL_INTEGRATION_PRECISION * SMALL_INTEGRATION_NUM_STEPS;
                    => current_positon = 1.0
                let input: f64 = x - (1.0 - current_positon) / current_positon;
                    => input = x - (1.0 - 1.0) / 1.0;
                    => input = x;
                let pdf_value: f64 = pdf_checked(input, &domain);
                    => pdf_value = pdf_checked(x, &domain);
                let integrand_value: f64 = pdf_value / (current_positon * current_positon);
                    => integrand_value = pdf_value / (1.0 * 1.0);
                    => integrand_value = pdf_value;
                 */
                let pdf_value: f64 = pdf_checked(x, &domain);
                pdf_value
            };

            cumulative_probability = SMALL_INTEGRATION_PRECISION / 3.0
                * (4.0 * odd_acc + 2.0 * even_acc + first_value + last_value);
        }

        return cumulative_probability;
    }

    /// Samples the distribution at random.
    ///
    /// The deafult method is [Inverse transform sampling](https://en.wikipedia.org/wiki/Inverse_transform_sampling)
    /// unless the deadult method is overriden. Inverse transform sampling simply 
    /// generates a random uniform number and evaluates the inverse cdf function 
    /// (the [Distribution::quantile] function) and returns the result. Note that 
    /// the deafult implemetation requieres numerical integration and may be expensive. 
    fn sample(&self) -> f64 {

        let mut rng: rand::prelude::ThreadRng = rand::thread_rng(); 
        let x: f64 = rng.gen::<f64>(); 

        // we know it is not err since we are sure that x is in [0, 1]
        return self.quantile(x).unwrap(); 
    }

    /// Evaluates the [quantile function](https://en.wikipedia.org/wiki/Quantile_function).
    ///  - `x` must be in the range [0.0, 1.0] or an error will be returned.
    ///
    /// The quantile function is the inverse function of [Distribution::cdf]. Note that 
    /// the deafult implemetation requieres numerical integration and may be expensive. 
    fn quantile(&self, x: f64) -> Result<f64, AdvStatError> {

        if x.is_nan() || x < 0.0 || 1.0 < x {
            // x is not valid
            return Err(AdvStatError::DomainErr);
        }

        // To evaluate the quantile function we will integrate the pdf until 
        // the accumulated area == x. Then (for more precision) we will use 1 
        // iteration of Newton's method for more precision. 

        let domain: &Domain = self.get_pdf_domain();
        let bounds: (f64, f64) = domain.get_bounds();

        if x == 0.0 {
            return Ok(bounds.0); 
        }
        if x == 1.0 {
            return Ok(bounds.1);
        }

        let pdf_checked = |x: f64, domain: &Domain| {
            if domain.contains(x) {
                self.pdf(x)
            } else {
                0.0
            }
        };

        match (bounds.0.is_finite(), bounds.1.is_finite()) {
            (true, true) => {
                // We will use [Simpson's rule](https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_1/3_rule) for integration.

                let (step_length, total_num_steps): (f64, usize) = {
                    let integration_range: f64 = x - bounds.0;
                    let alternative_step_length: f64 =
                        integration_range / DEFAULT_INTEGRATION_MAXIMUM_STEPS as f64;
                    if DEFAULT_INTEGRATION_PRECISION < alternative_step_length {
                        (alternative_step_length, DEFAULT_INTEGRATION_MAXIMUM_STEPS)
                        // DEFAULT_INTEGRATION_MAXIMUM_STEPS is even
                    } else {
                        let number_steps: usize =
                            ((integration_range / DEFAULT_INTEGRATION_PRECISION) as usize) | 1;
                        // x | 1 makes sure number_steps is even
                        let corrected_step_length: f64 = integration_range / (number_steps as f64);
                        (corrected_step_length, number_steps)
                    }
                };

                let double_step_length: f64 = 2.0 * step_length; 
                let constant: f64 = step_length / 3.0; 
                
                let mut accumulator: f64 = 0.0; 
                let mut last_pdf_evaluation: f64 = pdf_checked(bounds.0, &domain); 
                let mut num_step: f64 = 0.0; 
                loop {
                    let current_position: f64 = bounds.0 + double_step_length * num_step; 
                    let middle: f64 = pdf_checked(current_position + step_length, domain); 
                    let end: f64 = pdf_checked(current_position + double_step_length, domain); 

                    accumulator += constant * (last_pdf_evaluation + 4.0 * middle + end); 

                    if x <= accumulator {
                        break; 
                    }

                    last_pdf_evaluation = end; 
                    num_step += 1.0; 
                }
                let quantile_0: f64 = bounds.0 + double_step_length * (num_step + 1.0); 

                // we could return here, but if the pdf is "well behaved", we can 
                // use a single iteration of [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method)
                // to get a better aproximation
                let pdf_q: f64 = pdf_checked(quantile_0, domain); 
                if pdf_q.abs() < f64::EPSILON {
                    // pdf_q is essentially 0, skip newton's iteration and return
                    return Ok(quantile_0);
                }
                let quantile_1: f64 = quantile_0 - accumulator / pdf_q; 
                return Ok(quantile_1);

            },
            (true, false) => todo!(),
            (false, true) => todo!(),
            (false, false) => todo!(),
        }
        
        todo!("Implement deafult implementation. ");
    }

    // Multiple variants.
    // They are the same as the normal functions, but if they are overriden they may
    // provide a computational advantage.

    /// cdf_multiple allows to evaluate the [Distribution::cdf] at multiple points.
    /// It may provide a computational advantage.  
    fn cdf_multiple(&self, points: &[f64]) -> Vec<f64> {
        let mut ret: Vec<f64> = Vec::with_capacity(points.len());

        todo!("Implement deafult implementation. ");

        return ret;
    }

    /// sample_multiple allows to evaluate the [Distribution::sample] at multiple points.
    /// It may provide a computational advantage.
    fn sample_multiple(&self, n: usize) -> Vec<f64> {
        let mut ret: Vec<f64> = Vec::with_capacity(n);

        for _i in 0..n {
            ret.push(self.sample());
        }

        return ret;
    }

    /// quantile_multiple acts the same as [Distribution::quantile] but on multiple points.
    /// If this function is overriden, it may provide a computational advantage.
    fn quantile_multiple(&self, points: &[f64]) -> Result<Vec<f64>, AdvStatError> {
        let mut ret: Vec<f64> = Vec::with_capacity(points.len());

        for point in points {
            ret.push(self.quantile(*point)?);
        }

        return Ok(ret);
    }

    // Statistics

    /// Returns the [expected value](https://en.wikipedia.org/wiki/Expected_value)
    /// of the distribution if it exists.
    fn expected_value(&self) -> Option<f64> {
        todo!("Implement deafult implementation. ");
    }

    /// Returns the [variance](https://en.wikipedia.org/wiki/Variance) of
    /// the distribution if it exists.
    fn variance(&self) -> Option<f64> {
        todo!("Implement deafult implementation. ");
    }

    /// Returns the [mode](https://en.wikipedia.org/wiki/Mode_(statistics))
    /// of the distribution.
    fn mode(&self) -> f64 {
        todo!("Implement deafult implementation. ");
    }

    /// Returns the [skewness](https://en.wikipedia.org/wiki/Skewness)
    /// of the distribution.
    fn skewness(&self) -> f64 {
        todo!("Implement deafult implementation. ");
    }

    /// Returns the [kurtosis](https://en.wikipedia.org/wiki/Kurtosis)
    /// of the distribution.
    fn kurtosis(&self) -> f64 {
        todo!("Implement deafult implementation. ");
    }

    /// Returns the [excess kurtosis](https://en.wikipedia.org/wiki/Kurtosis#Excess_kurtosis)
    /// of the distribution.
    ///
    /// The excess kurtosis is defined as `kurtosis - 3`.
    fn excess_kurtosis(&self) -> f64 {
        return self.kurtosis() - 3.0;
    }

    /// Returns the [moment](https://en.wikipedia.org/wiki/Moment_(mathematics))
    /// of the distribution and the given order. Mode determines if the moment will be
    /// [Moments::Raw], [Moments::Central] or [Moments::Standarized].
    fn moments(&self, order: u8, mode: Moments) {
        #![allow(unused_variables)]
        todo!("Implement deafult implementation. ");
    }

    /// Returns the [entropy](https://en.wikipedia.org/wiki/Information_entropy)
    /// of the distribution
    fn entropy(&self) {
        todo!("Implement deafult implementation. ");
    }

    // Other
}
