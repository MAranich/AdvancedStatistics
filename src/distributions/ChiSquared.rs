//! # Chi-Squared distribution
//!
//! The [Chi Squared distribution](https://en.wikipedia.org/wiki/Chi-squared_distribution)
//! is a continuous distribution. It has 1 parameter: the degrees fo freedom (`k`). It
//! represents the distribution of the sum of k iid standard normal random variables.
//!
//! The Chi Squared distribution is a special case of the [Gamma distribution](crate::distributions::Gamma):
//!
//!  > ChiSquared(k) ~ Gamma(a = k/2, theta = 2)
//!
//!
//!

use std::{f64, hint::assert_unchecked, num::NonZero};

use crate::{
    configuration,
    distribution_trait::{Distribution, Parametric},
    domain::ContinuousDomain,
    euclid::{self, digamma, ln_gamma},
};

#[derive(Debug, Clone, PartialEq)]
pub struct ChiSquared {
    degrees_of_freedom: f64,
    normalitzation_constant: f64,
}

pub const CHI_SQUARED_DOMAIN: ContinuousDomain = ContinuousDomain::From(0.0);

impl ChiSquared {
    /// Creates a new [ChiSquared] distribution with parameter
    /// `k` = `degrees_of_freedom`.
    ///
    /// It will return error if `degrees_of_freedom` is 0.
    pub fn new(degrees_of_freedom: u64) -> Result<ChiSquared, ()> {
        if degrees_of_freedom == 0 {
            return Err(());
        }

        let c: f64 = ChiSquared::compute_normalitzation_constant(degrees_of_freedom as f64);

        return Ok(ChiSquared {
            degrees_of_freedom: degrees_of_freedom as f64,
            normalitzation_constant: c,
        });
    }

    /// Creates a new [ChiSquared] distribution with parameter
    /// `k` = `degrees_of_freedom` without checking if it is not 0 (or an integer).
    ///
    /// If the preconditions are not fullfiled, the returned distribution
    /// will be invalid.
    pub unsafe fn new_unchecked(degrees_of_freedom: f64) -> ChiSquared {
        let c: f64 = ChiSquared::compute_normalitzation_constant(degrees_of_freedom);

        return ChiSquared {
            degrees_of_freedom: degrees_of_freedom,
            normalitzation_constant: c,
        };
    }

    pub fn compute_normalitzation_constant(k: f64) -> f64 {
        assert!(0.0 < k);

        /*
            // original code: 
            let d: f64 = k * 0.5;
            return 1.0_f64 / (2.0_f64.powf(d) * gamma(d));

            ***
            
            c = 1/(2^(k*0.5) * gamma(k*0.5))
            ln(c) = ln(1/(2^(k*0.5) * gamma(k*0.5)))
            ln(c) = -ln(2^(k*0.5) * gamma(k*0.5))
            ln(c) = -ln(2^(k*0.5)) - ln(gamma(k*0.5))
            ln(c) = -(k*0.5)*ln(2) - ln_gamma(k*0.5)

            ***
            
            // alternative code: 

            let d: f64 = k * 0.5;
            let ln_c: f64 = -d * f64::consts::LN_2 - euclid::ln_gamma(d); 

            return ln_c.exp();

            // idk if the alternative version is better than the original one. 
         */


        let d: f64 = k * 0.5;
        let ln_c: f64 = -d * f64::consts::LN_2 - euclid::ln_gamma(d); 

        return ln_c.exp();
    }

    /// Get the parameter degrees of freedom
    pub fn get_degrees_of_freedom(&self) -> NonZero<u64> {
        // Safety: we checked it is non-zero in the creation of the struct.
        return unsafe { NonZero::new_unchecked(self.degrees_of_freedom as u64) };
    }

    pub fn get_normalitzation_constant(&self) -> f64 {
        return self.normalitzation_constant;
    }
}

impl Distribution for ChiSquared {
    fn pdf(&self, x: f64) -> f64 {
        // let norm(k) = 1.0 / (2^(k/2)*gamma(k/2))
        // pdf(x | k) = norm(k) * x^(k/2 - 1) * exp(-x/2)
        return x.powf(self.degrees_of_freedom * 0.5 - 1.0) * (-0.5 * x).exp() * self.normalitzation_constant;
    }

    fn get_domain(&self) -> &ContinuousDomain {
        return &CHI_SQUARED_DOMAIN;
    }

    // default cdf, sample and quantile

    fn cdf_multiple(&self, points: &[f64]) -> Vec<f64> {
        if points.is_empty() {
            return Vec::new();
        }

        // return error if NAN is found
        for point in points {
            if point.is_nan() {
                std::panic!("Found NaN in `ChiSquared::cdf_multiple`. \n");
            }
        }

        let mut ret: Vec<f64> = std::vec![0.0; points.len()];
        let bounds: (f64, f64) = (0.0, f64::INFINITY);
        let mut sorted_indicies: Vec<usize> = (0..points.len()).into_iter().collect::<Vec<usize>>();

        sorted_indicies.sort_unstable_by(|&i, &j| {
            let a: f64 = points[i];
            let b: f64 = points[j];
            a.partial_cmp(&b).unwrap()
        });

        let (step_length, max_iters): (f64, usize) =
            euclid::choose_integration_precision_and_steps(bounds, true);
        let half_step_length: f64 = 0.5 * step_length;
        let step_len_over_6: f64 = step_length / 6.0;

        let mut idx_iter: std::vec::IntoIter<usize> = sorted_indicies.into_iter();
        let mut current_index: usize = idx_iter.next().unwrap();
        // ^unwrap is safe

        let mut current_cdf_point: f64 = points[current_index];

        let mut num_step: f64 = 0.0;
        let mut accumulator: f64 = 0.0;

        // estimate the bound likelyhood with the next 2 values
        let mut last_pdf_evaluation: f64 = {
            let middle: f64 = self.pdf(bounds.0 + half_step_length);
            let end: f64 = self.pdf(bounds.0 + step_length);
            2.0 * middle - end
        };

        for _ in 0..max_iters {
            let current_position: f64;

            current_position = bounds.0 + step_length * num_step;
            while current_cdf_point <= current_position {
                ret[current_index] = accumulator;

                // update `current_cdf_point` to the next value or exit if we are done
                match idx_iter.next() {
                    Some(v) => current_index = v,
                    None => return ret,
                }
                current_cdf_point = points[current_index];
            }

            let middle: f64 = self.pdf(current_position + half_step_length);
            let end: f64 = self.pdf(current_position + step_length);

            accumulator += step_len_over_6 * (last_pdf_evaluation + 4.0 * middle + end);

            last_pdf_evaluation = end;
            num_step += 1.0;
        }

        ret[current_index] = accumulator;

        for idx in idx_iter {
            // use all remaining indicies
            ret[idx] = accumulator;
        }

        return ret;
    }

    fn sample_multiple(&self, n: usize) -> Vec<f64> {
        let gamma: super::Gamma::Gamma = crate::distributions::Gamma::Gamma::from_chi_squared(self); 

        return gamma.sample_multiple(n);
    }

    fn quantile_multiple(&self, points: &[f64]) -> Vec<f64> {
        if points.is_empty() {
            return Vec::new();
        }

        // return error if NAN is found
        for point in points {
            if point.is_nan() {
                std::panic!("Found NaN in `ChiSquared::quantile_multiple`. \n");
            }
        }

        let mut ret: Vec<f64> = std::vec![-0.0; points.len()];
        let bounds: (f64, f64) = (0.0, f64::INFINITY);
        let mut sorted_indicies: Vec<usize> = (0..points.len()).into_iter().collect::<Vec<usize>>();

        sorted_indicies.sort_unstable_by(|&i, &j| {
            let a: f64 = points[i];
            let b: f64 = points[j];
            a.partial_cmp(&b).unwrap()
        });

        let (step_length, max_iters): (f64, usize) =
            euclid::choose_integration_precision_and_steps(bounds, true);
        let half_step_length: f64 = 0.5 * step_length;
        let step_len_over_6: f64 = step_length / 6.0;

        let mut idx_iter: std::vec::IntoIter<usize> = sorted_indicies.into_iter();
        let mut current_index: usize = idx_iter.next().unwrap();
        // ^unwrap is safe

        let mut current_quantile: f64 = points[current_index];

        while current_quantile <= 0.0 {
            ret[current_index] = bounds.0;

            // update `current_quantile` to the next value or exit if we are done
            match idx_iter.next() {
                Some(v) => current_index = v,
                None => return ret,
            }
            current_quantile = points[current_index];
        }

        let mut num_step: f64 = 0.0;
        let mut accumulator: f64 = 0.0;

        // estimate the bound value with the next 2 values
        let mut last_pdf_evaluation: f64 = {
            let middle: f64 = self.pdf(bounds.0 + half_step_length);
            let end: f64 = self.pdf(bounds.0 + step_length);
            2.0 * middle - end
        };

        let use_newtons_method: bool = unsafe { configuration::QUANTILE_USE_NEWTONS_ITER };

        'integration_loop: for _ in 0..max_iters {
            let current_position: f64;

            current_position = bounds.0 + step_length * num_step;
            while current_quantile <= accumulator {
                let mut quantile: f64 = current_position;

                let pdf_q: f64 = self.pdf(quantile);
                if use_newtons_method && !(pdf_q.abs() < f64::EPSILON) {
                    // if pdf_q is essentially 0, skip this.
                    // newton's iteration
                    quantile = quantile - (accumulator - current_quantile) / pdf_q;
                }

                ret[current_index] = quantile;

                // update `current_quantile` to the next value or exit if we are done
                match idx_iter.next() {
                    Some(v) => current_index = v,
                    None => return ret,
                }
                current_quantile = points[current_index];
            }

            if bounds.1 <= current_position {
                ret[current_index] = current_position;
                break 'integration_loop;
            }

            let middle: f64 = self.pdf(current_position + half_step_length);
            let end: f64 = self.pdf(current_position + step_length);

            accumulator += step_len_over_6 * (last_pdf_evaluation + 4.0 * middle + end);

            last_pdf_evaluation = end;
            num_step += 1.0;
        }

        ret[current_index] = bounds.1;

        for idx in idx_iter {
            // use all remaining indicies
            ret[idx] = bounds.1;
        }

        return ret;
    }

    fn expected_value(&self) -> Option<f64> {
        return Some(self.degrees_of_freedom);
    }

    fn variance(&self) -> Option<f64> {
        return Some(2.0 * self.degrees_of_freedom);
    }

    fn mode(&self) -> f64 {
        return (self.degrees_of_freedom - 2.0).max(0.0);
    }

    fn median(&self) -> f64 {
        let term: f64 = 1.0 - 2.0 / (9.0 * self.degrees_of_freedom);
        return self.degrees_of_freedom * term * term * term;
    }

    fn skewness(&self) -> Option<f64> {
        return Some((8.0 / self.degrees_of_freedom).sqrt());
    }

    fn kurtosis(&self) -> Option<f64> {
        return self.excess_kurtosis().map(|x: f64| x + 3.0);
    }
    
    fn excess_kurtosis(&self) -> Option<f64> {
        return Some(12.0 / self.degrees_of_freedom);
    }

    fn moments(&self, order: u8, mode: euclid::Moments) -> f64 {
        /*

               Plan:

            Just to the integral. The integral that gives us the moments of order `k` is:

            ```
            integral {a -> b} ( (x - mu) / std )^k * f(x) dx
            ```
             - `k` is the order of the moment
             - `f(x)` is the pdf of the distribution.
             - `a` and `b` are the values that bound the domain of `f(x)`
                    (they can be `a = -inf` and `b = -inf`).
             - `mu` is the mean of the distribution (or `0` if we selected the `Raw` moment)
             - `std` is the standard deviation of the distribution
                    (or `1` if we did not select the `Standarized` moment)


           Distiguish between cases depending on the domain.

           We will integrate using [Simpson's rule](https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_1/3_rule)
           for integration.

           To compute integrals over an infinite range, we will perform a special
           [numerial integration](https://en.wikipedia.org/wiki/Numerical_integration#Integrals_over_infinite_intervals).

            let g(x) = ( (x - mu) / std )^k * f(x)
                For -infinite to const:
            integral {-inf -> a} g(x) dx = integral {0 -> 1} g(a - (1 - t)/t)  /  t^2  dt
            integral {-inf -> a} g(x) dx = integral {0 -> 1} ( (a - (1 - t)/t - mu) / std )^k * f(a - (1 - t)/t)  /  t^2  dt

                For const to infinite:
            integral {a -> inf} g(x) dx  = integral {0 -> 1} g(a + t/(t - 1))  /  (1 - t)^2  dt
            integral {a -> inf} g(x) dx  = integral {0 -> 1} ( (a + t/(t - 1) - mu) / std )^k * f(a + t/(t - 1))  /  (1 - t)^2  dt

                For -infinite to infinite:
            let inp = t/(1 - t^2)
            integral {-inf -> inf} g(x) dx  = integral {-1 -> 1} g(t/(1 - t^2))  *  (1 + t^2) / (1 - t^2)^2  dt
            integral {-inf -> inf} g(x) dx  = integral {-1 -> 1} ( (t/(1 - t^2) - mu) / std )^k * f(t/(1 - t^2))  *  (1 + t^2) / (1 - t^2)^2  dt


        */

        let domain: &ContinuousDomain = self.get_domain();
        let bounds: (f64, f64) = domain.get_bounds();

        // The values of 0.0 and 1.0 have no special meaning. They are not going to be used anyway.
        let (mean, std_dev): (f64, f64) = match mode {
            euclid::Moments::Raw => {
                let mut acc: f64 = 1.0; 
                let mut curr: f64 = self.degrees_of_freedom; 
                loop {
                    acc = acc * curr; 
                    
                    if self.degrees_of_freedom + (2 * order - 2) as f64 <= curr {
                        return acc; 
                    }

                    curr += 2.0; 
                }
            },
            euclid::Moments::Central => (
                self.expected_value()
                    .expect("Tried to compute a central moment but the expected value is undefined. "),
                1.0,
            ),
            euclid::Moments::Standarized => (
                self.expected_value()
                    .expect("Tried to compute a central/standarized moment but the Expected value is undefined. "),
                self.variance().expect("Tried to compute a standarized moment but the variance is undefined. "),
            ),
        };

        // Todo: give better error handling to the above. ^
        // println!("(mean, std_dev): {:?}", (mean, std_dev));

        let order_exp: i32 = order as i32;
        let (minus_mean, inv_std_dev) = (-mean, 1.0 / std_dev.sqrt());
        let (_, num_steps): (f64, usize) = euclid::choose_integration_precision_and_steps(bounds, true);

        let moment: f64 = {
            // integral {a -> inf} f(x) dx  = integral {0 -> 1} f(a + t/(t - 1))  /  (1 - t)^2  dt

            let integration_fn = |x: f64| 'integration: {
                // x will go from 0.0 to 1.0

                let x_minus: f64 = x - 1.0;
                if x_minus.abs() < f64::EPSILON {
                    // too near singularity, skip
                    break 'integration 0.0;
                }

                let u: f64 = 1.0 / x_minus;
                let fn_input: f64 = bounds.0 + x * u;
                let std_inp: f64 = (fn_input + minus_mean) * inv_std_dev;
                std_inp.powi(order_exp) * self.pdf(fn_input) * u * u
            };

            euclid::numerical_integration_finite(integration_fn, bounds, num_steps as u64)
        }; 
        
        return moment;
    }

    fn entropy(&self) -> f64 {
        let term_1: f64 = self.degrees_of_freedom * 0.5; 
        let term_2: f64 = 2.0_f64.ln() + ln_gamma(term_1); 
        let term_3: f64 = (1.0 - term_1) * digamma(term_1); 

        return term_1 + term_2 + term_3; 
    }

}

impl Parametric for ChiSquared {
    /// Evaluates the [PDF](https://en.wikipedia.org/wiki/Probability_density_function)
    /// (Probability Density function) of the Chi Squared distribution at point `x` with
    /// the given `parameters`.
    ///
    /// If follows the same constraits as the normal
    /// [Distribution::pdf]
    /// (or [DiscreteDistribution::pmf])
    /// but also taking the parameters into account.
    ///
    /// ### Parameters for Chi Squared:
    ///
    /// The Chi Squared distribution has only 1 parameter, `degrees_of_freedom`.
    fn general_pdf(&self, x: f64, parameters: &[f64]) -> f64 {
        // let norm(k) = 1.0 / (2^(k/2)*gamma(k/2))
        // pdf(x | k) = norm(k) * x^(k/2 - 1) * exp(-x/2)
        let k: f64 = parameters[0]; 
        let c: f64 = ChiSquared::compute_normalitzation_constant(k); 
        return x.powf(k * 0.5 - 1.0) * (-0.5 * x).exp() * c;
    }

    fn number_of_parameters() -> u16 {
        return 1; 
    }

    fn get_parameters(&self, parameters: &mut [f64]) {
        parameters[0] = self.degrees_of_freedom; 
    }
    
    fn derivative_pdf_parameters(&self, x: f64, parameters: &[f64]) -> Vec<f64> {
        // d/dx ln(f(x)) = f'(x)/f(x)
        // => f(x) * d/dx ln(f(x)) = f'(x)

        // let norm(k) = 1.0 / (2^(k/2)*gamma(k/2))
        // pdf(x | k) = norm(k) * x^(k/2 - 1) * exp(-x/2)
        // pdf(x | k) = 1.0 / (2^(k/2)*gamma(k/2)) * x^(k/2 - 1) * exp(-x/2)

        let mut log_der: Vec<f64> = self.log_derivative_pdf_parameters(x, parameters); 
        unsafe {assert_unchecked(log_der.len() == 2);}; 
        let pdf: f64 = self.general_pdf(x, parameters); 
        log_der[0] *= pdf; 
        log_der[1] *= pdf; 
    
        return log_der; 
    }
    
    fn log_derivative_pdf_parameters(&self, x: f64, parameters: &[f64]) -> Vec<f64> {
        // d/dx ln(f(x)) = f'(x)/f(x)

        // let norm(k) = 1.0 / (2^(k/2)*gamma(k/2))
        // pdf(x | k) = norm(k) * x^(k/2 - 1) * exp(-x/2)
        // pdf(x | k) = 1.0 / (2^(k/2)*gamma(k/2)) * x^(k/2 - 1) * exp(-x/2)
        // ln(pdf(x | k)) = ln(1.0 / (2^(k/2)*gamma(k/2)) * x^(k/2 - 1) * exp(-x/2))
        // ln(pdf(x | k)) = -ln(2^(k/2)) - ln_gamma(k/2) + (k/2 - 1)*ln(x) - x/2
        // ln(pdf(x | k)) = -(k/2)ln(2) - ln_gamma(k/2) + (k/2 - 1)*ln(x) - x/2

        // Reserve a vector with exacly 3 elements
        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(2);

        let k: f64 = parameters[0];

        {
            // ## Log derivative respect to x:
            /*
                d/dx ln(pdf(x | k)) = d/dx -(k/2)ln(2) - ln_gamma(k/2) + (k/2 - 1)*ln(x) + -x/2
                 = 0 - 0 + (k/2 - 1)/x - 1/2
                 = (k/2 - 1)/x - 1/2

            */

            ret.push((k * 0.5 - 1.0) / x - 0.5);
        }

        {
            // ## Log derivative respect to k:
            /*
                d/dk ln(pdf(x | k)) = d/dk -(k/2)ln(2) - ln_gamma(k/2) + (k/2 - 1)*ln(x) + -x/2
                 = -ln(2)/2 - digamma(k/2)/2 + 1/2*ln(x) + 0
                 = -ln(2)*0.5 - digamma(k*0.5)*0.5 + 0.5*ln(x)
                 = 0.5 * (-ln(2) - digamma(k*0.5) + ln(x))

            */

            ret.push(-0.5 * (2.0_f64.ln() - euclid::digamma(k * 0.5) + x.ln()));
        }

        return ret;
    }
    
    fn fit(&self, data: &mut crate::samples::Samples) -> Vec<f64> {
        /*
                Using Maximum Likelyhood estimation:
            Assuming n samples.

            pdf(x | k) = 1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t)
            d/dk ln(pdf(x | k) = 0.5 * (-ln(2) - digamma(k*0.5) + ln(x))
            0 = sumatory{x_i} 0.5 * (-ln(2) - digamma(k*0.5) + ln(x_i))
            0 = n*(-ln(2) - digamma(k*0.5)) + sumatory{x_i} ln(x_i)
            -n*(-ln(2) - digamma(k*0.5)) = sumatory{x_i} ln(x_i)
            ln(2) + digamma(k*0.5) = 1/n * sumatory{x_i} ln(x_i)
            ln(2) + digamma(k*0.5) = mean{x_i}[ ln(x_i) ]
            digamma(k*0.5) = -ln(2) + mean{x_i}[ ln(x_i) ]
            k*0.5 = digamma^-1( -ln(2) + mean{x_i}[ ln(x_i) ] )
            k = 2*digamma^-1( -ln(2) + mean{x_i}[ ln(x_i) ] )

                Implementation: 

            It improve on this there is a approximation for digamma: 

             > digamma(x) ~= ln(x - 0.5)

            The aproximation works well when `2.5 < x`. The maximum absolute error 
            is arround `~0.0100518357` at x = 2.5 and then decreases as `x` increases. 
            That means we can approximate `digamma^-1(x)` with: 

             > digamma^-1(x) ~= exp(x) + 0.5

            The treshold for using this second aproximation will be `digamma(2.5) = 0.70316`. 
            Otherwise we have to use the exact formula. We use the simplified formula iff 
            `0.70316 < -ln(2) + mean{x_i}[ ln(x_i) ]` <=> `0.0100128194 < mean{x_i}[ ln(x_i) ]`. 
            We will do a Newton's Method step with the real digamma values 
            in order to reduce inacuracies. 
            
            If the condition is not fullfilled, we will use Newton's 
            method to get the result. 

            As a final step we will **round** the result to the nearest 
            positive integer. 


                ### Deafult values:

            If there are not enough samples to compute the estimators, then
            this method will return the following deafult probabilities:
             - `degrees_of_freedom`: 1

         */

        let mut parameters: Vec<f64> = Vec::new();
        parameters.reserve_exact(1);

        let log_mean: f64 = match data.log_mean() {
            Some(v) => v,
            None => { 
                parameters.push(1.0);
                return parameters; 
            },
        }; 

        let threshold_approximation: f64 = 0.70316; 

        // digamma(k) = 
        let digamma_k: f64 = -f64::consts::LN_2 + log_mean; 
        
        let k: f64 = if threshold_approximation < digamma_k {
            // we CAN use the approxiamtion

            let approx_k: f64 = digamma_k.exp() + 0.5; 

            let final_approx: f64 = {
                // Extra Newton's method step: 

                let digamma_aprox: f64 = euclid::digamma(approx_k); 
                let der_digamma_aprox: f64 = euclid::fast_trigamma(digamma_aprox); 

                approx_k - (digamma_aprox - digamma_k)/der_digamma_aprox
            }; 

            final_approx * 2.0
        } else {

            // initial guess. 
            let mut k_approx: f64 = threshold_approximation * 0.5; 
            let conv_diff_criteria: f64 = unsafe {
                configuration::maximum_likelihood_estimation::CONVERGENCE_DIFFERENCE_CRITERIA
            };

            let LOWER_APROXIMATION_TRESHOLD: f64 = 0.01;

            loop {

                let num: f64 = if k_approx < LOWER_APROXIMATION_TRESHOLD {
                    euclid::fast_digamma(k_approx) - digamma_k
                } else {
                    euclid::digamma(k_approx) - digamma_k
                }; 

                if num < conv_diff_criteria {
                    break; 
                }

                let den: f64 = euclid::fast_trigamma(k_approx); 

                k_approx = k_approx - num/den; 
            }

            // Do final step with real digamma
            let num: f64 = euclid::digamma(k_approx) - digamma_k; 
            let den: f64 = euclid::fast_trigamma(k_approx); 

            k_approx = k_approx - num/den; 


            k_approx * 2.0
        }; 

        // round to nearest integer and take 1.0 as minimum value. 
        parameters.push(k.round().max(1.0));

        return parameters; 
    }



}
