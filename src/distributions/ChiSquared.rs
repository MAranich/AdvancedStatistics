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
    errors::AdvStatError,
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
    pub fn new(degrees_of_freedom: u64) -> Result<ChiSquared, AdvStatError> {
        if degrees_of_freedom == 0 {
            return Err(AdvStatError::InvalidNumber);
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
    /// ## Safety
    ///
    /// If the the following conditions are not fullfiled, the returned distribution
    /// will be invalid.
    ///
    ///  - `degrees_of_freedom` is finite.
    ///  - `degrees_of_freedom` is an integer.
    ///  - `0.0 < degrees_of_freedom`
    ///
    /// ## Note
    ///
    /// We let `degrees_of_freedom` be a float although theoretically it should be
    /// a positive integer. You may try using different non-integer numbers
    /// (and assuming the other conditions are fullfilled), you should get results
    /// without errors. However we do not make any guarantee if `degrees_of_freedom`
    /// is not an integer.
    ///
    #[must_use]
    pub unsafe fn new_unchecked(degrees_of_freedom: f64) -> ChiSquared {
        let c: f64 = ChiSquared::compute_normalitzation_constant(degrees_of_freedom);

        return ChiSquared {
            degrees_of_freedom,
            normalitzation_constant: c,
        };
    }

    #[must_use]
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
    #[must_use]
    pub const fn get_degrees_of_freedom(&self) -> NonZero<u64> {
        // Safety: we checked it is non-zero in the creation of the struct.
        return unsafe { NonZero::new_unchecked(self.degrees_of_freedom as u64) };
    }

    #[must_use]
    pub const fn get_normalitzation_constant(&self) -> f64 {
        return self.normalitzation_constant;
    }
}

impl Distribution for ChiSquared {
    fn pdf(&self, x: f64) -> f64 {
        // let norm(k) = 1.0 / (2^(k/2)*gamma(k/2))
        // pdf(x | k) = norm(k) * x^(k/2 - 1) * exp(-x/2)
        return x.powf(self.degrees_of_freedom * 0.5 - 1.0)
            * (-0.5 * x).exp()
            * self.normalitzation_constant;
    }

    fn get_domain(&self) -> &ContinuousDomain {
        return &CHI_SQUARED_DOMAIN;
    }

    // default cdf, sample and quantile

    fn sample_fill(&self, buffer: &mut [f64]) {
        let gamma: super::Gamma::Gamma = crate::distributions::Gamma::Gamma::from_chi_squared(self);

        gamma.sample_fill(buffer);
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

    fn excess_kurtosis(&self) -> Option<f64> {
        return Some(12.0 / self.degrees_of_freedom);
    }

    fn moments(&self, order: u8, mode: euclid::Moments) -> f64 {
        if let euclid::Moments::Raw = mode {
            let mut acc: f64 = 1.0;
            let mut curr: f64 = self.degrees_of_freedom;
            loop {
                acc = acc * curr;

                if self.degrees_of_freedom + f64::from(2 * order - 2) <= curr {
                    return acc;
                }

                curr += 2.0;
            }
        }

        let (mean, variance): (f64, f64) = match mode {
            euclid::Moments::Raw => unreachable!("Case hadled previously. "),
            euclid::Moments::Central => {
                // SAFETY: ChiSquared will always return the Some variant for expected_value
                let exp_val: f64 = unsafe { self.expected_value().unwrap_unchecked() };
                (exp_val, 1.0)
            }
            euclid::Moments::Standarized => {
                // SAFETY: ChiSquared will always return the Some variant for expected_value
                let exp_val: f64 = unsafe { self.expected_value().unwrap_unchecked() };

                // SAFETY: ChiSquared will always return the Some variant for variance
                let var: f64 = unsafe { self.variance().unwrap_unchecked() };
                (exp_val, var)
            }
        };

        return self.default_moments(order, mean, variance);
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
        // SAFETY: this is always true
        unsafe {
            assert_unchecked(log_der.len() == 2);
        };
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

        let log_mean: f64 = if let Some(v) = data.log_mean() {
            v
        } else {
            parameters.push(1.0);
            return parameters;
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

                approx_k - (digamma_aprox - digamma_k) / der_digamma_aprox
            };

            final_approx * 2.0
        } else {
            const LOWER_APROXIMATION_TRESHOLD: f64 = 0.01;

            // initial guess.
            let mut k_approx: f64 = threshold_approximation * 0.5;

            // SAFETY: should always be safe to only read
            let conv_diff_criteria: f64 = unsafe {
                configuration::maximum_likelihood_estimation::CONVERGENCE_DIFFERENCE_CRITERIA
            };

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

                k_approx = k_approx - num / den;
            }

            // Do final step with real digamma
            let num: f64 = euclid::digamma(k_approx) - digamma_k;
            let den: f64 = euclid::fast_trigamma(k_approx);

            k_approx = k_approx - num / den;

            k_approx * 2.0
        };

        // round to nearest integer and take 1.0 as minimum value.
        parameters.push(k.round().max(1.0));

        return parameters;
    }
}
