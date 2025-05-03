//! # The F distribution
//!
//! The [F distribution](https://en.wikipedia.org/wiki/F-distribution) is a
//! continuous probability distribution.
//!
//! It has 2 parameters: degrees_of_freedom_1 and degrees_of_freedom_2 (d1 and d2 reps.).
//!
//! If we have 2 IID random variables c_1 and c_2 wich have a [Chi Squared](crate::distributions::ChiSquared)
//! distribution with d1 and d2 degrees of freedom respectively, then:
//!
//! > (c_1 / d_1) / (c_2 / d2)
//!
//! Will have an F distribution. This statisitc plays a key role in the ANOVA test.
//!
//!
//!
//!

use crate::{
    distribution_trait::{Distribution, Parametric},
    domain::ContinuousDomain,
    errors::AdvStatError,
    euclid::{self, digamma, ln_gamma},
};

use super::ChiSquared::ChiSquared;

pub const F_DOMAIN: ContinuousDomain = ContinuousDomain::From(0.0);

#[derive(Debug, Clone, PartialEq)]
pub struct F {
    d1: f64,
    d2: f64,
    normalitzation_constant: f64,
}

impl F {
    /// Creates a new [F] distribution with parameters `d1` and `d2`.
    /// Where `d1` is the degrees of freedom of the numerator and `d2` are
    /// the degrees of freedom of the denominator.
    ///
    ///
    /// It will return error under the following conditions:
    ///  - `d1` is `+-inf` or a NaN
    ///  - `d2` is `+-inf` or a NaN
    ///  - `d1 <= 0.0`
    ///  - `d2 <= 0.0`
    ///  - The values for `d1` and `d2` are too large to model properly
    ///      - This means that a [f64] value is not precise enough.
    ///      - Use [F::new_unchecked] if you don't need to evaluate
    ///         the pdf direcly or indirecly.
    ///
    #[must_use]
    pub fn new(d1: f64, d2: f64) -> Result<F, AdvStatError> {
        if !d1.is_finite() {
            if d1.is_nan() {
                return Err(AdvStatError::NanErr);
            } else if d1.is_infinite() {
                return Err(AdvStatError::InvalidNumber);
            }
        }

        if !d2.is_finite() {
            if d2.is_nan() {
                return Err(AdvStatError::NanErr);
            } else if d2.is_infinite() {
                return Err(AdvStatError::InvalidNumber);
            }
        }

        if d1 <= 0.0 {
            return Err(AdvStatError::InvalidNumber);
        }

        if d2 <= 0.0 {
            return Err(AdvStatError::InvalidNumber);
        }

        let norm: f64 = F::compute_normalitzation_constant(d1, d2);

        if !norm.is_finite() || norm <= 0.0 {
            // we do not have enough precision to do the computations
            return Err(AdvStatError::NumericalError);
        }

        return Ok(F {
            d1,
            d2,
            normalitzation_constant: norm,
        });
    }

    /// Creates a new [F] distribution without checking for correctness
    /// with parameters `d1` and `d2`. Where `d1` is the degrees of
    /// freedom of the numerator and `d2` are the degrees of freedom
    /// of the denominator.
    ///
    /// ## Safety
    ///
    /// If the following conditions are not fullfiled, the returned distribution
    /// will be invalid.
    ///
    ///  - `d1` is finite (no `+-inf` or a NaN)
    ///  - `d2` is finite (no `+-inf` or a NaN)
    ///  - `0.0 < d1`
    ///  - `0.0 < d2`
    ///  - The values for `d1` and `d2` are too large to model properly
    ///      - This means that a [f64] value is not precise enough.
    ///
    #[must_use]
    pub unsafe fn new_unchecked(d1: f64, d2: f64) -> F {
        let norm: f64 = F::compute_normalitzation_constant(d1, d2);

        return F {
            d1,
            d2,
            normalitzation_constant: norm,
        };
    }

    #[must_use]
    fn compute_normalitzation_constant(d1: f64, d2: f64) -> f64 {
        assert!(0.0 < d1);
        assert!(0.0 < d2);

        let num: f64 = (d1 / d2).powf(d1 * 0.5);

        let beta: f64 = euclid::beta_fn(d1 * 0.5, d2 * 0.5);

        return num / beta;
    }

    #[must_use]
    pub fn get_d1(&self) -> f64 {
        return self.d1;
    }

    #[must_use]
    pub fn get_d2(&self) -> f64 {
        return self.d2;
    }

    #[must_use]
    pub fn get_normalitzation_constant(&self) -> f64 {
        return self.normalitzation_constant;
    }
}

impl Distribution for F {
    #[must_use]
    fn pdf(&self, x: f64) -> f64 {
        // norm(d1, d2) = (d1/d2)^(d1/2) / B(d1/2, d2/2)
        // norm(d1, d2) = (d1/d2)^(d1/2) * gamma(d1/2 + d2/2) / (gamma(d1/2) * gamma(d2/2))
        // pdf(x | d1, d2) = norm(d1, d2) * x^(d1/2 - 1) * (1 + d1/d2 * x)^-(d1+d2)/2
        let term_1: f64 = x.powf(self.d1 * 0.5 - 1.0);
        let term_2: f64 = (1.0 + self.d1 / self.d2 * x).powf(-(self.d1 + self.d2) * 0.5);
        return term_1 * term_2 * self.normalitzation_constant;
    }

    #[must_use]
    fn get_domain(&self) -> &crate::domain::ContinuousDomain {
        return &F_DOMAIN;
    }

    // cdf, sample and quantile are default

    #[must_use]
    fn cdf_multiple(&self, points: &[f64]) -> Vec<f64> {
        if points.is_empty() {
            return Vec::new();
        }

        // return error if NAN is found
        for point in points {
            if point.is_nan() {
                std::panic!("Found NaN in `F::cdf_multiple`. \n");
            }
        }

        let mut ret: Vec<f64> = std::vec![0.0; points.len()];
        let bounds: (f64, f64) = (0.0, f64::INFINITY);
        let mut sorted_indicies: Vec<usize> = (0..points.len()).collect::<Vec<usize>>();

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
            let current_position: f64 = bounds.0 + step_length * num_step;

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

    #[must_use]
    fn sample_multiple(&self, n: usize) -> Vec<f64> {
        let chi_num: ChiSquared = unsafe { ChiSquared::new_unchecked(self.d1) };
        let chi_den: ChiSquared = unsafe { ChiSquared::new_unchecked(self.d2) };

        let chi_num_samples: Vec<f64> = chi_num.sample_multiple(n);
        let chi_den_samples: Vec<f64> = chi_den.sample_multiple(n);

        let inv_d1: f64 = 1.0 / self.d1;
        let inv_d2: f64 = 1.0 / self.d2;

        return chi_num_samples
            .iter()
            .zip(chi_den_samples.iter())
            .map(|(&n, &d)| (n * inv_d1) / (d * inv_d2))
            .collect::<Vec<f64>>();
    }

    #[must_use]
    fn quantile_multiple(&self, points: &[f64]) -> Vec<f64> {
        if points.is_empty() {
            return Vec::new();
        }

        // return error if NAN is found
        for point in points {
            if point.is_nan() {
                std::panic!("Found NaN in `F::quantile_multiple`. \n");
            }
        }

        let mut ret: Vec<f64> = std::vec![-0.0; points.len()];
        let bounds: (f64, f64) = (0.0, f64::INFINITY);
        let mut sorted_indicies: Vec<usize> = (0..points.len()).collect::<Vec<usize>>();

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

        let use_newtons_method: bool = unsafe { crate::configuration::QUANTILE_USE_NEWTONS_ITER };

        for _ in 0..max_iters {
            let current_position: f64 = bounds.0 + step_length * num_step;

            while current_quantile <= accumulator {
                let mut quantile: f64 = current_position;

                let pdf_q: f64 = self.pdf(quantile);
                // result of pdf is always finite
                #[allow(clippy::neg_cmp_op_on_partial_ord)]
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

    #[must_use]
    fn expected_value(&self) -> Option<f64> {
        if self.d2 <= 2.0 {
            return None;
        }
        return Some(self.d2 / (self.d2 - 2.0));
    }

    #[must_use]
    fn variance(&self) -> Option<f64> {
        if self.d2 <= 4.0 {
            return None;
        }

        let a: f64 = self.d2 - 2.0;
        let num: f64 = 2.0 * self.d2 * self.d2 * (a + self.d1);
        let den: f64 = self.d1 * a * a * (self.d2 - 4.0);
        return Some(num / den);
    }

    #[must_use]
    fn mode(&self) -> f64 {
        if self.d2 <= 2.0 {
            return f64::NAN;
        }

        return (self.d1 - 2.0) * self.d2 / (self.d1 * (self.d2 + 2.0));
    }

    // default median

    #[must_use]
    fn skewness(&self) -> Option<f64> {
        if self.d2 <= 6.0 {
            return None;
        }

        let num: f64 = (2.0 * self.d1 + self.d2 - 2.0) * (8.0 * (self.d2 - 4.0)).sqrt();
        let den: f64 = (self.d2 - 6.0) * (self.d1 * (self.d1 + self.d2 - 2.0)).sqrt();

        return Some(num / den);
    }

    #[must_use]
    fn kurtosis(&self) -> Option<f64> {
        return self.excess_kurtosis().map(|x| x + 3.0);
    }

    #[must_use]
    fn excess_kurtosis(&self) -> Option<f64> {
        if self.d2 <= 8.0 {
            return None;
        }

        let a: f64 = self.d2 - 2.0;
        let num: f64 = 12.0 * self.d1 * (5.0 * self.d2 - 22.0) * (self.d1 + self.d2 - 2.0)
            + (self.d2 - 4.0) * a * a;

        let den: f64 = self.d1 * (self.d2 - 6.0) * (self.d2 - 8.0) * (self.d1 + self.d2 - 2.0);

        return Some(num / den);
    }

    // moments: TODO: there is a formula for the moments of the F distribution https://en.wikipedia.org/wiki/F-distribution#Properties
    #[must_use]
    fn entropy(&self) -> f64 {
        let d1: f64 = self.d1 * 0.5;
        let d2: f64 = self.d2 * 0.5;

        let log_gammas: f64 = ln_gamma(d1) + ln_gamma(d2) - ln_gamma(d1 + d2);

        let digammas: f64 = (1.0 - d1) * digamma(1.0 + d1) - (1.0 + d2) * digamma(1.0 + d2)
            + (d1 + d2) * digamma(d1 + d2);

        return log_gammas + digammas + (self.d2 / self.d1).ln();
    }
}

impl Parametric for F {
    /// Evaluates the [PDF](https://en.wikipedia.org/wiki/Probability_density_function)
    /// (Probability Density function) of the Chi Squared distribution at point `x` with
    /// the given `parameters`.
    ///
    /// If follows the same constraits as the normal
    /// [Distribution::pdf]
    /// (or [DiscreteDistribution::pmf])
    /// but also taking the parameters into account.
    ///
    /// ### Parameters for the F distribution:
    ///
    /// The [F] distribution has 2 parameters `degrees_freedom_numerator` = `d1` and
    /// `degrees_freedom_denomiator` = `d2`. The order of the `parameters` is:
    ///
    /// > \[degrees_freedom_numerator, degrees_freedom_denomiator\]
    ///
    /// Both must be stricly positive (usually also integers).
    #[must_use]
    fn general_pdf(&self, x: f64, parameters: &[f64]) -> f64 {
        // norm(d1, d2) = (d1/d2)^(d1/2) / B(d1/2, d2/2)
        // norm(d1, d2) = (d1/d2)^(d1/2) * gamma(d1/2 + d2/2) / (gamma(d1/2) * gamma(d2/2))
        // pdf(x | d1, d2) = norm(d1, d2) * x^(d1/2 - 1) * (1 + d1/d2 * x)^-(d1+d2)/2

        assert!(2 <= parameters.len(), "Error: Not enough parameters! ");
        let d1: f64 = parameters[0];
        let d2: f64 = parameters[1];

        let term_1: f64 = x.powf(d1 * 0.5 - 1.0);
        let term_2: f64 = (1.0 + d1 / d2 * x).powf(-(d1 + d2) * 0.5);

        let norm: f64 = F::compute_normalitzation_constant(d1, d2);

        return term_1 * term_2 * norm;
    }

    #[must_use]
    fn number_of_parameters() -> u16 {
        return 2;
    }

    fn get_parameters(&self, parameters: &mut [f64]) {
        assert!(
            2 <= parameters.len(),
            "Error: Not enough space for the parameters. "
        );
        parameters[0] = self.d1;
        parameters[1] = self.d2;
    }

    // deafult derivative_pdf_parameters
    
    #[must_use]
    fn log_derivative_pdf_parameters(&self, x: f64, parameters: &[f64]) -> Vec<f64> {
        // d/dx ln(f(x)) = f'(x)/f(x)

        // norm(d1, d2) = (d1/d2)^(d1/2) * gamma(d1/2 + d2/2) / (gamma(d1/2) * gamma(d2/2))
        // pdf(x | d1, d2) = norm(d1, d2) * x^(d1/2 - 1) * (1 + d1/d2 * x)^-(d1+d2)/2
        // ln(pdf(x | d1, d2)) = ln(norm(d1, d2) * x^(d1/2 - 1) * (1 + d1/d2 * x)^-(d1+d2)/2)
        // ln(pdf(x | d1, d2)) = ln(norm(d1, d2)) + (d1/2 - 1)*ln(x) - ((d1+d2)/2)*ln(1 + d1/d2 * x))
        // ln(norm(d1, d2)) = ln((d1/d2)^(d1/2) * gamma(d1/2 + d2/2) / (gamma(d1/2) * gamma(d2/2)))
        // ln(norm(d1, d2)) = (d1/2)*ln(d1/d2) + ln_gamma(d1/2 + d2/2) - ln_gamma(d1/2) - ln_gamma(d2/2)

        // Reserve a vector with exacly 3 elements
        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(3);

        let d1: f64 = parameters[0];
        let d2: f64 = parameters[1];

        let ratio: f64 = d1 / d2;
        let half_d1: f64 = d1 * 0.5;
        let half_d2: f64 = d2 * 0.5;
        let digamma_mix: f64 = digamma(half_d1 + half_d2);
        let term_1: f64 = 1.0 + ratio * x;
        let ln_term_1: f64 = term_1.ln();

        {
            // ## Log derivative respect to x:
            /*
                d/dx ln(pdf(x | a, t)) = d/dx ln(norm(d1, d2)) + (d1/2 - 1)*ln(x) - ((d1+d2)/2)*ln(1 + d1/d2 * x))
                 = 0 + (d1/2 - 1)/x - ((d1+d2)/2)/(1 + d1/d2 * x) * (d1/d2)
                 = (d1/2 - 1)/x - (d1*(d1+d2)/(d2*2))/(d2/d2 + d1/d2 * x)
                 = (d1/2 - 1)/x - (d1*(d1+d2)/2)/(d2 + d1 * x)

            */

            // = (d1/2 - 1)/x
            let term_1: f64 = (d1 * 0.5 - 1.0) / x;

            // = -(d1*(d1+d2)/2)/(d2 + d1 * x)
            let term_2: f64 = -(d1 * (d1 + d2) * 0.5) / (d2 + d1 * x);

            ret.push(term_1 + term_2);
        }

        {
            // ## Log derivative respect to d1:
            /*
                d/d_d1 ln(pdf(x | d1, d2)) = d/d_d1 ln(norm(d1, d2)) + (d1/2 - 1)*ln(x) - ((d1+d2)/2)ln(1 + d1/d2 * x))
                 = d/d_d1[ ln(norm(d1, d2)) ] + ln(x)/2 - (d/d_d1[ (d1+d2)/2) ]*ln(1 + d1/d2 * x) + (d1+d2)/2)*d/d_d1[ ln(1 + d1/d2 * x) ])
                 = d/d_d1[ ln(norm(d1, d2)) ] + ln(x)/2 - (0.5*ln(1 + d1/d2 * x) + (d1+d2)/2)/(1 + d1/d2 * x) * d/d_d1[ 1 + d1/d2 * x ])
                 = d/d_d1[ ln(norm(d1, d2)) ] + ln(x)/2 - (0.5*ln(1 + d1/d2 * x) + (d1+d2)/(1 + d1/d2 * x) * x/(2*d2) )
                 = d/d_d1[ ln(norm(d1, d2)) ] + 0.5 * (ln(x) - ln(1 + d1/d2 * x) + (d1+d2)/(1 + d1/d2 * x) * x/d2 ))

                d/d_d1[ ln(norm(d1, d2)) ] = d/d_d1[ (d1/2)*ln(d1/d2) + ln_gamma(d1/2 + d2/2) - ln_gamma(d1/2) - ln_gamma(d2/2) ]
                 = d/d_d1[ (d1/2)*ln(d1/d2) ] + d/d_d1[ ln_gamma(d1/2 + d2/2) ] - d/d_d1[ ln_gamma(d1/2) ] - d/d_d1[ ln_gamma(d2/2) ]
                 = (d/d_d1[ (d1/2) ]*ln(d1/d2) + (d1/2)*d/d_d1[ ln(d1/d2) ]) + d/d_d1[ ln_gamma(d1/2 + d2/2) ] - d/d_d1[ ln_gamma(d1/2) ] - d/d_d1[ ln_gamma(d2/2) ]
                 = (0.5*ln(d1/d2) + (d1/2)/(d1/d2) * 1/d2) + digamma(d1/2 + d2/2) * 0.5 - digamma(d1/2) * 0.5
                 = (0.5*ln(d1/d2) + 0.5 * d1*d2/d1 * 1/d2) + digamma(d1/2 + d2/2) * 0.5 - digamma(d1/2) * 0.5
                 = 0.5 * (ln(d1/d2) + 1 + digamma(d1/2 + d2/2) - digamma(d1/2))

                d/d_d1 ln(pdf(x | a, t)) = d/d_d1[ ln(norm(d1, d2)) ] + ln(x)/2 - (0.5*ln(1 + d1/d2 * x) + (d1+d2)/(1 + d1/d2 * x) * x/(2*d2) )
                d/d_d1 ln(pdf(x | a, t)) = 0.5 * (ln(d1/d2) + 1 + digamma(d1/2 + d2/2) - digamma(d1/2)) + ln(x)/2 - (0.5*ln(1 + d1/d2 * x) + (d1+d2)/(1 + d1/d2 * x) * x/(2*d2) )
            */

            let der_norm: f64 = {
                // = 0.5 * (ln(d1/d2) + 1 + digamma(d1/2 + d2/2) - digamma(d1/2))
                // = 0.5 * (ln(ratio) + 1 + digamma(half_d1 + half_d2) - digamma(half_d1))

                let ln_ratio: f64 = ratio.ln();

                let digammas: f64 = digamma_mix - digamma(half_d1);
                0.5 * (ln_ratio + 1.0 + digammas)
            };

            // 0.5 * (ln(x) - ln(1 + d1/d2 * x) + (d1+d2)/(1 + d1/d2 * x) * x/d2 ))

            let ln_x: f64 = x.ln();

            let term_2: f64 = x * (d1 + d2) / (d2 * term_1);

            ret.push(der_norm + 0.5 * (ln_x - ln_term_1 + term_2));
        }

        {
            // ## Log derivative respect to d2:
            /*
                d/d_d2 ln(pdf(x | a, t)) = d/d_d2 ln(norm(d1, d2)) + (d1/2 - 1)*ln(x) - 0.5*((d1+d2)*ln(1 + d1/d2 * x))
                 = d/d_d2[ ln(norm(d1, d2)) ] + 0 - 0.5*(ln(1 + d1/d2 * x) + (d1+d2)/(1 + d1/d2 * x) * d/d_d2[ 1 + d1/d2 * x ])
                 = d/d_d2[ ln(norm(d1, d2)) ] - 0.5*(ln(1 + d1/d2 * x) + (d1+d2)/(1 + d1/d2 * x) * - d1/d2^2 * x)

                d/d_d2[ ln(norm(d1, d2)) ] = d/d_d2 (d1/2)*ln(d1/d2) + ln_gamma(d1/2 + d2/2) - ln_gamma(d1/2) - ln_gamma(d2/2)
                 = (d1/2)/(d1/d2)*d/d_d2[ d1/d2 ] + digamma(d1/2 + d2/2) * 0.5 + 0 - digamma(d2/2) * 0.5
                 = (d1/2)/(d1/d2)*-d1/d2^2 + digamma(d1/2 + d2/2) * 0.5 - digamma(d2/2) * 0.5
                 = 0.5*d2*d1/d1*-d1/d2^2 + digamma(d1/2 + d2/2) * 0.5 - digamma(d2/2) * 0.5
                 = 0.5*d2*-d1/d2^2 + digamma(d1/2 + d2/2) * 0.5 - digamma(d2/2) * 0.5
                 = 0.5*-d1/d2 + digamma(d1/2 + d2/2) * 0.5 - digamma(d2/2) * 0.5
                 = 0.5*(-d1/d2 + digamma(d1/2 + d2/2) - digamma(d2/2))

            */

            let der_norm: f64 = {
                let digamma_d2: f64 = digamma(half_d2);

                0.5 * (-ratio + digamma_mix - digamma_d2)
            };
            // - 0.5*(ln(1 + d1/d2 * x) + (d1+d2)/(1 + d1/d2 * x) * - d1/d2^2 * x)

            let term_2: f64 = (d1 + d2) * (-d1 / (d2 * d2) * x) / term_1;

            ret.push(der_norm - 0.5 * (ln_term_1 + term_2));
        }

        return ret;
    }

    #[must_use]
    fn fit(&self, data: &mut crate::samples::Samples) -> Vec<f64> {
        /*
                Using Maximum Likelyhood estimation:
            Assuming n samples.

                Estimation of d1:

            norm(d1, d2) = (d1/d2)^(d1/2) * gamma(d1/2 + d2/2) / (gamma(d1/2) * gamma(d2/2))
            pdf(x | d1, d2) = norm(d1, d2) * x^(d1/2 - 1) * (1 + d1/d2 * x)^-(d1+d2)/2
            d/d_d1 ln(pdf(x | d1, d2)) = 0.5 * (ln(d1/d2) + 1 + digamma(d1/2 + d2/2) - digamma(d1/2)) + 0.5 * (ln(x) - ln(1 + d1/d2 * x) + (d1+d2)/(1 + d1/d2 * x) * x/d2 ))
            0 = sumatory{x_i} 0.5 * (ln(d1/d2) + 1 + digamma(d1/2 + d2/2) - digamma(d1/2)) + 0.5 * (ln(x_i) - ln(1 + d1/d2 * x_1) + (d1+d2)/(1 + d1/d2 * x_i) * x_i/d2 ))
            0 = sumatory{x_i} ln(d1/d2) + 1 + digamma(d1/2 + d2/2) - digamma(d1/2) + ln(x_i) - ln(1 + d1/d2 * x_i) + (d1+d2)/(1 + d1/d2 * x_i) * x_i/d2 )
            0 = n * (ln(d1/d2) + 1 + digamma(d1/2 + d2/2) - digamma(d1/2)) + sumatory{x_i} ln(x_i) - ln(1 + d1/d2 * x_i) + (d1+d2)/(1 + d1/d2 * x_i) * x_i/d2 )
            -n * (ln(d1/d2) + 1 + digamma(d1/2 + d2/2) - digamma(d1/2)) = sumatory{x_i} ln(x_i) - ln(1 + d1/d2 * x_i) + (d1+d2)/(d2 + d1 * x_i) * x_i )

            -ln(d1/d2) - 1 - digamma(d1/2 + d2/2) + digamma(d1/2) = mean{x_i}[ ln(x_i) ] - mean{x_i}[ ln(1 + d1/d2 * x_i) ] + (d1+d2) * mean{x_i}[1/(d2 + d1 * x_i) * x_i]


                Estimation of d2:

            norm(d1, d2) = (d1/d2)^(d1/2) * gamma(d1/2 + d2/2) / (gamma(d1/2) * gamma(d2/2))
            pdf(x | d1, d2) = norm(d1, d2) * x^(d1/2 - 1) * (1 + d1/d2 * x)^-(d1+d2)/2
            d/d_d2 ln(pdf(x | d1, d2)) = 0.5*(-d1/d2 + digamma(d1/2 + d2/2) - digamma(d2/2)) - 0.5*(ln(1 + d1/d2 * x) + (d1+d2)/(1 + d1/d2 * x) * - d1/d2^2 * x)
            0 = summatory{x_i} 0.5*(-d1/d2 + digamma(d1/2 + d2/2) - digamma(d2/2)) - 0.5*(ln(1 + d1/d2 * x_i) + (d1+d2)/(1 + d1/d2 * x_i) * - d1/d2^2 * x_i)
            0 = summatory{x_i} -d1/d2 + digamma(d1/2 + d2/2) - digamma(d2/2) - ln(1 + d1/d2 * x_i) + (d1+d2)/(1 + d1/d2 * x_i) * - d1/d2^2 * x_i
            0 = n * (-d1/d2 + digamma(d1/2 + d2/2) - digamma(d2/2)) + summatory{x_i} - ln(1 + d1/d2 * x_i) + (d1+d2)/(1 + d1/d2 * x_i) * - d1/d2^2 * x_i
            -n * (-d1/d2 + digamma(d1/2 + d2/2) - digamma(d2/2)) = summatory{x_i} - ln(1 + d1/d2 * x_i) - (d1+d2)/(d2 + d1 * x_i) * d1/d2 * x_i
            -n * (-d1/d2 + digamma(d1/2 + d2/2) - digamma(d2/2)) = summatory{x_i} - ln(1 + d1/d2 * x_i) - (d1+d2)/(d2 + d1 * x_i) * d1/d2 * x_i

            d1/d2 - digamma(d1/2 + d2/2) + digamma(d2/2) = -mean{x_i}[ ln(1 + d1/d2 * x_i) ] -(d1+d2) * d1/d2 * mean{x_i}[ 1/(d2 + d1 * x_i) * x_i]

            Here we have a system of 2 equations:

            -n * (ln(d1/d2) + 1 + digamma(d1/2 + d2/2) - digamma(d1/2)) = sumatory{x_i} ln(x_i) - ln(1 + d1/d2 * x_i) + (d1+d2)/(d2 + d1 * x_i) * x_i )
            -n * (-d1/d2 + digamma(d1/2 + d2/2) - digamma(d2/2)) = summatory{x_i} - ln(1 + d1/d2 * x_i) - (d1+d2)/(d2 + d1 * x_i) * d1/d2 * x_i

            -n * (ln(d1/d2) + 1 + digamma(d1/2 + d2/2) - digamma(d1/2)) + n * (-d1/d2 + digamma(d1/2 + d2/2) - digamma(d2/2)) = sumatory{x_i} ln(x_i) - ln(1 + d1/d2 * x_i) + (d1+d2)/(d2 + d1 * x_i) * x_i  - (- ln(1 + d1/d2 * x_i) - (d1+d2)/(d2 + d1 * x_i) * d1/d2 * x_i)
            -n * (ln(d1/d2) + 1 + digamma(d1/2 + d2/2) - digamma(d1/2) + d1/d2 - digamma(d1/2 + d2/2) + digamma(d2/2)) = sumatory{x_i}[ ln(x_i) ] - sumatory{x_i} [ ln(1 + d1/d2 * x_i) + (d1+d2)/(d2 + d1 * x_i) * x_i + ln(1 + d1/d2 * x_i) + (d1+d2)/(d2 + d1 * x_i) * x_i * d1/d2 ]
            -n * (ln(d1/d2) + d1/d2 + 1 - digamma(d1/2) + digamma(d2/2)) = sumatory{x_i}[ ln(x_i) ] - sumatory{x_i} [ 2 * ln(1 + d1/d2 * x_i) + (d1+d2)/(d2 + d1 * x_i) * x_i * (1 + d1/d2) ]
            -ln(d1/d2) - d1/d2 - 1 - digamma(d1/2) - digamma(d2/2) = mean{x_i}[ ln(x_i) ] - 2 * mean{x_i} [ ln(1 + d1/d2 * x_i) ] + mean{x_i}[ (d1+d2)/(d2 + d1 * x_i) * x_i * (1 + d1/d2) ]
            -ln(d1/d2) - d1/d2 - 1 - digamma(d1/2) - digamma(d2/2) = mean{x_i}[ ln(x_i) ] - 2 * mean{x_i} [ ln(1 + d1/d2 * x_i) ] + (d1+d2) * (1 + d1/d2) * mean{x_i}[ 1/(d2 + d1 * x_i) * x_i ]

            Here we are stuck and we cannot progress analytically.

            ## Option 1:

            Doing some re-labeling:

            mean_ln(x) = mean{x_i}[ ln(x_i) ]
            f(x, d1, d2) = - 2 * mean{x_i} [ ln(1 + d1/d2 * x_i) ] + (d1+d2) * (1 + d1/d2) * mean{x_i}[ 1/(d2 + d1 * x_i) * x_i ]
            g(d1, d2) = -ln(d1/d2) - d1/d2 - 1 - digamma(d1/2) - digamma(d2/2)

            Here `x` represents the vector containing all the x_i. Then our previous expression becomes:

            g(d1, d2) = mean_ln(x) + f(x, d1, d2)
            0 = mean_ln(x) + f(x, d1, d2) - g(d1, d2)

            And here we can use the gradient descent to get a numerical solution.

            ## Option 2:

            With our (separate) final equations:

            -ln(d1/d2) - 1 - digamma(d1/2 + d2/2) + digamma(d1/2) = mean{x_i}[ ln(x_i) ] - mean{x_i}[ ln(1 + d1/d2 * x_i) ] + (d1+d2) * mean{x_i}[ 1/(d2 + d1 * x_i) * x_i]
            d1/d2 - digamma(d1/2 + d2/2) + digamma(d2/2) = - mean{x_i}[ ln(1 + d1/d2 * x_i) ] - d1/d2 * (d1+d2) * mean{x_i}[ 1/(d2 + d1 * x_i) * x_i]

            Doing some re-labeling:
            mean_ln(x) = mean{x_i}[ ln(x_i) ]
            f1(d1, d2) = -ln(d1/d2) - 1 - digamma(d1/2 + d2/2) + digamma(d1/2)
            g(x, d1, d2) = - mean{x_i}[ ln(1 + d1/d2 * x_i) ]
            f2(d1, d2) = d1/d2 - digamma(d1/2 + d2/2) + digamma(d2/2)
            g2(x, d1, d2) = (d1+d2) * mean{x_i}[ 1/(d2 + d1 * x_i) * x_i]

            Then our equations become:

            f1(d1, d2) = mean_ln(x) + g(x, d1, d2) + g2(x, d1, d2)
            f2(d1, d2) = g(x, d1, d2) - d1/d2 * g2(x, d1, d2)

            Notice that goth g and g2 are evaluated in both equations. We will try to use
            multidimensiona Newton's method on:

            0 = mean_ln(x) + g(x, d1, d2) + g2(x, d1, d2) - f1(d1, d2)
            0 = g(x, d1, d2) - d1/d2 * g2(x, d1, d2) - f2(d1, d2)

            We define:

            F1(x, d1, d2) = mean_ln(x) + g(x, d1, d2) + g2(x, d1, d2) - f1(d1, d2)
            F2(x, d1, d2) = g(x, d1, d2) - d1/d2 * g2(x, d1, d2) - f2(d1, d2)

            And using matrix notation:

            F(x, d1, d2) = [F1(x, d1, d2), F2(x, d1, d2)]

            And the multidimensional Newton's method is:

            x_n+1 = x_n - J^-1 * F(x_n)

            Where J^-1 is the inverse of the Jacobian matrix J(x, d1, d2), wich is defined as:

            J(x, d1, d2) = [[d/d_d1 F1(x, d1, d2), d/d_d2 F1(x, d1, d2)], [d/d_d2 F1(x, d1, d2), d/d_d2 F1(x, d1, d2)]]

            In our notation, [[a, b], [c, d]] should be interpreted as:

            | a | b |
            | c | d |

            To compute the Jacobian we will first compute the necessary derivatives:

            d_d1 mean_ln(x) = d_d2 mean_ln(x) = 0

            d_d1 g(x, d1, d2) = d_d1 - mean{x_i}[ ln(1 + d1/d2 * x_i) ]
             = - mean{x_i}[ d_d1 ln(1 + d1/d2 * x_i) ]
             = - mean{x_i}[ 1/(1 + d1/d2 * x_i) * d_d1 [1 + d1/d2 * x_i] ]
             = - mean{x_i}[ (x_i/d2)/(1 + d1/d2 * x_i) ]
             = - mean{x_i}[ x_i/(d2 + d1 * x_i) ]

            d_d2 g(x, d1, d2) = d_d2 - mean{x_i}[ ln(1 + d1/d2 * x_i) ]
             = - mean{x_i}[ d_d2 ln(1 + d1/d2 * x_i) ]
             = - mean{x_i}[ 1/(1 + d1/d2 * x_i) * d_d2[ 1 + d1/d2 * x_i ] ]
             = - mean{x_i}[ 1/(1 + d1/d2 * x_i) * -d1/d2^2 * x_i ]
             = - mean{x_i}[ -d1 * x_i/(1 + d1/d2 * x_i) * 1/d2^2 ]
             = - mean{x_i}[ -d1 * x_i/(d2 + d1 * x_i) * 1/d2 ]
             = d1/d2 * mean{x_i}[ x_i/(d2 + d1 * x_i) ]

            d_d1 g2(x, d1, d2) = d_d1 (d1+d2) * mean{x_i}[ 1/(d2 + d1 * x_i) * x_i]
             = mean{x_i}[ 1/(d2 + d1 * x_i) * x_i] + (d1+d2) * mean{x_i}[ d/d_d1[ 1/(d2 + d1 * x_i) * x_i ]]
             = mean{x_i}[ 1/(d2 + d1 * x_i) * x_i] + (d1+d2) * mean{x_i}[ -1/(d2 + d1 * x_i)^2 * x_i^2 ]
             = mean{x_i}[ 1/(d2 + d1 * x_i) * x_i] - (d1+d2) * mean{x_i}[ x_i^2/(d2 + d1 * x_i)^2 ]
             = mean{x_i}[ x_i/(d2 + d1 * x_i)] + mean{x_i}[ - (d1+d2) * (x_i/(d2 + d1 * x_i))^2 ]
             = mean{x_i}[ x_i/(d2 + d1 * x_i) - (d1+d2) * (x_i/(d2 + d1 * x_i))^2 ]

            d_d2 g2(x, d1, d2) = d_d2 (d1+d2) * mean{x_i}[ 1/(d2 + d1 * x_i) * x_i]
             = d_d2 (d1+d2) * mean{x_i}[ x_1/(d2 + d1 * x_i) ]
             = mean{x_i}[ x_i/(d2 + d1 * x_i)] + (d1+d2) * mean{x_i}[ - x_i/(d2 + d1 * x_i)^2]
             = mean{x_i}[ x_i/(d2 + d1 * x_i)] + mean{x_i}[ - (d1+d2) * x_i/(d2 + d1 * x_i)^2]
             = mean{x_i}[ x_i/(d2 + d1 * x_i) - (d1+d2) * x_i/(d2 + d1 * x_i)^2]

            d/d_d1 f1(d1, d2) = d/d_d1 -ln(d1/d2) - 1 - digamma(d1/2 + d2/2) + digamma(d1/2)
             = -1/(d1/d2)*-d1/d2^2 + 0 - trigamma(d1/2 + d2/2)*0.5 + trigamma(d1/2)*0.5
             = d2/d1*d1/d2^2 - trigamma(d1/2 + d2/2)*0.5 + trigamma(d1/2)*0.5
             = 1/d2 - trigamma(d1/2 + d2/2)*0.5 + trigamma(d1/2)*0.5

            d/d_d2 f1(d1, d2) = d/d_d2 -ln(d1/d2) - 1 - digamma(d1/2 + d2/2) + digamma(d1/2)
             = -1/d1 - trigamma(d1/2 + d2/2)*0.5 + trigamma(d1/2)*0.5

            d_d1 f2(d1, d2) = d_d1 d1/d2 - digamma(d1/2 + d2/2) + digamma(d2/2)
             = 1/d2 - trigamma(d1/2 + d2/2) * 0.5

            d_d2 f2(d1, d2) = d_d2 d1/d2 - digamma(d1/2 + d2/2) + digamma(d2/2)
             = -d1/d2^2 -trigamma(d1/2 + d2/2)*0.5 + trigamma(d2/2)*0.5


            ### Computing d/d_d1 F1(x, d1, d2):

            d/d_d1 F1(x, d1, d2) = d_d1 mean_ln(x) + g(x, d1, d2) + g2(x, d1, d2) - f1(d1, d2)
             = 0 + (- mean{x_i}[ x_i/(d2 + d1 * x_i) ]) + (mean{x_i}[ x_i/(d2 + d1 * x_i) - (d1+d2) * (x_i/(d2 + d1 * x_i))^2 ]) - (1/d2 - trigamma(d1/2 + d2/2)*0.5 + trigamma(d1/2)*0.5)
             = - mean{x_i}[ x_i/(d2 + d1 * x_i) ] + mean{x_i}[ x_i/(d2 + d1 * x_i) ] - mean{x_i}[ (d1+d2) * (x_i/(d2 + d1 * x_i))^2 ] - 1/d2 + trigamma(d1/2 + d2/2)*0.5 - trigamma(d1/2)*0.5
             = - (d1+d2) * mean{x_i}[ (x_i/(d2 + d1 * x_i))^2 ] - 1/d2 + trigamma(d1/2 + d2/2)*0.5 - trigamma(d1/2)*0.5

            ### Computing d/d_d2 F1(x, d1, d2):

            d/d_d2 F1(x, d1, d2) = d_d2 mean_ln(x) + g(x, d1, d2) + g2(x, d1, d2) - f1(d1, d2)
             = 0 + (d1/d2 * mean{x_i}[ x_i/(d2 + d1 * x_i) ]) + (mean{x_i}[ x_i/(d2 + d1 * x_i) ] + (d1+d2) * mean{x_i}[ - x_i/(d2 + d1 * x_i)^2]) - (-1/d1 - trigamma(d1/2 + d2/2)*0.5 + trigamma(d1/2)*0.5)
             = (d1/d2 + 1) * mean{x_i}[ x_i/(d2 + d1 * x_i) ] + (d1+d2) * mean{x_i}[ - x_i/(d2 + d1 * x_i)^2] + 1/d1 + trigamma(d1/2 + d2/2)*0.5 - trigamma(d1/2)*0.5

            ### Computing d/d_d1 F2(x, d1, d2):

            d/d_d1 F2(x, d1, d2) = d/d_d1 g(x, d1, d2) - d1/d2 * g2(x, d1, d2) - f2(d1, d2)
             = (- mean{x_i}[ x_i/(d2 + d1 * x_i) ]) - (1/d2 * g2(x, d1, d2) + d1/d2 * d/d_d1[ g2(x, d1, d2)] ) - (1/d2 - trigamma(d1/2 + d2/2) * 0.5)
             = - mean{x_i}[ x_i/(d2 + d1 * x_i) ] - (1/d2 * (d1+d2) * mean{x_i}[ x_i/(d2 + d1 * x_i) ] + d1/d2 * (mean{x_i}[ x_i/(d2 + d1 * x_i) - (d1+d2) * (x_i/(d2 + d1 * x_i))^2 ]) ) - 1/d2 + trigamma(d1/2 + d2/2) * 0.5
             = - mean{x_i}[ x_i/(d2 + d1 * x_i) ] - (d1+d2)/d2 * mean{x_i}[ x_i/(d2 + d1 * x_i) ] - d1/d2 * (mean{x_i}[ x_i/(d2 + d1 * x_i - (d1+d2) * (x_i/(d2 + d1 * x_i))^2 ]) ) - 1/d2 + trigamma(d1/2 + d2/2) * 0.5
             = - ((d1+d2)/d2 + 1) * mean{x_i}[ x_i/(d2 + d1 * x_i) ] - d1/d2 * (mean{x_i}[ x_i/(d2 + d1 * x_i - (d1+d2) * (x_i/(d2 + d1 * x_i))^2 ]) ) - 1/d2 + trigamma(d1/2 + d2/2) * 0.5
                                                                                                           //^parenthesis ? )


            TODO: to complete. For now use default implementation

        */

        let d: usize = Self::number_of_parameters() as usize;
        let mut parameters: Vec<f64> = std::vec![0.0; d];
        self.get_parameters(&mut parameters);

        self.parameter_restriction(&mut parameters);

        let learning_rate: f64 =
            unsafe { crate::configuration::maximum_likelihood_estimation::LEARNING_RATE };
        let conv_diff_criteria: f64 = unsafe {
            crate::configuration::maximum_likelihood_estimation::CONVERGENCE_DIFFERENCE_CRITERIA
        };
        let max_iterations: u32 =
            unsafe { crate::configuration::maximum_likelihood_estimation::MAX_ITERATIONS };

        let inv_n: f64 = 1.0 / (data.peek_data().len() as f64);

        let mut gradient: Vec<f64> = std::vec![0.0; d];
        for _ in 0..max_iterations {
            // println!("{}: Parameters: {:?}", i, parameters);

            // set gradient to 0
            gradient = gradient.iter_mut().map(|_| 0.0).collect::<Vec<f64>>();

            for sample in data.peek_data() {
                let local_log_grad: Vec<f64> =
                    self.log_derivative_pdf_parameters(*sample, &parameters);

                for j in 0..d {
                    // ignore derivative respect x
                    gradient[j] += local_log_grad[j + 1];
                }
            }

            let mut total_change: f64 = 0.0;
            for k in 0..d {
                // We want to maximize likelyhood, so we won't put the `-`
                let change: f64 = learning_rate * inv_n * gradient[k];
                parameters[k] += change;
                total_change += change.abs();
            }

            self.parameter_restriction(&mut parameters);

            if total_change < conv_diff_criteria {
                // todo: if parameter_restriction changes something,
                // it is not taken into account in `total_change`.
                // Do something about it.
                break;
            }
        }

        return parameters;
    }
}
