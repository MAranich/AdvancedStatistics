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
    pub const fn get_d1(&self) -> f64 {
        return self.d1;
    }

    #[must_use]
    pub const fn get_d2(&self) -> f64 {
        return self.d2;
    }

    #[must_use]
    pub const fn get_normalitzation_constant(&self) -> f64 {
        return self.normalitzation_constant;
    }
}

impl Distribution for F {
    fn pdf(&self, x: f64) -> f64 {
        // norm(d1, d2) = (d1/d2)^(d1/2) / B(d1/2, d2/2)
        // norm(d1, d2) = (d1/d2)^(d1/2) * gamma(d1/2 + d2/2) / (gamma(d1/2) * gamma(d2/2))
        // pdf(x | d1, d2) = norm(d1, d2) * x^(d1/2 - 1) * (1 + d1/d2 * x)^-(d1+d2)/2

        let term_1: f64 = x.powf(self.d1.mul_add(0.5, -1.0));
        let term_2: f64 = (1.0 + self.d1 / self.d2 * x).powf((self.d1 + self.d2) * -0.5);
        return self.normalitzation_constant * term_1 * term_2;
    }

    fn get_domain(&self) -> &crate::domain::ContinuousDomain {
        return &F_DOMAIN;
    }

    // cdf and quantile are default

    fn sample_fill(&self, buffer: &mut [f64]) {
        // By definition we can obtain samples from a chi squared distribution over
        // their degrees of freedom and computing the ratio between the 2
        // https://en.wikipedia.org/wiki/F-distribution

        let n: usize = buffer.len();

        // we will store the numerator samples in the buffer but we will need to allocate more space
        // for the ones in the denominator.
        {
            // SAFETY: if self is valid, then self.d1 is positive and the call is safe
            let chi_num: ChiSquared = unsafe { ChiSquared::new_unchecked(self.d1) };

            chi_num.sample_fill(buffer);
        }

        let chi_den_samples: Vec<f64> = {
            // SAFETY: if self is valid, then self.d2 is positive and the call is safe
            let chi_den: ChiSquared = unsafe { ChiSquared::new_unchecked(self.d2) };

            chi_den.sample_multiple(n)
        };

        let inv_d1: f64 = 1.0 / self.d1;
        let inv_d2: f64 = 1.0 / self.d2;

        for (r, chi) in buffer.iter_mut().zip(chi_den_samples.iter()) {
            let num: f64 = (*r) * inv_d1;
            let den = (*chi) * inv_d2;
            *r = num / den;
        }
    }

    fn expected_value(&self) -> Option<f64> {
        if self.d2 <= 2.0 {
            return None;
        }
        return Some(self.d2 / (self.d2 - 2.0));
    }

    fn variance(&self) -> Option<f64> {
        // relabeling (will get optimized away)
        let n: f64 = self.d1;
        let d: f64 = self.d2;

        if d <= 4.0 {
            return None;
        }

        let a: f64 = d - 2.0;
        let num: f64 = 2.0 * d * d * (a + n);
        let den: f64 = n * a * a * (d - 4.0);
        return Some(num / den);
    }

    fn mode(&self) -> f64 {
        if self.d2 <= 2.0 {
            return f64::NAN;
        }

        return (self.d1 - 2.0) * self.d2 / (self.d1 * (self.d2 + 2.0));
    }

    // default median

    fn skewness(&self) -> Option<f64> {
        // The original expression from https://en.wikipedia.org/wiki/F-distribution
        // maybe could get simplified (joining the 2 sqrt into 1) but we don't know if
        // will provide better results

        // relabeling (will get optimized away)
        let n: f64 = self.d1;
        let d: f64 = self.d2;

        if self.d2 <= 6.0 {
            return None;
        }

        let num: f64 = (2.0 * n + d - 2.0) * (8.0 * (d - 4.0)).sqrt();
        let den: f64 = (d - 6.0) * (n * (n + d - 2.0)).sqrt();

        return Some(num / den);
    }

    fn excess_kurtosis(&self) -> Option<f64> {
        // relabeling (will get optimized away)
        let n: f64 = self.d1;
        let d: f64 = self.d2;

        if d <= 8.0 {
            return None;
        }

        let a: f64 = d - 2.0;
        let num: f64 = 12.0 * n * (5.0 * d - 22.0) * (n + a) + (d - 4.0) * a * a;

        let den: f64 = n * (d - 6.0) * (d - 8.0) * (n + a);

        return Some(num / den);
    }

    // moments: TODO: there is a formula for the moments of the F distribution https://en.wikipedia.org/wiki/F-distribution#Properties

    fn moments(&self, order: u8, mode: euclid::Moments) -> f64 {
        // relabeling (will get optimized away)
        let n: f64 = self.d1;
        let d: f64 = self.d2;

        if d <= 2.0 * f64::from(order) {
            return f64::NAN;
        }

        if let euclid::Moments::Raw = mode {
            // in this case we have a closed form formula: https://en.wikipedia.org/wiki/F-distribution#Properties

            let k: f64 = f64::from(order);

            // computing the gamma terms (logarithmically)
            let gamma_1: f64 = euclid::ln_gamma(n * 0.5 + k);
            let gamma_2: f64 = euclid::ln_gamma(d * 0.5 - k);
            let gamma_3: f64 = euclid::ln_gamma(n * 0.5);
            let gamma_4: f64 = euclid::ln_gamma(d * 0.5);

            let log_result: f64 = k * (d - n) + (gamma_1 - gamma_3) + (gamma_2 - gamma_4);

            return log_result.exp();
        }

        let (mean, variance): (f64, f64) = match mode {
            euclid::Moments::Raw => unreachable!("Case handled before. "),
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

        return self.default_moments(order, mean, variance);
    }

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

        // SAFETY: should always be safe to only read
        let learning_rate: f64 =
            unsafe { crate::configuration::maximum_likelihood_estimation::LEARNING_RATE };
        // SAFETY: should always be safe to only read
        let conv_diff_criteria: f64 = unsafe {
            crate::configuration::maximum_likelihood_estimation::CONVERGENCE_DIFFERENCE_CRITERIA
        };
        // SAFETY: should always be safe to only read
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
