//! # Student T
//!
//! The [Student T distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution#Probability_density_function)
//! is a continuous probability distribution.
//!
//! ### Parameters
//!
//! It has a single parameter, the degrees of freedom (usually denoted by the greek
//! letter `nu`).
//!  - The degrees of freedom is a stricly positive number (usually an integer).
//!  - If `nu = 1` then the distribution is a [Cauchy distribution](crate::distributions::Cauchy).
//!  - If `nu` diverges to infinity, the distribution becomes a [standard normal distribution](crate::distributions::Normal).
//!

use std::f64;

use rand::{Rng, rng};

use crate::{
    configuration,
    distribution_trait::{Distribution, Parametric},
    domain::ContinuousDomain,
    errors::AdvStatError,
    euclid::{self, ln_gamma},
};

use super::{ChiSquared::ChiSquared, Normal::StdNormal};

pub const STUDENT_T_DOMAIN: ContinuousDomain = ContinuousDomain::Reals;

pub struct StudentT {
    degrees_of_freedom: f64,
    normalitzation_constant: f64,
}

impl StudentT {
    /// Create a [StudentT] distribution.
    ///
    /// `degrees_of_freedom` determines how *normal* does the distribution look.
    ///  - Must be finite (no `+-inf` nor NaN)
    ///  - Must be stricly positive (`0.0 < degrees_of_freedom`)
    ///  - Altough we accept a float, `degrees_of_freedom` almost always is an integer.
    ///
    /// ***
    ///
    /// Notes:
    ///  - A [StudentT] distribution with 1 degree of freedom is a [Cauchy distribution](crate::distributions::Cauchy).
    ///  - A [StudentT] distribution with infinite degrees of freedom is a [standard normal distribution](crate::distributions::Normal).
    ///
    pub fn new(degrees_of_freedom: f64) -> Result<StudentT, AdvStatError> {
        if !degrees_of_freedom.is_finite() {
            if degrees_of_freedom.is_nan() {
                return Err(AdvStatError::NanErr);
            }
            // inf
            return Err(AdvStatError::InvalidNumber);
        }

        if degrees_of_freedom <= 0.0 {
            return Err(AdvStatError::InvalidNumber);
        }

        let norm: f64 = Self::compute_normalitzation_constant(degrees_of_freedom);

        return Ok(StudentT {
            degrees_of_freedom,
            normalitzation_constant: norm,
        });
    }

    /// Create a [StudentT] distribution without checking for the corrrectness of the inputs.
    ///
    /// `degrees_of_freedom` determines how *normal* does the distribution look.
    ///
    /// ## Safety
    ///
    /// If the following conditions are not fullfiled, the returned distribution
    /// will be invalid.
    ///
    ///  - Must be finite (no `+-inf` nor NaN)
    ///  - Must be stricly positive (`0.0 < degrees_of_freedom`)
    ///
    /// ***
    ///
    /// ## Notes:
    ///
    ///  - Altough we accept a float, `degrees_of_freedom` almost always is
    ///     an integer for most practical uses.
    ///  - A [StudentT] distribution with 1 degree of freedom is a [Cauchy distribution](crate::distributions::Cauchy).
    ///  - A [StudentT] distribution with infinite degrees of freedom is a [standard normal distribution](crate::distributions::Normal).
    ///      - In practice we do not need a lot of degrees of freedom to be very similar.
    ///
    /// If any of the 2 situations apply to your use case, we recommend using those
    /// distributions direcly since they will be more effitient and precise.
    #[must_use]
    pub unsafe fn new_unchecked(degrees_of_freedom: f64) -> StudentT {
        return StudentT {
            degrees_of_freedom,
            normalitzation_constant: Self::compute_normalitzation_constant(degrees_of_freedom),
        };
    }

    #[must_use]
    pub fn compute_normalitzation_constant(degrees_of_freedom: f64) -> f64 {
        /*

        c = gamma((nu+1)/2) / (sqrt(pi*nu) * gamma(nu/2))
        ln(c) = ln(gamma((nu+1)/2) / (sqrt(pi*nu) * gamma(nu/2)))
        ln(c) = ln_gamma((nu+1)/2) - ln(sqrt(pi*nu)) - ln_gamma(nu/2)
        ln(c) = ln_gamma((nu+1)/2) - 0.5*ln(pi*nu) - ln_gamma(nu/2)

        ***
        // original code:

        let num: f64 = gamma((degrees_of_freedom + 1.0) / 2.0);
        let den_1: f64 = gamma(degrees_of_freedom / 2.0);
        let den_2: f64 = (f64::consts::PI * degrees_of_freedom).sqrt();

        return num / (den_1 * den_2);

         */

        assert!(0.0 < degrees_of_freedom);

        let ln_c: f64 = ln_gamma(degrees_of_freedom * 0.5 + 0.5)
            - ln_gamma(degrees_of_freedom * 0.5)
            - 0.5 * (f64::consts::PI * degrees_of_freedom).ln();

        return ln_c.exp();
    }

    /// Returns the degrees_of_freedom.
    #[must_use]
    pub const fn get_degrees_of_freedom(&self) -> f64 {
        return self.degrees_of_freedom;
    }

    #[must_use]
    pub const fn get_normalitzation_constant(&self) -> f64 {
        return self.normalitzation_constant;
    }
}

impl Distribution for StudentT {
    #[must_use]
    fn pdf(&self, x: f64) -> f64 {
        // let norm = gamma((nu+1)/2) / (sqrt(pi * nu) * gamma(nu/2))
        // pdf(x | nu) = norm(nu) * (1 + x^2 / nu) ^ (-(nu+1)/2)

        let base: f64 = 1.0 + (x * x / self.degrees_of_freedom);
        let exponent: f64 = -0.5 * (self.degrees_of_freedom + 1.0);
        return base.powf(exponent) * self.normalitzation_constant;
    }

    #[must_use]
    fn get_domain(&self) -> &ContinuousDomain {
        return &STUDENT_T_DOMAIN;
    }

    // default cdf, sample and quantile

    #[must_use]
    fn cdf_multiple(&self, points: &[f64]) -> Vec<f64> {
        /*
            optimeize for StudentT
            We will take advantage that [StudentT] is symetric arround 0 to more
            effitiently integrate. So we will integrate from 0 to inf.
            We will use a ConstToInfinity domain.

        */

        if points.is_empty() {
            return Vec::new();
        }

        // return error if NAN is found
        for point in points {
            assert!(
                !point.is_nan(),
                "Found NaN in `cdf_multiple` for StudentT. \n"
            );
        }

        let mut ret: Vec<f64> = std::vec![-0.0; points.len()];
        let mut sorted_indicies: Vec<(usize, bool)> = (0..points.len())
            .map(|i| (i, points[i] < 0.5))
            .collect::<Vec<(usize, bool)>>();

        sorted_indicies.sort_unstable_by(|&i, &j| {
            let a: f64 = if i.1 { 1.0 - points[i.0] } else { points[i.0] };
            let b: f64 = if j.1 { 1.0 - points[j.0] } else { points[j.0] };
            a.partial_cmp(&b).unwrap()
        });

        let (step_length, max_iters): (f64, usize) = if self.degrees_of_freedom < 2.0 {
            //distribution may not have exponential bounded tails or be very diluded in space
            crate::euclid::choose_integration_precision_and_steps(
                (f64::NEG_INFINITY, f64::INFINITY),
                false,
            )
        } else if self.degrees_of_freedom < 10.0 {
            // not problematic, but not close enough to Normal distr
            let aux: (f64, usize) =
                crate::euclid::choose_integration_precision_and_steps((0.0, f64::INFINITY), false);

            // more precision and less steps. Nost of the mass is near 0.0.
            (aux.0 * 0.125, aux.1 / 8)
        } else {
            // We can assume it has a density distribution near the one from a
            // normal distribution.
            crate::euclid::choose_integration_precision_and_steps((0.0, 9.0), false)
        };

        //println!("Cdf params: (step_length: {step_length}, max_iters: {max_iters})");
        //println!("Degreed of freedom t: {}", self.degrees_of_freedom);

        let half_step_length: f64 = 0.5 * step_length;
        let step_len_over_6: f64 = step_length / 6.0;

        let mut idx_iter: std::vec::IntoIter<(usize, bool)> = sorted_indicies.into_iter();
        let (mut current_index, mut is_current_fliped): (usize, bool) = idx_iter.next().unwrap();
        // ^unwrap is safe

        let mut current_cdf_point: f64 = if is_current_fliped {
            1.0 - points[current_index]
        } else {
            points[current_index]
        };

        let mut num_step: f64 = 0.0;
        let mut accumulator: f64 = 0.5;

        // We can take the "bound" since it's well defined
        let mut last_pdf_evaluation: f64 = self.pdf(0.0);

        for _ in 0..max_iters {
            let current_position: f64 = step_length * num_step;

            while current_cdf_point <= current_position {
                ret[current_index] = if is_current_fliped {
                    1.0 - accumulator
                } else {
                    accumulator
                };

                // update `current_cdf_point` to the next value or exit if we are done
                match idx_iter.next() {
                    Some(v) => (current_index, is_current_fliped) = v,
                    None => return ret,
                }
                current_cdf_point = if is_current_fliped {
                    1.0 - points[current_index]
                } else {
                    points[current_index]
                };
            }

            let middle: f64 = self.pdf(current_position + half_step_length);
            let end: f64 = self.pdf(current_position + step_length);

            accumulator += step_len_over_6 * (last_pdf_evaluation + end + 4.0 * middle);

            last_pdf_evaluation = end;
            num_step += 1.0;
        }

        ret[current_index] = if is_current_fliped { 0.0 } else { 1.0 };

        for idx in idx_iter {
            // use all remaining indicies
            ret[idx.0] = if idx.1 { 0.0 } else { 1.0 };
        }

        return ret;
    }

    #[must_use]
    fn sample_multiple(&self, n: usize) -> Vec<f64> {
        // default sample_multiple, I have not found any better method than the
        // Inverse transform sampling or rejection sampling
        // Found: if Z is std_normal and V is chi^2 with nu deg.free., then
        // T = Z * sqrt(nu/V) is distributed as a StudentT distr. with nu deg.free.
        // ^is this better than inverse transform sampling?
        // To be researched and implemented.

        let std_norm: StdNormal = StdNormal::new();
        // SAFETY: `degrees_of_freedom` is always positive
        let chi_sq: ChiSquared = unsafe { ChiSquared::new_unchecked(self.degrees_of_freedom) };

        let normal_samples: Vec<f64> = std_norm.sample_multiple(n);
        let chi_sq_samples: Vec<f64> = chi_sq.sample_multiple(n);

        let d: f64 = self.degrees_of_freedom;

        return normal_samples
            .iter()
            .zip(chi_sq_samples.iter())
            .map(|(&n, &c)| n * (d / c).sqrt())
            .collect::<Vec<f64>>();
    }

    #[must_use]
    fn quantile_multiple(&self, points: &[f64]) -> Vec<f64> {
        /*
            Optimize for Full infinite domain
            We will take advantage that [StudentT] is symetric arround 0 to more
            effitiently integrate. So we will integrate from 0 to inf.
            We will use a ConstToInfinity domain.

        ***
        for dof = 2: Q(0.99999) ~= 225

        */

        if points.is_empty() {
            return Vec::new();
        }

        // return error if NAN is found
        for point in points {
            assert!(
                !point.is_nan(),
                "Found NaN in `quantile_multiple` for StudentT. \n"
            );
        }

        let mut ret: Vec<f64> = std::vec![-0.0; points.len()];
        let mut sorted_indicies: Vec<(usize, bool)> = (0..points.len())
            .map(|i| (i, points[i] < 0.5))
            .collect::<Vec<(usize, bool)>>();

        sorted_indicies.sort_unstable_by(|&i, &j| {
            let a: f64 = if i.1 { 1.0 - points[i.0] } else { points[i.0] };
            let b: f64 = if j.1 { 1.0 - points[j.0] } else { points[j.0] };
            a.partial_cmp(&b).unwrap()
        });

        let bounds: (f64, f64) = (f64::NEG_INFINITY, f64::INFINITY);
        let (step_length, max_iters): (f64, usize) = if self.degrees_of_freedom < 2.0 {
            //distribution may not have exponential bounded tails or be very diluded in space
            crate::euclid::choose_integration_precision_and_steps(
                (f64::NEG_INFINITY, f64::INFINITY),
                false,
            )
        } else if self.degrees_of_freedom < 10.0 {
            // not problematic, but not close enough to Normal distr
            let aux: (f64, usize) =
                crate::euclid::choose_integration_precision_and_steps((0.0, f64::INFINITY), false);

            // more precision and less steps. Nost of the mass is near 0.0.
            (aux.0 * 0.125, aux.1 / 8)
        } else {
            // We can assume it has a density distribution near the one from a
            // normal distribution.
            crate::euclid::choose_integration_precision_and_steps((0.0, 12.0), false)
        };

        let half_step_length: f64 = 0.5 * step_length;
        let step_len_over_6: f64 = step_length / 6.0;

        let mut idx_iter: std::vec::IntoIter<(usize, bool)> = sorted_indicies.into_iter();
        let (mut current_index, mut is_current_fliped): (usize, bool) = idx_iter.next().unwrap();
        // ^unwrap is safe

        let mut current_quantile: f64 = if is_current_fliped {
            1.0 - points[current_index]
        } else {
            points[current_index]
        };

        let mut num_step: f64 = 0.0;
        let mut accumulator: f64 = 0.5;

        // We can take the "bound" since it's well defined
        let mut last_pdf_evaluation: f64 = self.pdf(0.0);
        // SAFETY: should always be safe to only read
        let use_newtons_method: bool = unsafe { crate::configuration::QUANTILE_USE_NEWTONS_ITER };

        for _ in 0..max_iters {
            let current_position: f64 = step_length * num_step;
            while current_quantile < accumulator {
                let mut quantile: f64 = current_position;

                let pdf_q: f64 = self.pdf(quantile);

                // pdf should always return a finite value
                #[allow(clippy::neg_cmp_op_on_partial_ord)]
                if use_newtons_method && !(pdf_q.abs() < f64::EPSILON) {
                    // if pdf_q is essentially 0, skip this.
                    // newton's iteration
                    // q_n+1 = q_n - (cdf(q_n) - q)/pdf(q_n)
                    quantile = quantile - (accumulator - current_quantile) / pdf_q;
                }

                ret[current_index] = if is_current_fliped {
                    -quantile
                } else {
                    quantile
                };

                // update `current_quantile` to the next value or exit if we are done
                match idx_iter.next() {
                    Some(v) => (current_index, is_current_fliped) = v,
                    None => return ret,
                }
                current_quantile = if is_current_fliped {
                    1.0 - points[current_index]
                } else {
                    points[current_index]
                };
            }

            let middle: f64 = self.pdf(current_position + half_step_length);
            let end: f64 = self.pdf(current_position + step_length);

            accumulator += step_len_over_6 * (last_pdf_evaluation + end + 4.0 * middle);

            last_pdf_evaluation = end;
            num_step += 1.0;
        }

        ret[current_index] = if is_current_fliped {
            -bounds.1
        } else {
            bounds.1
        };

        for idx in idx_iter {
            // use all remaining indicies
            ret[idx.0] = if idx.1 { -bounds.1 } else { bounds.1 };
        }

        return ret;
    }

    #[must_use]
    fn expected_value(&self) -> Option<f64> {
        return Some(0.0);
    }

    #[must_use]
    fn variance(&self) -> Option<f64> {
        return if 2.0 < self.degrees_of_freedom {
            Some(self.degrees_of_freedom / (self.degrees_of_freedom - 2.0))
        } else if 1.0 < self.degrees_of_freedom {
            Some(f64::INFINITY)
        } else {
            // (dof <= 1)
            None
        };
    }

    #[must_use]
    fn mode(&self) -> f64 {
        return 0.0;
    }

    #[must_use]
    fn median(&self) -> f64 {
        return 0.0;
    }

    #[must_use]
    fn skewness(&self) -> Option<f64> {
        return if 3.0 < self.degrees_of_freedom {
            Some(0.0)
        } else {
            None
        };
    }

    #[must_use]
    fn kurtosis(&self) -> Option<f64> {
        return self.excess_kurtosis().map(|x| x + 3.0);
    }

    #[must_use]
    fn excess_kurtosis(&self) -> Option<f64> {
        return if 4.0 < self.degrees_of_freedom {
            Some(6.0 / (self.degrees_of_freedom - 4.0))
        } else if 2.0 < self.degrees_of_freedom {
            Some(f64::INFINITY)
        } else {
            // (dof <= 2)
            None
        };
    }

    #[must_use]
    fn moments(&self, order: u8, mode: crate::euclid::Moments) -> f64 {
        // https://en.wikipedia.org/wiki/Student%27s_t-distribution#Moments

        let dof: f64 = self.degrees_of_freedom;
        if dof <= f64::from(order) {
            // Moment does Not exist
            return f64::NAN;
        }

        let (mean, std_dev): (f64, f64) = match mode {
            crate::euclid::Moments::Raw => {
                if (order & 1) == 1 {
                    // order is odd,
                    return 0.0;
                }
                // order is even

                /*
                    let k: f64 = order as f64;
                    let num: f64 = gamma((k + 1.0) * 0.5) * gamma((dof - k) * 0.5) * dof.powf(k * 0.5);
                    let den: f64 = f64::consts::PI.sqrt() * gamma(dof * 0.5);
                    return num / den;
                */

                let mut acc: f64 = 1.0;
                for j in 1..(order >> 1) {
                    acc = acc * (2.0 * f64::from(j) - 1.0) / (dof - 2.0 * f64::from(j));
                }
                return acc * dof.powf(f64::from(order) * 0.5);
            }
            crate::euclid::Moments::Central => (0.0, 1.0),
            crate::euclid::Moments::Standarized => (
                0.0,
                self.variance().expect(
                    "Tried to compute a standarized moment but the variance is undefined. ",
                ),
            ),
        };

        let domain: &ContinuousDomain = self.get_domain();
        let bounds: (f64, f64) = domain.get_bounds();

        // println!("(mean, std_dev): {:?}", (mean, std_dev));

        let order_exp: i32 = i32::from(order);
        let (minus_mean, inv_std_dev) = (-mean, 1.0 / std_dev.sqrt());
        let (_, num_steps): (f64, usize) =
            crate::euclid::choose_integration_precision_and_steps(bounds, true);

        let moment: f64 = {
            // integral {a -> inf} f(x) dx  = integral {0 -> 1} f(a + t/(t - 1))  /  (1 - t)^2  dt

            let integration_fn = |x: f64| 'integration: {
                // x will go from -1.0 to 1.0

                let u: f64 = 1.0 - x * x;
                if u.abs() < f64::EPSILON {
                    // too near singularity, skip
                    break 'integration 0.0;
                }
                let v: f64 = 1.0 / u;
                let fn_input: f64 = x * v;
                let std_inp: f64 = (fn_input + minus_mean) * inv_std_dev;
                break 'integration std_inp.powi(order_exp)
                    * self.pdf(fn_input)
                    * (1.0 + x * x)
                    * v
                    * v;
            };

            crate::euclid::numerical_integration_finite(integration_fn, bounds, num_steps as u64)
        };

        return moment;
    }

    #[must_use]
    fn entropy(&self) -> f64 {
        let a: f64 = (self.degrees_of_freedom + 1.0) * 0.5;
        let b: f64 = self.degrees_of_freedom * 0.5;

        let term_1: f64 = a * (euclid::digamma(a) - euclid::digamma(b));

        let beta: f64 = euclid::beta_fn(b, 0.5);
        let term_2: f64 = (self.degrees_of_freedom.sqrt() * beta).ln();
        return term_1 + term_2;
    }

    #[must_use]
    fn confidence_interval(
        &self,
        hypothesys: crate::hypothesis::Hypothesis,
        significance_level: f64,
    ) -> (f64, f64) {
        assert!(
            significance_level.is_finite(),
            "Tried to call Distribution::confidence_interval with a non-finite `significance_level` (infinite or NaN). `significance_level` must be a probability. "
        );

        let mut bounds: (f64, f64) = self.get_domain().get_bounds();
        if significance_level <= 0.0 || bounds.1 - bounds.0 <= 0.0 {
            return euclid::DEFAULT_EMPTY_DOMAIN_BOUNDS;
        }

        if 1.0 <= significance_level {
            return bounds;
        }

        // significance_level is a value within (0.0, 1.0).

        match hypothesys {
            crate::hypothesis::Hypothesis::RightTail => {
                let quantile: f64 = self.quantile(1.0 - significance_level);
                bounds.1 = quantile;
            }
            crate::hypothesis::Hypothesis::LeftTail => {
                let quantile: f64 = self.quantile(significance_level);
                bounds.0 = quantile;
            }
            crate::hypothesis::Hypothesis::TwoTailed => {
                /*
                 */
                let quantiles: Vec<f64> =
                    self.quantile_multiple(&[significance_level, 1.0 - significance_level]);
                bounds.0 = quantiles[0];
                bounds.1 = quantiles[1];

                /*
                let quantile: f64 = self.quantile(significance_level);
                bounds.0 = quantile;
                bounds.1 = -quantile;
                 */
            }
        }

        return bounds;
    }

    #[must_use]
    fn p_value(&self, hypothesys: crate::hypothesis::Hypothesis, statistic: f64) -> f64 {
        // https://en.wikipedia.org/wiki/P-value#Definition
        let bounds: (f64, f64) = self.get_domain().get_bounds();
        if bounds.1 - bounds.0 <= 0.0 {
            return 0.0;
        }

        let density: f64 = self.cdf(statistic);
        let p: f64 = match hypothesys {
            crate::hypothesis::Hypothesis::RightTail => 1.0 - density,
            crate::hypothesis::Hypothesis::LeftTail => density,
            crate::hypothesis::Hypothesis::TwoTailed => 2.0 * density.min(1.0 - density),
        };

        return p;
    }
}

impl Parametric for StudentT {
    #[must_use]
    fn general_pdf(&self, x: f64, parameters: &[f64]) -> f64 {
        // let norm = gamma((nu-1)/2) / (sqrt(pi * nu) * gamma(nu/2))
        // pdf(x | nu) = norm(nu) * (1 + x^2 / nu) ^ (-(nu+1)/2)

        let dof: f64 = parameters[0];
        let norm: f64 = StudentT::compute_normalitzation_constant(dof);

        let base: f64 = 1.0 + x * x / dof;
        let exponent: f64 = -(dof + 1.0) / 2.0;
        return base.powf(exponent) * norm;
    }

    #[must_use]
    fn number_of_parameters() -> u16 {
        return 1;
    }

    fn get_parameters(&self, parameters: &mut [f64]) {
        parameters[0] = self.degrees_of_freedom;
    }

    #[must_use]
    fn derivative_pdf_parameters(&self, x: f64, parameters: &[f64]) -> Vec<f64> {
        // d/dx ln(f(x)) = f'(x)/f(x)
        // => f(x) * d/dx ln(f(x)) = f'(x)

        // Reserve a vector with exacly 2 elements
        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(2);

        // let norm = gamma((nu-1)/2) / (sqrt(pi * nu) * gamma(nu/2))
        // pdf(x | nu) = norm(nu) * (1 + x^2 / nu) ^ (-(nu+1)/2)

        //let nu: f64 = parameters[0];

        let pdf: f64 = self.general_pdf(x, parameters);
        let ln_der: Vec<f64> = self.log_derivative_pdf_parameters(x, parameters);
        assert!(ln_der.len() == StudentT::number_of_parameters() as usize);

        {
            // ## Derivative respect to x:
            /*
               d/dx pdf(x | nu) = d/dx norm(nu) * (1 + x^2 / nu) ^ (-(nu+1)/2)
                = norm(nu) * d/dx exp((-(nu+1)/2) * ln(1 + x^2 / nu))
                = norm(nu) * exp(-(nu+1)/2 * ln(1 + x^2 / nu)) * d/dx -(nu+1)/2 * ln(1 + x^2 / nu)
                = norm(nu) * (1 + x^2 / nu) ^ (-(nu+1)/2) * -(nu+1)/2 * d/dx ln(1 + x^2 / nu)
                = pdf(x | nu) * -(nu+1)/2 * (2*x / nu)/(1 + x^2 / nu)
                = pdf(x | nu) * -(nu + 1) * x/(nu + x^2)

               Confirmation:

               f(x) * d/dx ln(f(x)) = f'(x)
               d/dx ln(pdf(x | nu)) = -(nu + 1) * x/(nu + x^2)

               pdf(x | nu) * -(nu + 1) * x/(nu + x^2)

               Wich is identical to our result for the normal computation.
            */

            ret.push(pdf * ln_der[0]);
        }

        {
            // ## Derivative respect to nu:
            /*
                d/d_nu pdf(x | nu) = d/d_nu gamma((nu-1)/2) / (sqrt(pi * nu) * gamma(nu/2)) * (1 + x^2 / nu) ^ (-(nu+1)/2)
                 = d/d_nu gamma((nu-1)/2) / (sqrt(pi * nu) * gamma(nu/2)) * (1 + x^2 / nu) ^ (-(nu+1)/2)

                // I'm not doing that. Not because I can't, but because We can do it 10 times easier with
                // the log-derivative trick and if I do it normally it will be impossible to check it it is
                // correct anyways.

                f(x) * d/dx ln(f(x)) = f'(x)
                d/d_nu ln(pdf(x | nu)) = digamma(nu-1) - 0.5 * (1/nu + digamma(nu/2) + ln(1 + x^2 / nu) - (nu+1) * x^2 /(nu^2 + x^2 * nu))

                d/d_nu pdf(x | nu) = gamma((nu-1)/2) / (sqrt(pi * nu) * gamma(nu/2)) * (1 + x^2 / nu) ^ (-(nu+1)/2) * (digamma(nu-1) - 0.5 * (1/nu + digamma(nu/2) + ln(1 + x^2 / nu) - (nu+1) * x^2 /(nu^2 + x^2 * nu)))
                d/d_nu pdf(x | nu) = pdf(x | nu) * digamma(nu-1) - 0.5 * (1/nu + digamma(nu/2) + ln(1 + x^2 / nu) - (nu+1) * x^2 /(nu^2 + x^2 * nu))

            */

            ret.push(pdf * ln_der[1]);
        }

        return ret;
    }

    #[must_use]
    fn log_derivative_pdf_parameters(&self, x: f64, parameters: &[f64]) -> Vec<f64> {
        // d/dx ln(f(x)) = f'(x)/f(x)

        // let norm = gamma((nu-1)/2) / (sqrt(pi * nu) * gamma(nu/2))
        // pdf(x | nu) = norm(nu) * (1 + x^2 / nu) ^ (-(nu+1)/2)
        // ln(pdf(x | nu)) = ln(norm(nu) * (1 + x^2 / nu) ^ (-(nu+1)/2))
        // ln(pdf(x | nu)) = ln(norm(nu)) - (nu+1)/2 * ln(1 + x^2 / nu)
        // ln(pdf(x | nu)) = ln(gamma((nu-1)/2) / (sqrt(pi * nu) * gamma(nu/2))) - (nu+1)/2 * ln(1 + x^2 / nu)
        // ln(pdf(x | nu)) = ln(gamma((nu-1)/2)) - ln(sqrt(pi * nu) * gamma(nu/2))) - (nu+1)/2 * ln(1 + x^2 / nu)
        // ln(pdf(x | nu)) = ln(gamma(nu-1)) - ln(2) - 0.5*ln(pi * nu) - ln(gamma(nu/2)) - (nu+1)/2 * ln(1 + x^2 / nu)
        // ln(pdf(x | nu)) = ln(gamma(nu-1)) - ln(2) - 0.5*ln(pi) -0.5*ln(nu) - ln(gamma(nu/2)) - (nu+1)/2 * ln(1 + x^2 / nu)

        // Note:
        // Definition: digamma(x) = d/dx[ ln(Gamma(x)) ] = Gamma'(x) / Gamma(x)

        // Reserve a vector with exacly 2 elements
        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(2);

        let nu: f64 = parameters[0];

        {
            // ## Log derivative respect to x:
            /*

                d/dx ln(pdf(x | nu)) = d/dx ln(norm(nu)) - (nu+1)/2 * ln(1 + x^2 / nu)
                 = 0 - (nu+1)/2 * (2*x / nu)/(1 + x^2 / nu)
                 = -(nu + 1) * (x / nu)/(1 + x^2 / nu)
                 = -(nu + 1) * x/(nu*(1 + x^2 / nu))
                 = -(nu + 1) * x/(nu + x^2)

            */

            let ln_der: f64 = (nu + 1.0) * x / (nu + x * x);

            ret.push(ln_der);
        }

        {
            // ## Log derivative respect to a:
            /*

                d/d_nu ln(pdf(x | nu)) = d/d_nu ln(gamma(nu-1)) - ln(2) - 0.5*ln(pi) -0.5*ln(nu) - ln(gamma(nu/2)) - (nu+1)/2 * ln(1 + x^2 / nu) =
                = digamma(nu-1) - 0 - 0 -0.5/nu - 0.5*digamma(nu/2) - d/d_nu 0.5*(nu+1) * ln(1 + x^2 / nu)
                = digamma(nu-1) - 0.5/nu - 0.5*digamma(nu/2) - 0.5 * (ln(1 + x^2 / nu) + (nu+1) * d/d_nu[ 1 + x^2 / nu ]/(1 + x^2 / nu))
                = digamma(nu-1) - 0.5/nu - 0.5*digamma(nu/2) - 0.5 * (ln(1 + x^2 / nu) + (nu+1) * -x^2 / nu^2 /(1 + x^2 / nu))
                = digamma(nu-1) - 0.5/nu - 0.5*digamma(nu/2) - 0.5 * (ln(1 + x^2 / nu) - (nu+1) * x^2 /(nu^2 + x^2 * nu))
                = digamma(nu-1) - 0.5 * (1/nu + digamma(nu/2) + ln(1 + x^2 / nu) - (nu+1) * x^2 /(nu^2 + x^2 * nu))

            */

            let digamma_nu: f64 = euclid::digamma(nu - 1.0);
            let digamma_nu_half: f64 = euclid::digamma(nu * 0.5);

            let inv_nu: f64 = 1.0 / nu;

            // 1 + x^2 / nu
            let u: f64 = 1.0 + x * x * inv_nu;

            // 1/nu + digamma(nu/2) + ln(1 + x^2 / nu) - (nu+1) * x^2 /(nu^2 + x^2 * nu)
            //  = 1/nu + digamma(nu/2) + ln(u) - (nu+1) * x^2 /(u * nu^2)
            let term: f64 = inv_nu + digamma_nu_half + u.ln() - (nu + 1.0) * x * x / (u * nu * nu);

            ret.push(digamma_nu - 0.5 * term);
        }

        return ret;
    }

    // fn parameter_restriction(&self, parameters: &mut [f64]) {}

    #[must_use]
    fn fit(&self, data: &mut crate::samples::Samples) -> Vec<f64> {
        /*

            Using Maximum Likelyhood estimation:
            Assuming k samples.

                Estimation of nu:

            let norm = gamma((nu-1)/2) / (sqrt(pi * nu) * gamma(nu/2))
            pdf(x | nu) = norm(nu) * (1 + x^2 / nu) ^ (-(nu+1)/2)
            d/d_nu ln(pdf(x | nu) = digamma(nu-1) - 0.5 * (1/nu + digamma(nu/2) + ln(1 + x^2 / nu) - (nu+1) * x^2 /(nu^2 + x^2 * nu))


            0 = summatory{x_i} digamma(nu-1) - 0.5 * (1/nu + digamma(nu/2) + ln(1 + x_i^2 / nu) - (nu+1) * x_i^2 /(nu^2 + x_i^2 * nu))
            0 = k*digamma(nu-1) + summatory{x_i} - 0.5 * (1/nu + digamma(nu/2) + ln(1 + x_i^2 / nu) - (nu+1) * x_i^2 /(nu^2 + x_i^2 * nu))
            0 = k*digamma(nu-1) - 0.5 * summatory{x_i} 1/nu + digamma(nu/2) + ln(1 + x_i^2 / nu) - (nu+1) * x_i^2 /(nu^2 + x_i^2 * nu)
            0 = k*digamma(nu-1) - 0.5 * (k/nu + k*digamma(nu/2) + summatory{x_i} ln(1 + x_i^2 / nu) - (nu+1) * x_i^2 /(nu^2 + x_i^2 * nu) )
            2*k*digamma(nu-1) = k/nu + k*digamma(nu/2) + summatory{x_i}[ ln(1 + x_i^2 / nu) - (nu+1) * x_i^2 /(nu^2 + x_i^2 * nu) ]
            0 = -2*k*digamma(nu-1) + k/nu + k*digamma(nu/2) + summatory{x_i}[ ln(1 + x_i^2 / nu) - (nu+1) * x_i^2 /(nu^2 + x_i^2 * nu) ]
            0 = k*(-2*digamma(nu-1) + 1/nu + digamma(nu/2)) + summatory{x_i}[ ln(1 + x_i^2 / nu) - (nu+1) * x_i^2 /(nu^2 + x_i^2 * nu) ]

            In this position we are stuck and we cannot progress analytically any further. If we treat the whole
            RHS as a function that ew need to equate to 0, we can use gradient descent and newton's method to
            progress further. For this we will assume that for any valid dataset, there is exacly 1 zero in
            this function when `0 < nu < inf`. In this case we have the function:

            f(nu) = k*(-2*digamma(nu-1) + 1/nu + digamma(nu/2)) + summatory{x_i}[ ln(1 + x_i^2 / nu) - (nu+1) * x_i^2 /(nu^2 + x_i^2 * nu) ]

            And we need to find `nu*` such that f(nu*) = 0. We can initialize our initial guess to
            a sample from an exponentaial. The problem is if it converges to a negative solution. In
            that case we will just redo everything. Finding f'(nu):

            f'(nu) = d/d_nu k*(-2*digamma(nu-1) + 1/nu + digamma(nu/2)) + summatory{x_i}[ ln(1 + x_i^2 / nu) - (nu+1) * x_i^2 /(nu^2 + x_i^2 * nu) ]
            f'(nu) = k*(-2*digamma'(nu-1) - 1/nu^2 + 0.5*digamma'(nu/2)) + summatory{x_i}[ (-x_i^2 / nu^2)/(1 + x_i^2 / nu) -  x_i^2 * (nu^2 + x_i^2 * nu - (nu+1)*(2*nu + x_i^2))/(nu^2 + x_i^2 * nu)^2 ]

        */

        // TODO: heavy testing!!!

        let mut rng: rand::prelude::ThreadRng = rng();
        // SAFETY: value is safe to only read
        let h: f64 = unsafe { configuration::derivation::DEAFULT_H };
        // SAFETY: value is safe to only read
        let convergence_dist: f64 = unsafe {
            configuration::maximum_likelihood_estimation::CONVERGENCE_DIFFERENCE_CRITERIA
        };
        let der_digamma = |x: f64| (euclid::digamma(x + h) - euclid::digamma(x)) / h;
        let k: f64 = data.count() as f64;
        let mut nu: f64;
        'main: loop {
            nu = 10.0 * rng.random::<f64>().ln();
            // draw from exponential distribution with lambda = 0.1

            loop {
                let inv_nu: f64 = 1.0 / nu;
                // =  k*(-2*digamma(nu-1) + 1/nu + digamma(nu/2))
                let term_1: f64 =
                    k * (-2.0 * euclid::digamma(nu - 1.0) + inv_nu + euclid::digamma(nu * 0.5));

                // = k*(-2*digamma'(nu-1) - 1/nu^2 + 0.5*digamma'(nu/2))
                let term_1_der: f64 = k
                    * (-2.0 * der_digamma(nu - 1.0) - inv_nu * inv_nu
                        + 0.5 * der_digamma(nu * 0.5));

                let mut acc: f64 = 0.0;
                let mut acc_der: f64 = 0.0;

                for &s in data.peek_data() {
                    // ln(1 + x_i^2 / nu) - (nu+1) * x_i^2 /(nu^2 + x_i^2 * nu)
                    // -x_i^2 * ( 1/(nu^2 + x_i^2 * nu) + (nu^2 + x_i^2 * nu - (nu+1)*(2*nu + x_i^2))/(nu^2 + x_i^2 * nu)^2)
                    // -x_i^2 * ( 1/u + (u - (nu+1)*(2*nu + x_i^2))/u^2)

                    let sq: f64 = s * s;
                    // = nu^2 + x_i^2 * nu
                    let u: f64 = nu * (nu + sq);
                    let inv_u: f64 = 1.0 / u;

                    acc += (1.0 + sq * inv_nu).ln() - (nu - 1.0) * sq * inv_u;
                    acc_der +=
                        -sq * (inv_nu + (u - (nu + 1.0) * (2.0 * nu * sq) * inv_nu * inv_nu));
                }

                // Newton's method
                let displacement: f64 = (term_1 + acc) / (term_1_der + acc_der);
                nu = nu - displacement;

                if displacement < convergence_dist {
                    if nu.is_sign_negative() {
                        // converged to negative (invalid) value
                        break;
                    }
                    break 'main;
                }
            }
        }

        return vec![nu];
    }
}
