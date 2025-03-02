//! # Gamma distribution
//!
//! The [Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution)
//! is a continuous probability distribution.
//!
//! It has 2 parameters, but there are 2 ways to model it:
//!
//! 1. `alpha` or shape
//! 2. `theta` or scale
//!
//! The other way is:
//!
//! 1. `alpha` or shape
//! 2. `lambda` or rate
//!
//! `theta = 1/lambda`
//!
//! All parameters (in every possible parametritzations) are stricly positive.
//!
//!

use rand::Rng;

use crate::{
    configuration,
    distribution_trait::{Distribution, Parametric},
    domain::ContinuousDomain,
    euclid,
};

pub const GAMMA_DOMAIN: ContinuousDomain = ContinuousDomain::From(0.0);

#[derive(Debug, Clone, PartialEq)]
pub struct Gamma {
    /// alpha or shape
    alpha: f64,
    /// theta or scale
    theta: f64,
    normalitzation_constant: f64,
}

pub struct GammaGenerator {
    rng: rand::prelude::ThreadRng,
    exp: crate::distributions::Exponential::ExponentialGenerator,
    norm: crate::distributions::Normal::StdNormalGenerator,
    alpha: f64,
    theta: f64,
    inv_a: f64,
    b: f64,
    c: f64,
}

impl Gamma {
    /// Creates a new [Gamma] distribution with parameters `alpha` and `theta`.
    ///
    /// It will return error under the following conditions:
    ///  - `alpha` is `+-inf` or a NaN
    ///  - `theta` is `+-inf` or a NaN
    ///  - `alpha <= 0.0`
    ///  - `theta <= 0.0`
    ///  - The values for `alpha` and `theta` are too large to model properly
    ///      - This means that a [f64] value is not precise enough.
    ///
    pub fn new(alpha: f64, theta: f64) -> Result<Gamma, ()> {
        if !alpha.is_finite() {
            return Err(());
        }

        if !theta.is_finite() {
            return Err(());
        }

        if alpha <= 0.0 {
            return Err(());
        }

        if theta <= 0.0 {
            return Err(());
        }

        let norm_const: f64 = euclid::gamma(alpha) * theta.powf(alpha);

        if !norm_const.is_finite() {
            // we do not have enough precision to do the computations
            return Err(());
        }

        return Ok(Gamma {
            alpha,
            theta,
            normalitzation_constant: 1.0 / norm_const,
        });
    }

    /// Creates a new [Gamma] distribution with parameters `alpha` and `theta`
    /// without checking for their correcness.
    ///
    /// The [Gamma] structure will be invalid if:
    ///  - `alpha` is `+-inf` or a NaN
    ///  - `theta` is `+-inf` or a NaN
    ///  - `alpha <= 0.0`
    ///  - `theta <= 0.0`
    ///  - The values for `alpha` and `theta` are too large to model properly
    ///      - This means that a [f64] value is not precise enough.
    ///
    pub unsafe fn new_unchecked(alpha: f64, theta: f64) -> Gamma {
        let norm_const: f64 = euclid::gamma(alpha) * theta.powf(alpha);

        return Gamma {
            alpha,
            theta,
            normalitzation_constant: 1.0 / norm_const,
        };
    }

    /// Get the parameter alpha
    pub fn get_alpha(&self) -> f64 {
        return self.alpha;
    }

    pub fn get_theta(&self) -> f64 {
        return self.theta;
    }

    /// Returns an iterator that can generate [Gamma] samples even faster
    /// than normally calling [Gamma::sample] many times. Uscefull if you don't
    /// know exacly how many values you want for [Gamma::sample_multiple].
    ///
    /// It avoids the heap allocation of [Gamma::sample_multiple] and
    /// the repeated initialitzation processes in [Gamma::sample].
    pub fn iter(&self) -> GammaGenerator {
        let b: f64 = self.alpha - (1.0 / 3.0);
        let c: f64 = 1.0 / (3.0 * b.sqrt());

        return GammaGenerator {
            rng: rand::thread_rng(),
            exp: super::Exponential::Exponential::new(1.0).unwrap().iter(),
            norm: super::Normal::StdNormal::new().iter(),
            alpha: self.alpha,
            theta: self.theta,
            inv_a: 1.0 / self.alpha,
            b,
            c,
        };
    }
}

impl Distribution for Gamma {
    fn pdf(&self, x: f64) -> f64 {
        let shape: f64 = x.powf(self.alpha - 1.0) * (-x / self.theta).exp();
        return self.normalitzation_constant * shape;
    }

    fn get_domain(&self) -> &crate::domain::ContinuousDomain {
        return &GAMMA_DOMAIN;
    }

    fn cdf_multiple(&self, points: &[f64]) -> Vec<f64> {
        /*
            Plan: (sery similar to [Distribution::quantile_multiple])

            For cdf_multiple we will first return an error if we find a NaN.
            Otherwise we will need to sort them and integrate until we have
            integrated to the given number (and store the value).
            By sorting, we only need to integrate once through the pdf, reducing
            considerably computation costs (in particular for large inputs).

            However, this *cool* strategy has a problem and is that we will not
            return the values in the order we were asked. To account for this we will
            only sort the indices.

            We will integrate using [Simpson's rule](https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_1/3_rule)
            for integration.

            Considering the bounds:
             - If min is finite we just integrate normally.
             - If min is infinite but max is finite, we can integrate the area from the end
                    and then do .map(|x| 1-x )
             - If both are infinite, we will need to do integration with a change of variable

            To compute integrals over an infinite range, we will perform a special
            [numerial integration](https://en.wikipedia.org/wiki/Numerical_integration#Integrals_over_infinite_intervals).
            (change of variable)

                For -infinite to a (const):
            integral {-inf -> a} f(x) dx =
                        integral {0 -> 1} f(a - (1 - t)/t)  /  t^2  dt

                For -infinite to infinite:
            integral {-inf -> inf} f(x) dx =
                        integral {-1 -> 1} f( t / (1-t^2) ) * (1 + t^2) / (1 - t^2)^2  dt

            And "just" compute the new integral (taking care of the singularities at t = 0).

        */

        if points.is_empty() {
            return Vec::new();
        }

        // return error if NAN is found
        for point in points {
            if point.is_nan() {
                std::panic!("Found NaN in `cdf_multiple` of Gamma. \n");
                // return Err(AdvStatError::NanErr);
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
            euclid::choose_integration_precision_and_steps(bounds);
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

            let (middle, end): (f64, f64) = {
                let _middle: f64 = self.pdf(current_position + half_step_length);
                let _end: f64 = self.pdf(current_position + step_length);
                (_middle, _end)
            };

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
        // https://en.wikipedia.org/wiki/Gamma_distribution#Random_variate_generation
        // https://github.com/numpy/numpy/blob/main/numpy/random/src/distributions/distributions.c#L220

        let mut exp: crate::distributions::Exponential::ExponentialGenerator =
            super::Exponential::Exponential::new(1.0).unwrap().iter();

        if self.alpha == 1.0 {
            return exp.take(n).collect::<Vec<f64>>();
        }

        assert!(self.alpha != 0.0 && self.alpha != 1.0);

        let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(n);

        let inv_a: f64 = 1.0 / self.alpha;

        if 1.0 < self.alpha {
            for _ in 0..n {
                let r: f64 = 'gen: loop {
                    let u: f64 = rng.gen::<f64>();
                    let v: f64 = exp.next().unwrap();

                    /*
                        if u <= 1.0 - self.alpha {
                        if u - 1.0 <= - self.alpha {
                        if self.alpha <= -u + 1.0 {
                        if self.alpha <= 1.0 - u {
                        if self.alpha <= u {
                    */
                    if u <= self.alpha {
                        let x: f64 = u.powf(inv_a);
                        if x <= v {
                            break 'gen x;
                        }
                    } else {
                        let y: f64 = -(u * inv_a).ln();
                        let x: f64 = (1.0 - self.alpha + self.alpha * y).powf(inv_a);

                        if x <= (v + y) {
                            break 'gen x;
                        }
                    }
                };
                ret.push(r * self.theta);
            }
        } else {
            let mut norm: crate::distributions::Normal::StdNormalGenerator =
                super::Normal::StdNormal::new().iter();
            let b: f64 = self.alpha - (1.0 / 3.0);
            let c: f64 = 1.0 / (3.0 * b.sqrt());
            for _ in 0..n {
                let r: f64 = 'gen: loop {
                    let mut x: f64;
                    let mut v: f64;
                    's: loop {
                        x = norm.next().unwrap();
                        v = 1.0 + c * x;
                        if v <= 0.0 {
                            break 's;
                        }
                    }
                    v = v * v * v;
                    let u: f64 = rng.gen::<f64>();

                    let x_sq: f64 = x * x;
                    if u < 1.0 - 0.0331 * x_sq * x_sq {
                        break 'gen b * v;
                    }

                    if u.ln() < 0.5 * x_sq + b * (1.0 - v + v.ln()) {
                        break 'gen b * v;
                    }
                };
                ret.push(r * self.theta);
            }
        };

        return ret;
    }

    fn quantile_multiple(&self, points: &[f64]) -> Vec<f64> {
        /*
            Plan:

            For this function we will first return an error if we find a NaN.
            Otherwise we will need to sort them and integrate until the area under
            the pdf is = to the given number. By sorting, we only need to integrate
            once.

            However, this *cool* strategy has a problem and is that we will not
            return the values in the order we were asked. To account for this we will
            only sort the indices.

            Also, if we find any values smaller or greater than 0 or 1, the awnser will
            always be the edges of the domain (simplifying computations, although this
            case should not normally happen).

            We will integrate using [Simpson's rule](https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_1/3_rule)
            for integration.

            Considering the bounds:
             - If min is finite we just integrate normally.
             - If min is infinite but max is finite, we can integrate the area from the end
                    until `1.0 - point`
             - If both are infinite, we will need to do integration with a change of variable

            To compute integrals over an infinite range, we will perform a special
            [numerial integration](https://en.wikipedia.org/wiki/Numerical_integration#Integrals_over_infinite_intervals).

                For -infinite to infinite:
            integral {-inf -> inf} f(x) dx  = integral {-1 -> 1} f(t/(1 - t^2))  *  (1 + t^2) / (1 - t^2)^2  dt

            And "just" compute the new integral (taking care of the singularities at t = +-1).

        */

        if points.is_empty() {
            return Vec::new();
        }

        // return error if NAN is found
        for point in points {
            if point.is_nan() {
                std::panic!("Found NaN in `quantile_multiple` for Gamma. \n");
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
            euclid::choose_integration_precision_and_steps(bounds);
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

            let (middle, end): (f64, f64) = {
                let _middle: f64 = self.pdf(current_position + half_step_length);
                let _end: f64 = self.pdf(current_position + step_length);
                (_middle, _end)
            };

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
        return Some(self.alpha * self.theta);
    }

    fn variance(&self) -> Option<f64> {
        return Some(self.alpha * self.theta * self.theta);
    }

    fn mode(&self) -> f64 {
        return ((self.alpha - 1.0) * self.theta).max(0.0);
    }

    // median has no simple closed form

    fn skewness(&self) -> Option<f64> {
        return Some(2.0 / self.alpha.sqrt());
    }

    fn kurtosis(&self) -> Option<f64> {
        return self.excess_kurtosis().map(|x| x + 3.0);
    }

    fn excess_kurtosis(&self) -> Option<f64> {
        return Some(6.0 / self.alpha);
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

        if let euclid::Moments::Raw = mode {
            let mut acc: f64 = 1.0;
            let mut i: f64 = 1.0;
            while i <= (order as f64) {
                acc = acc * (self.alpha + i - 1.0);
                i += 1.0;
            }
            return self.theta.powi(order as i32) * acc;
        }

        let domain: &ContinuousDomain = self.get_domain();
        let bounds: (f64, f64) = domain.get_bounds();

        // The values of 0.0 and 1.0 have no special meaning. They are not going to be used anyway.
        let (mean, std_dev): (f64, f64) = match mode {
            euclid::Moments::Raw => unreachable!(),
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
        let integration_type: euclid::IntegrationType =
            euclid::IntegrationType::from_bounds(bounds);
        let (_, num_steps): (f64, usize) = euclid::choose_integration_precision_and_steps(bounds);

        let moment: f64 = match integration_type {
            euclid::IntegrationType::Finite => {
                let integration_fn = |x: f64| {
                    let std_inp: f64 = (x + minus_mean) * inv_std_dev;
                    std_inp.powi(order_exp) * self.pdf(x)
                };

                euclid::numerical_integration_finite(integration_fn, bounds, num_steps as u64)
            }
            euclid::IntegrationType::InfiniteToConst => {
                // integral {-inf -> a} f(x) dx = integral {0 -> 1} f(a - (1 - t)/t)  /  t^2  dt
                let integration_fn = |x: f64| 'integration: {
                    // x will go from 0.0 to 1.0
                    if x.abs() < f64::EPSILON {
                        // too near singularity, skip
                        break 'integration 0.0;
                    }
                    let inv_x: f64 = 1.0 / x;
                    let fn_input: f64 = bounds.1 - (1.0 - x) * inv_x;
                    let std_inp: f64 = (fn_input + minus_mean) * inv_std_dev;
                    std_inp.powi(order_exp) * self.pdf(fn_input) * inv_x * inv_x
                };

                euclid::numerical_integration_finite(integration_fn, bounds, num_steps as u64)
            }
            euclid::IntegrationType::ConstToInfinite => {
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
            }
            euclid::IntegrationType::FullInfinite => {
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
                    std_inp.powi(order_exp) * self.pdf(fn_input) * (1.0 + x * x) * v * v
                };

                euclid::numerical_integration_finite(integration_fn, bounds, num_steps as u64)
            }
        };

        return moment;
    }

    fn entropy(&self) -> f64 {
        return self.alpha
            + self.theta.ln()
            + euclid::ln_gamma(self.alpha)
            + (1.0 - self.alpha) * euclid::digamma(self.alpha);
    }
}

impl Parametric for Gamma {
    /// Evaluates the [Gamma::pdf] in a general way taking into account
    /// the parameters.
    ///
    /// ### Parameters for Gamma:
    ///
    /// The [Gamma] distribution has 2 parameters `alpha = a` and `theta = t`.
    /// The order of the `parameters` is:
    ///
    /// > \[alpha, theta\]
    ///
    /// Alpha and theta must be both stricly positive.
    fn general_pdf(&self, x: f64, parameters: &[f64]) -> f64 {
        // pdf(x | a, t) = 1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t)
        let a: f64 = parameters[0];
        let t: f64 = parameters[1];
        let shape: f64 = x.powf(a - 1.0) * (-x / t).exp();
        return self.normalitzation_constant * shape;
    }

    fn number_of_parameters() -> u16 {
        return 2;
    }

    fn get_parameters(&self, parameters: &mut [f64]) {
        parameters[0] = self.alpha;
        parameters[1] = self.theta;
    }

    fn derivative_pdf_parameters(&self, x: f64, parameters: &[f64]) -> Vec<f64> {
        // d/dx ln(f(x)) = f'(x)/f(x)
        // => f(x) * d/dx ln(f(x)) = f'(x)

        // pdf(x | a, t) = 1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t)

        // Reserve a vector with exacly 3 elements
        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(3);

        let a: f64 = parameters[0];
        let t: f64 = parameters[1];

        // = x/t
        let x_over_t: f64 = x / t;
        // = exp(-x/t)
        let exp_xt: f64 = (-x_over_t).exp();
        // = x^(a-2)
        let pow_a_2: f64 = x.powf(a - 2.0);
        // = t^a
        let t_raised_a: f64 = t.powf(a);
        // = 1 / (Gamma(a)*t^a)
        let norm_const: f64 = 1.0 / (euclid::gamma(a) * t_raised_a);

        {
            // ## Derivative respect to x:
            /*
                d/dx pdf(x | a, t) = d/dx 1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t)
                = 1/(Gamma(a)*t^a) * d/dx x^(a-1) * exp(-x/t)
                = 1/(Gamma(a)*t^a) * ((a-1)x^(a-2) * exp(-x/t) + x^(a-1) * exp(-x/t) * d/dx[ -x/t ])
                = 1/(Gamma(a)*t^a) * ((a-1)x^(a-2) * exp(-x/t) + x^(a-1) * exp(-x/t) * -1/t )
                = 1/(Gamma(a)*t^a) * ((a-1)x^(a-2) * exp(-x/t) + x^(a-2) * exp(-x/t) * x * -1/t )
                = 1/(Gamma(a)*t^a) * x^(a-2) * exp(-x/t) * (a - 1 - x/t)
                = pdf(x | a, t) / x * (a - 1 - x/t )

                    Confirmation:

                f(x) * d/dx ln(f(x)) = f'(x)
                d/dx ln(pdf(x | a, t)) = (a-1) / x - 1 / t

                1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t) * ((a-1) / x - 1 / t) =
                 = 1/(Gamma(a)*t^a) * x^(a-2) * x * exp(-x/t) * ((a-1) / x - 1 / t)
                 = 1/(Gamma(a)*t^a) * x^(a-2) * exp(-x/t) * (a - 1 - x/t)

                Wich is identical to our result for the normal computation.

            */
            // = (a - 1 - x/t )
            let term: f64 = a - 1.0 - x_over_t;
            // x^(a-2) * exp(-x/t)
            let shape: f64 = pow_a_2 * exp_xt;

            ret.push(norm_const * shape * term);
        }

        {
            // ## Derivative respect to a:
            /*
                d/da pdf(x | a, t) = d/da 1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t)
                 = exp(-x/t) * d/da x^(a-1)/(Gamma(a)*t^a)
                 = exp(-x/t) * d/da x^(a-1) * t^-a * Gamma(a)^-1
                 = exp(-x/t) * (d/da[ x^(a-1) ] * t^-a * Gamma(a)^-1 + x^(a-1) * d/da[ t^-a ] * Gamma(a)^-1 + x^(a-1) * t^-a * d/da[ Gamma(a)^-1 ] )
                 = exp(-x/t) * (ln(x) * x^(a-1) * t^-a * Gamma(a)^-1 + x^(a-1) * ln(t) * t^-a * d/da[ -a ] * Gamma(a)^-1 + x^(a-1) * t^-a * -1 * Gamma(a)^-2 * d/da[ Gamma(a) ] )
                 = exp(-x/t) * (ln(x) * x^(a-1) * t^-a * Gamma(a)^-1 + x^(a-1) * -ln(t) * t^-a * Gamma(a)^-1 + x^(a-1) * t^-a * -1 * Gamma(a)^-2 * Digamma(a) * Gamma(a) )
                 = exp(-x/t) * (ln(x) * x^(a-1) * t^-a * Gamma(a)^-1 + x^(a-1) * -ln(t) * t^-a * Gamma(a)^-1 + x^(a-1) * t^-a * -1 * Gamma(a)^-1 * Digamma(a) )
                 = exp(-x/t) * Gamma(a)^-1 * (ln(x) * x^(a-1) * t^-a + x^(a-1) * -ln(t) * t^-a + x^(a-1) * t^-a * -1 * Digamma(a) )
                 = exp(-x/t) * Gamma(a)^-1 * x^(a-1) * (ln(x) * t^-a + -ln(t) * t^-a + t^-a * -1 * Digamma(a) )
                 = exp(-x/t) * Gamma(a)^-1 * x^(a-1) * t^-a * (ln(x) + -ln(t) + -Digamma(a) )
                 = 1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t) * (ln(x) - ln(t) - Digamma(a))
                 = pdf(x | a, t) * (ln(x) + -ln(t) + -Digamma(a) )

                let norm_const = 1/(Gamma(a)*t^a);
                norm_const is already computed:

                 = norm_const * pow_a_2 * x * exp_xt * (ln(x) - ln(t) - Digamma(a))

                    Confirmation:

                f(x) * d/dx ln(f(x)) = f'(x)
                d/da ln(pdf(x | a, t)) = -Digamma(a) - ln(t) + ln(x)

                1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t) * (-Digamma(a) - ln(t) + ln(x)) =
                 = 1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t) * (ln(x) - ln(t) - Digamma(a))

                Wich is identical to our result for the normal computation.

            */

            // = ln(x) - ln(t) - Digamma(a)
            let term: f64 = x.ln() - t.ln() - euclid::digamma(a);

            ret.push(norm_const * pow_a_2 * x * exp_xt * term);
        }

        {
            // ## Derivative respect to t:
            /*
                d/dt pdf(x | a, t) = d/dt 1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t)
                 = x^(a-1)/Gamma(a) * d/dt 1/t^a * exp(-x/t)
                 = x^(a-1)/Gamma(a) * d/dt t^-a * exp(-x/t)
                 = x^(a-1)/Gamma(a) * d/dt exp(ln(t^-a)) * exp(-x/t)
                 = x^(a-1)/Gamma(a) * d/dt exp(-a * ln(t)) * exp(-x/t)
                 = x^(a-1)/Gamma(a) * d/dt exp(-a * ln(t) - x/t)
                 = x^(a-1)/Gamma(a) * exp(-a * ln(t) - x/t) * d/dt -a * ln(t) - x/t
                 = x^(a-1)/Gamma(a) * exp(-a * ln(t) - x/t) * (d/dt[ -a * ln(t) ] - d/dx[ x/t ])
                 = x^(a-1)/Gamma(a) * exp(-a * ln(t) - x/t) * (-a * d/dt[ ln(t) ] - x * d/dx[ 1/t ])
                 = x^(a-1)/Gamma(a) * exp(-a * ln(t) - x/t) * (-a * 1/t - x * -1/t^2 )
                 = x^(a-1)/Gamma(a) * exp(-a * ln(t) - x/t) * (-a/t + x/t^2)
                 = x^(a-1)/Gamma(a) * t^-a * exp(-x/t) * (-a/t + x/t^2)
                 = x^(a-1)/Gamma(a) * t^-a * exp(-x/t) * (-a + x/t)/t
                 = x^(a-1)/(Gamma(a)*t^a) * exp(-x/t) * (-a + x/t)/t
                 = 1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t) * (x/t - a)/t
                 = pdf(x | a, t) * (x/t - a)/t

                Substituting precomputed variables:
                 = pow_a_2 * x * norm_const * exp_xt * (-a + x_over_t)/t


                    Confirmation:

                f(x) * d/dx ln(f(x)) = f'(x)
                d/dx ln(pdf(x | a, t)) = [...]

                1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t) * (1/t * (x / t - a)) =
                 = 1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t) * (x/t - a)/t

                Wich is identical to our result for the normal computation.

            */
            // = (-a + x_over_t)/t
            let term: f64 = (x_over_t - a) / t;

            ret.push(pow_a_2 * x * norm_const * exp_xt * term);
        }

        return ret;
    }

    fn log_derivative_pdf_parameters(&self, x: f64, parameters: &[f64]) -> Vec<f64> {
        // d/dx ln(f(x)) = f'(x)/f(x)

        // pdf(x | a, t) = 1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t)
        // ln(pdf(x | a, t)) = ln(1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t))
        // ln(pdf(x | a, t)) = -ln(Gamma(a)*t^a) + ln(x^(a-1)) + ln(exp(-x/t))
        // ln(pdf(x | a, t)) = -(ln(Gamma(a)) + ln(t^a)) + (a-1)*ln(x) - x / t
        // ln(pdf(x | a, t)) = -(ln(Gamma(a)) + a*ln(t)) + (a-1)*ln(x) - x / t
        // ln(pdf(x | a, t)) = -ln(Gamma(a)) - a*ln(t) + (a-1)*ln(x) - x / t

        // Reserve a vector with exacly 3 elements
        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(3);

        let a: f64 = parameters[0];
        let t: f64 = parameters[1];

        let inv_t: f64 = 1.0 / t;

        {
            // ## Log derivative respect to x:
            /*
                d/dx ln(pdf(x | a, t)) = d/dx -ln(Gamma(a)) - a*ln(t) + (a-1)*ln(x) - x / t
                 = 0 - 0 + (a-1)/x - 1 / t
                 = (a-1) / x - 1 / t

            */

            ret.push((a - 1.0) / x - inv_t);
        }

        {
            // ## Log derivative respect to a:
            /*
                d/da ln(pdf(x | a, t)) = d/da -ln(Gamma(a)) - a*ln(t) + (a-1)*ln(x) - x / t
                 = -Digamma(a) - ln(t) + ln(x)

            */

            ret.push(-euclid::digamma(a) - t.ln() + x.ln());
        }

        {
            // ## Log derivative respect to t:
            /*
                d/dt ln(pdf(x | a, t)) = d/dt -ln(Gamma(a)) - a*ln(t) + (a-1)*ln(x) - x / t
                 = 0 - a/t + 0 + x / t^2
                 = - a/t + x / t^2
                 = 1/t * (-a + x / t)
                 = 1/t * (x / t - a)

            */

            ret.push(inv_t * (x * inv_t - a));
        }

        return ret;
    }

    fn parameter_restriction(&self, _parameters: &mut [f64]) {}

    fn fit(&self, data: &mut crate::Samples::Samples) -> Vec<f64> {
        /*
                Using Maximum Likelyhood estimation:
            Assuming k samples.

                Estimation of alpha:

            pdf(x | a, b) = 1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t)
            d/da ln(pdf(x | a, b) = -Digamma(a) - ln(t) + ln(x)
            0 = sumatory{x_i} -Digamma(a) - ln(t) + ln(x_i)
            0 = -k*Digamma(a) - k*ln(t) + sumatory{x_i} ln(x_i)
            k*Digamma(a) = - k*ln(t) + sumatory{x_i} ln(x_i)
            Digamma(a) = -ln(t) + 1/k * sumatory{x_i} ln(x_i)
            a = inv_digamma( -ln(t) + 1/k * sumatory{x_i} ln(x_i) )

                Estimation of theta:

            pdf(x | a, b) = 1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t)
            d/dt ln(pdf(x | a, b) = 1/t * (x / t - a)
            0 = sumatory{x_i} 1/t * (x_i / t - a)
            0 = 1/t * sumatory{x_i} x_i / t - a
            0 = 1/t * (-a * k + sumatory{x_i}[ x_i / t ])
            0 = 1/t * (-a * k + 1/t * sumatory{x_i}[ x_i ])
            0 * t = -a * k + 1/t * sumatory{x_i}[ x_i ]
            // deleting solution t = 0, wich is invalid
            0 = -a * k + 1/t * sumatory{x_i}[ x_i ]
            a * k = 1/t * sumatory{x_i}[ x_i ]
            a * k * t = 1/k * sumatory{x_i}[ x_i ]
            t = 1/a * 1/k * sumatory{x_i}[ x_i ]
            t = 1/a * mean{x_i}

            We have the following 2 equations:

            a * t = 1/k * sumatory{x_i}[ x_i ]
            Digamma(a) = -ln(t) + 1/k * sumatory{x_i} ln(x_i)

            We can compute `sumatory{x_i}[ x_i ]` and  `sumatory{x_i} ln(x_i)`.

            t = 1/a * mean{x_i}
            Digamma(a) = -ln(1/a * mean{x_i}) + 1/k * sumatory{x_i} ln(x_i)
            Digamma(a) = -ln(1/a) - ln(mean{x_i}) + 1/k * sumatory{x_i} ln(x_i)
            Digamma(a) = ln(a) - ln( mean{x_i} ) + mean{x_i}[ ln(x_i) ]
            Digamma(a) - ln(a) = mean{x_i}[ ln(x_i) ] - ln( mean{x_i} )

            We are stuck here, however we can try to find a numerical solution.
            If we create a statistic `s` and a function `f` such that:

            s = -( mean{x_i}[ ln(x_i) ] - ln( mean{x_i} ) )
            s = ln( mean{x_i} ) - mean{x_i}[ ln(x_i) ]
            f(a) = Digamma(a) - ln(a) + s

            Since `s` is considered contatn, the solution is just `a = f^-1(0)`.

            We will use Newton's method for that. Computing f'(a):

            d/da Digamma(a) = d/da Gamma'(a)/Gamma(a)
             = (Gamma''(a)*Gamma(a) - Gamma'(a)*Gamma'(a)) / Gamma(a)^2
             = (Gamma''(a)*Gamma(a) - Gamma'(a)^2) / Gamma(a)^2
             = (Gamma''(a)*Gamma(a) - (Digamma(x)*Gamma(a))^2) / Gamma(a)^2
             = (Gamma''(a)*Gamma(a) - Digamma(x)^2 * Gamma(a)^2) / Gamma(a)^2
             = Gamma''(a) / Gamma(a) - Digamma(x)^2
             = (polygamma_1(x) + Digamma(x)) * Gamma(a) / Gamma(a) - Digamma(x)^2
             = polygamma_1(x) + Digamma(x)^2 - Digamma(x)^2
             = polygamma_1(x)

            d/da f(a) = d/da Digamma(a) - ln(a) + s
            d/da f(a) = polygamma_1(a) - 1/a

            To compute the polygamma_1 function we will just take the derivative definition.
            Even if it contains some error, it should be accurate enough. Also, h can be decreased.
            Otherwise, a possible improvement for this function is to use an actual
            implemetation for oplygamma_1.

            d/da f(a) = lim{h->0}[ (Digamma(a + h) - Digamma(a))/h ] - 1/a

            Using Newton's method:

            a_i+1 = a_i - f(a_i)/f'(a_i)
            a_i+1 = a_i - (Digamma(a_i) - ln(a_i) + s)/(lim{h->0}[ (Digamma(a_i + h) - Digamma(a_i))/h ] - 1/a_i)
            a_i+1 = a_i - (Digamma(a_i) - ln(a_i) + s) / ((Digamma(a_i + h) - Digamma(a_i))/h - 1/a_i)
            a_i+1 = a_i - (Digamma(a_i) - ln(a_i) + s) / (Digamma(a_i + h) - Digamma(a_i) - h/a_i)/h
            a_i+1 = a_i - (Digamma(a_i) - ln(a_i) + s)*h / (Digamma(a_i + h) - Digamma(a_i) - h/a_i)

            For the initial value of a_0:

            Accordint to [wikipedia](https://en.wikipedia.org/wiki/Gamma_distribution#Maximum_likelihood_estimation),
            the following formula gives an acceptable estimate:

            a_0 = (3 - s + sqrt((s - 3)^2 + 24 * s)) / 12*s
            Note that s is always positive, therefore a_0 exists as long
            as samples are positive.

        */

        let mut parameters: Vec<f64> = Vec::new();
        parameters.reserve_exact(2);

        let mean: f64 = match data.mean() {
            Some(v) => {
                if v <= 0.0 {
                    return Vec::new();
                } else {
                    v
                }
            }
            None => return Vec::new(),
        };
        // if None, then means there was not enough data

        let mean_log: f64 = match data.log_mean() {
            Some(v) => v,
            None => return Vec::new(),
        };
        // If None, it means that the data did not come from a Gamma distribution

        //s = ln( mean{x_i} ) - mean{x_i}[ ln(x_i) ]
        let s: f64 = mean.ln() * mean_log;
        assert!(0.0 <= s);

        let mut a: f64 = {
            // a_0 = (3 - s + sqrt((3 - s)^2 + 24 * s)) / 12*s
            let aux: f64 = 3.0 - s;
            (aux - (aux * aux + 24.0 * s).sqrt()) / (12.0 * s)
        };

        // Newton's method:

        let h: f64 = unsafe { configuration::derivation::DEAFULT_H };
        let convergence_epsilon: f64 = unsafe {
            configuration::maximum_likelihood_estimation::CONVERGENCE_DIFFERENCE_CRITERIA
        };

        let mut difference: f64 = f64::MAX;
        while convergence_epsilon < difference {
            //a_i+1 = a_i - (Digamma(a_i) - ln(a_i) + s)*h / (Digamma(a_i + h) - Digamma(a_i) - h/a_i)

            let digamma: f64 = euclid::digamma(a);
            let digamma_h: f64 = euclid::digamma(a + h);

            let num: f64 = (digamma - a.ln() + s) * h;
            let den: f64 = digamma_h - digamma - h / a;

            difference = num / den;
            a = a - difference;
        }

        // a has converged

        // t = 1/a * mean{x_i}
        let t: f64 = mean / a;

        parameters.push(a);
        parameters.push(t);

        return parameters;
    }
}

impl Iterator for GammaGenerator {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        // similar implenentation as [Gamma::sample] but better.
        // removed comments

        // https://en.wikipedia.org/wiki/Gamma_distribution#Random_variate_generation
        // https://github.com/numpy/numpy/blob/main/numpy/random/src/distributions/distributions.c#L220

        if self.alpha == 1.0 {
            return self.exp.next();
        }

        assert!(self.alpha != 0.0 && self.alpha != 1.0);

        let r: f64 = if 1.0 < self.alpha {
            'gen: loop {
                let u: f64 = self.rng.gen::<f64>();
                let v: f64 = self.exp.next().unwrap();

                if u <= self.alpha {
                    let x: f64 = u.powf(self.inv_a);
                    if x <= v {
                        break 'gen x;
                    }
                } else {
                    let y: f64 = -(u * self.inv_a).ln();
                    let x: f64 = (1.0 - self.alpha + self.alpha * y).powf(self.inv_a);

                    if x <= (v + y) {
                        break 'gen x;
                    }
                }
            }
        } else {
            'gen: loop {
                let mut x: f64;
                let mut v: f64;
                's: loop {
                    x = self.norm.next().unwrap();
                    v = 1.0 + self.c * x;
                    if v <= 0.0 {
                        break 's;
                    }
                }
                v = v * v * v;
                let u: f64 = self.rng.gen::<f64>();

                let x_sq: f64 = x * x;
                if u < 1.0 - 0.0331 * x_sq * x_sq {
                    break 'gen self.b * v;
                }

                if u.ln() < 0.5 * x_sq + self.b * (1.0 - v + v.ln()) {
                    break 'gen self.b * v;
                }
            }
        };

        return Some(r * self.theta);
    }
}

impl Default for Gamma {
    fn default() -> Self {
        Gamma::new(1.0, 1.0).unwrap()
    }
}
