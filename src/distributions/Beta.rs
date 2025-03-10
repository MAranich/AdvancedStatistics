//! # Beta distribution
//!
//! The [Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution)
//! is a continuous probability distribution. It's domain is just `[0.0, 1.0]`
//! (wich can be uscefull to model probabilities). It has 2 parameters:
//! alpha and beta.
//!
//! It is the conjugate prior of the following distributions:
//!  - [Bernoulli](crate::distributions::Bernoulli),
//!  - [Binomial](crate::distributions::Binomial),
//!  - [Negative binomial](crate::distributions:: ...todo... )
//!  - [Geometric](crate::distributions::Geometric)
//!
//!

use rand::Rng;

use crate::{
    distribution_trait::{Distribution, Parametric},
    domain::ContinuousDomain,
    euclid::{self, digamma},
};

pub const BETA_DOMAIN: ContinuousDomain = ContinuousDomain::Range(0.0, 1.0);
#[derive(Debug, Clone, PartialEq)]
pub struct Beta {
    alpha: f64,
    beta: f64,
    normalitzation_constant: f64,
}

impl Beta {
    /// Creates a new [Beta] distribution with parameters `alpha` and `beta`.
    ///
    ///  - `alpha` must be finite and be stricly positive.
    ///  - `beta` must be finite and be stricly positive.
    ///
    /// Otherwise an error will be returned.
    pub fn new(alpha: f64, beta: f64) -> Result<Beta, ()> {
        if !alpha.is_finite() {
            return Err(());
        }

        if !beta.is_finite() {
            return Err(());
        }

        if alpha <= 0.0 {
            return Err(());
        }

        if beta <= 0.0 {
            return Err(());
        }

        let norm_ct: f64 = Beta::compute_normalitzation_constant(alpha, beta);

        return Ok(Beta {
            alpha,
            beta,
            normalitzation_constant: norm_ct,
        });
    }

    fn compute_normalitzation_constant(alpha: f64, beta: f64) -> f64 {
        // Assumes alpha and beta are valid

        /*
               Computation:
           The normalitzation constant is:
           norm = 1/B(alpha, beta) = Gamma(alpha + beta) / (Gamma(alpha)*Gamma(beta))

           However, the raw computation could overflow or be imprecise. Therefore
           we will compute:

           ln(norm) = ln( Gamma(alpha + beta) / (Gamma(alpha)*Gamma(beta)) )
            = ln(Gamma(alpha + beta)) - ln(Gamma(alpha)) - ln(Gamma(beta)))

           and exponentiate the result.


           ***************************
           OR INTEGRATE BETWEEN 0 AND 1

        */

        let ln_c: f64 =
            euclid::ln_gamma(alpha + beta) - euclid::ln_gamma(alpha) - euclid::ln_gamma(beta);

        return ln_c.exp();
    }

    /// Creates a new [Beta] distribution with parameters `alpha` and `beta`.
    ///
    ///  - `alpha` must be finite and be stricly positive.
    ///  - `beta` must be finite and be stricly positive.
    ///
    /// Otherwise an invalid Beta will be returned.
    pub unsafe fn new_unchecked(alpha: f64, beta: f64) -> Beta {
        let norm_ct: f64 = Beta::compute_normalitzation_constant(alpha, beta);

        return Beta {
            alpha,
            beta,
            normalitzation_constant: norm_ct,
        };
    }

    pub const fn get_alpha(&self) -> f64 {
        return self.alpha;
    }

    pub const fn get_beta(&self) -> f64 {
        return self.beta;
    }
}

impl Distribution for Beta {
    fn pdf(&self, x: f64) -> f64 {
        // let a = alpha, let b = beta for conciseness sake.
        // pdf(x | a, b) = Gamma(a+b-1)/(Gamma(a-1)*Gamma(b-1)) * x^(a-1) * (1 - x)^(b-1)
        // pdf(x | a, b) = norm * x^(a-1) * (1 - x)^(b-1)

        let pow_alpha: f64 = x.powf(self.alpha - 1.0);
        let pow_beta: f64 = (1.0 - x).powf(self.beta - 1.0);
        return self.normalitzation_constant * pow_alpha * pow_beta;
    }

    fn get_domain(&self) -> &crate::domain::ContinuousDomain {
        return &BETA_DOMAIN;
    }

    fn sample(&self) -> f64 {
        let aux: Vec<f64> = self.sample_multiple(1);
        return aux[0];
    }

    fn cdf_multiple(&self, points: &[f64]) -> Vec<f64> {
        /*
           Plan for Beta:

           > pdf(x | a, b) = norm * x^(a-1) * (1 - x)^(b-1)

           cdf(x | a, b) = integral{x: -inf->inf} pdf(x | a, b) dx
           cdf(x | a, b) = integral{x: 0->1} norm * x^(a-1) * (1 - x)^(b-1) dx

           We will reuse the deafult implementation but tune it for the Beta.

        */
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
                std::panic!("Found NaN in `cdf_multiple` of Beta. \n");
                // return Err(AdvStatError::NanErr);
            }
        }

        let mut ret: Vec<f64> = std::vec![0.0; points.len()];
        let bounds: (f64, f64) = (0.0, 1.0);
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
        // https://en.wikipedia.org/wiki/Beta_distribution#Random_variate_generation

        let mut gamma_alpha_samples: Vec<f64> = {
            let gamma_alpha: super::Gamma::Gamma =
                super::Gamma::Gamma::new(self.alpha, 1.0).unwrap();
            gamma_alpha.sample_multiple(n)
        };
        let gamma_beta_samples: Vec<f64> = {
            let gamma_beta: super::Gamma::Gamma = super::Gamma::Gamma::new(self.beta, 1.0).unwrap();
            gamma_beta.sample_multiple(n)
        };

        // all unsafe accesses are safe because of the following assert
        // wew will reuse the vector `gamma_alpha_samples` so we don't do an extra allocation
        assert!(gamma_alpha_samples.len() == n && gamma_beta_samples.len() == n);
        for i in 0..n {
            let a: f64 = unsafe { *gamma_alpha_samples.get_unchecked(i) };
            let b: f64 = unsafe { *gamma_beta_samples.get_unchecked(i) };
            let reference: &mut f64 = unsafe { gamma_alpha_samples.get_unchecked_mut(i) };

            *reference = a / (a + b);
        }

        return gamma_alpha_samples;
    }

    fn quantile_multiple(&self, points: &[f64]) -> Vec<f64> {
        /*
            Plan for Beta:

            > pdf(x | a, b) = norm * x^(a-1) * (1 - x)^(b-1)

            We will reuse the deafult implementation but tune it for the Beta.

        */

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
                std::panic!("Found NaN in `quantile_multiple` of Beta. \n");
            }
        }

        let mut ret: Vec<f64> = std::vec![-0.0; points.len()];
        let bounds: (f64, f64) = (0.0, 1.0);
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
        return Some(self.alpha / (self.alpha + self.beta));
    }

    fn variance(&self) -> Option<f64> {
        let ab: f64 = self.alpha + self.beta;
        return Some(self.alpha * self.beta / (ab * ab * (ab + 1.0)));
    }

    fn mode(&self) -> f64 {
        // alpha anb beta are non-nans, unwrap is safe
        match (
            self.alpha.partial_cmp(&1.0).unwrap(),
            self.beta.partial_cmp(&1.0).unwrap(),
        ) {
            (std::cmp::Ordering::Equal, std::cmp::Ordering::Equal) => 0.5,
            (std::cmp::Ordering::Less, _) => 0.0,
            (_, std::cmp::Ordering::Less) => 1.0,
            _ => {
                // "normal" case
                (self.alpha - 1.0) / (self.alpha + self.beta - 2.0)
            }
        }
    }

    // default median

    fn skewness(&self) -> Option<f64> {
        // formula from wiki
        // Sk = 2*(b-a) * sqrt(a+b+1) / ((a+b+2) * sqrt(a*b))
        // = 2*(b-a) * sqrt(a+b+1) * sqrt(a*b) / ((a+b+2) * sqrt(a*b) * sqrt(a*b))
        // = 2*(b-a) * sqrt(a+b+1) * sqrt(a*b) / ((a+b+2) * a*b)
        // = 2*(b-a) * sqrt((a+b+1) * a*b) / ((a+b+2) * a*b)
        // This formula is better because we do 1 sqrt less

        // = (a+b+1) * a*b
        let inner_sqrt: f64 = (self.alpha + self.beta + 1.0) * self.alpha * self.beta;
        let num: f64 = 2.0 * (self.beta - self.alpha) * inner_sqrt.sqrt();
        let den: f64 = (self.alpha + self.beta + 2.0) * self.alpha * self.beta;
        return Some(num / den);
    }

    fn kurtosis(&self) -> Option<f64> {
        return self.excess_kurtosis().map(|x| x + 3.0);
    }

    fn excess_kurtosis(&self) -> Option<f64> {
        let sum_a_b: f64 = self.alpha + self.beta;
        let sub_a_b: f64 = self.alpha - self.beta;
        let mul_a_b: f64 = self.alpha * self.beta;

        let num: f64 = sub_a_b * sub_a_b * (sum_a_b + 1.0) - mul_a_b * (sub_a_b + 2.0);
        let den: f64 = mul_a_b * (sub_a_b + 2.0) * (sub_a_b + 3.0);

        return Some(6.0 * num / den);
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
            let ab: f64 = self.alpha + self.beta;
            let mut acc: f64 = 1.0;
            for R in 0..order {
                let r: f64 = R as f64;
                acc = acc * (self.alpha + r) / (ab + r);
            }
            return acc;
        }

        let bounds: (f64, f64) = (0.0, 1.0);

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
        let (_, num_steps): (f64, usize) = euclid::choose_integration_precision_and_steps(bounds);

        let moment: f64 = {
            let integration_fn = |x: f64| {
                let std_inp: f64 = (x + minus_mean) * inv_std_dev;
                std_inp.powi(order_exp) * self.pdf(x)
            };

            euclid::numerical_integration_finite(integration_fn, bounds, num_steps as u64)
        };

        return moment;
    }

    fn entropy(&self) -> f64 {
        // https://en.wikipedia.org/wiki/Beta_distribution#Quantities_of_information_(entropy)

        // B(a, b) = gamma(a+b) / (gamma(a)*gamma(b)) = self.normalitzation_constant
        // ln(B(a, b)) - (a-1)*digamma(a) - (b-1)*digamma(b) + (a+b-2)*digamma(a+b)

        let term_1: f64 = self.normalitzation_constant.ln();
        let term_2: f64 = -(self.alpha - 1.0) * digamma(self.alpha);
        let term_3: f64 = -(self.beta - 1.0) * digamma(self.beta);
        let term_4: f64 = (self.alpha + self.beta - 2.0) * digamma(self.alpha + self.beta);

        return term_1 + term_2 + term_3 + term_4;
    }

    fn rejection_sample(&self, n: usize, pdf_max: f64) -> Vec<f64> {
        // Small modifications for the information that we know of beta.
        // Should be very effitient if alpah and beta are greater or equal to 1.0
        let mut rng: rand::prelude::ThreadRng = rand::rng();

        let mut ret: Vec<f64> = Vec::with_capacity(n);
        for _ in 0..n {
            let sample: f64 = loop {
                let x: f64 = rng.random();
                let y: f64 = rng.random();
                if y * pdf_max < self.pdf(x) {
                    break x;
                }
            };
            ret.push(sample);
        }

        return ret;
    }

    fn rejection_sample_range(&self, n: usize, pdf_max: f64, range: (f64, f64)) -> Vec<f64> {
        let mut rng: rand::prelude::ThreadRng = rand::rng();
        let range_magnitude: f64 = range.1 - range.0;

        if range_magnitude.is_sign_negative() || range.0 < 0.0 || 1.0 < range.1 {
            // possible early return
            return Vec::new();
        }

        let mut ret: Vec<f64> = Vec::with_capacity(n);
        for _ in 0..n {
            let sample: f64 = loop {
                let mut x: f64 = rng.random();
                x = range.0 + x * range_magnitude;
                let y: f64 = rng.random();
                if y * pdf_max < self.pdf(x) {
                    break x;
                }
            };
            ret.push(sample);
        }

        return ret;
    }
}

impl Parametric for Beta {
    /// Evaluates the [Beta::pdf] in a general way taking into account
    /// the parameters.
    ///
    /// ### Parameters for Beta:
    ///
    /// The [Beta] distribution has 2 parameters `alpha = a` and `beta = b`.
    /// The order of the `parameters` is:
    ///
    /// > \[alpha, beta\]
    ///
    /// Alpha and beta must be both stricly positive.
    fn general_pdf(&self, x: f64, parameters: &[f64]) -> f64 {
        // let a = alpha, let b = beta for conciseness sake.
        // pdf(x | a, b) = Gamma(a+b-1)/(Gamma(a-1)*Gamma(b-1)) * x^(a-1) * (1 - x)^(b-1)
        // pdf(x | a, b) = norm * x^(a-1) * (1 - x)^(b-1)

        let a: f64 = parameters[0];
        let b: f64 = parameters[1];
        let norm: f64 = Beta::compute_normalitzation_constant(a, b);

        let pow_alpha: f64 = x.powf(a - 1.0);
        let pow_beta: f64 = (1.0 - x).powf(b - 1.0);
        return norm * pow_alpha * pow_beta;
    }

    fn number_of_parameters() -> u16 {
        return 2;
    }

    fn get_parameters(&self, parameters: &mut [f64]) {
        parameters[0] = self.alpha;
        parameters[1] = self.beta;
    }

    fn derivative_pdf_parameters(&self, x: f64, parameters: &[f64]) -> Vec<f64> {
        // d/dx ln(f(x)) = f'(x)/f(x)
        // => f(x) * d/dx ln(f(x)) = f'(x)

        // Reserve a vector with exacly 3 elements
        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(3);

        let a: f64 = parameters[0];
        let b: f64 = parameters[1];
        let digamma_ab: f64 = euclid::digamma(a + b - 1.0);
        let pdf: f64 = self.general_pdf(x, parameters);

        {
            // ## Derivative respect to x:
            /*
               d/dx pdf(x | a, b) = d/dx norm * x^(a-1) * (1 - x)^(b-1)
                = norm * d/dx x^(a-1) * (1 - x)^(b-1)
                = norm * ( d/dx[ x^(a-1) ] * (1 - x)^(b-1) + x^(a-1) * d/dx[ (1 - x)^(b-1) ] )
                = norm * ( (a-1) * x^(a-2) * (1 - x)^(b-1) + x^(a-1) * (b-1) * (1 - x)^(b-2) d/dx[ 1 - x ] )
                = norm * ( (a-1) * x^(a-2) * (1 - x)^(b-1) + x^(a-1) * (b-1) * (1 - x)^(b-2) * -1 )
                = norm * x^(a-2) * ( (a-1) * (1 - x)^(b-1) + x * (b-1) * (1 - x)^(b-2) * -1 )
                = norm * x^(a-2) * (1 - x)^(b-2) * ( (a-1) * (1 - x) + x * (b-1) * -1 )
                = norm * x^(a-2) * (1 - x)^(b-2) * ( (a-1) * (1 - x) - (b-1) * x )

               Confirmation:

               f(x) * d/dx ln(f(x)) = f'(x)
               d/dx ln(pdf(x | a, b)) = (a-1)/x - (b-1)/(1 - x)

               norm * x^(a-1) * (1 - x)^(b-1) * ((a-1)/x - (b-1)/(1 - x)) =
                = norm * (x^(a-1) * (1 - x)^(b-1) * (a-1)/x - x^(a-1) * (1 - x)^(b-1) * (b-1)/(1 - x))
                = norm * (x^(a-2) * (1 - x)^(b-1) * (a-1) - x^(a-1) * (1 - x)^(b-2) * (b-1))
                = norm * (x^(a-2) * (1 - x)^(b-2) * (1 - x) * (a-1) - x^(a-2) * x * (1 - x)^(b-2) * (b-1))
                = norm * x^(a-2) * (1 - x)^(b-2) * ((1 - x) * (a-1) - x * (b-1))
                = norm * x^(a-2) * (1 - x)^(b-2) * ( (a-1) * (1 - x) - (b-1) * x )

               Wich is identical to our result for the normal computation.

            */

            // = (a-1) * (1 - x) - (b-1) * x
            let term: f64 = (self.alpha - 1.0) * (1.0 - x) - (self.beta - 1.0) * x;

            let alpha_pow: f64 = x.powf(self.alpha - 2.0);
            let beta_pow: f64 = (1.0 - x).powf(self.beta - 2.0);

            ret.push(self.normalitzation_constant * alpha_pow * beta_pow * term);
        }

        {
            // ## Derivative respect to alpha:
            /*
                let a = alpha
                let b = beta
                let norm(a) = Gamma(a+b-1)/(Gamma(a-1)*Gamma(b-1))

                d/da pdf(x | a, b) = d/da[ norm(a) * x^(a-1) * (1 - x)^(b-1) ]
                 = (1 - x)^(b-1) * d/da[ norm(a) * x^(a-1) ]
                 = (1 - x)^(b-1) * (d/da[ norm(a) ] * x^(a-1) + norm(a) * d/da[ exp((a-1) * ln(x)) ])
                 = (1 - x)^(b-1) * ( norm'(a) * x^(a-1) + norm(a) * x^(a-1) * ln(x) )

                Definition: d/dx Gamma(x) = Gamma(x) * PolyGamma_0(x)

                norm'(a) = d/da norm(a) = d/da Gamma(a+b-1)/(Gamma(a-1)*Gamma(b-1))
                 = d/da[ 1/Gamma(b-1) * Gamma(a+b-1)/Gamma(a-1) ]
                 = 1/Gamma(b-1) * d/da[ Gamma(a+b-1)/Gamma(a-1) ]
                 = 1/Gamma(b-1) * (d/da[ Gamma(a+b-1) ] * Gamma(a-1) - d/da[ Gamma(a-1) ] * Gamma(a+b-1)) / Gamma(a-1)^2
                 = 1/(Gamma(b-1) * Gamma(a-1)^2) * (d/da[ Gamma(a+b-1) ] * Gamma(a-1) - d/da[ Gamma(a-1) ] * Gamma(a+b-1))
                 = 1/(Gamma(b-1) * Gamma(a-1)^2) * (Gamma(a+b-1) * PolyGamma_0(a+b-1) * Gamma(a-1) - Gamma(a-1) * PolyGamma_0(a - 1) * Gamma(a+b-1))
                 = 1/(Gamma(b-1) * Gamma(a-1)) * (Gamma(a+b-1) * PolyGamma_0(a+b-1) - PolyGamma_0(a - 1) * Gamma(a+b-1))
                 = Gamma(a+b-1)/(Gamma(a-1)*Gamma(b-1)) * (PolyGamma_0(a+b-1) - PolyGamma_0(a - 1))
                 = norm(a) * (PolyGamma_0(a+b-1) - PolyGamma_0(a - 1))

                // using calculator, the results for norm'(a) are correct

                d/da pdf(x | a, b) =
                 = (1 - x)^(b-1) * ( norm'(a) * x^(a-1) + norm(a) * x^(a-1) * ln(x) )
                 = (1 - x)^(b-1) * ( norm(a) * (PolyGamma_0(a+b-1) - PolyGamma_0(a - 1)) * x^(a-1) + norm(a) * x^(a-1) * ln(x) )
                 = (1 - x)^(b-1) * norm(a) * ((PolyGamma_0(a+b-1) - PolyGamma_0(a - 1)) * x^(a-1) + x^(a-1) * ln(x) )
                 = (1 - x)^(b-1) * norm(a) * x^(a-1) * ((PolyGamma_0(a+b-1) - PolyGamma_0(a - 1)) + ln(x) )
                 = norm(a) * x^(a-1) * (1 - x)^(b-1) * (PolyGamma_0(a+b-1) - PolyGamma_0(a - 1) + ln(x) )
                 = pdf(x | a, b) * (PolyGamma_0(a+b-1) - PolyGamma_0(a - 1) + ln(x) )

                // using calculator, the final results for d/da pdf(x | a, b) are correct.

                Confirmation:

                f(x) * d/dx ln(f(x)) = f'(x)
                d/da ln(pdf(x | a, b)) = Digamma(a+b-1) - Digamma(a-1) + ln(x)

                norm(a) * x^(a-1) * (1 - x)^(b-1) * (Digamma(a+b-1) - Digamma(a-1) + ln(x))

                Note that Digamma(x) = PolyGamma_0(x)

                norm(a) * x^(a-1) * (1 - x)^(b-1) * (PolyGamma_0(a+b-1) - PolyGamma_0(a-1) + ln(x))
                pdf(x | a, b) * (PolyGamma_0(a+b-1) - PolyGamma_0(a - 1) + ln(x) )

                Wich is identical to our result for the normal computation.
            */

            // = PolyGamma_0(a+b-1) - PolyGamma_0(a - 1) + ln(x)
            let term: f64 = digamma_ab - euclid::digamma(a - 1.0) + x.ln();

            ret.push(pdf * term);
        }

        {
            // ## Derivative respect to beta:
            /*
                let a = alpha
                let b = beta
                let norm(b) = Gamma(a+b-1)/(Gamma(a-1)*Gamma(b-1))

                d/db pdf(x | a, b) = d/db[ norm(b) * x^(a-1) * (1 - x)^(b-1) ]
                 = x^(a-1) * d/db[ norm(b) * (1 - x)^(b-1) ]
                 = x^(a-1) * (norm'(b) * (1 - x)^(b-1) + norm(b) * d/db[ (1 - x)^(b-1) ])
                 = x^(a-1) * (norm'(b) * (1 - x)^(b-1) + norm(b) * (1 - x)^(b-1) * ln(1 - x))
                 = x^(a-1) * (1 - x)^(b-1) * (norm'(b) + norm(b) * ln(1 - x))

                Computing norm'(b):
                Definition: d/dx Gamma(x) = Gamma(x) * PolyGamma_0(x)

                norm'(b) = d/db norm(b) = d/db Gamma(a+b-1)/(Gamma(a-1)*Gamma(b-1))
                 = 1/Gamma(a-1) * d/db[ Gamma(a+b-1)/Gamma(b-1) ]
                 = 1/Gamma(a-1) * (d/db[ Gamma(a+b-1) ] * Gamma(b-1) - Gamma(a+b-1) * d/db[ Gamma(b-1) ]) / Gamma(b-1)^2
                 = 1/Gamma(a-1) * (Gamma(a+b-1) * PolyGamma_0(a+b-1) * Gamma(b-1) - Gamma(a+b-1) * Gamma(b-1) * PolyGamma_0(b-1)) / Gamma(b-1)^2
                 = Gamma(a+b-1) * Gamma(b-1) / Gamma(a-1) * (PolyGamma_0(a+b-1) - PolyGamma_0(b-1)) / Gamma(b-1)^2
                 = Gamma(a+b-1)/(Gamma(a-1)*Gamma(b-1)) * (PolyGamma_0(a+b-1) - PolyGamma_0(b-1))
                 = norm(b) * (PolyGamma_0(a+b-1) - PolyGamma_0(b-1))

                d/db pdf(x | a, b) = x^(a-1) * (1 - x)^(b-1) * (norm'(b) + norm(b) * ln(1 - x))
                 = x^(a-1) * (1 - x)^(b-1) * (norm(b) * (PolyGamma_0(a+b-1) - PolyGamma_0(b-1)) + norm(b) * ln(1 - x))
                 = norm(b) * x^(a-1) * (1 - x)^(b-1) * (PolyGamma_0(a+b-1) - PolyGamma_0(b-1) + ln(1 - x))
                 = pdf(x | a, b) * (PolyGamma_0(a+b-1) - PolyGamma_0(b-1) + ln(1 - x))

                Confirmation:

                f(x) * d/dx ln(f(x)) = f'(x)
                d/db ln(pdf(x | a, b)) = Digamma(a+b-1) - Digamma(b-1) + ln(1 - x)

                norm(a) * x^(a-1) * (1 - x)^(b-1) * (Digamma(a+b-1) - Digamma(b-1) + ln(1 - x))

                Note that Digamma(x) = PolyGamma_0(x)

                norm(a) * x^(a-1) * (1 - x)^(b-1) * (PolyGamma_0(a+b-1) - PolyGamma_0(b-1) + ln(1 - x))
                pdf(x | a, b) * (PolyGamma_0(a+b-1) - PolyGamma_0(b-1) + ln(1 - x))

                Wich is identical to our result for the normal computation.

            */

            // = PolyGamma_0(a+b-1) - PolyGamma_0(b-1) + ln(1 - x)
            let term: f64 = digamma_ab - euclid::digamma(b - 1.0) + (1.0 - x).ln();

            ret.push(pdf * term);
        }

        return ret;
    }

    fn log_derivative_pdf_parameters(&self, x: f64, parameters: &[f64]) -> Vec<f64> {
        // d/dx ln(f(x)) = f'(x)/f(x)

        // pdf(x | a, b) = Gamma(a+b-1)/(Gamma(a-1)*Gamma(b-1)) * x^(a-1) * (1 - x)^(b-1)
        // ln(pdf(x | a, b)) = ln(Gamma(a+b-1)/(Gamma(a-1)*Gamma(b-1)) * x^(a-1) * (1 - x)^(b-1))
        // ln(pdf(x | a, b)) = ln(Gamma(a+b-1)) - ln(Gamma(a-1)) - ln(Gamma(b-1)) + ln(x^(a-1)) + ln((1 - x)^(b-1))
        // ln(pdf(x | a, b)) = ln(Gamma(a+b-1)) - ln(Gamma(a-1)) - ln(Gamma(b-1)) + (a-1)*ln(x) + (b-1) * ln(1 - x)

        // Note:
        // Definition: Digamma(x) = d/dx[ ln(Gamma(x)) ] = Gamma'(x) / Gamma(x)

        // Reserve a vector with exacly 3 elements
        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(3);

        let a: f64 = parameters[0];
        let b: f64 = parameters[1];
        let digamma_ab: f64 = euclid::digamma(a + b - 1.0);

        {
            // ## Log derivative respect to x:
            /*

                d/dx ln(pdf(x | a, b)) = d/dx ln(Gamma(a+b-1)) - ln(Gamma(a-1)) - ln(Gamma(b-1)) + (a-1)*ln(x) + (b-1) * ln(1 - x)
                 = 0 - 0 - 0 + (a-1)/x + (b-1)/(1 - x) * d/dx[ 1 - x ]
                 = (a-1)/x + (b-1)/(1 - x) * -1
                 = (a-1)/x - (b-1)/(1 - x)

            */

            let term_1: f64 = (parameters[0] - 1.0) / x;
            let term_2: f64 = (parameters[2] - 1.0) / (1.0 - x);

            ret.push(term_1 - term_2);
        }

        {
            // ## Log derivative respect to a:
            /*

                d/da ln(pdf(x | a, b)) = d/da ln(Gamma(a+b-1)) - ln(Gamma(a-1)) - ln(Gamma(b-1)) + (a-1)*ln(x) + (b-1) * ln(1 - x)
                 = Digamma(a+b-1) - Digamma(a-1) - 0 + ln(x) + 0
                 = Digamma(a+b-1) - Digamma(a-1) + ln(x)

            */

            ret.push(digamma_ab - euclid::digamma(a - 1.0) + x.ln());
        }

        {
            // ## Log derivative respect to beta:
            /*

                d/db ln(pdf(x | a, b)) = d/db ln(Gamma(a+b-1)) - ln(Gamma(a-1)) - ln(Gamma(b-1)) + (a-1)*ln(x) + (b-1) * ln(1 - x)
                 = Digamma(a+b-1) - 0 - Digamma(b-1) + 0 + ln(1 - x)
                 = Digamma(a+b-1) - Digamma(b-1) + ln(1 - x)

            */

            ret.push(digamma_ab - euclid::digamma(b - 1.0) + (1.0 - x).ln());
        }

        return ret;
    }

    fn parameter_restriction(&self, parameters: &mut [f64]) {
        let ep: f64 = f64::EPSILON;
        parameters[0] = parameters[0].max(ep * ep * ep);
        parameters[1] = parameters[1].max(ep * ep * ep);
    }

    fn fit(&self, data: &mut crate::Samples::Samples) -> Vec<f64> {
        /*

            Using Maximum Likelyhood estimation:
            Assuming k samples.

                Estimation of alpha:

            pdf(x | a, b) = Gamma(a+b-1)/(Gamma(a-1)*Gamma(b-1)) * x^(a-1) * (1 - x)^(b-1)
            d/da ln(pdf(x | a, b) = Digamma(a+b-1) - Digamma(a-1) + ln(x)
            0 = sumatory{x_i} Digamma(a+b-1) - Digamma(a-1) + ln(x_i)
            0 = k * Digamma(a+b-1) + sumatory{x_i} - Digamma(a-1) + ln(x_i)
            0 = k * Digamma(a+b-1) - k * Digamma(a-1) + sumatory{x_i} ln(x_i)
            k * (Digamma(a-1) - Digamma(a+b-1)) = sumatory{x_i} ln(x_i)
            Digamma(a-1) - Digamma(a+b-1) = 1/k * sumatory{x_i} ln(x_i)

                Estimation of beta:

            pdf(x | a, b) = Gamma(a+b-1)/(Gamma(a-1)*Gamma(b-1)) * x^(a-1) * (1 - x)^(b-1)
            d/db ln(pdf(x | a, b) = Digamma(a+b-1) - Digamma(b-1) + ln(1 - x)
            0 = sumatory{x_i} Digamma(a+b-1) - Digamma(b-1) + ln(1 - x_i)
            0 = k * Digamma(a+b-1) + sumatory{x_i} - Digamma(b-1) + ln(1 - x_i)
            0 = k * Digamma(a+b-1) - k * Digamma(b-1) + sumatory{x_i} ln(1 - x_i)
            k * (Digamma(b-1) - Digamma(a+b-1)) = sumatory{x_i} ln(1 - x_i)
            Digamma(b-1) - Digamma(a+b-1) = 1/k * sumatory{x_i} ln(1 - x_i)

            We have the following 2 equations:

            Digamma(a-1) - Digamma(a+b-1) = 1/k * sumatory{x_i} ln(x_i)
            Digamma(b-1) - Digamma(a+b-1) = 1/k * sumatory{x_i} ln(1 - x_i)

            Substracting them:
            Digamma(a-1) - Digamma(a+b-1) - (Digamma(b-1) - Digamma(a+b-1)) = 1/k * sumatory{x_i}[ ln(x_i) ] - 1/k * sumatory{x_i}[ ln(1 - x_i) ]
            Digamma(a-1) - Digamma(b-1) = 1/k * (sumatory{x_i}[ ln(x_i) ] - sumatory{x_i}[ ln(1 - x_i) ])
            Digamma(a-1) - Digamma(b-1) = 1/k * sumatory{x_i}[ ln(x_i) - ln(1 - x_i) ]
            Digamma(a-1) - Digamma(b-1) = 1/k * sumatory{x_i}[ ln(x_i/(1 - x_i)) ]

            However we are stuck and we cannot progress further.
            If we knew one of them we could continue. There *may* exist
            multiple values for a and b that satisfy the equation
            (possibly infinite), but we cannot easly tell wich one
            of them is optimal.
            For this reason we will use the nuerical approach.

            We will do a few simplifications.

        */

        let d: usize = 2;
        let mut parameters: Vec<f64> = Vec::new();
        parameters.reserve_exact(2);
        parameters.push(1.0);
        parameters.push(1.0);
        // default values are a = 1, b = 1

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
            // set gradient to 0
            gradient[0] = 0.0;
            gradient[1] = 0.0;

            for sample in data.peek_data() {
                let local_log_grad: Vec<f64> =
                    self.log_derivative_pdf_parameters(*sample, &parameters);

                // unrolled loop
                gradient[0] += local_log_grad[1];
                gradient[1] += local_log_grad[2];
            }

            let mut total_change: f64 = 0.0;
            // We want to maximize likelyhood, so we won't put the `-`
            // unrolled loop
            let change: f64 = learning_rate * inv_n * gradient[0];
            parameters[0] += change;
            total_change += change.abs();
            let change: f64 = learning_rate * inv_n * gradient[1];
            parameters[1] += change;
            total_change += change.abs();

            self.parameter_restriction(&mut parameters);

            if total_change < conv_diff_criteria {
                break;
            }
        }

        return parameters;
    }
}

impl Default for Beta {
    fn default() -> Self {
        // special case of https://en.wikipedia.org/wiki/Wigner_semicircle_distribution
        // a sample from ([Beta::default] * 2 - 1) * r follows the Wigner semicircle distribution
        let alpha: f64 = 1.5;
        let beta: f64 = 1.5;
        Self {
            alpha,
            beta,
            normalitzation_constant: Beta::compute_normalitzation_constant(alpha, beta),
        }
    }
}
