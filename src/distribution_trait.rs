use std::usize;

use rand::Rng;

use crate::Samples::Samples;
use crate::configuration::{self, QUANTILE_USE_NEWTONS_ITER};
use crate::domain::{ContinuousDomain, DiscreteDomain};
///! This script contains the interfaces used to comunicate with the distributions.
use crate::euclid::{self, *};
use crate::hypothesis::Hypothesis;

/// The trait for any continuous distribution.
///
/// The [Distribution::pdf] must be a simple function. If you want a piecewise function,
/// look first at [crate::mixed_distribution]
///
/// None of the provided methods are guaranteed to work if the implemented [Distribution::pdf]
/// is NOT a [valid pdf](https://en.wikipedia.org/wiki/Probability_density_function).
/// So, it needs to fullfill:
///  - The function must be stricly non-negative
///  - The function must be real valued (no `+-inf` or NaNs)
///  - The function must have a total area of 1 under the curve.
pub trait Distribution {
    //Requiered method:

    /// Evaluates the [PDF](https://en.wikipedia.org/wiki/Probability_density_function)
    /// (Probability Density function) of the distribution at point `x`. The function
    /// should not be evaluated outside the domain (because it should return 0.0 anyway).
    ///
    /// The PDF is assumed to be a valid probability distribution. It is must fullfill:
    ///  - `0.0 <= pdf(x)`
    ///  - It is normalized. (It has an area under the curbe of `1.0`)
    ///      - If you are not sure if the PDF is normalized, you can use
    /// [crate::euclid::numerical_integration].
    ///  - The function must be real valued (no `+-inf` or NaNs)
    ///  - As `x` approaches `+-inf` (if inside the domain), `pdf(x)` should
    /// tend to `0.0`.
    fn pdf(&self, x: f64) -> f64;

    /// Returns a reference to the pdf [ContinuousDomain], wich indicates at wich points
    /// the pdf can be evaluated. The returned domain should be constant and not change.
    fn get_domain(&self) -> &ContinuousDomain;

    // Provided methods:
    // Manual implementation for a specific distribution is recommended.

    /// Evaluates the [CDF](https://en.wikipedia.org/wiki/Cumulative_distribution_function)
    /// (Cumulative distribution function).
    ///
    ///  > F(x) = cdf(x) = P(X <= x) = p
    ///
    /// The cdf includes the `x` itself. If the function is evaluated outside
    /// the domain of the pdf, it will either return either `0.0` or `1.0`.
    /// **Panicks** is `x` is a NaN.
    ///
    /// Note that the deafult implemetation requieres numerical integration and
    /// may be expensive.
    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            // x is not valid
            panic!("Tried to evaluate the cdf function with a NaN value. \n");
        }
        let aux: [f64; 1] = [x];
        let aux_2: Vec<f64> = self.cdf_multiple(&aux);
        return aux_2[0];
    }

    /// Samples the distribution at random.
    ///
    /// The deafult method is [Inverse transform sampling](https://en.wikipedia.org/wiki/Inverse_transform_sampling)
    /// unless the deadult method is overriden. Inverse transform sampling simply
    /// generates a random uniform number and evaluates the inverse cdf function
    /// (the [Distribution::quantile] function) and returns the result.
    ///
    /// Note that the deafult implemetation requieres numerical integration and
    /// may be expensive. The method [Distribution::sample_multiple] is more
    /// effitient for multiple sampling.
    fn sample(&self) -> f64 {
        let aux: Vec<f64> = self.sample_multiple(1);
        return aux[0];
    }

    /// Evaluates the [quantile function](https://en.wikipedia.org/wiki/Quantile_function).
    ///
    /// If the cdf is:
    ///
    ///  > F(x) = cdf(x) = P(X <= x) = p
    ///
    /// Then the quantile function is:
    ///
    ///  > Q(p) = x = F^-1(p)
    ///
    ///  - if `x` is outside the range [0.0, 1.0], the respective bound of the domain
    /// will be returned.
    ///  - **Panicks** is `x` is a NaN.
    ///
    /// The quantile function is the inverse function of [Distribution::cdf]. Note that
    /// the deafult implemetation requieres numerical integration and may be expensive.
    ///
    /// Also, if you are considering calling this function multiple times, use
    /// [Distribution::quantile_multiple] for better performance.
    fn quantile(&self, x: f64) -> f64 {
        // just call [Distribution::quantile_multiple]

        if x.is_nan() {
            // x is not valid
            panic!("Tried to evaluate the quantile function with a NaN value. \n");
        }

        let value: [f64; 1] = [x];
        let quantile_vec: Vec<f64> = self.quantile_multiple(&value);
        return quantile_vec[0];
    }

    // Multiple variants.
    // They are the same as the normal functions, but if they are overriden they may
    // provide a computational advantage.


    /// Evaluates the [CDF](https://en.wikipedia.org/wiki/Cumulative_distribution_function)
    /// (Cumulative distribution function) on multiple points.
    ///
    ///  > F(x) = cdf(x) = P(X <= x) = p
    ///
    /// The cdf includes the `x` itself. If the function is evaluated outside
    /// the domain of the pdf, it will either return either `0.0` or `1.0`.
    /// **Panicks** is `x` is a NaN.
    ///
    /// Note that the deafult implemetation requieres numerical integration and
    /// may be expensive.
    /// 
    /// ***
    /// 
    /// cdf_multiple allows to evaluate the [Distribution::cdf] at multiple points.
    /// It may provide a computational advantage.  
    ///
    /// If an effitient [Distribution::cdf] has been implemented, it can be replaced for:
    ///
    /// ```
    /// fn cdf_multiple(&self, points: &[f64]) -> Vec<f64> {
    ///     points.iter().map(|x| self.cdf(*x)).collect::<Vec<f64>>()
    /// }
    /// ```
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
                panic!("Found NaN in `cdf_multiple`. \n");
            }
        }

        let mut ret: Vec<f64> = vec![0.0; points.len()];
        let domain: &ContinuousDomain = self.get_domain();
        let bounds: (f64, f64) = domain.get_bounds();
        let integration_type: IntegrationType = IntegrationType::from_bounds(bounds);
        let mut sorted_indicies: Vec<usize> = (0..points.len()).into_iter().collect::<Vec<usize>>();

        sorted_indicies.sort_unstable_by(|&i, &j| {
            let a: f64 = points[i];
            let b: f64 = points[j];
            if let IntegrationType::InfiniteToConst = integration_type {
                // sort in reverse
                b.partial_cmp(&a).unwrap()
            } else {
                a.partial_cmp(&b).unwrap()
            }
        });

        
        let (step_length, max_iters): (f64, usize) = {
            let doing_substitutuon: bool = if let IntegrationType::FullInfinite = integration_type {
                true
            } else {false};
            choose_integration_precision_and_steps(bounds, doing_substitutuon)
        };
        let half_step_length: f64 = 0.5 * step_length;
        let step_len_over_6: f64 = step_length / 6.0;

        let mut idx_iter: std::vec::IntoIter<usize> = sorted_indicies.into_iter();
        let mut current_index: usize = idx_iter.next().unwrap();
        // ^unwrap is safe

        let mut current_cdf_point: f64 = points[current_index];

        let mut num_step: f64 = 0.0;
        let mut accumulator: f64 = 0.0;

        // estimate the bound likelyhood with the next 2 values
        let mut last_pdf_evaluation: f64 = match integration_type {
            IntegrationType::Finite | IntegrationType::ConstToInfinite => {
                let middle: f64 = self.pdf(bounds.0 + half_step_length);
                let end: f64 = self.pdf(bounds.0 + step_length);
                2.0 * middle - end
            }
            IntegrationType::InfiniteToConst => {
                let middle: f64 = self.pdf(bounds.1 - half_step_length);
                let end: f64 = self.pdf(bounds.1 - step_length);
                2.0 * middle - end
            }
            IntegrationType::FullInfinite => 0.0,
        };

        for _ in 0..max_iters {
            let current_position: f64;

            match integration_type {
                IntegrationType::Finite | IntegrationType::ConstToInfinite => {
                    current_position = bounds.0 + step_length * num_step;
                    while current_cdf_point < current_position {
                        ret[current_index] = accumulator;

                        // update `current_cdf_point` to the next value or exit if we are done
                        match idx_iter.next() {
                            Some(v) => current_index = v,
                            None => return ret,
                        }
                        current_cdf_point = points[current_index];
                    }
                }
                IntegrationType::InfiniteToConst => {
                    current_position = bounds.1 - step_length * num_step;
                    while current_position < current_cdf_point {
                        ret[current_index] = 1.0 - accumulator;

                        // update `current_cdf_point` to the next value or exit if we are done
                        match idx_iter.next() {
                            Some(v) => current_index = v,
                            None => return ret,
                        }
                        current_cdf_point = points[current_index];
                    }
                }
                IntegrationType::FullInfinite => {
                    // integral {-inf -> inf} f(x) dx = integral {-1 -> 1} f( t / (1-t^2) ) * (1 + t^2) / (1 - t^2)^2  dt

                    /*
                       Note that the real `current_position` (before the change of variables)
                       is `t/(1-t^2)` where t = `current_position`. Therefore the check
                       `current_cdf_point <= current_position` becomes:
                       let t = current_position;
                       current_cdf_point <= t/(1 - t * t)     =>
                       current_cdf_point * (1 - t * t) <= t   =>
                       current_cdf_point * u <= t
                    */

                    current_position = bounds.0 + step_length * num_step;
                    let u: f64 = 1.0 - current_position * current_position;

                    while current_cdf_point * u < current_position {
                        ret[current_index] = accumulator;

                        // update `current_cdf_point` to the next value or exit if we are done
                        match idx_iter.next() {
                            Some(v) => current_index = v,
                            None => return ret,
                        }
                        current_cdf_point = points[current_index];
                    }
                }
            };

            let (middle, end): (f64, f64) = match integration_type {
                IntegrationType::Finite | IntegrationType::ConstToInfinite => {
                    let _middle: f64 = self.pdf(current_position + half_step_length);
                    let _end: f64 = self.pdf(current_position + step_length);
                    (_middle, _end)
                }
                IntegrationType::InfiniteToConst => {
                    let _middle: f64 = self.pdf(current_position - half_step_length);
                    let _end: f64 = self.pdf(current_position - step_length);
                    (_middle, _end)
                }
                IntegrationType::FullInfinite => {
                    // integral {-inf -> inf} f(x) dx = integral {-1 -> 1} f( t / (1-t^2) ) * (1 + t^2) / (1 - t^2)^2  dt

                    let _middle: f64 = {
                        let t: f64 = current_position + half_step_length;
                        let u: f64 = 1.0 / (1.0 - t * t);
                        let v: f64 = 1.0 + t * t;
                        self.pdf(t * u) * v * u * u
                    };
                    let _end: f64 = {
                        let t: f64 = current_position + step_length;
                        let e: f64 = 1.0 - t * t;
                        if e.abs() < f64::EPSILON {
                            0.0
                        } else {
                            let u: f64 = 1.0 / e;
                            let v: f64 = 1.0 + t * t;
                            self.pdf(t * u) * v * u * u
                        }
                    };
                    (_middle, _end)
                }
            };

            accumulator += step_len_over_6 * (last_pdf_evaluation + 4.0 * middle + end);

            last_pdf_evaluation = end;
            num_step += 1.0;
        }

        ret[current_index] = if let IntegrationType::InfiniteToConst = integration_type {
            // this really should never happen
            1.0 - accumulator
        } else {
            accumulator
        };

        for idx in idx_iter {
            // use all remaining indicies
            if let IntegrationType::InfiniteToConst = integration_type {
                // this really should never happen
                ret[idx] = 1.0 - accumulator;
            } else {
                ret[idx] = accumulator;
            }
        }

        return ret;
    }


    /// Samples the distribution at random multiple times.
    ///
    /// The deafult method is [Inverse transform sampling](https://en.wikipedia.org/wiki/Inverse_transform_sampling)
    /// unless the deadult method is overriden. Inverse transform sampling simply
    /// generates a random uniform number and evaluates the inverse cdf function
    /// (the [Distribution::quantile] function) and returns the result.
    ///
    /// Note that the deafult implemetation requieres numerical integration and
    /// may be expensive. 
    /// 
    /// ***
    /// 
    /// [Distribution::sample_multiple] allows to evaluate the [Distribution::sample]
    /// at multiple points. It may provide a computational advantage in comparasion 
    /// to [Distribution::sample].
    ///
    /// The deafult implementation uses the [Distribution::quantile_multiple] function,
    /// wich may be expensive. Consider using [Distribution::rejection_sample] or 
    /// [Distribution::rejection_sample_range] if possible.
    ///
    /// If an effitient [Distribution::sample] has been implemented, it can be replaced for:
    ///
    /// ```
    /// fn sample_multiple(&self, n: usize) -> Vec<f64> {
    ///     (0..n).map(|_| self.sample()).collect::<Vec<f64>>()
    /// }
    /// ```
    fn sample_multiple(&self, n: usize) -> Vec<f64> {
        let mut rng: rand::prelude::ThreadRng = rand::rng();
        let mut rand_quantiles: Vec<f64> = vec![0.0; n];
        rng.fill(rand_quantiles.as_mut_slice());

        let ret: Vec<f64> = self.quantile_multiple(&rand_quantiles);

        return ret;
    }


    /// Evaluates the [quantile function](https://en.wikipedia.org/wiki/Quantile_function) 
    /// multiple times.
    ///
    /// If the cdf is:
    ///
    ///  > F(x) = cdf(x) = P(X <= x) = p
    ///
    /// Then the quantile function is:
    ///
    ///  > Q(p) = x = F^-1(p)
    ///
    ///  - if `x` is outside the range [0.0, 1.0], the respective bound of the domain
    /// will be returned.
    ///  - **Panicks** is `x` is a NaN.
    ///
    /// The quantile function is the inverse function of [Distribution::cdf_multiple]. Note that
    /// the deafult implemetation requieres numerical integration and may be expensive.
    ///
    /// ***
    /// 
    /// [Distribution::quantile_multiple] acts the same as [Distribution::quantile] but 
    /// on multiple points. It provides a computational advantage over calling the 
    /// normal [Distribution::quantile] multiple times.
    ///
    /// If an effitient [Distribution::quantile] has been implemented, it can be replaced for:
    ///
    /// ```
    /// fn quantile_multiple(&self, points: &[f64]) -> Vec<f64> {
    ///     return points
    ///         .iter()
    ///         .map(|&x| self.quantile(x))
    ///         .collect::<Vec<f64>>();
    /// }
    /// ```
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
                panic!("Found NaN in `quantile_multiple`. \n");
            }
        }

        let mut ret: Vec<f64> = vec![-0.0; points.len()];
        let domain: &ContinuousDomain = self.get_domain();
        let bounds: (f64, f64) = domain.get_bounds();
        let integration_type: IntegrationType = IntegrationType::from_bounds(bounds);
        let mut sorted_indicies: Vec<usize> = (0..points.len()).into_iter().collect::<Vec<usize>>();

        sorted_indicies.sort_unstable_by(|&i, &j| {
            let a: f64 = points[i];
            let b: f64 = points[j];
            if let IntegrationType::InfiniteToConst = integration_type {
                // sort in reverse
                b.partial_cmp(&a).unwrap()
            } else {
                a.partial_cmp(&b).unwrap()
            }
        });
        let (step_length, max_iters): (f64, usize) = {
            let doing_substitutuon: bool = if let IntegrationType::FullInfinite = integration_type {
                true
            } else {false};
            choose_integration_precision_and_steps(bounds, doing_substitutuon)
        };
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
        let mut last_pdf_evaluation: f64 = match integration_type {
            IntegrationType::Finite | IntegrationType::ConstToInfinite => {
                let middle: f64 = self.pdf(bounds.0 + half_step_length);
                let end: f64 = self.pdf(bounds.0 + step_length);
                2.0 * middle - end
            }
            IntegrationType::InfiniteToConst => {
                let middle: f64 = self.pdf(bounds.1 - half_step_length);
                let end: f64 = self.pdf(bounds.1 - step_length);
                2.0 * middle - end
            }
            IntegrationType::FullInfinite => 0.0,
        };

        let use_newtons_method: bool = unsafe { QUANTILE_USE_NEWTONS_ITER };

        'integration_loop: for _ in 0..max_iters {
            let current_position: f64;

            match integration_type {
                IntegrationType::Finite | IntegrationType::ConstToInfinite => {
                    current_position = bounds.0 + step_length * num_step;
                    while current_quantile < accumulator {
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
                }
                IntegrationType::InfiniteToConst => {
                    current_position = bounds.1 - step_length * num_step;
                    while 1.0 - accumulator < current_quantile {
                        let mut quantile: f64 = current_position;

                        let pdf_q: f64 = self.pdf(quantile);
                        if use_newtons_method && !(pdf_q.abs() < f64::EPSILON) {
                            // if pdf_q is essentially 0, skip this.
                            // newton's iteration
                            quantile += -((1.0 - accumulator) - current_quantile) / pdf_q;
                        }

                        ret[current_index] = quantile;

                        // update `current_cdf_point` to the next value or exit if we are done
                        match idx_iter.next() {
                            Some(v) => current_index = v,
                            None => return ret,
                        }
                        current_quantile = points[current_index];
                    }
                }
                IntegrationType::FullInfinite => {
                    // integral {-inf -> inf} f(x) dx = integral {-1 -> 1} f( t / (1-t^2) ) * (1 + t^2) / (1 - t^2)^2  dt

                    current_position = bounds.0 + step_length * num_step;
                    while current_quantile < accumulator {
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
                }
            };

            let (middle, end): (f64, f64) = match integration_type {
                IntegrationType::Finite | IntegrationType::ConstToInfinite => {
                    let _middle: f64 = self.pdf(current_position + half_step_length);
                    let _end: f64 = self.pdf(current_position + step_length);
                    (_middle, _end)
                }
                IntegrationType::InfiniteToConst => {
                    let _middle: f64 = self.pdf(current_position - half_step_length);
                    let _end: f64 = self.pdf(current_position - step_length);
                    (_middle, _end)
                }
                IntegrationType::FullInfinite => {
                    let _middle: f64 = {
                        let t: f64 = current_position + half_step_length;
                        let u: f64 = 1.0 / (1.0 - t * t);
                        self.pdf(t * u) * (1.0 + t * t) * u * u
                    };
                    let _end: f64 = {
                        let t: f64 = current_position + step_length;
                        let u: f64 = 1.0 / (1.0 - t * t);
                        self.pdf(t * u) * (1.0 + t * t) * u * u
                    };
                    (_middle, _end)
                }
            };

            accumulator += step_len_over_6 * (last_pdf_evaluation + 4.0 * middle + end);

            last_pdf_evaluation = end;
            num_step += 1.0;
        }

        if let IntegrationType::InfiniteToConst = integration_type {
            // this really should never happen
            ret[current_index] = bounds.0;
        } else {
            ret[current_index] = bounds.1;
        }

        for idx in idx_iter {
            // use all remaining indicies
            if let IntegrationType::InfiniteToConst = integration_type {
                // this really should never happen
                ret[idx] = bounds.0;
            } else {
                ret[idx] = bounds.1;
            }
        }

        return ret;
    }

    // Statistics

    /// Returns the [expected value](https://en.wikipedia.org/wiki/Expected_value)
    /// of the distribution if it exists. Represents the theorical mean of the distribution;
    /// the average of it's samples.
    fn expected_value(&self) -> Option<f64> {
        return Some(self.moments(1, Moments::Raw));
    }

    /// Returns the [variance](https://en.wikipedia.org/wiki/Variance) of
    /// the distribution if it exists. Represents how spread the
    /// distribution is.
    ///
    /// The variance is the square of the standard deviation.
    fn variance(&self) -> Option<f64> {
        return Some(self.moments(2, Moments::Central));
    }

    /// Returns the [mode](https://en.wikipedia.org/wiki/Mode_(statistics))
    /// of the distribution. It represents the most "likely" outcome.
    ///
    /// The deafult implementation uses gradient descent and has a random component,
    /// wich means that the returned value is guaranteed to be a **local maximum**, but not
    /// the global maximum. Fortunely for functions with only 1 single maximum,
    /// the function always returns the correct awnser, but be aware of it's limitations.
    /// Note that you may get incorrect results if the algorithm gets "stuck" in areas where
    /// the function is almost flat.
    fn mode(&self) -> f64 {
        let bounds: (f64, f64) = self.get_domain().get_bounds();
        let integration_type: IntegrationType = IntegrationType::from_bounds(bounds);

        let r: f64 = rand::random::<f64>();
        let seed: f64 = match integration_type {
            IntegrationType::Finite => bounds.0 + r * (bounds.1 - bounds.0),
            IntegrationType::InfiniteToConst => {
                let b: f64 = 2.0;
                // ^modificable parameter,

                b * r / (1.0 - r) + bounds.0
            }
            IntegrationType::ConstToInfinite => {
                let b: f64 = 2.0;
                // ^modificable parameter,

                b * r / (r - 1.0) + bounds.1
            }
            IntegrationType::FullInfinite => (r / (1.0 - r)).ln(),
        };

        let USE_LOG_DISTRIBUTION: bool =
            unsafe { configuration::distribution_mode_deafult::USE_LOG_DERIVATIVE };

        // the `f64::MIN_POSITIVE` is added to avoid problems if p is 0. It should be mostly
        // negligible. `ln(f64::MIN_POSITIVE) = -744.4400719213812`

        let h: f64 = 0.001;
        let derivative = |x: f64| (self.pdf(x + h) - self.pdf(x)) / h;
        let log_derivative = |x: f64| {
            let incr: f64 = (self.pdf(x + h) + f64::MIN_POSITIVE).ln();
            let curr: f64 = (self.pdf(x) + f64::MIN_POSITIVE).ln();
            (incr - curr) / h
        };

        let convergence_difference_criteria: f64 =
            unsafe { configuration::distribution_mode_deafult::CONVERGENCE_DIFFERENCE_CRITERIA };
        let mut learning_rate: f64 =
            unsafe { configuration::distribution_mode_deafult::LEARNING_RATE };
        let learning_rate_change: f64 =
            unsafe { configuration::distribution_mode_deafult::LEARNING_RATE_CHANGE };
        let min_iters: u32 = unsafe { configuration::distribution_mode_deafult::MIN_ITERATIONS };
        let max_iters: u32 = unsafe { configuration::distribution_mode_deafult::MAX_ITERATIONS };

        let mut ret: f64 = seed;
        let mut convergence: bool = false;
        let mut i: u32 = 0;

        while !convergence {
            let gradient: f64 = if USE_LOG_DISTRIBUTION {
                log_derivative(ret)
            } else {
                derivative(ret)
            };
            let updated: f64 = ret + learning_rate * gradient;

            if (ret - updated).abs() < convergence_difference_criteria && min_iters < i {
                convergence = true;
            }

            if max_iters < i {
                return ret;
            }

            match integration_type {
                IntegrationType::Finite => {
                    if ret <= bounds.0 {
                        // there is a maximum at bounds.0, but we cannot go further
                        return bounds.0;
                    }

                    if bounds.1 <= ret {
                        // there is a maximum at bounds.1, but we cannot go further
                        return bounds.1;
                    }
                }
                IntegrationType::InfiniteToConst => {
                    if bounds.1 <= ret {
                        // there is a maximum at bounds.1, but we cannot go further
                        return bounds.1;
                    }
                }
                IntegrationType::ConstToInfinite => {
                    if ret <= bounds.0 {
                        // there is a maximum at bounds.0, but we cannot go further
                        return bounds.0;
                    }
                }
                IntegrationType::FullInfinite => {}
            }

            /*
            if (i & 15) == 0 {
                println!("{}: {}\t\t(grad: {}, lr: {}, log: {}) ", i, ret, gradient, learning_rate, USE_LOG_DISTRIBUTION);
            }
            */
            ret = updated;
            i += 1;
            learning_rate = learning_rate * learning_rate_change;
        }

        return ret;
    }

    /// The [median](https://en.wikipedia.org/wiki/Median) of the distribution. If
    /// you sample a distribution, the median represnts the value that will be
    /// greater than 50% of your samples and also smaller than the other 50%.
    ///
    /// It may happen that the quantile distribution is hard to evaluate but that
    /// the median has a closed form solution. Otherwise, it will be equivalent to
    /// evaluating the [Distribution::quantile] function at `0.5`.
    fn median(&self) -> f64 {
        return self.quantile(0.5);
    }

    /// Returns the [skewness](https://en.wikipedia.org/wiki/Skewness)
    /// of the distribution. Measures how asymetric is the distribution.
    fn skewness(&self) -> Option<f64> {
        return Some(self.moments(3, Moments::Standarized));
    }

    /// Returns the [kurtosis](https://en.wikipedia.org/wiki/Kurtosis)
    /// of the distribution.
    fn kurtosis(&self) -> Option<f64> {
        return self.excess_kurtosis().map(|x| x + 3.0);
    }

    /// Returns the [excess kurtosis](https://en.wikipedia.org/wiki/Kurtosis#Excess_kurtosis)
    /// of the distribution.
    ///
    /// The excess kurtosis is defined as `kurtosis - 3`.
    fn excess_kurtosis(&self) -> Option<f64> {
        return Some(self.moments(4, Moments::Standarized) - 3.0);
    }

    /// Returns the [moment](https://en.wikipedia.org/wiki/Moment_(mathematics))
    /// of the distribution for the given order. ´mode´ determines if the moment will be
    /// [Moments::Raw], [Moments::Central] or [Moments::Standarized].
    fn moments(&self, order: u8, mode: Moments) -> f64 {
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
            Moments::Raw => (0.0, 1.0),
            Moments::Central => (
                self.expected_value()
                    .expect("Tried to compute a central moment but the expected value is undefined. "),
                1.0,
            ),
            Moments::Standarized => (
                self.expected_value()
                    .expect("Tried to compute a central/standarized moment but the Expected value is undefined. "),
                self.variance().expect("Tried to compute a standarized moment but the variance is undefined. "),
            ),
        };

        // Todo: give better error handling to the above. ^
        // println!("(mean, std_dev): {:?}", (mean, std_dev));

        let order_exp: i32 = order as i32;
        let (minus_mean, inv_std_dev) = (-mean, 1.0 / std_dev.sqrt());
        let integration_type: IntegrationType = IntegrationType::from_bounds(bounds);
        let (_, num_steps): (f64, usize) = choose_integration_precision_and_steps(bounds, true);

        let moment: f64 = match integration_type {
            IntegrationType::Finite => {
                let integration_fn = |x: f64| {
                    let std_inp: f64 = (x + minus_mean) * inv_std_dev;
                    std_inp.powi(order_exp) * self.pdf(x)
                };

                euclid::numerical_integration_finite(integration_fn, bounds, num_steps as u64)
            }
            IntegrationType::InfiniteToConst => {
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
            IntegrationType::ConstToInfinite => {
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
            IntegrationType::FullInfinite => {
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

    /// Returns the [entropy](https://en.wikipedia.org/wiki/Information_entropy)
    /// of the distribution. Measures how "uncertain" are the samples from the
    /// distribution.
    fn entropy(&self) -> f64 {
        // the `f64::MIN_POSITIVE` is added to avoid problems if p is 0. It should be mostly
        // negligible. `ln(f64::MIN_POSITIVE) = -744.4400719213812`

        let log_fn = |x| {
            let p: f64 = self.pdf(x);
            p * (p + f64::MIN_POSITIVE).ln()
        };

        let entropy: f64 = euclid::numerical_integration(log_fn, self.get_domain());

        return -entropy;
    }

    // Other provided methods:
    // (methods that don't need to be replaced and should be here)

    /// Sample the distribution with the [rejection sampling](https://en.wikipedia.org/wiki/Rejection_sampling)
    /// method. In general, it can be more effitient that the
    /// normal [Distribution::sample].
    ///
    /// Important:
    ///  - The **domain must be finite**. If it is not, consider the following:
    ///      - Use [Distribution::rejection_sample_range].
    ///      - Implement [Distribution::sample] yourself.
    ///
    ///  - `n`: represents the number of samples to be generated.
    ///  - `pmf_max`: the maximum probability of the [Distribution::pdf].
    ///      - Using a value smaller than the actual value will make the results
    ///         not follow the distribution.
    ///      - Using a value larger than the actual value will incur a extra
    ///         computational cost.
    ///      - Can be computed with [Distribution::mode].
    ///
    /// It is usually more effitient because it does **not** requiere the evaluation of the
    /// [Distribution::quantile] function, wich involves numerical integration. In exchange,
    /// it is needed to know `pdf_max`, the maximum value that the pdf achives.
    fn rejection_sample(&self, n: usize, pdf_max: f64) -> Vec<f64> {
        let mut rng: rand::prelude::ThreadRng = rand::rng();
        let domain: &ContinuousDomain = self.get_domain();
        let bounds: (f64, f64) = domain.get_bounds();
        let bound_range: f64 = bounds.1 - bounds.0;

        let mut ret: Vec<f64> = Vec::with_capacity(n);
        for _ in 0..n {
            let sample: f64 = loop {
                let mut x: f64 = rng.random();
                x = bounds.0 + x * bound_range;
                let y: f64 = rng.random();
                if y * pdf_max < self.pdf(x) {
                    break x;
                }
            };
            ret.push(sample);
        }

        return ret;
    }

    /// Same as [Distribution::rejection_sample] but only in the selected range. (Also
    /// same preconditions).
    ///
    /// This can be usefull for distributions with a stricly infinite domain but that
    /// virtually all their mass is concentrated in a smaller region (`range`).
    ///
    ///  - `n`: represents the number of samples to be generated.
    ///  - `pmf_max`: the maximum probability within the range.
    ///      - Using a value smaller than the actual value will make the results
    ///         not follow the distribution.
    ///      - Using a value larger than the actual value will incur a extra
    ///         computational cost.
    ///      - Can be computed with [Distribution::mode].
    ///  - `range`: the bounds of the region to be sampled from.
    fn rejection_sample_range(&self, n: usize, pdf_max: f64, range: (f64, f64)) -> Vec<f64> {
        let mut rng: rand::prelude::ThreadRng = rand::rng();
        let domain: &ContinuousDomain = self.get_domain();
        let bounds: (f64, f64) = domain.get_bounds();
        let range_magnitude: f64 = range.1 - range.0;

        if range_magnitude.is_sign_negative() || range.0 < bounds.0 || bounds.1 < range.1 {
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

    /// A [confidence interval](https://en.wikipedia.org/wiki/Confidence_interval)
    /// is the region where the samples of the distribution fall with probability
    /// `1 - significance_level`.
    ///
    /// It returns two values `(lower, upper)` sich that
    ///  > P(lower <= theta <= upper) = 1 - significance_level = 1 - alpha
    ///
    /// Special cases:
    ///  - **Panics** if `significance_level` is `+-inf` or a NaN.
    ///  - If `significance_level <= 0.0` then returns [DEFAULT_EMPTY_DOMAIN_BOUNDS].
    ///  - If `1.0 <= significance_level` then returns `self.get_domain().get_bounds()`.
    fn confidence_interval(&self, hypothesys: Hypothesis, significance_level: f64) -> (f64, f64) {
        if !significance_level.is_finite() {
            panic!(
                "Tried to call Distribution::confidence_interval with a non-finite `significance_level` (infinite or NaN). `significance_level` must be a probability. "
            );
        }

        let mut bounds: (f64, f64) = self.get_domain().get_bounds();
        if significance_level <= 0.0 || bounds.1 - bounds.0 <= 0.0 {
            return DEFAULT_EMPTY_DOMAIN_BOUNDS;
        }

        if 1.0 <= significance_level {
            return bounds;
        }

        // significance_level is a value within (0.0, 1.0).

        match hypothesys {
            Hypothesis::RightTail => {
                let quantile: f64 = self.quantile(1.0 - significance_level);
                bounds.1 = quantile;
            }
            Hypothesis::LeftTail => {
                let quantile: f64 = self.quantile(significance_level);
                bounds.0 = quantile;
            }
            Hypothesis::TwoTailed => {
                let quantiles: Vec<f64> =
                    self.quantile_multiple(&[significance_level, 1.0 - significance_level]);
                bounds.0 = quantiles[0];
                bounds.1 = quantiles[1];
            }
        }

        return bounds;
    }

    /// The [P value](https://en.wikipedia.org/wiki/P-value) is the probability
    /// of the null hypotesys having generated a statistic this *extreme* or more.
    fn p_value(&self, hypothesys: Hypothesis, statistic: f64) -> f64 {
        // https://en.wikipedia.org/wiki/P-value#Definition
        let bounds: (f64, f64) = self.get_domain().get_bounds();
        if bounds.1 - bounds.0 <= 0.0 {
            return 0.0;
        }

        let density: f64 = self.cdf(statistic);
        let p: f64 = match hypothesys {
            Hypothesis::RightTail => 1.0 - density,
            Hypothesis::LeftTail => density,
            Hypothesis::TwoTailed => 2.0 * density.min(1.0 - density),
        };

        return p;
    }
}

/// The trait for any continuous distribution.
pub trait DiscreteDistribution {
    //Requiered method:

    /// Evaluates the [PMF](https://en.wikipedia.org/wiki/Probability_density_function)
    /// (Probability Mass Function) of the distribution at point `x`.
    /// The function should not be evaluated outside the domain (because it
    /// should return 0.0 anyway).
    ///
    /// The PMF is assumed to be a valid probability distribution. It is must fullfill:
    ///  - `0.0 <= pmf(x)`
    ///  - It is normalized: `1.0 = sumatory{x} pmf(x)` for all `x` in the domain.
    ///      - If you are not sure if the PDF is normalized, you can use
    /// [crate::euclid::discrete_integration].
    ///  - The function must be real valued (no `+-inf` or NaNs)
    fn pmf(&self, x: f64) -> f64;

    /// Returns a reference to the pdf domain, wich indicates at wich points the pdf can
    /// be evaluated. The returned domain should be constant and not change.
    fn get_domain(&self) -> &DiscreteDomain;

    // Provided methods:
    // Manual implementation for a specific distribution is recommended.

    /// Evaluates the [CDF](https://en.wikipedia.org/wiki/Cumulative_distribution_function)
    /// (Cumulative distribution function).
    /// If the function is evaluated outside the domain of the pdf, it will either
    /// return either `0.0` or `1.0`. **Panicks** is `x` is a NaN.
    ///
    ///  > F(x) = cdf(x) = P(X <= x) = p
    ///
    /// The cdf **includes** the `x` itself. Note that the deafult implemetation 
    /// requieres evaluating the pmf many times and may be expensive.
    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            panic!("Tried to evaluate the cdf with a NaN value. \n");
        }

        let aux: [f64; 1] = [x];
        let aux_2: Vec<f64> = self.cdf_multiple(&aux);
        return aux_2[0];
    }

    /// Samples the distribution at random.
    ///
    /// The deafult method is [Inverse transform sampling](https://en.wikipedia.org/wiki/Inverse_transform_sampling)
    /// unless the deadult method is overriden. Inverse transform sampling simply
    /// generates a random uniform number and evaluates the inverse cdf function
    /// (the [Distribution::quantile] function) and returns the result.
    ///
    /// The method [Distribution::sample_multiple] is more effitient for
    /// multiple sampling.
    fn sample(&self) -> f64 {
        let aux: Vec<f64> = self.sample_multiple(1);
        return aux[0];
    }

    /// Evaluates the [quantile function](https://en.wikipedia.org/wiki/Quantile_function).
    ///
    /// If the cdf is:
    ///
    ///  > F(x) = cdf(x) = P(X <= x) = p
    ///
    /// Then the quantile function is:
    ///
    ///  > Q(p) = x = F^-1(p)
    ///
    ///  - if `x` is outside the range [0.0, 1.0], the bounds of the domain will be retruned.
    ///  - **Panicks** is `x` is a NaN.
    ///
    /// The quantile function is the inverse function of [Distribution::cdf]. Note that
    /// the deafult implemetation requieres numerical integration and may be expensive.
    ///
    /// Also, if you are considering calling this function multiple times, use
    /// [Distribution::quantile_multiple] for better performance.
    fn quantile(&self, x: f64) -> f64 {
        if x.is_nan() {
            // x is not valid
            panic!("Tried to evaluate the quantile function with a NaN value. \n");
        }

        let value: [f64; 1] = [x];
        let quantile_vec: Vec<f64> = self.quantile_multiple(&value);
        return quantile_vec[0];
    }

    // Multiple variants.
    // They are the same as the normal functions, but if they are overriden they may
    // provide a computational advantage.

    /// Evaluates the [CDF](https://en.wikipedia.org/wiki/Cumulative_distribution_function)
    /// (Cumulative distribution function) on multiple points.
    /// If the function is evaluated outside the domain of the pdf, it will either
    /// return either `0.0` or `1.0`. **Panicks** is `x` is a NaN.
    ///
    ///  > F(x) = cdf(x) = P(X <= x) = p
    ///
    /// The cdf **includes** the `x` itself. Note that the deafult implemetation 
    /// requieres evaluating the pmf many times and may be expensive.
    /// 
    /// ***
    /// 
    /// [Distribution::cdf_multiple] allows to evaluate the [Distribution::cdf] at 
    /// multiple points. It *may* provide a computational advantage over calling 
    /// [Distribution::cdf] in a loop.  
    ///
    /// ***
    /// ***
    ///
    /// If an effitient [Distribution::cdf] has been implemented, it can be replaced for:
    ///
    /// ```
    /// fn cdf_multiple(&self, points: &[f64]) -> Vec<f64> {
    ///     points.iter().map(|x| self.cdf(*x)).collect::<Vec<f64>>()
    /// }
    /// ```
    fn cdf_multiple(&self, points: &[f64]) -> Vec<f64> {
        /*
                Plan:
            Similarly to the continuous case, we will sort the points
            to avoid redundant computation (Check [Distribution::cdf_multiple]
            comment for better info).

            However if we have a domain with a full infinite domain `[-inf, inf]`
            then there is not a trivial way to compute the cdf in the general case.
            To solve this we will use an aproximation. We will find the range that
            contains most of the area (0.99999 by deafult when this was written),
            an we will update `bounds` to match these values. Then we can
            perform the discrete integration as if it was a finite range. We won't
            be able to caputre the full mass of the distribution but should be
            close enough for most cases.
        */

        if points.is_empty() {
            return Vec::new();
        }

        // panic if NAN is found
        for point in points {
            if point.is_nan() {
                panic!("Found NaN in `cdf_multiple`. \n");
            }
        }

        let mut ret: Vec<f64> = vec![0.0; points.len()];
        let domain: &DiscreteDomain = self.get_domain();
        let mut bounds: (f64, f64) = domain.get_bounds();
        let integration_type: IntegrationType = IntegrationType::from_bounds(bounds);

        if let IntegrationType::FullInfinite = integration_type {
            let max_area: f64 = unsafe { euclid::PROBABILITY_THRESHOLD_DISCRETE_INTEGRATION };
            bounds = euclid::discrete_region_with_area(|x: f64| self.pmf(x), domain, max_area)
        }

        let mut sorted_indicies: Vec<usize> = (0..points.len()).into_iter().collect::<Vec<usize>>();

        sorted_indicies.sort_unstable_by(|&i, &j| {
            let a: f64 = points[i];
            let b: f64 = points[j];
            if let IntegrationType::InfiniteToConst = integration_type {
                // sort in reverse
                b.partial_cmp(&a).unwrap()
            } else {
                a.partial_cmp(&b).unwrap()
            }
        });

        let mut idx_iter: std::vec::IntoIter<usize> = sorted_indicies.into_iter();
        let mut current_index: usize = idx_iter.next().unwrap();
        // ^unwrap is safe

        let mut current_cdf_point: f64 = points[current_index];

        let mut accumulator: f64 = 0.0;

        match integration_type {
            IntegrationType::Finite
            | IntegrationType::ConstToInfinite
            | IntegrationType::FullInfinite => {
                while current_cdf_point < bounds.0 {
                    ret[current_index] = 0.0;
                    match idx_iter.next() {
                        Some(v) => current_index = v,
                        None => return ret,
                    }
                    current_cdf_point = points[current_index];
                }
            }
            IntegrationType::InfiniteToConst => {
                while bounds.1 < current_cdf_point {
                    ret[current_index] = 1.0;
                    match idx_iter.next() {
                        Some(v) => current_index = v,
                        None => return ret,
                    }
                    current_cdf_point = points[current_index];
                }
            }
        }

        for x in domain.iter() {
            match integration_type {
                IntegrationType::Finite
                | IntegrationType::ConstToInfinite
                | IntegrationType::FullInfinite => {
                    while current_cdf_point < x {
                        ret[current_index] = accumulator;
                        match idx_iter.next() {
                            Some(v) => current_index = v,
                            None => return ret,
                        }
                        current_cdf_point = points[current_index];
                    }
                }
                IntegrationType::InfiniteToConst => {
                    while x < current_cdf_point {
                        ret[current_index] = 1.0 - accumulator;
                        match idx_iter.next() {
                            Some(v) => current_index = v,
                            None => return ret,
                        }
                        current_cdf_point = points[current_index];
                    }
                }
            }

            accumulator += self.pmf(x);
        }

        // If we reach this point it means that the domian is finite and the remaining
        // values are <= bounds.1

        ret[current_index] = 1.0;
        for idx in idx_iter {
            ret[idx] = 1.0;
        }

        return ret;
    }

    /// [Distribution::sample_multiple] samples the distribution 
    /// at random `n` times. 
    /// 
    /// The deafult method is [Inverse transform sampling](https://en.wikipedia.org/wiki/Inverse_transform_sampling)
    /// unless the deadult method is overriden. Inverse transform sampling simply
    /// generates a random uniform number and evaluates the inverse cdf function
    /// (the [Distribution::quantile_multiple] function) and returns the result.
    ///
    /// The deafult implementation uses the [Distribution::quantile_multiple] function,
    /// wich may be expensive. Consider using [Distribution::rejection_sample] if possible.
    /// 
    /// It *may* provide a computational advantage in
    /// comparasion to calling [Distribution::sample] in a loop.
    ///
    /// ***
    /// ***
    ///
    /// If an effitient [Distribution::sample] has been implemented, it can be replaced for:
    ///
    /// ```
    /// fn sample_multiple(&self, n: usize) -> Vec<f64> {
    ///     (0..n).map(|_| self.sample()).collect::<Vec<f64>>()
    /// }
    /// ```
    fn sample_multiple(&self, n: usize) -> Vec<f64> {
        let mut rng: rand::prelude::ThreadRng = rand::rng();
        let mut rand_quantiles: Vec<f64> = vec![0.0; n];
        rng.fill(rand_quantiles.as_mut_slice());

        let ret: Vec<f64> = self.quantile_multiple(&rand_quantiles);

        return ret;
    }

    /// `quantile_multiple` allows to evaluate the [Distribution::quantile] on multiple points.
    /// It *may* provide a computational advantage over calling [Distribution::quantile]
    /// in a loop.
    ///
    /// Notes:
    ///  - It **panics** if any value is a NaN.
    ///  - If a value in points is less (or equal) to 0, the minimum value
    /// in the domain will be returned.
    ///  - If a value in points is greater (or equal) to 1, the maximum value in the
    /// domain will be returned.
    ///
    /// ***
    /// ***
    ///
    /// If an effitient [Distribution::quantile] has been implemented, it can be replaced for:
    ///
    /// ```
    /// fn quantile_multiple(&self, points: &[f64]) -> Vec<f64> {
    ///     let list: Vec<f64> = points
    ///         .iter()
    ///         .map(|x| self.quantile(*x))
    ///         .collect::<Vec<f64>>();
    ///     return list;
    /// }
    /// ```
    fn quantile_multiple(&self, points: &[f64]) -> Vec<f64> {
        /*
                Plan:
            Similarly to the continuous case, we will sort the points
            to avoid redundant computation (Check [Distribution::quantile_multiple]
            comment for better info).

            However if we have a domain with a full infinite domain `[-inf, inf]`
            then there is not a trivial way to compute the quantile function
            in the general case. To solve this we will use an aproximation.
            We will find the range that contains most of the area (0.99999 by
            deafult when this was written), an we will update `bounds` to match
            these values. Then we can perform the discrete integration as if it
            was a finite range. We won't be able to caputre the full mass of the
            distribution but should be close enough for most cases.
        */

        if points.is_empty() {
            return Vec::new();
        }

        // panic if NAN is found
        for point in points {
            if point.is_nan() {
                panic!("Found NaN in `quantile_multiple`. \n");
            }
        }

        let mut ret: Vec<f64> = vec![0.0; points.len()];
        let domain: &DiscreteDomain = self.get_domain();
        let mut bounds: (f64, f64) = domain.get_bounds();
        let integration_type: IntegrationType = IntegrationType::from_bounds(bounds);
        if let IntegrationType::FullInfinite = integration_type {
            let max_area: f64 = unsafe { euclid::PROBABILITY_THRESHOLD_DISCRETE_INTEGRATION };

            bounds = euclid::discrete_region_with_area(|x: f64| self.pmf(x), domain, max_area);
            // Now treat the IntegrationType::FullInfinite the same as IntegrationType::Finite
        }

        let mut sorted_indicies: Vec<usize> = (0..points.len()).into_iter().collect::<Vec<usize>>();

        sorted_indicies.sort_unstable_by(|&i, &j| {
            let a: f64 = points[i];
            let b: f64 = points[j];
            if let IntegrationType::InfiniteToConst = integration_type {
                // sort in reverse
                b.partial_cmp(&a).unwrap()
            } else {
                a.partial_cmp(&b).unwrap()
            }
        });

        let mut idx_iter: std::vec::IntoIter<usize> = sorted_indicies.into_iter();
        let mut current_index: usize = idx_iter.next().unwrap();
        // ^unwrap is safe

        let mut current_quantile_point: f64 = points[current_index];

        let mut accumulator: f64 = 0.0;

        match integration_type {
            IntegrationType::Finite
            | IntegrationType::ConstToInfinite
            | IntegrationType::FullInfinite => {
                while current_quantile_point < 0.0 {
                    ret[current_index] = bounds.0;
                    match idx_iter.next() {
                        Some(v) => current_index = v,
                        None => return ret,
                    }
                    current_quantile_point = points[current_index];
                }
            }
            IntegrationType::InfiniteToConst => {
                while 1.0 <= current_quantile_point {
                    ret[current_index] = bounds.1;
                    match idx_iter.next() {
                        Some(v) => current_index = v,
                        None => return ret,
                    }
                    current_quantile_point = points[current_index];
                }
            }
        }

        for x in domain.iter() {
            
            accumulator += self.pmf(x);
            
            match integration_type {
                IntegrationType::Finite
                | IntegrationType::ConstToInfinite
                | IntegrationType::FullInfinite => {
                    while current_quantile_point < accumulator {
                        ret[current_index] = x;
                        match idx_iter.next() {
                            Some(v) => current_index = v,
                            None => return ret,
                        }
                        current_quantile_point = points[current_index];
                    }
                }
                IntegrationType::InfiniteToConst => {
                    while 1.0 - accumulator <= current_quantile_point {
                        ret[current_index] = x;
                        match idx_iter.next() {
                            Some(v) => current_index = v,
                            None => return ret,
                        }
                        current_quantile_point = points[current_index];
                    }
                }
            }
        }

        ret[current_index] = bounds.1;
        for idx in idx_iter {
            ret[idx] = bounds.1;
        }

        return ret;
    }

    // Statistics

    /// Returns the [expected value](https://en.wikipedia.org/wiki/Expected_value)
    /// of the distribution if it exists. Represents the theorical mean of the distribution;
    /// the average of it's samples.
    fn expected_value(&self) -> Option<f64> {
        return Some(self.moments(1, Moments::Raw));
    }

    /// Returns the [variance](https://en.wikipedia.org/wiki/Variance) of
    /// the distribution if it exists. Represents how spread the
    /// distribution is.
    ///
    /// The variance is the square of the standard deviation.
    fn variance(&self) -> Option<f64> {
        return Some(self.moments(2, Moments::Central));
    }

    /// Returns the [mode](https://en.wikipedia.org/wiki/Mode_(statistics))
    /// of the distribution. It represents the most likely outcome.
    ///
    /// If the distribution is very large or infinite, it only checks the first
    /// [configuration::disrete_distribution_deafults::MAXIMUM_STEPS]
    /// values.
    ///
    /// Panics if the domain contains no values.
    fn mode(&self) -> f64 {
        let domain: &DiscreteDomain = self.get_domain();
        let mut domain_iter: crate::domain::DiscreteDomainIterator<'_> = domain.iter();

        let (mut max, mut max_value): (f64, f64) = match domain_iter.next() {
            Some(v) => (v, self.pmf(v)),
            None => panic!(
                "Attempted to compute the mode of a distribution with empty domain. (Domain contains no elements)"
            ),
        };

        let finite_elemtents: bool = domain.contains_finite_elements();

        if finite_elemtents {
            for point in domain_iter {
                let mass: f64 = self.pmf(point);
                if max_value < mass {
                    max = point;
                    max_value = mass;
                }
            }
        } else {
            let area_treshold: f64 = unsafe { euclid::PROBABILITY_THRESHOLD_DISCRETE_INTEGRATION };
            let range: (f64, f64) =
                euclid::discrete_region_with_area(|x: f64| self.pmf(x), domain, area_treshold);

            for point in domain_iter {
                if !(range.0 <= point && point <= range.0) {
                    break;
                }
                let mass: f64 = self.pmf(point);
                if max_value < mass {
                    max = point;
                    max_value = mass;
                }
            }
        }

        return max;
    }

    /// The [median](https://en.wikipedia.org/wiki/Median) of the distribution. If
    /// you sample a distribution, the median represnts the value that will be
    /// greater than 50% of your samples and also smaller than the other 50%.
    ///
    /// It may happen that the quantile distribution is hard to evaluate but that
    /// the median has a closed form solution. Otherwise, it will be equivalent to
    /// evaluating the [DiscreteDistribution::quantile] function at `0.5`.
    fn median(&self) -> f64 {
        return self.quantile(0.5);
    }

    /// Returns the [skewness](https://en.wikipedia.org/wiki/Skewness)
    /// of the distribution. Measures how asymetric is the distribution.
    fn skewness(&self) -> Option<f64> {
        return Some(self.moments(3, Moments::Standarized));
    }

    /// Returns the [kurtosis](https://en.wikipedia.org/wiki/Kurtosis)
    /// of the distribution.
    fn kurtosis(&self) -> Option<f64> {
        // return self.kurtosis().map(|x| x + 3.0);
        return Some(self.moments(4, Moments::Standarized));
    }

    /// Returns the [excess kurtosis](https://en.wikipedia.org/wiki/Kurtosis#Excess_kurtosis)
    /// of the distribution.
    ///
    /// The excess kurtosis is defined as `kurtosis - 3`.
    fn excess_kurtosis(&self) -> Option<f64> {
        // return Some(self.moments(4, Moments::Standarized) - 3.0);
        return self.kurtosis().map(|x| x - 3.0);
    }

    /// Returns the [moment](https://en.wikipedia.org/wiki/Moment_(mathematics))
    /// of the distribution for the given order. `mode` determines if the moment will be
    /// [Moments::Raw], [Moments::Central] or [Moments::Standarized].
    fn moments(&self, order: u8, mode: Moments) -> f64 {
        let domain: &DiscreteDomain = self.get_domain();

        // The values of 0.0 and 1.0 have no special meaning. They are not going to be used anyway.
        let (mean, std_dev): (f64, f64) = match mode {
            Moments::Raw => (0.0, 1.0),
            Moments::Central => (
                self.expected_value().expect(
                    "Tried to compute a central moment but the expected value is undefined. ",
                ),
                1.0,
            ),
            Moments::Standarized => (
                self.expected_value().expect(
                    "Tried to compute a standarized moment but the Expected value is undefined. ",
                ),
                self.variance().expect(
                    "Tried to compute a standarized moment but the variance is undefined. ",
                ),
            ),
        };

        // Todo: give better error handling to the above. ^
        // println!("(mean, std_dev): {:?}", (mean, std_dev));

        let order_exp: i32 = order as i32;
        let (minus_mean, inv_std_dev): (f64, f64) = (-mean, 1.0 / std_dev.sqrt());

        let integration_fn = |x: f64| {
            let std_inp: f64 = (x + minus_mean) * inv_std_dev;
            std_inp.powi(order_exp) * self.pmf(x)
        };

        //let max_steps: u64 = unsafe { configuration::disrete_distribution_deafults::MAXIMUM_STEPS };
        //let max_steps_opt: Option<usize> = Some(max_steps.try_into().unwrap_or(usize::MAX));

        //let moment: f64 = euclid::discrete_integration(integration_fn, domain, max_steps_opt);

        let moment: (f64, f64) = euclid::discrete_integration_with_acumulation(
            integration_fn,
            |x: f64| self.pmf(x),
            domain,
        );

        return moment.0;
    }

    /// Returns the [entropy](https://en.wikipedia.org/wiki/Information_entropy)
    /// of the distribution. Measures how "uncertain" are the samples from the
    /// distribution.
    fn entropy(&self) -> f64 {
        let domain: &DiscreteDomain = self.get_domain();
        //let max_steps: u64 = unsafe { configuration::disrete_distribution_deafults::MAXIMUM_STEPS };
        //let max_steps_opt: Option<usize> = Some(max_steps.try_into().unwrap_or(usize::MAX));

        // the `f64::MIN_POSITIVE` is added to avoid problems if p is 0. It should be mostly
        // negligible. `ln(f64::MIN_POSITIVE) = -744.4400719213812`
        // In a more formal way, if `a << b` then `ln(a + b) ~= ln(b)`.
        // Since we are choosing a to be the smallest possible value, we should always fullfil
        // this criteria. When we do not is when `p = 0` but in that case
        // `p * (p + f64::MIN_POSITIVE).ln() = 0.0` so it dosen't matter anyway.

        let log_fn = |x: f64| {
            let p: f64 = self.pmf(x);
            p * (p + f64::MIN_POSITIVE).ln()
        };

        //let entropy: f64 = euclid::discrete_integration(log_fn, self.get_domain(), max_steps_opt);
        let entropy: (f64, f64) =
            euclid::discrete_integration_with_acumulation(log_fn, |x: f64| self.pmf(x), domain);

        return -entropy.0;
    }

    // Other provided methods:
    // (methods that don't need to be replaced and should be here)

    /// Sample the distribution but only in the specified range.
    ///
    /// This can be usefull for distributions with a stricly infinite domain but that
    /// virtually all their mass is concentrated in a smaller region (`range`). For
    /// example, we could sample from the a binomial distribution with a large `n` and
    /// low variance (`p` is close to 0 or 1), where most of the mass is centered very
    /// close to the mean but there is very little far away.
    ///
    /// Note that this does NOT work for [DiscreteDomain::Custom] variant and
    /// for the remaining variants, `range` needs to be contained within the domain
    /// of the distribution. If these conditions are notfullfilled, an empty vector
    /// will be returned.
    ///
    ///  - `n`: represents the number of samples to be generated.
    ///  - `pmf_max`: the maximum probability within the range.
    ///      - Using a value smaller than the actual value will make the results
    ///         not follow the distribution.
    ///      - Using a value larger than the actual value will incur a extra
    ///         computational cost.
    ///      - Can be computed with [DiscreteDistribution::mode].
    ///  - `range`: the bounds of the region to be sampled from.
    ///      - Both values are inclusive.
    fn rejection_sample_range(&self, n: usize, pmf_max: f64, range: (i64, i64)) -> Vec<f64> {
        let mut rng: rand::prelude::ThreadRng = rand::rng();
        let range_f: (f64, f64);

        {
            // possible early return
            let domain: &DiscreteDomain = self.get_domain();

            if let DiscreteDomain::Custom(_) = domain {
                return Vec::new();
            }

            let bounds: (f64, f64) = domain.get_bounds();
            range_f = (range.0 as f64, range.1 as f64);
            if range.1 < range.0 || (range_f.0 < bounds.0.floor()) || bounds.1.ceil() < range_f.1 {
                return Vec::new();
            }
        }

        // domain is not of the custom variant.
        // `range` is contained within the domain of the distribution

        /*

           let probability_region: f64 = {
               let cdf_values: Vec<f64> = self.cdf_multiple(&[(range.0 - 1) as f64, range.1 as f64]);
               cdf_values[1] - cdf_values[0]
           };

           If the desired region only contains `probability_region` mass, then
           we need to divide pmf() to get a new valid probability distribution.
           Then the real maximim of the new pmf cound be `pmf_max/probability_region`.
           The condition for being sampled is:
           (if `probability_region` = 1, (the normal way))
           `rand() < pmf(x) / pmf_max`
           (if `probability_region` != 1)
           `rand() < (pmf(x) / probability_region) / (pmf_max / probability_region)` =
           `rand() < pmf(x) / (probability_region * pmf_max / probability_region)` =
           `rand() < pmf(x) / pmf_max` =
           `rand() * pmf_max < pmf(x)`

           Therefore we do not need `probability_region`.
        */

        let bound_range: f64 = (range.1 - range.0) as f64;
        let mut ret: Vec<f64> = Vec::with_capacity(n);
        for _i in 0..n {
            let sample: f64 = loop {
                let mut x: f64 = rng.random();
                x = range_f.0 + x * bound_range;
                let y: f64 = rng.random();
                if y * pmf_max < self.pmf(x) {
                    break x;
                }
            };
            ret.push(sample);
        }

        return ret;
    }

    /// A [confidence interval](https://en.wikipedia.org/wiki/Confidence_interval)
    /// is the region where the samples of the distribution fall with probability
    /// `1 - significance_level`.
    ///
    /// It returns two values `(lower, upper)` sich that
    ///  > P(lower <= theta <= upper) = 1 - significance_level = 1 - alpha
    ///
    /// Special cases:
    ///  - **Panics** if `significance_level` is `+-inf` or a NaN.
    ///  - If `significance_level <= 0.0` then returns [DEFAULT_EMPTY_DOMAIN_BOUNDS].
    ///  - If `1.0 <= significance_level` then returns `self.get_domain().get_bounds()`.
    fn confidence_interval(&self, hypothesys: Hypothesis, significance_level: f64) -> (f64, f64) {
        if !significance_level.is_finite() {
            panic!(
                "Tried to call DiscreteDistribution::confidence_interval with a non-finite `significance_level` (infinite or NaN). `significance_level` must be a probability. "
            );
        }

        let mut bounds: (f64, f64) = self.get_domain().get_bounds();
        if significance_level <= 0.0 || bounds.1 <= bounds.0 {
            return DEFAULT_EMPTY_DOMAIN_BOUNDS;
        }

        if 1.0 <= significance_level {
            return bounds;
        }

        // significance_level is a value within (0.0, 1.0).

        match hypothesys {
            Hypothesis::RightTail => {
                let quantile: f64 = self.quantile(1.0 - significance_level);
                bounds.1 = quantile;
            }
            Hypothesis::LeftTail => {
                let quantile: f64 = self.quantile(significance_level);
                bounds.0 = quantile;
            }
            Hypothesis::TwoTailed => {
                let quantiles: Vec<f64> =
                    self.quantile_multiple(&[significance_level, 1.0 - significance_level]);
                bounds.0 = quantiles[0];
                bounds.1 = quantiles[1];
            }
        }

        return bounds;
    }

    /// The [P value](https://en.wikipedia.org/wiki/P-value) is the probability
    /// of the null hypotesys having generated a statistic this *extreme* or more.
    ///
    ///  - **Panics** if `statistic` is non-finite (`+-inf` or NaN)
    fn p_value(&self, hypothesys: Hypothesis, statistic: f64) -> f64 {
        // https://en.wikipedia.org/wiki/P-value#Definition

        if !statistic.is_finite() {
            panic!(
                "Tried to call DiscreteDistribution::p_value with a non-finite `statistic` (infinite or NaN). "
            );
        }

        let bounds: (f64, f64) = self.get_domain().get_bounds();
        if bounds.1 - bounds.0 <= 0.0 {
            return 0.0;
        }

        let density: f64 = self.cdf(statistic);
        let p: f64 = match hypothesys {
            Hypothesis::RightTail => 1.0 - density,
            Hypothesis::LeftTail => density,
            Hypothesis::TwoTailed => 2.0 * density.min(1.0 - density),
        };

        return p;
    }
}

/// This trait controls [Distributions](Distribution) with parameters.
///
/// This trait is uscefull to find the best fit from some data to
/// the given distribution. The parameters must be floats ([f64]).
///
/// Each distribution mush have a specific otdering for it's parameters.
/// This way, when they are passed trough functions it can be assumed that they
/// are always the same (and in the same order).
///
/// ## Requered methods:
///
///  1. [general_pdf](Parametric::general_pdf)
///  2. [number_of_parameters](Parametric::number_of_parameters)
///  3. [get_parameters](Parametric::get_parameters)
///  4. Either of: (ideally both)
///      - [derivative_pdf_parameters](Parametric::derivative_pdf_parameters)
///      - [log_derivative_pdf_parameters](Parametric::log_derivative_pdf_parameters)
///
/// (We recommend implementing log_derivative_pdf_parameters since it is usually easier).
/// ### Notes:
///
/// The `&self` in most methods is required because there can be other non-float
/// parameters that may be needed to compute [general_pdf](Parametric::general_pdf).
/// Otherwise there may be some a parameter we don't want to optimize for.
pub trait Parametric {
    /// Evaluates the [PDF](https://en.wikipedia.org/wiki/Probability_density_function)
    /// (Probability Density function) of the distribution at point `x` with
    /// the given `parameters`.
    ///
    /// If follows the same constraits as the normal
    /// [Distribution::pdf](crate::distribution_trait::Distribution::pdf)
    /// (or [DiscreteDistribution::pmf](crate::distribution_trait::DiscreteDistribution::pmf))
    /// but also taking the parameters into account.
    fn general_pdf(&self, x: f64, parameters: &[f64]) -> f64;

    /// Returns the gradient of the pdf in respect to the parameters and the point `x`.
    ///
    /// The returned vector has the derivative respect `x` in the first position
    /// (at index 0), the derivative respect to the first parameter at the second position
    /// (at index 1), the derivative respect to the second parameter at the third position
    /// (at index 2) and so on.
    ///
    /// If a parameter is discrete (for example can only be a natural number) then the
    /// derivative will be `0.0`.
    fn derivative_pdf_parameters(&self, x: f64, parameters: &[f64]) -> Vec<f64> {
        // d/dx ln(f(x)) = f'(x)/f(x)
        // => f(x) * d/dx ln(f(x)) = f'(x)

        let pdf: f64 = self.general_pdf(x, parameters);

        return self
            .log_derivative_pdf_parameters(x, parameters)
            .iter()
            .map(|log_d| log_d * pdf)
            .collect::<Vec<f64>>();
    }

    /// The natural logarithm of [Parametric::derivative_pdf_parameters]. The logarithm
    /// of the elements of the gradient of the pdf in respect to the parameters.
    ///
    /// The returned vector has the derivative respect `x` in the first position
    /// (at index 0), the derivative respect to the first parameter at the second position
    /// (at index 1), the derivative respect to the second parameter at the third position
    /// (at index 2) and so on.
    ///
    /// If a parameter is discrete (for example can only be a natural number) then the
    /// derivative will be `-inf` or a NaN.
    ///
    /// See: [logarithmic derivative](https://en.wikipedia.org/wiki/Logarithmic_derivative)
    /// `d/dx ln(f(x)) = f'(x)/f(x)`
    fn log_derivative_pdf_parameters(&self, x: f64, parameters: &[f64]) -> Vec<f64> {
        // d/dx ln(f(x)) = f'(x)/f(x)

        return self
            .derivative_pdf_parameters(x, parameters)
            .iter()
            .map(|der| {
                let mut pdf: f64 = self.general_pdf(x, parameters);
                pdf = pdf.max(f64::EPSILON);
                der / pdf
            })
            .collect::<Vec<f64>>();
    }

    /// Returns the number of parameters of the model.
    ///
    /// If [u16] is not enough, you may be interested in other machine leaning
    /// approaches that AdvancedStatistics does not focus on.
    fn number_of_parameters() -> u16;

    /// Writes the parameters of the model in order in the given
    /// slice.
    ///
    /// **IMPORTANT**: The caller must fullfill or the slice will be
    /// accessed out of bounds:
    /// `self.number_of_parameters() <= parameters.len()`
    fn get_parameters(&self, parameters: &mut [f64]);

    /// Restrict the parameters if necessary.
    ///
    /// For example, the std_dev in the normal distribution is stricly
    /// positive, therefore the body for the [Normal distribution](crate::distributions::Normal::Normal).
    /// should be `{ parameters[1] = parameters[1].max(f64::EPSILON)}` to
    /// ensure that the std_dev is always stricly positive.
    ///
    /// An empty body (like in the deafult implemetation) means no restrictions on
    /// any of the parameters.
    ///
    /// Used in [Parametric::fit]
    #[allow(unused_variables)]
    fn parameter_restriction(&self, parameters: &mut [f64]) {}

    /// Returns a vector of the parameters that best fit the distribution given
    /// the data.
    ///
    /// The method used is [Maximum Likelihood Estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)
    /// (MLE) with [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent).
    ///
    /// If there has been an error, returns an empty vector. If using the
    /// default implemetation, it gets the initial guess from the values in self.
    fn fit(&self, data: &mut Samples) -> Vec<f64> {
        let d: usize = Self::number_of_parameters() as usize;
        let mut parameters: Vec<f64> = vec![0.0; d];
        self.get_parameters(&mut parameters);

        self.parameter_restriction(&mut parameters);

        let learning_rate: f64 =
            unsafe { configuration::maximum_likelihood_estimation::LEARNING_RATE };
        let conv_diff_criteria: f64 = unsafe {
            configuration::maximum_likelihood_estimation::CONVERGENCE_DIFFERENCE_CRITERIA
        };
        let max_iterations: u32 =
            unsafe { configuration::maximum_likelihood_estimation::MAX_ITERATIONS };

        let inv_n: f64 = 1.0 / (data.peek_data().len() as f64);

        let mut gradient: Vec<f64> = vec![0.0; d];
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
