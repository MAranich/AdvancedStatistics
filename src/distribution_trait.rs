use std::usize;

use rand::Rng;

use crate::configuration::{self, QUANTILE_USE_NEWTONS_ITER};
use crate::domain::{ContinuousDomain, DiscreteDomain};
///! This script contains the interfaces used to comunicate with the distributions.
use crate::euclid::*;

/// The trait for any continuous distribution.
///
/// The [Distribution::pdf] must be a simple function. If you want a piecewise function,
/// look first at [crate::mixed_distribution]
///
/// None of the provided methods are guaranteed to work if the implemented [Distribution::pdf]
/// is NOT a [valid pdf](https://en.wikipedia.org/wiki/Probability_density_function).
/// So, it needs to fullfill:
///  - The function must be stricly non-negative
///  - The function must be real valued
///  - The function must have a total area of 1 under the curve.
pub trait Distribution {
    //Requiered method:

    /// Evaluates the [PDF](https://en.wikipedia.org/wiki/Probability_density_function)
    /// (Probability Density function) of the distribution at point `x`.
    ///
    /// The PDF is assumed to be a valid probability distribution. It is must fullfill: 
    ///  - `0.0 <= pdf(x)`
    ///  - It is normalized. (It has an area under the curbe of `1.0`)
    ///      - If you are not sure if the PDF is normalized, you can use 
    /// [crate::euclid::determine_normalitzation_constant_continuous]. 
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
    /// If the function is evaluated outside the domain of the pdf, it will either
    /// return either `0.0` or `1.0`. **Panicks** is `x` is a NaN.
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
    ///  - if `x` is outside the range [0.0, 1.0], the bounds of the domain will be returned.
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
                // return Err(AdvStatError::NanErr);
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

        let (step_length, max_iters): (f64, usize) = choose_integration_precision_and_steps(bounds);
        let half_step_length: f64 = 0.5 * step_length;
        let step_len_over_6: f64 = step_length / 6.0;

        let mut idx_iter: std::vec::IntoIter<usize> = sorted_indicies.into_iter();
        let mut current_index: usize = idx_iter.next().unwrap();
        // ^unwrap is safe

        let mut current_cdf_point: f64 = points[current_index];

        let mut num_step: f64 = 0.0;
        let mut accumulator: f64 = 0.0;

        let mut last_pdf_evaluation: f64 = match integration_type {
            IntegrationType::Finite | IntegrationType::ConstToInfinite => {
                self.pdf(bounds.0 + f64::EPSILON)
            }
            IntegrationType::InfiniteToConst => self.pdf(bounds.1 - f64::EPSILON),
            IntegrationType::FullInfinite => 0.0,
        };

        'integration_loop: for _ in 0..max_iters {
            //all_iterations_done = i + 1 == max_iters;
            let current_position: f64;

            match integration_type {
                IntegrationType::Finite | IntegrationType::ConstToInfinite => {
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
                }
                IntegrationType::InfiniteToConst => {
                    current_position = bounds.1 - step_length * num_step;
                    while current_position <= current_cdf_point {
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

                    while current_cdf_point * u <= current_position {
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

    /// [Distribution::sample_multiple] allows to evaluate the [Distribution::sample]
    /// at multiple points. It may provide a computational advantage in comparasion to [Distribution::sample].
    ///
    /// The deafult implementation uses the [Distribution::quantile_multiple] function,
    /// wich may be expensive. Consider using [Distribution::rejection_sample] if possible.
    ///
    /// If an effitient [Distribution::sample] has been implemented, it can be replaced for:
    ///
    /// ```
    /// fn sample_multiple(&self, n: usize) -> Vec<f64> {
    ///     (0..n).map(|_| self.sample()).collect::<Vec<f64>>()
    /// }
    /// ```
    fn sample_multiple(&self, n: usize) -> Vec<f64> {
        let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
        let mut rand_quantiles: Vec<f64> = vec![0.0; n];
        rng.fill(rand_quantiles.as_mut_slice());

        let ret: Vec<f64> = self.quantile_multiple(&rand_quantiles);

        return ret;
    }

    /// quantile_multiple acts the same as [Distribution::quantile] but on multiple points.
    /// It provides a computational advantage over calling the normal [Distribution::quantile]
    /// multiple times.
    ///
    /// **Panicks** is `x` is a NaN. If a value in points is less (or equal)
    /// to 0.0, the minimum value in the domain will be returned. If a value in
    /// points is greater (or equal) to 1, the maximum value in the domain
    /// will be returned.
    ///
    /// If an effitient [Distribution::quantile] has been implemented, it can be replaced for:
    ///
    /// ```
    /// fn quantile_multiple(&self, points: &[f64]) -> Result<Vec<f64>, crate::errors::AdvStatError> {
    ///     let list: Vec<f64> = points
    ///         .iter()
    ///         .map(|x| match self.quantile(*x) {
    ///             Ok(v) => v,
    ///             Err(_) => panic!("There has been an error! "),
    ///         })
    ///         .collect::<Vec<f64>>();
    ///     Ok(list)
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

        let (step_length, max_iters): (f64, usize) = choose_integration_precision_and_steps(bounds);
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

        let mut last_pdf_evaluation: f64 = match integration_type {
            IntegrationType::Finite | IntegrationType::ConstToInfinite => {
                self.pdf(bounds.0 + f64::EPSILON)
            }
            IntegrationType::InfiniteToConst => self.pdf(bounds.1 - f64::EPSILON),
            IntegrationType::FullInfinite => 0.0,
        };

        'integration_loop: for _ in 0..max_iters {
            let current_position: f64;

            match integration_type {
                IntegrationType::Finite | IntegrationType::ConstToInfinite => {
                    current_position = bounds.0 + step_length * num_step;
                    while current_quantile <= accumulator {
                        let mut quantile: f64 = current_position;

                        let pdf_q: f64 = self.pdf(quantile);
                        if QUANTILE_USE_NEWTONS_ITER && !(pdf_q.abs() < f64::EPSILON) {
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
                    while 1.0 - accumulator <= current_quantile {
                        let mut quantile: f64 = current_position;

                        let pdf_q: f64 = self.pdf(quantile);
                        if QUANTILE_USE_NEWTONS_ITER && !(pdf_q.abs() < f64::EPSILON) {
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
                    while current_quantile <= accumulator {
                        let mut quantile: f64 = current_position;

                        let pdf_q: f64 = self.pdf(quantile);
                        if QUANTILE_USE_NEWTONS_ITER && !(pdf_q.abs() < f64::EPSILON) {
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
                IntegrationType::FullInfinite => todo!(),
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

        /*
        // panick if NAN is found
        for point in points {
            if point.is_nan() {
                panic!("Tried to evaluate the quantile_multiple function with a NaN value. \n");
            }
        }

        let mut sorted_points: Vec<(usize, f64)> = points.iter().map(|x| *x).enumerate().collect();

        sorted_points.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut ret: Vec<(usize, f64)> = Vec::with_capacity(points.len());
        let domain: &ContinuousDomain = self.get_domain();
        let bounds: (f64, f64) = domain.get_bounds();

        // if points[i] <= 0  |||| Awnser is always bounds.0 (start of domain)
        let mut small_trivial_points: usize = 0;
        // if 1 <= points[i]  |||| Awnser is always bounds.1 (end of domain)
        let mut big_trivial_points: usize = 0;
        // The points that we actually need to process
        let mut non_trivial_points: Vec<(usize, f64)> = Vec::with_capacity(points.len());

        for point in sorted_points {
            if point.1 <= 0.0 {
                small_trivial_points += 1;
            } else if 1.0 <= point.1 {
                big_trivial_points += 1;
            } else {
                // move
                non_trivial_points.push(point);
            }
        }
        let non_trivial_points_len: usize = non_trivial_points.len();

        for i in 0..small_trivial_points {
            // quickly add all cases smaller than 0
            ret.push((i, bounds.0));
        }

        if non_trivial_points.is_empty() {
            let idx_offset: usize = small_trivial_points;
            for i in 0..big_trivial_points {
                ret.push((idx_offset + i, bounds.1));
            }
            ret.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            // ^this could be optimized by generating a vec and putting the values
            // according to indexes.  ( O(n) ) Also do below
            return ret.iter().map(|x| x.1).collect();
        }

        // Integration time!

        let pdf_checked = |x: f64| {
            if domain.contains(x) {
                self.pdf(x)
            } else {
                0.0
            }
        };

        let integration_type: IntegrationType = IntegrationType::from_bounds(bounds);

        let (step_length, _): (f64, usize) = choose_integration_precision_and_steps(bounds);

        let mut points_iter: std::vec::IntoIter<(usize, f64)> = non_trivial_points.into_iter();

        let (mut current_index, mut current_quantile): (usize, f64) = points_iter.next().unwrap(); // safe
                                                                                                   //let mut current_quantile: f64 = points_iter.next().unwrap().1; // safe

        let double_step_length: f64 = 2.0 * step_length;
        let step_len_over_3: f64 = step_length / 3.0;

        let mut accumulator: f64 = 0.0;
        // let mut last_pdf_evaluation: f64 = pdf_checked(bounds.0);
        let mut last_pdf_evaluation: f64 = match integration_type {
            IntegrationType::Finite => pdf_checked(bounds.0),
            IntegrationType::InfiniteToConst => {
                // t = 0, it would be a singularity. Skip point
                0.0
            }
            IntegrationType::ConstToInfinite => {
                // t = 0;     f(a + t/(t - 1))  /  (1 - t)^2
                pdf_checked(bounds.0)
            }
            IntegrationType::FullInfinite => {
                // t = -1;    f(t/(1 - t^2))  *  (1 + t^2) / (1 - t^2)^2
                // would be singularity, skip
                0.0
            }
        };

        let mut num_step: f64 = 0.0;
        'integration_loop: loop {
            let current_position: f64 = bounds.0 + double_step_length * num_step;
            //let middle: f64 = pdf_checked(current_position + step_length);
            //let end: f64 = pdf_checked(current_position + double_step_length);

            let (middle, end): (f64, f64) = match integration_type {
                IntegrationType::Finite => {
                    let middle_: f64 = pdf_checked(current_position + step_length);
                    let end_: f64 = pdf_checked(current_position + double_step_length);
                    (middle_, end_)
                }
                IntegrationType::InfiniteToConst => {
                    // In order to avoid the singularity at 0 we will split this into 2 parts: [-1, 0) and (0, 1]
                    //      For -infinite to const:
                    // integral {-inf -> a} f(x) dx = integral {0 -> 1} f(a - (1 - t)/t)  /  t^2  dt

                    let middle_: f64 = {
                        let t: f64 = current_position + step_length;
                        if t.abs() < f64::EPSILON {
                            // too near singularity, skip
                            0.0
                        } else {
                            pdf_checked(bounds.1 - (1.0 - t) / t) / (t * t)
                        }
                    };
                    let end_: f64 = {
                        let t: f64 = current_position + double_step_length;
                        if t.abs() < f64::EPSILON {
                            // too near singularity, skip
                            0.0
                        } else {
                            pdf_checked(bounds.1 - (1.0 - t) / t) / (t * t)
                        }
                    };
                    (middle_, end_)
                }
                IntegrationType::ConstToInfinite => {
                    //For const to infinite:
                    // integral {a -> inf} f(x) dx  = integral {0 -> 1} f(a + t/(t - 1))  /  (1 - t)^2  dt

                    let middle_: f64 = {
                        let t: f64 = current_position + step_length;
                        let t_minus: f64 = t - 1.0;
                        if t_minus.abs() < f64::EPSILON {
                            // too near singularity, skip
                            0.0
                        } else {
                            pdf_checked(bounds.0 + t / t_minus) / (t_minus * t_minus)
                        }
                    };
                    let end_: f64 = {
                        let t: f64 = current_position + double_step_length;
                        let t_minus: f64 = t - 1.0;
                        if t_minus.abs() < f64::EPSILON {
                            // too near singularity, skip
                            0.0
                        } else {
                            pdf_checked(bounds.0 + t / t_minus) / (t_minus * t_minus)
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
                            pdf_checked(t / u) * (1.0 + t * t) / (u * u)
                        }
                    };
                    let end_: f64 = {
                        let t: f64 = current_position + double_step_length;
                        let u: f64 = 1.0 - t * t;
                        if u.abs() < f64::EPSILON {
                            // too near singularity, skip
                            0.0
                        } else {
                            pdf_checked(t / u) * (1.0 + t * t) / (u * u)
                        }
                    };
                    (middle_, end_)
                }
            };

            accumulator += step_len_over_3 * (last_pdf_evaluation + 4.0 * middle + end);

            while current_quantile <= accumulator {
                let mut quantile: f64 = bounds.0 + double_step_length * (num_step + 1.0);

                let pdf_q: f64 = pdf_checked(quantile);
                if !(pdf_q.abs() < f64::EPSILON) && QUANTILE_USE_NEWTONS_ITER {
                    // if pdf_q is essentially 0, skip this.
                    // newton's iteration
                    quantile = quantile - (accumulator - current_quantile) / pdf_q;
                }

                ret.push((current_index, quantile));

                // update `current_quantile` to the next value or exit if we are done
                match points_iter.next() {
                    Some(p) => (current_index, current_quantile) = p,
                    None => break 'integration_loop,
                }
            }

            last_pdf_evaluation = end;
            num_step += 1.0;
            // we do 2 steps each iteration but at `current_position` we are mult. by `double_step_length`
        }

        for i in 0..big_trivial_points {
            let idx_offset = small_trivial_points + non_trivial_points_len;
            ret.push((idx_offset + i, bounds.1));
        }

        // put back to original order and return
        ret.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        // ^thic could be optimixed by generating a veg and putting the values
        // according to indexes.  ( O(n) ) Also do above
        return ret.iter().map(|x| x.1).collect();

        */
    }

    // Statistics

    /// Returns the [expected value](https://en.wikipedia.org/wiki/Expected_value)
    /// of the distribution if it exists.
    fn expected_value(&self) -> Option<f64> {
        return Some(self.moments(1, Moments::Raw));
    }

    /// Returns the [variance](https://en.wikipedia.org/wiki/Variance) of
    /// the distribution if it exists.
    fn variance(&self) -> Option<f64> {
        return Some(self.moments(2, Moments::Central));
    }

    /// Returns the [mode](https://en.wikipedia.org/wiki/Mode_(statistics))
    /// of the distribution.
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

        let USE_LOG_DISTRIBUTION: bool = configuration::distribution_deafult_mode::USE_LOG_DERIVATIVE; 

        let h: f64 = 0.001;
        let derivative = |x: f64| (self.pdf(x + h) - self.pdf(x)) / h; 
        let log_derivative = |x: f64| ((self.pdf(x + h) + f64::EPSILON).ln() - (self.pdf(x) + f64::EPSILON).ln()) / h; 
        

        let convergence_difference_criteria: f64 = configuration::distribution_deafult_mode::CONVERGENCE_DIFFERENCE_CRITERIA;
        let mut learning_rate: f64 = configuration::distribution_deafult_mode::LEARNING_RATE;
        let learning_rate_change: f64 = configuration::distribution_deafult_mode::LEARNING_RATE_CHANGE; 
        let min_iters: u32 = configuration::distribution_deafult_mode::MIN_ITERATIONS;
        let max_iters: u32 = configuration::distribution_deafult_mode::MAX_ITERATIONS;

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

    /// Returns the [skewness](https://en.wikipedia.org/wiki/Skewness)
    /// of the distribution.
    fn skewness(&self) -> Option<f64> {
        return Some(self.moments(3, Moments::Standarized));
    }

    /// Returns the [kurtosis](https://en.wikipedia.org/wiki/Kurtosis)
    /// of the distribution.
    fn kurtosis(&self) -> Option<f64> {
        return Some(self.moments(4, Moments::Standarized));
    }

    /// Returns the [excess kurtosis](https://en.wikipedia.org/wiki/Kurtosis#Excess_kurtosis)
    /// of the distribution.
    ///
    /// The excess kurtosis is defined as `kurtosis - 3`.
    fn excess_kurtosis(&self) -> Option<f64> {
        return self.kurtosis().map(|x| x - 3.0);
    }

    /// Returns the [moment](https://en.wikipedia.org/wiki/Moment_(mathematics))
    /// of the distribution and the given order. Mode determines if the moment will be
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

        let pdf_checked = |x: f64| {
            if domain.contains(x) {
                self.pdf(x)
            } else {
                0.0
            }
        };

        // The values of 0.0 and 1.0 have no special meaning. They are not going to be used anyway.
        let (mean, std_dev): (f64, f64) = match mode {
            Moments::Raw => (0.0, 1.0),
            Moments::Central => (
                self.expected_value()
                    .expect("We need expected value to continue"),
                1.0,
            ),
            Moments::Standarized => (
                self.expected_value()
                    .expect("We need expected value to continue"),
                self.variance().expect("We need variance value to continue"),
            ),
        };

        // Todo: give better error handling to the above. ^
        // println!("(mean, std_dev): {:?}", (mean, std_dev));

        let order_exp: i32 = order as i32;
        let (minus_mean, inv_std_dev) = (-mean, 1.0 / std_dev.sqrt());

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
                    let std_inp: f64 = (fn_input + minus_mean) * inv_std_dev;
                    let middle_: f64 = std_inp.powi(order_exp) * pdf_checked(fn_input);

                    let fn_input: f64 = current_position + double_step_length;
                    let std_inp: f64 = (fn_input + minus_mean) * inv_std_dev;
                    let end_: f64 = std_inp.powi(order_exp) * pdf_checked(fn_input);

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
                            let std_inp: f64 = (fn_input + minus_mean) * inv_std_dev;
                            std_inp.powi(order_exp) * pdf_checked(fn_input) / (t * t)
                        }
                    };
                    let end_: f64 = {
                        let t: f64 = current_position + double_step_length;
                        if t.abs() < f64::EPSILON {
                            // too near singularity, skip
                            0.0
                        } else {
                            let fn_input: f64 = bounds.1 - (1.0 - t) / t;
                            let std_inp: f64 = (fn_input + minus_mean) * inv_std_dev;
                            std_inp.powi(order_exp) * pdf_checked(fn_input) / (t * t)
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
                            let std_inp: f64 = (fn_input + minus_mean) * inv_std_dev;
                            std_inp.powi(order_exp) * pdf_checked(fn_input) / (t_minus * t_minus)
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
                            let std_inp: f64 = (fn_input + minus_mean) * inv_std_dev;
                            std_inp.powi(order_exp) * pdf_checked(fn_input) / (t_minus * t_minus)
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
                            let std_inp: f64 = (fn_input + minus_mean) * inv_std_dev;
                            std_inp.powi(order_exp) * pdf_checked(fn_input) * (1.0 + t * t)
                                / (u * u)
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
                            let std_inp: f64 = (fn_input + minus_mean) * inv_std_dev;
                            std_inp.powi(order_exp) * pdf_checked(fn_input) * (1.0 + t * t)
                                / (u * u)
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

    /// Returns the [entropy](https://en.wikipedia.org/wiki/Information_entropy)
    /// of the distribution
    fn entropy(&self) -> f64 {
        todo!("Implement deafult implementation. ");
    }

    // Other provided methods:
    // (methods that don't need to be replaced and should be here)

    /// Sample the distribution with the [rejection sampling](https://en.wikipedia.org/wiki/Rejection_sampling)
    /// method. In general, it is considerably more effitient that the normal [Distribution::sample]
    ///
    /// Important: [Distribution::rejection_sample] assumes a valid [Distribution::pdf] and
    /// a valid domain in [Distribution::get_domain]. Also the **domain must be finite**.
    /// If it is not, better use [Distribution::rejection_sample_range] or implement
    /// [Distribution::sample] yourself.
    ///
    /// It is more effitient because it does **not** requiere the evaluation of the
    /// [Distribution::quantile] function, wich involves numerical integration. In exchange,
    /// it is needed to know `pdf_max`, the maximum value that the pdf achives.
    ///
    /// Note: `pdf_max` does **not** need to be the real global maximum, it just needs
    /// to be equal or greater to it. Note that using a greater `pdf_max` value will incur
    /// a performance penalty.
    fn rejection_sample(&self, n: usize, pdf_max: f64) -> Vec<f64> {
        let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
        let domain: &ContinuousDomain = self.get_domain();
        let pdf_checked = |x: f64| {
            if domain.contains(x) {
                self.pdf(x)
            } else {
                0.0
            }
        };

        let bounds: (f64, f64) = domain.get_bounds();
        let bound_range: f64 = bounds.1 - bounds.0;

        let mut ret: Vec<f64> = Vec::with_capacity(n);
        for _i in 0..n {
            let sample: f64 = loop {
                let mut x: f64 = rng.gen();
                x = bounds.0 + x * bound_range;
                let y: f64 = rng.gen();
                if y * pdf_max < pdf_checked(x) {
                    break x;
                }
            };
            ret.push(sample);
        }

        return ret;
    }

    /// Same as [Distribution::rejection_sample] but only in the selected range.
    ///
    /// This can be usefull for distributions with a stricly infinite domain but that
    /// virtually all their mass is concentrated in a smaller region (`range`).
    ///
    /// For example, we could sample from the standard normal distribution with only
    /// the range `(-8.0, 8.0)` since the density left out of this range is negligible.
    fn rejection_sample_range(&self, n: usize, pdf_max: f64, range: (f64, f64)) -> Vec<f64> {
        let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
        let domain: &ContinuousDomain = self.get_domain();
        let pdf_checked = |x: f64| {
            if domain.contains(x) {
                self.pdf(x)
            } else {
                0.0
            }
        };

        let bounds: (f64, f64) = range;
        let bound_range: f64 = bounds.1 - bounds.0;

        let mut ret: Vec<f64> = Vec::with_capacity(n);
        for _i in 0..n {
            let sample: f64 = loop {
                let mut x: f64 = rng.gen();
                x = bounds.0 + x * bound_range;
                let y: f64 = rng.gen();
                if y * pdf_max < pdf_checked(x) {
                    break x;
                }
            };
            ret.push(sample);
        }

        return ret;
    }
}

/// The trait for any continuous distribution.
pub trait DiscreteDistribution {
    //Requiered method:

    /// Evaluates the [PMF](https://en.wikipedia.org/wiki/Probability_density_function)
    /// (Probability Mass Function) of the distribution at point x.
    /// The function should not be evaluated outside the domain (because it
    /// should return 0.0 anyway).
    ///
    /// The PMF is assumed to be a valid probability distribution. If you are not sure
    /// if the PMF is normalized to have a 1 unit of area under the curve of the pdf, you
    /// can use [crate::euclid::determine_normalitzation_constant_discrete].
    fn pdf(&self, x: f64) -> f64;

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
    /// Note that the deafult implemetation requieres numerical integration and
    /// may be expensive.
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

            For cdf_multiple we will first panic if we find a NaN.
            Otherwise we will need to sort them and (discretely) integrate until we have
            integrated to the given number (and store the value).
            By sorting, we only need to integrate once through the pdf.

            However, this *cool* strategy has a problem and is that we will not
            return the values in the order we were asked. To account for this we will
            keep track of the indicies of the origianl position and at the end re-sort
            the final array using them.

            Also, if we find any values smaller or greater than the bounds of the
            domain, the awnser will always be 0.0 or 1.0 (simplifying computations,
            although this case should not generally happen).

            If the lower bound is infinite, but the upper one is not, we will compute
            the probability from `x` to the upper bound and then do `1 - accumulator`
            to get the final awnser. If both bounds are infinite, we will start from 0
            and go adding the numbers and adding them on 2 accumulators (one for
            the mass in before `x` and another for after `x`), once the sum
            is greater than a threshold, we return the one that contains the awnser.

        */

        // return error if NAN is found
        for point in points {
            if point.is_nan() {
                panic!("Found NaN in `cdf_multiple`. \n");
                // return Err(AdvStatError::NanErr);
            }
        }
        todo!("Redo function");

        let mut sorted_points: Vec<(usize, f64)> = points.iter().map(|x| *x).enumerate().collect();

        sorted_points.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut ret: Vec<(usize, f64)> = Vec::with_capacity(points.len());
        let domain: &DiscreteDomain = self.get_domain();
        let bounds: (f64, f64) = domain.get_bounds();

        // if points[i] <= bounds.0  |||| Awnser is always 0
        let mut small_trivial_points: usize = 0;
        // if bounds.1 <= points[i]  |||| Awnser is always 1
        let mut big_trivial_points: usize = 0;
        // The points that we actually need to process
        let mut non_trivial_points: Vec<(usize, f64)> = Vec::with_capacity(points.len());

        for point in sorted_points {
            if point.1 <= bounds.0 {
                small_trivial_points += 1;
            } else if bounds.1 <= point.1 {
                big_trivial_points += 1;
            } else {
                // move
                non_trivial_points.push(point);
            }
        }
        let non_trivial_points_len: usize = non_trivial_points.len();

        for i in 0..small_trivial_points {
            // quickly add all cases smaller than bounds.0
            ret.push((i, 0.0));
        }

        if non_trivial_points.is_empty() {
            let idx_offset: usize = small_trivial_points;
            for i in 0..big_trivial_points {
                ret.push((idx_offset + i, 1.0));
            }
            ret.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            // ^this could be optimized by generating a vec and putting the values
            // according to indexes.  ( O(n) ) Also do below
            return ret.iter().map(|x| x.1).collect();
        }

        // Integration time!

        let pdf_checked = |x: f64| {
            if domain.contains(x) {
                self.pdf(x)
            } else {
                0.0
            }
        };

        let infinite_range: bool = bounds.0.is_infinite();

        let (step_length, _): (f64, usize) = choose_integration_precision_and_steps(bounds);

        let mut points_iter: std::vec::IntoIter<(usize, f64)> = non_trivial_points.into_iter();

        let (mut current_index, mut current_cdf_point): (usize, f64) = points_iter.next().unwrap(); // safe

        let double_step_length: f64 = 2.0 * step_length;
        let step_len_over_3: f64 = step_length / 3.0;

        let mut accumulator: f64 = 0.0;
        // let mut last_pdf_evaluation: f64 = pdf_checked(bounds.0);
        let mut last_pdf_evaluation: f64 = if infinite_range {
            // t = 0, it would be a singularity. Skip point
            0.0
        } else {
            pdf_checked(bounds.0)
        };

        let mut num_step: f64 = 0.0;
        'integration_loop: loop {
            let current_position: f64 = bounds.0 + double_step_length * num_step;

            while current_cdf_point <= current_position {
                ret.push((current_index, accumulator));

                // update `current_cdf_point` to the next value or exit if we are done
                match points_iter.next() {
                    Some(p) => (current_index, current_cdf_point) = p,
                    None => break 'integration_loop,
                }
            }

            let (middle, end): (f64, f64) = if infinite_range {
                // In order to avoid the singularity at 0 we will split this into 2 parts: [-1, 0) and (0, 1]
                //      For -infinite to const:
                // integral {-inf -> a} f(x) dx = integral {0 -> 1} f(a - (1 - t)/t)  /  t^2  dt

                let middle_: f64 = {
                    let t: f64 = current_position + step_length;
                    if t.abs() < f64::EPSILON {
                        // too near singularity, skip
                        0.0
                    } else {
                        pdf_checked(bounds.1 - (1.0 - t) / t) / (t * t)
                    }
                };
                let end_: f64 = {
                    let t: f64 = current_position + double_step_length;
                    if t.abs() < f64::EPSILON {
                        // too near singularity, skip
                        0.0
                    } else {
                        pdf_checked(bounds.1 - (1.0 - t) / t) / (t * t)
                    }
                };
                (middle_, end_)
            } else {
                let middle_: f64 = pdf_checked(current_position + step_length);
                let end_: f64 = pdf_checked(current_position + double_step_length);
                (middle_, end_)
            };

            accumulator += step_len_over_3 * (last_pdf_evaluation + 4.0 * middle + end);

            last_pdf_evaluation = end;
            num_step += 1.0;
            // we do 2 steps each iteration but at `current_position` we are mult. by `double_step_length`
        }

        for i in 0..big_trivial_points {
            let idx_offset = small_trivial_points + non_trivial_points_len;
            ret.push((idx_offset + i, 1.0));
        }

        // put back to original order and return
        ret.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        // ^thic could be optimixed by generating a veg and putting the values
        // according to indexes.  ( O(n) ) Also do above
        return ret.iter().map(|x| x.1).collect();
    }

    /// [Distribution::sample_multiple] allows to evaluate the [Distribution::sample]
    /// at multiple points. It may provide a computational advantage in comparasion to [Distribution::sample].
    ///
    /// The deafult implementation uses the [Distribution::quantile_multiple] function,
    /// wich may be expensive. Consider using [Distribution::rejection_sample] if possible.
    ///
    /// If an effitient [Distribution::sample] has been implemented, it can be replaced for:
    ///
    /// ```
    /// fn sample_multiple(&self, n: usize) -> Vec<f64> {
    ///     (0..n).map(|_| self.sample()).collect::<Vec<f64>>()
    /// }
    /// ```
    fn sample_multiple(&self, n: usize) -> Vec<f64> {
        let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
        let mut rand_quantiles: Vec<f64> = vec![0.0; n];
        rng.fill(rand_quantiles.as_mut_slice());

        let ret: Vec<f64> = self.quantile_multiple(&rand_quantiles);

        return ret;
    }

    /// quantile_multiple acts the same as [Distribution::quantile] but on multiple points.
    /// It provides a computational advantage over calling the normal [Distribution::quantile]
    /// multiple times.
    ///
    /// If there is any NaN in points, an error will be returned. If a value in points
    /// is less (or equal) to 0, the minimum value in the domain will be returned.
    /// If a value in points is greater (or equal) to 1, the maximum value in the
    /// domain will be returned.
    ///
    /// If an effitient [Distribution::quantile] has been implemented, it can be replaced for:
    ///
    /// ```
    /// fn quantile_multiple(&self, points: &[f64]) -> Result<Vec<f64>, crate::errors::AdvStatError> {
    ///     let list: Vec<f64> = points
    ///         .iter()
    ///         .map(|x| match self.quantile(*x) {
    ///             Ok(v) => v,
    ///             Err(_) => panic!("There has been an error! "),
    ///         })
    ///         .collect::<Vec<f64>>();
    ///     Ok(list)
    /// }
    /// ```
    fn quantile_multiple(&self, points: &[f64]) -> Vec<f64> {
        todo!("Redo function");

        /*
           Plan:

           For this function we will first return an error if we find a NaN.
           Otherwise we will need to sort them and integrate until the area under
           the pdf is = to the given number. By sorting, we only need to integrate
           once.

           However, this *cool* strategy has a problem and is that we will not
           return the values in the order we were asked. To account for this we will
           keep track of the indicies of the origianl position and at the end re-sort
           the final array using them.

           Also, if we find any values smaller or greater than 0 or 1, the awnser will
           always be the edges of the domain (simplifying computations, although this
           case should not normally happen).

           We will integrate using [Simpson's rule](https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_1/3_rule)
           for integration.

           To compute integrals over an infinite range, we will perform a special
           [numerial integration](https://en.wikipedia.org/wiki/Numerical_integration#Integrals_over_infinite_intervals).

               For -infinite to const:
           integral {-inf -> a} f(x) dx = integral {0 -> 1} f(a - (1 - t)/t)  /  t^2  dt

               For const to infinite:
           integral {a -> inf} f(x) dx  = integral {0 -> 1} f(a + t/(t - 1))  /  (1 - t)^2  dt

               For -infinite to infinite:
           integral {-inf -> inf} f(x) dx  = integral {-1 -> 1} f(t/(1 - t^2))  *  (1 + t^2) / (1 - t^2)^2  dt

           And "just" compute the new integrals (taking care of the singularities at t = 0).

        */

        // return error if NAN is found
        for point in points {
            if point.is_nan() {
                panic!("Tried to evaluate the quantile multiple function with a NaN value. \n");
                //return Err(AdvStatError::NanErr);
            }
        }

        let mut sorted_points: Vec<(usize, f64)> = points.iter().map(|x| *x).enumerate().collect();

        sorted_points.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut ret: Vec<(usize, f64)> = Vec::with_capacity(points.len());
        let domain: &DiscreteDomain = self.get_domain();
        let bounds: (f64, f64) = domain.get_bounds();

        // if points[i] <= 0  |||| Awnser is always bounds.0 (start of domain)
        let mut small_trivial_points: usize = 0;
        // if 1 <= points[i]  |||| Awnser is always bounds.1 (end of domain)
        let mut big_trivial_points: usize = 0;
        // The points that we actually need to process
        let mut non_trivial_points: Vec<(usize, f64)> = Vec::with_capacity(points.len());

        for point in sorted_points {
            if point.1 <= 0.0 {
                small_trivial_points += 1;
            } else if 1.0 <= point.1 {
                big_trivial_points += 1;
            } else {
                // move
                non_trivial_points.push(point);
            }
        }
        let non_trivial_points_len: usize = non_trivial_points.len();

        for i in 0..small_trivial_points {
            // quickly add all cases smaller than 0
            ret.push((i, bounds.0));
        }

        if non_trivial_points.is_empty() {
            let idx_offset: usize = small_trivial_points;
            for i in 0..big_trivial_points {
                ret.push((idx_offset + i, bounds.1));
            }
            ret.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            // ^this could be optimized by generating a vec and putting the values
            // according to indexes.  ( O(n) ) Also do below
            return ret.iter().map(|x| x.1).collect();
        }

        // Integration time!

        let pdf_checked = |x: f64| {
            if domain.contains(x) {
                self.pdf(x)
            } else {
                0.0
            }
        };

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

        let (step_length, _): (f64, usize) = choose_integration_precision_and_steps(bounds);

        let mut points_iter: std::vec::IntoIter<(usize, f64)> = non_trivial_points.into_iter();

        let (mut current_index, mut current_quantile): (usize, f64) = points_iter.next().unwrap(); // safe
                                                                                                   //let mut current_quantile: f64 = points_iter.next().unwrap().1; // safe

        let double_step_length: f64 = 2.0 * step_length;
        let step_len_over_3: f64 = step_length / 3.0;

        let mut accumulator: f64 = 0.0;
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

        let mut num_step: f64 = 0.0;
        'integration_loop: loop {
            let current_position: f64 = bounds.0 + double_step_length * num_step;
            //let middle: f64 = pdf_checked(current_position + step_length);
            //let end: f64 = pdf_checked(current_position + double_step_length);

            let (middle, end): (f64, f64) = match integration_type {
                IntegrationType::Finite => {
                    let middle_: f64 = pdf_checked(current_position + step_length);
                    let end_: f64 = pdf_checked(current_position + double_step_length);
                    (middle_, end_)
                }
                IntegrationType::InfiniteToPositive => {
                    // In order to avoid the singularity at 0 we will split this into 2 parts: [-1, 0) and (0, 1]
                    //      For -infinite to const:
                    // integral {-inf -> a} f(x) dx = integral {0 -> 1} f(a - (1 - t)/t)  /  t^2  dt

                    let middle_: f64 = {
                        let t: f64 = current_position + step_length;
                        if t.abs() < f64::EPSILON {
                            // too near singularity, skip
                            0.0
                        } else {
                            pdf_checked(bounds.1 - (1.0 - t) / t) / (t * t)
                        }
                    };
                    let end_: f64 = {
                        let t: f64 = current_position + double_step_length;
                        if t.abs() < f64::EPSILON {
                            // too near singularity, skip
                            0.0
                        } else {
                            pdf_checked(bounds.1 - (1.0 - t) / t) / (t * t)
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
                            pdf_checked(bounds.0 + t / t_minus) / (t_minus * t_minus)
                        }
                    };
                    let end_: f64 = {
                        let t: f64 = current_position + double_step_length;
                        let t_minus: f64 = t - 1.0;
                        if t_minus.abs() < f64::EPSILON {
                            // too near singularity, skip
                            0.0
                        } else {
                            pdf_checked(bounds.0 + t / t_minus) / (t_minus * t_minus)
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
                            pdf_checked(t / u) * (1.0 + t * t) / (u * u)
                        }
                    };
                    let end_: f64 = {
                        let t: f64 = current_position + double_step_length;
                        let u: f64 = 1.0 - t * t;
                        if u.abs() < f64::EPSILON {
                            // too near singularity, skip
                            0.0
                        } else {
                            pdf_checked(t / u) * (1.0 + t * t) / (u * u)
                        }
                    };
                    (middle_, end_)
                }
            };

            accumulator += step_len_over_3 * (last_pdf_evaluation + 4.0 * middle + end);

            while current_quantile <= accumulator {
                let mut quantile: f64 = bounds.0 + double_step_length * (num_step + 1.0);

                let pdf_q: f64 = pdf_checked(quantile);
                if !(pdf_q.abs() < f64::EPSILON) && QUANTILE_USE_NEWTONS_ITER {
                    // if pdf_q is essentially 0, skip this.
                    // newton's iteration
                    quantile = quantile - (accumulator - current_quantile) / pdf_q;
                }

                ret.push((current_index, quantile));

                // update `current_quantile` to the next value or exit if we are done
                match points_iter.next() {
                    Some(p) => (current_index, current_quantile) = p,
                    None => break 'integration_loop,
                }
            }

            last_pdf_evaluation = end;
            num_step += 1.0;
            // we do 2 steps each iteration but at `current_position` we are mult. by `double_step_length`
        }

        for i in 0..big_trivial_points {
            let idx_offset = small_trivial_points + non_trivial_points_len;
            ret.push((idx_offset + i, bounds.1));
        }

        // put back to original order and return
        ret.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        // ^thic could be optimixed by generating a veg and putting the values
        // according to indexes.  ( O(n) ) Also do above
        return ret.iter().map(|x| x.1).collect();
    }

    // Statistics

    /// Returns the [expected value](https://en.wikipedia.org/wiki/Expected_value)
    /// of the distribution if it exists.
    fn expected_value(&self) -> Option<f64> {
        return Some(self.moments(1, Moments::Raw));
    }

    /// Returns the [variance](https://en.wikipedia.org/wiki/Variance) of
    /// the distribution if it exists.
    fn variance(&self) -> Option<f64> {
        return Some(self.moments(2, Moments::Central));
    }

    /// Returns the [mode](https://en.wikipedia.org/wiki/Mode_(statistics))
    /// of the distribution.
    fn mode(&self) -> f64 {
        todo!("Implement deafult implementation. ");
    }

    /// Returns the [skewness](https://en.wikipedia.org/wiki/Skewness)
    /// of the distribution.
    fn skewness(&self) -> Option<f64> {
        return Some(self.moments(3, Moments::Standarized));
    }

    /// Returns the [kurtosis](https://en.wikipedia.org/wiki/Kurtosis)
    /// of the distribution.
    fn kurtosis(&self) -> Option<f64> {
        return Some(self.moments(4, Moments::Standarized));
    }

    /// Returns the [excess kurtosis](https://en.wikipedia.org/wiki/Kurtosis#Excess_kurtosis)
    /// of the distribution.
    ///
    /// The excess kurtosis is defined as `kurtosis - 3`.
    fn excess_kurtosis(&self) -> Option<f64> {
        return self.kurtosis().map(|x| x - 3.0);
    }

    /// Returns the [moment](https://en.wikipedia.org/wiki/Moment_(mathematics))
    /// of the distribution and the given order. Mode determines if the moment will be
    /// [Moments::Raw], [Moments::Central] or [Moments::Standarized].
    fn moments(&self, order: u8, mode: Moments) -> f64 {
        todo!("Redo function");

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

        let domain: &DiscreteDomain = self.get_domain();
        let bounds: (f64, f64) = domain.get_bounds();

        let pdf_checked = |x: f64| {
            if domain.contains(x) {
                self.pdf(x)
            } else {
                0.0
            }
        };

        // The values of 0.0 and 1.0 have no special meaning. They are not going to be used anyway.
        let (mean, std_dev): (f64, f64) = match mode {
            Moments::Raw => (0.0, 1.0),
            Moments::Central => (
                self.expected_value()
                    .expect("We need expected value to continue"),
                1.0,
            ),
            Moments::Standarized => (
                self.expected_value()
                    .expect("We need expected value to continue"),
                self.variance().expect("We need variance value to continue"),
            ),
        };

        // Todo: give better error handling to the above. ^
        // println!("(mean, std_dev): {:?}", (mean, std_dev));

        let order_exp: i32 = order as i32;
        let (minus_mean, inv_std_dev) = (-mean, 1.0 / std_dev.sqrt());

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
                    let std_inp: f64 = (fn_input + minus_mean) * inv_std_dev;
                    let middle_: f64 = std_inp.powi(order_exp) * pdf_checked(fn_input);

                    let fn_input: f64 = current_position + double_step_length;
                    let std_inp: f64 = (fn_input + minus_mean) * inv_std_dev;
                    let end_: f64 = std_inp.powi(order_exp) * pdf_checked(fn_input);

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
                            let std_inp: f64 = (fn_input + minus_mean) * inv_std_dev;
                            std_inp.powi(order_exp) * pdf_checked(fn_input) / (t * t)
                        }
                    };
                    let end_: f64 = {
                        let t: f64 = current_position + double_step_length;
                        if t.abs() < f64::EPSILON {
                            // too near singularity, skip
                            0.0
                        } else {
                            let fn_input: f64 = bounds.1 - (1.0 - t) / t;
                            let std_inp: f64 = (fn_input + minus_mean) * inv_std_dev;
                            std_inp.powi(order_exp) * pdf_checked(fn_input) / (t * t)
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
                            let std_inp: f64 = (fn_input + minus_mean) * inv_std_dev;
                            std_inp.powi(order_exp) * pdf_checked(fn_input) / (t_minus * t_minus)
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
                            let std_inp: f64 = (fn_input + minus_mean) * inv_std_dev;
                            std_inp.powi(order_exp) * pdf_checked(fn_input) / (t_minus * t_minus)
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
                            let std_inp: f64 = (fn_input + minus_mean) * inv_std_dev;
                            std_inp.powi(order_exp) * pdf_checked(fn_input) * (1.0 + t * t)
                                / (u * u)
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
                            let std_inp: f64 = (fn_input + minus_mean) * inv_std_dev;
                            std_inp.powi(order_exp) * pdf_checked(fn_input) * (1.0 + t * t)
                                / (u * u)
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

    /// Returns the [entropy](https://en.wikipedia.org/wiki/Information_entropy)
    /// of the distribution
    fn entropy(&self) -> f64 {
        todo!("Implement deafult implementation. ");
    }

    // Other provided methods:
    // (methods that don't need to be replaced and should be here)

    /// Sample the distribution with the [rejection sampling](https://en.wikipedia.org/wiki/Rejection_sampling)
    /// method. In general, it is considerably more effitient that the normal [Distribution::sample]
    ///
    /// Important: [Distribution::rejection_sample] assumes a valid [Distribution::pdf] and
    /// a valid domain in [Distribution::get_domain]. Also the **domain must be finite**.
    /// If it is not, better use [Distribution::rejection_sample_range] or implement
    /// [Distribution::sample] yourself.
    ///
    /// It is more effitient because it does **not** requiere the evaluation of the
    /// [Distribution::quantile] function, wich involves heavy computations. In exchange,
    /// it is needed to know `pdf_max`, the maximum value that the pdf achives.
    ///
    /// Note: `pdf_max` does **not** need to be the real global maximum, it just needs
    /// to be equal or greater to it. Note that using a greater `pdf_max` value will incur
    /// a performance penalty.
    fn rejection_sample(&self, n: usize, pdf_max: f64) -> Vec<f64> {
        todo!("Redo function");

        let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
        let domain: &DiscreteDomain = self.get_domain();
        let pdf_checked = |x: f64| {
            if domain.contains(x) {
                self.pdf(x)
            } else {
                0.0
            }
        };

        let bounds: (f64, f64) = domain.get_bounds();
        let bound_range: f64 = bounds.1 - bounds.0;

        let mut ret: Vec<f64> = Vec::with_capacity(n);
        for _i in 0..n {
            let sample: f64 = loop {
                let mut x: f64 = rng.gen();
                x = bounds.0 + x * bound_range;
                let y: f64 = rng.gen();
                if y * pdf_max < pdf_checked(x) {
                    break x;
                }
            };
            ret.push(sample);
        }

        return ret;
    }

    /// Same as [Distribution::rejection_sample] but only in the selected range.
    ///
    /// This can be usefull for distributions with a stricly infinite domain but that
    /// virtually all their mass is concentrated in a smaller region (`range`).
    ///
    /// For example, we could sample from the standard normal distribution with only
    /// the range `(-8.0, 8.0)` since the density left out of this range is negligible.
    fn rejection_sample_range(&self, n: usize, pdf_max: f64, range: (f64, f64)) -> Vec<f64> {
        todo!("Redo function");

        let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
        let domain: &DiscreteDomain = self.get_domain();
        let pdf_checked = |x: f64| {
            if domain.contains(x) {
                self.pdf(x)
            } else {
                0.0
            }
        };

        let bounds: (f64, f64) = range;
        let bound_range: f64 = bounds.1 - bounds.0;

        let mut ret: Vec<f64> = Vec::with_capacity(n);
        for _i in 0..n {
            let sample: f64 = loop {
                let mut x: f64 = rng.gen();
                x = bounds.0 + x * bound_range;
                let y: f64 = rng.gen();
                if y * pdf_max < pdf_checked(x) {
                    break x;
                }
            };
            ret.push(sample);
        }

        return ret;
    }
}
