use std::usize;

use rand::Rng;

///! This script contains the interfaces used to comunicate with the distributions.
use crate::errors::*;
use crate::euclid::*;
use crate::QUANTILE_USE_NEWTONS_ITER;
use crate::RNG;
use crate::SMALL_INTEGRATION_NUM_STEPS;
use crate::SMALL_INTEGRATION_PRECISION;
use crate::{
    DEFAULT_INTEGRATION_MAXIMUM_STEPS, DEFAULT_INTEGRATION_MAXIMUM_STEPS_F64,
    DEFAULT_INTEGRATION_PRECISION,
};

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
    ///  - if `x` is in the range [0.0, 1.0], the bounds of the domain will be retruned.
    ///  - `x` must be a non-Nan or an error will be returned.
    ///
    /// The quantile function is the inverse function of [Distribution::cdf]. Note that
    /// the deafult implemetation requieres numerical integration and may be expensive.
    ///
    /// Also, if you are considering calling this function multiple times, use
    /// [Distribution::quantile_multiple] for better performance.
    fn quantile(&self, x: f64) -> Result<f64, AdvStatError> {
        // just call [Distribution::quantile_multiple]

        if x.is_nan() {
            // x is not valid
            return Err(AdvStatError::DomainErr);
        }

        let value: [f64; 1] = [x];
        let quantile_vec: Vec<f64> = self.quantile_multiple(&value)?;
        return Ok(quantile_vec[0]);

        /*

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
                let step_len_over_3: f64 = step_length / 3.0;

                let mut accumulator: f64 = 0.0;
                let mut last_pdf_evaluation: f64 = pdf_checked(bounds.0, &domain);
                let mut num_step: f64 = 0.0;
                loop {
                    let current_position: f64 = bounds.0 + double_step_length * num_step;
                    let middle: f64 = pdf_checked(current_position + step_length, domain);
                    let end: f64 = pdf_checked(current_position + double_step_length, domain);

                    accumulator += step_len_over_3 * (last_pdf_evaluation + 4.0 * middle + end);

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
                let quantile_1: f64 = quantile_0 - (accumulator - x) / pdf_q;
                return Ok(quantile_1);

            },
            (true, false) => todo!(),
            (false, true) => todo!(),
            (false, false) => todo!(),
        }

        todo!("Implement deafult implementation. ");
        */
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
    /// It provides a computational advantage over calling the normal [Distribution::quantile]
    /// multiple times.
    ///
    /// If there is any NaN in points, an error will be returned. If a value in points
    /// is less (or equal) to 0, the minimum value in the domain will be returned.
    /// If a value in points is greater (or equal) to 1, the maximum value in the
    /// domain will be returned.
    fn quantile_multiple(&self, points: &[f64]) -> Result<Vec<f64>, AdvStatError> {
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
                return Err(AdvStatError::NanErr);
            }
        }

        let mut sorted_points: Vec<(usize, f64)> = points.iter().map(|x| *x).enumerate().collect();

        sorted_points.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut ret: Vec<(usize, f64)> = Vec::with_capacity(points.len());
        let domain: &Domain = self.get_pdf_domain();
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
            // ^thic could be optimixed by generating a veg and putting the values
            // according to indexes.  ( O(n) ) Also do bellow
            return Ok(ret.iter().map(|x| x.1).collect());
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

        let (step_length, total_num_steps): (f64, usize) = {
            /*
               To select the appropiate step_length (and total_num_steps, indirecly),
               we need to adapt between the possible cases.
                - For standard integration: Use DEFAULT_INTEGRATION_PRECISION unless it's
                   too small (if we would do more than DEFAULT_INTEGRATION_MAXIMUM_STEPS)

            */

            match integration_type {
                IntegrationType::Finite => {
                    // standard integration
                    let integration_range: f64 = bounds.1 - bounds.0;
                    if DEFAULT_INTEGRATION_PRECISION * DEFAULT_INTEGRATION_MAXIMUM_STEPS_F64
                        < integration_range
                    {
                        let alternative_step_length: f64 =
                            integration_range / DEFAULT_INTEGRATION_MAXIMUM_STEPS_F64;
                        (alternative_step_length, DEFAULT_INTEGRATION_MAXIMUM_STEPS)
                        // DEFAULT_INTEGRATION_MAXIMUM_STEPS is even
                    } else {
                        let number_steps: usize =
                            ((integration_range / DEFAULT_INTEGRATION_PRECISION) as usize) | 1;
                        // ` x | 1 ` makes sure number_steps is even
                        let corrected_step_length: f64 = integration_range / (number_steps as f64);
                        (corrected_step_length, number_steps)
                    }
                }
                IntegrationType::FullInfinite => {
                    // if the interval [0, 1] uses SMALL_INTEGRATION_NUM_STEPS,
                    // then [-1, 1] will use the doule. Ajust precision accordingly.
                    (
                        SMALL_INTEGRATION_PRECISION * 0.5,
                        (SMALL_INTEGRATION_NUM_STEPS * 2) as usize,
                    )
                }
                _ => {
                    // IntegrationType::InfiniteToPositive
                    // IntegrationType::InfiniteToNegative

                    // just use the following deafult values since the interval is [0, 1]
                    (
                        SMALL_INTEGRATION_PRECISION,
                        SMALL_INTEGRATION_NUM_STEPS as usize,
                    )
                }
            }
        };

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
                            pdf_checked(t / u) * (1.0 + t * t) / u
                        }
                    };
                    let end_: f64 = {
                        let t: f64 = current_position + double_step_length;
                        let u: f64 = 1.0 - t * t;
                        if u.abs() < f64::EPSILON {
                            // too near singularity, skip
                            0.0
                        } else {
                            pdf_checked(t / u) * (1.0 + t * t) / u
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
            num_step += 2.0; // we do 2 steps each iteration
        }

        for i in 0..big_trivial_points {
            let idx_offset = small_trivial_points + non_trivial_points_len;
            ret.push((idx_offset + i, bounds.1));
        }

        // put back to original order and return
        ret.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        // ^thic could be optimixed by generating a veg and putting the values
        // according to indexes.  ( O(n) ) Also do above
        return Ok(ret.iter().map(|x| x.1).collect());
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
