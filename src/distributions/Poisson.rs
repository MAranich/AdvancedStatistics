//! # Poisson distribution
//!
//! The [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution)
//! is a discrete distribution that counts the number of poisson events in a
//! given time with a given rate.
//!
//! The poisson distribution has a single parameter: the rate `lambda`. Lambda
//! represents the avarage number of events that happen in a given amount of time.
//!
//! Some properties of the Poisson distribution:
//!  - The sumation of 2 Poisson distributed random variables is also
//!     poisson distributed with the sum of the 2 original lambdas as
//!     it's own lambda.
//!      - This means that for large lambdas, the [CLT](https://en.wikipedia.org/wiki/Central_limit_theorem)
//!         aplies and the distributions get very close to a (discrete) normal distribution.

use std::num::NonZero;

use rand::Rng;

use crate::{
    distribution_trait::{DiscreteDistribution, Parametric},
    domain::DiscreteDomain,
    euclid::{ln_gamma, ln_gamma_int},
};

pub const POISSON_DOMAIN: DiscreteDomain = DiscreteDomain::From(0);

#[derive(Debug, Clone, PartialEq)]
pub struct Poisson {
    lambda: f64,
}

impl Poisson {
    /// Creates a new [Poisson] distribution.
    ///
    ///  - `lambda` indicates rate. And must fullfill:
    ///      - Must be finite (no `+-inf` nor NaNs)
    ///      - `0.0 < lambda`
    ///
    /// Otherwise an error will be returned.
    pub const fn new(lambda: f64) -> Result<Poisson, ()> {
        if !lambda.is_finite() {
            return Err(());
        }

        if lambda <= 0.0 {
            return Err(());
        }

        return Ok(Poisson { lambda });
    }

    /// Creates a new [Poisson] distribution without checking.
    ///
    ///  - `lambda` indicates rate. And must fullfill:
    ///      - Must be finite (no `+-inf` nor NaNs)
    ///      - `0.0 < lambda`
    ///
    /// If those conditions are not fullfiled, the returned distribution
    /// will be invalid.
    pub const unsafe fn new_unchecked(lambda: f64) -> Poisson {
        return Poisson { lambda };
    }

    /// Returns the value of `lambda`
    pub const fn get_lambda(&self) -> f64 {
        return self.lambda;
    }
}

impl DiscreteDistribution for Poisson {
    fn pmf(&self, mut x: f64) -> f64 {
        /* Usual definition:
         > P(x | lambda) = exp(-lambda) * lambda^x / x!

        But for better precision, we will use the following alternative equivalent:

         > P(x | lambda) = exp( x * ln(lambda) - lambda - ln(Gamma(k + 1)) )

        Also we will round `x` down to the nearest integer.

        */

        x = x.floor();

        // assuming `0.0 <= x`
        //let ln_gamma: f64 = ln_gamma_int(NonZero::new(x as u64 + 1).unwrap());
        let ln_gamma: f64 = ln_gamma(x + 1.0); 
        let inner_exp: f64 = x * self.lambda.ln() - self.lambda - ln_gamma;

        return inner_exp.exp();
    }

    fn get_domain(&self) -> &DiscreteDomain {
        return &POISSON_DOMAIN;
    }

    // use default cdf, sample and quantile

    fn cdf_multiple(&self, points: &[f64]) -> Vec<f64> {
        /*
               Plan:
           We will use the deafult implementation as base and we will
           make some ajustments to improve performance.
           We will remove the unnecessary matches to `integration_type`
           and optimitzation for the computation of `ln_gamma`.
           Instead of recomputing the value each time (wich can be very expensive),
           we use the previous vale to compute the next one.
        */
        if points.is_empty() {
            return Vec::new();
        }

        // panic if NAN is found
        for point in points {
            if point.is_nan() {
                std::panic!("Found NaN in `cdf_multiple` of Poisson. \n");
            }
        }

        let mut ret: Vec<f64> = std::vec![0.0; points.len()];
        let domain: &DiscreteDomain = self.get_domain();
        let bounds: (f64, f64) = domain.get_bounds();

        let mut sorted_indicies: Vec<usize> = (0..points.len()).into_iter().collect::<Vec<usize>>();

        sorted_indicies.sort_unstable_by(|&i, &j| {
            let a: f64 = points[i];
            let b: f64 = points[j];
            a.partial_cmp(&b).unwrap()
        });

        let mut idx_iter: std::vec::IntoIter<usize> = sorted_indicies.into_iter();
        let mut current_index: usize = idx_iter.next().unwrap();
        // ^unwrap is safe

        let mut current_cdf_point: f64 = points[current_index];

        let mut accumulator: f64 = 0.0;

        while current_cdf_point < bounds.0 {
            ret[current_index] = 0.0;
            match idx_iter.next() {
                Some(v) => current_index = v,
                None => return ret,
            }
            current_cdf_point = points[current_index];
        }

        let mut ln_gamma: f64 = 0.0;
        let mut x: f64 = 0.0;
        loop {
            while current_cdf_point < x {
                ret[current_index] = accumulator;
                match idx_iter.next() {
                    Some(v) => current_index = v,
                    None => return ret,
                }
                current_cdf_point = points[current_index];
            }

            let inner_exp: f64 = x * self.lambda.ln() - self.lambda - ln_gamma;
            let pmf: f64 = inner_exp.exp();

            accumulator += pmf;
            x += 1.0;
            ln_gamma += x.ln()
        }
    }

    fn sample_multiple(&self, n: usize) -> Vec<f64> {
        /*
            Source method:
            https://en.wikipedia.org/wiki/Poisson_distribution#Random_variate_generation

            The method is aprox. O(lambda), wich means that it is very fast for
            small lambdas but very slow for higher ones.

            Todo: search a better algorithm for bugger lambdas.

        */
        let mut rng: rand::prelude::ThreadRng = rand::rng();
        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(n);

        let L: f64 = (-self.lambda).exp();
        let mut p: f64;
        let mut k: f64;

        for _ in 0..n {
            p = 1.0;
            k = 0.0;
            loop {
                k += 1.0;
                let u: f64 = rng.random();
                p = p * u;

                if p <= L {
                    break;
                }
            }
            ret.push(k - 1.0);
        }

        return ret;
    }

    fn quantile_multiple(&self, points: &[f64]) -> Vec<f64> {
        /*
                Plan:
            Same plan as in [Poisson::cdf_multiple].
            We will use the deafult implementation as base and we will
            make some ajustments to improve performance.
            We will remove the unnecessary matches to `integration_type`
            and optimitzation for the computation of `ln_gamma`.
            Instead of recomputing the value each time (wich can be very expensive),
            we use the previous vale to compute the next one.
        */

        if points.is_empty() {
            return Vec::new();
        }

        // panic if NAN is found
        for point in points {
            if point.is_nan() {
                std::panic!("Found NaN in `quantile_multiple` for Poisson. \n");
            }
        }

        let mut ret: Vec<f64> = std::vec![0.0; points.len()];
        // let bounds: (f64, f64) = domain.get_bounds();
        // We already know: `bounds = (0.0, inf)`

        let mut sorted_indicies: Vec<usize> = (0..points.len()).into_iter().collect::<Vec<usize>>();

        sorted_indicies.sort_unstable_by(|&i, &j| {
            let a: f64 = points[i];
            let b: f64 = points[j];
            a.partial_cmp(&b).unwrap()
        });

        let mut idx_iter: std::vec::IntoIter<usize> = sorted_indicies.into_iter();
        let mut current_index: usize = idx_iter.next().unwrap();
        // ^unwrap is safe

        let mut current_quantile_point: f64 = points[current_index];

        let mut accumulator: f64 = 0.0;

        while current_quantile_point < 0.0 {
            ret[current_index] = 0.0;
            match idx_iter.next() {
                Some(v) => current_index = v,
                None => return ret,
            }
            current_quantile_point = points[current_index];
        }

        let mut ln_gamma: f64 = 0.0;
        let mut x: f64 = 0.0;
        loop {

            let inner_exp: f64 = x * self.lambda.ln() - self.lambda - ln_gamma;
            let pmf: f64 = inner_exp.exp();

            accumulator += pmf;
            //println!("pmf({})\t = {}", x, pmf); 
            //println!("cdf({})\t = {}", x, accumulator); 

            while current_quantile_point <= accumulator {
                ret[current_index] = x;
                match idx_iter.next() {
                    Some(v) => current_index = v,
                    None => return ret,
                }
                current_quantile_point = points[current_index];
            }

            x += 1.0;
            ln_gamma += x.ln();
        }
    }

    fn expected_value(&self) -> Option<f64> {
        return Some(self.lambda);
    }

    fn variance(&self) -> Option<f64> {
        return Some(self.lambda);
    }

    fn mode(&self) -> f64 {
        return self.lambda.floor();
    }

    // use defaut median because there is not any exact formula,
    // just bounds / aproximations (even if very good)

    fn skewness(&self) -> Option<f64> {
        return Some(1.0 / self.lambda.sqrt());
    }

    fn kurtosis(&self) -> Option<f64> {
        return self.excess_kurtosis().map(|x| x + 3.0);
    }

    fn excess_kurtosis(&self) -> Option<f64> {
        return Some(1.0 / self.lambda);
    }

    /*
        Todo:
        there are some formulas for `entropy`.
    */
}

impl Parametric for Poisson {
    /// Evaluates the [PDF](https://en.wikipedia.org/wiki/Probability_density_function)
    /// (Probability Density function) of the distribution at point `x` with
    /// the given `parameters`.
    ///
    /// If follows the same constraits as the normal
    /// [Distribution::pdf](crate::distribution_trait::Distribution::pdf)
    /// (or [DiscreteDistribution::pmf])
    /// but also taking the parameters into account.
    ///
    /// ### Parameters for Poisson:
    ///
    /// The Poisson distribution has only 1 parameter, `lambda`.
    fn general_pdf(&self, mut x: f64, parameters: &[f64]) -> f64 {
        /*
         > P(x | lambda) = exp(-lambda) * lambda^x / x!
         > P(x | lambda) = exp( x * ln(lambda) - lambda - ln(Gamma(k + 1)) )
        */

        x = x.floor();

        // assuming `0.0 <= x`
        let ln_gamma: f64 = ln_gamma_int(NonZero::new(x as u64 + 1).unwrap());
        let inner_exp: f64 = x * parameters[0].ln() - parameters[0] - ln_gamma;

        return inner_exp.exp();
    }

    fn number_of_parameters() -> u16 {
        return 1;
    }

    fn get_parameters(&self, parameters: &mut [f64]) {
        parameters[0] = self.lambda;
    }

    fn derivative_pdf_parameters(&self, x: f64, parameters: &[f64]) -> Vec<f64> {
        // d/dx ln(f(x)) = f'(x)/f(x)
        // => f(x) * d/dx ln(f(x)) = f'(x)

        // P(x | lambda) = exp(-lambda) * lambda^x / x!

        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(2);

        let lambda: f64 = parameters[0];

        //## Derivative respect to lambda:
        // there is no derivative for non-continuous functions:

        ret.push(0.0);

        {
            //## Derivative respect to lambda:
            /*
                d/dx P(x | lambda) = d/dx exp(-lambda) * lambda^x / x!
                 = 1.0 / x! * d/dx exp(-lambda) * lambda^x
                 = 1.0 / x! * (d/dx[ exp(-lambda) ] * lambda^x + d/dx[ lambda^x ] * exp(-lambda))
                 = 1.0 / x! * (exp(-lambda) * d/dx[ -lambda ] * lambda^x + x*lambda^(x-1) * exp(-lambda))
                 = 1.0 / x! * (exp(-lambda) * -1 * lambda^x + x*lambda^(x-1) * exp(-lambda))
                 = 1.0 / x! * (exp(-lambda) * -lambda^x + x*lambda^(x-1) * exp(-lambda))
                 = 1.0 / x! * exp(-lambda) * (-lambda^x + x*lambda^(x-1))
                 = exp(-lambda) * (x*lambda^(x-1) - lambda^x) / x!

                ### Confirmation:

                f(x) * d/dx ln(f(x)) = f'(x)

                d/d_lambda ln(pdf(x | lambda)) = x/lambda - 1

                pmf(x) * d/d_lambda pmf(f(x)) = exp(-lambda) * lambda^x / x! * (x/lambda - 1)
                 = exp(-lambda) / x! * (lambda^x * x/lambda - lambda^x)
                 = exp(-lambda) * (x*lambda^(x-1) - lambda^x) / x!

                Wich is identical to the solution we got for the normal derivative.

                ### Conclusion

                I think that for this case it will be simpler to just perform:
                 > exp(-lambda) * lambda^x / x! * (x/lambda - 1)
                 > = pdf(x | lambda) * (x/lambda - 1)
            */

            let term: f64 = x / lambda - 1.0;
            ret.push(self.general_pdf(x, parameters) * term);
        }

        return ret;
    }

    fn log_derivative_pdf_parameters(&self, mut x: f64, parameters: &[f64]) -> Vec<f64> {
        // d/dx ln(f(x)) = f'(x)/f(x)

        // pmf(x | lambda) = exp(-lambda) * lambda^x / x!
        // ln(P(x | lambda)) = ln(exp(-lambda) * lambda^x / x!)
        // ln(P(x | lambda)) = ln(exp(-lambda)) + ln(lambda^x) + ln(1.0 / x!)
        // ln(P(x | lambda)) = -lambda + x*ln(lambda) + -ln(x!)
        // ln(P(x | lambda)) = -lambda + x*ln(lambda) + -ln(gamma(x + 1)!)

        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(2);

        let lambda: f64 = parameters[0];
        x = x.floor();

        // ## Log derivative respect x:
        ret[0] = 0.0;

        {
            // ## Log derivative respect lambda:
            /*
               d/d_lambda ln(pmf(x | lambda)) = d/d_lambda -lambda + x*ln(lambda) + -ln(gamma(x + 1)!)
                = -1 + x/lambda + 0
                = x/lambda - 1

            */

            ret[1] = x / lambda - 1.0;
        }

        return ret;
    }

    fn parameter_restriction(&self, parameters: &mut [f64]) {
        let ep: f64 = f64::EPSILON;
        parameters[0] = parameters[0].max(ep * ep * ep)
    }

    fn fit(&self, data: &mut crate::samples::Samples) -> Vec<f64> {
        let mut parameters: Vec<f64> = Vec::new();
        parameters.reserve_exact(1);

        /*
                Estimation of p:

            Using Maximum Likelyhood estimation:
            Assuming k samples.

            pmf(x | lambda) = exp(-lambda) * lambda^x / x!
            d/d_lambda ln(pdf(x | lambda)) = x/lambda - 1
            0 = sumatory{x_i} x_i/lambda - 1
            0 = -k + sumatory{x_i} x_i/lambda
            k = 1/lambda * sumatory{x_i} x_i
            lambda * k = sumatory{x_i} x_i
            lambda = 1/k * sumatory{x_i} x_i
            lambda = mean{x_i}

                ### Deafult values:

            If there are not enough samples to compute the estimators, then
            this method will return the following deafult probabilities:
             - `lambda`: 1.0

        */

        let mean: f64 = match data.mean() {
            Some(m) => m,
            None => {
                // early return deafult lambda = 1.0
                parameters.push(1.0);
                return parameters;
            }
        };

        parameters.push(mean);
        return parameters;
    }
}

impl Default for Poisson {
    fn default() -> Self {
        Poisson::new(1.0).unwrap()
    }
}
