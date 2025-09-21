//! # Exponential distribution
//!
//! The [Exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution)
//! is a continuous distribution very important on statistics that measures
//! the time to the next poission event.
//!
//! A poission event does not have memory. Mathematically, if `e` follows
//! an Exponential distribution and `t_1 < t_2`
//! `P(t_1 < e) = P(t_1 < e | t_2 < e)`
//!
//! The Exponential distribution has a parameter: the rate `lambda` wich determines
//! how fast do events happen.

use rand::Rng;

use crate::{
    distribution_trait::{Distribution, Parametric},
    domain::ContinuousDomain,
    errors::AdvStatError,
    euclid,
};

pub const EXPONENTIAL_DOMAIN: ContinuousDomain = ContinuousDomain::From(0.0);

/// An [Exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution).
#[derive(Debug, Clone, PartialEq)]
pub struct Exponential {
    lambda: f64,
}

/// An iterator that generates infinites samples form the exponential distribution
/// faster than than normally calling [Exponential::sample] many times.
pub struct ExponentialGenerator {
    inv_lambda: f64,
    rng: rand::prelude::ThreadRng,
}

impl Exponential {
    /// Creates a new [Exponential] distribution.
    ///
    ///  - `lambda` must be finite (no `+-inf` nor NaNs)
    ///  - `0.0 < lambda`
    ///
    /// Otherwise an error will be returned.
    pub const fn new(lambda: f64) -> Result<Exponential, AdvStatError> {
        if !lambda.is_finite() {
            if lambda.is_nan() {
                return Err(AdvStatError::NanErr);
            } else if lambda.is_infinite() {
                return Err(AdvStatError::InvalidNumber);
            }
        }

        if lambda <= 0.0 {
            return Err(AdvStatError::InvalidNumber);
        }

        return Ok(Exponential { lambda });
    }

    /// Creates a new [Exponential] distribution.
    ///
    /// ## Safety
    ///
    /// If the following conditions are not fullfiled, the returned distribution
    /// will be invalid.
    ///
    ///  - `lambda` must be finite (no `+-inf` nor NaNs)
    ///  - `0.0 < lambda`
    ///
    #[must_use]
    pub const unsafe fn new_unchecked(lambda: f64) -> Exponential {
        return Exponential { lambda };
    }

    #[must_use]
    pub const fn get_lambda(&self) -> f64 {
        return self.lambda;
    }

    /// Returns an iterator that can generate [Exponential] samples even faster
    /// than normally calling [Exponential::sample] many times. Uscefull if you don't
    /// know exacly how many values you want for [Exponential::sample_multiple].
    ///
    /// It avoids the heap allocation of [Exponential::sample_multiple] and
    /// the repeated initialitzation processes in [Exponential::sample].
    #[must_use]
    pub fn iter(&self) -> ExponentialGenerator {
        return ExponentialGenerator {
            inv_lambda: 1.0 / self.lambda,
            rng: rand::rng(),
        };
    }
}

impl Distribution for Exponential {
    fn pdf(&self, x: f64) -> f64 {
        return self.lambda * (-self.lambda * x).exp();
    }

    fn get_domain(&self) -> &crate::domain::ContinuousDomain {
        return &EXPONENTIAL_DOMAIN;
    }

    fn cdf(&self, x: f64) -> f64 {
        assert!(
            !x.is_nan(),
            "Found NaN while attempting to compute the cdf of an Exponential. \n"
        );

        if x <= 0.0 {
            return 0.0;
        }

        return 1.0 - (-self.lambda * x).exp();
    }

    fn sample(&self) -> f64 {
        let mut rng: rand::prelude::ThreadRng = rand::rng();
        let r: f64 = rng.random();
        return -r.ln() / self.lambda;
    }

    fn quantile(&self, x: f64) -> f64 {
        assert!(
            !x.is_nan(),
            "Tried to evaluate the Exponential::quantile function with a NaN value. \n"
        );

        if x <= 0.0 {
            return 0.0;
        }

        if 1.0 <= x {
            return f64::INFINITY;
        }

        return -(1.0 - x).ln() / self.lambda;
    }

    fn cdf_fill(&self, points: &mut [f64]) {
        for x in points {
            assert!(
                !x.is_nan(),
                "Tried to evaluate the Exponential::cdf_fill function with a NaN value. \n"
            );

            *x = if *x <= 0.0 {
                0.0
            } else {
                1.0 - (-self.lambda * (*x)).exp()
            };
        }
    }

    fn sample_fill(&self, buffer: &mut [f64]) {
        let mut rng: rand::prelude::ThreadRng = rand::rng();
        rng.fill(buffer);

        let minus_inv_lambda: f64 = -1.0 / self.lambda;
        for r in buffer.iter_mut() {
            *r = minus_inv_lambda * r.ln();
        }
    }

    fn quantile_fill(&self, points: &mut [f64]) {
        let minus_inv_lambda: f64 = -1.0 / self.lambda;
        for x in points.iter_mut() {
            assert!(
                !x.is_nan(),
                "Tried to evaluate the Exponential::quantile_fill function with a NaN value. \n"
            );

            if *x <= 0.0 {
                *x = 0.0;
            } else if 1.0 <= *x {
                *x = f64::INFINITY;
            } else {
                *x = minus_inv_lambda * x.ln();
            }
        }
    }

    fn expected_value(&self) -> Option<f64> {
        return Some(1.0 / self.lambda);
    }

    fn variance(&self) -> Option<f64> {
        return Some(1.0 / (self.lambda * self.lambda));
    }

    fn mode(&self) -> f64 {
        0.0
    }

    fn median(&self) -> f64 {
        2.0_f64.ln() / self.lambda
    }

    fn skewness(&self) -> Option<f64> {
        return Some(2.0);
    }

    fn excess_kurtosis(&self) -> Option<f64> {
        return Some(6.0);
    }

    fn moments(&self, order: u8, mode: euclid::Moments) -> f64 {
        if let euclid::Moments::Standarized = mode {
            // use default implementation
            // SAFETY: the `expected_value` fn will always return the some variant for exponential
            let mean: f64 = unsafe { self.expected_value().unwrap_unchecked() };
            // SAFETY: the `variance` fn will always return the some variant for exponential
            let variance: f64 = unsafe { self.variance().unwrap_unchecked() };

            return self.default_moments(order, mean, variance);
        }

        // For central and raw moments we have formulas: https://en.wikipedia.org/wiki/Exponential_distribution#Mean,_variance,_moments,_and_median

        let factorial: f64 = match euclid::combinatorics::factorial(order) {
            Ok(v_int) => v_int as f64,
            Err(v) => v,
        };

        let power: f64 = self.lambda.powi(i32::from(order));

        let main_term: f64 = factorial / power;

        if let euclid::Moments::Raw = mode {
            return main_term;
        }

        // euclid::Moments::Central case

        if order == 0 {
            //special case for 0 because the formula does not work for 0
            // the central moment of order 0 is always 0
            return 0.0;
        }

        // formula obtained from the formula for the central moments and https://en.wikipedia.org/wiki/Derangement#

        return (main_term * euclid::INV_E).round();
    }

    fn entropy(&self) -> f64 {
        return 1.0 - self.lambda.ln();
    }
}

impl Parametric for Exponential {
    /// Evaluates the [PDF](https://en.wikipedia.org/wiki/Probability_density_function)
    /// (Probability Density function) of the distribution at point `x` with
    /// the given `parameters`.
    ///
    /// If follows the same constraits as the normal
    /// [Distribution::pdf]
    /// (or [DiscreteDistribution::pmf](crate::distribution_trait::DiscreteDistribution::pmf))
    /// but also taking the parameters into account.
    ///
    /// ### Parameters for [Exponential]:
    ///
    /// The exponential distribution has only 1 parameter, `lambda`.
    fn general_pdf(&self, x: f64, parameters: &[f64]) -> f64 {
        // pdf( x | lambda ) = lambda * exp(-lambda * x)
        let lambda: f64 = parameters[0];
        return lambda * (-lambda * x).exp();
    }

    fn number_of_parameters() -> u16 {
        1
    }

    fn get_parameters(&self, parameters: &mut [f64]) {
        parameters[0] = self.lambda;
    }

    fn derivative_pdf_parameters(&self, x: f64, parameters: &[f64]) -> Vec<f64> {
        // pdf:
        // pdf( x | lambda ) = lambda * exp(-lambda * x)

        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(2);

        let lambda: f64 = parameters[0];

        {
            //## Derivative respect to x:
            /*
               d/dx lambda * exp(-lambda * x) =
                = lambda * exp(-lambda * x) * -lambda
                = -lambda^2 * exp(-lambda * x)
                = pdf( x | lambda ) * -lambda

            */

            ret.push(-lambda * lambda * (-lambda * x).exp());
        }

        {
            // ## Derivative respect lambda:
            /*
               d/d_lambda  pdf( x | lambda ) = d/d_lambda  lambda * exp(-lambda * x)
                = lambda * -x*exp(-lambda * x) + exp(-lambda * x)
                = (lambda * -x + 1) * exp(-lambda * x)
                = (1 - x*lambda) * exp(-x * lambda)
               let aux = -x * lambda;
                = (1 + aux) * exp(aux)
            */

            let aux: f64 = -x * lambda;
            ret.push((1.0 + aux) * aux.exp());
        }

        return ret;
    }

    fn log_derivative_pdf_parameters(&self, x: f64, parameters: &[f64]) -> Vec<f64> {
        // pdf:
        // pdf( x | lambda ) = lambda * exp(-lambda * x)

        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(2);

        let lambda: f64 = parameters[0];

        {
            // ## Log derivative respect x:

            /*
               d/dx ln( lambda * exp(-lambda * x) ) =
               d/dx ln( lambda ) - lambda * x  =
                = 0 - lambda
                = -lambda
            */

            ret[0] = -lambda;
        }

        {
            // ## Log derivative respect lambda:
            /*
               d/d_lambda  ln(pdf( x | lambda )) = d/d_lambda  ln(lambda * exp(-lambda * x))
                = d/d_lambda  ln(lambda) + ln(exp(-lambda * x))
                = d/d_lambda  ln(lambda) - lambda * x
                =                 1/lambda   - x
                = 1/lambda - x

               Confirmation:
               d/dx ln(f(x)) = f'(x)/f(x) =>
               f(x) * d/dx ln(f(x)) = f'(x) =
                = lambda * exp(-lambda * x) * (1/lambda - x)
                = exp(-lambda * x) * (lambda * 1/lambda -lambda * x)
                = exp(-lambda * x) * (1 - lambda * x)
                = (1 - x * lambda) * exp(-x * lambda)

               Wich is identical for the espression we found in the derivation in the
               derivative_pdf_parameters function:
                `(1 - x*lambda) * exp(-x * lambda)`
            */

            ret[1] = 1.0 / lambda - x;
        }

        return ret;
    }

    fn parameter_restriction(&self, parameters: &mut [f64]) {
        // lambda must be stricly positive
        // epsilon cubed is arround `1.0947644252537633e-47` and should be enough for most puposes.
        let ep: f64 = f64::EPSILON;
        parameters[0] = parameters[0].max(ep * ep * ep);
    }

    fn fit(&self, data: &mut crate::samples::Samples) -> Vec<f64> {
        // pdf( x | lambda ) = lambda * exp(-lambda * x)

        /*
            If we want to maximize f(x), we should find f'(x) = 0.
            Assuming n samples:

            sumatory{x_i} d/d_lambda ln(P(x_i)) = sumatory{x_i} 1/lambda - x_i
            0 = sumatory{x_i} 1/lambda - x_i
              = n/lambda + sumatory{x_i} - x_i
              = -n/lambda + sumatory{x_i} x_i

            n/lambda = sumatory{x_i} x_i
            1/lambda = sumatory{x_i}[x_i] / n
            1/lambda = mean[x_i]
            lambda = 1/mean[x_i]

        */

        // Reserve vector for exacly 1 elements
        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(1);

        let estimator_lambda: f64 = 1.0 / data.mean().unwrap_or(1.0);
        ret.push(estimator_lambda);
        // returns lambda = 1.0 if no samples

        return ret;
    }
}

impl Iterator for ExponentialGenerator {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        let r: f64 = self.rng.random();
        return Some(-r.ln() * self.inv_lambda);
    }
}

impl Default for Exponential {
    fn default() -> Self {
        Exponential::new(1.0).unwrap()
    }
}
