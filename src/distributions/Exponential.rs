
use rand::Rng;

use crate::euclid::Domain;

use super::distribution_trait::Distribution;

/// An [Exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution).
#[derive(Debug, Clone, PartialEq)]
pub struct Exponential {
    lambda: f64,
    domain: Domain,
}

impl Exponential {
    /// Creates a new [Exponential] distribution. it is requiered that `0.0 < lambda` or an
    /// error will be returned.
    pub fn new(_lambda: f64) -> Result<Exponential, ()> {
        if _lambda <= 0.0 {
            return Err(());
        }

        // for performance reasons, the domain will only include up to the quantile 99.99%
        let max: f64 = -((1.0_f64 - 0.9999).ln()) / _lambda;

        let _domain: Domain = Domain::new_continuous_range(0.0, max);
        return Ok(Exponential {
            lambda: _lambda,
            domain: _domain,
        });
    }

    pub fn get_lambda(&self) -> f64 {
        return self.lambda;
    }
}

impl Distribution for Exponential {
    fn pdf(&self, x: f64) -> f64 {
        self.lambda * (-self.lambda * x).exp()
    }

    fn get_domain(&self) -> &crate::euclid::Domain {
        &self.domain
    }

    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            // x is not valid
            panic!("Found NaN while attempting to compute the cdf of an Exponential. ");
        }
        if x <= 0.0 {
            return 0.0;
        }
        return 1.0 - (-self.lambda * x).exp();
    }

    fn sample(&self) -> f64 {
        let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
        let r: f64 = rng.gen();
        return -r.ln() / self.lambda;
    }

    fn quantile(&self, x: f64) -> f64 {
        // just call [Distribution::quantile_multiple]

        if x.is_nan() {
            // x is not valid
            panic!("Tried to evaluate the quantile function with a NaN value. \n"); 
        }

        if x <= 0.0 {
            return 0.0;
        }

        if 1.0 <= x {
            return f64::INFINITY;
        }

        return -(1.0 - x).ln() / self.lambda;
    }

    fn cdf_multiple(&self, points: &[f64]) -> Vec<f64> {
        points.iter().map(|x| self.cdf(*x)).collect::<Vec<f64>>()
    }

    fn sample_multiple(&self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.sample()).collect::<Vec<f64>>()
    }

    fn quantile_multiple(&self, points: &[f64]) -> Vec<f64> {
        let list: Vec<f64> = points
            .iter()
            .map(|x| self.quantile(*x))
            .collect::<Vec<f64>>();
        return list; 
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

    fn skewness(&self) -> Option<f64> {
        return Some(2.0);
    }

    fn kurtosis(&self) -> Option<f64> {
        return Some(9.0);
    }

    fn excess_kurtosis(&self) -> Option<f64> {
        return self.kurtosis().map(|x| x - 3.0);
    }

    fn entropy(&self) -> f64 {
        return 1.0 - self.lambda.ln();
    }
}
