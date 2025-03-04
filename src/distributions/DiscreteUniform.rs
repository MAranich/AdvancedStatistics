//! # Discrete uniform
//! 
//! The [discrete uniform distribution](https://en.wikipedia.org/wiki/Discrete_uniform_distribution)
//! is a discrete distribution where all possible outcomes are integers in the
//! interval `[a, b]` have equal chance of happening.
//!
//! Note that this distribution is different from the
//! [Continuous uniform distribution](crate::distributions::Uniform).
//!

use rand::Rng;

use crate::{
    distribution_trait::DiscreteDistribution,
    domain::DiscreteDomain,
};

#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteUniform {
    domain: DiscreteDomain,
    /// The minimum value
    a: i64,
    /// The maximum value
    b: i64,
}

impl DiscreteUniform {
    /// Creates a new [DiscreteUniform] with parameters `a` (minimum)
    /// and `b` (maximum). Both `a` and `b` are inclusive.
    ///
    /// Returns [Err] if `b < a`.
    pub fn new(a: i64, b: i64) -> Result<DiscreteUniform, ()> {
        if b < a {
            return Err(());
        }
        let new_domain: DiscreteDomain = DiscreteDomain::Range(a, b);

        return Ok(DiscreteUniform {
            domain: new_domain,
            a,
            b,
        });
    }

    /// Creates a new [DiscreteUniform] with parameters `a` (minimum)
    /// and `b` (maximum). Both `a` and `b` are inclusive.
    ///
    /// Does not check if `b < a`.
    pub unsafe fn new_unchecked(a: i64, b: i64) -> DiscreteUniform {
        let new_domain: DiscreteDomain = DiscreteDomain::Range(a, b);

        return DiscreteUniform {
            domain: new_domain,
            a,
            b,
        };
    }

    /// Return `a` (minimum value).
    pub fn get_a(&self) -> i64 {
        return self.a;
    }

    /// Return `b` (maximum value).
    pub fn get_b(&self) -> i64 {
        return self.b;
    }
}

impl DiscreteDistribution for DiscreteUniform {
    fn pmf(&self, _x: f64) -> f64 {
        return 1.0 / (self.b as f64 - self.a as f64 + 1.0);
    }

    fn get_domain(&self) -> &DiscreteDomain {
        &self.domain
    }

    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            std::panic!("Tried to evaluate the cdf with a NaN value. \n");
        }

        let aux: [f64; 1] = [x];
        let aux_2: Vec<f64> = self.cdf_multiple(&aux);
        return aux_2[0];
    }

    fn sample(&self) -> f64 {
        let aux: Vec<f64> = self.sample_multiple(1);
        return aux[0];
    }

    fn quantile(&self, x: f64) -> f64 {
        if x.is_nan() {
            // x is not valid
            std::panic!("Tried to evaluate the quantile function with a NaN value. \n");
        }

        let value: [f64; 1] = [x];
        let quantile_vec: Vec<f64> = self.quantile_multiple(&value);
        return quantile_vec[0];
    }

    fn cdf_multiple(&self, points: &[f64]) -> Vec<f64> {
        if points.is_empty() {
            return Vec::new();
        }

        // panic if NAN is found
        for point in points {
            if point.is_nan() {
                std::panic!("Found NaN in `cdf_multiple` for DiscreteUniform. \n");
            }
        }

        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(points.len());

        let a_f: f64 = self.a as f64;
        let b_f: f64 = self.b as f64;
        let n_f: f64 = b_f - a_f + 1.0;

        for &p in points {
            if p < a_f {
                ret.push(0.0);
                continue;
            }

            if b_f <= p {
                ret.push(1.0);
                continue;
            }

            let cdf: f64 = (p.floor() - a_f + 1.0) / n_f;
            ret.push(cdf);
        }

        return ret;
    }

    fn sample_multiple(&self, n: usize) -> Vec<f64> {
        let mut rng: rand::prelude::ThreadRng = rand::rng();

        let ret: Vec<f64> = (0..n)
            .into_iter()
            .map(|_| rng.random_range(self.a..=self.b))
            .map(|q| q as f64)
            .collect::<Vec<f64>>();

        return ret;
    }

    fn quantile_multiple(&self, points: &[f64]) -> Vec<f64> {
        if points.is_empty() {
            return Vec::new();
        }

        // panic if NAN is found
        for point in points {
            if point.is_nan() {
                std::panic!("Found NaN in `quantile_multiple` for DiscreteUniform. \n");
            }
        }

        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(points.len());

        let a_f: f64 = self.a as f64;
        let b_f: f64 = self.b as f64;
        let n_f: f64 = b_f - a_f + 1.0;

        /*
           The cdf is:
           cdf(k) = (k.floor() - a + 1) / n
           Therefore, inverting the function:
           q = (k.floor() - a + 1) / n
           q*n = k.floor() - a + 1
           q*n - 1 + a = k.floor()

           Since q is not guaranteed to perfecly cancel everything
           and return an int, we will floor the left hand side.
        */

        for &q in points {
            if q < 0.0 {
                ret.push(a_f);
                continue;
            }

            if 1.0 <= q {
                ret.push(b_f);
                continue;
            }

            let quantile: f64 = (q * n_f + a_f - 1.0).floor();
            ret.push(quantile);
        }

        return ret;
    }

    fn expected_value(&self) -> Option<f64> {
        return Some((self.a + self.b) as f64 / 2.0);
    }

    fn variance(&self) -> Option<f64> {
        let n: f64 = (self.b - self.a + 1) as f64;
        return Some((n * n - 1.0) / 12.0);
    }

    ///Returns the mode of the distribution. It represents the most likely outcome.
    ///
    /// For the case of [DiscreteUniform], it will always return `self.a`
    /// (the minimum value), but take into account that every value in it's domain
    /// is the mode.
    fn mode(&self) -> f64 {
        return self.a as f64;
    }

    fn median(&self) -> f64 {
        return (self.a + self.b) as f64 / 2.0;
    }

    fn skewness(&self) -> Option<f64> {
        return Some(0.0);
    }

    fn kurtosis(&self) -> Option<f64> {
        return self.excess_kurtosis().map(|x| x + 3.0);
    }

    fn excess_kurtosis(&self) -> Option<f64> {
        let n: f64 = (self.b - self.a + 1) as f64;
        let num: f64 = -6.0 * (n * n + 1.0);
        let den: f64 = 5.0 * (n * n - 1.0);

        return Some(num / den);
    }

    fn moments(&self, order: u8, mode: crate::euclid::Moments) -> f64 {
        let domain: &DiscreteDomain = self.get_domain();

        // The values of 0.0 and 1.0 have no special meaning. They are not going to be used anyway.
        let (mean, std_dev): (f64, f64) = match mode {
            crate::euclid::Moments::Raw => (0.0, 1.0),
            crate::euclid::Moments::Central => (
                self.expected_value()
                    .expect("Tried to compute a central moment but the expected value is undefined. "),
                1.0,
            ),
            crate::euclid::Moments::Standarized => (
                self.expected_value()
                    .expect("Tried to compute a central/standarized moment but the Expected value is undefined. "),
                self.variance().expect("Tried to compute a standarized moment but the variance is undefined. "),
            ),
        };

        // Todo: give better error handling to the above. ^
        // println!("(mean, std_dev): {:?}", (mean, std_dev));

        let order_exp: i32 = order as i32;
        let (minus_mean, inv_std_dev) = (-mean, 1.0 / std_dev.sqrt());

        let integration_fn = |x: f64| {
            let std_inp: f64 = (x + minus_mean) * inv_std_dev;
            std_inp.powi(order_exp) * self.pmf(x)
        };

        let max_steps: u64 =
            unsafe { crate::configuration::disrete_distribution_deafults::MAXIMUM_STEPS };
        let max_steps_opt: Option<usize> = Some(max_steps.try_into().unwrap_or(usize::MAX));

        let moment: f64 =
            crate::euclid::discrete_integration(integration_fn, domain, max_steps_opt);

        return moment;
    }

    fn entropy(&self) -> f64 {
        let n: f64 = (self.b - self.a + 1) as f64;
        return n.ln(); 

    }
}

impl Default for DiscreteUniform {
    fn default() -> Self {
        DiscreteUniform::new(0, 9).unwrap()
    }
}
