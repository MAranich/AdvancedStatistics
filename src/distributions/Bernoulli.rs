//! # Bernoulli distribution
//!
//! The [Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution).
//!
//! It represents a value that can eiter take the value `1` with probability `p` or `0`
//! with probability `1 - p`. It is a special case of the [Binomial distribution] when
//! `n = 1`.
//!
//! It can be interpreted as a coin toss, where `p = 0.5` and the results `1` represents
//! heads and `0` represents tails. We can also have an unfair coin by changing the
//! paramenter `p` to another value. Note that `p` must belong to `[0, 1]`. If
//! we want a distribution to simulte the probability of getting a 6 on a fair dice,
//! we can do so by setting `p = 1/6`.
//!
//!

use rand::Rng;

use crate::{
    distribution_trait::{DiscreteDistribution, Parametric},
    domain::DiscreteDomain,
    errors::AdvStatError,
};

pub const BERNOULLI_DOMAIN: DiscreteDomain = DiscreteDomain::Range(0, 1);

/// Represnets a Bernoulli distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct Bernoulli {
    p: f64,
}

impl Bernoulli {
    /// Creates a new [bernulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution).
    ///
    ///  - `p` indicates the probability of success (returning `1.0`).
    ///     - `p` must belong in the interval `[0.0, 1.0]`. Otherwise an error will be returned.
    pub const fn new(p: f64) -> Result<Bernoulli, AdvStatError> {
        if !p.is_finite() {
            if p.is_nan() {
                return Err(AdvStatError::NanErr);
            } else if p.is_infinite() {
                return Err(AdvStatError::InvalidNumber);
            }
        }
        if !(0.0 <= p && p <= 1.0) {
            return Err(AdvStatError::InvalidNumber);
        }

        return Ok(Bernoulli { p });
    }

    /// Creates a new [bernulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution)
    /// without any checks.
    ///
    /// ## Safety
    ///
    /// If the following conditions are not fullfiled, the returned distribution
    /// will be invalid.
    ///
    ///  - `p` must be finite (no NaNs or `+-inf`)
    ///  - `p` must be a valid probability (`p` belongs to the interval `[0, 1]`)
    ///
    #[must_use]
    pub const unsafe fn new_unchecked(p: f64) -> Bernoulli {
        return Bernoulli { p };
    }

    /// Return `p` (probability of success).
    #[must_use]
    pub const fn get_p(&self) -> f64 {
        return self.p;
    }
}

impl DiscreteDistribution for Bernoulli {
    #[must_use]
    fn pmf(&self, x: f64) -> f64 {
        let mut ret: f64 = 0.0;
        if x == 1.0 {
            ret = self.p;
        } else if x == 0.0 {
            ret = 1.0 - self.p;
        }
        return ret;
    }

    #[must_use]
    fn get_domain(&self) -> &crate::domain::DiscreteDomain {
        return &BERNOULLI_DOMAIN;
    }

    #[must_use]
    fn cdf(&self, x: f64) -> f64 {
        assert!(
            !x.is_nan(),
            "Tried to evaluate the Bernoulli cdf with a NaN value. \n"
        );

        if x < 0.0 {
            return 0.0;
        }

        if 1.0 <= x {
            return 1.0;
        }

        return 1.0 - self.p;
    }

    #[must_use]
    fn sample(&self) -> f64 {
        let aux: Vec<f64> = self.sample_multiple(1);
        return aux[0];
    }

    #[must_use]
    fn quantile(&self, x: f64) -> f64 {
        assert!(
            !x.is_nan(),
            "Tried to evaluate the Bernoulli quantile function with a NaN value. \n"
        );

        if x <= 1.0 - self.p {
            return 0.0;
        }

        return 1.0;
    }

    #[must_use]
    fn cdf_multiple(&self, points: &[f64]) -> Vec<f64> {
        points.iter().map(|x| self.cdf(*x)).collect::<Vec<f64>>()
    }

    #[must_use]
    fn sample_multiple(&self, n: usize) -> Vec<f64> {
        let mut rng: rand::prelude::ThreadRng = rand::rng();
        let mut rand_quantiles: Vec<f64> = std::vec![0.0; n];
        rng.fill(rand_quantiles.as_mut_slice());

        return rand_quantiles
            .iter()
            .map(|&r| if self.p < r { 1.0 } else { 0.0 })
            .collect();
    }

    #[must_use]
    fn quantile_multiple(&self, points: &[f64]) -> Vec<f64> {
        return points
            .iter()
            .map(|&x| self.quantile(x))
            .collect::<Vec<f64>>();
    }

    #[must_use]
    fn expected_value(&self) -> Option<f64> {
        return Some(self.p);
    }

    #[must_use]
    fn variance(&self) -> Option<f64> {
        return Some(self.p * (1.0 - self.p));
    }

    #[must_use]
    fn mode(&self) -> f64 {
        if 0.5 <= self.p {
            return 1.0;
        }
        return 0.0;
    }

    #[must_use]
    fn skewness(&self) -> Option<f64> {
        let num: f64 = 1.0 - 2.0 * self.p;
        let den: f64 = (self.p * (1.0 - self.p)).sqrt();
        return Some(num / den);
    }

    #[must_use]
    fn kurtosis(&self) -> Option<f64> {
        let pq: f64 = self.p * (1.0 - self.p);
        return Some(3.0 + (1.0 - 6.0 * pq) / pq);
    }

    #[must_use]
    fn excess_kurtosis(&self) -> Option<f64> {
        let pq: f64 = self.p * (1.0 - self.p);
        return Some((1.0 - 6.0 * pq) / pq);
    }

    #[must_use]
    fn moments(&self, order: u8, mode: crate::euclid::Moments) -> f64 {
        let domain: &DiscreteDomain = self.get_domain();

        // The values of 0.0 and 1.0 have no special meaning. They are not going to be used anyway.
        let (mean, std_dev): (f64, f64) = match mode {
            crate::euclid::Moments::Raw => return self.expected_value().unwrap(),
            crate::euclid::Moments::Central => {
                let q: f64 = 1.0 - self.p;
                return q * (-self.p).powi(i32::from(order)) + self.p * q.powi(i32::from(order));
            }
            crate::euclid::Moments::Standarized => {
                (self.expected_value().unwrap(), self.variance().unwrap())
            }
        };

        // this is only for the standarized case and it is the deafult implementation.

        let order_exp: i32 = i32::from(order);
        let (minus_mean, inv_std_dev) = (-mean, 1.0 / std_dev.sqrt());

        let integration_fn = |x: f64| {
            let std_inp: f64 = (x + minus_mean) * inv_std_dev;
            std_inp.powi(order_exp) * self.pmf(x)
        };

        // SAFETY: should always be safe to only read
        let max_steps: u64 =
            unsafe { crate::configuration::disrete_distribution_deafults::MAXIMUM_STEPS };
        let max_steps_opt: Option<usize> = Some(max_steps.try_into().unwrap_or(usize::MAX));

        let moment: f64 =
            crate::euclid::discrete_integration(integration_fn, domain, max_steps_opt);

        return moment;
    }

    #[must_use]
    fn entropy(&self) -> f64 {
        let q: f64 = 1.0 - self.p;
        let entropy: f64 = -q * q.ln() - self.p * self.p.ln();
        return entropy;
    }
}

impl Parametric for Bernoulli {
    /// Evaluates the [Bernoulli::pmf] in a general way taking into account
    /// the parameter.
    ///
    /// > pmf(x | p) = {1-p if x == 0.0; p if x == 1.0; 0.0 otherwise}
    ///
    /// ### Parameters for [Bernoulli]:
    ///
    /// The [Bernoulli] distribution has 1 parameter: `p`
    /// (**p**robability of success (get 1))
    #[must_use]
    fn general_pdf(&self, x: f64, parameters: &[f64]) -> f64 {
        let mut ret: f64 = 0.0;
        let p: f64 = parameters[0];
        if x == 1.0 {
            ret = p;
        } else if x == 0.0 {
            ret = 1.0 - p;
        }
        return ret;
    }

    #[must_use]
    fn number_of_parameters() -> u16 {
        1
    }

    fn get_parameters(&self, parameters: &mut [f64]) {
        // assert!(self.number_of_parameters() <= parameters.len());
        parameters[0] = self.p;
    }

    #[must_use]
    fn derivative_pdf_parameters(&self, x: f64, _parameters: &[f64]) -> Vec<f64> {
        // d/dx ln(f(x)) = f'(x)/f(x)
        // => f(x) * d/dx ln(f(x)) = f'(x)

        // pmf(x | p) = {1-p if x == 0.0; p if x == 1.0; 0.0 otherwise}

        // Reserve a vector with exacly 3 elements
        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(3);

        {
            //## Derivative respect to x:
            /*
               d/dx pmf(x | p) = d/dx {1-p if x == 0.0; p if x == 1.0; 0.0 otherwise}

               The pmf of the bernoulli is not defined for values that are not 1.0 or 0.0,
               therefore the derivative of the pmf respect to x should be undefinied.

               We will return 0.0.

            */

            ret.push(0.0);
        }

        {
            //## Derivative respect to p:
            /*
               d/dp pmf(x | p) = d/dp {1-p if x == 0.0; p if x == 1.0; 0.0 otherwise}
                = {d/dp[1-p] if x == 0.0; d/dp[p] if x == 1.0; d/dp[0.0] otherwise}
                = {-1 if x == 0.0; 1 if x == 1.0; 0 otherwise}

            */

            let mut der: f64 = 0.0;
            if x == 1.0 {
                der = 1.0;
            } else if x == 0.0 {
                der = -1.0;
            }

            ret.push(der);
        }

        return ret;
    }

    #[must_use]
    fn log_derivative_pdf_parameters(&self, x: f64, parameters: &[f64]) -> Vec<f64> {
        // d/dx ln(f(x)) = f'(x)/f(x)

        // pmf(x | p) = {1-p if x == 0.0; p if x == 1.0; 0.0 otherwise}
        // ln(pmf(x | p)) = {ln(1-p) if x == 0.0; ln(p) if x == 1.0; ln(0.0) otherwise}

        // Reserve a vector with exacly 3 elements
        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(3);

        {
            //## Derivative respect to x:
            /*
               d/dx ln(pmf(x | p)) = d/dx {ln(1-p) if x == 0.0; ln(p) if x == 1.0; ln(0.0) otherwise}

               The pmf of the bernoulli is not defined for values that are not 1.0 or 0.0,
               therefore the derivative of the pmf respect to x should be undefinied.

               We will return 0.0 -> -inf.

            */

            ret.push(f64::NEG_INFINITY);
        }

        {
            //## Derivative respect to p:
            /*
               d/dp ln(pmf(x | p)) = d/dp {ln(1-p) if x == 0.0; ln(p) if x == 1.0; ln(0.0) otherwise}
                = {d/dp[ln(1-p)] if x == 0.0; d/dp[ln(p)] if x == 1.0; d/dp[ln(0.0)] otherwise}
                = {-1/(1-p) if x == 0.0; 1/p if x == 1.0; 0.0 otherwise}
                = {1/(p-1) if x == 0.0; 1/p if x == 1.0; 0.0 otherwise}

                We decided to let d/dp[ln(0.0)] = 0
            */

            let mut der: f64 = 0.0;
            let p: f64 = parameters[0];
            if (x - 1.0).abs() < 0.5 {
                der = 1.0 / p;
            } else if x == 0.0 {
                der = 1.0 / (p - 1.0);
            }

            ret.push(der);
        }

        return ret;
    }

    fn parameter_restriction(&self, parameters: &mut [f64]) {
        parameters[0] = parameters[0].clamp(0.0, 1.0);
    }

    #[must_use]
    fn fit(&self, data: &mut crate::samples::Samples) -> Vec<f64> {
        // Reserve a vector with exacly 1 elements
        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(1);

        // ln(pmf(x | p)) = {ln(1-p) if x == 0.0; ln(p) if x == 1.0; ln(0.0) otherwise}

        /*
                Maximum likelyhood estimation:

            Assuming `n` samples, wich `a` of them are 0, `b` are 1 and `c` are
            anything else. It is true that: `n = a + b + c`.
            Since all samples SHOULD come from the Bernoulli distribution `c` = 0
            and `n = a + b`

            ### For mean:

            0 = sumatory{x_i} d/d_mean ln(pdf(x_i | p))
            0 = sumatory{x_i} {1/(p-1) if x == 0.0; 1/p if x == 1.0; 0.0 otherwise}
            0 = a/(p-1) + b/p + c*0.0
            0 = a/(p-1) + b/p
            -a/(p-1) = b/p
            a/(1-p) = b/p
            a/b = (1-p)/p
            a/b = 1/p - 1
            a/b + 1 = 1/p
            (a + b)/b = 1/p
            b/(a + b) = p
            b/n = p


            ## Conclusion:

            To estimate the mean we will use:

            mean = mean[x_i]

            But for std we will use the **UNBIASED** formula instead of the obtained one.
            std = sqrt(1/(n-1) * sumatory{x_i} (x - mean)^2 )

            ## Deafult values:
            If there are not enough samples for the computation, the deafults are:
             - p: 0.5

        */

        let mut num_ones: u32 = 0;
        for observation in data.peek_data() {
            if *observation == 0.0 {
                num_ones += 1;
            }
        }

        ret.push(f64::from(num_ones) / (data.peek_data().len() as f64));

        return ret;
    }
}

impl Default for Bernoulli {
    fn default() -> Self {
        Self { p: 0.5 }
    }
}
