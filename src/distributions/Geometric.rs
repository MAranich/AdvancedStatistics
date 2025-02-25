//! # Geometric distribution
//!
//! The [geometric distribution](https://en.wikipedia.org/wiki/Geometric_distribution)
//! is a discrete distribution that represents the number of
//! [Bernoulli](crate::distributions::Bernoulli::Bernoulli) trials needed
//! to get a success.
//!
//! This distribution *should not be confused* with the **shifted exponential**.
//!

use rand::Rng;

use crate::{
    distribution_trait::{DiscreteDistribution, Parametric},
    domain::DiscreteDomain,
};

pub const GEOMETRIC_DOMAIN: DiscreteDomain = DiscreteDomain::From(1);

pub struct Geometric {
    p: f64,
}

impl Geometric {
    /// Creates a new [Geometric] distribution.
    ///
    ///  - `p` indicates the probability of success.
    ///  - `p` must belong in the interval `[0.0, 1.0]`.
    ///      - Otherwise an error will be returned.
    pub fn new(p: f64) -> Result<Geometric, ()> {
        if p.is_infinite() || p.is_nan() {
            return Err(());
        }
        if !(0.0 <= p && p <= 1.0) {
            return Err(());
        }

        return Ok(Geometric { p });
    }

    /// Creates a new [Geometric] distribution without checking if `p` is valid.
    ///
    /// In order to generate a valid Geometric, `p` must fullfill:
    ///  - `p` indicates the probability of success.
    ///  - `p` must belong in the interval `[0.0, 1.0]`.
    pub unsafe fn new_unchecked(p: f64) -> Geometric {
        return Geometric { p };
    }

    /// Return `p` (probability of success).
    pub fn get_p(&self) -> f64 {
        return self.p;
    }
}

impl DiscreteDistribution for Geometric {
    fn pmf(&self, x: f64) -> f64 {
        // pmf(x | p) = (1 - p)^(x.floor() - 1) * p
        let k: i32 = x as i32 - 1;
        let q: f64 = 1.0 - self.p;
        return self.p * q.powi(k);
    }

    fn get_domain(&self) -> &DiscreteDomain {
        return &GEOMETRIC_DOMAIN;
    }

    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            std::panic!("Tried to evaluate the cdf of Geometric with a NaN value. \n");
        }
        // 1 - (1 - p)^x.floor()
        return 1.0 - (1.0 - self.p).powi(x as i32);
    }

    fn quantile(&self, x: f64) -> f64 {
        if x.is_nan() {
            // x is not valid
            std::panic!(
                "Tried to evaluate the quantile function of Geometric with a NaN value. \n"
            );
        }

        let value: [f64; 1] = [x];
        let quantile_vec: Vec<f64> = self.quantile_multiple(&value);
        return quantile_vec[0];
    }

    fn cdf_multiple(&self, points: &[f64]) -> Vec<f64> {
        points.iter().map(|x| self.cdf(*x)).collect::<Vec<f64>>()
    }

    // use deafult sample_multiple. Although we could easly simulate it by
    // simulating Bernoulli trials until we succeed, it is too much work
    // to avould computing a `ln()`

    fn quantile_multiple(&self, points: &[f64]) -> Vec<f64> {
        if points.is_empty() {
            return Vec::new();
        }

        // panic if NAN is found
        for point in points {
            if point.is_nan() {
                std::panic!("Found NaN in `quantile_multiple` of Geometric. \n");
            }
        }

        /*
            cdf(x) = 1 - (1 - p)^x.floor()

            q = 1 - (1 - p)^x.floor()
            1 - q = (1 - p)^x.floor()
            ln(1 - q) = ln((1 - p)^x.floor())
            ln(1 - q) = x.floor() * ln(1 - p)
            ln(1 - q) / ln(1 - p) = x.floor()

            We will floor the result of the LHS
        */

        // reserve exacly the elements needed
        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(points.len());

        // precompute: `1 / ln(1 - p)`
        let term: f64 = 1.0 / (1.0 - self.p).ln();

        points
            .iter()
            .map(|q: &f64| q.clamp(0.0, 1.0))
            .map(|q: f64| ((1.0 - q).ln() * term).floor())
            .for_each(|q| ret.push(q));

        return ret;
    }

    fn expected_value(&self) -> Option<f64> {
        return Some(1.0 / self.p);
    }

    fn variance(&self) -> Option<f64> {
        return Some((1.0 * self.p) / (self.p * self.p));
    }

    fn mode(&self) -> f64 {
        return 1.0;
    }

    /// The [median](https://en.wikipedia.org/wiki/Median) of the distribution. If
    /// you sample a distribution, the median represnts the value that will be
    /// greater than 50% of your samples and also smaller than the other 50%.
    ///
    /// It may happen that the quantile distribution is hard to evaluate but that
    /// the median has a closed form solution. Otherwise, it will be equivalent to
    /// evaluating the [DiscreteDistribution::quantile] function at `0.5`.
    ///
    /// ### Geometric:
    ///
    /// The median is not unique if `-1/log_2(1-p)` is an integer.
    fn median(&self) -> f64 {
        return (-1.0 / (1.0 - self.p).log2()).floor();
    }

    fn skewness(&self) -> Option<f64> {
        return Some((2.0 - self.p) / (1.0 - self.p).sqrt());
    }

    fn kurtosis(&self) -> Option<f64> {
        return self.excess_kurtosis().map(|x| x + 3.0);
    }

    fn excess_kurtosis(&self) -> Option<f64> {
        return Some(6.0 + self.p * self.p / (1.0 - self.p));
    }

    fn entropy(&self) -> f64 {
        let q: f64 = 1.0 - self.p;

        let num = -q * q.ln() - self.p * self.p.ln();
        return num / self.p;
    }
}

impl Parametric for Geometric {
    /// Evaluates the [Geometric::pmf] in a general way taking into account
    /// the parameters.
    ///
    /// ### Parameters for Geometric:
    ///
    /// The [Geometric] distribution has 1 parameter: `p` (**p**robability of success).
    ///
    fn general_pdf(&self, x: f64, parameters: &[f64]) -> f64 {
        // pmf(x | p) = (1 - p)^(x.floor() - 1) * p
        let k: i32 = x as i32 - 1;
        let q: f64 = 1.0 - parameters[0];
        return self.p * q.powi(k);
    }

    fn number_of_parameters() -> u16 {
        return 1;
    }

    fn get_parameters(&self, parameters: &mut [f64]) {
        parameters[0] = self.p;
    }

    fn derivative_pdf_parameters(&self, x: f64, parameters: &[f64]) -> Vec<f64> {
        // d/dx ln(f(x)) = f'(x)/f(x)
        // => f(x) * d/dx ln(f(x)) = f'(x)

        // pmf(x | p) = (1 - p)^(x.floor() - 1) * p

        // Reserve a vector with exacly 2 elements
        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(2);

        /*
            The derivative respect to x will always be 0 since it's a pmf.
            The derivative is not defined for discrete functions.
        */

        ret.push(0.0);

        {
            /*
                    Derivative of pmf(x | p) respect to p:

                pmf(x | p) = (1 - p)^(x.floor() - 1) * p
                d/dp pmf(x | p) = d/dp (1 - p)^(x.floor() - 1) * p
                 = d/dp[ (1 - p)^(x.floor() - 1) ] * p + (1 - p)^(x.floor() - 1) * d/dp [ p ]
                 = (x.floor() - 1) * (1 - p)^(x.floor() - 2) * d/dp[ 1 - p ] * p + (1 - p)^(x.floor() - 1) * 1
                 = (x.floor() - 1) * (1 - p)^(x.floor() - 2) * -1 * p + (1 - p)^(x.floor() - 1)
                 = -(x.floor() - 1) * (1 - p)^(x.floor() - 2) * p + (1 - p)^(x.floor() - 2) * (1 - p)
                 = (1 - p)^(x.floor() - 2) * ( -(x.floor() - 1) * p + 1 - p )


                ### Check:

                d/dx ln(f(x)) = f'(x)/f(x)
                => f(x) * d/dx ln(f(x)) = f'(x)

                d/dp ln(pmf(x | p)) = -(x.floor() - 1)/(1 - p) + 1/p

                (1 - p)^(x.floor() - 1) * p * (-(x.floor() - 1)/(1 - p) + 1/p)
                 = ((1 - p)^(x.floor() - 1) * p * -(x.floor() - 1)/(1 - p) + (1 - p)^(x.floor() - 1) * p * 1/p)
                 = ((1 - p)^(x.floor() - 2) * p * -(x.floor() - 1) + (1 - p)^(x.floor() - 1) )
                 = ((1 - p)^(x.floor() - 2) * p * -(x.floor() - 1) + (1 - p)^(x.floor() - 2) * (1 - p) )
                 = (1 - p)^(x.floor() - 2) * (p * -(x.floor() - 1) + 1 - p )
                 = (1 - p)^(x.floor() - 2) * ( -(x.floor() - 1) * p + 1 - p )

                Wich is identical to 1 of the results we obtained for the normal
                derivative of pmf.

                We will implement:
                > (1 - p)^(x.floor() - 2) * ( -(x.floor() - 1) * p + 1 - p )
            */

            let x_floor: f64 = x.floor();
            let p: f64 = parameters[0];
            // (1 - p)^(x.floor() - 2)
            let term_1: f64 = (1.0 - p).powi(x_floor as i32 - 2);
            // ( -(x.floor() - 1) * p + 1 - p )
            let term_2: f64 = -(x_floor - 1.0) * p + 1.0 - p;

            ret.push(term_1 * term_2);
        }

        return ret;
    }

    fn log_derivative_pdf_parameters(&self, x: f64, parameters: &[f64]) -> Vec<f64> {
        // d/dx ln(f(x)) = f'(x)/f(x)

        // pmf(x | p) = (1 - p)^(x.floor() - 1) * p
        // ln(pmf(x | p)) = ln((1 - p)^(x.floor() - 1) * p)
        // ln(pmf(x | p)) = ln((1 - p)^(x.floor() - 1)) + ln(p)
        // ln(pmf(x | p)) = (x.floor() - 1) * ln(1 - p) + ln(p)

        // Reserve a vector with exacly 3 elements
        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(2);

        /*
            The derivative respect to x will always be 0 since it's a pmf.
            The derivative is not defined for discrete functions.
        */

        ret.push(0.0);

        {
            /*
                   Derivative of ln(pmf(x | p)) respect to p:

               ln(pmf(x | p)) = (x.floor() - 1) * ln(1 - p) + ln(p)
               d/dp ln(pmf(x | p)) = d/dp (x.floor() - 1) * ln(1 - p) + ln(p)
                = d/dp (x.floor() - 1) * ln(1 - p) + ln(p)
                = d/dp[ (x.floor() - 1) * ln(1 - p) ] + d/dp[ ln(p) ]
                = (x.floor() - 1) * d/dp[ ln(1 - p) ] + 1/p
                = (x.floor() - 1) * 1/(1 - p) * d/dp[ 1 - p ] + 1/p
                = (x.floor() - 1)/(1 - p) * -1 + 1/p
               d/dp ln(pmf(x | p)) = -(x.floor() - 1)/(1 - p) + 1/p

            */

            ret.push(-(x.floor() - 1.0) / (1.0 - parameters[0]) + 1.0 / parameters[0]);
        }

        return ret;
    }

    fn parameter_restriction(&self, parameters: &mut [f64]) {
        let ep: f64 = f64::EPSILON;
        parameters[0] = parameters[0].max(ep * ep * ep)
    }

    fn fit(&self, data: &mut crate::Samples::Samples) -> Vec<f64> {
        let mut parameters: Vec<f64> = Vec::new();
        parameters.reserve_exact(1);

        /*
                Estimation of p:

            Using Maximum Likelyhood estimation:
            Assuming k samples.

            pmf(x | p) = (1 - p)^x.floor() * p
            d/dp ln(pmf(x | p)) = -(x.floor() - 1)/(1 - p) + 1/p
            0 = sumatory{x_i} -(x_i.floor() - 1)/(1 - p) + 1/p

            Since we assume all samples come from [Geometric],
            we will drop the `.floor()` since it's redundant computation
            given our assumptions.

            0 = sumatory{x_i} -(x_i - 1)/(1 - p) + 1/p
            0 = k/p + sumatory{x_i} -(x_i - 1)/(1 - p)
            0 = k/p + -1/(1 - p) * sumatory{x_i}[ x_i - 1 ]
            -k/p = -1/(1 - p) * sumatory{x_i}[ x_i - 1 ]
            k/p = 1/(1 - p) * sumatory{x_i}[ x_i - 1 ]
            k*(1 - p)/p = sumatory{x_i}[ x_i - 1 ]
            k*(1 - p)/p = -k + sumatory{x_i}[ x_i ]
            (1 - p)/p = -1 + 1/k * sumatory{x_i}[ x_i ]
            (1 - p)/p = -1 + mean{x_i}
            1/p - 1 = -1 + mean{x_i}
            1/p = mean{x_i}
            p = 1/mean{x_i}

            However, it can be proven that this estimatior is biased with a bias:

            b = p * (1 - p) / n
            p = p_mle - b

            p_mle = 1/mean{x_i}

            As we cannot compute b because it depends on the real unknown p, we
            will estimate it with:

            est_b = p_mle * (1 - p_mle) / n
            p = p_mle - est_b
            p = p_mle - p_mle * (1 - p_mle) / n

            ### Deafult values:

            If there are not enough samples to compute the estimators, then
            this method will return the following deafult probabilities:
             - `p`: 0.5

        */

        let mean: f64 = match data.mean() {
            Some(m) => m,
            None => {
                // early return deafult probability of 0.5
                parameters.push(0.5);
                return parameters;
            }
        };

        let p_mle: f64 = 1.0 / mean;

        let p: f64 = p_mle - p_mle * (1.0 - p_mle) / (data.peek_data().len() as f64);
        parameters.push(p);

        return parameters;
    }
}
