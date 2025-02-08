//! The [Binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution)
//! is the distribution that models the number of successes of `n`
//! [Bernoulli trials](crate::distributions::Bernoulli) with succes probability `p`.
//!
//! For example, if you whant to know the probability to get exacly 17 heads in 22
//! throws of a coin, we can model this as a binomial distribution with parameters
//! `n = 22` and `p = 0.5` and evaluate the pmf at `17` (adduming head = 1 and tail = 0).
//! To get the probability of getting 14 heads or less we can compute the cdf at `14`.
//!
//! The [Bernoulli distribution](crate::distributions::Bernoulli) is equivalent to a
//! Binomial distribution with `n = 1`

use crate::{distribution_trait::DiscreteDistribution, domain::DiscreteDomain, euclid};

use rand::Rng;

pub struct Binomial {
    domain: DiscreteDomain,
    p: f64,
    n: u64,
}

impl Binomial {
    /// Creates a new [Binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution).
    ///
    ///  - `p` indicates the probability of success (returning `1.0`) of each Bernoulli trial.
    ///      - `p` must belong in the interval `[0.0, 1.0]`. Otherwise an error will be returned.
    ///  - `n` indicates the number of trials
    ///
    pub fn new(p: f64, n: u64) -> Result<Binomial, ()> {
        if p.is_infinite() || p.is_nan() {
            return Err(());
        }
        if !(0.0 <= p && p <= 1.0) {
            return Err(());
        }

        let domain: DiscreteDomain = DiscreteDomain::Range(0, n.try_into().unwrap_or(i64::MAX));

        return Ok(Binomial { domain, p, n });
    }

    /// Creates a new [Binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution).
    ///
    ///   - `p` indicates the probability of success (returning `1.0`) of each Bernoulli trial.
    ///      - `p` must belong in the interval `[0.0, 1.0]`. Otherwise the structure will be invalid.
    ///  - `n` indicates the number of trials
    pub unsafe fn new_unchecked(p: f64, n: u64) -> Binomial {
        let domain: DiscreteDomain = DiscreteDomain::Range(0, n.try_into().unwrap_or(i64::MAX));

        return Binomial { domain, p, n };
    }

    /// Return `p` (probability of success).
    pub fn get_p(&self) -> f64 {
        return self.p.clone();
    }

    /// Return `n` (number of trials).
    pub fn get_n(&self) -> u64 {
        return self.n.clone();
    }
}

impl DiscreteDistribution for Binomial {
    fn pmf(&self, x: f64) -> f64 {
        // Todo: update this basic 1:1 implementation for fomething better
        // `binomial_coef` can be very big and `prob_p * prob_q` can be very small
        // causing problems in their computations that could be avoided if
        // their computations were joined

        let X: u64 = x.floor() as u64;
        let binomial_coef: u128 = euclid::combinatorics::binomial_coeffitient(self.n, X).expect("The parameters of the binomial are too big. Our current implementation is not good enough. ");

        let prob_p: f64 = self.p.powi(X as i32);
        let prob_q: f64 = (1.0 - self.p).powi((self.n - X) as i32);

        return (binomial_coef as f64) * prob_p * prob_q;
    }

    fn get_domain(&self) -> &DiscreteDomain {
        return &self.domain;
    }

    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            std::panic!("Tried to evaluate the cdf with a NaN value. \n");
        }

        let aux: [f64; 1] = [x];
        let aux_2: Vec<f64> = self.cdf_multiple(&aux);
        return aux_2[0];
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
        // todo: This could be improved by aproximating the distribution to a normal when
        // n is large

        /*
               Special plan for Binomial.

           Binomial is symetric and we can use that to our advantage to reduce
           computational costs.
        */

        if points.is_empty() {
            return Vec::new();
        }

        // panic if NAN is found
        for point in points {
            if point.is_nan() {
                std::panic!("Found NaN in `cdf_multiple`. \n");
            }
        }

        let mut ret: Vec<f64> = std::vec![0.0; points.len()];
        let domain: &DiscreteDomain = self.get_domain();
        let bounds: (f64, f64) = domain.get_bounds();

        let mut sorted_indicies: Vec<usize> = (0..points.len()).into_iter().collect::<Vec<usize>>();
        // the bool determines if it is flipped or not

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

        while current_cdf_point <= bounds.0 {
            ret[current_index] = 0.0;
            match idx_iter.next() {
                Some(v) => current_index = v,
                None => return ret,
            }
            current_cdf_point = points[current_index];
        }

        for x in domain.iter() {
            while current_cdf_point <= x {
                ret[current_index] = accumulator;
                match idx_iter.next() {
                    Some(v) => current_index = v,
                    None => return ret,
                }
                current_cdf_point = points[current_index];
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

    fn sample_multiple(&self, n: usize) -> Vec<f64> {
        let mut rng: rand::prelude::ThreadRng = rand::thread_rng();

        let mut ret: Vec<f64> = Vec::with_capacity(n);
        let mut aux: Vec<f64> = Vec::with_capacity(self.n as usize);

        for _ in 0..n {
            rng.fill(aux.as_mut_slice());
            let mut count: f64 = 0.0;

            for r in &aux {
                // Todo: when stabilized, change implementation:
                // count = std::intrinsics::select_unpredictable(self.p < *r, count + 1.0, count);

                if self.p < *r {
                    count += 1.0;
                }

                /*  Equivalent to:
                if self.p < r {
                    count += 1.0;
                }
                 */
            }

            ret.push(count);
        }

        return ret;
    }

    fn quantile_multiple(&self, points: &[f64]) -> Vec<f64> {
        // todo: This could be improved by aproximating the distribution to a normal when
        // n is large

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

        let mut current_quantile_point: f64 = points[current_index];

        let mut accumulator: f64 = 0.0;

        while current_quantile_point <= 0.0 {
            ret[current_index] = bounds.0;
            match idx_iter.next() {
                Some(v) => current_index = v,
                None => return ret,
            }
            current_quantile_point = points[current_index];
        }

        for x in domain.iter() {
            while current_quantile_point <= accumulator {
                ret[current_index] = x;
                match idx_iter.next() {
                    Some(v) => current_index = v,
                    None => return ret,
                }
                current_quantile_point = points[current_index];
            }

            accumulator += self.pmf(x);
        }

        ret[current_index] = bounds.1;
        for idx in idx_iter {
            ret[idx] = bounds.1;
        }

        return ret;
    }

    fn expected_value(&self) -> Option<f64> {
        return Some(self.n as f64 * self.p);
    }

    fn variance(&self) -> Option<f64> {
        return Some(self.n as f64 * self.p * (1.0 - self.p));
    }

    fn mode(&self) -> f64 {
        let floor: f64 = (self.n as f64 * self.p).floor();
        let ceil: f64 = (self.n as f64 * self.p).ceil();

        if self.pmf(floor) < self.pmf(ceil) {
            return floor;
        } else {
            return ceil;
        }
    }

    fn median(&self) -> f64 {
        return (self.n as f64 * self.p).floor() + 0.5;
    }

    fn skewness(&self) -> Option<f64> {
        // = (q - p) / sqrt(p*q*n) = ((1-p) - p) / sqrt(p*(1-p)*n)
        // = (1 - 2*p) / sqrt(p*(1-p)*n)

        let numerator: f64 = 1.0 - 2.0 * self.p;
        let denomiantor: f64 = (self.p * (1.0 - self.p) * (self.n as f64)).sqrt();
        return Some(numerator / denomiantor);
    }

    fn kurtosis(&self) -> Option<f64> {
        let numerator: f64 = 1.0 - 6.0 * self.p * (1.0 - self.p);
        let denomiantor: f64 = self.p * (1.0 - self.p) * (self.n as f64);
        return Some(3.0 + numerator / denomiantor);
    }

    fn excess_kurtosis(&self) -> Option<f64> {
        let numerator: f64 = 1.0 - 6.0 * self.p * (1.0 - self.p);
        let denomiantor: f64 = self.p * (1.0 - self.p) * (self.n as f64);
        return Some(numerator / denomiantor);
    }

    fn moments(&self, order: u8, mode: euclid::Moments) -> f64 {
        // This moments function can be improved:
        // https://en.wikipedia.org/wiki/Binomial_distribution#Higher_moments

        let domain: &DiscreteDomain = self.get_domain();

        // The values of 0.0 and 1.0 have no special meaning. They are not going to be used anyway.
        let (mean, std_dev): (f64, f64) = match mode {
            euclid::Moments::Raw => (0.0, 1.0),
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

        let integration_fn = |x: f64| {
            let std_inp: f64 = (x + minus_mean) * inv_std_dev;
            std_inp.powi(order_exp) * self.pmf(x)
        };

        let max_steps: u64 = unsafe { crate::configuration::disrete_distribution_deafults::MAXIMUM_STEPS };
        let max_steps_opt: Option<usize> = Some(max_steps.try_into().unwrap_or(usize::MAX));

        let moment: f64 = euclid::discrete_integration(integration_fn, domain, max_steps_opt);

        return moment;
    }

    // The results don't seem to agree. The O(1/n) term must be too big to neglect.
    /*
    fn entropy(&self) -> f64 {
        // 2*pi*e*p*q*n
        let input_log: f64 = 2.0 * PI * E * self.p * (1.0 - self.p) * (self.n as f64);
        let log: f64 = input_log.log2();

        return 0.5 * log; // + O(1/n)
    }
    */
}
