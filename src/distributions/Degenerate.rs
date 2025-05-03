//! # Degenerate distribution
//!
//! The degenerate distribution is a distribution that has probability of `1.0` for
//! the value `0.0` and `0.0` everywhere else. For this reason it always returns the
//! same value, `0.0`.
//!
//! If you want the continuous version of this distribution, use
//! [Dirac](crate::distributions::Dirac).
//!
//! If you want to move this distribution so it outputs another value that is not `0.0`,
//! use [Location](crate::Loation).
//!

use crate::{distribution_trait::DiscreteDistribution, domain::DiscreteDomain};

pub const DEGENERATE_DISTRIBUTION_DOMAIN: DiscreteDomain = DiscreteDomain::Range(0, 0);

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Degenerate {}

impl Degenerate {
    #[must_use]
    pub const fn new() -> Degenerate {
        return Degenerate {};
    }
}

impl DiscreteDistribution for Degenerate {
    #[must_use]
    fn pmf(&self, x: f64) -> f64 {
        return if x == 0.0 { 1.0 } else { 0.0 };
    }

    #[must_use]
    fn get_domain(&self) -> &DiscreteDomain {
        return &DEGENERATE_DISTRIBUTION_DOMAIN;
    }

    #[must_use]
    fn cdf(&self, x: f64) -> f64 {
        assert!(
            !x.is_nan(),
            "Tried to evaluate the cdf of Degenerate with a NaN value. \n"
        );

        return (x.signum() + 1.0) * 0.5;
    }

    #[must_use]
    fn sample(&self) -> f64 {
        return 0.0;
    }

    #[must_use]
    fn quantile(&self, x: f64) -> f64 {
        assert!(
            !x.is_nan(),
            "Tried to evaluate the quantile function of Degenerate with a NaN value. \n"
        );

        return 0.0;
    }

    #[must_use]
    fn cdf_multiple(&self, points: &[f64]) -> Vec<f64> {
        points.iter().map(|x| self.cdf(*x)).collect::<Vec<f64>>()
    }

    #[must_use]
    fn sample_multiple(&self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.sample()).collect::<Vec<f64>>()
    }

    #[must_use]
    fn quantile_multiple(&self, points: &[f64]) -> Vec<f64> {
        let list: Vec<f64> = points
            .iter()
            .map(|x| self.quantile(*x))
            .collect::<Vec<f64>>();
        return list;
    }

    #[must_use]
    fn expected_value(&self) -> Option<f64> {
        // who would have guessed...!
        return Some(0.0);
    }

    fn variance(&self) -> Option<f64> {
        // Only discrete distribution with 0 varaince!
        return Some(0.0);
    }

    #[must_use]
    fn mode(&self) -> f64 {
        return 0.0;
    }

    #[must_use]
    fn median(&self) -> f64 {
        return 0.0;
    }

    #[must_use]
    fn skewness(&self) -> Option<f64> {
        // Since Variance is 0.0, skewness is also undefined
        return None;
    }

    #[must_use]
    fn kurtosis(&self) -> Option<f64> {
        // Since Variance is 0.0, kurtosis is also undefined
        return None;
    }

    #[must_use]
    fn excess_kurtosis(&self) -> Option<f64> {
        // Since Variance is 0.0, excess_kurtosis is also undefined
        return None;
    }

    #[must_use]
    fn moments(&self, _order: u8, mode: crate::euclid::Moments) -> f64 {
        return match mode {
            crate::euclid::Moments::Raw | crate::euclid::Moments::Central => 0.0,
            crate::euclid::Moments::Standarized => f64::NAN,
        };
    }

    #[must_use]
    fn entropy(&self) -> f64 {
        return 0.0;
    }

    #[must_use]
    fn rejection_sample_range(&self, n: usize, _pmf_max: f64, _range: (i64, i64)) -> Vec<f64> {
        return vec![0.0; n];
    }
}
