//! # Dirac distribution
//!
//! The Dirac distribution is a distribution that has probability of `1.0` for
//! the value `0.0` and `0.0` everywhere else. For this reason it always returns the
//! same value, `0.0`. It is the limit of the normal distribution if `mean = 0` when
//! the standard deviation approaches to 0.0.
//!
//! If you want the discrete version of this distribution, use
//! [Degenerate](crate::distributions::Degenerate).
//!
//! If you want to move this distribution so it outputs another value that is not `0.0`,
//! use [Location](crate::Loation).

use crate::{distribution_trait::Distribution, domain::ContinuousDomain};

pub const DIRAC_DOMAIN: ContinuousDomain = ContinuousDomain::Range(0.0, 0.0);

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Dirac {}

impl Dirac {
    #[must_use]
    pub const fn new() -> Dirac {
        return Dirac {};
    }
}

impl Distribution for Dirac {
    #[must_use]
    fn pdf(&self, x: f64) -> f64 {
        return if x == 0.0 { f64::INFINITY } else { 0.0 };
    }

    #[must_use]
    fn get_domain(&self) -> &ContinuousDomain {
        &DIRAC_DOMAIN
    }

    #[must_use]
    fn cdf(&self, x: f64) -> f64 {
        assert!(
            !x.is_nan(),
            "Tried to evaluate the cdf function of Dirac with a NaN value. \n"
        );

        return x.signum();
    }

    #[must_use]
    fn sample(&self) -> f64 {
        return 0.0;
    }

    #[must_use]
    fn quantile(&self, x: f64) -> f64 {
        // just call [Distribution::quantile_multiple]

        assert!(
            !x.is_nan(),
            "Tried to evaluate the quantile function of Dirac with a NaN value. \n"
        );

        // I think that this is what makes most sense
        return 0.0;
    }

    #[must_use]
    fn cdf_multiple(&self, points: &[f64]) -> Vec<f64> {
        points.iter().map(|x| self.cdf(*x)).collect::<Vec<f64>>()
    }

    #[must_use]
    fn sample_multiple(&self, n: usize) -> Vec<f64> {
        return vec![0.0; n];
    }

    #[must_use]
    fn quantile_multiple(&self, points: &[f64]) -> Vec<f64> {
        return vec![0.0; points.len()];
    }

    #[must_use]
    fn expected_value(&self) -> Option<f64> {
        // who would have guessed...! (continuous ver.)
        return Some(0.0);
    }

    #[must_use]
    fn variance(&self) -> Option<f64> {
        // Only continuous distribution with 0 varaince!
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

    fn rejection_sample(&self, n: usize, _pdf_max: f64) -> Vec<f64> {
        return self.sample_multiple(n);
    }

    fn rejection_sample_range(&self, n: usize, _pmf_max: f64, _range: (f64, f64)) -> Vec<f64> {
        return vec![0.0; n];
    }
}
