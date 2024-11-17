///! This script contains the interfaces used to comunicate with the distributions.
use crate::errors::*;
use crate::euclid::*;

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

    fn get_pdf_domain(&self) -> &Domain;

    // Provided methods:
    // Manual implementation for a specific distribution is recommended.

    /// Evaluates the [CDF](https://en.wikipedia.org/wiki/Cumulative_distribution_function)
    /// (Cumulative distribution function).
    /// If the function is evaluated outside the domain of the pdf,
    /// it will return  either `0.0` or  `1.0`.
    fn cdf(&self, x: f64) -> f64 {
        #![allow(unused_variables)]

        let mut cumulative_probability: f64 = 0.0;

        let domain = self.get_pdf_domain();

        todo!("Implement deafult implementation. ");
    }

    /// Samples the distribution at random.
    ///
    /// The deafult method is [rejection sampling](https://en.wikipedia.org/wiki/Rejection_sampling)
    /// unless the deadult method is overriden.
    fn sample(&self) -> f64 {
        todo!("Implement deafult implementation. ");
        // Use rejection sampling
        todo!("Add documentation for how random numbers are generated (state/seed). ");
    }

    /// Evaluates the [quantile function](https://en.wikipedia.org/wiki/Quantile_function).
    ///  - `x` must be in the range [0.0, 1.0] or an error will be returned.
    ///
    /// The quantile function also happens to be the inverse function of [Distribution::cdf].
    fn quantile(&self, x: f64) -> Result<f64, AdvStatError> {
        #![allow(unused_variables)]
        todo!("Implement deafult implementation. ");
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
    /// If this function is overriden, it may provide a computational advantage.
    fn quantile_multiple(&self, points: &[f64]) -> Result<Vec<f64>, AdvStatError> {
        let mut ret: Vec<f64> = Vec::with_capacity(points.len());

        for point in points {
            ret.push(self.quantile(*point)?);
        }

        return Ok(ret);
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
