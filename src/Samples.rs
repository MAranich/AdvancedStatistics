use bon::bon;
use rand::Rng;

use crate::errors::AdvStatError;

pub struct Samples {
    /// All the stored values
    data: Vec<f64>,
    /// Properties related to `data`. Can contain values such as the mean, variance, skewness...
    properties: SampleProperties,
}

/// Stores the sample properties of the data if they have been computed.
#[non_exhaustive]
pub struct SampleProperties {
    /// the average of the sample
    ///
    /// Or None if `data.len() == 0`
    pub mean: Option<f64>,
    /// the variance of the sample
    ///
    /// Or None if `data.len() <= 1`
    pub variance: Option<f64>,
    /// the skewness of the sample
    pub skewness: Option<f64>,
    /// the excess kurtosis of the sample
    pub excess_kurtosis: Option<f64>,
    /// All the computed quantiles for this distribution.
    ///
    /// Stored as (q, quantile), where q [0, 1]. It is not guaranteed to be sorted or
    /// de-duplicated (use [Samples::sort_dedup_quantiles] to fix that).
    /// There are no NaNs.
    pub quantiles: Vec<(f64, f64)>,
    /// The maximum value of the sample
    pub maximum: Option<f64>,
    /// The minimum value of the sample
    pub minimum: Option<f64>,
    /// Determines if the data is sorted
    pub is_sorted: bool,
    /// Expected logarithmic value,
    /// the estimator for `E(ln(x))`
    pub log_mean: Option<f64>,
}

/// Decides wich Outlier detection algorithm use when calling [Samples::outlier_detection].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum OutlierDetectionMethod {
    /// Marks as outliers all elements that are outside `x` times the
    /// [IQR](https://en.wikipedia.org/wiki/Interquartile_range) (**I**nter **Q**uartile **R**ange).
    ///
    /// The default for [OutlierDetectionMethod] is `OutlierDetectionMethod::IQR(1.5)`
    IQR(f64),
    /// Marks as outliers all elements with a [Z-Score](https://en.wikipedia.org/wiki/Standard_score)
    /// greater in absolut value than the given value.
    ZScore(f64),
    /// Marks as outliers all values outside the provided range.
    Range(f64, f64),
}

#[bon]
impl Samples {
    /// Creates a new instance of [Samples] with the given `data`.
    ///
    /// `data` must not contain NaNs or infinities (`+-inf`).
    ///
    /// If you want to just move the data without copying it,
    /// use [Samples::new_move].
    pub fn new(data: &[f64]) -> Result<Samples, AdvStatError> {
        for elem in data {
            if !elem.is_finite() {
                if elem.is_nan() {
                    return Err(AdvStatError::NanErr);
                } else if elem.is_infinite() {
                    return Err(AdvStatError::InvalidNumber);
                }
            }
        }

        return Ok(Samples {
            data: Vec::from(data),
            properties: SampleProperties::empty(),
        });
    }

    /// Creates a new instance of [Samples] with the given `data` without checking
    /// for the absence of NaNs or infinities (`+-inf`).
    ///
    /// ## Safety
    ///
    ///  - `data` must not contain NaNs or infinities (`+-inf`).
    ///
    /// If the condition is not fullfilled, the datastructure will be invalid
    /// and may return wrong/nonsensical results.
    ///
    /// ## Alternatives
    ///
    /// If you want to just move the data without copying it,
    /// use [Samples::new_move] or [Samples::new_move_uncheched].
    #[must_use]
    pub unsafe fn new_unchecked(data: &[f64]) -> Samples {
        return Samples {
            data: Vec::from(data),
            properties: SampleProperties::empty(),
        };
    }

    /// Creates a new instance of [Samples] with the given `data`.
    ///
    /// ## Errors
    ///
    /// `data` must not contain NaNs or infinities (`+-inf`).
    ///
    /// ## Alternatives
    ///
    /// If you don't want to move the data (to keep ownership of it),
    /// use [Samples::new].
    pub fn new_move(data: Vec<f64>) -> Result<Samples, AdvStatError> {
        for elem in &data {
            if !elem.is_finite() {
                if elem.is_nan() {
                    return Err(AdvStatError::NanErr);
                } else if elem.is_infinite() {
                    return Err(AdvStatError::InvalidNumber);
                }
            }
        }

        return Ok(Samples {
            data,
            properties: SampleProperties::empty(),
        });
    }

    /// Creates a new instance of [Samples] with the given `data` without checking
    /// for the absence of NaNs or infinities (`+-inf`).
    ///
    /// ## Safety
    ///
    /// The data must not contain NaNs or infinities (`+-inf`).
    ///
    /// If the condition is not fullfilled, the datastructure will be invalid
    /// and may return wrong/nonsensical results.
    ///
    /// ## Alternatives
    ///
    /// If you don't want to move the data (to keep ownership of it),
    /// use [Samples::new] / [Samples::new_unchecked].
    #[must_use]
    pub const unsafe fn new_move_uncheched(data: Vec<f64>) -> Samples {
        return Samples {
            data,
            properties: SampleProperties::empty(),
        };
    }

    /// Gives a reference to the contained data.
    ///
    /// Note that the data may be sorted or not (depending on
    /// calls to other methods).
    #[must_use]
    pub const fn peek_data(&self) -> &Vec<f64> {
        return &self.data;
    }

    /// Returns the contained data and drops self.
    ///
    /// If you do not want to drop self, use [Samples::peek_data]
    #[must_use]
    pub fn get_data(self) -> Vec<f64> {
        return self.data;
    }

    /// Drops the previous self and creates a new one
    /// with the new data appended to the old data.
    ///
    /// If you want to just move the data without copying it,
    /// use [Samples::add_data_move].
    ///
    /// `data` must not contain NaNs or infinities (`+-inf`).
    /// If it does, the original self will be returned.
    ///
    /// Note that the internal [SampleProperties] is emptied.
    #[allow(clippy::result_large_err)]
    pub fn add_data(self, data: &[f64]) -> Result<Samples, Samples> {
        let invalid_contained: bool = data.iter().any(|f: &f64| !f.is_finite());
        if invalid_contained {
            return Err(self);
        }

        let mut join_data: Vec<f64> = self.data;
        join_data.reserve(data.len());

        for s in data {
            join_data.push(*s);
        }

        return Ok(Samples {
            data: join_data,
            properties: SampleProperties::empty(),
        });
    }

    /// Drops the previous self and creates a new one
    /// with the new data appended to the old data.
    ///
    /// If you don't want to move the data (to keep ownership of it),
    /// use [Samples::add_data].
    ///
    /// `data` must not contain NaNs or infinities (`+-inf`).
    /// If it does, the original self will be returned.
    ///
    /// Note that the internal [SampleProperties] is emptied.
    #[allow(clippy::result_large_err)]
    pub fn add_data_move(self, data: Vec<f64>) -> Result<Samples, Samples> {
        let invalid_contained: bool = data.iter().any(|f: &f64| !f.is_finite());
        if invalid_contained {
            return Err(self);
        }

        let join_data: Vec<f64> = [self.data, data].concat();
        return Ok(Samples {
            data: join_data,
            properties: SampleProperties::empty(),
        });
    }

    /// Returns a reference to the internal field that contains
    /// all computed statistics ([SampleProperties]).
    ///
    /// If you want to also compute the properties, use [Samples::get_properties].
    #[must_use]
    pub const fn peek_properties(&self) -> &SampleProperties {
        return &self.properties;
    }

    /// Computes the sample [mean](https://en.wikipedia.org/wiki/Mean) and returns it.
    ///
    /// Returns [None] if there is not enough samples to compute the mean.
    ///
    /// If the mean was already computed, it just returns the value
    /// stored in [SampleProperties] and the operation is constant time.
    pub fn mean(&mut self) -> Option<f64> {
        // If it is already computed, jut return it.
        if self.properties.mean.is_some() {
            return self.properties.mean;
        }

        let n: usize = self.data.len();
        if n == 0 {
            // No mean for 0 samples.
            return None;
        }

        // actual computation of the mean.
        let mut mean: f64 = 0.0;

        for &s in &self.data {
            mean += s;
        }

        mean = mean / (n as f64);

        // Store for use in the future.
        self.properties.mean = Some(mean);
        return Some(mean);
    }

    /// Computes the sample [variance](https://en.wikipedia.org/wiki/Variance)
    /// and returns it.
    ///
    /// Returns [None] if there is not enough samples to compute the variance
    /// (0 or 1 samples only).
    ///
    /// If the variance was already computed, it just returns the value
    /// stored in [SampleProperties] and the operation is constant time.
    pub fn variance(&mut self) -> Option<f64> {
        // If it is already computed, jut return it.
        if self.properties.variance.is_some() {
            return self.properties.variance;
        }

        let n: usize = self.data.len();
        if n < 2 {
            // No variance for 0 or 1 samples.
            return None;
        }

        let n: f64 = n as f64;

        // actual computation of the variance.

        // get mean, it always exists because there is more than 1 sample
        let mean: f64 = self.mean().unwrap();
        let minus_mean: f64 = -mean;
        let mut variance: f64 = 0.0;

        for &s in &self.data {
            let a: f64 = s + minus_mean;
            variance += a * a;
        }

        variance = variance / (n - 1.0);

        self.properties.variance = Some(variance);
        return Some(variance);
    }

    /// Computes the sample [skewness](https://en.wikipedia.org/wiki/Skewness)
    /// and returns it.
    ///
    /// Returns [None] if there is not enough samples to compute the skewness
    /// (0 or 1 samples only).
    ///
    /// If the skewness was already computed, it just returns the value
    /// stored in [SampleProperties] and the operation is constant time.
    pub fn skewness(&mut self) -> Option<f64> {
        // If it is already computed, just return it.
        if self.properties.skewness.is_some() {
            return self.properties.skewness;
        }

        let n: usize = self.data.len();
        if n < 2 {
            // No skewness for 0 or 1 samples.
            return None;
        }

        let n: f64 = n as f64;

        // actual computation of the skewness.

        // get mean, it always exists because there is more than 1 sample
        let mean: f64 = self.mean().unwrap();
        let variance: f64 = self.variance().unwrap();
        let std_dev: f64 = variance.sqrt();

        /*

                Computation of Skewness:

            We will use the formula provided at [wikipedia](https://en.wikipedia.org/wiki/Skewness#Sample_skewness):

            **1)**
            `Sk = n /((n - 1) * (n - 2)) * sumatory[ ((x_i - m)/s)^3 ]`

            However, this is equivalent to:

            **2)**
            `Sk = n / ((n - 1) * (n - 2) * s^3) * sumatory[ (x_i - m)^3 ] `

            And:

            **3)**

            `Sk = n / ((n - 1) * (n - 2) * s^3) * ( m^3*(2*n - 3) - 3*m*var*(n-1) + sumatory[ x_i^3 ] )`

            We do not currenly know wich one is better.

            ***

            Considering that we are assumed to know the mean (`m`) and the variance (`var`),
            The only problem is figuring out wich of the following is better in terms of
            speed and acuracy:

            1. sumatory[ ((x_i - m)/s)^3 ]
            2. sumatory[ (x_i - m)^3 ]
            3. sumatory[ x_i^3 ]

            Intuitively method **3** is better because it does less operations, but it is
            also more vulnerable to overflow realted problems. For now I will implement the
            3 methods and test later.

            I will leave method 2 active by deafult.
            This is subject to change in the future.

        */

        let mut skewness: f64 = 0.0;

        let method: u8 = 2;
        assert!(method == 1 || method == 2 || method == 3);

        match method {
            1 => {
                // ## Method 1:
                // `Sk = n /((n - 1) * (n - 2)) * sumatory[ ((x_i - m)/s)^3 ]`

                let minus_mean: f64 = -mean;
                let inv_std_dev: f64 = 1.0 / std_dev;

                for &s in &self.data {
                    let standardized_value: f64 = (s + minus_mean) * inv_std_dev;
                    skewness += standardized_value * standardized_value * standardized_value;
                }

                skewness = skewness * (n / ((n - 1.0) * (n - 2.0)));
            }
            2 => {
                // ## Method 2:
                // `Sk = n / ((n - 1) * (n - 2) * s^3) * sumatory[ (x_i - m)^3 ] `

                let minus_mean: f64 = -mean;

                for &s in &self.data {
                    let centered_value: f64 = s + minus_mean;
                    skewness += centered_value * centered_value * centered_value;
                }

                let std_dev_cubed: f64 = std_dev * std_dev * std_dev;
                skewness = skewness * (n / ((n - 1.0) * (n - 2.0) * std_dev_cubed));
            }
            3 => {
                // ## Method 3:
                // `Sk = n / ((n - 1) * (n - 2) * s^3) * ( m^3*(2*n - 3) - 3*m*var*(n-1) + sumatory[ x_i^3 ] )`

                for &s in &self.data {
                    skewness += s * s * s;
                }

                // m^3*(2*n - 3)
                // let coef_1: f64 = mean * mean * mean * (2.0 * n - 3.0);
                let coef_1: f64 = mean * mean * mean * 2.0_f64.mul_add(n, -3.0);

                // -3*m*var*(n-1)
                let coef_2: f64 = -3.0 * mean * variance * (n - 1.0);

                skewness = skewness + coef_1 + coef_2;

                let std_dev_cubed: f64 = std_dev * std_dev * std_dev;
                skewness = skewness * (n / ((n - 1.0) * (n - 2.0) * std_dev_cubed));
            }
            _ => unreachable!(),
        };

        self.properties.skewness = Some(skewness);
        return Some(skewness);
    }

    pub fn excess_kurtosis(&mut self) -> Option<f64> {
        /*
               Excess kurtosis computation:

           We will use the unbiased estimator [formula](https://en.wikipedia.org/wiki/Kurtosis#Standard_unbiased_estimator).

           ```
           Ex_k = (n+1)*n / ((n-1)*(n-2)*(n-3)) * sumatory[ (x_i - m)^4 ] / (sumatory [ (x_i - m)^2 ])^2 - 3*(n-1)^2 / ((n-2)*(n-3))
           Ex_k = (n+1)*n / ((n-1)*(n-2)*(n-3)) * sumatory[ (x_i - m)^4 ] / var^2 - 3*(n-1)^2 / ((n-2)*(n-3))

           ```

           To organize:

           ```
           let coef_1 = (n+1)*n / ((n-1)*(n-2)*(n-3));
           let coef_2 = - 3*(n-1)^2 / ((n-2)*(n-3));

           Ex_k = coef_1 * sumatory[ (x_i - m)^4 ] / var^2 + coef_2

           ```

        */

        // If it is already computed, just return it.
        if self.properties.excess_kurtosis.is_some() {
            return self.properties.excess_kurtosis;
        }

        let n: usize = self.data.len();
        if n < 3 {
            // No excess_kurtosis for 0 or 1 or 2 samples.
            return None;
        }

        let n: f64 = n as f64;

        // actual computation of the skewness.

        // get mean, it always exists because there is more than 1 sample
        let mean: f64 = self.mean().unwrap();
        let variance: f64 = self.variance().unwrap();

        // let coef_1 = (n+1)*n / ((n-1)*(n-2)*(n-3));
        let coef_1: f64 = ((n + 1.0) * n) / ((n - 1.0) * (n - 2.0) * (n - 3.0));

        // let coef_2 = - 3*(n-1)^2 / ((n-2)*(n-3));
        let coef_2: f64 = -3.0 * (n - 1.0) * (n - 1.0) / ((n - 2.0) * (n - 3.0));

        let minus_mean: f64 = -mean;

        let mut kurtosis: f64 = 0.0;
        for &s in &self.data {
            let centered_value: f64 = s + minus_mean;
            let centered_value_sq: f64 = centered_value * centered_value;
            kurtosis += centered_value_sq * centered_value_sq;
        }

        kurtosis = coef_1 * kurtosis / (variance * variance);
        let excess_kurtosis: f64 = kurtosis + coef_2;

        self.properties.excess_kurtosis = Some(excess_kurtosis);
        return Some(excess_kurtosis);
    }

    /// Returns the minimum among the data.
    ///
    /// Returns `None` if there are 0 samples. If the data is already sorted,
    /// returns in constant time. Otherwise it's `O(n)` (does not sort the data).
    pub fn minimum(&mut self) -> Option<f64> {
        let n: usize = self.data.len();
        if n == 0 {
            return None;
        }

        // If it is already computed, just return it.
        if self.properties.minimum.is_some() {
            return self.properties.minimum;
        }

        let min: Option<f64> = self.data.first().copied();
        if self.properties.is_sorted {
            self.properties.minimum = min;
            return min;
        }

        let mut min: f64 = min.unwrap();

        // find the value manually. Also skip the first value since it's min
        for &v in &self.data {
            if min < v {
                min = v;
            }
        }

        self.properties.minimum = Some(min);
        return Some(min);
    }

    /// Returns the maximum among the data.
    ///
    /// Returns `None` if there are 0 samples. If the data is already sorted,
    /// returns in constant time. Otherwise it's `O(n)` (does not sort the data).
    pub fn maximum(&mut self) -> Option<f64> {
        let n: usize = self.data.len();
        if n == 0 {
            return None;
        }

        // If it is already computed, just return it.
        if self.properties.maximum.is_some() {
            return self.properties.minimum;
        }

        let max: Option<f64> = self.data.last().copied();
        if self.properties.is_sorted {
            self.properties.maximum = max;
            return max;
        }

        let mut max: f64 = max.unwrap();

        // find the value manually. Will check last value twice but that's ok.
        for &v in &self.data {
            if max < v {
                max = v;
            }
        }

        self.properties.maximum = Some(max);
        return Some(max);
    }

    /// Forces to sort the internal data if it is not sorted already.
    #[inline]
    pub fn sort_data(&mut self) {
        if !self.properties.is_sorted {
            self.data
                .sort_unstable_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());
            self.properties.is_sorted = true;
        }
    }

    pub fn sort_dedup_quantiles(&mut self) {
        self.properties
            .quantiles
            .sort_unstable_by(|a: &(f64, f64), b: &(f64, f64)| a.0.partial_cmp(&b.0).unwrap());

        self.properties
            .quantiles
            .dedup_by(|a: &mut (f64, f64), b: &mut (f64, f64)| {
                a.0.partial_cmp(&b.0).unwrap().is_eq()
            });
    }

    /// Returns the [quantile](https://en.wikipedia.org/wiki/Quantile) of `q`.
    ///
    /// Sorts the data if it is not sorted already. Returns `None` if data
    /// is empty.
    ///
    /// If `q <= 0.0` returns a the smallest value in data.
    /// If `1.0 <= q` returns a the greatest value in data.
    pub fn quantile(&mut self, mut q: f64) -> Option<f64> {
        // sort data if it has not been sorted already.
        self.sort_data();

        // handle edge cases
        let n: usize = self.data.len();
        if n == 0 {
            return None;
        }

        q = q.clamp(0.0, 1.0);

        // We use the [nearest rank method](https://en.wikipedia.org/wiki/Percentile#The_nearest-rank_method).
        // but adapted in order to work from quantiles, where q ranges from [0, 1] instead of
        // ranging from (0, 100].

        let index: usize = ((n as f64) * q).ceil() as usize;
        // SAFETY: index belongs to `0..self.data.len()`
        let quantile: f64 = unsafe { self.data.get(index).copied().unwrap_unchecked() };
        self.properties.quantiles.push((q, quantile));

        return Some(quantile);
    }

    /// Returns the mean of the logarithm
    ///
    /// It is the estimatior for `E(ln(x))`.
    ///
    /// It will return [None] under any of the following conditions:
    ///  - There are 0 samples in [Samples]
    ///  - If any of the samples is negative.
    ///
    /// Also, if any sample is exacly 0, `Some(-inf)` is returned.
    pub fn log_mean(&mut self) -> Option<f64> {
        let mut acc: f64 = 0.0;

        for &sample in &self.data {
            if sample.is_sign_negative() {
                // Value in properties should already be None, so no need to set it.
                return None;
            }

            acc += sample.ln();
        }

        let mean_ln: f64 = acc / (self.data.len() as f64);
        self.properties.log_mean = Some(mean_ln);
        return Some(mean_ln);
    }

    /// Compute multiple properties in bulk.
    ///
    /// Returns a reference to the updated [SampleProperties]
    /// (same as using [Samples::peek_properties]). Note that the computation of some
    /// values implie indirecly the computation of others. For example, if you ask
    /// to compute only the `variance`, the `mean` will also be computed as a by-product.
    #[builder]
    pub fn get_properties(
        &mut self,
        #[builder(default)] mean: bool,
        #[builder(default)] variance: bool,
        #[builder(default)] skewness: bool,
        #[builder(default)] excess_kurtosis: bool,
        #[builder(default)] quantiles: Vec<f64>,
        #[builder(default)] maximum: bool,
        #[builder(default)] minimum: bool,
        #[builder(default)] sort: bool,
        #[builder(default)] log_mean: bool,
    ) -> &SampleProperties {
        if !quantiles.is_empty() || sort {
            self.sort_data();
        }

        // The functions already assing the values to [SampleProperties] internally.
        if mean {
            let _: Option<f64> = self.mean();
        }
        if variance {
            let _: Option<f64> = self.variance();
        }
        if skewness {
            let _: Option<f64> = self.skewness();
        }
        if excess_kurtosis {
            let _: Option<f64> = self.excess_kurtosis();
        }
        if minimum {
            let _: Option<f64> = self.minimum();
        }
        if maximum {
            let _: Option<f64> = self.maximum();
        }

        if !quantiles.is_empty() {
            for q in quantiles {
                let _: Option<f64> = self.quantile(q);
            }
            self.sort_dedup_quantiles();
        }

        if log_mean {
            let _: Option<f64> = self.log_mean();
        }

        return &self.properties;
    }

    /// Returns the number of elements in [Samples]
    ///
    /// Identical to [Samples::count]
    #[must_use]
    pub fn len(&self) -> usize {
        return self.data.len();
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        return self.data.len() == 0;
    }

    /// Returns the number of elements in [Samples]
    ///
    /// Identical to [Samples::len]
    #[must_use]
    // #[confusable = len]
    pub fn count(&self) -> usize {
        //#[rustc_confusables("len", "size", "count", "shape", "n")]
        return self.data.len();
    }

    /// Get a resample of the data with repetition.
    ///
    /// A classical [bootstrap resample](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)).
    ///
    /// See also: [Samples::get_resample_multiple], [Samples::get_permutation]
    #[must_use]
    pub fn get_resample(&self) -> Vec<f64> {
        let mut rng: rand::prelude::ThreadRng = rand::rng();
        let len: usize = self.data.len();
        let mut resample: Vec<f64> = Vec::with_capacity(len);

        for _ in 0..len {
            let index: usize = rng.random_range(0..len);
            // SAFETY: index belongs to `0..len`
            let value: f64 = unsafe { *self.data.get_unchecked(index) };
            resample.push(value);
            // Safety: it is safe because we generated an index
            // that is specifically within the valid values.
        }

        return resample;
    }

    /// Get multiple resamples of the data with repetition.
    ///
    /// A classical [bootstrap resample](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)).
    ///
    /// See also: [Samples::get_resample_multiple], [Samples::get_permutation]
    #[must_use]
    pub fn get_resample_multiple(&self, n: usize) -> Vec<Vec<f64>> {
        let mut rng: rand::prelude::ThreadRng = rand::rng();
        let len: usize = self.data.len();
        let mut ret: Vec<Vec<f64>> = Vec::with_capacity(n);

        for _ in 0..n {
            let mut resample: Vec<f64> = Vec::with_capacity(len);

            for _ in 0..len {
                let index: usize = rng.random_range(0..len);
                // SAFETY: index belongs to `0..len`
                let value: f64 = unsafe { *self.data.get_unchecked(index) };
                resample.push(value);
                // Safety: it is safe because we generated an index
                // that is specifically within the valid values.
            }
            ret.push(resample);
        }
        return ret;
    }

    /// Returns a random permutation of the data.
    ///
    /// Identical to [random_permutation](crate::euclid::random_permutation).
    /// Sorts in `O(n)` time using the [Fisher–Yates shuffle](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle).
    ///
    /// See also: [Samples::get_resample], [Samples::get_permutation_multiple]
    #[must_use]
    pub fn get_permutation(&self) -> Vec<f64> {
        // [Fisher–Yates shuffle](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle)

        let mut rng: rand::prelude::ThreadRng = rand::rng();
        let len: usize = self.data.len();
        let mut new_data: Vec<f64> = self.data.clone();

        for i in (1..len).rev() {
            let mut j: f64 = rng.random::<f64>();
            j = j * ((i + 1) as f64);
            let j: usize = j as usize;
            // k belongs to  [0, i - 1]

            new_data.swap(i, j);
        }

        return new_data;
    }

    /// Returns a random permutation of the data.
    ///
    /// Same as [Samples::get_permutation] but for `n` samples at once.
    /// Identical implementation to [random_permutation](crate::euclid::random_permutation).
    /// Sorts in `O(n)` time using the [Fisher–Yates shuffle](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle).
    ///
    /// See also: [Samples::get_permutation], [Samples::get_resample]
    #[must_use]
    pub fn get_permutation_multiple(&self, n: usize) -> Vec<Vec<f64>> {
        // [Fisher–Yates shuffle](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle)

        let mut rng: rand::prelude::ThreadRng = rand::rng();
        let len: usize = self.data.len();
        let mut ret: Vec<Vec<f64>> = Vec::with_capacity(n);

        for _ in 0..n {
            let mut new_data: Vec<f64> = self.data.clone();

            for i in (1..len).rev() {
                let mut j: f64 = rng.random::<f64>();
                j = j * ((i + 1) as f64);
                let j: usize = j as usize;
                // k belongs to  [0, i - 1]

                new_data.swap(i, j);
            }
            ret.push(new_data);
        }

        return ret;
    }

    /// Returns a vector of indicies of all the samples marked as otliers byt the
    /// selected [OutlierDetectionMethod].
    #[must_use]
    pub fn outlier_detection(&mut self, method: OutlierDetectionMethod) -> Vec<usize> {
        let mut ret: Vec<usize> = Vec::new();

        if self.count() < 3 {
            return ret;
        }

        match method {
            OutlierDetectionMethod::IQR(m) => {
                if m.is_nan() || m.is_sign_negative() {
                    // return empty vector
                    return ret;
                }

                // Q1 − 1.5 IQR ////// Q3 + 1.5 IQR
                // SAFETY: we already checked for `self.count() < 3`
                let q1: f64 = unsafe { self.quantile(0.25).unwrap_unchecked() };
                // SAFETY: we already checked for `self.count() < 3`
                let q3: f64 = unsafe { self.quantile(0.75).unwrap_unchecked() };

                let iqr: f64 = q3 - q1;
                let mult_iqr: f64 = m * iqr;

                let range: (f64, f64) = (q1 - mult_iqr, q3 + mult_iqr);

                for (i, &sampl) in self.data.iter().enumerate() {
                    if !(range.0 <= sampl && sampl <= range.1) {
                        ret.push(i);
                    }
                }
            }
            OutlierDetectionMethod::ZScore(s) => {
                if s.is_nan() || s.is_sign_negative() {
                    // return empty vector
                    return ret;
                }

                // SAFETY: we already checked for `self.count() < 3`
                let mean: f64 = unsafe { self.mean().unwrap_unchecked() };
                // SAFETY: we already checked for `self.count() < 3`
                let variance: f64 = unsafe { self.variance().unwrap_unchecked() };

                let minus_mean: f64 = -mean;
                let inv_std_dev: f64 = 1.0 / variance.sqrt();

                for (i, &sampl) in self.data.iter().enumerate() {
                    let score: f64 = (sampl + minus_mean) * inv_std_dev;
                    if s < score.abs() {
                        ret.push(i);
                    }
                }
            }
            OutlierDetectionMethod::Range(min, max) => {
                if min.is_nan() || max.is_nan() {
                    // return empty vector
                    return ret;
                }

                if max <= min {
                    // return full vector
                    return (0..self.count()).collect::<Vec<usize>>();
                }

                for (i, &sampl) in self.data.iter().enumerate() {
                    if !(min <= sampl && sampl <= max) {
                        ret.push(i);
                    }
                }
            }
        }

        return ret;
    }

    /// Returns 2 Samples structures `(inliners, outliers)` from self and the given
    /// outliers.
    #[must_use]
    pub fn outlier_removal(self, outliers_idx: Vec<usize>) -> (Samples, Samples) {
        let mut original_data: Vec<f64> = self.get_data();

        let mut outliers_data: Vec<f64> = Vec::with_capacity(outliers_idx.len());
        for o in outliers_idx {
            let outlier: f64 = original_data.swap_remove(o);
            outliers_data.push(outlier);
        }

        // SAFETY: since the data comes from a valid Samples, both parts of
        // the splitted data are also valid.
        let inliners: Samples = unsafe { Samples::new_move_uncheched(original_data) };
        // SAFETY: Same as before
        let outliers: Samples = unsafe { Samples::new_move_uncheched(outliers_data) };

        return (inliners, outliers);
    }
}

impl SampleProperties {
    #[must_use]
    #[inline]
    pub const fn empty() -> SampleProperties {
        return SampleProperties {
            mean: None,
            variance: None,
            skewness: None,
            excess_kurtosis: None,
            quantiles: Vec::new(),
            maximum: None,
            minimum: None,
            is_sorted: false,
            log_mean: None,
        };
        // is_sorted: data MAY be actually sorted but we cannot assume it is.
        // And we don't want to check.
    }
}

impl Default for OutlierDetectionMethod {
    #[inline]
    fn default() -> Self {
        // Default value of 1.5 for the IQR variant
        return OutlierDetectionMethod::IQR(1.5);
    }
}
