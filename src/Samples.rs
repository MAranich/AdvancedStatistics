pub struct Samples {
    // non-empty
    data: Vec<f64>,
    properties: SampleProperties,
}

/// Stores the sample properties of the data if they have been computed.
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
    /// The maximum value of the sample
    pub maximum: Option<f64>,
    /// The minimum value of the sample
    pub minimum: Option<f64>,
    /// Determines if the data is sorted
    pub is_sorted: bool,
}

impl Samples {
    /// Creates a new instance of [Samples] with the given `data`.
    ///
    /// `data` must not contain NaNs or infinities (`+-inf`).
    ///
    /// If you want to just move the data without copying it,
    /// use [Samples::new_move].
    pub fn new(data: &[f64]) -> Result<Samples, ()> {
        let invalid_contained: bool = data.iter().any(|f: &f64| !f.is_finite());
        if invalid_contained {
            return Err(());
        }

        return Ok(Samples {
            data: Vec::from(data),
            properties: SampleProperties::empty(),
        });
    }

    /// Creates a new instance of [Samples] with the given `data`.
    ///
    /// `data` must not contain NaNs or infinities (`+-inf`).
    ///
    /// If you don't want to move the data (to keep ownership of it),
    /// use [Samples::new].
    pub fn new_move(data: Vec<f64>) -> Result<Samples, ()> {
        let invalid_contained: bool = data.iter().any(|f: &f64| !f.is_finite());
        if invalid_contained {
            return Err(());
        }

        return Ok(Samples {
            data,
            properties: SampleProperties::empty(),
        });
    }

    /// Gives a reference to the contained data.
    ///
    /// Note that the data may be sorted or not (depending on
    /// calls to other methods).
    pub fn peek_data(&self) -> &Vec<f64> {
        return &self.data;
    }

    /// Returns the contained data and drops self.
    ///
    /// If you do not want to drop self, use [Samples::peek_data]
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
    pub fn peek_properties(&self) -> &SampleProperties {
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
            return self.properties.mean.clone();
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

        mean = mean / n as f64;

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
            return self.properties.variance.clone();
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
        let mut variance: f64 = 0.0;

        for &s in &self.data {
            variance += s * s;
        }

        variance = variance / (n - 1.0);
        variance = variance - mean * mean;

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
            return self.properties.skewness.clone();
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
                let coef_1: f64 = mean * mean * mean * (2.0 * n - 3.0);

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
            return self.properties.excess_kurtosis.clone();
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

        let min: Option<f64> = self.data.first().copied();
        if self.properties.is_sorted {
            self.properties.minimum = min.clone();
            return min;
        }

        let mut min: f64 = min.unwrap();

        // find the value manually. Also skip the first value since it's min
        for &v in &self.data[1..] {
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

        let max: Option<f64> = self.data.last().copied();
        if self.properties.is_sorted {
            self.properties.maximum = max.clone();
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
        if self.properties.is_sorted {
            return;
        }

        self.data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        self.properties.is_sorted = true;
    }

    /// Returns the quantile.
    ///
    /// Sorts the data if it is not sorted already. Returns `None` if data
    /// is empty. If `q <= 0.0` returns a the smallest value in data. 
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

        return self.data.get(index).copied();
    }
}

impl SampleProperties {
    pub fn empty() -> SampleProperties {
        SampleProperties {
            mean: None,
            variance: None,
            skewness: None,
            excess_kurtosis: None,
            maximum: None,
            minimum: None,
            is_sorted: false,
        }
        // is_sorted: data MAY be actually sorted but we cannot assume it is.
    }
}
