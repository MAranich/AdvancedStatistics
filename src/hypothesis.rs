//! # Hypothesys testing
//!
//! This module contains the definition for [Hypothesis] and some common
//! statistical tests.
//!
//! ## Introduction:
//!
//! *This section is a brief introduction to statistical concepts*
//!
//! In statistics, when we want to make a claim about reality, we perform
//! an hypotesys test.
//!
//! ***
//!
//! We generate 2 hypothesys:
//!  - The null hypothesys (`H0`)
//!  - The alternative hypothesys (`Ha` or `H1`)
//!
//! The **null hypothesys** claims that there does not exist any effect. Under this
//! hypothesys anything we observe is just a product of random chance.
//!
//! The **alternative hypothesys** claims that there is an effect and is usually
//! the one that we want to prove it's true.
//!
//! To do a hypotesys test we must:
//!
//!  - Have a claim must be testable using data.
//!  - Then we have to select an statistic that can allow us to test the hypothesys
//!     - The expected value, for example.
//!  - Then we need to know the distribution of the test statistic under the null
//!     hypothesys (known as null distribution)
//!
//! Here we can take 2 approaches: selecting a significance level (denoted by
//! `alpha`) or using the p value.
//!
//! ### Using the significance level
//!
//! The significance level (denoted by `alpha`) is the probability of commiting
//! a Type 1 error. In other words, `alpha` is probability of accepting `Ha`
//! given `H0` is true:
//!
//!  > alpha = P( Rejecting H0 | H0 is true )
//!
//! `alpha` is a probability that we choose wich is defined as the maximum
//! accptable rate of type 1 error. That means that when we reject `H0` we also
//! accept the possibility of being wrong one every `1/alpha` times. Common values
//! for `alpha` are `0.05` and `0.01`. Smaller values means less probability of
//! being wrong but comes at a cost of more sample size of less power (failing to reject
//! `H0` when it was actually false).
//!
//! When we select a certain significance level (`alpha`) we can generate a certain
//! conficence interval of the null distribution. The interval is designed in such
//! a way that there is a `1 - alpha` chance of the statistic falling inside the
//! interval assuming `H0`. In other words, there is `alpha` chance of getting a
//! statistic so *extreme* that actaually falls outside the confidence interval.
//!
//! If our statistic falls outside the confidence interval, it means that probably
//! it comes from some other distribution. Wich means that the null hypothesys `H0`
//! is probably false and we can **reject the null hypotesys**.
//!
//! On the other hand, if the statistic falls inside the confidence interval, means
//! that our data supports the null hypothesys `H0`. In that case we **fail to reject
//! the null hypothesys**.
//!
//! ### Using the P value
//!
//! The P value is a number that intends to summarize all the information
//! requiered to reject or fail to reject the null hypothesys `H0`. The P
//! value can be interpreted on the following (equivalent ways):
//!  - The probability of the null distribution generating a statistic
//!     as extreme or more than the one obtained.
//!  - The value for alpha that makes the statistic lay on the boundary of the
//!     confidence interval.
//!      - That means that if we use the significance level approach and we select
//!         `alpha < p` we will fail to reject `H0` and `p < alpha` we will
//!         reject `H0`.
//!
//! If we obtain a very extreme P value, we can forget about the significance
//! level (wich usually is an arvitrary value choice) and jump to the ç
//! conclusions immidiately:
//!  - If the P value is **very small** (for example `P < 0.01`) => reject `H0`.
//!  - If the P value is **very large** (for example `0.1 < p`) => fail to reject `H0`.
//!
//! ## Implementation
//!
//! We have implemeted some common tests. Each of the functions take the
//! necessary arguments and return the corresponding [P value](https://en.wikipedia.org/wiki/P-value)
//! of the test.
//!
//! However, take into account that there are assumptions of the tests that we cannot
//! check. For this reason there is the necessary documentation that explains:
//!  - What the test does (introduction)
//!  - The assumptions
//!  - The inputs
//!  - The results (and errors)
//!
//! But keep in mind that the user needs to make sure that the necessary assumprions
//! for the test to give results that are valid.
//!
//! ## Hypotesys struct
//!
//! Todo:
//!
//!

use std::hint::assert_unchecked;

use crate::{
    distribution_trait::Distribution,
    distributions::{Normal::STD_NORMAL, StudentT::StudentT},
    errors::TestError,
    samples::Samples,
};

/// Defines Wich kind of test are we doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
#[allow(clippy::exhaustive_enums)]
pub enum Hypothesis {
    /// A [Hypothesis::RightTail] will test if our statisitc is *significantly*
    /// bigger than what `H0` claims. (`theta_0 < theta_obs`)
    RightTail,
    /// A [Hypothesis::LeftTail] will test if our statisitc is *significantly*
    /// smaller than what `H0` claims. (`theta_obs < theta_0`)
    LeftTail,
    /// A [Hypothesis::TwoTailed] will test if our statisitc is *significantly*
    /// different (far away) of what `H0` claims. (`theta_obs != theta_0`)
    ///
    /// Divides the probability evenly between both sides.
    #[default]
    TwoTailed,
}

/// Contains the result of the test
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum TestResult {
    /// The obtained statistic and [P value](https://en.wikipedia.org/wiki/P-value)
    PValue(f64, f64),
    /// The obtained statistic, [P value](https://en.wikipedia.org/wiki/P-value) and the confidence interval
    PValueCI(f64, f64, (f64, f64)),
}

impl TestResult {
    /// Returns the [P value](https://en.wikipedia.org/wiki/P-value).
    #[must_use]
    pub const fn p(&self) -> f64 {
        // convinience method for quickly retriving the p value
        return match *self {
            TestResult::PValue(_, p) | TestResult::PValueCI(_, p, _) => p,
        };
    }

    #[must_use]
    pub const fn statisitc(&self) -> f64 {
        // convinience method for quickly retriving the statisitc value
        return match *self {
            TestResult::PValue(s, _) | TestResult::PValueCI(s, _, _) => s,
        };
    }
}

impl Default for TestResult {
    fn default() -> Self {
        return TestResult::PValue(f64::NAN, -0.0);
    }
}

/// Performs a general test and returns the probability (P value) of `statistic` being
/// drawn from the `null` distribution.
///
/// *do not use*: may be deprecated in the future or completely re-written
#[must_use]
pub fn general_test<T: crate::distribution_trait::Distribution>(
    hypothesys: Hypothesis,
    statistic: f64,
    null: &T,
) -> f64 {
    return null.p_value(hypothesys, statistic);
}

/// Performs a [Z-test](https://en.wikipedia.org/wiki/Z-test) **for the mean**
/// with the given `data` and `hypotesys`. This test can be used to test if the mean of
/// a dataset is different from the null hypothesys when the variance is known.
///
/// Usually a [t-test](t_test) is prefered for most cases unless the sample size is *very big*.
///
/// ## Assumptions of the test
///
/// 1. [IID samples](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)
/// 2. Assumes that the null distribution of the test statistic
///     (the mean) can be aproximated with a
///     [Normal](crate::distributions::Normal) distribution.
///      - This can be assumed trough the [CLT](https://en.wikipedia.org/wiki/Central_limit_theorem)
///         if it applies.
///      - Can also be assumed if it is known that the samples are drawn
///             from a [Normal](crate::distributions::Normal) distribution.
/// 3. The mean (= 0.0) and standard deviation of the null distribution are known
///      - (Or estimated with high accuracy)
///
/// If the conditions for the test are **not** fullfilled, then the result is meaningless.
///
/// ## Inputs:
///
/// 1. `data`: all the samples collected to perform the test.
/// 2. `hypothesys`: (optional) determines if a 2-tailed/left-tailed/right-tailed will be used
///      - The default is a 2 tailed test.
/// 3. `null`: (optional) the [null distribution](https://en.wikipedia.org/wiki/Null_distribution)
/// of the mean.
///      - Contains the null hypothesys mean
///      - Contains the **known** standard deviation of the null distribution.
///      - The default is a standard normal (0 mean, 1 std_dev).
/// 4. `significance`: (optional) If left empty, only the P-value will be computed.
///     Otherwise, a confidence interval with the given significance level (alpha)
///     will be computed.
///      - It needs to be a valid probability (`0 < significance < 1`, tipically 0.05 or less)
///      - (The P-value is always computed)
///
/// ## Results
///
/// If the test is performed correcly, it returns a [TestResult] with the P value
/// and the confidence interval if `significance` was provided.
///
/// If there is not enough samples in `data`, returns [TestError::NotEnoughSamples].
///
#[bon::builder]
pub fn z_test(
    data: &mut Samples,
    #[builder(default)] hypothesys: Hypothesis,
    #[builder(default)] null: crate::distributions::Normal::Normal,
    significance: Option<f64>,
) -> Result<TestResult, TestError> {
    let sample_mean: f64 = match data.mean() {
        Some(m) => m,
        None => return Err(TestError::NotEnoughSamples),
    };

    let statistic: f64 = (sample_mean - null.get_mean()) * (data.count() as f64).sqrt()
        / (null.get_standard_deviation());

    let p: f64 = STD_NORMAL.p_value(hypothesys, statistic);

    let ret: TestResult = if let Some(alpha) = significance {
        #[allow(clippy::nonminimal_bool)]
        if !alpha.is_finite() || !(0.0 < alpha && alpha < 1.0) {
            return Err(TestError::InvalidSignificance);
        }
        let confidence_interval: (f64, f64) = null.confidence_interval(hypothesys, alpha);
        TestResult::PValueCI(statistic, p, confidence_interval)
    } else {
        TestResult::PValue(statistic, p)
    };

    return Ok(ret);
}

/// Performs a one sample [t-test](https://en.wikipedia.org/wiki/Student%27s_t-test) for the mean.
/// Can be used to determine if a the mean of the data is different to the one form
/// the null distribution (usally mean = 0).
///
/// ## Assumptions of the test
///
/// 1. [IID samples](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)
/// 2. Assumes that the null distribution of the mean can be aproximated
///     with a [Normal](crate::distributions::Normal) distribution.
///      - This can be assumed trough the [CLT](https://en.wikipedia.org/wiki/Central_limit_theorem)
///         if it applies.
///      - Can also be assumed if it is known that the samples are drawn
///         from a [Normal](crate::distributions::Normal) distribution.
/// 3. The mean of the null distribution are known
///      - (Or estimated with high accuracy)
///
/// If the conditions for the test are **not** fullfilled, then the result is meaningless.
///
/// ## Inputs:
///
/// 1. `data`: all the samples collected to perform the test.
/// 2. `hypothesys`: (optional) determines if a 2-tailed/left-tailed/right-tailed will be used
///      - The default is a 2 tailed test.
/// 3. `null_mean`: the mean of the null distribution.
///      - The default is 0.
/// 4. `significance`: (optional) If left empty, only the P-value will be computed.
///     Otherwise, a confidence interval with the given significance level (alpha)
///     will be computed.
///      - It needs to be a valid probability (`0 < significance < 1`, tipically 0.05 or less)
///      - (The P-value is always computed)
///
/// ## Results
///
/// If the test is performed correcly, it returns a [TestResult] with the P value
/// and the confidence interval if `significance` was provided.
///
/// If there is not enough samples in `data`, returns [TestError::NotEnoughSamples].
///
/// ## Notes:
///
/// If the sample size is very big, the t-test is equivalent to the [Z-test](z_test).
///
/// Although teoretically we cannot do it, in practice we know that the t-tests are
/// robust to violations on it's assumptions.
///
#[bon::builder]
pub fn t_test(
    data: &mut Samples,
    #[builder(default)] hypothesys: Hypothesis,
    #[builder(default)] null_mean: f64,
    significance: Option<f64>,
) -> Result<TestResult, TestError> {
    let len: usize = data.count();
    if len < 2 {
        return Err(TestError::NotEnoughSamples);
    }
    // all following `unwrap` are safe because the length is at least 2

    let mean: f64 = data.mean().unwrap();
    let sample_std_dev: f64 = data.variance().unwrap().sqrt();

    let t: f64 = (mean - null_mean) * (len as f64).sqrt() / sample_std_dev;
    let t_distr: StudentT = StudentT::new(len as f64 - 1.0).unwrap();

    let p: f64 = t_distr.p_value(hypothesys, t);

    let ret: TestResult = if let Some(alpha) = significance {
        #[allow(clippy::nonminimal_bool)]
        if !alpha.is_finite() || !(0.0 < alpha && alpha < 1.0) {
            return Err(TestError::InvalidSignificance);
        }
        // compute confidence interval:
        // knowing that the statistic is the mean,
        let std_err: f64 = sample_std_dev / (len as f64).sqrt();
        let precentile: f64 = if hypothesys == Hypothesis::TwoTailed {
            t_distr.quantile(1.0 - 0.5 * alpha)
        } else {
            t_distr.quantile(1.0 - alpha)
        };

        let confidence_interval: (f64, f64) =
            (mean - precentile * std_err, mean + precentile * std_err);

        TestResult::PValueCI(t, p, confidence_interval)
    } else {
        TestResult::PValue(t, p)
    };

    return Ok(ret);
}

/// Performs a two sample location [t-test](https://en.wikipedia.org/wiki/Student%27s_t-test#Two-sample_t-tests)
/// for the mean. We have 2 datasets `data_a` and `data_b` and we wish to determine if
/// their means are different or not. The null hypotesys assumes that the 2 means
/// are equal (there is no difference).
///
/// ## Handling differnet variances or sample size
///
/// This function will ajust to the given datasets and perform different computations:
///  - [Equal sample sizes and variance](https://en.wikipedia.org/wiki/Student%27s_t-test#Equal_sample_sizes_and_variance)
///  - [Equal or unequal sample sizes, similar variances](https://en.wikipedia.org/wiki/Student%27s_t-test#Equal_or_unequal_sample_sizes,_similar_variances_(%E2%81%A01/2%E2%81%A0_%3C_%E2%81%A0sX1/sX2%E2%81%A0_%3C_2))
///  - [Equal or unequal sample sizes, unequal variances](https://en.wikipedia.org/wiki/Student%27s_t-test#Equal_or_unequal_sample_sizes,_unequal_variances_(sX1_%3E_2sX2_or_sX2_%3E_2sX1))
///      - Known as [Welch's t-test](https://en.wikipedia.org/wiki/Welch%27s_t-test)
///
/// Variances are considered similar iff `1/2 <= var(a)/var(b) <= 2.0`, where `var(x)`
/// represents the unbiased sample variance of the dataset x.
///
/// ## Assumptions of the test
///
/// 1. [IID samples](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)
/// 2. Assumes that the null distribution of the mean can be aproximated
///     with a [Normal](crate::distributions::Normal) distribution.
///      - This can be assumed trough the [CLT](https://en.wikipedia.org/wiki/Central_limit_theorem)
///         if it applies (by having **many** samples).
///      - Can also be assumed if it is known that the samples are drawn
///             from a [Normal](crate::distributions::Normal) distribution.
///
/// If the conditions for the test are **not** fullfilled, then the result is meaningless.
///
/// ## Inputs:
/// 1. `data_a`: The samples collected for group A.
/// 2. `data_b`: The samples collected for group B.
/// 3. `hypothesys`: determines if a 2-tailed/left-tailed/right-tailed will be used
///      - The default is a 2 tailed test.
/// 4. `significance`: (optional) If left empty, only the P-value will be computed.
///     Otherwise, a confidence interval with the given significance level (alpha)
///     will be computed.
///      - It needs to be a valid probability (`0 < significance < 1`, tipically 0.05 or less)
///      - (The P-value is always computed)
/// 5. `use_welch`: (optional) If set to `true`, will always use the [Welch's t-test](https://en.wikipedia.org/wiki/Welch%27s_t-test),
///     wich accounts for different amount of samples and different variance.
///
/// ## Results
///
/// If the test is performed correcly, it returns a [TestResult] with the P value
/// and the confidence interval if `significance` was provided.
///
/// If there is not enough samples in `data`, returns [TestError::NotEnoughSamples].
///
#[bon::builder]
pub fn two_sample_t_test(
    data_a: &mut Samples,
    data_b: &mut Samples,
    #[builder(default)] hypothesys: Hypothesis,
    significance: Option<f64>,
    #[builder(default)] use_welch: bool,
) -> Result<TestResult, TestError> {
    let n_a: f64 = data_a.count() as f64;
    let n_b: f64 = data_b.count() as f64;

    if n_a < 2.0 || n_b < 2.0 {
        return Err(TestError::NotEnoughSamples);
    }
    // All the following unwraps are safe since the dataset has at least 2 samples

    let mean_diff: f64 = data_a.mean().unwrap() - data_b.mean().unwrap();

    let var_a: f64 = data_a.variance().unwrap();
    let var_b: f64 = data_b.variance().unwrap();
    let similar_var: bool = {
        let var_ratio: f64 = var_a / var_b;
        (0.5..=2.0).contains(&var_ratio)
    };

    #[allow(clippy::float_cmp)]
    if n_a == n_b && similar_var && !use_welch {
        // Equal sample sizes and variance
        // https://en.wikipedia.org/wiki/Student%27s_t-test#Equal_sample_sizes_and_variance

        let n: f64 = n_a;

        let s_pool: f64 = ((var_a + var_b) * 0.5).sqrt();

        let t: f64 = mean_diff / (s_pool * (2.0 / n).sqrt());

        let degrees_of_freedom: f64 = 2.0 * (n - 1.0);

        // SAFETY: as the assert indicates, the degrees of freedom are always greater or equal to 2
        #[allow(clippy::multiple_unsafe_ops_per_block)]
        let null_student_t: StudentT = unsafe {
            // safe because of the cond in the assert is always fullfilled
            assert_unchecked(2.0 <= degrees_of_freedom);
            StudentT::new_unchecked(degrees_of_freedom)
        };

        let p: f64 = null_student_t.p_value(hypothesys, t);
        let ret: TestResult = if let Some(alpha) = significance {
            #[allow(clippy::nonminimal_bool)]
            if !alpha.is_finite() || !(0.0 < alpha && alpha < 1.0) {
                return Err(TestError::InvalidSignificance);
            }
            let confidence_interval: (f64, f64) =
                null_student_t.confidence_interval(hypothesys, alpha);
            TestResult::PValueCI(t, p, confidence_interval)
        } else {
            TestResult::PValue(t, p)
        };

        return Ok(ret);
    }

    if similar_var && !use_welch {
        // Equal or unequal sample sizes, similar variances
        // https://en.wikipedia.org/wiki/Student%27s_t-test#Equal_or_unequal_sample_sizes,_similar_variances_(%E2%81%A01/2%E2%81%A0_%3C_%E2%81%A0sX1/sX2%E2%81%A0_%3C_2)

        let degrees_of_freedom: f64 = n_a + n_b - 2.0;

        let s_pool: f64 = (((n_a - 1.0) * var_a + (n_b - 1.0) * var_b) / degrees_of_freedom).sqrt();
        let t: f64 = mean_diff / (s_pool * (1.0 / n_a + 1.0 / n_b).sqrt());

        // SAFETY: as the assert indicates, the degrees of freedom are always greater or equal to 2
        #[allow(clippy::multiple_unsafe_ops_per_block)]
        let null_student_t: StudentT = unsafe {
            // safe because of the cond in the assert is always fullfilled
            assert_unchecked(2.0 <= degrees_of_freedom);
            StudentT::new_unchecked(degrees_of_freedom)
        };

        let p: f64 = null_student_t.p_value(hypothesys, t);
        let ret: TestResult = if let Some(alpha) = significance {
            #[allow(clippy::nonminimal_bool)]
            if !alpha.is_finite() || !(0.0 < alpha && alpha < 1.0) {
                return Err(TestError::InvalidSignificance);
            }
            let confidence_interval: (f64, f64) =
                null_student_t.confidence_interval(hypothesys, alpha);
            TestResult::PValueCI(t, p, confidence_interval)
        } else {
            TestResult::PValue(t, p)
        };

        return Ok(ret);
    }

    // Welch's t-test
    // https://en.wikipedia.org/wiki/Welch%27s_t-test

    let aux_a: f64 = var_a / n_a;
    let aux_b: f64 = var_b / n_b;
    let aux: f64 = aux_a + var_b / n_b;

    //corrected sample standard deviation
    let s_corr: f64 = aux.sqrt();
    let t: f64 = mean_diff / s_corr;

    let degrees_of_freedom: f64 = {
        let num: f64 = aux * aux;
        let den_1: f64 = aux_a * aux_a / (n_a - 1.0);
        let den_2: f64 = aux_b * aux_b / (n_a - 1.0);
        num / (den_1 + den_2)
    };

    // SAFETY: as the assert indicates, the degrees of freedom are always greater or equal to 2
    #[allow(clippy::multiple_unsafe_ops_per_block)]
    let null_student_t: StudentT = unsafe {
        // safe because of the cond in the assert is always fullfilled
        assert_unchecked(2.0 <= degrees_of_freedom);
        StudentT::new_unchecked(degrees_of_freedom)
    };

    let p: f64 = null_student_t.p_value(hypothesys, t);
    let ret: TestResult = if let Some(alpha) = significance {
        #[allow(clippy::nonminimal_bool)]
        if !alpha.is_finite() || !(0.0 < alpha && alpha < 1.0) {
            return Err(TestError::InvalidSignificance);
        }
        let confidence_interval: (f64, f64) = null_student_t.confidence_interval(hypothesys, alpha);
        TestResult::PValueCI(t, p, confidence_interval)
    } else {
        TestResult::PValue(t, p)
    };

    return Ok(ret);
}

/// Performs a paired [t-test](https://en.wikipedia.org/wiki/Student%27s_t-test) for the mean.
/// Can be used to determine if a there is a shift from the first observation to the second.
///
/// To do this, we compute a difference dataset where each sample is the difference between
/// each pair of samples.
///
/// ## Assumptions of the test
///
/// 1. [IID samples](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)
/// 2. Assumes that the null distribution of the mean of the *difference dataset* can
/// be aproximated with a [Normal](crate::distributions::Normal) distribution.
///      - This can be assumed trough the [CLT](https://en.wikipedia.org/wiki/Central_limit_theorem)
///     if it applies.
///      - Can also be assumed if it is known that the samples of the difference
///         distribution are drawn from a [Normal](crate::distributions::Normal) distribution.
///          - Note that this does not mean that the original samples need to come
///             form a normal distribution, only their difference.
///
/// ## Inputs:
///
/// 1. `data_pre`: the samples collected to perform the test before
/// the treatment (gropu A).
/// 2. `data_post`: the samples collected to perform the test after
/// the treatment (gropu B).
/// 3. `hypothesys`: determines if a 2-tailed/left-tailed/right-tailed will be used
///      - The default is a 2 tailed test.
/// 4. `significance`: (optional) If left empty, only the P-value will be computed.
///     Otherwise, a confidence interval with the given significance level (alpha)
///     will be computed.
///      - It needs to be a valid probability (`0 < significance < 1`, tipically 0.05 or less)
///      - (The P-value is always computed)
///
/// If the conditions for the test are **not** fullfilled, then the result is meaningless.
///
/// ## Results
///
/// If the test is performed correcly, it returns a [TestResult] with the P value
/// and the confidence interval if `significance` was provided.
///
/// If there are an unequal number of sampes, returns [TestError::InvalidArguments].
/// If there is not enough samples in `data`, returns [TestError::NotEnoughSamples].
///
#[bon::builder]
pub fn paired_t_test(
    data_pre: &Samples,
    data_post: &Samples,
    #[builder(default)] hypothesys: Hypothesis,
    significance: Option<f64>,
) -> Result<TestResult, TestError> {
    let n_pre: usize = data_pre.count();
    let n_post: usize = data_post.count();

    if n_pre != n_post {
        return Err(TestError::InvalidArguments);
    }
    let n: usize = n_pre;

    let mut difference_dataset_values: Vec<f64> = Vec::new();
    difference_dataset_values.reserve_exact(n);

    for (pre, post) in data_pre
        .peek_data()
        .iter()
        .zip(data_post.peek_data().iter())
    {
        difference_dataset_values.push(*post - *pre);
    }

    let mut difference_dataset: Samples = match Samples::new_move(difference_dataset_values) {
        Ok(v) => v,
        Err(_) => return Err(TestError::NanErr),
    };

    //return t_test(&mut difference_dataset, hypothesys, 0.0);
    return t_test()
        .data(&mut difference_dataset)
        .hypothesys(hypothesys)
        .null_mean(0.0)
        .maybe_significance(significance)
        .call();
}

mod simulation_study {
    //! This mudule is dedicated to performing simulation studies for the different tests.
    //!

    use std::ptr::null;

    use crate::{distribution_trait::Distribution, errors::SimulationError};

    /// The default number of repetitions for simulation studies.
    pub const DEFAULT_NUM_REPETITIONS: usize = 128;

    pub enum SimulationResult {
        SignificanceLevel(f64),
        Power(f64),
        SampleSize(usize),
        Feasibility(bool),
        Error(SimulationError),
    }

    /// Performs a simulation strudy for the [one sample t-test](hypothesis::t_test).
    ///
    /// It will comupte the optional variable wich is not set.
    ///
    /// ## Inputs
    ///
    ///  - `null_distribution`: The distribution of the samples under the null hypothesys.
    ///  - `alternative_distribution`: The distribution of the samples under the alternative hypothesys.
    ///  - `significance_level`: The [significance level](https://en.wikipedia.org/wiki/Statistical_significance) of the test.
    ///      - (Optional)
    ///      - If it is **not** set, then the function will compute it.
    ///  - `power`: The [statistical power](https://en.wikipedia.org/wiki/Power_(statistics)) of the test.
    ///      - (Optional)
    ///      - If it is **not** set, then the function will compute it.
    ///  - `sample_size`: the number of samples used in the test.
    ///      - (Optional)
    ///      - If it is **not** set, then the function will compute it.
    ///  - `number_of_repetitions`: the number of times this experiment will be repeated.
    ///      - (Optional)
    ///      - Deafult is [DEFAULT_NUM_REPETITIONS].
    ///
    /// In the case where all 3 of `significance_level`, `power` and `sample_size` are set,
    /// then it will compute if these results are achievable. The result will be on
    /// `SimulationResult::Feasibility`. If it is true, it means that the statistical test
    /// has the desired propreties or better.
    ///
    ///
    ///
    #[bon::builder]
    pub fn simulation_t_test(
        null_distribution: &dyn Distribution,
        alternative_distribution: &dyn Distribution,
        significance_level: Option<f64>,
        power: Option<f64>,
        sample_size: Option<usize>,
        number_of_repetitions: Option<usize>,
    ) -> SimulationResult {
        let n_repetitions: usize = number_of_repetitions.unwrap_or(DEFAULT_NUM_REPETITIONS);

        let alpha: f64 = if let Some(a) = significance_level {
            if a.is_nan() || !(0.0 < a && a < 1.0) {
                return SimulationResult::Error(SimulationError::InvalidSignificanceLevel);
            }
            a
        } else {
            // solving for alpha

            let powr: f64 = if let Some(p) = power {
                if p.is_nan() || !(0.0 < p && p < 1.0) {
                    return SimulationResult::Error(SimulationError::InvalidPower);
                }
                p
            } else {
                return SimulationResult::Error(SimulationError::MissingArguments);
            };

            let n_samples: usize = if let Some(n) = sample_size {
                n
                // maybe 0
            } else {
                return SimulationResult::Error(SimulationError::MissingArguments);
            };

            return significance_level_t_test()
                .null_distribution(null_distribution)
                .alternative_distribution(alternative_distribution)
                .power(powr)
                .sample_size(n_samples)
                .number_of_repetitions(n_repetitions)
                .call();
        };

        let powr: f64 = if let Some(p) = power {
            p
        } else {
            let n_samples: usize = if let Some(n) = sample_size {
                n
            } else {
                return SimulationResult::Error(SimulationError::MissingArguments);
            };

            return power_t_test()
                .null_distribution(null_distribution)
                .alternative_distribution(alternative_distribution)
                .significance_level(alpha)
                .sample_size(n_samples)
                .number_of_repetitions(n_repetitions)
                .call();
        };

        let n_samples: usize = if let Some(n) = sample_size {
            n
        } else {
            return sample_size_t_test()
                .null_distribution(null_distribution)
                .alternative_distribution(alternative_distribution)
                .significance_level(alpha)
                .power(powr)
                .number_of_repetitions(n_repetitions)
                .call();
        };

        return feasibility_t_test()
            .null_distribution(null_distribution)
            .alternative_distribution(alternative_distribution)
            .significance_level(alpha)
            .power(powr)
            .sample_size(n_samples)
            .number_of_repetitions(n_repetitions)
            .call();
    }

    #[bon::builder]
    fn significance_level_t_test(
        null_distribution: &dyn Distribution,
        alternative_distribution: &dyn Distribution,
        power: f64,
        sample_size: usize,
        number_of_repetitions: usize,
    ) -> SimulationResult {
        todo!();
    }

    #[bon::builder]
    fn power_t_test(
        null_distribution: &dyn Distribution,
        alternative_distribution: &dyn Distribution,
        significance_level: f64,
        sample_size: usize,
        number_of_repetitions: usize,
    ) -> SimulationResult {
        todo!();
    }

    #[bon::builder]
    fn sample_size_t_test(
        null_distribution: &dyn Distribution,
        alternative_distribution: &dyn Distribution,
        significance_level: f64,
        power: f64,
        number_of_repetitions: usize,
    ) -> SimulationResult {
        todo!();
    }

    #[bon::builder]
    fn feasibility_t_test(
        null_distribution: &dyn Distribution,
        alternative_distribution: &dyn Distribution,
        significance_level: f64,
        power: f64,
        sample_size: usize,
        number_of_repetitions: usize,
    ) -> SimulationResult {
        todo!();
    }
}
