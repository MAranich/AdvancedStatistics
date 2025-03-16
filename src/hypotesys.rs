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
//! hypothesys (known as null distribution)
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
//! as extreme or more than the one obtained.
//!  - The value for alpha that makes the statistic lay on the boundary of the
//! confidence interval.
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
    Samples::Samples, distribution_trait::Distribution, distributions::StudentT::StudentT,
    errors::TestError,
};

/// Defines Wich kind of test are we doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
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

/// Performs a [Z-test](https://en.wikipedia.org/wiki/Z-test) **for the mean**
/// with the given `data` and `hypotesys`. This test can be used to test 
///
/// ## Assumptions of the test
///
/// 1. [IID samples](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)
/// 2. Assumes that the null distribution of the test statistic
/// (the mean) can be aproximated with a
/// [Normal](crate::distributions::Normal) distribution.
///      - This can be assumed trough the [CLT](https://en.wikipedia.org/wiki/Central_limit_theorem)
///     if it applies.
///      - Can also be assumed if it is known that the samples are drawn
///         from a [Normal](crate::distributions::Normal) distribution.
/// 3. The mean and standard deviation of the null distribution are known
///      - (Or estimated with high accuracy)
///
/// ## Inputs:
/// 1. `data`: all the samples collected to perform the test.
/// 2. `hypothesys`: determines if a 2-tailed/left-tailed/right-tailed will be used
/// 3. `null`: the [null distribution](https://en.wikipedia.org/wiki/Null_distribution)
/// of the mean.
///      - Contains the null hypothesys mean
///      - Contains the **known** standard deviation of the null distribution.
///
/// ## Results
///
/// Returns the [P value](https://en.wikipedia.org/wiki/P-value).
///  - If the P value is **very small** (for example `P < 0.01`), the null hypothesys
/// can be immidiately rejected.
///  - If the P value is **very large** (for example `0.1 < p`), the null hypothesys
/// cannot be rejected.
/// 
/// If there is not enough samples in `data`, returns [TestError::NotEnoughSamples].
///
pub fn z_test(
    data: &mut Samples,
    hypothesys: Hypothesis,
    null: crate::distributions::Normal::Normal,
) -> Result<f64, TestError> {
    let sample_mean: f64 = match data.mean() {
        Some(m) => m,
        None => return Err(TestError::NotEnoughSamples),
    };

    let p: f64 = null.p_value(hypothesys, sample_mean); 
    return Ok(p);
}

/// Performs a general test and returns the probability (P value) of `statistic` being
/// drawn from the `null` distribution. 
pub fn general_test<T: crate::distribution_trait::Distribution>(
    hypothesys: Hypothesis,
    statistic: f64,
    null: T,
) -> f64 {
    return null.p_value(hypothesys, statistic);
}

/// Performs a one sample [t-test](https://en.wikipedia.org/wiki/Z-test) for the mean. 
/// Can be used to determine if a the mean of the data is different to the one form 
/// the null distribution. 
///
/// ## Assumptions of the test
///
/// 1. [IID samples](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)
/// 2. Assumes that the null distribution of the mean can be aproximated
/// with a [Normal](crate::distributions::Normal) distribution.
///      - This can be assumed trough the [CLT](https://en.wikipedia.org/wiki/Central_limit_theorem)
///     if it applies.
///      - Can also be assumed if it is known that the samples are drawn
///         from a [Normal](crate::distributions::Normal) distribution.
/// 3. The mean of the null distribution are known
///      - (Or estimated with high accuracy)
///
/// ## Inputs:
/// 1. `data`: all the samples collected to perform the test.
/// 2. `hypothesys`: determines if a 2-tailed/left-tailed/right-tailed will be used
/// 3. `null_mean`: the mean of the null distribution.
///
/// ## Results
///
/// Returns the [P value](https://en.wikipedia.org/wiki/P-value).
///  - If the P value is **very small** (for example `P < 0.01`), the null hypothesys
/// can be immidiately rejected.
///  - If the P value is **very large** (for example `0.1 < p`), the null hypothesys
/// cannot be rejected.
///
/// If there is not enough samples in `data`, returns [TestError::NotEnoughSamples].
///
pub fn t_test(
    data: &mut Samples,
    hypothesys: Hypothesis,
    null_mean: f64,
) -> Result<f64, TestError> {
    let len: usize = data.count();
    if len < 2 {
        return Err(TestError::NotEnoughSamples);
    }

    let mean: f64 = data.mean().unwrap();
    let sample_std_dev: f64 = data.variance().unwrap().sqrt();

    let t: f64 = (mean - null_mean) * (len as f64).sqrt() / sample_std_dev;
    let t_distr: StudentT = StudentT::new(len as f64 - 1.0).unwrap();

    let p: f64 = t_distr.p_value(hypothesys, t);

    return Ok(p);
}

/// Performs a two sample location [t-test](https://en.wikipedia.org/wiki/Student%27s_t-test#Two-sample_t-tests)
/// for the mean. We have 2 datasets `data_a` and `data_b` and we wish to determine if
/// their means are different or not. The null hypotesys assumes that the 2 means
/// are equal (there is no difference).
///
/// This function will ajust to the given datasets and perform different computations:
///  - [Equal sample sizes and variance](https://en.wikipedia.org/wiki/Student%27s_t-test#Equal_sample_sizes_and_variance)
///  - [Equal or unequal sample sizes, similar variances](https://en.wikipedia.org/wiki/Student%27s_t-test#Equal_or_unequal_sample_sizes,_similar_variances_(%E2%81%A01/2%E2%81%A0_%3C_%E2%81%A0sX1/sX2%E2%81%A0_%3C_2))
///  - [Equal or unequal sample sizes, unequal variances](https://en.wikipedia.org/wiki/Student%27s_t-test#Equal_or_unequal_sample_sizes,_unequal_variances_(sX1_%3E_2sX2_or_sX2_%3E_2sX1))
///      - Known as [Welch's t-test](https://en.wikipedia.org/wiki/Welch%27s_t-test)
///
/// Variances are considered similar iff `1/2 <= var(a)/var(b) <= 2.0`, where `var(x)`
/// is the unbiased sample variance of the dataset x.
///
/// ## Assumptions of the test
///
/// 1. [IID samples](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)
/// 2. Assumes that the null distribution of the mean can be aproximated
/// with a [Normal](crate::distributions::Normal) distribution.
///      - This can be assumed trough the [CLT](https://en.wikipedia.org/wiki/Central_limit_theorem)
///     if it applies (by having **many** samples).
///      - Can also be assumed if it is known that the samples are drawn
///         from a [Normal](crate::distributions::Normal) distribution.
///
/// ## Inputs:
/// 1. `data_a`: The samples collected for group A.
/// 2. `data_b`: The samples collected for group b.
/// 3. `hypothesys`: determines if a 2-tailed/left-tailed/right-tailed will be used
///
/// ## Results
///
/// Returns the [P value](https://en.wikipedia.org/wiki/P-value).
///  - If the P value is **very small** (for example `P < 0.01`), the null hypothesys
/// can be immidiately rejected.
///  - If the P value is **very large** (for example `0.1 < p`), the null hypothesys
/// cannot be rejected.
///
/// If there is not enough samples in `data`, returns [TestError::NotEnoughSamples].
///
pub fn two_sample_t_test(
    data_a: &mut Samples,
    data_b: &mut Samples,
    hypothesys: Hypothesis,
) -> Result<f64, TestError> {
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
        0.5 <= var_ratio && var_ratio <= 2.0
    };

    if n_a == n_b && similar_var {
        // Equal sample sizes and variance
        // https://en.wikipedia.org/wiki/Student%27s_t-test#Equal_sample_sizes_and_variance

        let n: f64 = n_a;

        let s_pool: f64 = ((var_a + var_b) * 0.5).sqrt();

        let t: f64 = mean_diff / (s_pool * (2.0 / n).sqrt());

        let degrees_of_freedom: f64 = 2.0 * (n - 1.0);
        let null_student_t: StudentT = unsafe {
            // safe because of the cond in the assert is always fullfilled
            assert_unchecked(2.0 <= degrees_of_freedom);
            StudentT::new_unchecked(degrees_of_freedom)
        };

        let p: f64 = null_student_t.p_value(hypothesys, t);
        return Ok(p);
    }

    if similar_var {
        // Equal or unequal sample sizes, similar variances
        // https://en.wikipedia.org/wiki/Student%27s_t-test#Equal_or_unequal_sample_sizes,_similar_variances_(%E2%81%A01/2%E2%81%A0_%3C_%E2%81%A0sX1/sX2%E2%81%A0_%3C_2)

        let degrees_of_freedom: f64 = n_a + n_b - 2.0;

        let s_pool: f64 = (((n_a - 1.0) * var_a + (n_b - 1.0) * var_b) / degrees_of_freedom).sqrt();
        let t: f64 = mean_diff / (s_pool * (1.0 / n_a + 1.0 / n_b).sqrt());

        let null_student_t: StudentT = unsafe {
            // safe because of the cond in the assert is always fullfilled
            assert_unchecked(2.0 <= degrees_of_freedom);
            StudentT::new_unchecked(degrees_of_freedom)
        };

        let p: f64 = null_student_t.p_value(hypothesys, t);
        return Ok(p);
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

    let null_student_t: StudentT = unsafe {
        // safe because of the cond in the assert is always fullfilled
        assert_unchecked(2.0 <= degrees_of_freedom);
        StudentT::new_unchecked(degrees_of_freedom)
    };

    let p: f64 = null_student_t.p_value(hypothesys, t);
    return Ok(p);
}
