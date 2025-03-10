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
//!  
//!
//!
//!

use crate::{Samples::Samples, distribution_trait::Distribution};

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
/// with the given `data` and `hypotesys`.
///
/// ## Assumptions
///
/// 1. Assumes that the null distribution of the test statistic
/// (the mean) can be aproximated with a
/// [Normal](crate::distributions::Normal) distribution.
///     - This can be assumed trough the [CLT](https://en.wikipedia.org/wiki/Central_limit_theorem)
///     if it applies.
/// 2. The mean and standard deviation of the null distribution are known
///      - (Or estimated with high accuracy)
///
/// ## Inputs:
/// 1. `data`: all the samples collected to perform the test.
///   - **Panics** if `data` contains no samples.
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
pub fn z_test(
    data: &mut Samples,
    hypothesys: Hypothesis,
    null: crate::distributions::Normal::Normal,
) -> f64 {
    let sample_mean: f64 = match data.mean() {
        Some(m) => m,
        None => panic!("Attempted a hypotesys::z_test_mean with an empty dataset. \n"),
    };

    return null.p_value(hypothesys, sample_mean);
}

/// Performs a [Z-test](https://en.wikipedia.org/wiki/Z-test) for a general `statistic`
/// with the given `data` and `hypotesys`.
///
/// ## Assumptions
///
/// 1. Assumes that the null distribution of the test statistic
/// can be aproximated with a [Normal](crate::distributions::Normal) distribution.
///     - This can be assumed trough the [CLT](https://en.wikipedia.org/wiki/Central_limit_theorem)
///     if it applies.
/// 2. The mean and standard deviation of the null distribution are known
///      - (Or estimated with high accuracy)
///
/// ## Inputs:
/// 1. `data`: all the samples collected to perform the test.
///      - **Panics** if `data` contains no samples.
/// 2. `hypothesys`: determines if a 2-tailed/left-tailed/right-tailed will be used
/// 3. `null_mean_statistic`: the mean of the null distribution.
/// 4. `std_dev`: the *known* standard deviation of the null distribution.
///      - **Panics** if it is negative, `+-inf` or NaN.
/// 5. `statistic`: the found statistic.
///
/// ## Results
///
/// Returns the [P value](https://en.wikipedia.org/wiki/P-value).
///  - If the P value is **very small** (for example `P < 0.01`), the null hypothesys
/// can be immidiately rejected.
///  - If the P value is **very large** (for example `0.1 < p`), the null hypothesys
/// cannot be rejected.
///
pub fn z_test_general(
    hypothesys: Hypothesis,
    null_mean_statistic: f64,
    std_dev: f64,
    statistic: f64,
) -> f64 {
    let null: crate::distributions::Normal::Normal = match crate::distributions::Normal::Normal::new(
        null_mean_statistic,
        std_dev,
    ) {
        Ok(d) => d,
        Err(_) => panic!(
            "Attempted a hypotesys::z_test_mean with a invalif value for `std_dev`. It is either not stricly positive, `+-inf` or a NaN. \n"
        ),
    };

    return null.p_value(hypothesys, statistic);
}
