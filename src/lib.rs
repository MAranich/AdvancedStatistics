#![allow(
    non_snake_case,
    clippy::needless_return,
    clippy::assign_op_pattern,
    clippy::excessive_precision
)]

#![warn(
    clippy::all,
    clippy::restriction,
    clippy::pedantic,
    clippy::nursery,
    clippy::cargo,
)]
// ^Disable warning "crate `AdvancedStatistics` should have a snake case name convert the identifier to snake case: `advanced_statistics`"
// The rest of the names will follow the snake_case convention.

//! # Advanced Statistics
//!
//!
//! This library is a statistics library that provides:
//!
//! - [x] Interface to create distributions
//! - [ ] Common distributions (ready to be used in any calculation)
//! - [x] Data computation
//! - [ ] Multiple tests (hypothesys testing)
//! - [ ] Computing P values and confidence intervals
//! - [x] Parameter estimation
//! - [x] Outlier detection
//! - [ ] Resampling methods
//! - [ ] Multidimensional data support
//! - [ ] ANOVA / MANOVA
//! - [ ] Bayesian statistics
//! - [x] Updated to rust 2024 version
//!
//! This is an open source project: [github](https://github.com/MAranich/AdvancedStatistics).
//!
//! ## Distributions
//!
//! We have defined the trait [Distribution] and
//! [DiscreteDistribution](distribution_trait::DiscreteDistribution) that define a
//! basic trait (interface) to work with distributions. The only requiered methods
//! to implement are:
//!  - [pdf](distribution_trait::Distribution::pdf): the pdf of the distribution.
//!  - [get_domain](distribution_trait::Distribution::get_domain): the [domain]
//!     of the pdf of the distribution.
//!
//! After this, a wide array of funcions are avaliable (check
//! [distribution_trait::Distribution] for more details). Note that this deafult
//! implementations can be computationally costly, therefore we recommend
//! implementing the other methods if there is an avaliable analytical solution
//! for them.
//!
//! But if you are interested on a more common distribution, you may find it among the
//! ones that we have already implemented:
//!
//! ### Continuous distributions:
//!
//!  - [x] [Normal distribution](crate::distributions::Normal) ([Wiki](https://en.wikipedia.org/wiki/Normal_distribution))
//!  - [x] [Uniform distribution](crate::distributions::Exponential) ([Wiki](https://en.wikipedia.org/wiki/Continuous_uniform_distribution))
//!  - [x] [Exponential](crate::distributions::Exponential) ([Wiki](https://en.wikipedia.org/wiki/Exponential_distribution))
//!  - [x] [Gamma distribution](crate::distributions::Gamma) ([Wiki](https://en.wikipedia.org/wiki/Gamma_distribution))
//!  - [x] [Chi-squared distribution](crate::distributions::ChiSquared) ([Wiki](https://en.wikipedia.org/wiki/Chi-squared_distribution))
//!  - [ ] [Chi distribution]() ([Wiki](https://en.wikipedia.org/wiki/Chi_distribution))
//!  - [x] [F distribution](crate::distributions::F) ([Wiki](https://en.wikipedia.org/wiki/F-distribution))
//!  - [x] [Beta distribution](crate::distributions::Beta) ([Wiki](https://en.wikipedia.org/wiki/Beta_distribution))
//!  - [x] [Student's T distribution](crate::distributions::StudentT) ([Wiki](https://en.wikipedia.org/wiki/Student%27s_t-distribution))
//!  - [ ] [Hypergeometric]() ([Wiki](https://en.wikipedia.org/wiki/Hypergeometric_distribution))
//!  - [ ] [Beta prime distribution]() ([Wiki](https://en.wikipedia.org/wiki/Beta_prime_distribution))
//!  - [ ] [Pareto distribution]() ([Wiki](https://en.wikipedia.org/wiki/Pareto_distribution))
//!  - [ ] [Cauchy distribution]() ([Wiki](https://en.wikipedia.org/wiki/Cauchy_distribution))
//!  - [ ] [Laplace distribution]() ([Wiki](https://en.wikipedia.org/wiki/Laplace_distribution))
//!  - [ ] [Logistic distribution]() ([Wiki](https://en.wikipedia.org/wiki/Logistic_distribution))
//!  - [ ] [Quantile parameterized distribution]() ([Wiki](https://en.wikipedia.org/wiki/Quantile-parameterized_distribution)) (?)
//!  - [x] [Dirac distribution](distributions::Dirac) ([Wiki](https://en.wikipedia.org/wiki/Dirac_delta_function))
//!  - [ ] ... (more to come (?))
//!
//! ### Discrete distributions:
//!
//!  - [x] [Bernoulli](distributions::Bernoulli) ([Wiki](https://en.wikipedia.org/wiki/Bernoulli_distribution))
//!  - [x] [Binomial](distributions::Binomial) ([Wiki](https://en.wikipedia.org/wiki/Binomial_distribution))
//!  - [x] [Discrete Uniform](distributions::DiscreteUniform) ([Wiki](https://en.wikipedia.org/wiki/Discrete_uniform_distribution))
//!  - [x] [Geometric distribution](distributions::Geometric) ([Wiki](https://en.wikipedia.org/wiki/Geometric_distribution))
//!  - [ ] [Negative binomial distribution]() ([Wiki](https://en.wikipedia.org/wiki/Negative_binomial_distribution))
//!  - [x] [Poisson distribution](distributions::Poisson)([Wiki](https://en.wikipedia.org/wiki/Poisson_distribution))
//!  - [x] [Degenerate distribution](distributions::Degenerate) ([Wiki](https://en.wikipedia.org/wiki/Degenerate_distribution)) (Continuous version is [Dirac](distributions::Dirac))
//!  - [ ] ... (more to come (?))
//!
//!
//! ## Tests
//!
//! We have provided some structures to simplify the computation of tests.
//! Collected data can be stored in [Samples](Samples::Samples), wich also
//! allows effitient computation of common statistics (mean, variance, median...).
//!
//! We also provided an implemetation of common tests:
//!
//!  - [x] [Z-test](hypothesis::z_test) ([Wiki](https://en.wikipedia.org/wiki/Z-test))
//!  - [x] [t-test](hypothesis::t_test) ([Wiki](https://en.wikipedia.org/wiki/Student%27s_t-test#One-sample_t-test))
//!  - [x] [2 sample t-test](hypothesis::two_sample_t_test) ([Wiki](https://en.wikipedia.org/wiki/Student%27s_t-test#Two-sample_t-tests))
//!  - [x] [Paired t-test](hypothesis::paired_t_test) ([Wiki](https://en.wikipedia.org/wiki/Student%27s_t-test))
//!  - [ ] [F-test](hypothesis::test_function) ([Wiki](https://en.wikipedia.org/wiki/F-test))
//!  - [ ] [Sign test](hypothesis::test_function) ([Wiki](https://en.wikipedia.org/wiki/Sign_test))
//!  - [ ] [Wilcoxon signed-rank test](hypothesis::test_function) ([Wiki](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test))
//!  - [ ] [Chi-squared test](hypothesis::test_function) ([Wiki](https://en.wikipedia.org/wiki/Chi-squared_test))
//!  - [ ] [Pearson's chi-squared test](hypothesis::test_function) ([Wiki](https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test))
//!  - [ ] ... (more to come (?))
//!
//!
//! ***
//!

use distribution_trait::Distribution;

pub mod configuration;
pub mod distribution_trait;
pub mod distributions;
pub mod domain;
pub mod errors;
pub mod euclid;
pub mod hypothesis;
pub mod mixed_distribution;
pub mod samples;

// Todo: Do this struct that allows to shift the location of a distribution
/// This Structure is not implemented yet. It is currenly useless.
///
/// In a future it will allow to "move" a Distribution.
/// For example, the pdf of an Exponential always starts at 0, but with this
/// Structure you can make it start at any value you desire while maintaining
/// the same interface.
pub struct Loation<D>
where
    D: Distribution,
{
    /// The main distribution to be modified
    pub distribution: D,
    /// Indicates wich value is now the `0.0` of the distribution
    pub location: f64,
    /// Flips the distribution (horizontally). The flip is done after the location shift.
    pub flipped: bool,
}
