#![allow(non_snake_case)]
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
//! - [ ] Outlier detection
//! - [ ] Resampling methods
//! - [ ] Multidimensional data support
//! - [ ] ANOVA / MANOVA
//! - [ ] Bayesian statistics
//!
//! ## Distributions
//!
//! We have defined the trait [Distribution] and
//! [DiscreteDistribution](distribution_trait::DiscreteDistribution) that define a
//! basic trait (interface) to work with distributions. The only requiered methods
//! to implement are:
//!  - [pdf](distribution_trait::Distribution::pdf): the pdf of the distribution.
//!  - [get_domain](distribution_trait::Distribution::get_domain): the [domain]
//! of the pdf of the distribution.
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
//!  - [ ] [Chi squared distribution](https://en.wikipedia.org/wiki/Chi-squared_distribution)
//!  - [ ] [Chi distribution](https://en.wikipedia.org/wiki/Chi_distribution)
//!  - [ ] [F distribution](https://en.wikipedia.org/wiki/F-distribution)
//!  - [ ] [Gamma distribution](crate::distributions::Gamma) ([Wiki](https://en.wikipedia.org/wiki/Gamma_distribution))
//!  - [x] [Beta distribution](crate::distributions::Beta) ([Wiki](https://en.wikipedia.org/wiki/Beta_distribution))
//!  - [ ] [Student's T distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution)
//!  - [ ] [Hypergeometric](https://en.wikipedia.org/wiki/Hypergeometric_distribution) (?)
//!  - [ ] [Beta prime distribution](https://en.wikipedia.org/wiki/Beta_prime_distribution)
//!  - [ ] [Pareto distribution](https://en.wikipedia.org/wiki/Pareto_distribution)
//!  - [ ] [Cauchy distribution](https://en.wikipedia.org/wiki/Cauchy_distribution)
//!  - [ ] [Laplace distribution](https://en.wikipedia.org/wiki/Laplace_distribution)
//!  - [ ] [Logistic distribution](https://en.wikipedia.org/wiki/Logistic_distribution)
//!  - [ ] [Quantile parameterized distribution](https://en.wikipedia.org/wiki/Quantile-parameterized_distribution) (?)
//!  - [ ] [Dirac distribution](distributions::Dirac) ([Wiki](https://en.wikipedia.org/wiki/Dirac_delta_function))
//!  - [ ] ... (more to come (?))
//!
//! ### Discrete distributions:
//!
//!  - [x] [Bernoulli](distributions::Bernoulli) ([Wiki](https://en.wikipedia.org/wiki/Bernoulli_distribution))
//!  - [x] [Binomial](distributions::Binomial) ([Wiki](https://en.wikipedia.org/wiki/Binomial_distribution))
//!  - [x] [Discrete Uniform](distributions::DiscreteUniform) ([Wiki](https://en.wikipedia.org/wiki/Discrete_uniform_distribution))
//!  - [x] [Geometric distribution](distributions::Geometric) ([Wiki](https://en.wikipedia.org/wiki/Geometric_distribution))
//!  - [ ] [Negative binomial distribution](https://en.wikipedia.org/wiki/Negative_binomial_distribution)
//!  - [x] [Poisson distribution](distributions::Poisson)([Wiki](https://en.wikipedia.org/wiki/Poisson_distribution))
//!  - [x] [Degenerate distribution](distributions::Degenerate) ([Wiki](https://en.wikipedia.org/wiki/Degenerate_distribution)) (Continuous version is [Dirac](distributions::Dirac))
//!  - [ ] ... (more to come (?))
//!
//!
//!
//!

use distribution_trait::Distribution;

pub mod Samples;
pub mod configuration;
pub mod distribution_trait;
pub mod distributions;
pub mod domain;
pub mod errors;
pub mod euclid;
pub mod mixed_distribution;

// Todo: Do this struct that allows to shift the location of a distribution
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
