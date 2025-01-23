#![allow(non_snake_case)]
// ^Disable warning "crate `AdvancedStatistics` should have a snake case name convert the identifier to snake case: `advanced_statistics`"
// The rest of the names will follow the snake_case convention.

//! # Advanced Statistics
//!
//!
//! This library is a statistics library that provides:
//!
//! - [ ] Interface to create distributions
//!     - (only implemented for continuous case)
//! - [ ] Common distributions (ready to be used in any calculation)
//! - [ ] Data computation
//! - [ ] Multiple tests (hypothesys testing)
//! - [ ] Computing P values and confidence intervals
//! - [ ] Parameter estimation
//! - [ ] Outlier detection
//! - [ ] Resampling methods
//! - [ ] Multidimensional data support
//! - [ ] ANOVA / MANOVA
//! - [ ] Bayesian statistics
//!
//! ## Distributions
//!
//! We have defined the trait [distribution_trait::Distribution] and
//! [distribution_trait::DiscreteDistribution] that define a basic interface
//! to work with distributions. The only requiered methods to implement are:
//!  - [distribution_trait::Distribution::pdf]: the pdf of the distribution.
//!  - [distribution_trait::Distribution::get_domain]: the [domain]
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
//! ### Continuous distributions
//!
//!  - [x] Exponential
//!  - [ ] ... (more to come)
//!
//! ### Discrete distributions
//!
//!  - [ ] ... (more to come)
//!  - [ ] ... (more to come)
//!
//!
//!
//!

pub mod configuration;
pub mod distribution_trait;
pub mod distributions;
pub mod domain;
pub mod errors;
pub mod euclid;
pub mod mixed_distribution;
