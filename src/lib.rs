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
//! We have defined the trait [distributions::distribution_trait::Distribution] and
//! [distributions::distribution_trait::DiscreteDistribution] that define a basic interface
//! to work with distributions. The only requiered methods to implement are:
//!  - [distributions::distribution_trait::Distribution::pdf]: the pdf of the distribution.
//!  - [distributions::distribution_trait::Distribution::get_domain]: the [euclid::Domain]
//! of the pdf of the distribution.
//!
//! After this, a wide array of funcions are avaliable (check [distributions::distribution_trait::Distribution] for more details).
//! Note that this deafult implementations can be computationally costly, therefore we
//! recommend implementing the other methods if there is an avaliable solution for them.
//!
//! But if you are interested on a more common distribution, you may find it among the
//! ones that we have already implemented:
//!
//!  - [x] Exponential
//!  - [ ] ... (more to come)
//!
//!

use rand::rngs::ThreadRng;
use std::cell::RefCell;

pub mod distributions;
pub mod distribution_trait;
pub mod errors;
pub mod euclid;
pub mod configuration; 
pub mod domain;



