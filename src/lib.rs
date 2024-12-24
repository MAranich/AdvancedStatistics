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
pub mod errors;
pub mod euclid;

// only 1 RNG (Random Number Generator per thread)
thread_local! {
    pub static RNG: RefCell<ThreadRng> = RefCell::new(rand::thread_rng());
}
//pub static RNG: ThreadRng = rand::thread_rng();

/// The deafult integration precision is `1/8`.
pub const DEFAULT_INTEGRATION_PRECISION: f64 = 0.125;

/// The maximum number of subdivisions of the domains to numerically integrate functions.
/// `1 << 20 + 1 = 1 048 577`
///
/// The `+ 1` is because odd numbers are better for Simpson's integration.
pub const DEFAULT_INTEGRATION_MAXIMUM_STEPS: usize = (1 << 20) + 1;
/// [DEFAULT_INTEGRATION_MAXIMUM_STEPS]
pub const DEFAULT_INTEGRATION_MAXIMUM_STEPS_F64: f64 = DEFAULT_INTEGRATION_MAXIMUM_STEPS as f64;

/// The minimum number of subdivisions of the domains to numerically integrate functions.
/// `(1 << 10) + 1 = 1 025`
///
/// The `+ 1` is because odd numbers are better for Simpson's integration.
pub const DEFAULT_INTEGRATION_MINIMUM_STEPS: usize = (1 << 12) + 1;
/// [DEFAULT_INTEGRATION_MINIMUM_STEPS]
pub const DEFAULT_INTEGRATION_MINIMUM_STEPS_F64: f64 = DEFAULT_INTEGRATION_MINIMUM_STEPS as f64;

/// The number of steps used to integrate among a small interval
/// (such as `[0, 1]` or `[-1, 1]`). The value is relatively large because we want
/// to make sure to capture the shape of the function in the small interval.
///
/// `32769` = `2^15 + 1`
pub const SMALL_INTEGRATION_NUM_STEPS: usize = 32769;
/// [SMALL_INTEGRATION_NUM_STEPS]
pub const SMALL_INTEGRATION_NUM_STEPS_F64: f64 = SMALL_INTEGRATION_NUM_STEPS as f64;

/// The integration precision for small intervals (such as `[0, 1]` or `[-1, 1]`).
/// Equal to `1/32768`
pub const SMALL_INTEGRATION_PRECISION: f64 = 1.0 / SMALL_INTEGRATION_NUM_STEPS_F64;

/// Determines if a Newton's method iteration is used in the (deafult)
/// quantile function, mainly for testing.
pub const QUANTILE_USE_NEWTONS_ITER: bool = true;
