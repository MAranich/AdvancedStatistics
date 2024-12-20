#![allow(non_snake_case)]
// ^Disable warning "crate `AdvancedStatistics` should have a snake case name convert the identifier to snake case: `advanced_statistics`"
// The rest of the names will follow the snake_case convention.

use std::{cell::RefCell, sync::LazyLock};
use rand::rngs::ThreadRng; 

pub mod distributions;
pub mod euclid;
pub mod errors;

// only 1 RNG (Random Number Generator per thread)
thread_local! {
    pub static RNG: RefCell<ThreadRng> = RefCell::new(rand::thread_rng()); 
}
//pub static RNG: ThreadRng = rand::thread_rng(); 

/// The deafult integration precision is `1/8`.
pub const DEFAULT_INTEGRATION_PRECISION: f64 = 0.125;

/// The maximum number of subdivisions of the domains to numerically integrate functions. 
/// `1 << 20 = 1 048 577`
/// 
/// The `+ 1` is because odd numbers are better for Simpson's integration. 
pub const DEFAULT_INTEGRATION_MAXIMUM_STEPS: usize = (1 << 20) + 1;
pub const DEFAULT_INTEGRATION_MAXIMUM_STEPS_F64: f64 = DEFAULT_INTEGRATION_MAXIMUM_STEPS as f64;

/// The number of steps used to integrate among a small interval 
/// (such as `[0, 1]` or `[-1, 1]`). The value is relatively large because we want 
/// to make sure to capture the shape of the function in the small interval. 
/// 
/// `32769` = `2^15 + 1` 
pub const SMALL_INTEGRATION_NUM_STEPS: usize = 32769; 
pub const SMALL_INTEGRATION_NUM_STEPS_F64: f64 = 32769.0; 


/// The integration precision for small intervals (such as `[0, 1]` or `[-1, 1]`). 
/// Equal to `1/32768` 
pub const SMALL_INTEGRATION_PRECISION: f64 = 1.0 / SMALL_INTEGRATION_NUM_STEPS_F64; 

/// Determines if a Newton's method iteration is used in the (deafult) 
/// quantile function, mainly for testing. 
pub const QUANTILE_USE_NEWTONS_ITER: bool = true; 




















