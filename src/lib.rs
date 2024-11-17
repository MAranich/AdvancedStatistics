#![allow(non_snake_case)]
// ^Disable warning "crate `AdvancedStatistics` should have a snake case name convert the identifier to snake case: `advanced_statistics`"
// The rest of the names will follow the snake_case convention.

pub mod errors;

pub mod distributions;

pub mod euclid;

/// The deafult integration precision is `1/8`.
pub const DEAFULT_INTEGRATION_PRECISION: f64 = 0.125;
