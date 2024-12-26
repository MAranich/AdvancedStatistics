
//! This file contains the deafult values and other value choices used trough the library. 
//! 


/// The library uses numerical integration in a few instances. 
/// In order to do this we have decided to use the [Simpson's rule](https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_1/3_rule)
/// to untegrate. But even considering this, for a given integral we still need 
/// to choose the step length or the number of steps. 
/// 
/// For most intervals, we integrate using a step length of [DEFAULT_INTEGRATION_PRECISION], 
/// however if the interval is too large (it would requiere more than [DEFAULT_INTEGRATION_MAXIMUM_STEPS]
/// to integrate), we will keep the number of steps fixed and just change the precision to not do more than 
/// that many steps. In the other hand, is the interval is very small (a 2 units or less), 
/// we will just use [SMALL_INTEGRATION_NUM_STEPS] with the appropiate precision. 
/// We will also make sure that if we are using the [DEFAULT_INTEGRATION_PRECISION], we do 
/// at least [DEFAULT_INTEGRATION_MINIMUM_STEPS] (and adapt the precision if needed). 
/// 
/// In order to choose the correct values, we use the [crate::euclid::choose_integration_precision_and_steps]
/// function. 
/// 
/// There are no perfect values that will work with every distribution. Increasing the 
/// precision comes with an extra computational cost. We recommend changing the values 
/// to fit your needs. This values are just a mere recomendation. 
pub mod integration {
    


/// The deafult integration precision is `1/8`.
pub static DEFAULT_INTEGRATION_PRECISION: f64 = 0.125;

/// The maximum number of subdivisions of the domains to numerically integrate functions.
/// `1 << 20 + 1 = 1 048 577`
///
/// The `+ 1` is because odd numbers are better for Simpson's integration.
pub static DEFAULT_INTEGRATION_MAXIMUM_STEPS: usize = (1 << 20) + 1;
/// [DEFAULT_INTEGRATION_MAXIMUM_STEPS] but as float. 
pub static DEFAULT_INTEGRATION_MAXIMUM_STEPS_F64: f64 = DEFAULT_INTEGRATION_MAXIMUM_STEPS as f64;

/// The minimum number of subdivisions of the domains to numerically integrate functions.
/// `(1 << 10) + 1 = 4 097`
///
/// The `+ 1` is because odd numbers are better for Simpson's integration.
pub static DEFAULT_INTEGRATION_MINIMUM_STEPS: usize = (1 << 12) + 1;
/// [DEFAULT_INTEGRATION_MINIMUM_STEPS] but as float. 
pub static DEFAULT_INTEGRATION_MINIMUM_STEPS_F64: f64 = DEFAULT_INTEGRATION_MINIMUM_STEPS as f64;

/// The number of steps used to integrate among a small interval
/// (such as `[0, 1]` or `[-1, 1]`). The value is relatively large because we want
/// to make sure to capture the shape of the function in the small interval.
///
/// `32769` = `2^15 + 1`
pub static SMALL_INTEGRATION_NUM_STEPS: usize = 32769;
/// [SMALL_INTEGRATION_NUM_STEPS]
pub static SMALL_INTEGRATION_NUM_STEPS_F64: f64 = SMALL_INTEGRATION_NUM_STEPS as f64;

/// The integration precision for small intervals (such as `[0, 1]` or `[-1, 1]`).
/// Equal to `1/32768`
pub static SMALL_INTEGRATION_PRECISION: f64 = 1.0 / SMALL_INTEGRATION_NUM_STEPS_F64;


}


/// Determines if a Newton's method iteration is used in the (deafult)
/// quantile function (continuous). 
/// 
/// It generally improves precision, but you may want to disable it 
/// if it leads to errors. 
pub static QUANTILE_USE_NEWTONS_ITER: bool = true;










