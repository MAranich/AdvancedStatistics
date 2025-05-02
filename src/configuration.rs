//! This file contains the deafult values and other value choices used trough the library.
//!

pub mod integration {
    //! The library uses numerical integration in a few instances.
    //! In order to do this we have decided to use the [Simpson's rule](https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_1/3_rule)
    //! to untegrate. But even considering this, for a given integral we still need
    //! to choose the step length or the number of steps.
    //!
    //! For most intervals, we integrate using a step length of [DEFAULT_INTEGRATION_PRECISION],
    //! however if the interval is too large (it would requiere more than [DEFAULT_INTEGRATION_MAXIMUM_STEPS]
    //! to integrate), we will keep the number of steps fixed and just change the precision to not do more than
    //! that many steps. In the other hand, is the interval is very small (a 2 units or less),
    //! we will just use [SMALL_INTEGRATION_NUM_STEPS] with the appropiate precision.
    //! We will also make sure that if we are using the [DEFAULT_INTEGRATION_PRECISION], we do
    //! at least [DEFAULT_INTEGRATION_MINIMUM_STEPS] (and adapt the precision if needed).
    //!
    //! In order to choose the correct values, we use the [crate::euclid::choose_integration_precision_and_steps]
    //! function.
    //!
    //! There are no perfect values that will work with every distribution. Increasing the
    //! precision comes with an extra computational cost. We recommend changing the values
    //! to fit your needs. This values are just a mere recomendation.

    /// Multiplies the number of steps when performing a numerical integration
    /// in a **finite** domain.
    pub static mut MULTIPLIER_STEPS_FINITE_INTEGRATION: f64 = 1.0;

    /// The deafult integration precision is `1/8`.
    pub static mut DEFAULT_INTEGRATION_PRECISION: f64 = 0.125;

    /// The maximum number of subdivisions of the domains to numerically integrate functions.
    /// `1 << 20 = 1 048 576`
    pub static mut DEFAULT_INTEGRATION_MAXIMUM_STEPS: usize = 1 << 20;
    /// [DEFAULT_INTEGRATION_MAXIMUM_STEPS] but as float.
    pub static mut DEFAULT_INTEGRATION_MAXIMUM_STEPS_F64: f64 =
        unsafe { DEFAULT_INTEGRATION_MAXIMUM_STEPS } as f64;

    /// The minimum number of subdivisions of the domains to numerically integrate functions.
    /// `1 << 12 = 4 096`
    pub static mut DEFAULT_INTEGRATION_MINIMUM_STEPS: usize = 1 << 12;
    /// [DEFAULT_INTEGRATION_MINIMUM_STEPS] but as float.
    pub static mut DEFAULT_INTEGRATION_MINIMUM_STEPS_F64: f64 =
        unsafe { DEFAULT_INTEGRATION_MINIMUM_STEPS } as f64;

    /// The number of steps used to integrate among a small interval
    /// (such as `[0, 1]` or `[-1, 1]`). The value is relatively large because we want
    /// to make sure to capture the shape of the function in the small interval.
    ///
    /// `32768` = `2^15`
    pub static mut SMALL_INTEGRATION_NUM_STEPS: usize = 1 << 15;
    /// [SMALL_INTEGRATION_NUM_STEPS]
    pub static mut SMALL_INTEGRATION_NUM_STEPS_F64: f64 =
        unsafe { SMALL_INTEGRATION_NUM_STEPS } as f64;

    /// The integration precision for small intervals (such as `[0, 1]` or `[-1, 1]`).
    /// Equal to `1/32768`
    pub static mut SMALL_INTEGRATION_PRECISION: f64 =
        1.0 / unsafe { SMALL_INTEGRATION_NUM_STEPS_F64 };
}

pub mod disrete_distribution_deafults {

    /// `1 << 20 = 1 048 576`
    ///
    /// When computing a discrete integration of [crate::distribution_trait::DiscreteDistribution],
    /// if the distribution has an infinite number of steps, there will be an infinite
    /// loop. To avoid this, we set a maximum number of steps to terminate at some point.
    pub static mut MAXIMUM_STEPS: u64 = 1 << 20;
}

pub mod derivation {

    /// Default `h` when performing numerical derivation:
    ///
    /// `f'(x) = (f(x + h) - f(x))/h`
    pub static mut DEAFULT_H: f64 = 1.0 / 1024.0;
}

pub mod distribution_mode_deafult {
    //! The [mode](https://en.wikipedia.org/wiki/Mode_(statistics)) for a continuous distribution
    //! is the value `m` wich maximizes the pdf. This means that it is a function
    //! optimitzation problem. To solve this problem the solution we have settled on is
    //! [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent)
    //! (but adapted for our use case). This module allows the user to change the deafult
    //! values of the parameters of the algorithm.

    /// Use the [Logarithmic derivative](https://en.wikipedia.org/wiki/Logarithmic_derivative)
    /// instead of the normal derivarive to compute the gradient. We have found that it
    /// usally performs better by using it.
    pub static mut USE_LOG_DERIVATIVE: bool = true;

    /// The [learning rate](https://en.wikipedia.org/wiki/Learning_rate). It indicates
    /// how big is the "step" we take to the solution.
    ///  - A big learning rate will lead to the solution faster, but may overshoot.
    ///  - A small learning rate will better converge once near a local optimum but
    ///     may also get stuck in an undesirable local maximum (not the global maximum).
    ///     It will also have a slow convergence.
    pub static mut LEARNING_RATE: f64 = 0.02;

    /// The change of the [learning rate](https://en.wikipedia.org/wiki/Learning_rate)
    /// ([LEARNING_RATE]). After every iteration the learning rate us updated as:
    ///
    /// ```
    /// current_learning_rate = current_learning_rate * LEARNING_RATE_CHANGE;
    /// ```
    ///
    /// Using this strategy to update the learning rate can lead to bigger steps
    /// at the initial iterations while smaller at the latter ones. Setting
    /// `LEARNING_RATE_CHANGE = 1.0` essentially disables this feature. We do not
    /// reccomend using values greater than `1.0` or smaller than `0.5`.
    pub static mut LEARNING_RATE_CHANGE: f64 = 0.9999;

    /// How close need to be the old and updated value in order to declare that
    /// convergence has been achived and return the result.
    pub static mut CONVERGENCE_DIFFERENCE_CRITERIA: f64 = 0.0001;

    /// Sets the number of iterations that the algorithm must do before
    /// reaching convergence.
    ///
    /// Setting this to a big number may be uscefull
    /// if the algorithm is in a very flat part of the function.
    pub static mut MIN_ITERATIONS: u32 = 0;

    /// Sets the maximum number of iterations.
    ///
    /// Deafult: `1 << 16 = 65 536`
    pub static mut MAX_ITERATIONS: u32 = 1 << 16;
}

pub mod maximum_likelihood_estimation {
    //! Under development

    pub static mut USE_LOG_DERIVATIVE: bool = false;

    pub static mut LEARNING_RATE: f64 = 0.02;

    pub static mut LEARNING_RATE_CHANGE: f64 = 0.9999;

    pub static mut CONVERGENCE_DIFFERENCE_CRITERIA: f64 = 0.0001;

    pub static mut MAX_ITERATIONS: u32 = 1 << 16;
}

/// Determines if a Newton's method iteration is used in the (deafult)
/// quantile function (continuous).
///
/// It generally improves precision, but you may want to disable it
/// if it leads to errors.
pub static mut QUANTILE_USE_NEWTONS_ITER: bool = true;

pub mod newtons_method {
    /// Convergence criteria for Newton's method.
    ///
    /// =`1.52587890625e-05`
    pub static mut NEWTONS_CONVERGENCE_DIFFERENCE_CRITERIA: f64 =
        f64::from_bits(4535124824762089472);
}
