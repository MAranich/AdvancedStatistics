use thiserror::Error;

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum AdvStatError {
    /// A function was evaluated outside it's domain.
    #[error("A function was evaluated outside it's domain. ")]
    DomainErr,

    /// A NaN (Not a Number) was found in the input.
    #[error("A NaN (Not a Number) was found in the input. ")]
    NanErr,

    /// The a number did not fullfill the conditions of the function.
    /// Maybe it was infinite when it was not allowed, was negative when the function
    /// only takes positive number, or was a big numer when the function asks for a
    /// probability. It may also be a NaN.
    #[error(
        "The a number did not fullfill the conditions of the function. Maybe it was infinite when it was not allowed, was negative when the function only takes positive number, or was a big numer when the function asks for a probability. It may also be a NaN. "
    )]
    InvalidNumber,

    /// There was an error when performing some numerical computation. Overflow/underflow/division by 0
    #[error(
        "There was an error when performing some numerical computation. Overflow/underflow/division by 0"
    )]
    NumericalError,

    /// Unknown error
    #[error("Unknown error")]
    Unknown,
}

/// An enum that indicates what went wrong with the test.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum TestError {
    /// A NaN (Not a Number) was found in the input. (Or maybe `+- inf` depending on the function)
    #[error(
        "A NaN (Not a Number) was found in the input. (Or maybe `+- inf` depending on the function)"
    )]
    NanErr,

    /// There were not enough samples to do the operation.
    #[error("There were not enough samples to do the operation. ")]
    NotEnoughSamples,

    /// The arguments violated some of the function preconditions.
    #[error("The arguments violated some of the function preconditions. ")]
    InvalidArguments,

    /// The significance level was set to an invalid value. (`0.0 < significance < 1.0`)
    #[error("The significance level was set to an invalid value. (`0.0 < significance < 1.0`)")]
    InvalidSignificance,

    /// Unknown error / unimplemented
    #[error("Unknown error / unimplemented")]
    Unknown,
}

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum SimulationError {
    #[error("There was an error when using the distribution. ")]
    DistributionError(AdvStatError),

    #[error("There was an error when performing the test. ")]
    TestError(TestError),

    #[error(
        "There are not enough arguments for the simulation. At maximum, only one of `significance_level`, `power` or `sample_size` must be left as None. "
    )]
    MissingArguments,

    #[error("The variable `number_of_repetitions` was set to 0. ")]
    ZeroRepetitions,

    #[error(
        "The variable `significance_level` was **NOT** a valid probability. It must be non-Nan and fullfill `0.0 < significance_level < 1`. "
    )]
    InvalidSignificanceLevel,

    #[error(
        "The variable `power` was **NOT** a valid probability. It must be non-Nan and fullfill `0.0 < power < 1`. "
    )]
    InvalidPower,

    #[error(
        "The sample size we are requiered to operate on may be too large for the integer datatypes. Perhaps there is an error in the arguments. "
    )]
    MassiveSampleSize,
}
