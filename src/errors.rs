use thiserror::Error;

#[derive(Error, Debug)]
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
    /// There were not enough samples to do the operation.
    InvalidNumber,
    #[error("There were not enough samples to do the operation. ")]
    NotEnoughSamples,
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
