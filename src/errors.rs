use thiserror::Error;

#[derive(Error, Debug)]
pub enum AdvStatError {
    #[error("Unknown error")]
    Unknown,
    #[error("A function was evaluated outside it's domain. ")]
    DomainErr,
    #[error("A NaN (Not a Number) was found in the input. ")]
    NanErr, 
    #[error("There were not enough samples to do the operation. ")]
    NotEnoughSamples, 
}

/// An enum that indicates what went wrong with the test. 
#[derive(Error, Debug)]
pub enum TestError {
    /// A NaN (Not a Number) was found in the input. (Or maybe `+- inf` depending on the function)
    #[error("A NaN (Not a Number) was found in the input. (Or maybe `+- inf` depending on the function)")]
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
