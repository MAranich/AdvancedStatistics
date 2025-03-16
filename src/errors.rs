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

#[derive(Error, Debug)]
pub enum TestError {
    #[error("A NaN (Not a Number) was found in the input. ")]
    NanErr, 
    #[error("There were not enough samples to do the operation. ")]
    NotEnoughSamples, 
    #[error("The arguments violated some of the function preconditions. ")]
    InvalidArguments, 
    #[error("Unknown error")]
    Unknown,
}
