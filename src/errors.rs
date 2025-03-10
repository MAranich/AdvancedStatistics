use thiserror::Error;

#[derive(Error, Debug)]
pub enum AdvStatError {
    #[error("Unknown error")]
    Unknown,
    #[error("A function was evaluated outside it's domain. ")]
    DomainErr,
    #[error("A NaN (Not a Number) was found in the input. ")]
    NanErr,
}
