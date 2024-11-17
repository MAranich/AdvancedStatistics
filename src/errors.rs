use thiserror::Error; 

#[derive(Error, Debug)]
pub enum AdvStatError { 
    #[error("Unknown error")]
    Unknown, 
    #[error("A function was evaluated outside it's domain. ")]
    DomainErr, 

}

