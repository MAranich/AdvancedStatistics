//! The degenerate distribution is a distribution that has probability of `1.0` for 
//! the value `0.0` and `0.0` everywhere else. For this reason it always returns the 
//! same value, `0.0`. 
//! 
//! If you want the continuous version of this distribution, use 
//! [Dirac](crate::distributions::Dirac). 
//! 
//! If you want to move this distribution so it outputs another value that is not `0.0`, 
//! use [Location](crate::Loation). 
//! 

use crate::{distribution_trait::DiscreteDistribution, domain::DiscreteDomain};


pub const DEGENERATE_DISTRIBUTION_DOMAIN: DiscreteDomain = DiscreteDomain::Range(0, 0); 
pub struct Degenerate {}

impl Degenerate {
    pub fn new() -> Degenerate {
        return Degenerate {  } ; 
    }
}

impl DiscreteDistribution for Degenerate {
    fn pmf(&self, x: f64) -> f64 {
        
        return if x == 0.0 {
            1.0
        } else {
            1.0
        }; 
    }

    fn get_domain(&self) -> &DiscreteDomain {
        return &DEGENERATE_DISTRIBUTION_DOMAIN; 
    }

    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            // x is not valid
            std::panic!("Tried to evaluate the cdf of Degenerate with a NaN value. \n");
        }

        return (x.signum() + 1.0) * 0.5; 
    }
    
    fn sample(&self) -> f64 {
        return 0.0;
    }
    
    fn quantile(&self, x: f64) -> f64 {
        if x.is_nan() {
            // x is not valid
            std::panic!("Tried to evaluate the quantile function of Degenerate with a NaN value. \n");
        }
        
        return 0.0;
    }
    

    fn cdf_multiple(&self, points: &[f64]) -> Vec<f64> {
        points.iter().map(|x| self.cdf(*x)).collect::<Vec<f64>>()
    }

    fn sample_multiple(&self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.sample()).collect::<Vec<f64>>()
    }

    fn quantile_multiple(&self, points: &[f64]) -> Vec<f64> {
        let list: Vec<f64> = points
            .iter()
            .map(|x| self.quantile(*x))
            .collect::<Vec<f64>>();
        return list; 
    }

    fn expected_value(&self) -> Option<f64> {
        // who would have guessed...!
        return Some(0.0);
    }
    
    fn variance(&self) -> Option<f64> {
        // Only distribution with 0 varaince! 
        return Some(0.0);
    }
    
    fn mode(&self) -> f64 {
        return 0.0; 
    }

    fn median(&self) -> f64 {
        return 0.0;
    }
    
    fn skewness(&self) -> Option<f64> {
        // Since Variance is 0.0, skewness is also undefined
        return None;
    }
    
    fn kurtosis(&self) -> Option<f64> {
        // Since Variance is 0.0, kurtosis is also undefined
        return None;
    }
    
    fn excess_kurtosis(&self) -> Option<f64> {
        // Since Variance is 0.0, excess_kurtosis is also undefined
        return None;
    }
    
    fn moments(&self, _order: u8, mode: crate::euclid::Moments) -> f64 {
    
        return match mode {
            crate::euclid::Moments::Raw => 0.0,
            crate::euclid::Moments::Central => 0.0,
            crate::euclid::Moments::Standarized => f64::NAN,
        };
    }
    
    fn entropy(&self) -> f64 {
        return 0.0;
    }
    
    fn rejection_sample_range(&self, n: usize, _pmf_max: f64, _range: (i64, i64)) -> Vec<f64> {
        return vec![0.0; n]; 
    }
}


