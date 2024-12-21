

use distributions::{distribution_interface::Distribution, Exponential::Exponential};
use euclid::Domain;
use AdvancedStatistics::*; 


#[test]
fn dummy() {}

#[test]
fn temporal() {
    let value: f64 = 1.0 / 2.0_f64.powi(15); 
    print!("{} | {} \n", value, value.to_bits()); 
    //panic!(); 
}

struct my_exp {
    lambda: f64,
    domain: Domain,
}

impl Distribution for my_exp {
    fn pdf(&self, x: f64) -> f64 {
        self.lambda * (-self.lambda * x).exp()
    }

    fn get_pdf_domain(&self) -> &AdvancedStatistics::euclid::Domain {
        &self.domain
    }
}

#[test]
fn comparing_exponential_distr() {
    let lambda: f64 = 1.2; 
    let max: f64 = -(0.0001_f64.ln()) / lambda;


    let ground: Exponential = Exponential::new(lambda).unwrap(); 
    let test: my_exp = my_exp {
        lambda: lambda, 
        domain: Domain::new_continuous_range(0.0, max)
    }; 


    // ***

    println!("\nTesting cdf: \n\n"); 

    let points: Vec<f64> = vec![0.0, 0.1, 0.5, 0.75, 1.0, 1.2, 1.6, 2.5, 4.0, 10.0]; 
    let r1: Vec<f64> = ground.cdf_multiple(&points); 
    let r2: Vec<f64> = test.cdf_multiple(&points); 


    for (i, x) in points.iter().enumerate() {
        let diff: f64 = r1[i] - r2[i];  
        println!("cdf({}) || \tground: {}\ttest: {} \tdiff: {}", x, r1[i], r2[i], diff); 
    }

    // ***

    println!("\nTesting quantile: \n\n"); 

    let points: Vec<f64> = vec![-1.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0]; 
    let r1: Vec<f64> = ground.quantile_multiple(&points).unwrap(); 
    let r2: Vec<f64> = test.quantile_multiple(&points).unwrap(); 


    for (i, x) in points.iter().enumerate() {
        let diff: f64 = r1[i] - r2[i];  
        println!("quantile({}) || \tground: {}\ttest: {} \tdiff: {}", x, r1[i], r2[i], diff); 
    }

    /*
    for lambda = 1.2, 
    cdf(7.67528364331348) = 0.9998999999999999
    
    7.675283643313579
    
     */


    // ***

    println!("\nStatistics: \n\n"); 
    
    println!("Expected value: \n\t - ground: {:?}\n\t - test: {:?}\n", ground.expected_value(), test.expected_value()); 
    
    println!("Variance: \n\t - ground: {:?}\n\t - test: {:?}\n", ground.variance(), test.variance()); 
    
    println!("Skewness: \n\t - ground: {:?}\n\t - test: {:?}\n", ground.skewness(), test.skewness()); 










    println!("\n\n================================\n\n"); 

    panic!("Show me the results. "); 

}







