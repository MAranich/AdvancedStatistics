
//use AdvancedStatistics::distribution_trait::*; 
use AdvancedStatistics::{distribution_trait::Distribution, distributions::Normal::*}; 

use std::fs::File;
use std::io::prelude::*;

fn print_paired_vectors(input: &[f64], output_opt: Option<&[f64]>) {

    if let Some(output) = output_opt {
        assert!(input.len() == output.len()); 
        
        for (i, (a, b)) in input.iter().zip(output.iter()).enumerate() {
            println!("{i}\t{:.3}: \t{:.3}", a, b); 
        }
        println!(""); 
    } else {
        for (i, a) in input.iter().enumerate() {
            println!("{i}\t{:.3}", a); 
        }
        println!(""); 
    }
}

fn main() {

    println!("*****************************************************************\n\
    This script shows how to use the distribution interface. We will use the \
    normal distribution, but the interface is common to all distriutions. \
    We will start with a standard normal distribution (0 mean and 1 variance). \n"); 

    // Just for StdNormal, this step can be omited, since it holds no information
    // (by using AdvancedStatistics::distributions::Normal::STD_NORMAL)

    let std_normal: StdNormal = StdNormal::new(); 

    println!("Expected value: {} \t (Correct one: {})", std_normal.expected_value().unwrap(), 0.0); 
    println!("Variance: {} \t (Correct one: {})", std_normal.variance().unwrap(), 1.0); 
    println!("Skewness: {} \t (Correct one: {})", std_normal.skewness().unwrap(), 0.0); 
    println!("Excess Kurtosis: {} \t (Correct one: {})", std_normal.excess_kurtosis().unwrap(), 0.0); 

    let inputs: Vec<f64> = (0..=16).into_iter().map(|x: i32| (x as f64) * 0.5 - 4.0).collect::<Vec<f64>>(); 

    let cdf_values: Vec<f64> = std_normal.cdf_multiple(&inputs); 

    println!("***********************************************************"); 
    println!("Cdf: "); 

    print_paired_vectors(&inputs, Some(&cdf_values));

    let inputs: Vec<f64> = (1..=19).into_iter().map(|x: i32| x as f64 * (1.0/20.0)).collect::<Vec<f64>>(); 

    let quantile_values: Vec<f64> = std_normal.quantile_multiple(&inputs); 

    println!("Quantiles: "); 

    print_paired_vectors(&inputs, Some(&quantile_values));

    println!("***********************************************************"); 

    // Generate n samples and print them 

    let n: usize = 50; 
    println!("{} samples: ", n); 


    // instead of printing them, save tehm in a file. 
    
    let samples: Vec<f64> = std_normal.sample_multiple(n); 
    
    let generate_file_with_samples: bool = false; 
    if generate_file_with_samples {
        
        /*
            // Write floats in a binary file (effitient but not human readable)
            let mut stringified_samples: Vec<u8> = Vec::with_capacity(8 * samples.len()); 

            for s in &samples {
                stringified_samples.extend_from_slice(&s.to_le_bytes()) ;
            }

            let mut file: File = File::create("output.b").expect("create failed");
            file.write_all(&stringified_samples).expect("write failed");
        */

        let mut stringified_samples: Vec<u8> = Vec::with_capacity(8 * samples.len()); 

        for s in &samples {
            let str: String = format!("{}\n", *s); 
            stringified_samples.extend_from_slice(str.as_bytes()) ;
        }

        let mut file: File = File::create("output.txt").expect("create failed");
        file.write_all(&stringified_samples).expect("write failed");
        
    } else {
        print_paired_vectors(&samples, None);
    }

}
