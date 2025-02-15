//! Testing module for parameter estimation calculations.
//!
//!
//!

use distribution_trait::{DiscreteDistribution, Distribution};
use distributions::{Binomial::Binomial, Exponential::Exponential};
use domain::*;
use AdvancedStatistics::{distribution_trait::Parametric, Samples::Samples, *};

#[test]
fn exponential_samples() {
    let real_lambda: f64 = 0.85;
    let my_exp: Exponential =
        Exponential::new(real_lambda).expect("lambda should be stricly positive. ");

    let sampl: Vec<f64> = my_exp.sample_multiple(5000);

    // println!("|{:?}|", sampl);

    let mut my_samples: Samples = Samples::new_move(sampl).expect("No inf nor nans");
    let mean: f64 = my_samples.mean().unwrap();

    // 1.0 is the seed
    let infered_lambda: f64 = Exponential::new(1.0).unwrap().fit(&mut my_samples)[0];

    println!(
        "Real parameter: \t{} \nInfered one: \t\t{} \nMethod of moments: \t{}\n\n",
        real_lambda,
        infered_lambda,
        1.0 / mean
    );

    panic!("Show results! ");
    // for now it looks mostly good
}
