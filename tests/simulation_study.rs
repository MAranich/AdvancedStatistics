//! This testinc module is dedicated to test the correctness of the
//! [simulation study](crate::hypothesis::simulation_study) methods.
//!
//! ***
//!
//! For now we are only going to make tests for the simpler cases. The reason for this
//! is that the logic and the code for all the simulation studies is the same.
//! The major difference is the test used and any small changes to account for it.
//!
//! ***
//!
//! To verify that our results are correct, we will use the results provided by
//! the open source packet [pwr](https://github.com/heliosdrm/pwr) for R.
//!
//! ***
//!
//! Note that all of these tests are **probabilistic**. Therefore it is possible
//! that they fail from time to time. However they should pass *most* of the times
//! and if not, get a result *close* to the correct one.
//!
//!
//!

use AdvancedStatistics::hypothesis::Hypothesis;
use AdvancedStatistics::{distributions::Normal::Normal, hypothesis::simulation_study::*};
use assert_approx_eq::assert_approx_eq;

#[test]
fn simulation_simple_t_test_1() {
    /*
       Here we will make sure that the t-test has the adequate power.

       Using the following R code we obtain:
       ```R
       library(pwr)


       null_mean = 0.0
       alt_mean = 0.5
       std = 1.0
       cohens_d = (alt_mean - null_mean) / std

       n = 32

       pwr.t.test(n = n, d = cohens_d, sig.level = 0.05, type = "one.sample")

       ```

       We obtain:

       ```
            One-sample t test power calculation

                 n = 32
                 d = 0.5
         sig.level = 0.05
             power = 0.782276
       alternative = two.sided
       ```

       ***

       From our manual supervision, we have seen that the absolute error is generally
       arround 1% with some variation.

    */

    let theoretical_power: f64 = 0.782276;
    let tolerance: f64 = 0.05;

    // ********************

    let null_distr: Normal = Normal::new(0.0, 1.0).expect("Parameters are valid. ");
    let alt_distr: Normal = Normal::new(0.5, 1.0).expect("Parameters are valid. ");

    let n: usize = 32;
    let significance_level: f64 = 0.05;

    // We will perform quite a bit of repetitions. Monte Carlo
    // methods only work for large sample sizes. (could be increased)
    let repetitions: usize = 2048;

    let result: SimulationResult = simulation_t_test()
        .null_distribution(&null_distr)
        .alternative_distribution(&alt_distr)
        .hypothesys(Hypothesis::TwoTailed)
        .significance_level(significance_level)
        .sample_size(n)
        .number_of_repetitions(repetitions)
        .call();

    let estimated_power: f64 = match result {
        SimulationResult::Power(p) => p,
        _ => panic!("Power was not successfully computed. "),
    };

    assert_approx_eq!(theoretical_power, estimated_power, tolerance); 

    // panic!("estimated_power: {} , {}, {}", estimated_power, (theoretical_power - estimated_power).abs(), (theoretical_power - estimated_power).abs().ln());
}

#[test]
fn simulation_simple_t_test_2() {
    /*
        Here we will make sure that the t-test has the adequate power.

        Using the following R code we obtain:
        ```R
        library(pwr)


        null_mean = 1.3
        alt_mean = 1.0
        std = 2.0
        cohens_d = (alt_mean - null_mean) / std

        n = 16

        pwr.t.test(n = n, d = cohens_d, sig.level = 0.05, type = "one.sample")

        ```

        We obtain:

        ```
        One-sample t test power calculation

                    n = 16
                    d = 0.15
            sig.level = 0.05
                power = 0.08695976
          alternative = two.sided
        ```

        ***


    */

    let theoretical_power: f64 = 0.08695976; 
    let tolerance: f64 = 0.05;

    // ********************

    let null_distr: Normal = Normal::new(1.3, 2.0).expect("Parameters are valid. ");
    let alt_distr: Normal = Normal::new(1.0, 2.0).expect("Parameters are valid. ");

    let n: usize = 16; 
    let significance_level: f64 = 0.05;

    // We will perform quite a bit of repetitions. Monte Carlo
    // methods only work for large sample sizes. (could be increased)
    let repetitions: usize = 2048;

    let result: SimulationResult = simulation_t_test()
        .null_distribution(&null_distr)
        .alternative_distribution(&alt_distr)
        .hypothesys(Hypothesis::TwoTailed)
        .significance_level(significance_level)
        .sample_size(n)
        .number_of_repetitions(repetitions)
        .call();

    let estimated_power: f64 = match result {
        SimulationResult::Power(p) => p,
        _ => panic!("Power was not successfully computed. "),
    };

    assert_approx_eq!(theoretical_power, estimated_power, tolerance); 

    // panic!("estimated_power: {} , {}, {}", estimated_power, (theoretical_power - estimated_power).abs(), (theoretical_power - estimated_power).abs().ln());
}
