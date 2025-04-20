//! This file contains a simple example on how to do some t-tests:
//!
//!
//!
//!

use AdvancedStatistics::distribution_trait::Distribution;
use AdvancedStatistics::distributions::Normal::*;
use AdvancedStatistics::hypothesis::*;
use AdvancedStatistics::samples::*;
use rand::prelude::*;
use rand::rngs::SmallRng;

static mut TRUE_MEAN: f64 = 0.0;
static mut TRUE_VARIANCE: f64 = 0.0;

fn main() {
    let mut rng: SmallRng = {
        // use seeded or random
        if true {
            let seed: u64 = 1_157_447;
            SmallRng::seed_from_u64(seed)
        } else {
            SmallRng::from_os_rng()
        }
    };

    {
        // one sample t_test:
        println!("Performing one sample t-test: **********************************\n");

        // store the true values in static variable so they can be accessed from everywhere.
        // (not optimal)
        unsafe { TRUE_MEAN = rng.random::<f64>() + 0.1 };
        let true_mean: f64 = unsafe { TRUE_MEAN };
        println!("The true mean of our data: {}\n\t[0.1, 1.1]", true_mean);

        unsafe { TRUE_VARIANCE = rng.random::<f64>() * 1.2 + 0.5 };
        let true_variance: f64 = unsafe { TRUE_VARIANCE };
        println!(
            "The true variance of our data: {}\n\t[1.0, 3.0]",
            true_variance
        );

        let n_samples: usize = 30;
        let shown_samples: usize = n_samples;
        assert!(shown_samples <= n_samples); // or the slicing later may panic

        let normal: Normal =
            Normal::new(true_mean, true_variance.sqrt()).expect("Parameters should be valid. ");

        let samples: Vec<f64> = normal.sample_multiple(n_samples);
        //let samples: Vec<f64> = vec![1.692, -0.070, 0.446, 2.292, -0.532, -0.186, -0.806, 2.888, 0.143, 0.820, 0.000, 1.025, 0.298, 3.535, 0.032, 1.476, 1.209, 1.029, -0.745, 1.369, -0.141, -0.413, 0.722, 0.510, -0.951, -0.164, 1.599, 1.163, 0.940, 0.649];
        let mut data: Samples = Samples::new_move(samples).unwrap();

        let samples: &Vec<f64> = data.peek_data();
        print!("We generate {n_samples} form the normal distribution ({shown_samples} shown): \n[",);

        for i in 0..(shown_samples - 1) {
            print!("{:.3}, ", samples[i]);
        }
        println!("{:.3}]", samples[shown_samples - 1]);

        println!(
            "Sample mean of data: {:.4} \tSample variance: {:.4}\n",
            data.mean().unwrap(),
            data.variance().unwrap()
        );

        println!(
            "We define H0 as the \"mean of our data is 0.0\" and Ha as \"the mean of the data is != 0.0\". Therefore we perform a 2 tailed test. \n"
        );

        let significance: f64 = 0.05;

        one_sample_t_test_example(&mut data, significance);
    }
}

fn one_sample_t_test_example(data: &mut Samples, significance: f64) {
    println!("Performing the test computations... ");

    let results: Result<TestResult, AdvancedStatistics::errors::TestError> = t_test()
        .data(data)
        .hypothesys(Hypothesis::TwoTailed)
        .null_mean(0.0)
        .significance(significance)
        .call();

    let results: TestResult = match results {
        Ok(v) => v,
        Err(e) => panic!("Error: {:?}", e),
    };

    match results {
        TestResult::PValue(_, _) => unreachable!("We have given the t-test a significance value. "),
        TestResult::PValueCI(t, p, ci) => {
            let confidence: f64 = 100.0 * (1.0 - significance);
            println!(
                "t: {:.5} \nThe {confidence:.2}% confidence interval is: ({:.5}, {:.5}) \nThe P value is: {p}",
                t, ci.0, ci.1
            );

            let true_mean: f64 = unsafe { TRUE_MEAN };
            if ci.0 <= true_mean && true_mean <= ci.1 {
                println!(
                    "The true mean is within the confidence interval. (this is normally unknowable)"
                );
            } else {
                println!(
                    "The true mean is NOT within the confidence interval. (this is normally unknowable)"
                );
            }

            if p <= significance {
                println!(
                    "Result: We reject H0. \n\nKnowing the true parameters of the true distribution, this is the correct result. "
                );
            } else {
                println!(
                    "Result: We fail to reject H0. \n\nKnowing the true parameters of the true distribution, this is result is wrong. "
                );

                println!(
                    "In this case, our statistical analisys provided us with a wrong result. This can happen just by random chance. "
                );
            }
        }
    }

    /*

       **********************************************
               R
       **********************************************

       Results for the test on R when the data is:
       [1.692, -0.070, 0.446, 2.292, -0.532, -0.186, -0.806, 2.888, 0.143, 0.820, 0.000, 1.025, 0.298, 3.535, 0.032, 1.476, 1.209, 1.029, -0.745, 1.369, -0.141, -0.413, 0.722, 0.510, -0.951, -0.164, 1.599, 1.163, 0.940, 0.649]

       Code:

       ```
       # Sample data
       samples <- c(0.649, 1.444, 0.616, -1.295, -0.414, 0.133, 1.033, 1.351, -0.258, 1.218, 0.343, -0.218, 0.399, 2.380, -0.362, 1.349, 1.159, 1.862, -0.622, 2.772, 0.502, 0.577, 1.245, 1.705, -0.027, -0.155, 0.793, 2.044, 0.563, 1.231)

       # Perform the one-sample t-test
       t_test_result <- t.test(samples, mu = 0.0)

       # Print the results
       print(t_test_result)
       ```

       Results:
       data:  samples
        t = 4.2815, df = 29, p-value = 0.0001854
        alternative hypothesis: true mean is not equal to 0
        95 percent confidence interval:
        0.3833213 1.0844787
        sample estimates:
        mean of x
        0.7339

       **********************************************
               AdvancedStatistics results:
       **********************************************

        Sample mean of data: 0.6610     Sample variance: 1.1413

        We define H0 as the "mean of our data is 0.0" and Ha as "the mean of the data is != 0.0". Therefore we perform a 2 tailed test.

        Performing the test computations...
        t: 4.28054
        The 95.00% confidence interval is: (0.38468, 1.08292)
        The P value is: 0.00014372368126691626
        The true mean is within the confidence interval.
        Result: We reject H0.

        *************
        Parameters original:    step_length: 0.125, max_iters: 96
        Parameters 2:           step_length: 0.1603093, max_iters: 74
        Parameters 3:           step_length: 0.0801546, max_iters: 149


        **********************************************
                Conclusion:
        **********************************************

        The results are mostly precise, however there are slight
        differences in precision.

    */
}
