//! This file contains a simple example on how to do some t-tests:
//!
//!
//!
//!

use AdvancedStatistics::Samples::*;
use AdvancedStatistics::distribution_trait::Distribution;
use AdvancedStatistics::distributions::Normal::*;
use AdvancedStatistics::hypothesis::*;
use rand::prelude::*;
use rand::rngs::SmallRng;

fn main() {
    let seed: u64 = 1_157_447;
    let mut rng: SmallRng = SmallRng::seed_from_u64(seed);
    {
        // one sample t_test:
        println!("Performing one sample t-test: **********************************\n");

        let true_mean: f64 = rng.random::<f64>() + 0.1;
        println!("The true mean of our data: {}\n\t[0.1, 1.1]", true_mean);

        let true_variance: f64 = rng.random::<f64>() * 1.2 + 0.5;
        println!(
            "The true variance of our data: {}\n\t[1.0, 3.0]",
            true_variance
        );

        let n_samples: usize = 30;
        let shown_samples: usize = 30;
        assert!(shown_samples <= n_samples); // or the slicing later may panic

        let normal: Normal =
            Normal::new(true_mean, true_variance.sqrt()).expect("Parameters should be valid. ");

        let samples: Vec<f64> = normal.sample_multiple(n_samples); 
        //let samples: Vec<f64> = vec![1.692, -0.070, 0.446, 2.292, -0.532, -0.186, -0.806, 2.888, 0.143, 0.820, 0.000, 1.025, 0.298, 3.535, 0.032, 1.476, 1.209, 1.029, -0.745, 1.369, -0.141, -0.413, 0.722, 0.510, -0.951, -0.164, 1.599, 1.163, 0.940, 0.649]; 
        let mut data: Samples = Samples::new_move(samples).unwrap();

        let samples: &Vec<f64> = data.peek_data();
        print!("We generate {n_samples} form the normal distribution: \n[",);

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
        samples <- c(1.692, -0.070, 0.446, 2.292, -0.532, -0.186, -0.806, 2.888, 0.143, 0.820, 0.000, 1.025, 0.298, 3.535, 0.032, 1.476, 1.209, 1.029, -0.745, 1.369, -0.141, -0.413, 0.722, 0.510, -0.951, -0.164, 1.599, 1.163, 0.940, 0.649)

        # Perform the one-sample t-test
        t_test_result <- t.test(samples, mu = 0.0)

        # Print the results
        print(t_test_result)
        ```

        Results: 
        data:  samples
        t = 3.3888, df = 29, p-value = 0.002039
        alternative hypothesis: true mean is not equal to 0
        95 percent confidence interval:
        0.2620577 1.0598756
        sample estimates:
        mean of x 
        0.6609667 

        **********************************************
                AdvancedStatistics results: 
        **********************************************

        Sample mean of data: 0.6610     Sample variance: 1.1413

        We define H0 as the "mean of our data is 0.0" and Ha as "the mean of the data is != 0.0". Therefore we perform a 2 tailed test.

        Performing the test computations...
        t: 3.38881
            The 95.00% confidence interval is: (0.32998, 0.99195)
            The 95.00% confidence interval is: (0.33024, 0.99169)
        The 95.00% confidence interval is: (0.33164, 0.99029)
            The P value is: 0.0015244510675893608
            The P value is: 0.0017533729375256435
        The P value is: 0.0017552924987749297
        Result: We reject H0.

        ************
        Biased:     t: 3.44675
        Unbiased:   t: 3.36667
        Unbiased 2: t: 3.38881
        R:          t: 3.3888

        *************
        Parameters original:    step_length: 0.125, max_iters: 96
        Parameters 2:           step_length: 0.1603093, max_iters: 74
        Parameters 3:           step_length: 0.0801546, max_iters: 149

     */
}
