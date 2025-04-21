use AdvancedStatistics::{
    distribution_trait::DiscreteDistribution, distributions::Bernoulli::*,
    distributions::Poisson::*, euclid::*, hypothesis::Hypothesis,
};

#[cfg(test)]
use assert_approx_eq::assert_approx_eq;

macro_rules! assert_range {
    ($min:expr, $x:expr, $max: expr) => {{
        let (min, x, max) = (&$min, &$x, &$max);

        assert!(
            min <= max,
            "assertion failed: `(min !<= max)` \
            (min: `{:?}`, max: `{:?}`)",
            *min,
            *max
        );

        assert!(
            min <= x,
            "assertion failed: `(min !<= x)` \
            (min: `{:?}`, x: `{:?}`)",
            *min,
            *x
        );

        assert!(
            x <= max,
            "assertion failed: `(x !<= max)` \
            (min: `{:?}`, x: `{:?}`)",
            *x,
            *max
        );
    }};
}
#[cfg(test)]
mod bernoulli_tests {

    use super::*;

    #[test]
    fn test_pdf() {
        let distribution: Bernoulli =
            Bernoulli::new(0.3).expect("Parameter should be a valid probability");
        assert_eq!(distribution.pmf(0.0), 0.7);
        assert_eq!(distribution.pmf(1.0), 0.3);
        assert_eq!(distribution.pmf(0.5), 0.0);
    }

    #[test]
    fn test_sample_multiple() {
        let distribution: Bernoulli =
            Bernoulli::new(0.3).expect("Parameter should be a valid probability");
        let samples: Vec<f64> = distribution.sample_multiple(1000);
        assert_eq!(samples.len(), 1000);
        assert!(samples.iter().all(|&x| x == 0.0 || x == 1.0));
    }

    #[test]
    fn test_cdf_multiple() {
        let distribution: Bernoulli =
            Bernoulli::new(0.3).expect("Parameter should be a valid probability");
        let points: Vec<f64> = vec![-1.0, 0.5, 1.5];
        let cdf_values: Vec<f64> = distribution.cdf_multiple(&points);
        assert_eq!(cdf_values, vec![0.0, 0.7, 1.0]);
    }
}

#[cfg(test)]
mod poisson_tests {
    use super::*;

    #[test]
    fn test_poisson_pdf() {
        let poisson: Poisson = Poisson::new(3.0).expect("Parameter should be valid");
        assert_approx_eq!(poisson.pmf(0.0), 0.049787068367863944);
        assert_approx_eq!(poisson.pmf(1.0), 0.14936120510359183);
        assert_approx_eq!(poisson.pmf(3.0), 0.22404180765538775);
        assert_approx_eq!(poisson.pmf(5.0), 0.1008181344474244);
    }

    #[test]
    fn test_poisson_cdf_multiple() {
        let poisson: Poisson = Poisson::new(2.0).expect("Parameter should be valid");
        let points: Vec<f64> = vec![0.0, 1.0, 2.0, 5.0];
        let cdf_values: Vec<f64> = poisson.cdf_multiple(&points);
        assert_approx_eq!(cdf_values[3], 0.9834371942939481);
        assert_approx_eq!(cdf_values[2], 0.6766764161830634);
        assert_approx_eq!(cdf_values[1], 0.4060058497098381);
        //println!("First pass. ");
        assert_approx_eq!(cdf_values[0], 0.1353352832366127);
    }

    #[test]
    fn test_poisson_sample_multiple() {
        let poisson: Poisson = Poisson::new(4.0).expect("Parameter should be valid");
        let samples: Vec<f64> = poisson.sample_multiple(1000);
        assert_eq!(samples.len(), 1000);
        // We can't test exact values, but we can verify that the values are within a reasonable range.
        // For example, we can check that the mean of the samples is close to the expected value.
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!((mean - 4.0).abs() < 0.5); //allow some tolerance
    }

    #[test]
    fn test_poisson_quantile_multiple() {
        let poisson: Poisson = Poisson::new(1.5).expect("Parameter should be valid");
        let points: Vec<f64> = vec![0.1, 0.5, 0.9];
        let quantiles: Vec<f64> = poisson.quantile_multiple(&points);

        assert_range!(0.0, quantiles[0], 0.999);
        assert_range!(0.0, quantiles[1], 1.999);
        assert_range!(2.0, quantiles[2], 4.999);
        //assert_approx_eq!(quantiles[0], 0.0);
        //assert_approx_eq!(quantiles[1], 1.0);
        //assert_approx_eq!(quantiles[2], 3.0);
    }

    #[test]
    fn test_poisson_expected_value() {
        let poisson: Poisson = Poisson::new(5.0).expect("Parameter should be valid");
        assert_eq!(poisson.expected_value().unwrap(), 5.0);
    }

    #[test]
    fn test_poisson_mode() {
        let poisson: Poisson = Poisson::new(6.0).expect("Parameter should be valid");
        assert_eq!(poisson.mode(), 6.0);

        let poisson: Poisson = Poisson::new(6.5).expect("Parameter should be valid");
        assert_eq!(poisson.mode(), 6.0); //floor of lambda in non integer cases.
    }

    #[test]
    fn test_poisson_median() {
        {
            let poisson: Poisson = Poisson::new(4.0).expect("Parameter should be valid");
            let median: f64 = poisson.median();
            //assert_approx_eq!(median, 3.968482618751532); // Approximate median.
            assert!(3.0 <= median && median <= 4.0, "{}", median);
        }

        {
            let poisson: Poisson = Poisson::new(8.25).expect("Parameter should be valid");
            let median: f64 = poisson.median();
            assert!(7.0 <= median && median <= 8.0, "{}", median);
        }

        {
            let poisson: Poisson = Poisson::new(1.45).expect("Parameter should be valid");
            let median: f64 = poisson.median();
            assert!(0.0 <= median && median <= 1.0, "{}", median);
        }
    }

    #[test]
    fn test_poisson_skewness() {
        let poisson: Poisson = Poisson::new(9.0).expect("Parameter should be valid");
        assert_approx_eq!(poisson.skewness().unwrap(), 1.0 / 3.0);
    }

    #[test]
    fn test_poisson_kurtosis() {
        let poisson: Poisson = Poisson::new(16.0).expect("Parameter should be valid");
        assert_approx_eq!(poisson.kurtosis().unwrap(), 3.0 + 1.0 / 16.0);
    }

    #[test]
    fn test_poisson_excess_kurtosis() {
        let poisson: Poisson = Poisson::new(25.0).expect("Parameter should be valid");
        assert_approx_eq!(poisson.excess_kurtosis().unwrap(), 1.0 / 25.0);
    }

    #[test]
    fn test_poisson_moments() {
        let poisson: Poisson = Poisson::new(7.0).expect("Parameter should be valid");

        // we are performing discrete integration, so we can expect less precision
        assert_approx_eq!(poisson.moments(1, Moments::Raw), 7.0, 0.001);
        assert_approx_eq!(poisson.moments(2, Moments::Raw), 56.0, 0.01);
        assert_approx_eq!(poisson.moments(2, Moments::Central), 7.0, 0.002);
        assert_approx_eq!(poisson.moments(3, Moments::Central), 7.0, 0.02);
        assert_approx_eq!(poisson.moments(2, Moments::Standarized), 1.0, 0.001);
        // 2nd standardized moment is always 1.0 for all distributions
    }

    #[test]
    fn test_poisson_entropy() {
        {
            let poisson: Poisson = Poisson::new(3.5).expect("Parameter should be valid");
            assert_approx_eq!(poisson.entropy(), 2.01517, 0.0001);
            // result from fromula wiki + wolframalpha
        }

        {
            let poisson: Poisson = Poisson::new(20.0).expect("Parameter should be valid");
            assert_approx_eq!(poisson.entropy(), 2.91253, 0.0001);
            // result from fromula wiki + wolframalpha
        }

        {
            let poisson: Poisson = Poisson::new(0.8).expect("Parameter should be valid");
            assert_approx_eq!(poisson.entropy(), 1.17838, 0.0001);
            // result from fromula wiki + wolframalpha
        }
    }

    #[test]
    fn test_poisson_confidence_interval() {
        let poisson: Poisson = Poisson::new(10.0).expect("Parameter should be valid");

        {
            let significance_level: f64 = 0.05; // 0.025 for each side
            let (lower, upper): (f64, f64) =
                poisson.confidence_interval(Hypothesis::default(), significance_level);

            // let (lower, upper): (f64, f64) = (4.0, 17.0); // correct

            // stricly
            let area_within: f64 = {
                let mut acc: f64 = 0.0;
                for x in (lower as i64)..=(upper as i64) {
                    acc += poisson.pmf(x as f64);
                }
                acc
            };

            let area_before: f64 = {
                let mut acc: f64 = 0.0;
                for x in 0..(lower as i64) {
                    acc += poisson.pmf(x as f64);
                }
                acc
            };

            let area_after: f64 = {
                let mut acc: f64 = 0.0;
                for x in (upper as i64 + 1)..200 {
                    acc += poisson.pmf(x as f64);
                }
                acc
            };

            println!("Area stricly within the c.i.: {:.6}", area_within);
            println!("lower: {} \tupper: {}", lower, upper);
            println!("Area before the c.i.: {:.6}", area_before);
            println!("Area after the c.i.: {:.6}", area_after);

            assert!(0.95 <= area_within);
            // panic!("Show results");
        }

        // let (lower, upper) = poisson.confidence_interval(Hypothesis::LeftTail, 0.05);
        // assert_approx_eq!(lower, 4.966031766487979);
        // assert_approx_eq!(upper, f64::INFINITY);

        // let (lower, upper) = poisson.confidence_interval(Hypothesis::RightTail, 0.05);
        // assert_approx_eq!(lower, f64::NEG_INFINITY);
        // assert_approx_eq!(upper, 15.033968233512021);
    }

    // TODO: creates test properly

    #[test]
    fn test_poisson_p_value() {
        /*
            Area stricly within the c.i.: 0.975386
            lower: 4        upper: 17
            Area before the c.i.: 0.010336
            Area after the c.i.: 0.014278
        */

        let poisson: Poisson = Poisson::new(10.0).expect("Parameter should be valid");
        let p_value: f64 = poisson.p_value(Hypothesis::TwoTailed, 18.0);
        println!("P value: {:.6}", p_value);
        assert!(p_value < 0.05);
        // panic!("Show results! ");
    }
}

#[cfg(test)]
mod normal_tests {

    use super::*;
    use AdvancedStatistics::{distribution_trait::Distribution, distributions::Normal::Normal};

    #[test]
    fn normal_pdf() {
        let params = [(0.0, 1.0), (2.0, 1.0), (0.0, 3.0), (-1.0, 0.4)];

        // results of pdf at 0.0, 1.0, -1.0, 4.0
        let correct_results = [
            [
                0.398942280401,
                0.241970724519,
                0.241970724519,
                0.000133830225765,
            ],
            [
                0.0539909665132,
                0.241970724519,
                0.00443184841194,
                0.0539909665132,
            ],
            [
                0.132980760134,
                0.125794409231,
                0.125794409231,
                0.054670024892,
            ],
            [
                0.0438207512339,
                0.00000371679878684,
                0.997355701004,
                1.1737988395e-34,
            ],
        ];
        // data obtained with desmos

        assert!(params.len() == correct_results.len());

        for (i, (m, s)) in params.iter().enumerate() {
            let normal: Normal = Normal::new(*m, *s).expect("Parameters should be ok. ");

            let results: Vec<f64> = vec![
                normal.pdf(0.0),
                normal.pdf(1.0),
                normal.pdf(-1.0),
                normal.pdf(4.0),
            ];
            assert_approx_eq!(results[0], correct_results[i][0]);
            assert_approx_eq!(results[1], correct_results[i][1]);
            assert_approx_eq!(results[2], correct_results[i][2]);
            assert_approx_eq!(results[3], correct_results[i][3]);
        }
    }

    #[test]
    fn normal_cdf() {
        let params = [(0.0, 1.0), (2.0, 1.0), (0.0, 3.0), (-1.0, 0.4)];

        // results of cdf at 0.0, 1.0, -1.0, 4.0
        let correct_results = [
            [0.5, 0.841344746069, 0.158655253931, 0.999968328758],
            [
                0.0227501319482,
                0.158655253931,
                0.00134989803163,
                0.977249868052,
            ],
            [0.5, 0.630558659818, 0.369441340182, 0.908788780274],
            [0.993790334674, 0.999999713348, 0.5, 1.0],
        ];
        // data obtained with desmos

        assert!(params.len() == correct_results.len());

        for (i, (m, s)) in params.iter().enumerate() {
            let normal: Normal = Normal::new(*m, *s).expect("Parameters should be ok. ");

            let results: Vec<f64> = vec![
                normal.cdf(0.0),
                normal.cdf(1.0),
                normal.cdf(-1.0),
                normal.cdf(4.0),
            ];

            assert_approx_eq!(results[0], correct_results[i][0]);
            assert_approx_eq!(results[1], correct_results[i][1]);
            assert_approx_eq!(results[2], correct_results[i][2]);
            assert_approx_eq!(results[3], correct_results[i][3]);
            println!("({m}, {s})");

            normal
                .cdf_multiple(&[0.0, 1.0, -1.0, 4.0])
                .iter()
                .zip(results.iter())
                .for_each(|(a, b)| assert_approx_eq!(a, b))
        }
    }
}
