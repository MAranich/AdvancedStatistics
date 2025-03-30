use AdvancedStatistics::{
    distribution_trait::DiscreteDistribution, distributions::Bernoulli::*,
    distributions::Poisson::*, euclid::*, hypothesis::Hypothesis,
};

#[inline]
fn assert_approx_eq(a: f64, b: f64) {
    let eps: f64 = 1.0e-6;

    assert!(
        (a - b).abs() < eps,
        "assertion failed: `(left !== right)` \
         (left: `{:?}`, right: `{:?}`, expect diff: `{:?}`, real diff: `{:?}`)",
        a,
        b,
        eps,
        (a - b).abs()
    );
}

#[inline]
fn assert_range_eq(min: f64, x: f64, max: f64) {

    if x < min {
        panic!("assertion failed: `(x < min)` \n(min: {:?}, x: {:?}, max: {:?})"
        , min, x, max); 
    }

    if x < min {
        panic!("assertion failed: `(max < x)` \n(min: {:?}, x: {:?}, max: {:?})"
        , min, x, max); 
    }

    if max < min {
        panic!("assertion failed: `(max < min)` \n(min: {:?}, x: {:?}, max: {:?})"
        , min, x, max); 
    }
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
        assert_approx_eq(poisson.pmf(0.0), 0.049787068367863944);
        assert_approx_eq(poisson.pmf(1.0), 0.14936120510359183);
        assert_approx_eq(poisson.pmf(3.0), 0.22404180765538775);
        assert_approx_eq(poisson.pmf(5.0), 0.1008181344474244);
    }

    #[test]
    fn test_poisson_cdf_multiple() {
        let poisson: Poisson = Poisson::new(2.0).expect("Parameter should be valid");
        let points: Vec<f64> = vec![0.0, 1.0, 2.0, 5.0];
        let cdf_values: Vec<f64> = poisson.cdf_multiple(&points);
        assert_approx_eq(cdf_values[3], 0.9834371942939481);
        assert_approx_eq(cdf_values[2], 0.6766764161830634);
        assert_approx_eq(cdf_values[1], 0.4060058497098381);
        //println!("First pass. ");
        assert_approx_eq(cdf_values[0], 0.1353352832366127);
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

        assert_range_eq(0.0, quantiles[0], 0.999); 
        assert_range_eq(0.0, quantiles[1], 1.999); 
        assert_range_eq(2.0, quantiles[2], 4.999); 
        //assert_approx_eq(quantiles[0], 0.0);
        //assert_approx_eq(quantiles[1], 1.0);
        //assert_approx_eq(quantiles[2], 3.0);
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
            //assert_approx_eq(median, 3.968482618751532); // Approximate median.
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
        assert_approx_eq(poisson.skewness().unwrap(), 1.0 / 3.0);
    }

    #[test]
    fn test_poisson_kurtosis() {
        let poisson: Poisson = Poisson::new(16.0).expect("Parameter should be valid");
        assert_approx_eq(poisson.kurtosis().unwrap(), 3.0+1.0/16.0);
    }

    #[test]
    fn test_poisson_excess_kurtosis() {
        let poisson: Poisson = Poisson::new(25.0).expect("Parameter should be valid");
        assert_approx_eq(poisson.excess_kurtosis().unwrap(), 1.0/25.0);
    }

    #[test]
    fn test_poisson_moments() {
        let poisson: Poisson = Poisson::new(7.0).expect("Parameter should be valid");
        assert_approx_eq(poisson.moments(1, Moments::Raw), 7.0);
        assert_approx_eq(poisson.moments(2, Moments::Raw), 56.0);
        assert_approx_eq(poisson.moments(2, Moments::Central), 7.0);
        assert_approx_eq(poisson.moments(3, Moments::Central), 7.0);
        assert_approx_eq(
            poisson.moments(2, Moments::Standarized),
            1.0 / 7.0f64.sqrt(),
        );
    }

    #[test]
    fn test_poisson_entropy() {
        let poisson: Poisson = Poisson::new(3.5).expect("Parameter should be valid");
        assert_approx_eq(poisson.entropy(), 2.0538965935397463);
    }

    #[test]
    fn test_poisson_confidence_interval() {
        let poisson: Poisson = Poisson::new(10.0).expect("Parameter should be valid");
        let (lower, upper) = poisson.confidence_interval(Hypothesis::default(), 0.05);

        // These values are approximate and depend on the exact implementation.
        assert_approx_eq(lower, 4.091176527552552);
        assert_approx_eq(upper, 15.908823472447448);

        let (lower, upper) = poisson.confidence_interval(Hypothesis::LeftTail, 0.05);
        assert_approx_eq(lower, 4.966031766487979);
        assert_approx_eq(upper, f64::INFINITY);

        let (lower, upper) = poisson.confidence_interval(Hypothesis::RightTail, 0.05);
        assert_approx_eq(lower, f64::NEG_INFINITY);
        assert_approx_eq(upper, 15.033968233512021);
    }

    #[test]
    fn test_poisson_p_value() {
        let poisson: Poisson = Poisson::new(5.0).expect("Parameter should be valid");
        let p_value: f64 = poisson.p_value(Hypothesis::TwoTailed, 7.0);
        assert_approx_eq(p_value, 0.616333535978712);

        let p_value: f64 = poisson.p_value(Hypothesis::LeftTail, 3.0);
        assert_approx_eq(p_value, 0.2650259152973715);

        let p_value: f64 = poisson.p_value(Hypothesis::RightTail, 8.0);
        assert_approx_eq(p_value, 0.1334033238692683);
    }
}
