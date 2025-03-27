use std::num::NonZero;

use AdvancedStatistics::{
    euclid::{digamma, fast_digamma},
    *,
};
///! General testing section
use distribution_trait::{DiscreteDistribution, Distribution};
use distributions::{Binomial::Binomial, Exponential::Exponential};
use domain::*;

#[test]
fn temporal() {
    println!("{:?}", 0.0);

    //panic!();
}

struct MyExp {
    lambda: f64,
    domain: ContinuousDomain,
}

impl Distribution for MyExp {
    fn pdf(&self, x: f64) -> f64 {
        self.lambda * (-self.lambda * x).exp()
    }

    fn get_domain(&self) -> &ContinuousDomain {
        &self.domain
    }

    /*

    fn expected_value(&self) -> Option<f64> {
        return Some(1.0 / self.lambda);
    }


    fn variance(&self) -> Option<f64> {
        return Some(1.0 / (self.lambda * self.lambda));
    }
     */
}

#[test]
fn deafult_methods_comparasion_continuous_exponential() {
    let lambda: f64 = 1.2;
    let max: f64 = -(0.0001_f64.ln()) / lambda;

    let ground: Exponential = Exponential::new(lambda).unwrap();
    let test: MyExp = MyExp {
        lambda: lambda,
        domain: ContinuousDomain::Range(0.0, max),
    };

    // ***

    println!("\nIntegral of the area of the pdf: \n\n");

    let area_ground: f64 =
        euclid::numerical_integration(|x| Distribution::pdf(&ground, x), &test.domain);
    let area_test: f64 =
        euclid::numerical_integration(|x| Distribution::pdf(&test, x), &test.domain);

    println!("Area || \tground: {}\ttest: {} ", area_ground, area_test);

    // ***

    println!("\nTesting cdf: \n\n");

    let points: Vec<f64> = vec![0.0, 0.1, 0.5, 0.75, 1.0, 1.2, 1.6, 2.5, 4.0, 10.0];
    let r1: Vec<f64> = ground.cdf_multiple(&points);
    let r2: Vec<f64> = test.cdf_multiple(&points);

    for (i, x) in points.iter().enumerate() {
        let diff: f64 = r1[i] - r2[i];
        println!(
            "\tcdf({}) || \nground:\t{}\ntest : \t{} \ndiff : \t{}\n",
            x, r1[i], r2[i], diff
        );
    }

    // ***

    println!("\nTesting quantile: \n\n");

    let points: Vec<f64> = vec![
        -1.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0,
    ];
    let r1: Vec<f64> = ground.quantile_multiple(&points);
    let r2: Vec<f64> = test.quantile_multiple(&points);

    for (i, x) in points.iter().enumerate() {
        let diff: f64 = r1[i] - r2[i];
        println!(
            "\tquantile({}) || \nground:\t{}\ntest : \t{} \ndiff : \t{}\n",
            x, r1[i], r2[i], diff
        );
    }

    /*
    for lambda = 1.2,
    cdf(7.67528364331348) = 0.9998999999999999

    7.675283643313579

     */

    // ***

    println!("\nStatistics: \n\n");

    println!(
        "Expected value: \n\t - ground: {:?}\n\t - test: {:?}\n",
        ground.expected_value(),
        test.expected_value()
    );

    println!(
        "Variance: \n\t - ground: {:?}\n\t - test: {:?}\n",
        ground.variance(),
        test.variance()
    );

    println!(
        "Skewness: \n\t - ground: {:?}\n\t - test: {:?}\n",
        ground.skewness(),
        test.skewness()
    );

    println!(
        "Mode: \n\t - ground: {:?}\n\t - test: {:?}\n",
        ground.mode(),
        test.mode()
    );

    println!("\n\n================================\n\n");

    panic!("Show me the results. ");
}

#[test]
fn discrete_domain_iterators() {
    print!("\n\n");

    let amount: usize = 20;

    let tests: Vec<(&str, DiscreteDomain)> = vec![
        ("new_discrete_integers", DiscreteDomain::Integers),
        ("new_discrete_positives", DiscreteDomain::From(0)),
        ("new_discrete_negatives", DiscreteDomain::To(0)),
        ("new_discrete_range (-3, 13)", DiscreteDomain::Range(-3, 13)),
        ("new_discrete_from (-5)", DiscreteDomain::From(-5)),
        ("new_discrete_to (2)", DiscreteDomain::To(2)),
        (
            "new_discrete_custom",
            DiscreteDomain::new_discrete_custom(&[-1.0, -2.0, -5.0, 5.0, 22.0, 344.0, 866.0]),
        ),
    ];

    for (log_text, domain) in tests {
        print!("Iterator for {}: \n\t", log_text);

        let mut iterator: DiscreteDomainIterator<'_> = domain.iter();
        let results: Vec<Option<f64>> = (0..amount).into_iter().map(|_i| iterator.next()).collect();

        print!("{:?}\n\n", results);
    }

    print!("\n\n");
    panic!("Show the results. ");
}

#[test]
fn binomial_coef_computation() {
    //https://www.omnicalculator.com/math/binomial-coefficient

    // This test only checks for valid cases

    // (n, k, `n choose k`)
    let results: Vec<(u64, u64, u128)> = vec![
        (0, 0, 1),
        (3, 0, 1),
        (4, 2, 6),
        (10, 10, 1),
        (12, 5, 792),
        (12, 6, 924),
        (14, 7, 3432),
        (14, 3, 364),
        (14, 2, 91),
        (18, 2, 153),
        (18, 16, 153),
        (18, 8, 43758),
        (19, 8, 75582),
        (50, 15, 2250829575120),
        (32, 1, 32),
        (32, 2, 496),
        (32, 3, 4960),
        (32, 4, 35960),
        (32, 5, 201376),
        (32, 6, 906192),
        (32, 7, 3365856),
        (32, 8, 10518300),
        (32, 9, 28048800),
        (32, 10, 64512240),
        (32, 11, 129024480),
        (32, 12, 225792840),
        (32, 13, 347373600),
        (32, 14, 471435600),
        (32, 15, 565722720),
        (32, 16, 601080390),
        (32, 17, 565722720),
        (32, 18, 471435600),
        (32, 19, 347373600),
        (32, 20, 225792840),
    ];

    for (n, k, correct_n_k) in results {
        let bin_coef: u128 = euclid::combinatorics::binomial_coefficient(n, k)
            .expect("No overflows for the examples. ");

        println!("({}|{}) = {}  ||| Result: {}", n, k, correct_n_k, bin_coef);

        assert!(bin_coef == correct_n_k);
    }
}

struct MyBinomial {
    domain: DiscreteDomain,
    p: f64,
    n: u64,
}

impl DiscreteDistribution for MyBinomial {
    fn pmf(&self, x: f64) -> f64 {
        let x_u: u64 = x.floor() as u64;
        let binomial_coef: u128 = euclid::combinatorics::binomial_coefficient(self.n, x_u).expect("The parameters of the binomial are too big. Our current implementation is not good enough. ");

        let prob_p: f64 = self.p.powi(x_u as i32);
        let prob_q: f64 = (1.0 - self.p).powi((self.n - x_u) as i32);

        return (binomial_coef as f64) * prob_p * prob_q;
    }

    fn get_domain(&self) -> &DiscreteDomain {
        return &self.domain;
    }
}

#[test]
fn deafult_methods_comparasion_discrete_binomial() {
    let p: f64 = 0.4;
    let n: u64 = 40;

    let ground: Binomial = Binomial::new(p, n).expect("Creation of this bimomial should succeed");
    let test: MyBinomial = MyBinomial {
        domain: DiscreteDomain::Range(0, n as i64),
        p,
        n,
    };

    // ***

    println!("\nTotal probability of the pmf: \n\n");

    let area_ground: f64 =
        euclid::discrete_integration(|x| ground.pmf(x), ground.get_domain(), Some(2 * n as usize));
    let area_test: f64 =
        euclid::discrete_integration(|x| test.pmf(x), test.get_domain(), Some(n as usize + 1));

    println!("Area || \tground: {}\ttest: {} ", area_ground, area_test);

    // ***

    println!("\nTesting cdf: \n\n");

    let points: Vec<f64> = vec![
        -1.0, 0.0, 2.0, 5.0, 10.0, 16.0, 17.0, 25.0, 32.0, 40.0, 45.0, 100.0,
    ];
    let r1: Vec<f64> = ground.cdf_multiple(&points);
    let r2: Vec<f64> = test.cdf_multiple(&points);

    for (i, x) in points.iter().enumerate() {
        let diff: f64 = r1[i] - r2[i];
        println!(
            "\tcdf({}) || \nground:\t{}\ntest : \t{} \ndiff : \t{}\n",
            x, r1[i], r2[i], diff
        );
    }

    // ***

    println!("\nTesting quantile: \n\n");

    let points: Vec<f64> = vec![
        -1.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0,
    ];
    let r1: Vec<f64> = ground.quantile_multiple(&points);
    let r2: Vec<f64> = test.quantile_multiple(&points);

    for (i, x) in points.iter().enumerate() {
        let diff: f64 = r1[i] - r2[i];
        println!(
            "\tquantile({}) || \nground:\t{}\ntest : \t{} \ndiff : \t{}\n",
            x, r1[i], r2[i], diff
        );
    }

    // ***

    println!("\nStatistics: \n\n");

    println!(
        "Expected value: \n\t - ground: {:?}\n\t - test: {:?}\n",
        ground.expected_value(),
        test.expected_value()
    );

    println!(
        "Variance: \n\t - ground: {:?}\n\t - test: {:?}\n",
        ground.variance(),
        test.variance()
    );

    println!(
        "Skewness: \n\t - ground: {:?}\n\t - test: {:?}\n",
        ground.skewness(),
        test.skewness()
    );

    println!(
        "Excess kurtosis: \n\t - ground: {:?}\n\t - test: {:?}\n",
        ground.excess_kurtosis(),
        test.excess_kurtosis()
    );

    println!(
        "Mode: \n\t - ground: {:?}\n\t - test: {:?}\n",
        ground.mode(),
        test.mode()
    );

    println!(
        "Median: \n\t - ground: {:?}\n\t - test: {:?}\n",
        ground.median(),
        test.median()
    );

    println!(
        "Entropy: \n\t - ground: {:?}\n\t - test: {:?}\n",
        ground.entropy(),
        test.entropy()
    );

    println!("\n\n================================\n\n");

    panic!("Show me the results. ");
}

#[test]
fn ln_gamma_precision() {
    /*
        we know that ln_gamma_int is correct because ti's derivated from it's definition,
        but it may be slow O(n) and only works for integers.
    */
    // maximum absolute error allowed
    let max_diff: f64 = 0.000045;
    // maximum relative error allowed
    let max_rel: f64 = 0.0000000000001;

    for i in 1..=260 {
        let ground: f64 = euclid::ln_gamma_int(NonZero::new(i as u64).unwrap());
        let alternative: f64 = euclid::ln_gamma(i as f64);
        let abs_diff: f64 = (ground - alternative).abs();
        let rel_err: f64 = abs_diff / ground;

        println!(
            "{}: Ground: {} \t Test: {}\t abs_diff: {}\tRelative: {}",
            i, ground, alternative, abs_diff, rel_err
        );
        assert!(abs_diff < max_diff);
        assert!(rel_err < max_rel || !rel_err.is_finite());
        // relative error when the true value is 0 explodes to inf
    }

    println!("**************************************************************************");
    println!("**************************************************************************");

    for i in 18..=52 {
        let ground: f64 = euclid::ln_gamma_int(NonZero::new((i * i) as u64).unwrap());
        let alternative: f64 = euclid::ln_gamma((i * i) as f64);
        let abs_diff: f64 = (ground - alternative).abs();
        let rel_err: f64 = abs_diff / ground;

        println!(
            "{}: Ground: {} \t Test: {}\t abs_diff: {}\tRelative: {}",
            i * i,
            ground,
            alternative,
            abs_diff,
            rel_err
        );
        assert!(abs_diff < max_diff);
        assert!(rel_err < max_rel);
    }

    println!("**************************************************************************");
    println!("**************************************************************************");
    println!("**************************************************************************");

    // doing powers of 1.5 so it does not syncronize with the ground impl.
    // max: 50 => 31.24s
    //      48 => 13.81s
    for i in 20..=48 {
        let x: f64 = 1.5_f64.powi(i).floor();
        let ground: f64 = euclid::ln_gamma_int(NonZero::new(x as u64).unwrap());
        let alternative: f64 = euclid::ln_gamma(x);
        let abs_diff: f64 = (ground - alternative).abs();
        let rel_err: f64 = abs_diff / ground;

        println!(
            "{}: Ground: {} \t Test: {}\t abs_diff: {}\tRelative: {}",
            x, ground, alternative, abs_diff, rel_err
        );
        assert!(abs_diff < max_diff);
        assert!(rel_err < max_rel);
    }

    // panic!("Show results! ");
}

#[test]
fn precision_fast_digamma() {
    // We will assume `digamma` is the absolute correct value.

    let points: [f64; 19] = [
        0.02, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.4, 2.6, 3.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0,
        512.0, 1024.0, 2048.0,
    ];
    // we don't care for values less than 0.01

    let max_abs_error_treshols: f64 = 0.01;

    for v in points {
        let ground: f64 = digamma(v);
        let aprox: f64 = fast_digamma(v);
        let diff: f64 = (ground - aprox).abs();
        println!("\t{}: (ground: {}, aprox: {})", v, ground, aprox);
        println!("abs_diff: {}, log_abs_diff: {}", diff, diff.ln());

        if max_abs_error_treshols <= diff {
            // some values near x = 2.5 or x = 0.0 may fulfill this but
            // we will just ignre them because they are very few.
            panic!("Error over the trehold in precision_fast_digamma");
        }
    }
}
