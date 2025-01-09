///! General testing section
use distribution_trait::Distribution;
use distributions::Exponential::Exponential;
use domain::*;
use AdvancedStatistics::*;

#[test]
fn temporal() {
    let value: f64 = 1.0 / 2.0_f64.powi(15);
    print!("{} | {} \n", value, value.to_bits());
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
fn comparing_exponential_distr() {
    let lambda: f64 = 1.2;
    let max: f64 = -(0.0001_f64.ln()) / lambda;

    let ground: Exponential = Exponential::new(lambda).unwrap();
    let test: MyExp = MyExp {
        lambda: lambda,
        domain: ContinuousDomain::Range(0.0, max),
    };

    // ***
    println!("\nIntegral of the area of the pdf: \n\n");

    let area_ground: f64 = euclid::determine_normalitzation_constant_continuous(
        |x| Distribution::pdf(&ground, x),
        &test.domain,
    );
    let area_test: f64 = euclid::determine_normalitzation_constant_continuous(
        |x| Distribution::pdf(&test, x),
        &test.domain,
    );

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

    println!("\n\n================================\n\n");

    panic!("Show me the results. ");
}

#[test]
fn discrete_domain_iterators() {
    print!("\n\n");

    let amount: usize = 20;

    let tests: Vec<(&str, DiscreteDomain)> = vec![
        (
            "new_discrete_integers",
            DiscreteDomain::Integers
        ),
        (
            "new_discrete_positives",
            DiscreteDomain::From(0)
        ),
        (
            "new_discrete_negatives",
            DiscreteDomain::To(0)
        ),
        (
            "new_discrete_range (-3, 13)",
            DiscreteDomain::Range(-3, 13),
        ),
        (
            "new_discrete_from (-5)",
            DiscreteDomain::From(-5),
        ),
        (
            "new_discrete_to (2)",
            DiscreteDomain::To(2),
        ),
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
