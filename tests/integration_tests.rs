use std::f64::consts::PI;

use AdvancedStatistics::{distribution_trait::*, domain::*, euclid};

struct FiniteDistr {
    domain: ContinuousDomain,
}

impl Distribution for FiniteDistr {
    fn pdf(&self, x: f64) -> f64 {
        let c: f64 = 5.0;
        let w: f64 = x - c;

        /*
        for c = 5, interval [0, 5.47347] area: 7.57585312457
         */
        let num: f64 = w * w * w + 4.0 * w - 2.0;
        let den: f64 = -3.0 * w * w - 2.0; // always non-zero for all real `w`
        return num / den;
    }

    fn get_domain(&self) -> &ContinuousDomain {
        &self.domain
    }
}

#[test]
fn integration_tests_finite() {
    println!("Integration test Finite");
    let d: FiniteDistr = FiniteDistr {
        domain: ContinuousDomain::Range(0.0, 5.47347),
    };

    let c: f64 = euclid::numerical_integration(|x| d.pdf(x), d.get_domain());

    assert!((c - 7.57585312457).abs() < 0.0000001);
}

struct InfToConstDistr {
    domain: ContinuousDomain,
}

impl Distribution for InfToConstDistr {
    fn pdf(&self, x: f64) -> f64 {
        // defined for (-inf, c]
        let c: f64 = 2.0;

        /*
           // for c = 2, area : 1.99571445585
           for c = 2, area : 2.39574891663
        */
        let log: f64 = (c - x + 1.0).ln();
        let den: f64 = x * x + c;
        let ret: f64 = log / den;

        if ret.is_nan() {
            panic!("Evaluated pdf at {x} and returned NaN. \nlog: {log}\tden: {den}");
        }
        return ret;
    }

    fn get_domain(&self) -> &ContinuousDomain {
        &self.domain
    }
}
#[test]
fn integration_tests_inf_to_const() {
    println!("Integration test inf to const");

    let c: f64 = 2.0;
    let d: InfToConstDistr = InfToConstDistr {
        domain: ContinuousDomain::To(c),
    };

    let c: f64 = euclid::numerical_integration(|x| d.pdf(x), d.get_domain());
    let expected_ret: f64 = 2.39574891663;

    println!("{} || {}", c, expected_ret);
    assert!((c - expected_ret).abs() < 0.0001);
}

struct ConstToInfDistr {
    domain: ContinuousDomain,
}

impl Distribution for ConstToInfDistr {
    fn pdf(&self, x: f64) -> f64 {
        // defined for [c, inf]
        let c: f64 = -5.0;
        /*
           area : 12 for any c
        */
        let sqrt: f64 = (x - c).sqrt();
        let log: f64 = (x - c).ln();
        let ret: f64 = (log - sqrt).exp();

        if ret.is_nan() {
            panic!("Evaluated pdf at {x} and returned NaN. ");
        }
        return ret;
    }

    fn get_domain(&self) -> &ContinuousDomain {
        &self.domain
    }
}

#[test]
fn integration_tests_const_to_inf() {
    println!("Integration test const to inf");

    let D: ConstToInfDistr = ConstToInfDistr {
        domain: ContinuousDomain::From(-5.0),
    };

    let c: f64 = euclid::numerical_integration(|x| D.pdf(x), D.get_domain());

    println!("{} || {}", c, 12.0);
    assert!((c - 12.0).abs() < 0.0000001);
}

struct InfiniteDistr {
    domain: ContinuousDomain,
}

impl Distribution for InfiniteDistr {
    fn pdf(&self, x: f64) -> f64 {
        // area: sqrt(pi)
        let ret: f64 = (-x * x).exp();
        if ret.is_nan() {
            panic!("Evaluated pdf at {x} and returned NaN. ");
        }
        return ret;
    }

    fn get_domain(&self) -> &ContinuousDomain {
        &self.domain
    }
}

#[test]
fn integration_tests_infinite() {
    println!("Integration test inf");

    let d: InfiniteDistr = InfiniteDistr {
        domain: ContinuousDomain::Reals,
    };

    let c: f64 = euclid::numerical_integration(|x| d.pdf(x), d.get_domain());

    println!("{} || {}", c * c, PI);
    assert!((c * c - PI).abs() < 0.0000001);
}
