//! # Normal distribution
//!
//! The [Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution)
//! ia a very important continuous probability distribution.
//!
//! This distribution is very frequent in statistics and extremly well studied.
//! It also has many properties, like maximum entropy or being it's own
//! conjugate distribution. It also has a key role en the [Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem)
//! (CLT), wich is a key theorem that says that the sum of `n` random variables
//! of **any** distribution will give a new random variable that is normally distributed
//! as `n` grows to infinity (although 30 samples are usually enough for most practical
//! univariate cases).  
//!
//! We implement the [Normal] distribution and the [StdNormal], wich is the same as [Normal]
//! but for fixed `mean = 0.0` and `std_dev = 1.0`.
//!

use rand::Rng;
use std::f64::consts::{E, PI};

use crate::{
    distribution_trait::{Distribution, Parametric},
    domain::ContinuousDomain,
    euclid,
};

// coefitients for the (aprox) computation of the inverse cdf of the std normal
const B_ZERO_COEFITIENT: f64 = 2.92678600515804815402;
const B_ONE_COEFITIENTS: [f64; 5] = [
    8.97280659046817350354,
    10.27157061171363078863,
    12.72323261907760928036,
    16.88639562007936907786,
    24.12333774572479110372,
];

const B_TWO_COEFITIENTS: [f64; 5] = [
    5.81582518933527390512,
    5.70347935898051436684,
    5.51862483025707963145,
    5.26184239579604207321,
    4.92081346632882032881,
];

const C_ONE_COEFITIENTS: [f64; 5] = [
    11.61511226260603247078,
    18.25323235347346524796,
    18.38871225773938486923,
    18.61193318971775795045,
    24.14804072812762821134,
];

const C_TWO_COEFITIENTS: [f64; 5] = [
    3.83362947800146179416,
    7.30756258553673541139,
    8.42742300458043240405,
    5.66479518878470764762,
    4.91396098895240075156,
];

// The numbers are sorted from greatest power coef. to least for better caching.
const SECTION_0_NUM: [f64; 7] = [
    0.0003208627452884811,
    0.0029441556362517027,
    0.009257272344987084,
    0.020003594190859522,
    0.10840412072985979,
    0.3989422804014327,
    0.5,
];
const SECTION_0_DEN: [f64; 7] = [
    0.0006417254905769622,
    0.0,
    0.01851454468997417,
    0.0,
    0.21680824145971958,
    0.0,
    1.0,
];

// setction near 2.5
const SECTION_25_NUM: [f64; 7] = [
    0.0004777055635740545,
    0.004959895473075369,
    0.02716949987611552,
    0.1170167930359141,
    0.33675493562026104,
    0.6331862336229754,
    0.99379,
];
const SECTION_25_DEN: [f64; 7] = [
    0.0005150688335036621,
    0.004518125531525654,
    0.029528985481783798,
    0.10980043913651164,
    0.34997983988192594,
    0.6195050600458603,
    1.0,
];

// section near 4.0
const SECTION_4_NUM: [f64; 7] = [
    0.004050436501176754,
    0.03868273057953287,
    0.18470344342551243,
    0.560398719825338,
    1.138259393319791,
    1.4781165616099314,
    0.999968,
];
const SECTION_4_DEN: [f64; 7] = [
    0.004051295899278642,
    0.03867514518753098,
    0.18473451563610907,
    0.5603253380140956,
    1.138365676762724,
    1.4780300285708456,
    1.0,
];

#[derive(Debug, Clone, PartialEq)]
pub struct StdNormal {
    domain: ContinuousDomain,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Normal {
    std_normal: StdNormal,
    /// The mean of the distribution
    mean: f64,
    /// The standard deviation of the distribution
    standard_deviation: f64,
}

pub struct StdNormalGenerator {
    rng: rand::prelude::ThreadRng,
}

pub struct NormalGenerator {
    rng: StdNormalGenerator,
    minus_mean: f64,
    inv_std_dev: f64,
}

impl StdNormal {
    /// Create a Standard normal distribution. Has a mean of `0.0` and a standard
    /// deviation of `1.0`.
    pub const fn new() -> StdNormal {
        return StdNormal {
            domain: ContinuousDomain::Reals,
        };
    }

    /// Returns an iterator that can generate [StdNormal] samples even faster
    /// than normally calling [StdNormal::sample] many times. Uscefull if you don't
    /// know exacly how many values you want for [StdNormal::sample_multiple].
    ///
    /// It avoids the heap allocation of [StdNormal::sample_multiple] and
    /// the repeated initialitzation processes in [StdNormal::sample].
    pub fn iter(&self) -> StdNormalGenerator {
        StdNormalGenerator {
            rng: rand::rng(),
        }
    }
}

impl Normal {
    /// Create a [Normal] distribution.
    ///
    ///  - The `mean` must be finite (No `+-inf` or NaNs)
    ///  - The `standard_deviation` must be finite (No `+-inf` or NaNs)
    ///  - The `standard_deviation` must be stricly greater than `0.0`.
    ///
    /// If those conditions are not fullfiled, an error will be returned.
    pub const fn new(mean: f64, standard_deviation: f64) -> Result<Normal, ()> {
        if !mean.is_finite() || !standard_deviation.is_finite() || standard_deviation < 0.0 {
            return Err(());
        }

        let std_normal: StdNormal = StdNormal::new();
        return Ok(Normal {
            std_normal,
            mean,
            standard_deviation,
        });
    }

    /// Create a [Normal] distribution without checking for the corrrectness of the inputs.
    ///
    ///  - The `mean` must be finite (No `+-inf` or NaNs)
    ///  - The `standard_deviation` must be finite (No `+-inf` or NaNs)
    ///  - The `standard_deviation` must be stricly greater than `0.0`.
    ///
    /// If those conditions are not fullfiled, the returned distribution
    /// will be invalid.
    pub const unsafe fn new_unchecked(mean: f64, standard_deviation: f64) -> Normal {
        let std_normal: StdNormal = StdNormal::new();
        return Normal {
            std_normal,
            mean,
            standard_deviation,
        };
    }

    /// Returns the mean, the first parameter of the normal distribution.
    pub const fn get_mean(&self) -> f64 {
        return self.mean;
    }

    /// Returns the standard deviation, the second parameter of the normal distribution.
    pub const fn get_standard_deviation(&self) -> f64 {
        return self.standard_deviation;
    }

    /// Returns an iterator that can generate [Normal] samples even faster
    /// than normally calling [Normal::sample] many times. Uscefull if you don't
    /// know exacly how many values you want for [Normal::sample_multiple].
    ///
    /// It avoids the heap allocation of [Normal::sample_multiple] and
    /// the repeated initialitzation processes in [Normal::sample].
    pub fn iter(&self) -> NormalGenerator {
        let std: StdNormalGenerator = StdNormalGenerator {
            rng: rand::rng(),
        };

        return NormalGenerator {
            rng: std,
            minus_mean: -self.mean,
            inv_std_dev: 1.0 / self.standard_deviation,
        };
    }
}

impl Distribution for StdNormal {
    fn pdf(&self, x: f64) -> f64 {
        return euclid::INV_SQRT_2_PI * (-x * x * 0.5).exp();
    }

    fn get_domain(&self) -> &ContinuousDomain {
        return &self.domain;
    }

    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            // x is not valid
            std::panic!("Tried to evaluate the cdf function of StdNormal with a NaN value. \n");
        }
        let aux: [f64; 1] = [x];
        let aux_2: Vec<f64> = self.cdf_multiple(&aux);
        return aux_2[0];
    }

    fn sample(&self) -> f64 {
        let aux: Vec<f64> = self.sample_multiple(1);
        return aux[0];
    }

    fn quantile(&self, x: f64) -> f64 {
        // just call [Distribution::quantile_multiple]

        if x.is_nan() {
            // x is not valid
            std::panic!(
                "Tried to evaluate the quantile function of StdNormal with a NaN value. \n"
            );
        }

        let value: [f64; 1] = [x];
        let quantile_vec: Vec<f64> = self.quantile_multiple(&value);
        return quantile_vec[0];
    }

    fn cdf_multiple(&self, points: &[f64]) -> Vec<f64> {
        /*
        We will use the aproximation by:
        Dia, Yaya D. (2023). "Approximate Incomplete Integrals, Application to Complementary Error Function". SSRN. doi:10.2139/ssrn.4487559. S2CID 259689086.

        The precision of this method is extremly high: an error of less than
        `~1.1 * 10^-16 ~= 2^-53`. Considering that
        `f64::EPSILON = 2.220446049250313e-16 ~= 2.22 * 10^-16`, this solution
        may as well be considered exact if we are working with `f64`.

        ***

        To evaluate the polynomials we will do Horner's rule for efficiency:
        https://en.wikipedia.org/wiki/Polynomial_evaluation#Horner's_rule

        ```
        x^2 + a_1 * x + a_2 =
         = (x + a_1) * x + a_2
        ```

        ***
        For better efficiency we will use `f64::mul_add`.
        `x.mul_add(a, b) = x * a + b`

        */

        let mut ret: Vec<f64> = Vec::with_capacity(points.len());
        for pnt in points {
            let (point, flipped): (f64, bool) = if *pnt < 0.0 {
                (-*pnt, true)
            } else {
                (*pnt, false)
            };

            // let term_zero: f64 = 1.0 / (point + b_zero_coefitient);

            let term_1_num: f64 =
                (point + C_TWO_COEFITIENTS[0]).mul_add(point, C_ONE_COEFITIENTS[0]);
            let term_1_den: f64 =
                (point + B_TWO_COEFITIENTS[0]).mul_add(point, B_ONE_COEFITIENTS[0]);

            let term_2_num: f64 =
                (point + C_TWO_COEFITIENTS[1]).mul_add(point, C_ONE_COEFITIENTS[1]);
            let term_2_den: f64 =
                (point + B_TWO_COEFITIENTS[1]).mul_add(point, B_ONE_COEFITIENTS[1]);

            let term_3_num: f64 =
                (point + C_TWO_COEFITIENTS[2]).mul_add(point, C_ONE_COEFITIENTS[2]);
            let term_3_den: f64 =
                (point + B_TWO_COEFITIENTS[2]).mul_add(point, B_ONE_COEFITIENTS[2]);

            let term_4_num: f64 =
                (point + C_TWO_COEFITIENTS[3]).mul_add(point, C_ONE_COEFITIENTS[3]);
            let term_4_den: f64 =
                (point + B_TWO_COEFITIENTS[3]).mul_add(point, B_ONE_COEFITIENTS[3]);

            let term_5_num: f64 =
                (point + C_TWO_COEFITIENTS[4]).mul_add(point, C_ONE_COEFITIENTS[4]);
            let term_5_den: f64 =
                (point + B_TWO_COEFITIENTS[4]).mul_add(point, B_ONE_COEFITIENTS[4]);

            let numerator: f64 = term_1_num * term_2_num * term_3_num * term_4_num * term_5_num;
            let denomiantor: f64 = term_1_den * term_2_den * term_3_den * term_4_den * term_5_den;

            let m: f64 = numerator / (denomiantor * (point + B_ZERO_COEFITIENT));
            // `aproximation` = `1 - cdf(x)`
            let aproximation: f64 = m * self.pdf(point);

            ret.push(if flipped {
                aproximation
            } else {
                1.0 - aproximation
            });
        }

        return ret;
    }

    fn sample_multiple(&self, n: usize) -> Vec<f64> {
        /*
           For sampling, we do not need as much precision as for
           [StdNormal::quantile] or [StdNormal::cdf]. Therefore
           we will use a faster but sligtly less precise aproximation
           for the cdf and then use Newton's method to get the solution.

           Note that `1 - Phi(x) = Phi(-x)` therefore we only need to solve
           for `x <= 0`. The aproximations for the cdf(x) are the following:

           Interval [-1.4, 1.4]:
           > (0.5 + 0.3989422804014327*x + 0.10840412072985979*x^2 + 0.020003594190859522*x^3 + 0.009257272344987084*x^4 + 0.0029441556362517027*x^5 + 0.0003208627452884811*x^6) / (1 + 0.21680824145971958*x^2 + 0.01851454468997417*x^4 + 0.0006417254905769622*x^6)

           Interval [1.2, 4.3]:
           > (0.99379 + 0.6331862336229754*(x-2.5)^1 + 0.33675493562026104*(x-2.5)^2 + 0.1170167930359141*(x-2.5)^3 + 0.02716949987611552*(x-2.5)^4 + 0.004959895473075369*(x-2.5)^5 + 0.0004777055635740545*(x-2.5)^6) / (1 + 0.6195050600458603*(x-2.5)^1 + 0.34997983988192594*(x-2.5)^2 + 0.10980043913651164*(x-2.5)^3 + 0.029528985481783798*(x-2.5)^4 + 0.004518125531525654*(x-2.5)^5 + 0.0005150688335036621*(x-2.5)^6)

           Interval [3, 6]:
           > (0.999968 + 1.4781165616099314*(x-4.0) + 1.138259393319791*(x-4.0)^2 + 0.560398719825338*(x-4.0)^3 + 0.18470344342551243*(x-4.0)^4 + 0.03868273057953287*(x-4.0)^5 + 0.004050436501176754*(x-4.0)^6) / (1 + 1.4780300285708456*(x-4.0) + 1.138365676762724*(x-4.0)^2 + 0.5603253380140956*(x-4.0)^3 + 0.18473451563610907*(x-4.0)^4 + 0.03867514518753098*(x-4.0)^5 + 0.004051295899278642*(x-4.0)^6)

           For their respective intervals their maximum absolute error is arround or under
           `e^-15 ~= 3.06 * 10^-7`, wich is an error considerably higher than the one at
           [StdNormal::cdf] (`1.1 * 10^-16`)

           Note, for values  `6.0 <= x <= 8.0` we will still use the aproximation centered
           at 4.0 because it's still very precise and it would be a very rate occurence.
           For values `8.0 <= z` we will map them direcly to 1.0, the chanve of this happening
           is extermly unlikely.

           ***

           To evaluate the polynomials we will do Horner's rule for efficiency:
           https://en.wikipedia.org/wiki/Polynomial_evaluation#Horner's_rule

        */
        let mut rng: rand::prelude::ThreadRng = rand::rng();
        let mut rand_quantiles: Vec<f64> = std::vec![0.0; n];
        rng.fill(rand_quantiles.as_mut_slice());

        for rand_q in &mut rand_quantiles {
            // just map r to the awnser

            let (q, flipped): (f64, bool) = if *rand_q < 0.5 {
                (1.0 - *rand_q, true)
            } else {
                (*rand_q, false)
            };

            assert!(0.5 <= q && q <= 1.0);

            // [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method)

            let mut r: f64 = 4.0 * q - 2.2;
            let mut last: f64 = -128.0; // arbitrary number, just far away from r
                                        // ^seed (bad aproximation for inv cdf(x) [0.5, 1.0] but good enough as first guess)
            'newton_loop: loop {
                let cdf: f64 = if r < 1.4 {
                    // covers 83.8487 % of cases
                    // method 0

                    //Horner's rule
                    let numerator: f64 = SECTION_0_NUM[0]
                        .mul_add(r, SECTION_0_NUM[1])
                        .mul_add(r, SECTION_0_NUM[2])
                        .mul_add(r, SECTION_0_NUM[3])
                        .mul_add(r, SECTION_0_NUM[4])
                        .mul_add(r, SECTION_0_NUM[5])
                        .mul_add(r, SECTION_0_NUM[6]);

                    let denominator: f64 = SECTION_0_DEN[0]
                        .mul_add(r, SECTION_0_DEN[1])
                        .mul_add(r, SECTION_0_DEN[2])
                        .mul_add(r, SECTION_0_DEN[3])
                        .mul_add(r, SECTION_0_DEN[4])
                        .mul_add(r, SECTION_0_DEN[5])
                        .mul_add(r, SECTION_0_DEN[6]);

                    numerator / denominator
                } else if r < 3.0 {
                    // covers 99.73% of cases
                    // method 2.5

                    //Horner's rule
                    let numerator: f64 = SECTION_25_NUM[0]
                        .mul_add(r, SECTION_25_NUM[1])
                        .mul_add(r, SECTION_25_NUM[2])
                        .mul_add(r, SECTION_25_NUM[3])
                        .mul_add(r, SECTION_25_NUM[4])
                        .mul_add(r, SECTION_25_NUM[5])
                        .mul_add(r, SECTION_25_NUM[6]);

                    let denominator: f64 = SECTION_25_DEN[0]
                        .mul_add(r, SECTION_25_DEN[1])
                        .mul_add(r, SECTION_25_DEN[2])
                        .mul_add(r, SECTION_25_DEN[3])
                        .mul_add(r, SECTION_25_DEN[4])
                        .mul_add(r, SECTION_25_DEN[5])
                        .mul_add(r, SECTION_25_DEN[6]);

                    numerator / denominator
                } else if r < 8.0 {
                    // covers 99.99999999999987558078851456431752968% of cases (according to wolframalpha)
                    // method 4
                    // remember that we will still use this method up to 8.0
                    // even if it doen not retain as much precision.
                    //Horner's rule
                    let numerator: f64 = SECTION_4_NUM[0]
                        .mul_add(r, SECTION_4_NUM[1])
                        .mul_add(r, SECTION_4_NUM[2])
                        .mul_add(r, SECTION_4_NUM[3])
                        .mul_add(r, SECTION_4_NUM[4])
                        .mul_add(r, SECTION_4_NUM[5])
                        .mul_add(r, SECTION_4_NUM[6]);

                    let denominator: f64 = SECTION_4_DEN[0]
                        .mul_add(r, SECTION_4_DEN[1])
                        .mul_add(r, SECTION_4_DEN[2])
                        .mul_add(r, SECTION_4_DEN[3])
                        .mul_add(r, SECTION_4_DEN[4])
                        .mul_add(r, SECTION_4_DEN[5])
                        .mul_add(r, SECTION_4_DEN[6]);

                    numerator / denominator
                } else {
                    // numbers greater than 8.0 are mapped to 0.0 to avoid prolems
                    // this is **extremly** unlikely anyway. probability of ending
                    // in this region: 0.00000000000012441921148543568247032% (according to wolframalpha)
                    r = 0.0;
                    break 'newton_loop;
                };

                /*
                Newton's method: for finding a root for a function f(x)

                x_n+1 = x_n - f(x_n)/f'(x_n)

                In pur particular case:

                r_n+1 = r_n - (cdf(r_n) - q)/pdf(r_n)

                knowing that:
                pdf(x) = 1/sqrt(2*pi) * exp(-0.5 * x^2)
                1/pdf(x) = sqrt(2*pi) * exp(0.5 * x^2)
                 = inv_pdf(x)

                Therefore:
                r_n+1 = r_n - (cdf(r_n) - q) * inv_pdf(r_n)
                r_n+1 = r_n - (cdf(r_n) - q) * sqrt(2*pi) * exp(0.5 * r_n^2)
                r_n+1 = r_n + (q - cdf(r_n)) * sqrt(2*pi) * exp(0.5 * r_n * r_n)

                So, the only non-trivial computations are `cdf(x)` and `exp(x)`.

                */
                // Instead of computing 1/pdf; compute a modified version of pdf
                let sqrt_2_pi: f64 = (2.0 * PI).sqrt();
                let inv_pdf: f64 = sqrt_2_pi * (0.5 * r * r).exp();
                r = r + (q - cdf) * inv_pdf;

                if (r - last).abs() < 0.00001 {
                    break;
                }
                last = r;
            }

            *rand_q = if flipped { -r } else { r };
        }

        return rand_quantiles;
    }

    fn quantile_multiple(&self, points: &[f64]) -> Vec<f64> {
        /*
            Plan:

            We will just use Newton's method with the `cdf` function.
            It will give us a lot of precision and a better performance
            than just integrating numerically.
        */

        if points.is_empty() {
            return Vec::new();
        }

        // return error if NAN is found
        for (i, point) in points.iter().enumerate() {
            if point.is_nan() {
                std::panic!(
                    "Found NaN in `quantile_multiple` for StdNormal in position {}. \n",
                    i
                );
            }
        }

        let ret: Vec<f64> = points
            .iter()
            .map(|&q| 'quantile: {
                if q <= 0.0 {
                    break 'quantile f64::NEG_INFINITY;
                } else if 1.0 <= q {
                    break 'quantile f64::INFINITY;
                }

                let mut last_guess: f64 = -128.0;
                // ^arbitrary value but far away from `guess`
                let mut guess: f64 = 1.25331413732 + 2.50662827463 * q;
                // ^initial guess, 1 deg. Taylor series of quantile(x).
                let final_guess: f64 = loop {
                    /*
                    Newton's method: for finding a root for a function f(x)

                    x_n+1 = x_n - f(x_n)/f'(x_n)

                    In pur particular case:

                    g_n+1 = g_n - (cdf(g_n) - q)/pdf(g_n)

                    knowing that:
                    pdf(x) = 1/sqrt(2*pi) * exp(-0.5 * x^2)
                    1/pdf(x) = sqrt(2*pi) * exp(0.5 * x^2)
                     = inv_pdf(x)

                    Therefore:
                    g_n+1 = g_n - (cdf(g_n) - q) * inv_pdf(g_n)
                    g_n+1 = g_n - (cdf(g_n) - q) * sqrt(2*pi) * exp(0.5 * g_n^2)
                    g_n+1 = g_n + (q - cdf(g_n)) * sqrt(2*pi) * exp(0.5 * g_n * g_n)

                    So, the only non-trivial computations are `cdf(x)` and `exp(x)`.

                    */

                    let sqrt_2_pi: f64 = (2.0 * PI).sqrt();
                    let inv_pdf: f64 = sqrt_2_pi * (0.5 * guess * guess).exp();
                    guess = guess + (q - self.cdf(q)) * inv_pdf;

                    if (guess - last_guess).abs() < 0.000000001 {
                        break guess;
                    }
                    last_guess = guess;
                };

                final_guess
            })
            .collect::<Vec<f64>>();

        return ret;
    }

    fn expected_value(&self) -> Option<f64> {
        return Some(0.0);
    }

    fn variance(&self) -> Option<f64> {
        return Some(1.0);
    }

    fn mode(&self) -> f64 {
        return 0.0;
    }

    fn median(&self) -> f64 {
        return 0.0;
    }

    fn skewness(&self) -> Option<f64> {
        return Some(0.0);
    }

    fn kurtosis(&self) -> Option<f64> {
        return self.excess_kurtosis().map(|x| x + 3.0);
    }

    fn excess_kurtosis(&self) -> Option<f64> {
        return Some(0.0);
    }

    fn moments(&self, order: u8, mode: crate::euclid::Moments) -> f64 {
        /*

               Plan:

            Just to the integral. The integral that gives us the moments of order `k` is:

            ```
            integral {a -> b} ( (x - mu) / std )^k * f(x) dx
            ```
             - `k` is the order of the moment
             - `f(x)` is the pdf of the distribution.
             - `a` and `b` are the values that bound the domain of `f(x)`
                    (they can be `a = -inf` and `b = -inf`).
             - `mu` is the mean of the distribution (or `0` if we selected the `Raw` moment)
             - `std` is the standard deviation of the distribution
                    (or `1` if we did not select the `Standarized` moment)


           Distiguish between cases depending on the domain.

           We will integrate using [Simpson's rule](https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_1/3_rule)
           for integration.

           To compute integrals over an infinite range, we will perform a special
           [numerial integration](https://en.wikipedia.org/wiki/Numerical_integration#Integrals_over_infinite_intervals).

            let g(x) = ( (x - mu) / std )^k * f(x)
                For -infinite to const:
            integral {-inf -> a} g(x) dx = integral {0 -> 1} g(a - (1 - t)/t)  /  t^2  dt
            integral {-inf -> a} g(x) dx = integral {0 -> 1} ( (a - (1 - t)/t - mu) / std )^k * f(a - (1 - t)/t)  /  t^2  dt

                For const to infinite:
            integral {a -> inf} g(x) dx  = integral {0 -> 1} g(a + t/(t - 1))  /  (1 - t)^2  dt
            integral {a -> inf} g(x) dx  = integral {0 -> 1} ( (a + t/(t - 1) - mu) / std )^k * f(a + t/(t - 1))  /  (1 - t)^2  dt

                For -infinite to infinite:
            let inp = t/(1 - t^2)
            integral {-inf -> inf} g(x) dx  = integral {-1 -> 1} g(t/(1 - t^2))  *  (1 + t^2) / (1 - t^2)^2  dt
            integral {-inf -> inf} g(x) dx  = integral {-1 -> 1} ( (t/(1 - t^2) - mu) / std )^k * f(t/(1 - t^2))  *  (1 + t^2) / (1 - t^2)^2  dt


        */

        let domain: &ContinuousDomain = self.get_domain();
        let bounds: (f64, f64) = domain.get_bounds();

        // The values of 0.0 and 1.0 have no special meaning. They are not going to be used anyway.
        let (mean, std_dev): (f64, f64) = match mode {
            crate::euclid::Moments::Raw => (0.0, 1.0),
            crate::euclid::Moments::Central => (
                self.expected_value()
                    .expect("Tried to compute a central moment but the expected value is undefined. "),
                1.0,
            ),
            crate::euclid::Moments::Standarized => (
                self.expected_value()
                    .expect("Tried to compute a central/standarized moment but the Expected value is undefined. "),
                self.variance().expect("Tried to compute a standarized moment but the variance is undefined. "),
            ),
        };

        // Todo: give better error handling to the above. ^
        // println!("(mean, std_dev): {:?}", (mean, std_dev));

        let order_exp: i32 = order as i32;
        let (minus_mean, inv_std_dev) = (-mean, 1.0 / std_dev.sqrt());
        let integration_type: crate::euclid::IntegrationType =
            crate::euclid::IntegrationType::from_bounds(bounds);
        let (_, num_steps): (f64, usize) =
            crate::euclid::choose_integration_precision_and_steps(bounds);

        let moment: f64 = match integration_type {
            crate::euclid::IntegrationType::Finite => {
                let integration_fn = |x: f64| {
                    let std_inp: f64 = (x + minus_mean) * inv_std_dev;
                    std_inp.powi(order_exp) * self.pdf(x)
                };

                crate::euclid::numerical_integration_finite(
                    integration_fn,
                    bounds,
                    num_steps as u64,
                )
            }
            crate::euclid::IntegrationType::InfiniteToConst => {
                // integral {-inf -> a} f(x) dx = integral {0 -> 1} f(a - (1 - t)/t)  /  t^2  dt
                let integration_fn = |x: f64| 'integration: {
                    // x will go from 0.0 to 1.0
                    if x.abs() < f64::EPSILON {
                        // too near singularity, skip
                        break 'integration 0.0;
                    }
                    let inv_x: f64 = 1.0 / x;
                    let fn_input: f64 = bounds.1 - (1.0 - x) * inv_x;
                    let std_inp: f64 = (fn_input + minus_mean) * inv_std_dev;
                    std_inp.powi(order_exp) * self.pdf(fn_input) * inv_x * inv_x
                };

                crate::euclid::numerical_integration_finite(
                    integration_fn,
                    bounds,
                    num_steps as u64,
                )
            }
            crate::euclid::IntegrationType::ConstToInfinite => {
                // integral {a -> inf} f(x) dx  = integral {0 -> 1} f(a + t/(t - 1))  /  (1 - t)^2  dt

                let integration_fn = |x: f64| 'integration: {
                    // x will go from 0.0 to 1.0

                    let x_minus: f64 = x - 1.0;
                    if x_minus.abs() < f64::EPSILON {
                        // too near singularity, skip
                        break 'integration 0.0;
                    }

                    let u: f64 = 1.0 / x_minus;
                    let fn_input: f64 = bounds.0 + x * u;
                    let std_inp: f64 = (fn_input + minus_mean) * inv_std_dev;
                    std_inp.powi(order_exp) * self.pdf(fn_input) * u * u
                };

                crate::euclid::numerical_integration_finite(
                    integration_fn,
                    bounds,
                    num_steps as u64,
                )
            }
            crate::euclid::IntegrationType::FullInfinite => {
                // integral {a -> inf} f(x) dx  = integral {0 -> 1} f(a + t/(t - 1))  /  (1 - t)^2  dt

                let integration_fn = |x: f64| 'integration: {
                    // x will go from -1.0 to 1.0

                    let u: f64 = 1.0 - x * x;
                    if u.abs() < f64::EPSILON {
                        // too near singularity, skip
                        break 'integration 0.0;
                    }
                    let v: f64 = 1.0 / u;
                    let fn_input: f64 = x * v;
                    let std_inp: f64 = (fn_input + minus_mean) * inv_std_dev;
                    std_inp.powi(order_exp) * self.pdf(fn_input) * (1.0 + x * x) * v * v
                };

                crate::euclid::numerical_integration_finite(
                    integration_fn,
                    bounds,
                    num_steps as u64,
                )
            }
        };

        return moment;
    }

    fn entropy(&self) -> f64 {
        return 0.5 * (2.0 * PI * E).ln();
    }

    fn rejection_sample(&self, n: usize, _pdf_max: f64) -> Vec<f64> {
        let pdf_max: f64 = 1.0 / (2.0 * PI).sqrt();
        // ^We know the correct value analytically, so there is no need for the user to put anything
        let mut rng: rand::prelude::ThreadRng = rand::rng();
        let domain: &ContinuousDomain = &ContinuousDomain::Range(-4.5, 4.5);
        //^contains essentially all density
        let bounds: (f64, f64) = domain.get_bounds();
        let bound_range: f64 = bounds.1 - bounds.0;

        let mut ret: Vec<f64> = Vec::with_capacity(n);
        for _ in 0..n {
            let sample: f64 = loop {
                let mut x: f64 = rng.random();
                x = bounds.0 + x * bound_range;
                let y: f64 = rng.random();
                if y * pdf_max < self.pdf(x) {
                    break x;
                }
            };
            ret.push(sample);
        }

        return ret;
    }

    fn rejection_sample_range(&self, n: usize, mut pdf_max: f64, range: (f64, f64)) -> Vec<f64> {
        {
            let absolute_pdf_max: f64 = 1.0 / (2.0 * PI).sqrt();
            pdf_max = pdf_max.min(absolute_pdf_max);
        }

        let mut rng: rand::prelude::ThreadRng = rand::rng();
        let range_magnitude: f64 = range.1 - range.0;

        if range_magnitude.is_sign_negative() {
            // possible early return
            return Vec::new();
        }

        let mut ret: Vec<f64> = Vec::with_capacity(n);
        for _ in 0..n {
            let sample: f64 = loop {
                let mut x: f64 = rng.random();
                x = range.0 + x * range_magnitude;
                let y: f64 = rng.random();
                if y * pdf_max < self.pdf(x) {
                    break x;
                }
            };
            ret.push(sample);
        }

        return ret;
    }
}

impl Distribution for Normal {
    fn pdf(&self, x: f64) -> f64 {
        let inv_std: f64 = 1.0 / self.standard_deviation;
        return self.std_normal.pdf((x - self.mean) * inv_std) * inv_std;
    }

    fn get_domain(&self) -> &ContinuousDomain {
        return self.std_normal.get_domain();
    }

    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            // x is not valid
            std::panic!("Tried to evaluate the cdf function of Normal with a NaN value. \n");
        }
        let aux: [f64; 1] = [x];
        let aux_2: Vec<f64> = self.cdf_multiple(&aux);
        return aux_2[0];
    }

    fn sample(&self) -> f64 {
        let aux: Vec<f64> = self.sample_multiple(1);
        return aux[0];
    }

    fn quantile(&self, x: f64) -> f64 {
        // just call [Distribution::quantile_multiple]

        if x.is_nan() {
            // x is not valid
            std::panic!("Tried to evaluate the quantile function of Normal with a NaN value. \n");
        }

        let value: [f64; 1] = [x];
        let quantile_vec: Vec<f64> = self.quantile_multiple(&value);
        return quantile_vec[0];
    }

    fn cdf_multiple(&self, points: &[f64]) -> Vec<f64> {
        let neg_mean: f64 = -self.mean;
        let inv_std_dev: f64 = 1.0 / self.standard_deviation;
        self.std_normal.cdf_multiple(
            &points
                .iter()
                .map(|&x| (x + neg_mean) * inv_std_dev)
                .collect::<Vec<f64>>(),
        )
    }

    fn sample_multiple(&self, n: usize) -> Vec<f64> {
        let ret: Vec<f64> = self
            .std_normal
            .sample_multiple(n)
            .iter()
            .map(|&x| (x + self.mean) * self.standard_deviation)
            .collect::<Vec<f64>>();

        return ret;
    }

    fn quantile_multiple(&self, points: &[f64]) -> Vec<f64> {
        if points.is_empty() {
            return Vec::new();
        }

        // return error if NAN is found
        for point in points {
            if point.is_nan() {
                std::panic!("Found NaN in `quantile_multiple` of Normal. \n");
            }
        }

        let neg_mean: f64 = -self.mean;
        let inv_std_dev: f64 = 1.0 / self.standard_deviation;
        return self.std_normal.quantile_multiple(
            &points
                .iter()
                .map(|&x| (x + neg_mean) * inv_std_dev)
                .collect::<Vec<f64>>(),
        );
    }

    fn expected_value(&self) -> Option<f64> {
        return Some(self.mean);
    }

    fn variance(&self) -> Option<f64> {
        return Some(self.standard_deviation);
    }

    fn mode(&self) -> f64 {
        return self.mean;
    }

    fn median(&self) -> f64 {
        return self.mean;
    }

    fn skewness(&self) -> Option<f64> {
        return Some(0.0);
    }

    fn kurtosis(&self) -> Option<f64> {
        return Some(3.0);
    }

    fn excess_kurtosis(&self) -> Option<f64> {
        return Some(0.0);
    }

    fn moments(&self, order: u8, mode: crate::euclid::Moments) -> f64 {
        /*

               Plan:

            Just to the integral. The integral that gives us the moments of order `k` is:

            ```
            integral {a -> b} ( (x - mu) / std )^k * f(x) dx
            ```
             - `k` is the order of the moment
             - `f(x)` is the pdf of the distribution.
             - `a` and `b` are the values that bound the domain of `f(x)`
                    (they can be `a = -inf` and `b = -inf`).
             - `mu` is the mean of the distribution (or `0` if we selected the `Raw` moment)
             - `std` is the standard deviation of the distribution
                    (or `1` if we did not select the `Standarized` moment)


           Distiguish between cases depending on the domain.

           We will integrate using [Simpson's rule](https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_1/3_rule)
           for integration.

           To compute integrals over an infinite range, we will perform a special
           [numerial integration](https://en.wikipedia.org/wiki/Numerical_integration#Integrals_over_infinite_intervals).

            let g(x) = ( (x - mu) / std )^k * f(x)
                For -infinite to const:
            integral {-inf -> a} g(x) dx = integral {0 -> 1} g(a - (1 - t)/t)  /  t^2  dt
            integral {-inf -> a} g(x) dx = integral {0 -> 1} ( (a - (1 - t)/t - mu) / std )^k * f(a - (1 - t)/t)  /  t^2  dt

                For const to infinite:
            integral {a -> inf} g(x) dx  = integral {0 -> 1} g(a + t/(t - 1))  /  (1 - t)^2  dt
            integral {a -> inf} g(x) dx  = integral {0 -> 1} ( (a + t/(t - 1) - mu) / std )^k * f(a + t/(t - 1))  /  (1 - t)^2  dt

                For -infinite to infinite:
            let inp = t/(1 - t^2)
            integral {-inf -> inf} g(x) dx  = integral {-1 -> 1} g(t/(1 - t^2))  *  (1 + t^2) / (1 - t^2)^2  dt
            integral {-inf -> inf} g(x) dx  = integral {-1 -> 1} ( (t/(1 - t^2) - mu) / std )^k * f(t/(1 - t^2))  *  (1 + t^2) / (1 - t^2)^2  dt


        */

        let domain: &ContinuousDomain = self.get_domain();
        let bounds: (f64, f64) = domain.get_bounds();

        // The values of 0.0 and 1.0 have no special meaning. They are not going to be used anyway.
        let (mean, std_dev): (f64, f64) = match mode {
            crate::euclid::Moments::Raw => (0.0, 1.0),
            crate::euclid::Moments::Central => (
                self.expected_value()
                    .expect("Tried to compute a central moment but the expected value is undefined. "),
                1.0,
            ),
            crate::euclid::Moments::Standarized => (
                self.expected_value()
                    .expect("Tried to compute a central/standarized moment but the Expected value is undefined. "),
                self.variance().expect("Tried to compute a standarized moment but the variance is undefined. "),
            ),
        };

        // Todo: give better error handling to the above. ^
        // println!("(mean, std_dev): {:?}", (mean, std_dev));

        let order_exp: i32 = order as i32;
        let (minus_mean, inv_std_dev) = (-mean, 1.0 / std_dev.sqrt());
        let integration_type: crate::euclid::IntegrationType =
            crate::euclid::IntegrationType::from_bounds(bounds);
        let (_, num_steps): (f64, usize) =
            crate::euclid::choose_integration_precision_and_steps(bounds);

        let moment: f64 = match integration_type {
            crate::euclid::IntegrationType::Finite => {
                let integration_fn = |x: f64| {
                    let std_inp: f64 = (x + minus_mean) * inv_std_dev;
                    std_inp.powi(order_exp) * self.pdf(x)
                };

                crate::euclid::numerical_integration_finite(
                    integration_fn,
                    bounds,
                    num_steps as u64,
                )
            }
            crate::euclid::IntegrationType::InfiniteToConst => {
                // integral {-inf -> a} f(x) dx = integral {0 -> 1} f(a - (1 - t)/t)  /  t^2  dt
                let integration_fn = |x: f64| 'integration: {
                    // x will go from 0.0 to 1.0
                    if x.abs() < f64::EPSILON {
                        // too near singularity, skip
                        break 'integration 0.0;
                    }
                    let inv_x: f64 = 1.0 / x;
                    let fn_input: f64 = bounds.1 - (1.0 - x) * inv_x;
                    let std_inp: f64 = (fn_input + minus_mean) * inv_std_dev;
                    std_inp.powi(order_exp) * self.pdf(fn_input) * inv_x * inv_x
                };

                crate::euclid::numerical_integration_finite(
                    integration_fn,
                    bounds,
                    num_steps as u64,
                )
            }
            crate::euclid::IntegrationType::ConstToInfinite => {
                // integral {a -> inf} f(x) dx  = integral {0 -> 1} f(a + t/(t - 1))  /  (1 - t)^2  dt

                let integration_fn = |x: f64| 'integration: {
                    // x will go from 0.0 to 1.0

                    let x_minus: f64 = x - 1.0;
                    if x_minus.abs() < f64::EPSILON {
                        // too near singularity, skip
                        break 'integration 0.0;
                    }

                    let u: f64 = 1.0 / x_minus;
                    let fn_input: f64 = bounds.0 + x * u;
                    let std_inp: f64 = (fn_input + minus_mean) * inv_std_dev;
                    std_inp.powi(order_exp) * self.pdf(fn_input) * u * u
                };

                crate::euclid::numerical_integration_finite(
                    integration_fn,
                    bounds,
                    num_steps as u64,
                )
            }
            crate::euclid::IntegrationType::FullInfinite => {
                // integral {a -> inf} f(x) dx  = integral {0 -> 1} f(a + t/(t - 1))  /  (1 - t)^2  dt

                let integration_fn = |x: f64| 'integration: {
                    // x will go from -1.0 to 1.0

                    let u: f64 = 1.0 - x * x;
                    if u.abs() < f64::EPSILON {
                        // too near singularity, skip
                        break 'integration 0.0;
                    }
                    let v: f64 = 1.0 / u;
                    let fn_input: f64 = x * v;
                    let std_inp: f64 = (fn_input + minus_mean) * inv_std_dev;
                    std_inp.powi(order_exp) * self.pdf(fn_input) * (1.0 + x * x) * v * v
                };

                crate::euclid::numerical_integration_finite(
                    integration_fn,
                    bounds,
                    num_steps as u64,
                )
            }
        };

        return moment;
    }

    fn entropy(&self) -> f64 {
        let s_squared: f64 = self.standard_deviation * self.standard_deviation;
        return 0.5 * (2.0 * PI * E * s_squared).ln();
    }

    fn rejection_sample(&self, n: usize, pdf_max: f64) -> Vec<f64> {
        return self
            .std_normal
            .rejection_sample(n, pdf_max)
            .into_iter()
            .map(|x| (x + self.mean) * self.standard_deviation)
            .collect::<Vec<f64>>();
    }

    fn rejection_sample_range(&self, n: usize, pdf_max: f64, range: (f64, f64)) -> Vec<f64> {
        // Todo: this should be probably be put in terms of the [rejection_sample_range]
        // implemetation os StdNormal (like we did with [rejection_sample])
        let mut rng: rand::prelude::ThreadRng = rand::rng();
        let range_magnitude: f64 = range.1 - range.0;

        if range_magnitude.is_sign_negative() {
            // possible early return
            return Vec::new();
        }

        let mut ret: Vec<f64> = Vec::with_capacity(n);
        for _ in 0..n {
            let sample: f64 = loop {
                let mut x: f64 = rng.random();
                x = range.0 + x * range_magnitude;
                let y: f64 = rng.random();
                if y * pdf_max * self.standard_deviation < self.std_normal.pdf(x - self.mean) {
                    // ^small optimitzation it avoid a division
                    break x;
                }
            };
            ret.push(sample);
        }

        return ret;
    }
}

// Note: no Parametric for StdNormal because it does not have any parameters,
// since it has no parameters (mean is always 0 and std_dev is always 1).
// If it is needed to compute the derivatives/log derivatives of the std normal
// we recommend simply creating a Normal with parameters 0 and 1.

impl Parametric for Normal {
    /// Evaluates the [Normal::pdf] in a general way taking into account
    /// the parameters.
    ///
    /// > pdf(x | mean, std) = 1/(std * sqrt(2*pi)) * exp( -(x - mean)^2 / (2*std^2) )
    ///
    /// ### Parameters for Normal:
    ///
    /// The [Normal] distribution has 2 parameters: `mean`
    /// and `std` (standard deviation). The order of the `parameters` is:
    ///
    /// > \[mean, std\]
    fn general_pdf(&self, x: f64, parameters: &[f64]) -> f64 {
        let input: f64 = x - parameters[0];
        let inv_std_dev: f64 = 1.0 / parameters[1];
        let inv_var: f64 = inv_std_dev * inv_std_dev;

        let inv_sqrt_2_pi: f64 = 1.0 / (2.0 * PI).sqrt();
        let normalitzation_constant: f64 = inv_sqrt_2_pi * inv_std_dev;

        return normalitzation_constant * (-0.5 * input * input * inv_var).exp();
    }

    /// Returns the number of parameters of the model: `2`
    fn number_of_parameters() -> u16 {
        2
    }

    fn get_parameters(&self, parameters: &mut [f64]) {
        parameters[0] = self.mean;
        parameters[1] = self.standard_deviation;
    }

    fn derivative_pdf_parameters(&self, x: f64, parameters: &[f64]) -> Vec<f64> {
        // d/dx ln(f(x)) = f'(x)/f(x)
        // => f(x) * d/dx ln(f(x)) = f'(x)

        // pdf(x | mean, std) = 1/(std * sqrt(2*pi)) * exp( -(x - mean)^2 / (2*std^2) )

        // Reserve a vector with exacly 3 elements
        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(3);

        {
            //## Derivative respect to x:
            /*
               d/dx pdf(x | mean, std) = d/dx 1/(std * sqrt(2*pi)) * exp( -(x - mean)^2 / (2*std^2) )
                = 1/(std * sqrt(2*pi)) * d/dx exp( -(x - mean)^2 / (2*std^2) )
                = 1/(std * sqrt(2*pi)) * exp( -(x - mean)^2 / (2*std^2) ) * d/dx -(x - mean)^2 / (2*std^2)
                = 1/(std * sqrt(2*pi)) * exp( -(x - mean)^2 / (2*std^2) ) / (2*std^2) * d/dx -(x - mean)^2
                = 1/(std * sqrt(2*pi)) * exp( -(x - mean)^2 / (2*std^2) ) / (2*std^2) * -2*(x - mean) d/dx x - mean
                = 1/(std * sqrt(2*pi)) * exp( -(x - mean)^2 / (2*std^2) ) / (2*std^2) * -2*(x - mean)
                = 1/(std * sqrt(2*pi) * 2*std^2) * exp( -(x - mean)^2 / (2*std^2) )  * -2*(x - mean)
                = 1/(sqrt(8*pi) * std^3) * exp( -(x - mean)^2 / (2*std^2) ) * -2*(x - mean)
                = -2/(sqrt(8*pi) * std^3) * exp( -(x - mean)^2 / (2*std^2) ) * (x - mean)

            */

            let input: f64 = x - parameters[0];
            let inv_std_dev: f64 = 1.0 / parameters[1];
            let inv_var: f64 = inv_std_dev * inv_std_dev;

            // norm_const = -2/(sqrt(8*pi) * std^3)
            let inv_sqrt_8_pi: f64 = 1.0 / (8.0 * PI).sqrt();
            let normalitzation_constant: f64 = -2.0 * inv_sqrt_8_pi * inv_std_dev * inv_var;

            let der: f64 = normalitzation_constant * (-0.5 * input * input * inv_var).exp() * input;
            ret.push(der);
        }

        {
            //## Derivative respect to mean:
            /*
               d/d_mean pdf(x | mean, std) = d/d_mean 1/(std * sqrt(2*pi)) * exp( -(x - mean)^2 / (2*std^2) )
                = 1/(std * sqrt(2*pi)) * d/d_mean exp( -(x - mean)^2 / (2*std^2) )
                = 1/(std * sqrt(2*pi)) * exp( -(x - mean)^2 / (2*std^2) ) * d/d_mean -(x - mean)^2 / (2*std^2)
                = 1/(std * sqrt(2*pi)) * exp( -(x - mean)^2 / (2*std^2) ) / (2*std^2) * d/d_mean -(x - mean)^2
                = 1/(std * sqrt(2*pi)) * exp( -(x - mean)^2 / (2*std^2) ) / (2*std^2) * -2*(x - mean) d/d_mean x - mean
                = 1/(std * sqrt(2*pi)) * exp( -(x - mean)^2 / (2*std^2) ) / (2*std^2) * -2*(x - mean) * -1
                = 1/(std * sqrt(2*pi)) * exp( -(x - mean)^2 / (2*std^2) ) / (2*std^2) * 2*(x - mean)
                = 2/(std * sqrt(2*pi) * 2*std^2) * exp( -(x - mean)^2 / (2*std^2) ) * (x - mean)
                = 2/(sqrt(8*pi) * std^3) * exp( -(x - mean)^2 / (2*std^2) ) * (x - mean)

                Notice how:
                > d/d_mean pdf(x | mean, std) = -d/d_x pdf(x | mean, std)

                Therefore we only need to reuse the prevously computed number:
            */

            ret.push(-ret[0]);
        }

        {
            //## Derivative respect to std:
            /*
               d/d_std pdf(x | mean, std) = d/d_std 1/(std * sqrt(2*pi)) * exp( -(x - mean)^2 / (2*std^2) )
                = 1/(sqrt(2*pi)) d/d_std 1/std * exp( -(x - mean)^2 / (2*std^2) )
                = 1/(sqrt(2*pi)) d/d_std exp( -(x - mean)^2 / (2*std^2) ) / std
                = 1/(sqrt(2*pi)) ( d/d_std[exp( -(x - mean)^2 / (2*std^2) )] * std - 1*exp( -(x - mean)^2 / (2*std^2) )) / std^2
                = 1/(sqrt(2*pi)) ( exp( -(x - mean)^2 / (2*std^2) ) * d/d_std[-(x - mean)^2 / (2*std^2) ] * std - exp( -(x - mean)^2 / (2*std^2) )) / std^2
                = 1/(sqrt(2*pi)) ( exp( -(x - mean)^2 / (2*std^2) ) * d/d_std[-(x - mean)^2 * (2*std^2)^-1 ] * std - exp( -(x - mean)^2 / (2*std^2) )) / std^2
                = 1/(sqrt(2*pi)) ( exp( -(x - mean)^2 / (2*std^2) ) * -(x - mean)^2 * d/d_std[(2*std^2)^-1 ] * std - exp( -(x - mean)^2 / (2*std^2) )) / std^2
                = 1/(sqrt(2*pi)) ( exp( -(x - mean)^2 / (2*std^2) ) * -(x - mean)^2 * -(2*std^2)^-2 * 4 * std * std - exp( -(x - mean)^2 / (2*std^2) )) / std^2
                = 1/(sqrt(2*pi)) ( exp( -(x - mean)^2 / (2*std^2) ) * -(x - mean)^2 * -1/(2*std^2)^2 * 4 * std * std - exp( -(x - mean)^2 / (2*std^2) )) / std^2
                = 1/(sqrt(2*pi)) ( exp( -(x - mean)^2 / (2*std^2) ) * -(x - mean)^2 * -1/(4*std^4) * 4 * std * std - exp( -(x - mean)^2 / (2*std^2) )) / std^2
                = 1/(sqrt(2*pi)) ( exp( -(x - mean)^2 / (2*std^2) ) * -(x - mean)^2 * -1/std^4 * std * std - exp( -(x - mean)^2 / (2*std^2) )) / std^2
                = 1/(sqrt(2*pi)) ( exp( -(x - mean)^2 / (2*std^2) ) * -(x - mean)^2 * -1/std^2 - exp( -(x - mean)^2 / (2*std^2) )) / std^2
                = 1/(sqrt(2*pi)*std^2) ( exp( -(x - mean)^2 / (2*std^2) ) * (-(x - mean)^2 * -1/std^2 - 1))
                = 1/(sqrt(2*pi)*std^2) ( exp( -(x - mean)^2 / (2*std^2) ) * ((x - mean)^2 * 1/std^2 - 1))
                = 1/(sqrt(2*pi)*std^2) ( exp( -(x - mean)^2 / (2*std^2) ) * ((x - mean)^2/std^2 - 1))
                = 1/(std^2 * sqrt(2*pi)) * exp( -(x - mean)^2 / (2*std^2) ) * (((x - mean)/std)^2 - 1)
                = 1/(std * sqrt(2*pi)) * exp( -(x - mean)^2 / (2*std^2) ) / std * (((x - mean)/std)^2 - 1)
                = pdf(x | mean, std)/std * (((x - mean)/std)^2 - 1)
                = pdf(x | mean, std) * (((x - mean)/std)^2 - 1)/std

            *********************************

            Alt proof:
            pdf(x | mean, std) = pdf_std((x - mean)/std) / std_dev
            d/d_std pdf(x | mean, std) = d/d_std pdf_std((x - mean)/std) / std_dev
             = d/d_std pdf_std((x - mean)/std) / std_dev
             = (pdf_std'((x - mean)/std) * d/d_std[(x - mean)/std_dev] * std_dev - 1 * pdf_std((x - mean)/std)) / std_dev^2
             = (pdf_std'((x - mean)/std) * (x - mean) * d/d_std[1/std_dev] * std_dev - pdf_std((x - mean)/std)) / std_dev^2
             = (pdf_std'((x - mean)/std) * (x - mean) * -1/std_dev^2 * std_dev - pdf_std((x - mean)/std)) / std_dev^2
             = (pdf_std'((x - mean)/std) * (x - mean) * -1/std_dev - pdf_std((x - mean)/std)) / std_dev^2
             = (pdf_std'((x - mean)/std) * -(x - mean)/std_dev - pdf_std((x - mean)/std)) / std_dev^2

            Computing pdf_std'(s):

            pdf_std'(s) = d/ds pdf_std(s) = d/ds 1/(sqrt(2*pi)) * exp( -0.5*s^2 )
             = 1/(sqrt(2*pi)) * d/ds exp( -0.5*s^2 )
             = 1/(sqrt(2*pi)) * exp( -0.5*s^2 ) * d/ds -0.5*s^2
             = 1/(sqrt(2*pi)) * exp( -0.5*s^2 ) * -0.5 * d/ds s^2
             = 1/(sqrt(2*pi)) * exp( -0.5*s^2 ) * -0.5 * 2*s
             = 1/(sqrt(2*pi)) * exp( -0.5*s^2 ) * -s
             = pdf_std(s) * -s

            Continuing on d/d_std pdf(x | mean, std):
            d/d_std pdf(x | mean, std) =
             = (pdf_std'((x - mean)/std_dev) * -(x - mean)/std_dev - pdf_std((x - mean)/std)) / std_dev^2
             = (pdf_std((x - mean)/std_dev) * -(x - mean)/std_dev * -(x - mean)/std_dev - pdf_std((x - mean)/std)) / std_dev^2
             = (pdf_std((x - mean)/std_dev) * ((x - mean)/std_dev)^2 - pdf_std((x - mean)/std)) / std_dev^2
             = pdf_std((x - mean)/std_dev) * ( ((x - mean)/std_dev)^2 - 1 ) / std_dev^2
             = pdf_std((x - mean)/std_dev) / std_dev * (((x - mean)/std_dev)^2 - 1) / std_dev
             = pdf(x | mean, std) * (((x - mean)/std_dev)^2 - 1) / std_dev

            Wich is identical for the solution we found in the normal derivation :)

            *********************************
            ### Check:

            d/dx ln(f(x)) = f'(x)/f(x)  =>  f(x) * d/dx ln(f(x)) = f'(x)
            d/d_std ln(pdf(x | mean, std)) = 1/std * (-1 + ((x - mean)/std)^2 )

            pdf(x | mean, std) * d/d_std ln(pdf(x | mean, std)) =
             = 1/(std * sqrt(2*pi)) * exp( -(x - mean)^2 / (2*std^2) ) * 1/std * (-1 + ((x - mean)/std)^2 )
             = 1/(std^2 * sqrt(2*pi)) * exp( -(x - mean)^2 / (2*std^2) ) * (-1 + ((x - mean)/std)^2 )
             = 1/(std^2 * sqrt(2*pi)) * exp( -(x - mean)^2 / (2*std^2) ) * (((x - mean)/std)^2 - 1)

            Wich is identical to 1 of the results we obtained for d/d_std pdf(x | mean, std).

            */
            // pdf(x | mean, std) * (((x - mean)/std)^2 - 1)/std

            let input: f64 = x - parameters[0];
            let inv_std_dev: f64 = 1.0 / parameters[1];
            let s: f64 = input * inv_std_dev;

            let inv_sqrt_2_pi: f64 = 1.0 / (2.0 * PI).sqrt();

            // pdf(x | mean, std)
            let pdf: f64 = inv_sqrt_2_pi * inv_std_dev * (-0.5 * s * s).exp();

            // = (((x - mean)/std)^2 - 1)/std
            let term: f64 = (s * s - 1.0) * inv_std_dev;

            ret.push(pdf * term);
        }

        return ret;
    }

    fn log_derivative_pdf_parameters(&self, x: f64, parameters: &[f64]) -> Vec<f64> {
        // d/dx ln(f(x)) = f'(x)/f(x)

        // pdf(x | mean, std) = 1/(std * sqrt(2*pi)) * exp( -(x - mean)^2 / (2*std^2) )
        // ln(pdf(x | mean, std)) = ln(1/(std * sqrt(2*pi)) * exp( -(x - mean)^2 / (2*std^2) ))
        // ln(pdf(x | mean, std)) = ln(1/(std * sqrt(2*pi)) + -(x - mean)^2 / (2*std^2)
        // ln(pdf(x | mean, std)) = -ln(std * sqrt(2*pi) + -(x - mean)^2 / (2*std^2)
        // ln(pdf(x | mean, std)) = -ln(std) -ln(sqrt(2*pi)) - (x - mean)^2 / (2*std^2)
        // ln(pdf(x | mean, std)) = -ln(std) -0.5*ln(2*pi) - (x - mean)^2 / (2*std^2)

        // Reserve a vector with exacly 3 elements
        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(3);

        {
            //## Log derivative respect to x:
            /*
               d/dx ln(pdf(x | mean, std)) = d/dx -ln(std) -0.5*ln(2*pi) - (x - mean)^2 / (2*std^2)
                = 0 + 0 - d/dx (x - mean)^2 / (2*std^2)
                = - 1/(2*std^2) * d/dx (x - mean)^2
                = - 1/(2*std^2) * 2 * (x - mean) * 1
                = - 1/(std^2) * (x - mean)
                = - (x - mean)/(std^2)
                = (mean - x)/(std^2)
                let s = (x - mean)/std
                d/dx ln(pdf(x | mean, std)) = -(x - mean)/(std^2)
                 = -s/std

                We will use the result: -(x - mean)/(std^2)
            */

            let input: f64 = x - parameters[0];
            let inv_std_dev: f64 = 1.0 / parameters[1];

            ret.push(-input * inv_std_dev * inv_std_dev);
        }

        {
            //## Log derivative respect to mean:
            /*
               d/d_mean ln(pdf(x | mean, std)) = d/d_mean -ln(std) -0.5*ln(2*pi) - (x - mean)^2 / (2*std^2)
                = 0 + 0 - d/d_mean (x - mean)^2 / (2*std^2)
                = -d/d_mean (x - mean)^2 / (2*std^2)
                = -1/(2*std^2) * d/d_mean (x - mean)^2
                = -1/(2*std^2) * 2*(x - mean) * -1
                = -1/(std^2) * (x - mean) * -1
                = 1/(std^2) * (x - mean)
                = (x - mean)/(std^2)

                Note that (similarly with the normal derivatives), this reuslt is
                the same as the negated prevous one:

                d/d_mean ln(pdf(x | mean, std)) = -d/dx ln(pdf(x | mean, std))
            */

            ret.push(-ret[0]);
        }

        {
            //## Log derivative respect to std:
            /*
                d/d_std ln(pdf(x | mean, std)) = d/d_std -ln(std) -0.5*ln(2*pi) - (x - mean)^2 / (2*std^2)
                = -1/std + 0 - (x - mean)^2 * d/d_std 1 / (2*std^2)
                = -1/std + 0 - (x - mean)^2 * 0.5 * d/d_std 1 / (std^2)
                = -1/std + 0 - (x - mean)^2 * 0.5 * d/d_std std^-2
                = -1/std + 0 - (x - mean)^2 * 0.5 * -2 * std^-3
                = -1/std - (x - mean)^2 * -1 * std^-3
                = -1/std + (x - mean)^2 * std^-3
                = -1/std + ((x - mean)/std)^2 * 1/std
                = 1/std * (-1 + ((x - mean)/std)^2 )
                let s = (x - mean)/std
                d/d_std ln(pdf(x | mean, std)) = 1/std * (-1 + s^2 )
                = 1/std * ( s^2 - 1 )

                We will use the result: = 1/std * (-1 + ((x - mean)/std)^2 )
            */

            let input: f64 = x - parameters[0];
            let inv_std_dev: f64 = 1.0 / parameters[1];

            let log_der: f64 = inv_std_dev * (input * input * inv_std_dev * inv_std_dev - 1.0);
            ret.push(log_der);
        }

        return ret;
    }

    fn parameter_restriction(&self, parameters: &mut [f64]) {
        parameters[1] = parameters[1].max(f64::MIN_POSITIVE);
        // std cannot be 0 or negative
    }

    fn fit(&self, data: &mut crate::Samples::Samples) -> Vec<f64> {
        // Reserve a vector with exacly 2 elements
        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(2);

        /*
                Maximum likelyhood estimation:

            Assuming n samples.

            ### For mean:

            0 = sumatory{x_i} d/d_mean ln(pdf(x_i | mean, std_dev))
            0 = sumatory{x_i} (x_i - mean)/(std_dev^2)
            0 = 1/std_dev^2 * sumatory{x_i} x_i - mean
            0 = sumatory{x_i} x_i - mean
            0 = -mean * n + sumatory{x_i} x_i
            mean * n = sumatory{x_i} x_i
            mean = 1/m * sumatory{x_i} x_i
            mean = mean[x_i]

            //(wow, what a surprise)

            ### For std_dev:

            0 = sumatory{x_i} d/d_std ln(pdf(x_i | mean, std_dev))
            0 = sumatory{x_i} 1/std * (-1 + ((x - mean)/std)^2 )
            0 = 1/std * sumatory{x_i} (-1 + ((x - mean)/std)^2 )
            0 = sumatory{x_i} -1 + ((x - mean)/std)^2
            0 = -n + sumatory{x_i} ((x - mean)/std)^2
            n = sumatory{x_i} ((x - mean)/std)^2
            n = sumatory{x_i} (x - mean)^2/std^2
            n = 1/std^2 * sumatory{x_i} (x - mean)^2
            n * std^2 = sumatory{x_i} (x - mean)^2
            std^2 = 1/n * sumatory{x_i} (x - mean)^2
            std = sqrt(1/n * sumatory{x_i} (x - mean)^2 )

            // The biased definition of std_dev

            ## Conclusion:

            To estimate the mean we will use:

            mean = mean[x_i]

            But for std we will use the **UNBIASED** formula instead of the obtained one.
            std = sqrt(1/(n-1) * sumatory{x_i} (x - mean)^2 )

            If there are not enough samples for the computation, the deafults are:
             - mean: 0
             - std_dev: 1

        */

        let mean: f64 = data.mean().unwrap_or(0.0);
        let std_dev: f64 = data.variance().map(|v| v.sqrt()).unwrap_or(1.0);
        // note that .variance() uses the unbiased extimator

        ret.push(mean);
        ret.push(std_dev);

        return ret;
    }
}

impl Iterator for StdNormalGenerator {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        // similar implenentation as [StdNormal::sample] but better.
        // removed comments

        let rand_q: f64 = self.rng.random();

        // just map r to the awnser

        let (q, flipped): (f64, bool) = if rand_q < 0.5 {
            (1.0 - rand_q, true)
        } else {
            (rand_q, false)
        };

        assert!(0.5 <= q && q <= 1.0);

        // [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method)

        let mut r: f64 = 4.0 * q - 2.2;
        let mut last: f64 = -128.0; // arbitrary number, just far away from r
                                    // ^seed (bad aproximation for inv cdf(x) [0.5, 1.0] but good enough as first guess)
        let ret: f64 = 'newton_loop: loop {
            let cdf: f64 = if r < 1.4 {
                // covers 83.8487 % of cases
                // method 0

                //Horner's rule
                let numerator: f64 = SECTION_0_NUM[0]
                    .mul_add(r, SECTION_0_NUM[1])
                    .mul_add(r, SECTION_0_NUM[2])
                    .mul_add(r, SECTION_0_NUM[3])
                    .mul_add(r, SECTION_0_NUM[4])
                    .mul_add(r, SECTION_0_NUM[5])
                    .mul_add(r, SECTION_0_NUM[6]);

                let denominator: f64 = SECTION_0_DEN[0]
                    .mul_add(r, SECTION_0_DEN[1])
                    .mul_add(r, SECTION_0_DEN[2])
                    .mul_add(r, SECTION_0_DEN[3])
                    .mul_add(r, SECTION_0_DEN[4])
                    .mul_add(r, SECTION_0_DEN[5])
                    .mul_add(r, SECTION_0_DEN[6]);

                numerator / denominator
            } else if r < 3.0 {
                // covers 99.73% of cases
                // method 2.5

                //Horner's rule
                let numerator: f64 = SECTION_25_NUM[0]
                    .mul_add(r, SECTION_25_NUM[1])
                    .mul_add(r, SECTION_25_NUM[2])
                    .mul_add(r, SECTION_25_NUM[3])
                    .mul_add(r, SECTION_25_NUM[4])
                    .mul_add(r, SECTION_25_NUM[5])
                    .mul_add(r, SECTION_25_NUM[6]);

                let denominator: f64 = SECTION_25_DEN[0]
                    .mul_add(r, SECTION_25_DEN[1])
                    .mul_add(r, SECTION_25_DEN[2])
                    .mul_add(r, SECTION_25_DEN[3])
                    .mul_add(r, SECTION_25_DEN[4])
                    .mul_add(r, SECTION_25_DEN[5])
                    .mul_add(r, SECTION_25_DEN[6]);

                numerator / denominator
            } else if r < 8.0 {
                // covers 99.99999999999987558078851456431752968% of cases (according to wolframalpha)
                // method 4
                // remember that we will still use this method up to 8.0
                // even if it doen not retain as much precision.
                //Horner's rule
                let numerator: f64 = SECTION_4_NUM[0]
                    .mul_add(r, SECTION_4_NUM[1])
                    .mul_add(r, SECTION_4_NUM[2])
                    .mul_add(r, SECTION_4_NUM[3])
                    .mul_add(r, SECTION_4_NUM[4])
                    .mul_add(r, SECTION_4_NUM[5])
                    .mul_add(r, SECTION_4_NUM[6]);

                let denominator: f64 = SECTION_4_DEN[0]
                    .mul_add(r, SECTION_4_DEN[1])
                    .mul_add(r, SECTION_4_DEN[2])
                    .mul_add(r, SECTION_4_DEN[3])
                    .mul_add(r, SECTION_4_DEN[4])
                    .mul_add(r, SECTION_4_DEN[5])
                    .mul_add(r, SECTION_4_DEN[6]);

                numerator / denominator
            } else {
                // numbers greater than 8.0 are mapped to 0.0 to avoid prolems
                // this is **extremly** unlikely anyway. probability of ending
                // in this region: 0.00000000000012441921148543568247032% (according to wolframalpha)
                break 'newton_loop 0.0;
            };

            // Newton's method
            //r_n+1 = r_n + (q - cdf(r_n)) * sqrt(2*pi) * exp(0.5 * r_n * r_n)

            // Instead of computing 1/pdf; compute a modified version of pdf
            let inv_pdf: f64 = euclid::SQRT_2_PI * (0.5 * r * r).exp();
            r = r + (q - cdf) * inv_pdf;

            if (r - last).abs() < 0.00001 {
                // since we are using Newton's method, this shoud be rather fast
                // (3 iters or less for over 99% of cases)
                break 'newton_loop if flipped { -r } else { r };
            }
            last = r;
        };

        return Some(ret);
    }
}

impl Iterator for NormalGenerator {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        let r: f64 = self.rng.next().unwrap();
        return Some((r + self.minus_mean) * self.inv_std_dev);
    }
}

impl Default for Normal {
    fn default() -> Self {
        Normal::new(0.0, 1.0).unwrap()
    }
}
