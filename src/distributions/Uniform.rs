//! The [uniform distribution](https://en.wikipedia.org/wiki/Continuous_uniform_distribution)
//! is a continuous distributions where all possible outcomes in the interval `[a, b]` have
//! equal likelyhood of happening.
//!
//! Note that this distribution is different from the
//! [discrete uniform distribution](crate::distributions::DiscreteUniform).
//!

use rand::Rng;

use crate::{distribution_trait::Distribution, domain::ContinuousDomain};

pub struct Uniform {
    domain: ContinuousDomain,
    /// The minimum value
    a: f64,
    /// The maximum value
    b: f64,
}

impl Uniform {
    /// Creates a new [uniform distribution](https://en.wikipedia.org/wiki/Continuous_uniform_distribution).
    ///
    ///  - `a` indicates the minimum value.
    ///  - `b` indicates the maximum value.
    ///  - `a <= b` must be fulfilled or an error will be returned.
    ///  - `a` and `b` must both be finite values (no `+-inf` or NaNs)
    pub fn new(a: f64, b: f64) -> Result<Uniform, ()> {
        if !a.is_finite() || !b.is_finite() {
            return Err(());
        }

        if b < a {
            return Err(());
        }

        let domain: ContinuousDomain = ContinuousDomain::Range(a, b);

        return Ok(Uniform { domain, a, b });
    }

    /// Creates a new [uniform distribution](https://en.wikipedia.org/wiki/Continuous_uniform_distribution).
    /// Does not do any checks. If the invariants are not fullfilled, results may be
    /// wrong or inconsistent.
    ///
    ///  - `a` indicates the minimum value.
    ///  - `b` indicates the maximum value.
    ///  - `a <= b` must be fulfilled or an error will be returned.
    ///  - `a` and `b` must both be finite values (no `+-inf` or NaNs)
    pub unsafe fn new_unchecked(a: f64, b: f64) -> Result<Uniform, ()> {
        let domain: ContinuousDomain = ContinuousDomain::Range(a, b);

        return Ok(Uniform { domain, a, b });
    }

    /// Return `a` (minimum value).
    pub fn get_a(&self) -> f64 {
        return self.a.clone();
    }

    /// Return `b` (maximum value).
    pub fn get_b(&self) -> f64 {
        return self.b.clone();
    }
}

impl Distribution for Uniform {
    fn pdf(&self, _x: f64) -> f64 {
        // it is **oviously** assumed that the pdf is evaluated inside the domain. 
        return 1.0 / (self.b - self.a); 
    }

    fn get_domain(&self) -> &ContinuousDomain {
        return &self.domain; 
    }
    
    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            // x is not valid
            std::panic!("Tried to evaluate the cdf function with a NaN value. \n");
        }

        if x < self.a {
            return 0.0; 
        }

        if self.b < x {
            return 1.0; 
        }

        return (x - self.a) / (self.b - self.a);
    }
    
    fn sample(&self) -> f64 {
        let ret: f64 = self.a + rand::prelude::random::<f64>() * (self.b - self.a);
        return ret;
    }
    
    fn quantile(&self, x: f64) -> f64 {
        // just call [Distribution::quantile_multiple]
    
        if x.is_nan() {
            // x is not valid
            std::panic!("Tried to evaluate the quantile function with a NaN value. \n");
        }

        if x <= 0.0 {
            return self.a;
        }

        if 1.0 <= x {
            return self.b;
        }
    
        return self.a + x * (self.b - self.a);
    }
    
    fn cdf_multiple(&self, points: &[f64]) -> Vec<f64> {
        points.iter().map(|x| self.cdf(*x)).collect::<Vec<f64>>()
    }

    fn sample_multiple(&self, n: usize) -> Vec<f64> {
        let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
        let mut rand_quantiles: Vec<f64> = std::vec![0.0; n];
        rng.fill(rand_quantiles.as_mut_slice());
    
        let d: f64 = self.b - self.a; 
        return rand_quantiles.iter().map(|q| self.a + q * d).collect::<Vec<f64>>();
    }
    
    fn quantile_multiple(&self, points: &[f64]) -> Vec<f64> {
        return points
            .iter()
            .map(|&x| self.quantile(x))
            .collect::<Vec<f64>>();
    }

    fn expected_value(&self) -> Option<f64> {
        return Some(0.5 * (self.a + self.b));
    }
    
    fn variance(&self) -> Option<f64> {
        return Some((1.0 / 12.0) * (self.b - self.a));
    }
    
    /// In this case it just returns the mean, but aly value in `[a, b]` is by definition
    /// the mode, since they are all equally likely. s
    fn mode(&self) -> f64 {
        return self.expected_value().unwrap();
    }
    
    fn median(&self) -> f64 {
        return self.expected_value().unwrap();
    }
    
    fn skewness(&self) -> Option<f64> {
        return Some(0.0);
    }
    
    fn kurtosis(&self) -> Option<f64> {
        return Some(3.0 - 6.0 / 5.0);
    }
    
    fn excess_kurtosis(&self) -> Option<f64> {
        return Some(-6.0 / 5.0);
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
            crate::euclid::Moments::Raw => {
                let ord: i32 = order as i32; 
                let num: f64 = self.b.powi(ord + 1) - self.a.powi(ord + 1); 
                let den: f64 = (order as f64 + 1.0) * (self.b - self.a); 
                return num / den; 
            },
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
        let integration_type: crate::euclid::IntegrationType = crate::euclid::IntegrationType::from_bounds(bounds);
        let (_, num_steps): (f64, usize) = crate::euclid::choose_integration_precision_and_steps(bounds);
    
        let moment: f64 = match integration_type {
            crate::euclid::IntegrationType::Finite => {
                let integration_fn = |x: f64| {
                    let std_inp: f64 = (x + minus_mean) * inv_std_dev;
                    std_inp.powi(order_exp) * self.pdf(x)
                };
    
                crate::euclid::numerical_integration_finite(integration_fn, bounds, num_steps as u64)
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
    
                crate::euclid::numerical_integration_finite(integration_fn, bounds, num_steps as u64)
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
    
                crate::euclid::numerical_integration_finite(integration_fn, bounds, num_steps as u64)
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
    
                crate::euclid::numerical_integration_finite(integration_fn, bounds, num_steps as u64)
            }
        };
    
        return moment;
    }
    
    fn entropy(&self) -> f64 {
        return (self.b - self.a).ln();
    }
    
    fn rejection_sample(&self, n: usize, _pdf_max: f64) -> Vec<f64> {
        // In this case it is equivalent to just normal sampling. 
        // It is already very effitient. 
        return self.sample_multiple(n);
    }
    
    fn rejection_sample_range(&self, n: usize, _pdf_max: f64, range: (f64, f64)) -> Vec<f64> {
        let new_uniform: Uniform = match Uniform::new(range.0, range.1) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        }; 
    
        return new_uniform.sample_multiple(n);
    }
}
