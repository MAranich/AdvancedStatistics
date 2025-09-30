//! # Gamma distribution
//!
//! The [Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution)
//! is a continuous probability distribution.
//!
//! It has 2 parameters, but there are 2 ways to model it:
//!
//! 1. `alpha` or shape
//! 2. `theta` or scale
//!
//! The other way is:
//!
//! 1. `alpha` or shape
//! 2. `lambda` or rate
//!
//! `theta = 1/lambda`
//!
//! All parameters (in every possible parametritzations) are stricly positive.
//!
//!

use rand::Rng;

use crate::{
    configuration,
    distribution_trait::{Distribution, Parametric},
    domain::ContinuousDomain,
    errors::AdvStatError,
    euclid,
};

use super::ChiSquared::ChiSquared;

// Values obtained from:
// Berg, Christian & Pedersen, Henrik L. (March 2006).
// ["The Chen–Rubin conjecture in a continuous setting"](https://www.intlpress.com/site/pub/files/_fulltext/journals/maa/2006/0013/0001/MAA-2006-0013-0001-a004.pdf)
//(PDF). Methods and Applications
// of Analysis. 13 (1): 63–88. doi:10.4310/MAA.2006.v13.n1.a4. S2CID 6704865. [Archived](https://web.archive.org/web/20210116114105/https://www.intlpress.com/site/pub/files/_fulltext/journals/maa/2006/0013/0001/MAA-2006-0013-0001-a004.pdf)
// (PDF) from the original on 16 January 2021. Retrieved 1 April 2020.
const GAMMA_MEDIAN_LAURENT_APPROX: [f64; 10] = [
    -1.0 / 3.0,
    f64::from_bits(4581350660937346637),
    f64::from_bits(4574964251307030381),
    f64::from_bits(4559158671624784028),
    f64::from_bits(13786726787819272789),
    f64::from_bits(13780326936167835008),
    f64::from_bits(4559263177061591461),
    f64::from_bits(4556933621898462772),
    f64::from_bits(13783575470603033594),
    f64::from_bits(13783519043403274501),
];

pub const GAMMA_DOMAIN: ContinuousDomain = ContinuousDomain::From(0.0);

#[derive(Debug, Clone, PartialEq)]
pub struct Gamma {
    /// alpha or shape
    alpha: f64,
    /// theta or scale
    theta: f64,
    normalitzation_constant: f64,
}

pub struct GammaGenerator {
    rng: rand::prelude::ThreadRng,
    exp: crate::distributions::Exponential::ExponentialGenerator,
    norm: crate::distributions::Normal::StdNormalGenerator,
    alpha: f64,
    theta: f64,
    inv_a: f64,
    b: f64,
    c: f64,
}

impl Gamma {
    /// Creates a new [Gamma] distribution with parameters `alpha` and `theta`.
    ///
    /// It will return error under any of the following conditions:
    ///  - `alpha` is `+-inf` or a NaN
    ///  - `theta` is `+-inf` or a NaN
    ///  - `alpha <= 0.0`
    ///  - `theta <= 0.0`
    ///  - The values for `alpha` and `theta` are too large to model properly
    ///      - This means that a [f64] value is not precise enough.
    ///      - Use [Gamma::new_unchecked] if you don't need to evaluate
    ///         the pdf direcly or indirecly.
    ///
    pub fn new(alpha: f64, theta: f64) -> Result<Gamma, AdvStatError> {
        if !alpha.is_finite() {
            if alpha.is_nan() {
                return Err(AdvStatError::NanErr);
            } else if alpha.is_infinite() {
                return Err(AdvStatError::InvalidNumber);
            }
        }

        if !theta.is_finite() {
            if theta.is_nan() {
                return Err(AdvStatError::NanErr);
            } else if theta.is_infinite() {
                return Err(AdvStatError::InvalidNumber);
            }
        }

        if alpha <= 0.0 {
            return Err(AdvStatError::InvalidNumber);
        }

        if theta <= 0.0 {
            return Err(AdvStatError::InvalidNumber);
        }

        let norm_const: f64 = euclid::gamma(alpha) * theta.powf(alpha);

        if !norm_const.is_finite() {
            // we do not have enough precision to do the computations
            return Err(AdvStatError::NumericalError);
        }

        return Ok(Gamma {
            alpha,
            theta,
            normalitzation_constant: 1.0 / norm_const,
        });
    }

    /// Creates a new [Gamma] distribution with parameters `alpha` and `theta`
    /// without checking for their correcness.
    ///
    /// ## Safety
    ///
    /// If any of the following conditions are met, the returned distribution
    /// will be invalid.
    ///
    ///  - `alpha` is `+-inf` or a NaN
    ///  - `theta` is `+-inf` or a NaN
    ///  - `alpha <= 0.0`
    ///  - `theta <= 0.0`
    ///  - The values for `alpha` and `theta` are too large to model properly
    ///      - This means that a [f64] value is not precise enough.
    ///
    #[must_use]
    pub unsafe fn new_unchecked(alpha: f64, theta: f64) -> Gamma {
        let norm_const: f64 = euclid::gamma(alpha) * theta.powf(alpha);

        return Gamma {
            alpha,
            theta,
            normalitzation_constant: 1.0 / norm_const,
        };
    }

    /// Get the parameter alpha
    #[must_use]
    pub const fn get_alpha(&self) -> f64 {
        return self.alpha;
    }

    /// Get the parameter theta
    #[must_use]
    pub const fn get_theta(&self) -> f64 {
        return self.theta;
    }

    /// Returns an iterator that can generate [Gamma] samples even faster
    /// than normally calling [Gamma::sample] many times. Uscefull if you don't
    /// know exacly how many values you want for [Gamma::sample_multiple].
    ///
    /// It avoids the heap allocation of [Gamma::sample_multiple] and
    /// the repeated initialitzation processes in [Gamma::sample].
    #[must_use]
    pub fn iter(&self) -> GammaGenerator {
        let b: f64 = self.alpha - (1.0 / 3.0);
        let c: f64 = 1.0 / (3.0 * b.sqrt());

        return GammaGenerator {
            rng: rand::rng(),
            exp: super::Exponential::Exponential::new(1.0).unwrap().iter(),
            norm: super::Normal::StdNormal::new().iter(),
            alpha: self.alpha,
            theta: self.theta,
            inv_a: 1.0 / self.alpha,
            b,
            c,
        };
    }

    #[must_use]
    pub fn from_chi_squared(chi_sq: &ChiSquared) -> Gamma {
        // SAFETY: if chi squared is valid, so will be the new gamma
        unsafe { Gamma::new_unchecked(chi_sq.get_degrees_of_freedom().get() as f64 * 0.5, 2.0) }
    }
}

impl Distribution for Gamma {
    fn pdf(&self, x: f64) -> f64 {
        /*
           $$
           f(x; \alpha, \theta) = \frac{1}{\theta^{\alpha} \cdot \Gamma(\alpha)} \cdot x^{\alpha - 1} \cdot e^{ \frac{-x}{\theta} }
           $$
           $$
           \frac{1}{\Gamma(\alpha)} \cdot x^{\alpha - 1} \cdot e^{-x }
           $$
        */

        let shape: f64 = x.powf(self.alpha - 1.0) * (-x / self.theta).exp();
        return self.normalitzation_constant * shape;
    }

    fn get_domain(&self) -> &crate::domain::ContinuousDomain {
        return &GAMMA_DOMAIN;
    }

    fn sample_fill(&self, buffer: &mut [f64]) {
        // https://en.wikipedia.org/wiki/Gamma_distribution#Random_variate_generation
        // https://github.com/numpy/numpy/blob/main/numpy/random/src/distributions/distributions.c#L220

        // let n: usize = buffer.len();
        let exp_distr: crate::distributions::Exponential::Exponential =
            super::Exponential::Exponential::new(1.0).unwrap();

        if self.alpha == 1.0 {
            exp_distr.sample_fill(buffer);
            return;
        } else if self.alpha == 0.0 {
            buffer.fill(0.0);
            return;
        }

        assert!(self.alpha != 0.0 && self.alpha != 1.0);

        let mut rng: rand::prelude::ThreadRng = rand::rng();

        let inv_a: f64 = 1.0 / self.alpha;

        if self.alpha < 1.0 {
            let mut exp: crate::distributions::Exponential::ExponentialGenerator = exp_distr.iter();
            for s in buffer.iter_mut() {
                let r: f64 = 'generate: loop {
                    let u: f64 = rng.random::<f64>();
                    // SAFETY: the iterator will always return the Some variant
                    let v: f64 = unsafe { exp.next().unwrap_unchecked() };

                    if u <= 1.0 - self.alpha {
                        let x: f64 = u.powf(inv_a);
                        if x <= v {
                            break 'generate x;
                        }
                    } else {
                        let y: f64 = -((1.0 - u) * inv_a).ln();
                        let x: f64 = (1.0 + self.alpha * (y - 1.0)).powf(inv_a);

                        if x <= (v + y) {
                            break 'generate x;
                        }
                    }
                };
                *s = r * self.theta;
            }
        } else {
            let mut norm: crate::distributions::Normal::StdNormalGenerator =
                super::Normal::StdNormal::new().iter();
            let b: f64 = self.alpha - (1.0 / 3.0);
            let c: f64 = 1.0 / (9.0 * b).sqrt();
            for s in buffer.iter_mut() {
                let r: f64 = 'generate: loop {
                    let mut x: f64;
                    let mut v: f64;
                    'sampl: loop {
                        // SAFETY: the iterator will always return the Some variant
                        x = unsafe { norm.next().unwrap_unchecked() };
                        v = 1.0 + c * x;
                        if v <= 0.0 {
                            break 'sampl;
                        }
                    }
                    v = v * v * v;
                    let u: f64 = rng.random::<f64>();

                    let x_sq: f64 = x * x;
                    let x_forth: f64 = x_sq * x_sq;
                    if u < 1.0 - 0.0331 * x_forth {
                        break 'generate b * v;
                    }

                    // log(0.0) ok here
                    if u.ln() < 0.5 * x_sq + b * (1.0 - v + v.ln()) {
                        break 'generate b * v;
                    }
                };
                *s = r * self.theta;
            }
        }
    }

    /*
        TODO: update gamma cdf, quantile and median.

        //https://en.wikipedia.org/wiki/Gamma_distribution#Characterization_using_shape_%CE%B1_and_scale_%CE%B8

        /*

        Desmos fn: g\left(x\right)\ =\frac{1}{t^{a}\cdot(a-1)!}\cdot\int_{0}^{x}u^{a-1}\cdot e^{\frac{-u}{t}}du

        */
    */

    // fn cdf_fill(&self, points: &mut [f64]) { todo!();}

    fn expected_value(&self) -> Option<f64> {
        return Some(self.alpha * self.theta);
    }

    fn variance(&self) -> Option<f64> {
        return Some(self.alpha * self.theta * self.theta);
    }

    fn mode(&self) -> f64 {
        return ((self.alpha - 1.0) * self.theta).max(0.0);
    }

    fn median(&self) -> f64 {
        // median has no simple closed form

        // desmos fn: \frac{1}{(x-1)!}\cdot\int_{0}^{y}t^{x-1}\cdot e^{-t}\ dt\ =\ 0.5
        // x is the shape alpha

        /*
        const TRESHOLD_LARGE_APPROXIMATION: f64 = 16.0;

        if TRESHOLD_LARGE_APPROXIMATION <= self.alpha {
            // very good and cheap approximation for large alpha.
            // is asymptotically accurate at high alpha.
            // at alpha = 8.0 it has an error of arround 1% and *should* decrease as alpha increases.
            let inner: f64 = 1.0 - 1.0 / (9.0 * self.alpha);
            let cubed: f64 = inner * inner * inner;

            let standardized_median: f64 = self.alpha * cubed;
            return standardized_median * self.theta;
        }
        */

        /*

        let $v(a)$ denote the median function and $a$ be the shape parameter (alpha).

        v(1.0) = ln(2) ~= 0.69314718056

        v(1.0001) ~= 0.6932439854
         */

        // relabeling (will get optimized away)
        let a: f64 = self.alpha;

        const TRESHOLD_LARGE_APPROXIMATION: f64 = 129.0;
        const TRESHOLD_MID_APPROXIMATION: f64 = 1.75;
        const TRESHOLD_LOW_APPROXIMATION: f64 = 0.05;

        if TRESHOLD_LARGE_APPROXIMATION <= a {
            // After this value, we can drop some terms since they get rounded down to 0

            // Desmos fn: f\left(x\right)\ =\ x\ -\ \frac{1}{3}\ +\ \frac{0.019753086419753086}{x}+\frac{0.007211444248481286}{x^{\ 2}}+\frac{0.0006526298981717363}{x^{\ 3}}\ -\frac{0.0012385769635511374}{x^{\ 4}}

            let inv: f64 = 1.0 / a;
            let laurent: f64 = GAMMA_MEDIAN_LAURENT_APPROX[4]
                .mul_add(inv, GAMMA_MEDIAN_LAURENT_APPROX[3])
                .mul_add(inv, GAMMA_MEDIAN_LAURENT_APPROX[2])
                .mul_add(inv, GAMMA_MEDIAN_LAURENT_APPROX[1])
                .mul_add(inv, GAMMA_MEDIAN_LAURENT_APPROX[0]);

            return laurent + a;
        } else if TRESHOLD_MID_APPROXIMATION <= a {
            // Desmos fn: f\left(x\right)\ =\ x\ -\ \frac{1}{3}\ +\ \frac{0.019753086419753086}{x}+\frac{0.007211444248481286}{x^{\ 2}}+\frac{0.0006526298981717363}{x^{\ 3}}\ -\frac{0.0012385769635511374}{x^{\ 4}}\ -\ \frac{0.0004509888376840951}{x^{\ 5}}+\frac{0.0006639604003340528}{x^{6}}+\frac{0.000449835386727579}{x^{\ 7}}-\frac{0.000765903234571818}{x^{8}}-\frac{0.0007597853853152098}{x^{\ 9}}

            let inv: f64 = 1.0 / a;
            let laurent: f64 = GAMMA_MEDIAN_LAURENT_APPROX[10]
                .mul_add(inv, GAMMA_MEDIAN_LAURENT_APPROX[9])
                .mul_add(inv, GAMMA_MEDIAN_LAURENT_APPROX[8])
                .mul_add(inv, GAMMA_MEDIAN_LAURENT_APPROX[7])
                .mul_add(inv, GAMMA_MEDIAN_LAURENT_APPROX[6])
                .mul_add(inv, GAMMA_MEDIAN_LAURENT_APPROX[5])
                .mul_add(inv, GAMMA_MEDIAN_LAURENT_APPROX[4])
                .mul_add(inv, GAMMA_MEDIAN_LAURENT_APPROX[3])
                .mul_add(inv, GAMMA_MEDIAN_LAURENT_APPROX[2])
                .mul_add(inv, GAMMA_MEDIAN_LAURENT_APPROX[1])
                .mul_add(inv, GAMMA_MEDIAN_LAURENT_APPROX[0]);

            return laurent + a;
        } else {
            /*
            return euclid::EXP_NEG_MASCH * (-euclid::LN_2 / a).exp();
            // at 0.1: pred: 0.0005483, real ~ 0.0007336

             */
        }

        // fallback
        return self.quantile(0.5);
    }

    fn skewness(&self) -> Option<f64> {
        return Some(2.0 / self.alpha.sqrt());
    }

    fn excess_kurtosis(&self) -> Option<f64> {
        return Some(6.0 / self.alpha);
    }

    fn moments(&self, order: u8, mode: euclid::Moments) -> f64 {
        if let euclid::Moments::Raw = mode {
            let mut acc: f64 = 1.0;
            let mut i: f64 = 1.0;
            while i <= f64::from(order) {
                acc = acc * (self.alpha + i - 1.0);
                i += 1.0;
            }
            return self.theta.powi(i32::from(order)) * acc;
        }

        // The values of 0.0 and 1.0 have no special meaning. They are not going to be used anyway.
        let (mean, variance): (f64, f64) = match mode {
            euclid::Moments::Raw => unreachable!(),
            euclid::Moments::Central => {
                // SAFETY: the `expected_value` fn will always return the some variant for the gamma distribution
                let mean: f64 = unsafe { self.expected_value().unwrap_unchecked() };
                (mean, 1.0)
            },
            euclid::Moments::Standarized => {
                // SAFETY: the `expected_value` fn will always return the some variant for the gamma distribution
                let mean: f64 = unsafe { self.expected_value().unwrap_unchecked() };
                // SAFETY: the `variance` fn will always return the some variant for the gamma distribution
                let variance: f64 = unsafe { self.variance().unwrap_unchecked() };
                (mean, variance)
            }
        };

        return self.default_moments(order, mean, variance);
    }

    fn entropy(&self) -> f64 {
        return self.alpha
            + self.theta.ln()
            + euclid::ln_gamma(self.alpha)
            + (1.0 - self.alpha) * euclid::digamma(self.alpha);
    }
}

impl Parametric for Gamma {
    /// Evaluates the [Gamma::pdf] in a general way taking into account
    /// the parameters.
    ///
    /// ### Parameters for Gamma:
    ///
    /// The [Gamma] distribution has 2 parameters `alpha = a` and `theta = t`.
    /// The order of the `parameters` is:
    ///
    /// > \[alpha, theta\]
    ///
    /// Alpha and theta must be both stricly positive.
    fn general_pdf(&self, x: f64, parameters: &[f64]) -> f64 {
        // pdf(x | a, t) = 1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t)
        let a: f64 = parameters[0];
        let t: f64 = parameters[1];
        let shape: f64 = x.powf(a - 1.0) * (-x / t).exp();
        return self.normalitzation_constant * shape;
    }

    fn number_of_parameters() -> u16 {
        return 2;
    }

    fn get_parameters(&self, parameters: &mut [f64]) {
        assert!(2 <= parameters.len(), "The buffer has not enough capacity. "); 
        parameters[0] = self.alpha;
        parameters[1] = self.theta;
    }

    fn derivative_pdf_parameters(&self, x: f64, parameters: &[f64]) -> Vec<f64> {
        // d/dx ln(f(x)) = f'(x)/f(x)
        // => f(x) * d/dx ln(f(x)) = f'(x)

        // pdf(x | a, t) = 1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t)

        // Reserve a vector with exacly 3 elements
        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(3);

        let a: f64 = parameters[0];
        let t: f64 = parameters[1];

        // = x/t
        let x_over_t: f64 = x / t;
        // = exp(-x/t)
        let exp_xt: f64 = (-x_over_t).exp();
        // = x^(a-2)
        let pow_a_2: f64 = x.powf(a - 2.0);
        // = t^a
        let t_raised_a: f64 = t.powf(a);
        // = 1 / (Gamma(a)*t^a)
        let norm_const: f64 = 1.0 / (euclid::gamma(a) * t_raised_a);

        {
            // ## Derivative respect to x:
            /*
                d/dx pdf(x | a, t) = d/dx 1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t)
                = 1/(Gamma(a)*t^a) * d/dx x^(a-1) * exp(-x/t)
                = 1/(Gamma(a)*t^a) * ((a-1)x^(a-2) * exp(-x/t) + x^(a-1) * exp(-x/t) * d/dx[ -x/t ])
                = 1/(Gamma(a)*t^a) * ((a-1)x^(a-2) * exp(-x/t) + x^(a-1) * exp(-x/t) * -1/t )
                = 1/(Gamma(a)*t^a) * ((a-1)x^(a-2) * exp(-x/t) + x^(a-2) * exp(-x/t) * x * -1/t )
                = 1/(Gamma(a)*t^a) * x^(a-2) * exp(-x/t) * (a - 1 - x/t)
                = pdf(x | a, t) / x * (a - 1 - x/t )

                    Confirmation:

                f(x) * d/dx ln(f(x)) = f'(x)
                d/dx ln(pdf(x | a, t)) = (a-1) / x - 1 / t

                1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t) * ((a-1) / x - 1 / t) =
                 = 1/(Gamma(a)*t^a) * x^(a-2) * x * exp(-x/t) * ((a-1) / x - 1 / t)
                 = 1/(Gamma(a)*t^a) * x^(a-2) * exp(-x/t) * (a - 1 - x/t)

                Wich is identical to our result for the normal computation.

            */
            // = (a - 1 - x/t )
            let term: f64 = a - 1.0 - x_over_t;
            // x^(a-2) * exp(-x/t)
            let shape: f64 = pow_a_2 * exp_xt;

            ret.push(norm_const * shape * term);
        }

        {
            // ## Derivative respect to a:
            /*
                d/da pdf(x | a, t) = d/da 1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t)
                 = exp(-x/t) * d/da x^(a-1)/(Gamma(a)*t^a)
                 = exp(-x/t) * d/da x^(a-1) * t^-a * Gamma(a)^-1
                 = exp(-x/t) * (d/da[ x^(a-1) ] * t^-a * Gamma(a)^-1 + x^(a-1) * d/da[ t^-a ] * Gamma(a)^-1 + x^(a-1) * t^-a * d/da[ Gamma(a)^-1 ] )
                 = exp(-x/t) * (ln(x) * x^(a-1) * t^-a * Gamma(a)^-1 + x^(a-1) * ln(t) * t^-a * d/da[ -a ] * Gamma(a)^-1 + x^(a-1) * t^-a * -1 * Gamma(a)^-2 * d/da[ Gamma(a) ] )
                 = exp(-x/t) * (ln(x) * x^(a-1) * t^-a * Gamma(a)^-1 + x^(a-1) * -ln(t) * t^-a * Gamma(a)^-1 + x^(a-1) * t^-a * -1 * Gamma(a)^-2 * Digamma(a) * Gamma(a) )
                 = exp(-x/t) * (ln(x) * x^(a-1) * t^-a * Gamma(a)^-1 + x^(a-1) * -ln(t) * t^-a * Gamma(a)^-1 + x^(a-1) * t^-a * -1 * Gamma(a)^-1 * Digamma(a) )
                 = exp(-x/t) * Gamma(a)^-1 * (ln(x) * x^(a-1) * t^-a + x^(a-1) * -ln(t) * t^-a + x^(a-1) * t^-a * -1 * Digamma(a) )
                 = exp(-x/t) * Gamma(a)^-1 * x^(a-1) * (ln(x) * t^-a + -ln(t) * t^-a + t^-a * -1 * Digamma(a) )
                 = exp(-x/t) * Gamma(a)^-1 * x^(a-1) * t^-a * (ln(x) + -ln(t) + -Digamma(a) )
                 = 1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t) * (ln(x) - ln(t) - Digamma(a))
                 = pdf(x | a, t) * (ln(x) + -ln(t) + -Digamma(a) )

                let norm_const = 1/(Gamma(a)*t^a);
                norm_const is already computed:

                 = norm_const * pow_a_2 * x * exp_xt * (ln(x) - ln(t) - Digamma(a))

                    Confirmation:

                f(x) * d/dx ln(f(x)) = f'(x)
                d/da ln(pdf(x | a, t)) = -Digamma(a) - ln(t) + ln(x)

                1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t) * (-Digamma(a) - ln(t) + ln(x)) =
                 = 1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t) * (ln(x) - ln(t) - Digamma(a))

                Wich is identical to our result for the normal computation.

            */

            // = ln(x) - ln(t) - Digamma(a)
            let term: f64 = x.ln() - t.ln() - euclid::digamma(a);

            ret.push(norm_const * pow_a_2 * x * exp_xt * term);
        }

        {
            // ## Derivative respect to t:
            /*
                d/dt pdf(x | a, t) = d/dt 1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t)
                 = x^(a-1)/Gamma(a) * d/dt 1/t^a * exp(-x/t)
                 = x^(a-1)/Gamma(a) * d/dt t^-a * exp(-x/t)
                 = x^(a-1)/Gamma(a) * d/dt exp(ln(t^-a)) * exp(-x/t)
                 = x^(a-1)/Gamma(a) * d/dt exp(-a * ln(t)) * exp(-x/t)
                 = x^(a-1)/Gamma(a) * d/dt exp(-a * ln(t) - x/t)
                 = x^(a-1)/Gamma(a) * exp(-a * ln(t) - x/t) * d/dt -a * ln(t) - x/t
                 = x^(a-1)/Gamma(a) * exp(-a * ln(t) - x/t) * (d/dt[ -a * ln(t) ] - d/dx[ x/t ])
                 = x^(a-1)/Gamma(a) * exp(-a * ln(t) - x/t) * (-a * d/dt[ ln(t) ] - x * d/dx[ 1/t ])
                 = x^(a-1)/Gamma(a) * exp(-a * ln(t) - x/t) * (-a * 1/t - x * -1/t^2 )
                 = x^(a-1)/Gamma(a) * exp(-a * ln(t) - x/t) * (-a/t + x/t^2)
                 = x^(a-1)/Gamma(a) * t^-a * exp(-x/t) * (-a/t + x/t^2)
                 = x^(a-1)/Gamma(a) * t^-a * exp(-x/t) * (-a + x/t)/t
                 = x^(a-1)/(Gamma(a)*t^a) * exp(-x/t) * (-a + x/t)/t
                 = 1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t) * (x/t - a)/t
                 = pdf(x | a, t) * (x/t - a)/t

                Substituting precomputed variables:
                 = pow_a_2 * x * norm_const * exp_xt * (-a + x_over_t)/t


                    Confirmation:

                f(x) * d/dx ln(f(x)) = f'(x)
                d/dx ln(pdf(x | a, t)) = [...]

                1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t) * (1/t * (x / t - a)) =
                 = 1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t) * (x/t - a)/t

                Wich is identical to our result for the normal computation.

            */
            // = (-a + x_over_t)/t
            let term: f64 = (x_over_t - a) / t;

            ret.push(pow_a_2 * x * norm_const * exp_xt * term);
        }

        return ret;
    }

    fn log_derivative_pdf_parameters(&self, x: f64, parameters: &[f64]) -> Vec<f64> {
        // d/dx ln(f(x)) = f'(x)/f(x)

        // pdf(x | a, t) = 1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t)
        // ln(pdf(x | a, t)) = ln(1/(Gamma(a)*t^a) * x^(a-1) * exp(-x/t))
        // ln(pdf(x | a, t)) = -ln(Gamma(a)*t^a) + ln(x^(a-1)) + ln(exp(-x/t))
        // ln(pdf(x | a, t)) = -(ln(Gamma(a)) + ln(t^a)) + (a-1)*ln(x) - x / t
        // ln(pdf(x | a, t)) = -(ln(Gamma(a)) + a*ln(t)) + (a-1)*ln(x) - x / t
        // ln(pdf(x | a, t)) = -ln(Gamma(a)) - a*ln(t) + (a-1)*ln(x) - x / t

        // Reserve a vector with exacly 3 elements
        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(3);

        let a: f64 = parameters[0];
        let t: f64 = parameters[1];

        let inv_t: f64 = 1.0 / t;

        {
            // ## Log derivative respect to x:
            /*
                d/dx ln(pdf(x | a, t)) = d/dx -ln(Gamma(a)) - a*ln(t) + (a-1)*ln(x) - x / t
                 = 0 - 0 + (a-1)/x - 1 / t
                 = (a-1) / x - 1 / t

            */

            ret.push((a - 1.0) / x - inv_t);
        }

        {
            // ## Log derivative respect to a:
            /*
                d/da ln(pdf(x | a, t)) = d/da -ln(Gamma(a)) - a*ln(t) + (a-1)*ln(x) - x / t
                 = -Digamma(a) - ln(t) + ln(x)

            */

            ret.push(-euclid::digamma(a) - t.ln() + x.ln());
        }

        {
            // ## Log derivative respect to t:
            /*
                d/dt ln(pdf(x | a, t)) = d/dt -ln(Gamma(a)) - a*ln(t) + (a-1)*ln(x) - x / t
                 = 0 - a/t + 0 + x / t^2
                 = - a/t + x / t^2
                 = 1/t * (-a + x / t)
                 = 1/t * (x / t - a)

            */

            ret.push(inv_t * (x * inv_t - a));
        }

        return ret;
    }

    fn parameter_restriction(&self, _parameters: &mut [f64]) {}

    fn fit(&self, data: &mut crate::samples::Samples) -> Vec<f64> {
        /*
            **Recommended wiewing with Latex visualizer / obsidian or something**
                Using Maximum Likelyhood estimation:
            Assuming n samples.

            
            $$
            f(x; \alpha, \theta) = \frac{1}{\theta^{\alpha} \cdot \Gamma(\alpha)} \cdot x^{\alpha - 1} \cdot e^{ \frac{-x}{\theta} }
            $$
            $$
            \ln (f(x; \alpha, \theta)) = \ln\left( \frac{1}{\theta^{\alpha} \cdot \Gamma(\alpha)} \cdot x^{\alpha - 1} \cdot e^{ \frac{-x}{\theta} } \right)
            $$
            $$
            = -\alpha \ln(\theta) - \ln\Gamma(\alpha) + (\alpha - 1) \ln(x) - \frac{x}{\theta}
            $$



            The maximum likelyhood estimation is defined as: 
            $$
            L(\alpha, \theta) = \prod_{i=1}^n f(x_{i}; \alpha, \theta)
            $$
            $$
            \ell(\alpha, \theta) 
            = \ln(L(\alpha, \theta))
            = \sum_{i=1}^n -\alpha \ln(\theta) - \ln\Gamma(\alpha) + (\alpha - 1) \ln(x_{i}) - \frac{x_{i}}{\theta}
            $$
            $$
            = n \cdot (-\alpha \ln(\theta) - \ln\Gamma(\alpha)) + \sum_{i=1}^n  (\alpha - 1) \ln(x_{i}) - \frac{x_{i}}{\theta}
            $$
            By using $n \cdot \overline {x} = \sum_{i} x_{i}$ , we can simplify further: 
            $$
            = 
            n \cdot 
            \left(
            - \alpha \ln(\theta) 
            - \ln\Gamma(\alpha)) 
            + (\alpha - 1) \overline{\ln(x)} 
            - \frac{\overline{x}}{\theta}
            \right)
            $$
            Differentiating respect to $\theta$ and equating to $0$ (the $n$ can be 
            discarded as $n \neq 0$): 
            $$
            0 = \frac{ \partial  }{ \partial \theta }  \ell(\alpha, \theta) = 
            \frac{ \partial  }{ \partial \theta }  
            - \alpha \ln(\theta) 
            - \ln\Gamma(\alpha)) 
            + (\alpha - 1) \overline{\ln(x)} 
            - \frac{\overline{x}}{\theta}
            $$
            $$
            0 = 
            - \frac{\alpha}{\theta} 
            + \frac{\overline{x}}{\theta^{2}} 
            = 
            \frac{1}{\theta} \cdot 
            \left(
                - \alpha 
                + \frac{\overline{x}}{\theta}
            \right) 
            $$
            Since $\frac{1}{\theta} \neq 0$, the other term must equal $0$ . 

            $$
            0 = 
            - \alpha 
            + \frac{\overline{x}}{\theta}
            $$
            Now solving for $\theta$ : 
            $$
            \theta
            = 
            \frac{\overline{x}}{ \alpha }
            $$
            Now we have a very simple closed form solution for $\theta$ given 
            $\alpha$ . Plugging this result back into the log likelyhood function: 

            $$
            \ell(\alpha) 
            = 
            n \cdot 
            \left(
                - (\alpha - 1) \overline{\ln(x)} 
                - \alpha
                + \alpha \ln(\frac{\overline{x}}{ \alpha }) 
                - \ln\Gamma(\alpha)) 
            \right)
            $$

            According to wikipedia, solving for $\alpha$ when equating to $0$ 
            has no closed form solution, but is a well behaved function. Differentiating 
            respect to $\alpha$ and equating to $0$ : 
            $$
            0 = \frac{ \partial  }{ \partial \alpha }  \ell(\alpha) = 
            n \cdot 
            \left(
                \frac{ \partial  }{ \partial \alpha }
                
                (\alpha - 1) \overline{\ln(x)} 
                - \alpha
                - \alpha \ln(\overline{x}) 
                + \alpha \ln( \alpha ) 
                - \ln\Gamma(\alpha)) 
            \right)
            $$
            $$
            0 = 
            \overline{\ln(x)} 
            - 1
            - \ln(\overline{x}) 
            + \ln( \alpha ) + 1 
            - \psi_{0}(\alpha)) 
            $$
            $$
            \ln( \alpha ) 
            - \psi_{0}(\alpha)
            = 
            \ln(\overline{x}) 
            -\overline{\ln(x)} 
            $$
            We can use this result to use Newton's method to find a numerical 
            value for $\alpha$ . Let $s = \ln(\overline{x}) -\overline{\ln(x)}$. 
            Then we can define the function $g(x) = \ln( \alpha ) - \psi_{0}(\alpha) - s$ . 
            Therefore the Newton method update step is: 
            $$
            \alpha_{k+1} = \alpha_{k} - \frac{g(x)}{g^\prime (x)} = \alpha_{k} - \frac{\ln( \alpha ) - \psi_{0}(\alpha) - s}{\frac{1}{\alpha} - \psi_{1}(\alpha)}
            $$

            For the initial value of $a_0$, accordint to [wikipedia](https://en.wikipedia.org/wiki/Gamma_distribution#Maximum_likelihood_estimation), 
            the following formula gives an acceptable estimate: 
            $$
            a_0 = \frac{3 - s + \sqrt{ (s - 3)^2 + 24s }}{12s} 
            $$
            Where $s = \ln(\overline{x}) - \overline{\ln(x)}$ . 
            
            Note that $s$ is always positive, therefore $a_0$ exists as long as samples are positive.

        */

        let mut parameters: Vec<f64> = Vec::new();
        parameters.reserve_exact(2);

        let mean: f64 = match data.mean() {
            Some(v) => {
                if v <= 0.0 {
                    return Vec::new();
                }
                v
            }
            None => return Vec::new(),
        };
        // if None, then means there was not enough data

        let mean_log: f64 = match data.log_mean() {
            Some(v) => v,
            None => return Vec::new(),
        };
        // If None, it means that the data did not come from a Gamma distribution

        // s = ln( mean{x_i} ) - mean{x_i}[ ln(x_i) ]
        let s: f64 = mean.ln() * mean_log;
        assert!(0.0 <= s);

        let mut a: f64 = {
            // a_0 = (3 - s + sqrt((3 - s)^2 + 24 * s)) / 12*s
            let aux: f64 = 3.0 - s;
            (aux - (aux * aux + 24.0 * s).sqrt()) / (12.0 * s)
        };

        // Newton's method:

        // SAFETY: should always be safe to only read
        let convergence_epsilon: f64 = unsafe {
            configuration::maximum_likelihood_estimation::CONVERGENCE_DIFFERENCE_CRITERIA
        };

        let mut difference: f64 = f64::MAX;

        // max iterations
        let mut i: i32 = 0; 
        const MAX_ITTERATIONS_NEWTON_GAMMA: i32 = 200; // arbitrary

        while convergence_epsilon < difference && i < MAX_ITTERATIONS_NEWTON_GAMMA {
            //a_i+1 = a_i - (Digamma(a_i) - ln(a_i) + s)*h / (Digamma(a_i + h) - Digamma(a_i) - h/a_i)

            let digamma: f64 = euclid::digamma(a);
            let trigamma: f64 = euclid::fast_trigamma(a);

            let num: f64 = digamma - a.ln() + s;
            let den: f64 = trigamma - 1.0 / a;

            difference = num / den;
            a = a - difference;
            i = i + 1; 
            if !a.is_finite() {
                // we are not in normal numbers, newton's method has not converged. 
                i = MAX_ITTERATIONS_NEWTON_GAMMA; 
            }
        }

        if i == MAX_ITTERATIONS_NEWTON_GAMMA {
            // convergence has not been achieved. 
            return Vec::new();
        }

        // a has converged

        // t = 1/a * mean{x_i}
        let t: f64 = mean / a;

        parameters.push(a);
        parameters.push(t);

        return parameters;
    }
}

impl Iterator for GammaGenerator {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        // similar implenentation as [Gamma::sample] but better.
        // removed comments

        // https://en.wikipedia.org/wiki/Gamma_distribution#Random_variate_generation
        // https://github.com/numpy/numpy/blob/main/numpy/random/src/distributions/distributions.c#L220

        if self.alpha == 1.0 {
            return self.exp.next();
        }

        assert!(self.alpha != 0.0 && self.alpha != 1.0);

        let r: f64 = if 1.0 < self.alpha {
            'generate: loop {
                let u: f64 = self.rng.random::<f64>();
                let v: f64 = self.exp.next().unwrap();

                if u <= self.alpha {
                    let x: f64 = u.powf(self.inv_a);
                    if x <= v {
                        break 'generate x;
                    }
                } else {
                    let y: f64 = -(u * self.inv_a).ln();
                    let x: f64 = (1.0 - self.alpha + self.alpha * y).powf(self.inv_a);

                    if x <= (v + y) {
                        break 'generate x;
                    }
                }
            }
        } else {
            'generate: loop {
                let mut x: f64;
                let mut v: f64;
                's: loop {
                    x = self.norm.next().unwrap();
                    v = 1.0 + self.c * x;
                    if v <= 0.0 {
                        break 's;
                    }
                }
                v = v * v * v;
                let u: f64 = self.rng.random::<f64>();

                let x_sq: f64 = x * x;
                if u < 1.0 - 0.0331 * x_sq * x_sq {
                    break 'generate self.b * v;
                }

                if u.ln() < 0.5 * x_sq + self.b * (1.0 - v + v.ln()) {
                    break 'generate self.b * v;
                }
            }
        };

        return Some(r * self.theta);
    }
}

impl Default for Gamma {
    fn default() -> Self {
        return Gamma::new(1.0, 1.0).unwrap();
    }
}
