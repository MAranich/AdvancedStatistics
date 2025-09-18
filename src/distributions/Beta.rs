//! # Beta distribution
//!
//! The [Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution)
//! is a continuous probability distribution. It's domain is just `[0.0, 1.0]`
//! (wich can be uscefull to model probabilities). It has 2 parameters:
//! alpha and beta.
//!
//! It is the conjugate prior of the following distributions:
//!  - [Bernoulli](crate::distributions::Bernoulli),
//!  - [Binomial](crate::distributions::Binomial),
//!  - [Negative binomial](crate::distributions:: ...todo... )
//!  - [Geometric](crate::distributions::Geometric)
//!
//!

use crate::{
    distribution_trait::{Distribution, Parametric},
    domain::ContinuousDomain,
    errors::AdvStatError,
    euclid::{self, digamma},
};

pub const BETA_DOMAIN: ContinuousDomain = ContinuousDomain::Range(0.0, 1.0);
#[derive(Debug, Clone, PartialEq)]
pub struct Beta {
    alpha: f64,
    beta: f64,
    normalitzation_constant: f64,
}

impl Beta {
    /// Creates a new [Beta] distribution with parameters `alpha` and `beta`.
    ///
    ///  - `alpha` must be finite and be stricly positive.
    ///  - `beta` must be finite and be stricly positive.
    ///
    /// Otherwise an error will be returned.
    ///
    /// An error will also be returned if `alpha` and `beta` are too
    /// large to model properly.
    /// - This means that a [f64] value is not precise enough.
    /// - Use [Beta::new_unchecked] if you don't need to evaluate
    ///         the pdf direcly or indirecly.
    pub fn new(alpha: f64, beta: f64) -> Result<Beta, AdvStatError> {
        if !alpha.is_finite() {
            if alpha.is_nan() {
                return Err(AdvStatError::NanErr);
            } else if alpha.is_infinite() {
                return Err(AdvStatError::InvalidNumber);
            }
        }

        if !beta.is_finite() {
            if beta.is_nan() {
                return Err(AdvStatError::NanErr);
            } else if beta.is_infinite() {
                return Err(AdvStatError::InvalidNumber);
            }
        }

        if alpha <= 0.0 {
            return Err(AdvStatError::InvalidNumber);
        }

        if beta <= 0.0 {
            return Err(AdvStatError::InvalidNumber);
        }

        let norm: f64 = 1.0 / euclid::beta_fn(alpha, beta);

        if !norm.is_finite() {
            // we do not have enough precision to do the computations
            return Err(AdvStatError::NumericalError);
        }

        return Ok(Beta {
            alpha,
            beta,
            normalitzation_constant: norm,
        });
    }

    /// Creates a new [Beta] distribution with parameters `alpha` and `beta`.
    ///
    /// ## Safety
    ///
    /// If the following conditions are not fullfiled, the returned distribution
    /// will be invalid.
    ///
    ///  - `alpha` must be finite and be stricly positive.
    ///  - `beta` must be finite and be stricly positive.
    ///  - `alpha` and `beta` are too large to model properly.
    ///
    #[must_use]
    pub unsafe fn new_unchecked(alpha: f64, beta: f64) -> Beta {
        let norm_ct: f64 = 1.0 / euclid::beta_fn(alpha, beta);

        return Beta {
            alpha,
            beta,
            normalitzation_constant: norm_ct,
        };
    }

    #[must_use]
    pub const fn get_alpha(&self) -> f64 {
        return self.alpha;
    }

    #[must_use]
    pub const fn get_beta(&self) -> f64 {
        return self.beta;
    }
}

impl Distribution for Beta {
    fn pdf(&self, x: f64) -> f64 {
        // let a = alpha, let b = beta for conciseness sake.
        // pdf(x | a, b) = Gamma(a+b-1)/(Gamma(a-1)*Gamma(b-1)) * x^(a-1) * (1 - x)^(b-1)
        // pdf(x | a, b) = norm * x^(a-1) * (1 - x)^(b-1)

        let pow_alpha: f64 = x.powf(self.alpha - 1.0);
        let pow_beta: f64 = (1.0 - x).powf(self.beta - 1.0);
        return self.normalitzation_constant * pow_alpha * pow_beta;
    }

    fn get_domain(&self) -> &crate::domain::ContinuousDomain {
        return &BETA_DOMAIN;
    }

    // default cdf and quantile fn

    fn sample_fill(&self, buffer: &mut [f64]) {
        // https://en.wikipedia.org/wiki/Beta_distribution#Random_variate_generation

        let n: usize = buffer.len();

        // we will reuse the space in buffer to avoid a memory allocation
        let gamma_alpha_distr: super::Gamma::Gamma =
            super::Gamma::Gamma::new(self.alpha, 1.0).unwrap();
        gamma_alpha_distr.sample_fill(buffer);

        // we do not have more *free* space, allocate more space
        let gamma_beta: Vec<f64> = {
            let gamma_beta_distr: super::Gamma::Gamma =
                super::Gamma::Gamma::new(self.beta, 1.0).unwrap();
            gamma_beta_distr.sample_multiple(n)
        };

        // all unsafe accesses are safe because of the following assert
        // wew will reuse the vector `gamma_alpha_samples` so we don't do an extra allocation
        for i in 0..n {
            // SAFETY: `i` is in range
            let a: f64 = unsafe { *buffer.get_unchecked(i) };
            // SAFETY: `i` is in range
            let b: f64 = unsafe { *gamma_beta.get_unchecked(i) };
            // SAFETY: `i` is in range
            let reference: &mut f64 = unsafe { buffer.get_unchecked_mut(i) };

            *reference = a / (a + b);
        }
    }

    fn expected_value(&self) -> Option<f64> {
        return Some(self.alpha / (self.alpha + self.beta));
    }

    fn variance(&self) -> Option<f64> {
        let ab: f64 = self.alpha + self.beta;
        return Some(self.alpha * self.beta / (ab * ab * (ab + 1.0)));
    }

    fn mode(&self) -> f64 {
        // alpha anb beta are non-nans, unwrap is safe
        match (
            self.alpha.partial_cmp(&1.0).unwrap(),
            self.beta.partial_cmp(&1.0).unwrap(),
        ) {
            (std::cmp::Ordering::Equal, std::cmp::Ordering::Equal) => 0.5,
            (std::cmp::Ordering::Less, _) => 0.0,
            (_, std::cmp::Ordering::Less) => 1.0,
            _ => {
                // "normal" case
                (self.alpha - 1.0) / (self.alpha + self.beta - 2.0)
            }
        }
    }

    // default median

    fn skewness(&self) -> Option<f64> {
        // formula from wiki
        // Sk = 2*(b-a) * sqrt(a+b+1) / ((a+b+2) * sqrt(a*b))
        // Sk = 2*(b-a) / (a+b+2) * sqrt(a+b+1) / sqrt(a*b)
        // Sk = 2*(b-a) / (a+b+2) * sqrt((a+b+1) / a*b)
        // Sk = 2*(b-a) * sqrt((a+b+1) / a*b)  /  (a+b+2)
        // This formula is better because we do 1 sqrt less

        // relabeling for convinience (will get optimized away)
        let a: f64 = self.alpha; 
        let b: f64 = self.beta; 

        let u: f64 = a + b + 1.0; 

        let inner_sqrt: f64 = u / (a * b); 
        let num: f64 = 2.0 * (b - a) * inner_sqrt.sqrt();
        let den: f64 = u + 1.0;
        return Some(num / den);
    }

    fn excess_kurtosis(&self) -> Option<f64> {
        let a_plus_b: f64 = self.alpha + self.beta;
        let a_minus_b: f64 = self.alpha - self.beta;
        let mul_a_b: f64 = self.alpha * self.beta;

        let num: f64 = a_minus_b * a_minus_b * (a_plus_b + 1.0) - mul_a_b * (a_minus_b + 2.0);
        let den: f64 = mul_a_b * (a_minus_b + 2.0) * (a_minus_b + 3.0);

        return Some(6.0 * num / den);
    }

    fn moments(&self, order: u8, mode: euclid::Moments) -> f64 {

        if let euclid::Moments::Raw = mode {
            let ab: f64 = self.alpha + self.beta;
            let mut acc: f64 = 1.0;
            for r in 0..order {
                let r: f64 = f64::from(r);
                acc = acc * (self.alpha + r) / (ab + r);
            }
            return acc;
        }

        // The values of 0.0 and 1.0 have no special meaning. They are not going to be used anyway.
        let (mean, variance): (f64, f64) = match mode {
            euclid::Moments::Raw => unreachable!(),
            euclid::Moments::Central => (
                self.expected_value()
                    .expect("Tried to compute a central moment but the expected value is undefined. "),
                1.0,
            ),
            euclid::Moments::Standarized => (
                self.expected_value()
                    .expect("Tried to compute a central/standarized moment but the Expected value is undefined. "),
                self.variance().expect("Tried to compute a standarized moment but the variance is undefined. "),
            ),
        };

        return self.default_moments(order, mean, variance); 
    }

    fn entropy(&self) -> f64 {
        // https://en.wikipedia.org/wiki/Beta_distribution#Quantities_of_information_(entropy)

        // B(a, b) = gamma(a+b) / (gamma(a)*gamma(b)) = self.normalitzation_constant
        // ln(B(a, b)) - (a-1)*digamma(a) - (b-1)*digamma(b) + (a+b-2)*digamma(a+b)

        let term_1: f64 = self.normalitzation_constant.ln();
        let term_2: f64 = -(self.alpha - 1.0) * digamma(self.alpha);
        let term_3: f64 = -(self.beta - 1.0) * digamma(self.beta);
        let term_4: f64 = (self.alpha + self.beta - 2.0) * digamma(self.alpha + self.beta);

        return term_1 + term_2 + term_3 + term_4;
    }
}

impl Parametric for Beta {
    /// Evaluates the [Beta::pdf] in a general way taking into account
    /// the parameters.
    ///
    /// ### Parameters for Beta:
    ///
    /// The [Beta] distribution has 2 parameters `alpha = a` and `beta = b`.
    /// The order of the `parameters` is:
    ///
    /// > \[alpha, beta\]
    ///
    /// Alpha and beta must be both stricly positive.
    fn general_pdf(&self, x: f64, parameters: &[f64]) -> f64 {
        // let a = alpha, let b = beta for conciseness sake.
        // pdf(x | a, b) = Gamma(a+b-1)/(Gamma(a-1)*Gamma(b-1)) * x^(a-1) * (1 - x)^(b-1)
        // pdf(x | a, b) = norm * x^(a-1) * (1 - x)^(b-1)

        let a: f64 = parameters[0];
        let b: f64 = parameters[1];
        let norm: f64 = 1.0 / euclid::beta_fn(a, b);

        let pow_alpha: f64 = x.powf(a - 1.0);
        let pow_beta: f64 = (1.0 - x).powf(b - 1.0);
        return norm * pow_alpha * pow_beta;
    }

    fn number_of_parameters() -> u16 {
        return 2;
    }

    fn get_parameters(&self, parameters: &mut [f64]) {
        parameters[0] = self.alpha;
        parameters[1] = self.beta;
    }

    fn derivative_pdf_parameters(&self, x: f64, parameters: &[f64]) -> Vec<f64> {
        // d/dx ln(f(x)) = f'(x)/f(x)
        // => f(x) * d/dx ln(f(x)) = f'(x)

        // Reserve a vector with exacly 3 elements
        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(3);

        let a: f64 = parameters[0];
        let b: f64 = parameters[1];
        let digamma_ab: f64 = euclid::digamma(a + b - 1.0);
        let pdf: f64 = self.general_pdf(x, parameters);

        {
            // ## Derivative respect to x:
            /*
               d/dx pdf(x | a, b) = d/dx norm * x^(a-1) * (1 - x)^(b-1)
                = norm * d/dx x^(a-1) * (1 - x)^(b-1)
                = norm * ( d/dx[ x^(a-1) ] * (1 - x)^(b-1) + x^(a-1) * d/dx[ (1 - x)^(b-1) ] )
                = norm * ( (a-1) * x^(a-2) * (1 - x)^(b-1) + x^(a-1) * (b-1) * (1 - x)^(b-2) d/dx[ 1 - x ] )
                = norm * ( (a-1) * x^(a-2) * (1 - x)^(b-1) + x^(a-1) * (b-1) * (1 - x)^(b-2) * -1 )
                = norm * x^(a-2) * ( (a-1) * (1 - x)^(b-1) + x * (b-1) * (1 - x)^(b-2) * -1 )
                = norm * x^(a-2) * (1 - x)^(b-2) * ( (a-1) * (1 - x) + x * (b-1) * -1 )
                = norm * x^(a-2) * (1 - x)^(b-2) * ( (a-1) * (1 - x) - (b-1) * x )

               Confirmation:

               f(x) * d/dx ln(f(x)) = f'(x)
               d/dx ln(pdf(x | a, b)) = (a-1)/x - (b-1)/(1 - x)

               norm * x^(a-1) * (1 - x)^(b-1) * ((a-1)/x - (b-1)/(1 - x)) =
                = norm * (x^(a-1) * (1 - x)^(b-1) * (a-1)/x - x^(a-1) * (1 - x)^(b-1) * (b-1)/(1 - x))
                = norm * (x^(a-2) * (1 - x)^(b-1) * (a-1) - x^(a-1) * (1 - x)^(b-2) * (b-1))
                = norm * (x^(a-2) * (1 - x)^(b-2) * (1 - x) * (a-1) - x^(a-2) * x * (1 - x)^(b-2) * (b-1))
                = norm * x^(a-2) * (1 - x)^(b-2) * ((1 - x) * (a-1) - x * (b-1))
                = norm * x^(a-2) * (1 - x)^(b-2) * ( (a-1) * (1 - x) - (b-1) * x )

               Wich is identical to our result for the normal computation.

            */

            // = (a-1) * (1 - x) - (b-1) * x
            let term: f64 = (self.alpha - 1.0) * (1.0 - x) - (self.beta - 1.0) * x;

            let alpha_pow: f64 = x.powf(self.alpha - 2.0);
            let beta_pow: f64 = (1.0 - x).powf(self.beta - 2.0);

            ret.push(self.normalitzation_constant * alpha_pow * beta_pow * term);
        }

        {
            // ## Derivative respect to alpha:
            /*
                let a = alpha
                let b = beta
                let norm(a) = Gamma(a+b-1)/(Gamma(a-1)*Gamma(b-1))

                d/da pdf(x | a, b) = d/da[ norm(a) * x^(a-1) * (1 - x)^(b-1) ]
                 = (1 - x)^(b-1) * d/da[ norm(a) * x^(a-1) ]
                 = (1 - x)^(b-1) * (d/da[ norm(a) ] * x^(a-1) + norm(a) * d/da[ exp((a-1) * ln(x)) ])
                 = (1 - x)^(b-1) * ( norm'(a) * x^(a-1) + norm(a) * x^(a-1) * ln(x) )

                Definition: d/dx Gamma(x) = Gamma(x) * PolyGamma_0(x)

                norm'(a) = d/da norm(a) = d/da Gamma(a+b-1)/(Gamma(a-1)*Gamma(b-1))
                 = d/da[ 1/Gamma(b-1) * Gamma(a+b-1)/Gamma(a-1) ]
                 = 1/Gamma(b-1) * d/da[ Gamma(a+b-1)/Gamma(a-1) ]
                 = 1/Gamma(b-1) * (d/da[ Gamma(a+b-1) ] * Gamma(a-1) - d/da[ Gamma(a-1) ] * Gamma(a+b-1)) / Gamma(a-1)^2
                 = 1/(Gamma(b-1) * Gamma(a-1)^2) * (d/da[ Gamma(a+b-1) ] * Gamma(a-1) - d/da[ Gamma(a-1) ] * Gamma(a+b-1))
                 = 1/(Gamma(b-1) * Gamma(a-1)^2) * (Gamma(a+b-1) * PolyGamma_0(a+b-1) * Gamma(a-1) - Gamma(a-1) * PolyGamma_0(a - 1) * Gamma(a+b-1))
                 = 1/(Gamma(b-1) * Gamma(a-1)) * (Gamma(a+b-1) * PolyGamma_0(a+b-1) - PolyGamma_0(a - 1) * Gamma(a+b-1))
                 = Gamma(a+b-1)/(Gamma(a-1)*Gamma(b-1)) * (PolyGamma_0(a+b-1) - PolyGamma_0(a - 1))
                 = norm(a) * (PolyGamma_0(a+b-1) - PolyGamma_0(a - 1))

                // using calculator, the results for norm'(a) are correct

                d/da pdf(x | a, b) =
                 = (1 - x)^(b-1) * ( norm'(a) * x^(a-1) + norm(a) * x^(a-1) * ln(x) )
                 = (1 - x)^(b-1) * ( norm(a) * (PolyGamma_0(a+b-1) - PolyGamma_0(a - 1)) * x^(a-1) + norm(a) * x^(a-1) * ln(x) )
                 = (1 - x)^(b-1) * norm(a) * ((PolyGamma_0(a+b-1) - PolyGamma_0(a - 1)) * x^(a-1) + x^(a-1) * ln(x) )
                 = (1 - x)^(b-1) * norm(a) * x^(a-1) * ((PolyGamma_0(a+b-1) - PolyGamma_0(a - 1)) + ln(x) )
                 = norm(a) * x^(a-1) * (1 - x)^(b-1) * (PolyGamma_0(a+b-1) - PolyGamma_0(a - 1) + ln(x) )
                 = pdf(x | a, b) * (PolyGamma_0(a+b-1) - PolyGamma_0(a - 1) + ln(x) )

                // using calculator, the final results for d/da pdf(x | a, b) are correct.

                Confirmation:

                f(x) * d/dx ln(f(x)) = f'(x)
                d/da ln(pdf(x | a, b)) = Digamma(a+b-1) - Digamma(a-1) + ln(x)

                norm(a) * x^(a-1) * (1 - x)^(b-1) * (Digamma(a+b-1) - Digamma(a-1) + ln(x))

                Note that Digamma(x) = PolyGamma_0(x)

                norm(a) * x^(a-1) * (1 - x)^(b-1) * (PolyGamma_0(a+b-1) - PolyGamma_0(a-1) + ln(x))
                pdf(x | a, b) * (PolyGamma_0(a+b-1) - PolyGamma_0(a - 1) + ln(x) )

                Wich is identical to our result for the normal computation.
            */

            // = PolyGamma_0(a+b-1) - PolyGamma_0(a - 1) + ln(x)
            let term: f64 = digamma_ab - euclid::digamma(a - 1.0) + x.ln();

            ret.push(pdf * term);
        }

        {
            // ## Derivative respect to beta:
            /*
                let a = alpha
                let b = beta
                let norm(b) = Gamma(a+b-1)/(Gamma(a-1)*Gamma(b-1))

                d/db pdf(x | a, b) = d/db[ norm(b) * x^(a-1) * (1 - x)^(b-1) ]
                 = x^(a-1) * d/db[ norm(b) * (1 - x)^(b-1) ]
                 = x^(a-1) * (norm'(b) * (1 - x)^(b-1) + norm(b) * d/db[ (1 - x)^(b-1) ])
                 = x^(a-1) * (norm'(b) * (1 - x)^(b-1) + norm(b) * (1 - x)^(b-1) * ln(1 - x))
                 = x^(a-1) * (1 - x)^(b-1) * (norm'(b) + norm(b) * ln(1 - x))

                Computing norm'(b):
                Definition: d/dx Gamma(x) = Gamma(x) * PolyGamma_0(x)

                norm'(b) = d/db norm(b) = d/db Gamma(a+b-1)/(Gamma(a-1)*Gamma(b-1))
                 = 1/Gamma(a-1) * d/db[ Gamma(a+b-1)/Gamma(b-1) ]
                 = 1/Gamma(a-1) * (d/db[ Gamma(a+b-1) ] * Gamma(b-1) - Gamma(a+b-1) * d/db[ Gamma(b-1) ]) / Gamma(b-1)^2
                 = 1/Gamma(a-1) * (Gamma(a+b-1) * PolyGamma_0(a+b-1) * Gamma(b-1) - Gamma(a+b-1) * Gamma(b-1) * PolyGamma_0(b-1)) / Gamma(b-1)^2
                 = Gamma(a+b-1) * Gamma(b-1) / Gamma(a-1) * (PolyGamma_0(a+b-1) - PolyGamma_0(b-1)) / Gamma(b-1)^2
                 = Gamma(a+b-1)/(Gamma(a-1)*Gamma(b-1)) * (PolyGamma_0(a+b-1) - PolyGamma_0(b-1))
                 = norm(b) * (PolyGamma_0(a+b-1) - PolyGamma_0(b-1))

                d/db pdf(x | a, b) = x^(a-1) * (1 - x)^(b-1) * (norm'(b) + norm(b) * ln(1 - x))
                 = x^(a-1) * (1 - x)^(b-1) * (norm(b) * (PolyGamma_0(a+b-1) - PolyGamma_0(b-1)) + norm(b) * ln(1 - x))
                 = norm(b) * x^(a-1) * (1 - x)^(b-1) * (PolyGamma_0(a+b-1) - PolyGamma_0(b-1) + ln(1 - x))
                 = pdf(x | a, b) * (PolyGamma_0(a+b-1) - PolyGamma_0(b-1) + ln(1 - x))

                Confirmation:

                f(x) * d/dx ln(f(x)) = f'(x)
                d/db ln(pdf(x | a, b)) = Digamma(a+b-1) - Digamma(b-1) + ln(1 - x)

                norm(a) * x^(a-1) * (1 - x)^(b-1) * (Digamma(a+b-1) - Digamma(b-1) + ln(1 - x))

                Note that Digamma(x) = PolyGamma_0(x)

                norm(a) * x^(a-1) * (1 - x)^(b-1) * (PolyGamma_0(a+b-1) - PolyGamma_0(b-1) + ln(1 - x))
                pdf(x | a, b) * (PolyGamma_0(a+b-1) - PolyGamma_0(b-1) + ln(1 - x))

                Wich is identical to our result for the normal computation.

            */

            // = PolyGamma_0(a+b-1) - PolyGamma_0(b-1) + ln(1 - x)
            let term: f64 = digamma_ab - euclid::digamma(b - 1.0) + (1.0 - x).ln();

            ret.push(pdf * term);
        }

        return ret;
    }

    fn log_derivative_pdf_parameters(&self, x: f64, parameters: &[f64]) -> Vec<f64> {
        // d/dx ln(f(x)) = f'(x)/f(x)

        // pdf(x | a, b) = Gamma(a+b-1)/(Gamma(a-1)*Gamma(b-1)) * x^(a-1) * (1 - x)^(b-1)
        // ln(pdf(x | a, b)) = ln(Gamma(a+b-1)/(Gamma(a-1)*Gamma(b-1)) * x^(a-1) * (1 - x)^(b-1))
        // ln(pdf(x | a, b)) = ln(Gamma(a+b-1)) - ln(Gamma(a-1)) - ln(Gamma(b-1)) + ln(x^(a-1)) + ln((1 - x)^(b-1))
        // ln(pdf(x | a, b)) = ln(Gamma(a+b-1)) - ln(Gamma(a-1)) - ln(Gamma(b-1)) + (a-1)*ln(x) + (b-1) * ln(1 - x)

        // Note:
        // Definition: Digamma(x) = d/dx[ ln(Gamma(x)) ] = Gamma'(x) / Gamma(x)

        // Reserve a vector with exacly 3 elements
        let mut ret: Vec<f64> = Vec::new();
        ret.reserve_exact(3);

        let a: f64 = parameters[0];
        let b: f64 = parameters[1];
        let digamma_ab: f64 = euclid::digamma(a + b - 1.0);

        {
            // ## Log derivative respect to x:
            /*

                d/dx ln(pdf(x | a, b)) = d/dx ln(Gamma(a+b-1)) - ln(Gamma(a-1)) - ln(Gamma(b-1)) + (a-1)*ln(x) + (b-1) * ln(1 - x)
                 = 0 - 0 - 0 + (a-1)/x + (b-1)/(1 - x) * d/dx[ 1 - x ]
                 = (a-1)/x + (b-1)/(1 - x) * -1
                 = (a-1)/x - (b-1)/(1 - x)

            */

            let term_1: f64 = (parameters[0] - 1.0) / x;
            let term_2: f64 = (parameters[2] - 1.0) / (1.0 - x);

            ret.push(term_1 - term_2);
        }

        {
            // ## Log derivative respect to a:
            /*

                d/da ln(pdf(x | a, b)) = d/da ln(Gamma(a+b-1)) - ln(Gamma(a-1)) - ln(Gamma(b-1)) + (a-1)*ln(x) + (b-1) * ln(1 - x)
                 = Digamma(a+b-1) - Digamma(a-1) - 0 + ln(x) + 0
                 = Digamma(a+b-1) - Digamma(a-1) + ln(x)

            */

            ret.push(digamma_ab - euclid::digamma(a - 1.0) + x.ln());
        }

        {
            // ## Log derivative respect to beta:
            /*

                d/db ln(pdf(x | a, b)) = d/db ln(Gamma(a+b-1)) - ln(Gamma(a-1)) - ln(Gamma(b-1)) + (a-1)*ln(x) + (b-1) * ln(1 - x)
                 = Digamma(a+b-1) - 0 - Digamma(b-1) + 0 + ln(1 - x)
                 = Digamma(a+b-1) - Digamma(b-1) + ln(1 - x)

            */

            ret.push(digamma_ab - euclid::digamma(b - 1.0) + (1.0 - x).ln());
        }

        return ret;
    }

    fn parameter_restriction(&self, parameters: &mut [f64]) {
        let ep: f64 = f64::EPSILON;
        parameters[0] = parameters[0].max(ep * ep * ep);
        parameters[1] = parameters[1].max(ep * ep * ep);
    }

    fn fit(&self, data: &mut crate::samples::Samples) -> Vec<f64> {
        /*

            Using Maximum Likelyhood estimation:
            Assuming k samples.

                Estimation of alpha:

            pdf(x | a, b) = Gamma(a+b-1)/(Gamma(a-1)*Gamma(b-1)) * x^(a-1) * (1 - x)^(b-1)
            d/da ln(pdf(x | a, b) = Digamma(a+b-1) - Digamma(a-1) + ln(x)
            0 = sumatory{x_i} Digamma(a+b-1) - Digamma(a-1) + ln(x_i)
            0 = k * Digamma(a+b-1) + sumatory{x_i} - Digamma(a-1) + ln(x_i)
            0 = k * Digamma(a+b-1) - k * Digamma(a-1) + sumatory{x_i} ln(x_i)
            k * (Digamma(a-1) - Digamma(a+b-1)) = sumatory{x_i} ln(x_i)
            Digamma(a-1) - Digamma(a+b-1) = 1/k * sumatory{x_i} ln(x_i)

                Estimation of beta:

            pdf(x | a, b) = Gamma(a+b-1)/(Gamma(a-1)*Gamma(b-1)) * x^(a-1) * (1 - x)^(b-1)
            d/db ln(pdf(x | a, b) = Digamma(a+b-1) - Digamma(b-1) + ln(1 - x)
            0 = sumatory{x_i} Digamma(a+b-1) - Digamma(b-1) + ln(1 - x_i)
            0 = k * Digamma(a+b-1) + sumatory{x_i} - Digamma(b-1) + ln(1 - x_i)
            0 = k * Digamma(a+b-1) - k * Digamma(b-1) + sumatory{x_i} ln(1 - x_i)
            k * (Digamma(b-1) - Digamma(a+b-1)) = sumatory{x_i} ln(1 - x_i)
            Digamma(b-1) - Digamma(a+b-1) = 1/k * sumatory{x_i} ln(1 - x_i)

            We have the following 2 equations:

            Digamma(a-1) - Digamma(a+b-1) = 1/k * sumatory{x_i} ln(x_i)
            Digamma(b-1) - Digamma(a+b-1) = 1/k * sumatory{x_i} ln(1 - x_i)

            Substracting them:
            Digamma(a-1) - Digamma(a+b-1) - (Digamma(b-1) - Digamma(a+b-1)) = 1/k * sumatory{x_i}[ ln(x_i) ] - 1/k * sumatory{x_i}[ ln(1 - x_i) ]
            Digamma(a-1) - Digamma(b-1) = 1/k * (sumatory{x_i}[ ln(x_i) ] - sumatory{x_i}[ ln(1 - x_i) ])
            Digamma(a-1) - Digamma(b-1) = 1/k * sumatory{x_i}[ ln(x_i) - ln(1 - x_i) ]
            Digamma(a-1) - Digamma(b-1) = 1/k * sumatory{x_i}[ ln(x_i/(1 - x_i)) ]

            However we are stuck and we cannot progress further.
            If we knew one of them we could continue. There *may* exist
            multiple values for a and b that satisfy the equation
            (possibly infinite), but we cannot easly tell wich one
            of them is optimal.
            For this reason we will use the nuerical approach.

            We will do a few simplifications.

        */

        let d: usize = 2;
        let mut parameters: Vec<f64> = Vec::new();
        parameters.reserve_exact(2);
        parameters.push(1.0);
        parameters.push(1.0);
        // default values are a = 1, b = 1

        // SAFETY: should always be safe to only read
        let learning_rate: f64 =
            unsafe { crate::configuration::maximum_likelihood_estimation::LEARNING_RATE };
        // SAFETY: should always be safe to only read
        let conv_diff_criteria: f64 = unsafe {
            crate::configuration::maximum_likelihood_estimation::CONVERGENCE_DIFFERENCE_CRITERIA
        };
        // SAFETY: should always be safe to only read
        let max_iterations: u32 =
            unsafe { crate::configuration::maximum_likelihood_estimation::MAX_ITERATIONS };

        let inv_n: f64 = 1.0 / (data.peek_data().len() as f64);

        let mut gradient: Vec<f64> = std::vec![0.0; d];
        for _ in 0..max_iterations {
            // set gradient to 0
            gradient[0] = 0.0;
            gradient[1] = 0.0;

            for sample in data.peek_data() {
                let local_log_grad: Vec<f64> =
                    self.log_derivative_pdf_parameters(*sample, &parameters);

                // unrolled loop
                gradient[0] += local_log_grad[1];
                gradient[1] += local_log_grad[2];
            }

            let mut total_change: f64 = 0.0;
            // We want to maximize likelyhood, so we won't put the `-`
            // unrolled loop
            let change: f64 = learning_rate * inv_n * gradient[0];
            parameters[0] += change;
            total_change += change.abs();
            let change: f64 = learning_rate * inv_n * gradient[1];
            parameters[1] += change;
            total_change += change.abs();

            self.parameter_restriction(&mut parameters);

            if total_change < conv_diff_criteria {
                break;
            }
        }

        return parameters;
    }
}

impl Default for Beta {
    fn default() -> Self {
        // special case of https://en.wikipedia.org/wiki/Wigner_semicircle_distribution
        // a sample from ([Beta::default] * 2 - 1) * r follows the Wigner semicircle distribution
        let alpha: f64 = 1.5;
        let beta: f64 = 1.5;
        let norm_ct: f64 = 1.0 / euclid::beta_fn(alpha, beta);

        Self {
            alpha,
            beta,
            normalitzation_constant: norm_ct,
        }
    }
}
