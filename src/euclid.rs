//! Euclid contains uscefull math functions

use rand::Rng;
use std::{num::NonZero, usize};

use crate::{
    configuration::integration::{
        DEFAULT_INTEGRATION_MAXIMUM_STEPS, DEFAULT_INTEGRATION_MAXIMUM_STEPS_F64,
        DEFAULT_INTEGRATION_MINIMUM_STEPS, DEFAULT_INTEGRATION_MINIMUM_STEPS_F64,
        DEFAULT_INTEGRATION_PRECISION, SMALL_INTEGRATION_NUM_STEPS, SMALL_INTEGRATION_PRECISION,
    },
    domain::{ContinuousDomain, DiscreteDomain},
};

/// The [moments](https://en.wikipedia.org/wiki/Moment_(mathematics)) of a function
/// are some values that provide information about the shape of the function.
/// If the function is a valid pdf, the mean, variance, skewness and other values
/// of interest can be expressed as moments.
///
/// The moments can have be of any of the 3 variants in this enum:
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum Moments {
    #[default]
    Raw,
    Central,
    Standarized,
}

/// Deafult return value if a domain is empty
pub const DEFAULT_EMPTY_DOMAIN_BOUNDS: (f64, f64) = (-0.0, 0.0);

/// Integrate a function it it's whole domain.
/// Can be used to determine the normalitzation constant of a pdf.
///
/// This function assumes that `pdf` contains a finite area in it's `domain`.
///
/// You need to divide the value given by `pdf` by the returned value in order to have
/// a valid probability distribution.
///
/// If you already created a distribution `d`, and want to make sure it integrates
/// to `1.0`, you can use:
/// ```
/// let c: f64 = euclid::numerical_integration(|x| d.pdf(x), d.get_domain());
/// assert!((c - 1.0).abs() < 0.0001)
/// ```
pub fn numerical_integration(pdf: impl Fn(f64) -> f64, domain: &ContinuousDomain) -> f64 {
    /*
           Plan:

       To compute integrals over an infinite range, we will perform a special
       [numerial integration](https://en.wikipedia.org/wiki/Numerical_integration#Integrals_over_infinite_intervals).
       (change of variable)

           For a (const) to infinite:
        integral {a -> inf} f(x) dx =
                    integral {0 -> 1} f(a + t/(1 - t))  /  (1 - t)^2  dt
        let u = 1/(1-t);
                    integral {0 -> 1} f(a + t * u) * u * u   dt

           For -infinite to a (const):
        integral {-inf -> a} f(x) dx =
                    integral {0 -> 1} f(a - (1 - t)/t)  /  t^2  dt
        let u = 1/t;
                    integral {0 -> 1} f(a - (1 - t) * u) * u * u  dt

           For -infinite to infinite:
        integral {-inf -> inf} f(x) dx =
                   integral {-1 -> 1} f( t / (1-t^2) ) * (1 + t^2) / (1 - t^2)^2  dt
        let u = 1/(1-t^2);
                   integral {-1 -> 1} f( t * u ) ) * (1 + t^2) * u * u  dt

    */
    let mut ret: f64 = -0.0;

    let bounds: (f64, f64) = domain.get_bounds();
    let integration_type: IntegrationType = IntegrationType::from_bounds(bounds);
    let (step_length, max_iters): (f64, usize) = choose_integration_precision_and_steps(bounds);
    let half_step_length: f64 = 0.5 * step_length;
    let step_len_over_6: f64 = step_length / 6.0;

    let mut num_step: f64 = 0.0;

    // estimate the bound value with the next 2 values
    let mut last_pdf_evaluation: f64 = match integration_type {
        IntegrationType::Finite | IntegrationType::ConstToInfinite => {
            let middle: f64 = pdf(bounds.0 + half_step_length);
            let end: f64 = pdf(bounds.0 + step_length);
            2.0 * middle - end
        }
        IntegrationType::InfiniteToConst => {
            let middle: f64 = pdf(bounds.1 - half_step_length);
            let end: f64 = pdf(bounds.1 - step_length);
            2.0 * middle - end
        }
        IntegrationType::FullInfinite => 0.0,
    };

    for _i in 0..max_iters {
        let current_position: f64;

        let (middle, end): (f64, f64) = match integration_type {
            IntegrationType::Finite => {
                current_position = bounds.0 + step_length * num_step;
                let _middle: f64 = pdf(current_position + half_step_length);
                let _end: f64 = pdf(current_position + step_length);
                (_middle, _end)
            }
            IntegrationType::ConstToInfinite => {
                // integral {a -> inf} f(x) dx = integral {0 -> 1} f(a + t/(1 - t))  /  (1 - t)^2  dt

                current_position = step_length * num_step;

                let _middle: f64 = 'mid: {
                    let t: f64 = current_position + half_step_length;
                    let e: f64 = 1.0 - t;
                    if e.abs() < f64::EPSILON {
                        break 'mid 0.0;
                        // todo: implement something better here
                    }
                    let u: f64 = 1.0 / e; // 1/(1-t)
                    pdf(bounds.0 + t * u) * u * u
                };
                let _end: f64 = 'end: {
                    let t: f64 = current_position + step_length;
                    let e: f64 = 1.0 - t;
                    if e.abs() < f64::EPSILON {
                        break 'end 0.0;
                        // todo: implement something better here
                    }
                    let u: f64 = 1.0 / e; // = 1/(1 - t)
                    pdf(bounds.0 + t * u) * u * u
                };
                (_middle, _end)
            }
            IntegrationType::InfiniteToConst => {
                // integral {-inf -> a} f(x) dx = integral {0 -> 1} f(a - (1 - t)/t)  /  t^2  dt
                // this integral is done in "reverse". Form 1 to 0.

                current_position = 1.0 - step_length * num_step;

                let _middle: f64 = 'mid: {
                    let t: f64 = current_position - half_step_length;
                    let u: f64 = 1.0 / t;
                    if u.is_infinite() {
                        break 'mid 0.0;
                        // todo: implement something better here
                    }
                    pdf(bounds.1 - (1.0 - t) * u) * u * u
                };
                let _end: f64 = 'end: {
                    let t: f64 = current_position - step_length;
                    let u: f64 = 1.0 / t;
                    if u.is_infinite() {
                        break 'end 0.0;
                        // todo: implement something better here
                    }
                    pdf(bounds.1 - (1.0 - t) * u) * u * u
                };
                (_middle, _end)
            }
            IntegrationType::FullInfinite => {
                // integral {-inf -> inf} f(x) dx = integral {-1 -> 1} f( t / (1-t^2) ) * (1 + t^2) / (1 - t^2)^2  dt

                current_position = -1.0 + step_length * num_step;

                let _middle: f64 = {
                    let t: f64 = current_position + half_step_length;
                    let u: f64 = 1.0 / (1.0 - t * t);
                    let v: f64 = 1.0 + t * t;
                    pdf(t * u) * v * u * u
                };
                let _end: f64 = {
                    let t: f64 = current_position + step_length;
                    let e: f64 = 1.0 - t * t;
                    if e.abs() < f64::EPSILON {
                        0.0
                    } else {
                        let u: f64 = 1.0 / e; // =1/(1-t^2)
                        let v: f64 = 1.0 + t * t;
                        pdf(t * u) * v * u * u
                    }
                };
                (_middle, _end)
            }
        };

        ret += step_len_over_6 * (last_pdf_evaluation + 4.0 * middle + end);

        last_pdf_evaluation = end;
        num_step += 1.0;
    }

    return ret;
}

/// Sum all the discrete values in a distribution.
/// Can be used to determine the total probability of a pmf.
///
/// If you want to have a pmf wich represents a valid probability distribution,
/// the result of this function shouls be one. If it is not, you can divise the
/// result of the pmf by the returned value in order to nomalize it.
///
/// This function assumes that `pmf` contains a finite area in it's `domain`.
///
/// **Warning:** if `max_steps` is set to None and the domain contains infinitely
/// many values, the function will not terminate (infinite loop).
pub fn discrete_integration(
    pmf: impl Fn(f64) -> f64,
    domain: &DiscreteDomain,
    max_steps: Option<usize>,
) -> f64 {
    let mut ret: f64 = 0.0;

    if let Some(max) = max_steps {
        let mut i: usize = 0;
        for point in domain.iter() {
            ret += pmf(point);
            i += 1;
            if max <= i {
                break;
            }
        }
    } else {
        for point in domain.iter() {
            ret += pmf(point);
        }
    }

    return ret;
}

/// Numerical integration but for a finite range.
///
/// Numerical integration for a function `func` within a finite range. The
/// integration is performed with
/// [Simpson's rule](https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_1/3_rule).
pub fn numerical_integration_finite(
    func: impl Fn(f64) -> f64,
    integration_range: (f64, f64),
    num_steps: u64,
) -> f64 {
    let mut ret: f64 = -0.0;

    let bounds: (f64, f64) = integration_range;
    let step_length: f64 = (bounds.1 - bounds.0) / num_steps as f64;
    let half_step_length: f64 = 0.5 * step_length;
    let step_len_over_6: f64 = step_length / 6.0;

    let mut num_step: f64 = 0.0;

    let mut last_pdf_evaluation: f64 = func(bounds.0 + f64::EPSILON);

    for _ in 0..num_steps {
        let current_position: f64 = bounds.0 + step_length * num_step;

        let middle: f64 = func(current_position + half_step_length);
        let end: f64 = func(current_position + step_length);

        ret += step_len_over_6 * (last_pdf_evaluation + 4.0 * middle + end);

        last_pdf_evaluation = end;
        num_step += 1.0;
    }

    return ret;
}

/// Randomly permute a slice.
///
/// This is an implementation of [Fisher–Yates shuffle](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle),
/// an efficient algorithm to randomly permute a list in `O(n)`.
pub fn random_permutation<T>(arr: &mut [T]) {
    // [Fisher–Yates shuffle](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle)

    let len: usize = arr.len();
    let mut rng: rand::prelude::ThreadRng = rand::thread_rng();

    for i in (1..=(len - 1)).rev() {
        let mut j: f64 = rng.gen();
        j = j * ((i + 1) as f64);
        let j: usize = j as usize;
        // k belongs to  [0, i - 1]

        arr.swap(i, j);
    }
}

/// Returns (step_length, num_steps) depending on the bounds of integration.
///
/// This function is internal to the library and wraps up everything
/// needed to choose the appropiate step_length (precision), and by consequence
/// the number of steps.
///
/// If the bounds are non-finite a change of variables for the integration is assumed.
///
/// Guarantees:
///  - `bounds.0 < bounds.1`
///  -
pub fn choose_integration_precision_and_steps(bounds: (f64, f64)) -> (f64, usize) {
    /*
        To select the appropiate step_length (and total_num_steps, indirecly),
        we need to adapt between the possible cases.
        - For standard integration: Use DEFAULT_INTEGRATION_PRECISION unless it's
            too small (if we would do more than DEFAULT_INTEGRATION_MAXIMUM_STEPS)


        num_steps needs to be odd

        // SAFETY: all the accesses to static variables shoulb be safe because
        the values at the config file should not be changed during this function call.
    */

    match (bounds.0.is_finite(), bounds.1.is_finite()) {
        (true, true) => {
            let range: f64 = bounds.1 - bounds.0;
            let step_length: f64;
            let num_steps: usize;

            if range <= 1.0 {
                // small range (less than an unit)
                num_steps = unsafe { SMALL_INTEGRATION_NUM_STEPS };
                // ^already odd
                step_length = range / num_steps as f64;
            } else if unsafe {
                DEFAULT_INTEGRATION_PRECISION * DEFAULT_INTEGRATION_MAXIMUM_STEPS_F64
            } < range
            {
                // interval is very big, we will increase the step_lenght.

                step_length = range / unsafe { DEFAULT_INTEGRATION_MAXIMUM_STEPS_F64 };
                num_steps = unsafe { DEFAULT_INTEGRATION_MAXIMUM_STEPS };
                // ^already odd
            } else {
                // *normal* interval

                let first_num: f64 = range / unsafe { DEFAULT_INTEGRATION_PRECISION };
                if first_num < unsafe { DEFAULT_INTEGRATION_MINIMUM_STEPS_F64 } {
                    // too litle steps
                    num_steps = unsafe { DEFAULT_INTEGRATION_MINIMUM_STEPS };
                } else {
                    num_steps = (first_num as usize) | 1;
                }
                step_length = range / (num_steps as f64);
            }

            (step_length, num_steps)
        }
        (false, false) => {
            // if the interval [0, 1] uses SMALL_INTEGRATION_NUM_STEPS,
            // then [-1, 1] will use the doule.
            unsafe {
                (
                    SMALL_INTEGRATION_PRECISION,
                    (SMALL_INTEGRATION_NUM_STEPS * 2) as usize,
                )
            }
        }
        _ => {
            /* Cases:
                (true, false) => {},
                (false, true) => {},
            */

            // The range in this case is [0, 1], so we can return the same values.
            unsafe { (SMALL_INTEGRATION_PRECISION, SMALL_INTEGRATION_NUM_STEPS) }
        }
    }
}

/// Indicates how big is the range to integrate.
///
/// Mainly ised for readability
pub enum IntegrationType {
    /// closed interval: `[a, b]`
    Finite,
    /// interval: `(-inf, a]`
    InfiniteToConst,
    /// interval: `[b, inf)`
    ConstToInfinite,
    /// interval: `(-inf, inf)`
    FullInfinite,
}

impl IntegrationType {
    pub fn from_bounds(bounds: (f64, f64)) -> IntegrationType {
        match (bounds.0.is_finite(), bounds.1.is_finite()) {
            (true, true) => IntegrationType::Finite,
            (true, false) => IntegrationType::ConstToInfinite,
            (false, true) => IntegrationType::InfiniteToConst,
            (false, false) => IntegrationType::FullInfinite,
        }
    }
}

pub mod combinatorics {

    /// Compute the [binomial coeffitient](https://en.wikipedia.org/wiki/Binomial_coefficient).
    ///
    /// Returns an error if there was a problem with the computation (overflow),
    /// if `n` or `k` are negative or if `n < k`.
    pub fn binomial_coeffitient(n: u64, mut k: u64) -> Result<u128, ()> {
        // todo: https://math.stackexchange.com/questions/202554/how-do-i-compute-binomial-coefficients-efficiently

        /*
               Plan:

           I have 2 methods.

           let (n|k) = `n choose k`;

           Method recursive:
           *****************************************************************************
           We will use the recursive formula:

           let (n|k) = `n choose k`;

            - ( n | k ) = (n - 1 | k - 1) + (n - 1 | k )
            - ( n | 0 ) = 1
                - (for all `0 <= n`)
            - ( n | n ) = 1

           With the dymanic programming principles we can create a table with the
           results like:

           | k \ n 	| 0 	| 1 	| 2 	| 3 	| 4 	| 5 	| 6 	| 7 	|
           |-------	|---	|---	|---	|---	|---	|---	|---	|---	|
           | 0     	| 1 	| 1 	| 1 	| 1 	| ^ 	| ^ 	| ^ 	| _ 	|
           | 1     	| u 	| 1 	| 2 	| 3 	| 4 	| ^ 	| ^ 	| _ 	|
           | 2     	| u 	| u 	| 1 	| _ 	| _ 	| a 	| ^ 	| _ 	|
           | 3     	| u 	| u 	| u 	| 1 	| _ 	| b 	| x 	| _ 	|
           | 4     	| u 	| u 	| u 	| u 	| 1 	| _ 	| _ 	| _ 	|
           | 5     	| u 	| u 	| u 	| u 	| u 	| 1 	| _ 	| _ 	|

           If we ant to find x, we just need to add a and b. values marked by  _
           represent uncomputed values and u represnt the values where the binomial
           coefitient is undefined. Note that we kust need a matrix of size k x n
           (even if the example is bigger). The values marked by u or ^ do not need
           to be computed.

           I have just noticed that this is just making the pascal triangle.
           *****************************************************************************

           Method optimized iterative approach (MOIA (?)):

           (n | k) = productory {i = 1 -> k} (n - k + i) / i




        */

        // Since C(n, k) = C(n, n-k)
        if n - k < k {
            k = n - k;
        }

        let N: u128 = n as u128;
        let K: u128 = k as u128;

        let mut ret: u128 = 1;
        for i in 1..=K {
            match ret.checked_mul(N - K + i) {
                Some(v) => ret = v,
                None => return Err(()),
            }
            ret = ret / i;
        }

        return Ok(ret);
    }
}

/// Evaluetes the natural logarithm of gamma of `x` when `x` is an integer.
///
/// O(input)
pub fn ln_gamma_int(input: NonZero<u64>) -> f64 {
    // full continuous case implementation: https://www.netlib.org/fdlibm/e_lgamma_r.c

    /*
      ln(gamma(x)) = ln(gamma(x + 1)) - ln(x)
      ln(gamma(x + 1)) = ln(gamma(x)) + ln(x)

      Gamma(x) = (x-1)!
      Gamma(x) = productory{i: 1 -> x-1} i
      ln(Gamma(x)) = sumatory{i: 1 -> x-1} ln(i)

      Therefore:
      ln(gamma(x + 1)) = ln(gamma(x - 1)) + ln(x - 1) + ln(x)
      ln(gamma(x + 1)) = ln(gamma(x - 2)) + ln(x - 2) + ln(x - 1) + ln(x)
      [...]
      ln(gamma(x + 1)) = sumatory{i: 2 -> x} ln(x)
      ln(gamma(x)) = sumatory{i: 2 -> x-1} ln(x)

      ***
      To reduce the computational cost we will precompute the some awnsers in
      order to have some boost in speed (particularly for larger inputs. )

      Note: `ln(f64::MAX) = 709.7827`, therefore if the result is greater than
      this number, an overflow will ocurr if exponentiation is attempted.

      ***

      # Better way to do this:

      x! = x*(x-1)*(x-2)*(x-1)*[...]*4*3*2*1

      This basic formula essentially gives us someting **REALLY** close to the
      factoritzation of x!. Therefore, if we can make a function `prime_factors(k)`
      that is very fast **up to x**, then we can do:

      x! = 2^a * 3^b * 5^c * 7^d * 11^e * 13^f [...]
      ln(x!) = ln(2^a * 3^b * 5^c * 7^d * 11^e * 13^f [...])
      ln(x!) = ln(2^a) + ln(3^b) + ln(5^c) + ln(7^d) + ln(11^e) + ln(13^f) + [...]
      ln(x!) = a*ln(2) + b*ln(3) + c*ln(5) + d*ln(7) + e*ln(11) + f*ln(13) + [...]

      Wich should be very effitient assuming we have precompute the ln() of the furst prime
      numbers. This could be implemented (aprox.) as:

      ```
      //list_primes = [2, 3, 5, 7, 11, 13, ...]
      list_ln_primes = [ln(2), ln(3), ln(5), ln(7), ln(11), ln(13), ...]

      let prime_factor_decomp = (2..=x)
          .iter().
          .map(|k| prime_factors(k))
          .reduce(|fact, acc| acc + fact);
      // acc + fact is done element-wise onto the largest vector

      let mut ret = 0.0;
      for (ln_prime, pow) in list_ln_primes.iter().zip(prime_factor_decomp.iter()) {
          ret += pow * ln_prime;
      }

      return ret;
      ```
    */

    let x: u64 = input.get();

    // numberical values obtained by [WorframAlpha](https://www.wolframalpha.com/input?i=lngamma%283%29)
    let (mut accumulator, mut i): (f64, f64) = match x {
        0 => unreachable!(),
        1 | 2 => return 0.0,
        3 => return 0.69314718055994530941723212145817656807550,
        4 => return 1.79175946922805500081247735838070227272299,
        5 => return 3.17805383034794561964694160129705540887399,
        6 => return 4.78749174278204599424770093452324304839959,
        7 => return 6.57925121201010099506017829290394532112258,
        8 => return 8.52516136106541430016553103634712505075966,
        9 => return 10.6046029027452502284172274007216547549861,
        10 => return 12.8018274800814696112077178745667061642811,
        11 => return 15.1044125730755152952257093292510703718822,
        12 => return 17.5023078458738858392876529072161996717039,
        13 => return 19.9872144956618861495173623870550785125024,
        14 => return 22.5521638531234228855708498286203971173077,
        15 => return 25.1912211827386815000934346935217534150203,
        16 => return 27.8992713838408915660894392636704667591933,
        17..=31 => (27.8992713838408915660894392636704667591933, 17.0),
        32..=63 => (78.092223553315310631416808058720323846721783, 32.0),
        64..=95 => (201.0093163992815266792820391565502964125081888, 64.0),
        96..=127 => (340.8150588707990178689655113342148226173214543, 96.0),
        128..=191 => (491.553448223298003498872193835691609891142996, 128.0),
        192..=255 => (815.7297363039101614174116323592750280049309917, 192.0),
        _ => (1161.712101118400650788039632401011094238739485, 256.0),
    };

    let x: f64 = x as f64;

    while i < x {
        accumulator += i.ln();
        i += 1.0;
    }

    return accumulator;
}
