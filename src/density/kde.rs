use core::f64::consts::{PI, SQRT_2};

/// The implemented one dimensional [kernel functions][source]
///
/// source: https://en.wikipedia.org/wiki/Kernel_(statistics)
pub enum Kernel1d {
    Epanechnikov,
    Gaussian { sigma: f64 },
    Uniform,
    Triangular,
    Biweigth,
    Triweight,
    Tricube,
    Cosine,
    Logistic,
    Sigmoid,
    Silverman,
}

impl Kernel1d {
    pub fn evaluate(&self, u: f64) -> f64 {
        match self {
            Self::Epanechnikov => {
                if u.abs() >= 1. {
                    0.0
                } else {
                    0.75 * (1. - u.powi(2))
                }
            }
            Self::Gaussian { sigma } => {
                (-0.5 * (u / sigma).powi(2)).exp() / (crate::consts::SQRT_2PI * sigma)
            }
            Self::Uniform => {
                if u.abs() > 1. {
                    0.0
                } else {
                    0.5
                }
            }
            Self::Triangular => {
                let abs_u = u.abs();
                if abs_u >= 1. {
                    0.0
                } else {
                    1. - abs_u
                }
            }
            Self::Biweigth => {
                if u.abs() >= 1. {
                    0.0
                } else {
                    (15. / 16.) * (1. - u.powi(2)).powi(2)
                }
            }
            Self::Triweight => {
                if u.abs() >= 1. {
                    0.0
                } else {
                    (35. / 32.) * (1. - u.powi(2)).powi(3)
                }
            }
            Self::Tricube => {
                let abs_u = u.abs();
                if abs_u >= 1. {
                    0.0
                } else {
                    (70. / 81.) * (1. - abs_u.powi(3)).powi(3)
                }
            }
            Self::Cosine => {
                if u.abs() >= 1. {
                    0.0
                } else {
                    (PI / 4.) * ((PI / 2.) * u).cos()
                }
            }
            Self::Logistic => 0.5 / (1. + u.cosh()),
            Self::Sigmoid => 1. / (PI * u.cosh()),
            Self::Silverman => {
                let abs_u_over_sqrt2 = u.abs() / SQRT_2;
                0.5 * (-abs_u_over_sqrt2).exp() * (PI / 4. + abs_u_over_sqrt2).sin()
            }
        }
    }
}
