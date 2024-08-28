use super::Activation;
use std::f64::consts::E;

pub(super) struct Sigmoid {}

impl Activation for Sigmoid {
    fn activate(x: f64) -> f64 {
        1.0 / (1.0 + E.powf(-x))
    }
    fn derive(x: f64) -> f64 {
        Self::activate(x) * (1.0 - Self::activate(x))
    }
}
