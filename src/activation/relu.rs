use super::Activation;

pub(super) struct ReLU {}

impl Activation for ReLU {
    fn activate(x: f64) -> f64 {
        x.max(0.0)
    }

    fn derive(x: f64) -> f64 {
        match x > 0.0 {
            true => 1.0,
            false => 0.0,
        }
    }
}