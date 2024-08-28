use super::Activation;

pub(super) struct Tanh {}
impl Activation for Tanh {
    fn activate(x: f64) -> f64 {
        x.tanh()
    }
    fn derive(x: f64) -> f64 {
        1.0 - Self::activate(x).powi(2)
    }
}
