use super::Activation;

pub(super) struct Step {}
impl Activation for Step {
    fn activate(x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
    fn derive(x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}