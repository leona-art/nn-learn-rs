pub(self) mod relu;
pub(self) mod sigmoid;
pub(self) mod step;
pub(self) mod tanh;
use relu::ReLU;
use sigmoid::Sigmoid;
use step::Step;
use tanh::Tanh;

pub(self) trait Activation {
    fn activate(x: f64) -> f64;
    fn derive(x: f64) -> f64;
}

pub enum ActivationType {
    Step,
    Sigmoid,
    ReLU,
    Tanh,
}
impl ActivationType {
    pub fn activate(&self, x: f64) -> f64 {
        match self {
            ActivationType::Step => Step::activate(x),
            ActivationType::Sigmoid => Sigmoid::activate(x),
            ActivationType::ReLU => ReLU::activate(x),
            ActivationType::Tanh => Tanh::activate(x),
        }
    }
    pub fn derive(&self, x: f64) -> f64 {
        match self {
            ActivationType::Step => Step::derive(x),
            ActivationType::Sigmoid => Sigmoid::derive(x),
            ActivationType::ReLU => ReLU::derive(x),
            ActivationType::Tanh => Tanh::derive(x),
        }
    }
}
