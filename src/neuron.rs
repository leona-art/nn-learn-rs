use super::activation::ActivationType;

#[derive(Debug)]
pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl Neuron {
    pub fn new(weights: &[f64], bias: f64) -> Neuron {
        Neuron {
            weights: weights.to_vec(),
            bias,
        }
    }

    pub fn feedforward(&self, inputs: &[f64], activation: &ActivationType) -> f64 {
        if self.weights.len() != inputs.len() {
            panic!("The number of inputs must match the number of weights");
        }
        let sum = self
            .weights
            .iter()
            .zip(inputs.iter())
            .map(|(w, i)| w * i)
            .sum::<f64>()
            + self.bias;
        activation.activate(sum)
    }

    pub fn backward(
        &mut self,
        inputs: &[f64],
        activation: &ActivationType,
        output_gradient: f64,
        learning_rate: f64,
    ) -> Vec<f64> {
        let z = self
            .weights
            .iter()
            .zip(inputs.iter())
            .map(|(w, &x)| w * x)
            .sum::<f64>()
            + self.bias;
        let activation_derivative = activation.derive(z);
        let delta = output_gradient * activation_derivative;

        let mut input_gradient = vec![0.0; inputs.len()];
        for (i, (&input, weight)) in inputs.iter().zip(self.weights.iter_mut()).enumerate() {
            input_gradient[i] = *weight * delta;
            *weight -= learning_rate * delta * input;
        }
        self.bias -= learning_rate * delta;
        input_gradient
    }
}
