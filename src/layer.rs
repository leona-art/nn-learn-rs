use super::activation::ActivationType;
use super::neuron::Neuron;
use rand::prelude::*;

pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub activation: ActivationType,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize, activation: ActivationType) -> Layer {
        let mut neurons = Vec::with_capacity(output_size);
        let mut rng = thread_rng();
        for _ in 0..output_size {
            // ランダムな値で初期化
            let weights: Vec<f64> = (0..input_size).map(|_| rng.gen_range(-0.1..0.1)).collect();
            let bias = rng.gen_range(-0.1..0.1);
            neurons.push(Neuron::new(&weights, bias));
        }
        Layer {
            neurons,
            activation,
        }
    }

    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        if inputs.len() != self.neurons[0].weights.len() {
            panic!("The number of inputs must match the number of weights");
        }
        self.neurons
            .iter()
            .map(|neuron| neuron.feedforward(inputs, &self.activation))
            .collect()
    }

    pub fn backward(
        &mut self,
        input: &[f64],
        output_gradient: &[f64],
        learning_rate: f64,
    ) -> Vec<f64> {
        let mut input_gradient = vec![0.0; input.len()];
        for (neuron, &output_grad) in self.neurons.iter_mut().zip(output_gradient) {
            let neuron_gradient =
                neuron.backward(input, &self.activation, output_grad, learning_rate);
            for (i, &g) in neuron_gradient.iter().enumerate() {
                input_gradient[i] += g;
            }
        }
        input_gradient
    }
}
