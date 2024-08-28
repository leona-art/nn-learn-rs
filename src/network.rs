use crate::{activation::ActivationType, layer::Layer};

pub struct Network {
    layers: Vec<Layer>,
    input_size: usize,
    output_size: usize,
}

impl Network {
    pub fn new(input_size: usize) -> Self {
        Network {
            layers: Vec::new(),
            input_size,
            output_size: 0,
        }
    }
    pub fn add_layer(&mut self, output_size: usize, activation: ActivationType) -> &mut Self {
        let input_size = match self.layers.last() {
            Some(layer) => layer.neurons.len(),
            None => self.input_size,
        };
        self.layers
            .push(Layer::new(input_size, output_size, activation));
        self.output_size = output_size;
        self
    }
    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        let output = self
            .layers
            .iter()
            .fold(inputs.to_vec(), |outputs, layer| layer.forward(&outputs));
        output
    }

    pub fn backward(
        &mut self,
        input: &[f64],
        target: &[f64],
        learning_rate: f64,
    ) -> f64 {
        let mut forward_outputs = vec![input.to_vec()];

        // Forward pass
        for layer in &self.layers {
            let output = layer.forward(forward_outputs.last().unwrap());
            forward_outputs.push(output);
        }

        // Compute initial gradient
        let mut gradient = Self::compute_output_gradient(forward_outputs.last().unwrap(), target);
        let mut loss = Self::compute_loss(forward_outputs.last().unwrap(), target);

        // Backward pass
        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            let input = &forward_outputs[i];
            gradient = layer.backward(input, &gradient, learning_rate);
        }

        loss
    }

    fn compute_output_gradient(output: &[f64], target: &[f64]) -> Vec<f64> {
        // 例：二乗誤差の勾配
        output
            .iter()
            .zip(target)
            .map(|(&o, &t)| 2.0 * (o - t))
            .collect()
    }

    fn compute_loss(output: &[f64], target: &[f64]) -> f64 {
        // 例：二乗誤差
        output
            .iter()
            .zip(target)
            .map(|(&o, &t)| (o - t).powi(2))
            .sum()
    }

    pub fn train(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>], epochs: usize, learning_rate: f64) {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            for (input, target) in inputs.iter().zip(targets) {
                let loss = self.backward(input, target, learning_rate);
                total_loss += loss;
            }
            println!("Epoch {}: Average loss = {}", epoch, total_loss / inputs.len() as f64);
        }
    }
}
