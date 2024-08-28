use activation::ActivationType;
use network::Network;
pub mod activation;
pub mod layer;
pub mod network;
pub mod neuron;

fn main() {
    let mut nw = Network::new(2);
    nw.add_layer(2, ActivationType::Sigmoid)
        .add_layer(2, ActivationType::Sigmoid)
        .add_layer(1, ActivationType::Sigmoid);
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
    nw.train(&inputs, &targets, 100, 0.1);
}
