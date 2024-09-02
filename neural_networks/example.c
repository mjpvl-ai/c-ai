#include <stdio.h>
#include <stdlib.h>
#include "dense_layer.h"
#include "neural_network_training.h"
#include "neural_networks.h"
#include "simple_neural_network.h"

int main() {
    // Define the network structure
    int layer_sizes[] = {2, 3, 1}; // 2 input neurons, 3 hidden neurons, 1 output neuron
    int num_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]);

    // Create a neural network
    NeuralNetwork *network = create_neural_network(num_layers, layer_sizes);

    // Toy dataset (XOR problem)
    double X[4][2] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    double y[4] = {0.0, 1.0, 1.0, 0.0}; // XOR output

    // Training parameters
    int num_samples = 4;
    int epochs = 1000;
    double learning_rate = 0.01;

    // Train the neural network
    train_neural_network(network, (const double *)X, y, num_samples, epochs, learning_rate);

    // Make predictions
    printf("Predictions:\n");
    for (int i = 0; i < num_samples; i++) {
        double output[1];
        forward_neural_network(network, X[i]);
        output[0] = network->layers[num_layers - 2]->output[0];
        printf("Input: (%.1f, %.1f) -> Output: %.2f\n", X[i][0], X[i][1], output[0]);
    }

    // Free resources
    free_neural_network(network);

    return 0;
}
