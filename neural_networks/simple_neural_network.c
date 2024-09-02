// simple_neural_network.c
#include "simple_neural_network.h"
#include <stdlib.h>

NeuralNetwork* create_neural_network(int num_layers, int *layer_sizes) {
    NeuralNetwork *network = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    network->num_layers = num_layers;
    network->layers = (DenseLayer **)malloc(num_layers * sizeof(DenseLayer *));

    for (int i = 0; i < num_layers - 1; i++) {
        network->layers[i] = create_dense_layer(layer_sizes[i], layer_sizes[i + 1]);
    }

    return network;
}

void forward_neural_network(NeuralNetwork *network, const double *input) {
    const double *current_input = input;
    for (int i = 0; i < network->num_layers - 1; i++) {
        forward_dense(network->layers[i], current_input);
        current_input = network->layers[i]->output;
    }
}

void free_neural_network(NeuralNetwork *network) {
    for (int i = 0; i < network->num_layers - 1; i++) {
        free_dense_layer(network->layers[i]);
    }
    free(network->layers);
    free(network);
}
