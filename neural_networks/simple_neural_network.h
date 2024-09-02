// simple_neural_network.h
#ifndef SIMPLE_NEURAL_NETWORK_H
#define SIMPLE_NEURAL_NETWORK_H

#include "dense_layer.h"

typedef struct {
    DenseLayer **layers;
    int num_layers;
} NeuralNetwork;

NeuralNetwork* create_neural_network(int num_layers, int *layer_sizes);
void forward_neural_network(NeuralNetwork *network, const double *input);
void free_neural_network(NeuralNetwork *network);

#endif // SIMPLE_NEURAL_NETWORK_H
