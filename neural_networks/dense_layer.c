// dense_layer.c
#include "dense_layer.h"
#include <stdlib.h>

DenseLayer* create_dense_layer(int input_size, int output_size) {
    DenseLayer *layer = (DenseLayer *)malloc(sizeof(DenseLayer));
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->weights = (double *)malloc(input_size * output_size * sizeof(double));
    layer->biases = (double *)malloc(output_size * sizeof(double));
    layer->output = (double *)malloc(output_size * sizeof(double));

    // Initialize weights and biases (example: small random values)
    for (int i = 0; i < input_size * output_size; i++) {
        layer->weights[i] = ((double)rand() / RAND_MAX) * 0.01;
    }
    for (int i = 0; i < output_size; i++) {
        layer->biases[i] = 0.0;
    }

    return layer;
}

void forward_dense(DenseLayer *layer, const double *input) {
    for (int i = 0; i < layer->output_size; i++) {
        layer->output[i] = layer->biases[i];
        for (int j = 0; j < layer->input_size; j++) {
            layer->output[i] += input[j] * layer->weights[i * layer->input_size + j];
        }
    }
}

void free_dense_layer(DenseLayer *layer) {
    free(layer->weights);
    free(layer->biases);
    free(layer->output);
    free(layer);
}
