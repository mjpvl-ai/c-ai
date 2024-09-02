#include "neural_network_training.h"
#include "../Basic_Linear_Algebra_Operations/loss_functions.h"
#include "../optimization/gradient_descent.h"
#include"../neural_networks/simple_neural_network.h"
#include <stdlib.h>

void train_neural_network(NeuralNetwork *network, const double *X, const double *y, int num_samples, int epochs, double learning_rate) {
    int output_size = network->layers[network->num_layers - 2]->output_size;
    double *output = (double *)malloc(output_size * sizeof(double));
    double *error = (double *)malloc(output_size * sizeof(double));

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < num_samples; i++) {
            // Forward pass
            forward_neural_network(network, &X[i * network->layers[0]->input_size]);

            // Calculate the output error (derivative of loss with respect to output)
            for (int j = 0; j < output_size; j++) {
                output[j] = network->layers[network->num_layers - 2]->output[j];
                error[j] = output[j] - y[i * output_size + j]; // dL/dOutput
            }

            // Backpropagation
            for (int l = network->num_layers - 2; l >= 0; l--) {
                DenseLayer *layer = network->layers[l];
                int input_size = layer->input_size;
                int output_size = layer->output_size;

                // Gradient for biases
                for (int j = 0; j < output_size; j++) {
                    layer->biases[j] -= learning_rate * error[j];
                }

                // Gradient for weights
                for (int j = 0; j < output_size; j++) {
                    for (int k = 0; k < input_size; k++) {
                        layer->weights[j * input_size + k] -= learning_rate * error[j] * (l == 0 ? X[i * input_size + k] : network->layers[l - 1]->output[k]);
                    }
                }

                // If not the first layer, propagate the error backward
                if (l > 0) {
                    double *next_error = (double *)calloc(input_size, sizeof(double));
                    for (int j = 0; j < output_size; j++) {
                        for (int k = 0; k < input_size; k++) {
                            next_error[k] += error[j] * layer->weights[j * input_size + k];
                        }
                    }
                    free(error);
                    error = next_error;
                }
            }
        }
    }

    free(output);
    free(error);
}
