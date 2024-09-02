#include "neural_networks.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Helper functions
static float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

static float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1 - s);
}

static void matrix_multiply(float *a, float *b, float *result, int a_rows, int a_cols, int b_cols) {
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_cols; j++) {
            result[i * b_cols + j] = 0;
            for (int k = 0; k < a_cols; k++) {
                result[i * b_cols + j] += a[i * a_cols + k] * b[k * b_cols + j];
            }
        }
    }
}

static void matrix_add(float *a, float *b, float *result, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        result[i] = a[i] + b[i];
    }
}

// Perceptron Implementation
void perceptron_init(Perceptron *perceptron, int num_inputs, float learning_rate) {
    perceptron->num_inputs = num_inputs;
    perceptron->learning_rate = learning_rate;
    perceptron->weights = (float *)malloc(num_inputs * sizeof(float));
    perceptron->bias = 0.0;
    
    for (int i = 0; i < num_inputs; i++) {
        perceptron->weights[i] = (float)rand() / RAND_MAX;
    }
}

void perceptron_train(Perceptron *perceptron, float *inputs, int target, int num_samples) {
    for (int i = 0; i < num_samples; i++) {
        float weighted_sum = perceptron->bias;
        for (int j = 0; j < perceptron->num_inputs; j++) {
            weighted_sum += perceptron->weights[j] * inputs[j];
        }
        int prediction = sigmoid(weighted_sum) >= 0.5 ? 1 : 0;
        int error = target - prediction;
        
        perceptron->bias += perceptron->learning_rate * error;
        for (int j = 0; j < perceptron->num_inputs; j++) {
            perceptron->weights[j] += perceptron->learning_rate * error * inputs[j];
        }
    }
}

int perceptron_predict(Perceptron *perceptron, float *inputs) {
    float weighted_sum = perceptron->bias;
    for (int i = 0; i < perceptron->num_inputs; i++) {
        weighted_sum += perceptron->weights[i] * inputs[i];
    }
    return sigmoid(weighted_sum) >= 0.5 ? 1 : 0;
}

// Feedforward Neural Network Implementation
void feedforward_nn_init(FeedforwardNN *nn, int num_layers, int *layer_sizes, float learning_rate) {
    nn->num_layers = num_layers;
    nn->layer_sizes = (int *)malloc(num_layers * sizeof(int));
    nn->weights = (float **)malloc((num_layers - 1) * sizeof(float *));
    nn->biases = (float **)malloc((num_layers - 1) * sizeof(float *));
    nn->activations = (float **)malloc(num_layers * sizeof(float *));
    nn->z_values = (float **)malloc((num_layers - 1) * sizeof(float *));
    nn->learning_rate = learning_rate;
    
    for (int i = 0; i < num_layers; i++) {
        nn->layer_sizes[i] = layer_sizes[i];
        nn->activations[i] = (float *)malloc(layer_sizes[i] * sizeof(float));
        if (i < num_layers - 1) {
            nn->weights[i] = (float *)malloc(layer_sizes[i] * layer_sizes[i + 1] * sizeof(float));
            nn->biases[i] = (float *)malloc(layer_sizes[i + 1] * sizeof(float));
            nn->z_values[i] = (float *)malloc(layer_sizes[i + 1] * sizeof(float));
            for (int j = 0; j < layer_sizes[i] * layer_sizes[i + 1]; j++) {
                nn->weights[i][j] = (float)rand() / RAND_MAX;
            }
            for (int j = 0; j < layer_sizes[i + 1]; j++) {
                nn->biases[i][j] = (float)rand() / RAND_MAX;
            }
        }
    }
}

void feedforward_nn_forward(FeedforwardNN *nn, float *inputs) {
    memcpy(nn->activations[0], inputs, nn->layer_sizes[0] * sizeof(float));
    
    for (int i = 0; i < nn->num_layers - 1; i++) {
        matrix_multiply(nn->activations[i], nn->weights[i], nn->z_values[i], nn->layer_sizes[i], nn->layer_sizes[i + 1], 1);
        matrix_add(nn->z_values[i], nn->biases[i], nn->activations[i + 1], 1, nn->layer_sizes[i + 1]);
        for (int j = 0; j < nn->layer_sizes[i + 1]; j++) {
            nn->activations[i + 1][j] = sigmoid(nn->activations[i + 1][j]);
        }
    }
}

void feedforward_nn_backpropagate(FeedforwardNN *nn, float *inputs, float *targets, float *outputs) {
    float *deltas = (float *)malloc(nn->layer_sizes[nn->num_layers - 1] * sizeof(float));
    float *errors = (float *)malloc(nn->layer_sizes[nn->num_layers - 1] * sizeof(float));
    
    for (int i = 0; i < nn->layer_sizes[nn->num_layers - 1]; i++) {
        errors[i] = targets[i] - nn->activations[nn->num_layers - 1][i];
        deltas[i] = errors[i] * sigmoid_derivative(nn->activations[nn->num_layers - 1][i]);
    }
    
    for (int i = nn->num_layers - 2; i >= 0; i--) {
        float *next_deltas = (float *)malloc(nn->layer_sizes[i] * sizeof(float));
        for (int j = 0; j < nn->layer_sizes[i]; j++) {
            next_deltas[j] = 0;
            for (int k = 0; k < nn->layer_sizes[i + 1]; k++) {
                next_deltas[j] += deltas[k] * nn->weights[i][j * nn->layer_sizes[i + 1] + k];
                nn->weights[i][j * nn->layer_sizes[i + 1] + k] += nn->learning_rate * deltas[k] * nn->activations[i][j];
            }
            nn->biases[i][j] += nn->learning_rate * deltas[j];
        }
        free(deltas);
        deltas = next_deltas;
    }
    
    free(deltas);
    free(errors);
}

void feedforward_nn_train(FeedforwardNN *nn, float *inputs, float *targets, int num_samples, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < num_samples; i++) {
            feedforward_nn_forward(nn, inputs);
            feedforward_nn_backpropagate(nn, inputs, targets, nn->activations[nn->num_layers - 1]);
        }
    }
}

void feedforward_nn_predict(FeedforwardNN *nn, float *inputs, float *outputs) {
    feedforward_nn_forward(nn, inputs);
    memcpy(outputs, nn->activations[nn->num_layers - 1], nn->layer_sizes[nn->num_layers - 1] * sizeof(float));
}

void feedforward_nn_free(FeedforwardNN *nn) {
    for (int i = 0; i < nn->num_layers; i++) {
        free(nn->activations[i]);
        if (i < nn->num_layers - 1) {
            free(nn->weights[i]);
            free(nn->biases[i]);
            free(nn->z_values[i]);
        }
    }
    free(nn->layer_sizes);
    free(nn->weights);
    free(nn->biases);
    free(nn->activations);
    free(nn->z_values);
}
