#ifndef NEURAL_NETWORKS_H
#define NEURAL_NETWORKS_H

#include <stdbool.h>

// Perceptron
typedef struct {
    int num_inputs;
    float *weights;
    float bias;
    float learning_rate;
} Perceptron;

void perceptron_init(Perceptron *perceptron, int num_inputs, float learning_rate);
void perceptron_train(Perceptron *perceptron, float *inputs, int target, int num_samples);
int perceptron_predict(Perceptron *perceptron, float *inputs);

// Feedforward Neural Network
typedef struct {
    int num_layers;
    int *layer_sizes;
    float **weights;
    float **biases;
    float **activations;
    float **z_values;
    float learning_rate;
} FeedforwardNN;

void feedforward_nn_init(FeedforwardNN *nn, int num_layers, int *layer_sizes, float learning_rate);
void feedforward_nn_forward(FeedforwardNN *nn, float *inputs);
void feedforward_nn_backpropagate(FeedforwardNN *nn, float *inputs, float *targets, float *outputs);
void feedforward_nn_train(FeedforwardNN *nn, float *inputs, float *targets, int num_samples, int epochs);
void feedforward_nn_predict(FeedforwardNN *nn, float *inputs, float *outputs);
void feedforward_nn_free(FeedforwardNN *nn);

// Convolutional Neural Network
typedef struct {
    int num_layers;
    int *layer_sizes;
    float **filters;
    float **biases;
    int input_height, input_width, num_channels;
    int filter_height, filter_width;
} CNN;

void cnn_init(CNN *cnn, int num_layers, int *layer_sizes, int input_height, int input_width, int num_channels);
void cnn_forward(CNN *cnn, float *input, float *output);
void cnn_backpropagate(CNN *cnn, float *input, float *target, float *output);
void cnn_train(CNN *cnn, float *input, float *target, int epochs);
void cnn_predict(CNN *cnn, float *input, float *output);
void cnn_free(CNN *cnn);

#endif // NEURAL_NETWORKS_H
