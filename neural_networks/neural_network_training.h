// neural_network_training.h
#ifndef NEURAL_NETWORK_TRAINING_H
#define NEURAL_NETWORK_TRAINING_H

#include "simple_neural_network.h"

void train_neural_network(NeuralNetwork *network, const double *X, const double *y, int num_samples, int epochs, double learning_rate);

#endif // NEURAL_NETWORK_TRAINING_H
