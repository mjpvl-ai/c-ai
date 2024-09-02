# C AI Library

This C AI library is designed to provide a comprehensive set of tools for building and training neural networks. It covers both basic and advanced techniques, allowing for the creation of sophisticated AI models. The library is modular, meaning each feature is implemented in its own module, which can be easily imported and used in various C projects.

## Table of Contents
1. [Basic Neural Network Components](#basic-neural-network-components)
   - [Dense Layer](#dense-layer)
   - [Neural Network](#neural-network)
2. [Training Algorithms](#training-algorithms)
   - [Gradient Descent](#gradient-descent)
   - [Backpropagation](#backpropagation)
3. [Advanced Techniques](#advanced-techniques)
   - [Momentum](#momentum)
   - [Regularization](#regularization)
   - [Learning Rate Schedules](#learning-rate-schedules)
   - [Adam Optimizer](#adam-optimizer)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)

## Basic Neural Network Components

### Dense Layer

**Module:** `dense_layer`

A dense (fully connected) layer is the fundamental building block of a neural network. Each neuron in a dense layer receives input from all the neurons in the previous layer.

- **Functions:**
  - `DenseLayer* create_dense_layer(int input_size, int output_size);`: Creates a dense layer with specified input and output sizes.
  - `void forward_dense_layer(DenseLayer *layer, const double *input);`: Performs the forward pass by calculating the weighted sum of inputs and applying the activation function.
  - `void free_dense_layer(DenseLayer *layer);`: Frees the memory allocated for the dense layer.

### Neural Network

**Module:** `neural_network`

The neural network module manages a collection of layers and handles the forward pass through the entire network.

- **Functions:**
  - `NeuralNetwork* create_neural_network(int num_layers);`: Initializes a neural network with a specified number of layers.
  - `void add_dense_layer(NeuralNetwork *network, DenseLayer *layer);`: Adds a dense layer to the neural network.
  - `void forward_neural_network(NeuralNetwork *network, const double *input);`: Executes the forward pass through all layers.
  - `void free_neural_network(NeuralNetwork *network);`: Frees the memory allocated for the neural network.

## Training Algorithms

### Gradient Descent

**Module:** `gradient_descent`

Gradient descent is an optimization algorithm used to minimize the loss function by iteratively updating the model's parameters.

- **Functions:**
  - `void gradient_descent(double *weights, const double *gradients, int size, double learning_rate);`: Updates weights based on the computed gradients and learning rate.

### Backpropagation

**Module:** `backpropagation`

Backpropagation is the key algorithm used to train neural networks. It calculates the gradients of the loss function with respect to each weight by propagating the error backward through the network.

- **Functions:**
  - `void train_neural_network(NeuralNetwork *network, const double *X, const double *y, int num_samples, int epochs, double learning_rate);`: Trains the neural network using forward and backward passes, applying gradient descent to update weights.

## Advanced Techniques

### Momentum

**Module:** `momentum`

Momentum helps accelerate the gradient descent optimization by adding a fraction of the previous update to the current update.

- **Functions:**
  - `MomentumOptimizer* create_momentum_optimizer(int size, double momentum_factor);`: Initializes a momentum optimizer.
  - `void apply_momentum(MomentumOptimizer *optimizer, double *weights, double *gradients, int size, double learning_rate);`: Applies momentum during the weight update step.
  - `void free_momentum_optimizer(MomentumOptimizer *optimizer);`: Frees the memory allocated for the momentum optimizer.

### Regularization

**Module:** `regularization`

Regularization techniques like L2 regularization help prevent overfitting by penalizing large weights, encouraging the model to use smaller, more distributed weights.

- **Functions:**
  - `void apply_l2_regularization(double *weights, int size, double lambda, double learning_rate);`: Applies L2 regularization to the weights during training.

### Learning Rate Schedules

**Module:** `learning_rate_schedule`

Learning rate schedules adjust the learning rate during training, allowing the model to converge more effectively by reducing the learning rate as training progresses.

- **Functions:**
  - `LearningRateSchedule* create_learning_rate_schedule(double initial_learning_rate, double decay_rate, int decay_steps);`: Creates a learning rate schedule with exponential decay.
  - `double get_current_learning_rate(LearningRateSchedule *schedule);`: Returns the current learning rate based on the schedule.
  - `void update_learning_rate(LearningRateSchedule *schedule);`: Updates the learning rate as the training progresses.
  - `void free_learning_rate_schedule(LearningRateSchedule *schedule);`: Frees the memory allocated for the learning rate schedule.

### Adam Optimizer

**Module:** `adam`

The Adam optimizer is an adaptive learning rate optimization algorithm that combines the advantages of both momentum and RMSProp, handling sparse gradients effectively.

- **Functions:**
  - `AdamOptimizer* create_adam_optimizer(int size, double beta1, double beta2, double epsilon);`: Initializes an Adam optimizer.
  - `void apply_adam(AdamOptimizer *optimizer, double *weights, double *gradients, int size, double learning_rate);`: Applies the Adam optimization algorithm during the weight update step.
  - `void free_adam_optimizer(AdamOptimizer *optimizer);`: Frees the memory allocated for the Adam optimizer.

## Usage

To use any of these modules in your C project, include the respective header file in your source code. For example:

```c
#include "dense_layer.h"
#include "neural_network.h"
#include "gradient_descent.h"
#include "momentum.h"
#include "regularization.h"
#include "learning_rate_schedule.h"
#include "adam.h"
