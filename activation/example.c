// example.c
#include <stdio.h>
#include "activation_functions.h"

int main() {
    double input = -0.5;

    // Using the Sigmoid function
    double sigmoid_result = sigmoid(input);
    printf("Sigmoid(%f) = %f\n", input, sigmoid_result);

    // Using the ReLU function
    double relu_result = relu(input);
    printf("ReLU(%f) = %f\n", input, relu_result);

    // Using the Tanh function
    double tanh_result = tanh_activation(input);
    printf("Tanh(%f) = %f\n", input, tanh_result);

    return 0;
}
