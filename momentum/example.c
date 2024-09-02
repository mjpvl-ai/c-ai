// example.c
#include <stdio.h>
#include "momentum.h"

int main() {
    // Define the size of the weight and gradient arrays
    int size = 3;

    // Example weights and gradients
    double weights[3] = {0.5, -0.2, 0.8};
    double gradients[3] = {0.1, -0.3, 0.2};

    // Learning rate and momentum factor
    double learning_rate = 0.01;
    double momentum_factor = 0.9;

    // Create the momentum optimizer
    MomentumOptimizer *optimizer = create_momentum_optimizer(size, momentum_factor);

    // Print initial weights
    printf("Initial weights:\n");
    for (int i = 0; i < size; i++) {
        printf("%f ", weights[i]);
    }
    printf("\n");

    // Apply the momentum optimizer for a few iterations
    for (int iteration = 0; iteration < 5; iteration++) {
        apply_momentum(optimizer, weights, gradients, size, learning_rate);

        // Print updated weights
        printf("Weights after iteration %d:\n", iteration + 1);
        for (int i = 0; i < size; i++) {
            printf("%f ", weights[i]);
        }
        printf("\n");
    }

    // Clean up and free the optimizer
    free_momentum_optimizer(optimizer);

    return 0;
}
