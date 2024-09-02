#include <stdio.h>
#include "adam.h"
#include "gradient_descent.h"
#include "learning_rate_schedule.h"

#define SIZE 5

int main() {
    // Initialize weights and gradients
    double weights[SIZE] = {0.5, 0.2, -0.3, 0.1, -0.1};
    double gradients[SIZE] = {0.1, -0.2, 0.3, -0.1, 0.05};

    // Create Adam optimizer
    AdamOptimizer *adam_optimizer = create_adam_optimizer(SIZE, 0.9, 0.999, 1e-8);

    // Create learning rate schedule
    LearningRateSchedule *schedule = create_learning_rate_schedule(0.01, 0.96, 100);

    // Perform optimization
    printf("Initial weights: ");
    for (int i = 0; i < SIZE; i++) {
        printf("%f ", weights[i]);
    }
    printf("\n");

    // Training loop
    for (int epoch = 0; epoch < 10; epoch++) {
        // Update learning rate
        double learning_rate = get_current_learning_rate(schedule);
        update_learning_rate(schedule);

        // Apply Adam optimization
        apply_adam(adam_optimizer, weights, gradients, SIZE, learning_rate);

        // Print updated weights
        printf("Epoch %d - Learning Rate: %f, Weights: ", epoch + 1, learning_rate);
        for (int i = 0; i < SIZE; i++) {
            printf("%f ", weights[i]);
        }
        printf("\n");
    }

    // Free resources
    free_adam_optimizer(adam_optimizer);
    free_learning_rate_schedule(schedule);

    return 0;
}
