#include <stdio.h>
#include "dot_product.h"
#include "loss_functions.h"
#include "vector_ops.h"

int main() {
    int size = 3;

    // Example vectors
    double v1[] = {1.0, 2.0, 3.0};
    double v2[] = {4.0, 5.0, 6.0};

    // Result vector for addition and subtraction
    double result_addition[3];
    double result_subtraction[3];

    // Use dot_product
    double dot_result = dot_product(v1, v2, size);
    printf("Dot Product: %f\n", dot_result);

    // Use vector_addition
    vector_addition(v1, v2, result_addition, size);
    printf("Vector Addition: ");
    for (int i = 0; i < size; i++) {
        printf("%f ", result_addition[i]);
    }
    printf("\n");

    // Use vector_subtraction
    vector_subtraction(v1, v2, result_subtraction, size);
    printf("Vector Subtraction: ");
    for (int i = 0; i < size; i++) {
        printf("%f ", result_subtraction[i]);
    }
    printf("\n");

    // Example true and predicted values for loss functions
    double y_true[] = {1.0, 0.0, 1.0};
    double y_pred[] = {0.9, 0.1, 0.8};

    // Use mean_squared_error
    double mse = mean_squared_error(y_true, y_pred, size);
    printf("Mean Squared Error: %f\n", mse);

    // Use cross_entropy_loss
    double cross_entropy = cross_entropy_loss(y_true, y_pred, size);
    printf("Cross Entropy Loss: %f\n", cross_entropy);

    return 0;
}
