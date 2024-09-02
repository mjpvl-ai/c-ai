// gradient_descent.c
#include "gradient_descent.h"

void gradient_descent(double *weights, const double *gradients, int size, double learning_rate) {
    for (int i = 0; i < size; i++) {
        weights[i] -= learning_rate * gradients[i];
    }
}
