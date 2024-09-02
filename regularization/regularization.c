#include "regularization.h"

void apply_l2_regularization(double *weights, int size, double lambda, double learning_rate) {
    for (int i = 0; i < size; i++) {
        weights[i] -= learning_rate * lambda * weights[i];
    }
}
