#include "adam.h"
#include <stdlib.h>
#include <math.h>

AdamOptimizer* create_adam_optimizer(int size, double beta1, double beta2, double epsilon) {
    AdamOptimizer *optimizer = (AdamOptimizer *)malloc(sizeof(AdamOptimizer));
    optimizer->m = (double *)calloc(size, sizeof(double));
    optimizer->v = (double *)calloc(size, sizeof(double));
    optimizer->beta1 = beta1;
    optimizer->beta2 = beta2;
    optimizer->epsilon = epsilon;
    optimizer->t = 0;
    return optimizer;
}

void apply_adam(AdamOptimizer *optimizer, double *weights, double *gradients, int size, double learning_rate) {
    optimizer->t++;
    for (int i = 0; i < size; i++) {
        optimizer->m[i] = optimizer->beta1 * optimizer->m[i] + (1 - optimizer->beta1) * gradients[i];
        optimizer->v[i] = optimizer->beta2 * optimizer->v[i] + (1 - optimizer->beta2) * gradients[i] * gradients[i];

        double m_hat = optimizer->m[i] / (1 - pow(optimizer->beta1, optimizer->t));
        double v_hat = optimizer->v[i] / (1 - pow(optimizer->beta2, optimizer->t));

        weights[i] -= learning_rate * m_hat / (sqrt(v_hat) + optimizer->epsilon);
    }
}

void free_adam_optimizer(AdamOptimizer *optimizer) {
    free(optimizer->m);
    free(optimizer->v);
    free(optimizer);
}
