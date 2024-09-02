#include "momentum.h"
#include <stdlib.h>

MomentumOptimizer* create_momentum_optimizer(int size, double momentum_factor) {
    MomentumOptimizer *optimizer = (MomentumOptimizer *)malloc(sizeof(MomentumOptimizer));
    optimizer->velocity = (double *)calloc(size, sizeof(double));
    optimizer->momentum_factor = momentum_factor;
    return optimizer;
}

void apply_momentum(MomentumOptimizer *optimizer, double *weights, double *gradients, int size, double learning_rate) {
    for (int i = 0; i < size; i++) {
        optimizer->velocity[i] = optimizer->momentum_factor * optimizer->velocity[i] - learning_rate * gradients[i];
        weights[i] += optimizer->velocity[i];
    }
}

void free_momentum_optimizer(MomentumOptimizer *optimizer) {
    free(optimizer->velocity);
    free(optimizer);
}
