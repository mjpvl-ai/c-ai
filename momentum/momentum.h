#ifndef MOMENTUM_H
#define MOMENTUM_H

typedef struct {
    double *velocity;
    double momentum_factor;
} MomentumOptimizer;

MomentumOptimizer* create_momentum_optimizer(int size, double momentum_factor);
void apply_momentum(MomentumOptimizer *optimizer, double *weights, double *gradients, int size, double learning_rate);
void free_momentum_optimizer(MomentumOptimizer *optimizer);

#endif // MOMENTUM_H
