#ifndef ADAM_H
#define ADAM_H

typedef struct {
    double *m;
    double *v;
    double beta1;
    double beta2;
    double epsilon;
    int t;
} AdamOptimizer;

AdamOptimizer* create_adam_optimizer(int size, double beta1, double beta2, double epsilon);
void apply_adam(AdamOptimizer *optimizer, double *weights, double *gradients, int size, double learning_rate);
void free_adam_optimizer(AdamOptimizer *optimizer);

#endif // ADAM_H
