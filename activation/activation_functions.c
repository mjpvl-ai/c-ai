// activation_functions.c
#include "activation_functions.h"
#include <math.h>

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double relu(double x) {
    return fmax(0.0, x);
}

double tanh_activation(double x) {
    return tanh(x);
}
