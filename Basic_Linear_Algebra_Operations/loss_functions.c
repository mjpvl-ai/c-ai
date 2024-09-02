// loss_functions.c
#include "loss_functions.h"
#include <math.h>

double mean_squared_error(const double *y_true, const double *y_pred, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += pow(y_true[i] - y_pred[i], 2);
    }
    return sum / size;
}

double cross_entropy_loss(const double *y_true, const double *y_pred, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += -y_true[i] * log(y_pred[i]) - (1.0 - y_true[i]) * log(1.0 - y_pred[i]);
    }
    return sum / size;
}
