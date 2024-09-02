// loss_functions.h
#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

double mean_squared_error(const double *y_true, const double *y_pred, int size);
double cross_entropy_loss(const double *y_true, const double *y_pred, int size);

#endif // LOSS_FUNCTIONS_H
