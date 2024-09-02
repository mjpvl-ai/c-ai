// vector_ops.c
#include "vector_ops.h"

void vector_addition(const double *v1, const double *v2, double *result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = v1[i] + v2[i];
    }
}

void vector_subtraction(const double *v1, const double *v2, double *result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = v1[i] - v2[i];
    }
}
