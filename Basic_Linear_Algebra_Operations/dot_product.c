// dot_product.c
#include "dot_product.h"

double dot_product(const double *v1, const double *v2, int size) {
    double result = 0.0;
    for (int i = 0; i < size; i++) {
        result += v1[i] * v2[i];
    }
    return result;
}
