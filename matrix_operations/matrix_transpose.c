// matrix_transpose.c
#include "matrix_transpose.h"

void transpose_matrix(const double *matrix, double *result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j * rows + i] = matrix[i * cols + j];
        }
    }
}
