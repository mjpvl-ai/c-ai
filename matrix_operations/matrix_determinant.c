// matrix_determinant.c
#include "matrix_determinant.h"
#include <stdlib.h>

double determinant(const double *matrix, int n) {
    if (n == 1) {
        return matrix[0];
    }
    if (n == 2) {
        return matrix[0] * matrix[3] - matrix[1] * matrix[2];
    }

    double det = 0.0;
    for (int p = 0; p < n; p++) {
        double *sub_matrix = (double *)malloc((n - 1) * (n - 1) * sizeof(double));
        for (int i = 1; i < n; i++) {
            int sub_col = 0;
            for (int j = 0; j < n; j++) {
                if (j == p) {
                    continue;
                }
                sub_matrix[(i - 1) * (n - 1) + sub_col] = matrix[i * n + j];
                sub_col++;
            }
        }
        det += matrix[p] * determinant(sub_matrix, n - 1) * ((p % 2 == 0) ? 1 : -1);
        free(sub_matrix);
    }
    return det;
}
