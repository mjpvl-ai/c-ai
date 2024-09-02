// matrix_inversion.c
#include "matrix_inversion.h"
#include <stdlib.h>
#include <string.h>

int invert_matrix(const double *matrix, double *result, int n) {
    int i, j, k;
    double temp;
    double *augmented_matrix = (double *)malloc(n * 2 * n * sizeof(double));
    if (augmented_matrix == NULL) {
        return 0; // Memory allocation failed
    }

    // Initialize the augmented matrix with the input matrix and identity matrix
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            augmented_matrix[i * 2 * n + j] = matrix[i * n + j];
            augmented_matrix[i * 2 * n + (n + j)] = (i == j) ? 1 : 0;
        }
    }

    // Perform Gaussian elimination
    for (i = 0; i < n; i++) {
        if (augmented_matrix[i * 2 * n + i] == 0) {
            free(augmented_matrix);
            return 0; // Singular matrix, no inverse exists
        }

        temp = augmented_matrix[i * 2 * n + i];
        for (j = 0; j < 2 * n; j++) {
            augmented_matrix[i * 2 * n + j] /= temp;
        }

        for (j = 0; j < n; j++) {
            if (j != i) {
                temp = augmented_matrix[j * 2 * n + i];
                for (k = 0; k < 2 * n; k++) {
                    augmented_matrix[j * 2 * n + k] -= augmented_matrix[i * 2 * n + k] * temp;
                }
            }
        }
    }

    // Extract the inverted matrix from the augmented matrix
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            result[i * n + j] = augmented_matrix[i * 2 * n + (n + j)];
        }
    }

    free(augmented_matrix);
    return 1; // Success
}
