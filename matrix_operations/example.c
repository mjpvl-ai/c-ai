// example.c
#include <stdio.h>
#include "matrix_determinant.h"
#include "matrix_inversion.h"
#include "matrix_ops.h"
#include "matrix_transpose.h"

int main() {
    // Example matrix (3x3)
    double matrix[9] = {
        4, 7, 2,
        3, 6, 1,
        2, 5, 9
    };

    // Calculate the determinant of a 3x3 matrix
    double det = determinant(matrix, 3);
    printf("Determinant: %f\n", det);

    // Invert the matrix (3x3)
    double inverse[9];
    if (invert_matrix(matrix, inverse, 3)) {
        printf("Inverse Matrix:\n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                printf("%f ", inverse[i * 3 + j]);
            }
            printf("\n");
        }
    } else {
        printf("Matrix inversion failed (matrix might be singular).\n");
    }

    // Multiply two matrices (3x3 * 3x3)
    double matrix2[9] = {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };
    double result[9];
    matrix_multiply(matrix, matrix2, result, 3, 3, 3);
    printf("Matrix Multiplication Result (Matrix * Identity):\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%f ", result[i * 3 + j]);
        }
        printf("\n");
    }

    // Transpose the matrix (3x3)
    double transpose[9];
    transpose_matrix(matrix, transpose, 3, 3);
    printf("Transposed Matrix:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%f ", transpose[i * 3 + j]);
        }
        printf("\n");
    }

    return 0;
}
