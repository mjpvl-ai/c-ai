#include <stdio.h>
#include <stdlib.h>
#include "data_preprocessing.h"

#define SIZE 10
#define CATEGORY_SIZE 3

int main() {
    // Example data
    double data[SIZE] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0};
    int categories[SIZE] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};
    double encoded_output[SIZE * CATEGORY_SIZE];
    
    // Normalization
    printf("Original Data:\n");
    for (int i = 0; i < SIZE; i++) {
        printf("%f ", data[i]);
    }
    printf("\n");

    normalize(data, SIZE);
    printf("Normalized Data:\n");
    for (int i = 0; i < SIZE; i++) {
        printf("%f ", data[i]);
    }
    printf("\n");

    // Standardization
    double data2[SIZE] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0};
    standardize(data2, SIZE);
    printf("Standardized Data:\n");
    for (int i = 0; i < SIZE; i++) {
        printf("%f ", data2[i]);
    }
    printf("\n");

    // Min-Max Scaling
    double data3[SIZE] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0};
    min_max_scaling(data3, SIZE, 0.0, 1.0);
    printf("Min-Max Scaled Data (0 to 1):\n");
    for (int i = 0; i < SIZE; i++) {
        printf("%f ", data3[i]);
    }
    printf("\n");

    // One-Hot Encoding
    one_hot_encode(categories, SIZE, CATEGORY_SIZE, encoded_output);
    printf("One-Hot Encoded Data:\n");
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < CATEGORY_SIZE; j++) {
            printf("%f ", encoded_output[i * CATEGORY_SIZE + j]);
        }
        printf("\n");
    }

    return 0;
}