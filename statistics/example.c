// example.c
#include <stdio.h>
#include "statistics.h"

int main() {
    double data1[] = {2.3, 4.1, 6.5, 7.2, 5.6, 3.3};
    double data2[] = {1.2, 3.8, 5.5, 6.1, 4.8, 2.9};
    int size = sizeof(data1) / sizeof(data1[0]);

    // Calculate and print mean
    double mean_value = mean(data1, size);
    printf("Mean: %f\n", mean_value);

    // Calculate and print median
    double median_value = median(data1, size);
    printf("Median: %f\n", median_value);

    // Calculate and print mode
    double mode_value = mode(data1, size);
    printf("Mode: %f\n", mode_value);

    // Calculate and print variance
    double variance_value = variance(data1, size);
    printf("Variance: %f\n", variance_value);

    // Calculate and print standard deviation
    double stddev_value = standard_deviation(data1, size);
    printf("Standard Deviation: %f\n", stddev_value);

    // Calculate and print covariance
    double covariance_value = covariance(data1, data2, size);
    printf("Covariance: %f\n", covariance_value);

    // Calculate and print correlation coefficient
    double correlation_value = correlation_coefficient(data1, data2, size);
    printf("Correlation Coefficient: %f\n", correlation_value);

    return 0;
}
