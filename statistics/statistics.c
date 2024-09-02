// statistics.c
#include "statistics.h"
#include <stdlib.h>
#include <math.h>

double mean(const double *data, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum / size;
}

int compare(const void *a, const void *b) {
    double da = *(double *)a;
    double db = *(double *)b;
    return (da > db) - (da < db);
}


double median(double *data, int size) {
    qsort(data, size, sizeof(double), compare);
    if (size % 2 == 0) {
        return (data[size / 2 - 1] + data[size / 2]) / 2;
    } else {
        return data[size / 2];
    }
}

double mode(const double *data, int size) {
    double mode = data[0];
    int max_count = 0;
    for (int i = 0; i < size; i++) {
        int count = 0;
        for (int j = 0; j < size; j++) {
            if (data[j] == data[i]) count++;
        }
        if (count > max_count) {
            max_count = count;
            mode = data[i];
        }
    }
    return mode;
}

// statistics.c (continued)

double variance(const double *data, int size) {
    double m = mean(data, size);
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += pow(data[i] - m, 2);
    }
    return sum / size;
}

double standard_deviation(const double *data, int size) {
    return sqrt(variance(data, size));
}

double covariance(const double *data1, const double *data2, int size) {
    double mean1 = mean(data1, size);
    double mean2 = mean(data2, size);
    double cov = 0.0;
    for (int i = 0; i < size; i++) {
        cov += (data1[i] - mean1) * (data2[i] - mean2);
    }
    return cov / size;
}

double correlation_coefficient(const double *data1, const double *data2, int size) {
    double cov = covariance(data1, data2, size);
    double stddev1 = standard_deviation(data1, size);
    double stddev2 = standard_deviation(data2, size);
    return cov / (stddev1 * stddev2);
}

