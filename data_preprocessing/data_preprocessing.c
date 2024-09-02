// data_preprocessing.c

#include <math.h>
#include <float.h>
#include "data_preprocessing.h"
#include "../statistics/statistics.h"

void normalize(double *data, int size) {
    double data_mean = mean(data, size);
    double data_stddev = standard_deviation(data, size);
    for (int i = 0; i < size; i++) {
        data[i] = (data[i] - data_mean) / data_stddev;
    }
}

void standardize(double *data, int size) {
    double data_min = DBL_MAX;
    double data_max = -DBL_MAX;
    for (int i = 0; i < size; i++) {
        if (data[i] < data_min) data_min = data[i];
        if (data[i] > data_max) data_max = data[i];
    }
    for (int i = 0; i < size; i++) {
        data[i] = (data[i] - data_min) / (data_max - data_min);
    }
}

void min_max_scaling(double *data, int size, double new_min, double new_max) {
    double data_min = DBL_MAX;
    double data_max = -DBL_MAX;
    for (int i = 0; i < size; i++) {
        if (data[i] < data_min) data_min = data[i];
        if (data[i] > data_max) data_max = data[i];
    }
    for (int i = 0; i < size; i++) {
        data[i] = new_min + (data[i] - data_min) * (new_max - new_min) / (data_max - data_min);
    }
}

void one_hot_encode(const int *categories, int num_categories, int category_size, double *encoded_output) {
    for (int i = 0; i < num_categories; i++) {
        for (int j = 0; j < category_size; j++) {
            encoded_output[i * category_size + j] = (categories[i] == j) ? 1.0 : 0.0;
        }
    }
}
