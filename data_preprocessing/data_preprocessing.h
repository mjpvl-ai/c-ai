// data_preprocessing.h

#ifndef DATA_PREPROCESSING_H
#define DATA_PREPROCESSING_H

void normalize(double *data, int size);
void standardize(double *data, int size);
void min_max_scaling(double *data, int size, double new_min, double new_max);
void one_hot_encode(const int *categories, int num_categories, int category_size, double *encoded_output);

#endif // DATA_PREPROCESSING_H
