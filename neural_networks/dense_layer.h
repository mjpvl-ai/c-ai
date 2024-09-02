// dense_layer.h
#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

typedef struct {
    int input_size;
    int output_size;
    double *weights;
    double *biases;
    double *output;
} DenseLayer;

DenseLayer* create_dense_layer(int input_size, int output_size);
void forward_dense(DenseLayer *layer, const double *input);
void free_dense_layer(DenseLayer *layer);

#endif // DENSE_LAYER_H
