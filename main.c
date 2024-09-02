// #include <stdio.h>
// #include <stdlib.h>
// #include "data_preprocessing/data_preprocessing.h"

// #define SIZE 10
// #define CATEGORY_SIZE 3

// int main() {
//     // Example data
//     double data[SIZE] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0};
//     int categories[SIZE] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};
//     double encoded_output[SIZE * CATEGORY_SIZE];
    
//     // Normalization
//     printf("Original Data:\n");
//     for (int i = 0; i < SIZE; i++) {
//         printf("%f ", data[i]);
//     }
//     printf("\n");

//     normalize(data, SIZE);
//     printf("Normalized Data:\n");
//     for (int i = 0; i < SIZE; i++) {
//         printf("%f ", data[i]);
//     }
//     printf("\n");

//     // Standardization
//     double data2[SIZE] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0};
//     standardize(data2, SIZE);
//     printf("Standardized Data:\n");
//     for (int i = 0; i < SIZE; i++) {
//         printf("%f ", data2[i]);
//     }
//     printf("\n");

//     // Min-Max Scaling
//     double data3[SIZE] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0};
//     min_max_scaling(data3, SIZE, 0.0, 1.0);
//     printf("Min-Max Scaled Data (0 to 1):\n");
//     for (int i = 0; i < SIZE; i++) {
//         printf("%f ", data3[i]);
//     }
//     printf("\n");

//     // One-Hot Encoding
//     one_hot_encode(categories, SIZE, CATEGORY_SIZE, encoded_output);
//     printf("One-Hot Encoded Data:\n");
//     for (int i = 0; i < SIZE; i++) {
//         for (int j = 0; j < CATEGORY_SIZE; j++) {
//             printf("%f ", encoded_output[i * CATEGORY_SIZE + j]);
//         }
//         printf("\n");
//     }

//     return 0;
// }


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "activation/activation_functions.h"
#include "Basic Linear Algebra Operations/dot_product.h"
#include "Basic Linear Algebra Operations/loss_functions.h"
#include "Basic Linear Algebra Operations/vector_ops.h"
#include "data_preprocessing/data_preprocessing.h"
#include "matrix_operations/matrix_determinant.h"
#include "matrix_operations/matrix_inversion.h"
#include "matrix_operations/matrix_ops.h"
#include "matrix_operations/matrix_transpose.h"
#include "momentum/momentum.h"
#include "neural_networks/dense_layer.h"
#include "neural_networks/neural_network_training.h"
#include "neural_networks/neural_networks.h"
#include "optimization/adam.h"
#include "optimization/gradient_descent.h"
#include "optimization/learning_rate_schedule.h"
#include "regularization/regularization.h"
#include "statistics/statistics.h"
#include "supervised_learning/supervised_learning.h"
#include "unsupervised_learning/unsupervised_learning.h"

void test_activation_functions() {
    double x = 1.0;
    printf("Sigmoid(%f) = %f\n", x, sigmoid(x));
    printf("ReLU(%f) = %f\n", x, relu(x));
    printf("Tanh(%f) = %f\n", x, tanh_activation(x));
}

void test_dot_product() {
    double v1[] = {1.0, 2.0, 3.0};
    double v2[] = {4.0, 5.0, 6.0};
    int size = 3;
    printf("Dot Product = %f\n", dot_product(v1, v2, size));
}

void test_loss_functions() {
    double y_true[] = {1.0, 0.0, 1.0};
    double y_pred[] = {0.8, 0.2, 0.6};
    int size = 3;
    printf("Mean Squared Error = %f\n", mean_squared_error(y_true, y_pred, size));
    printf("Cross Entropy Loss = %f\n", cross_entropy_loss(y_true, y_pred, size));
}

void test_vector_operations() {
    double v1[] = {1.0, 2.0, 3.0};
    double v2[] = {4.0, 5.0, 6.0};
    int size = 3;
    double result[3];

    vector_addition(v1, v2, result, size);
    printf("Vector Addition: ");
    for (int i = 0; i < size; i++) {
        printf("%f ", result[i]);
    }
    printf("\n");

    vector_subtraction(v1, v2, result, size);
    printf("Vector Subtraction: ");
    for (int i = 0; i < size; i++) {
        printf("%f ", result[i]);
    }
    printf("\n");
}

void test_data_preprocessing() {
    double data[] = {10.0, 20.0, 30.0, 40.0};
    int size = 4;

    normalize(data, size);
    printf("Normalized Data: ");
    for (int i = 0; i < size; i++) {
        printf("%f ", data[i]);
    }
    printf("\n");

    standardize(data, size);
    printf("Standardized Data: ");
    for (int i = 0; i < size; i++) {
        printf("%f ", data[i]);
    }
    printf("\n");

    min_max_scaling(data, size, 0.0, 1.0);
    printf("Min-Max Scaled Data: ");
    for (int i = 0; i < size; i++) {
        printf("%f ", data[i]);
    }
    printf("\n");

    int categories[] = {0, 1, 2};
    int num_categories = 3;
    double encoded_output[9];
    one_hot_encode(categories, num_categories, num_categories, encoded_output);
    printf("One-Hot Encoded Data: ");
    for (int i = 0; i < 9; i++) {
        printf("%f ", encoded_output[i]);
    }
    printf("\n");
}

void test_matrix_operations() {
    double matrix1[] = {1.0, 2.0, 3.0, 4.0};
    double matrix2[] = {5.0, 6.0, 7.0, 8.0};
    double result[4];
    int rows = 2, cols = 2;

    matrix_multiply(matrix1, matrix2, result, rows, cols, cols);
    printf("Matrix Multiplication Result:\n");
    for (int i = 0; i < 4; i++) {
        printf("%f ", result[i]);
        if ((i + 1) % cols == 0) printf("\n");
    }

    transpose_matrix(matrix1, result, rows, cols);
    printf("Matrix Transpose Result:\n");
    for (int i = 0; i < 4; i++) {
        printf("%f ", result[i]);
        if ((i + 1) % cols == 0) printf("\n");
    }

    double determinant_result = determinant(matrix1, rows);
    printf("Matrix Determinant = %f\n", determinant_result);

    int invert_status = invert_matrix(matrix1, result, rows);
    if (invert_status == 0) {
        printf("Matrix Inversion Result:\n");
        for (int i = 0; i < 4; i++) {
            printf("%f ", result[i]);
            if ((i + 1) % cols == 0) printf("\n");
        }
    } else {
        printf("Matrix inversion failed\n");
    }
}

void test_momentum_optimizer() {
    int size = 3;
    double weights[] = {1.0, 2.0, 3.0};
    double gradients[] = {0.1, 0.2, 0.3};
    double learning_rate = 0.01;

    MomentumOptimizer* optimizer = create_momentum_optimizer(size, 0.9);
    apply_momentum(optimizer, weights, gradients, size, learning_rate);

    printf("Weights after Momentum Update: ");
    for (int i = 0; i < size; i++) {
        printf("%f ", weights[i]);
    }
    printf("\n");

    free_momentum_optimizer(optimizer);
}

void test_neural_networks() {
    int input_size = 3, output_size = 2;
    DenseLayer *dense_layer = create_dense_layer(input_size, output_size);

    double input[] = {1.0, 2.0, 3.0};
    forward_dense(dense_layer, input);
    printf("Dense Layer Output: ");
    for (int i = 0; i < output_size; i++) {
        printf("%f ", dense_layer->output[i]);
    }
    printf("\n");

    free_dense_layer(dense_layer);
}

void test_learning_rate_schedule() {
    LearningRateSchedule* schedule = create_learning_rate_schedule(0.1, 0.01, 10);

    for (int i = 0; i < 20; i++) {
        printf("Learning Rate at step %d: %f\n", i, get_current_learning_rate(schedule));
        update_learning_rate(schedule);
    }

    free_learning_rate_schedule(schedule);
}

void test_statistics() {
    double data1[] = {1.0, 2.0, 3.0, 4.0};
    double data2[] = {5.0, 6.0, 7.0, 8.0};
    int size = 4;

    printf("Mean = %f\n", mean(data1, size));
    printf("Median = %f\n", median(data1, size));
    printf("Variance = %f\n", variance(data1, size));
    printf("Standard Deviation = %f\n", standard_deviation(data1, size));
    printf("Covariance = %f\n", covariance(data1, data2, size));
    printf("Correlation Coefficient = %f\n", correlation_coefficient(data1, data2, size));
}

void test_supervised_learning() {
    float x[] = {1.0, 2.0, 3.0};
    float y[] = {2.0, 4.0, 6.0};
    int n = 3;
    float slope, intercept;

    linear_regression(x, y, n, &slope, &intercept);
    printf("Linear Regression: slope = %f, intercept = %f\n", slope, intercept);

    float weights[] = {0.0, 0.0};
    logistic_regression(x, (int*)y, n, weights, 0.1, 100);
    printf("Logistic Regression Weights: %f, %f\n", weights[0], weights[1]);

    float query_point[] = {2.5};
    int label = knn_classify(x, (int*)y, n, 1, query_point, 1);
    printf("k-NN Classification Result: %d\n", label);

    Node* tree = create_node(0, 1.5, 1);
    int tree_prediction = predict_decision_tree(tree, query_point);
    printf("Decision Tree Prediction: %d\n", tree_prediction);

    RandomForest rf;
    rf.num_trees = NUM_TREES;
    rf.trees = (Node**)malloc(NUM_TREES * sizeof(Node*));
    int forest_prediction = predict_random_forest(&rf, query_point);
    printf("Random Forest Prediction: %d\n", forest_prediction);

    float svm_weights[] = {0.0, 0.0};
    float bias = 0.0;
    svm_train(x, (int*)y, n, svm_weights, &bias, 0.1, 100);
    printf("SVM Weights: %f, %f; Bias: %f\n", svm_weights[0], svm_weights[1], bias);

    float probabilities[2];
    naive_bayes_classifier(x, (int*)y, n, query_point, 1, probabilities);
    printf("Naive Bayes Classification Probabilities: %f, %f\n", probabilities[0], probabilities[1]);

    free(rf.trees);
    free_tree(tree);
}

void test_unsupervised_learning() {
    float data[] = {1.0, 2.0, 3.0, 4.0};
    int n = 4;
    int k = 2;
    int clusters[4];

    kmeans(data, n, k, clusters);
    printf("k-Means Clustering Results: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", clusters[i]);
    }
    printf("\n");

    pca(data, n, 2, clusters,2);  // Using clusters as a placeholder for PCA results
    printf("PCA Results: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", clusters[i]);
    }
    printf("\n");
}

int main() {
    test_activation_functions();
    test_dot_product();
    test_loss_functions();
    test_vector_operations();
    test_data_preprocessing();
    test_matrix_operations();
    test_momentum_optimizer();
    test_neural_networks();
    test_learning_rate_schedule();
    test_statistics();
    test_supervised_learning();
    test_unsupervised_learning();

    return 0;
}
