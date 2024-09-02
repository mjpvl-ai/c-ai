// supervised_learning.c
#include "supervised_learning.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Linear Regression
void linear_regression(float *x, float *y, int n, float *slope, float *intercept) {
    float x_mean = 0, y_mean = 0, numerator = 0, denominator = 0;
    
    for (int i = 0; i < n; i++) {
        x_mean += x[i];
        y_mean += y[i];
    }
    
    x_mean /= n;
    y_mean /= n;
    
    for (int i = 0; i < n; i++) {
        numerator += (x[i] - x_mean) * (y[i] - y_mean);
        denominator += (x[i] - x_mean) * (x[i] - x_mean);
    }
    
    *slope = numerator / denominator;
    *intercept = y_mean - (*slope * x_mean);
}

// Logistic Regression
void logistic_regression(float *x, int *y, int n, float *weights, float learning_rate, int epochs) {
    int m = 1; // Number of features
    for (int epoch = 0; epoch < epochs; epoch++) {
        float gradient = 0;
        for (int i = 0; i < n; i++) {
            float prediction = 1.0 / (1.0 + exp(-weights[0] * x[i]));
            gradient += (y[i] - prediction) * x[i];
        }
        gradient /= n;
        weights[0] += learning_rate * gradient;
    }
}

// k-Nearest Neighbors (k-NN)
float euclidean_distance(float *a, float *b, int dim) {
    float sum = 0;
    for (int i = 0; i < dim; i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(sum);
}

int knn_classify(float *data_points, int *labels, int num_points, int num_features, float *query_point, int k) {
    float *distances = (float *)malloc(num_points * sizeof(float));
    int *sorted_indices = (int *)malloc(num_points * sizeof(int));
    
    // Calculate distances
    for (int i = 0; i < num_points; i++) {
        distances[i] = euclidean_distance(&data_points[i * num_features], query_point, num_features);
        sorted_indices[i] = i;
    }
    
    // Sort distances
    for (int i = 0; i < num_points - 1; i++) {
        for (int j = i + 1; j < num_points; j++) {
            if (distances[j] < distances[i]) {
                float temp_dist = distances[i];
                distances[i] = distances[j];
                distances[j] = temp_dist;
                
                int temp_idx = sorted_indices[i];
                sorted_indices[i] = sorted_indices[j];
                sorted_indices[j] = temp_idx;
            }
        }
    }
    
    // Find the most common label among the k nearest neighbors
    int *votes = (int *)calloc(10, sizeof(int)); // Assuming 10 classes
    for (int i = 0; i < k; i++) {
        votes[labels[sorted_indices[i]]]++;
    }
    
    int max_votes = 0;
    int predicted_label = -1;
    for (int i = 0; i < 10; i++) {
        if (votes[i] > max_votes) {
            max_votes = votes[i];
            predicted_label = i;
        }
    }
    
    free(distances);
    free(sorted_indices);
    free(votes);
    
    return predicted_label;
}

// Decision Trees
Node *create_node(int feature_index, float threshold, int label) {
    Node *node = (Node *)malloc(sizeof(Node));
    node->feature_index = feature_index;
    node->threshold = threshold;
    node->label = label;
    node->left = NULL;
    node->right = NULL;
    return node;
}

int predict_decision_tree(Node *node, float *features) {
    if (node->left == NULL && node->right == NULL) {
        return node->label;
    }
    if (features[node->feature_index] <= node->threshold) {
        return predict_decision_tree(node->left, features);
    } else {
        return predict_decision_tree(node->right, features);
    }
}

// Random Forests
int predict_random_forest(RandomForest *rf, float *features) {
    int *votes = (int *)calloc(10, sizeof(int)); // Assuming 10 classes
    for (int i = 0; i < rf->num_trees; i++) {
        int label = predict_decision_tree(rf->trees[i], features);
        votes[label]++;
    }
    
    int max_votes = 0;
    int predicted_label = -1;
    for (int i = 0; i < 10; i++) {
        if (votes[i] > max_votes) {
            max_votes = votes[i];
            predicted_label = i;
        }
    }
    
    free(votes);
    
    return predicted_label;
}

// Support Vector Machines (SVM)
float svm_predict(float *weights, float *bias, float *features, int num_features) {
    float sum = 0;
    for (int i = 0; i < num_features; i++) {
        sum += weights[i] * features[i];
    }
    return sum + *bias;
}

// Naive Bayes Classifier
int naive_bayes_predict(float *features, float *mean, float *variance, float *prior, int num_features, int num_classes) {
    float *posteriors = (float *)malloc(num_classes * sizeof(float));
    
    for (int i = 0; i < num_classes; i++) {
        float posterior = log(prior[i]);
        for (int j = 0; j < num_features; j++) {
            float diff = features[j] - mean[i * num_features + j];
            posterior += -0.5 * log(2 * M_PI * variance[i * num_features + j])
                         - (diff * diff) / (2 * variance[i * num_features + j]);
        }
        posteriors[i] = posterior;
    }
    
    int predicted_class = 0;
    float max_posterior = posteriors[0];
    for (int i = 1; i < num_classes; i++) {
        if (posteriors[i] > max_posterior) {
            max_posterior = posteriors[i];
            predicted_class = i;
        }
    }
    
    free(posteriors);
    
    return predicted_class;
}
