#include "supervised_learning.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_TREES 5

int main() {
    // Linear Regression
    float x[] = {1, 2, 3, 4, 5};
    float y[] = {2, 4, 6, 8, 10};
    int n = 5;
    float slope, intercept;
    linear_regression(x, y, n, &slope, &intercept);
    printf("Linear Regression: Slope = %.6f, Intercept = %.6f\n", slope, intercept);

    // Logistic Regression
    float log_x[] = {0.1, 0.2, 0.3, 0.4, 0.5};
    int log_y[] = {0, 0, 1, 1, 1};
    int log_n = 5;
    float weights[] = {0.0};
    float learning_rate = 0.1;
    int epochs = 1000;
    logistic_regression(log_x, log_y, log_n, weights, learning_rate, epochs);
    printf("Logistic Regression: Weight = %.6f\n", weights[0]);

    // k-Nearest Neighbors (k-NN)
    float data_points[] = {1, 2, 2, 4, 5, 6, 7, 8, 9, 10};
    int labels[] = {0, 0, 1, 1, 0};
    int num_points = 5;
    int num_features = 2;
    float query_point[] = {3, 4};
    int k = 3;
    int predicted_label = knn_classify(data_points, labels, num_points, num_features, query_point, k);
    printf("k-NN Classification: Predicted Label = %d\n", predicted_label);

    // Decision Trees
    Node *root = create_node(0, 5.0, 1);
    if (root == NULL) {
        perror("Failed to create decision tree root node");
        return 1; // Exit with error code
    }
    float test_features[] = {4.0};
    int predicted_label_dt = predict_decision_tree(root, test_features);
    printf("Decision Tree Classification: Predicted Label = %d\n", predicted_label_dt);

    // Random Forests
    RandomForest rf;
    rf.num_trees = NUM_TREES;
    rf.trees = (Node **)malloc(NUM_TREES * sizeof(Node *));
    if (rf.trees == NULL) {
        perror("Failed to allocate memory for random forest trees");
        return 1; // Exit with error code
    }
    for (int i = 0; i < NUM_TREES; i++) {
        rf.trees[i] = create_node(0, 5.0, 1);
        if (rf.trees[i] == NULL) {
            perror("Failed to allocate memory for decision tree node");
            return 1; // Exit with error code
        }
    }
    int predicted_label_rf = predict_random_forest(&rf, test_features);
    printf("Random Forest Classification: Predicted Label = %d\n", predicted_label_rf);
    for (int i = 0; i < NUM_TREES; i++) {
        free(rf.trees[i]);
    }
    free(rf.trees);

    // Support Vector Machines (SVM)
    float svm_weights[] = {1.0};
    float svm_bias = 0.5;
    float svm_features[] = {2.0};
    float svm_prediction = svm_predict(svm_weights, &svm_bias, svm_features, 1);
    printf("SVM Prediction: %.2f\n", svm_prediction);

    // Naive Bayes Classifier
    float nb_features[] = {1.0};
    float nb_mean[] = {1.0};
    float nb_variance[] = {0.1};
    float nb_prior[] = {0.5};
    num_features = 1;
    int num_classes = 2;
    int predicted_class_nb = naive_bayes_predict(nb_features, nb_mean, nb_variance, nb_prior, num_features, num_classes);
    printf("Naive Bayes Classification: Predicted Class = %d\n", predicted_class_nb);

    return 0;
}
