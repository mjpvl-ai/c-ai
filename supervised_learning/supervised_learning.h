// supervised_learning.h
#ifndef SUPERVISED_LEARNING_H
#define SUPERVISED_LEARNING_H

typedef struct Node {
    int feature_index;
    float threshold;
    int label;
    struct Node *left;
    struct Node *right;
} Node;

typedef struct RandomForest {
    Node **trees;  // Array of pointers to decision trees
    int num_trees; // Number of trees in the forest
} RandomForest;

#define NUM_TREES 5

// Linear Regression
void linear_regression(float *x, float *y, int n, float *slope, float *intercept);

// Logistic Regression
void logistic_regression(float *x, int *y, int n, float *weights, float learning_rate, int epochs);

// k-Nearest Neighbors (k-NN)
float euclidean_distance(float *a, float *b, int dim);
int knn_classify(float *data_points, int *labels, int num_points, int num_features, float *query_point, int k);

// Decision Trees
Node *create_node(int feature_index, float threshold, int label);
int predict_decision_tree(Node *node, float *features);

// Random Forests
int predict_random_forest(RandomForest *rf, float *features);

// Support Vector Machines (SVM)
float svm_predict(float *weights, float *bias, float *features, int num_features);

// Naive Bayes Classifier
int naive_bayes_predict(float *features, float *mean, float *variance, float *prior, int num_features, int num_classes);

#endif // SUPERVISED_LEARNING_H
