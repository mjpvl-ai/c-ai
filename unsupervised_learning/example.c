#include "unsupervised_learning.h"
#include <stdio.h>
#include <stdlib.h>

#define NUM_POINTS 8
#define NUM_FEATURES 2
#define K 2
#define NUM_COMPONENTS 2

int main() {
    // Sample data for clustering
    float data[NUM_POINTS][NUM_FEATURES] = {
        {1.0, 2.0}, {1.5, 1.8}, {5.0, 8.0}, {8.0, 8.0},
        {1.0, 0.6}, {9.0, 11.0}, {8.0, 2.0}, {10.0, 2.0}
    };
    
    int labels[NUM_POINTS];
    float centroids[K][NUM_FEATURES] = {
        {1.0, 2.0}, {5.0, 8.0} // Initial centroids
    };
    
    // Call k-Means Clustering
    printf("k-Means Clustering:\n");
    kmeans_clustering((float *)data, NUM_POINTS, NUM_FEATURES, K, labels, (float *)centroids);
    printf("Labels: ");
    for (int i = 0; i < NUM_POINTS; i++) {
        printf("%d ", labels[i]);
    }
    printf("\nCentroids:\n");
    for (int i = 0; i < K; i++) {
        printf("Centroid %d: (%.2f, %.2f)\n", i, centroids[i][0], centroids[i][1]);
    }
    printf("\n");
    
    // Call Hierarchical Clustering
    printf("Hierarchical Clustering:\n");
    HierarchicalClusteringResult hc_result = hierarchical_clustering((float *)data, NUM_POINTS, NUM_FEATURES);
    printf("Clusters: ");
    for (int i = 0; i < NUM_POINTS; i++) {
        printf("%d ", hc_result.clusters[i]);
    }
    printf("\n");
    free(hc_result.clusters);
    free(hc_result.distances);
    
    // Call Principal Component Analysis (PCA)
    printf("Principal Component Analysis (PCA):\n");
    float principal_components[NUM_FEATURES * NUM_FEATURES];
    pca((float *)data, NUM_POINTS, NUM_FEATURES, principal_components, NUM_FEATURES);
    printf("Principal Components:\n");
    for (int i = 0; i < NUM_FEATURES; i++) {
        printf("PC %d: (%.2f, %.2f)\n", i + 1, principal_components[i * NUM_FEATURES], principal_components[i * NUM_FEATURES + 1]);
    }
    printf("\n");

    // Call t-Distributed Stochastic Neighbor Embedding (t-SNE)
    printf("t-Distributed Stochastic Neighbor Embedding (t-SNE):\n");
    float embedded_data[NUM_POINTS][NUM_FEATURES];
    tsne((float *)data, NUM_POINTS, NUM_FEATURES, (float *)embedded_data, NUM_FEATURES, 30, 1000);
    printf("Embedded Data:\n");
    for (int i = 0; i < NUM_POINTS; i++) {
        printf("Point %d: (%.2f, %.2f)\n", i, embedded_data[i][0], embedded_data[i][1]);
    }
    printf("\n");

    // Call Gaussian Mixture Models (GMM)
    printf("Gaussian Mixture Models (GMM):\n");
    GMM gmm;
    gmm.means = (float *)malloc(NUM_COMPONENTS * NUM_FEATURES * sizeof(float));
    gmm.covariances = (float *)malloc(NUM_COMPONENTS * NUM_FEATURES * sizeof(float));
    gmm.weights = (float *)malloc(NUM_COMPONENTS * sizeof(float));

    gmm_fit((float *)data, NUM_POINTS, NUM_FEATURES, NUM_COMPONENTS, &gmm, 100);
    for (int i = 0; i < NUM_POINTS; i++) {
        int predicted_cluster = gmm_predict(&gmm, (float *)data[i], NUM_FEATURES, NUM_COMPONENTS);
        printf("Point %d belongs to cluster %d\n", i, predicted_cluster);
    }

    free(gmm.means);
    free(gmm.covariances);
    free(gmm.weights);

    return 0;
}
