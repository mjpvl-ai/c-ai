#ifndef UNSUPERVISED_LEARNING_H
#define UNSUPERVISED_LEARNING_H

// k-Means Clustering
void kmeans_clustering(float *data, int num_points, int num_features, int k, int *labels, float *centroids);

// Hierarchical Clustering
typedef struct {
    int *clusters;
    float *distances;
} HierarchicalClusteringResult;

HierarchicalClusteringResult hierarchical_clustering(float *data, int num_points, int num_features);

// Principal Component Analysis (PCA)
void pca(float *data, int num_points, int num_features, float *principal_components, int num_components);

// t-Distributed Stochastic Neighbor Embedding (t-SNE)
void tsne(float *data, int num_points, int num_features, float *embedded_data, int num_dimensions, int perplexity, int iterations);

// Gaussian Mixture Models (GMM)
typedef struct {
    float *means;
    float *covariances;
    float *weights;
} GMM;

void gmm_fit(float *data, int num_points, int num_features, int num_components, GMM *gmm, int max_iterations);
int gmm_predict(GMM *gmm, float *point, int num_features, int num_components);

#endif // UNSUPERVISED_LEARNING_H
