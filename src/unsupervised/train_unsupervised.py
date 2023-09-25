import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

def standardize_features(X):
    """
    This function standardizes the features using sklearn's StandardScaler. 
    It assumes that the input is a pandas DataFrame.

    Args:
        X (pd.DataFrame): The input features to be standardized.

    Returns:
        X_scaled_df (pd.DataFrame): The standardized features, returned as a pandas DataFrame.

    Note:
        The function does not handle exceptions, so the input DataFrame should not contain any non-numeric columns.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    return X_scaled_df

def perform_pca(X, explained_variance = 0.85):
    """
    This function performs Principal Component Analysis (PCA) on the input data.
    It determines the number of components required to explain the specified amount of variance.

    Args:
        X (np.array or pd.DataFrame): The input data for PCA.
        explained_variance (float): The amount of variance to be explained by the PCA.

    Returns:
        X_pca (np.array): The transformed data after applying PCA.
        d (int): The optimal number of components.

    Note:
        The function assumes that you have imported the necessary libraries (like sklearn, numpy, pandas etc.)
    """

    pca = PCA()
    X_pca = pca.fit_transform(X)
    
    cum_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    
    d = np.argmax(cum_explained_variance >= explained_variance) + 1

    print('Optimal number of components:', d)
    
    return PCA(n_components=d).fit_transform(X), d

def perform_kmeans(X, n_clusters, n_init=10, random_state=42):
    """
    This function performs K-Means clustering on the input data and returns the cluster labels.
    The function tries multiple initializations and chooses the one with the best silhouette score.

    Args:
        X (np.array or pd.DataFrame): The input data for clustering.
        n_clusters (int): The number of clusters for KMeans.
        n_init (int): The number of time the k-means algorithm will be run with different centroid seeds.
        random_state (int): Determines random number generation for centroid initialization.

    Returns:
        labels_df (pd.DataFrame): The cluster labels for each sample.

    Note:
        The function assumes that you have imported the necessary libraries (like sklearn, numpy, pandas etc.)
    """

    if not isinstance(n_clusters, int) or n_clusters <= 0:
        raise ValueError("n_clusters must be a positive integer")

    if not isinstance(n_init, int) or n_init <= 0:
        raise ValueError("n_init must be a positive integer")

    best_silhouette = -1
    best_model = None
    best_labels = None

    for _ in range(n_init):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=random_state)
        kmeans.fit(X)
        labels = kmeans.labels_
        silhouette = silhouette_score(X, labels)

        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_model = kmeans
            best_labels = labels

    davies_bouldin = davies_bouldin_score(X, best_labels)

    print(f'Best Silhouette score for {n_clusters} clusters: {best_silhouette}')
    print(f'Davies-Bouldin score for {n_clusters} clusters: {davies_bouldin}')

    labels_df = pd.DataFrame(best_labels + 1, columns=['cluster'])
    return labels_df, best_model, best_silhouette, davies_bouldin

