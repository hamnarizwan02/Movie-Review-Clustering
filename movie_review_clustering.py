import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df = df.dropna(subset=['Rating'])
    return df

def assign_rating_clusters(ratings):      #as given in assignment
    return np.select([ratings >= 8, (ratings >= 6) & (ratings < 8), ratings < 6],
                     [2, 1, 0], default=-1)

def kmeans_clustering(X, k, max_iters=100):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    iteration_count = 0

    for i in range(max_iters):
        iteration_count += 1
        distances = np.abs(X[:, np.newaxis] - centroids)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([X[labels == i].mean() for i in range(k)])

        if np.all(np.abs(centroids - new_centroids) < 1e-5):
            break

        centroids = new_centroids

    print(f"Converged after {iteration_count} iterations")
    return labels, centroids

def compare_genres_within_clusters(df, labels):
    for i in range(3):
        cluster_genres = df[labels == i]['Genre'].value_counts()
        print(f"Cluster {i} top genres:")
        print(cluster_genres.head())
        print()

def confusion_matrix(true_labels, pred_labels):       #create confusion matrix
    k = max(max(true_labels), max(pred_labels)) + 1
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[t, p] += 1
    return cm

def visualize_clusters(X, labels, centroids):
    plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.scatter(X[labels == i], np.zeros_like(X[labels == i]), alpha=0.5, label=f'Cluster {i}')
    for i, c in enumerate(centroids):
        plt.axvline(x=c, color='red', linestyle='--', label=f'Centroid {i}')
    plt.yticks([])
    plt.xlabel('Rating')
    plt.title('Movie Rating Clusters')
    plt.legend()
    plt.show()

def main():
    df = load_and_preprocess_data('IMDB-Movie-Data.csv')

    true_clusters = assign_rating_clusters(df['Rating'])

    ratings = df['Rating'].values                           #K-means clustering
    labels, centroids = kmeans_clustering(ratings, k=3)

    print("Final centroid values:")
    for i, centroid in enumerate(centroids):
        print(f"Cluster {i}: Rating = {centroid:.2f}")

    compare_genres_within_clusters(df, labels)

    cm = confusion_matrix(true_clusters, labels)
    print("Confusion Matrix:")
    print(cm)

    visualize_clusters(ratings, labels, centroids)

if __name__ == "__main__":
    main()