import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os

# Load extracted features
features, labels = joblib.load("D:/work/code/ML2Proj/3. Exploratory data analysis/animals_features.pkl")

# Parameter grid to test
eps_values = range(18,21)
min_samples_values = range(7,11)

# Output directory
output_dir = "D:/work/code/ML2Proj/3. Exploratory data analysis/"
os.makedirs(output_dir, exist_ok=True)

# Iterate over different parameter combinations
for eps in eps_values:
    for min_samples in min_samples_values:
        print(f"Running DBSCAN with eps={eps}, min_samples={min_samples}")
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(features)
        
        # Save clustering labels
        label_filename = f"dbscan_labels_eps{eps}_min{min_samples}.pkl"
        joblib.dump(cluster_labels, os.path.join(output_dir, label_filename))
        
        # Compute metrics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        silhouette = silhouette_score(features, cluster_labels) if n_clusters > 1 else -1
        
        # Save evaluation results
        eval_filename = f"dbscan_eval_eps{eps}_min{min_samples}.txt"
        with open(os.path.join(output_dir, eval_filename), "w") as f:
            f.write(f"DBSCAN with eps={eps}, min_samples={min_samples}\n")
            f.write(f"Number of clusters: {n_clusters}\n")
            f.write(f"Number of noise points: {n_noise}\n")
            f.write(f"Silhouette Score: {silhouette:.4f}\n")
        
        # PCA for visualization
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(features)
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='viridis', s=5)
        plt.colorbar(scatter, label="Cluster Label")
        plt.title(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.savefig(os.path.join(output_dir, f"dbscan_pca_eps{eps}_min{min_samples}.png"))
        plt.close()

print("DBSCAN parameter tuning completed. All results saved.")
