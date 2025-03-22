import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Load extracted features
features, _ = joblib.load("D:/work/code/ML2Proj/3. Exploratory data analysis/animals_features.pkl")

# DBSCAN with best parameters
eps = 19
min_samples = 7
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(features)

# Save the DBSCAN labels
joblib.dump(labels, "D:/work/code/ML2Proj/3. Exploratory data analysis/dbscan_final_labels.pkl")

# Cluster distribution analysis
unique_labels, counts = np.unique(labels, return_counts=True)
cluster_info = {label: count for label, count in zip(unique_labels, counts)}

# Save cluster distribution to file
with open("D:/work/code/ML2Proj/3. Exploratory data analysis/dbscan_cluster_distribution.txt", "w") as f:
    for label, count in cluster_info.items():
        f.write(f"Cluster {label}: {count} samples\n")

# Compute silhouette score
valid_labels = labels[labels != -1]  # Remove noise points for silhouette score
if len(set(valid_labels)) > 1:  # Silhouette score needs at least 2 clusters
    silhouette = silhouette_score(features[labels != -1], valid_labels)
else:
    silhouette = -1  # Invalid score if only one cluster

# Save silhouette score
with open("D:/work/code/ML2Proj/3. Exploratory data analysis/dbscan_silhouette_score.txt", "w") as f:
    f.write(f"Silhouette Score: {silhouette:.4f}\n")

# PCA for visualization
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features)

# Scatter plot of clusters
plt.figure(figsize=(8, 6))
for label in np.unique(labels):
    mask = labels == label
    plt.scatter(features_pca[mask, 0], features_pca[mask, 1], label=f"Cluster {label}", alpha=0.5)
plt.legend()
plt.title(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("D:/work/code/ML2Proj/3. Exploratory data analysis/dbscan_pca_plot.png")
plt.close()
