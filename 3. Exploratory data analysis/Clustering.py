import os
import joblib
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd

# Load extracted features
features_path = "D:/work/code/ML2Proj/3. Exploratory data analysis/animals_features.pkl"
features, labels = joblib.load(features_path)

# Ensure it's in NumPy array format
if isinstance(features, dict):  
    features = np.array(list(features.values()))  

# Step 1: K-Means Clustering
def kmeans_clustering(features, max_k=10):
    distortions = []
    silhouette_scores = []

    for k in range(2, max_k + 1):  
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(features, cluster_labels))

    return distortions, silhouette_scores


# Run K-Means clustering
distortions, silhouette_scores = kmeans_clustering(features)

# Run K-Means again to get final labels (using optimal k from elbow method)
optimal_k = np.argmin(distortions) + 2  # +2 because k starts from 2
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(features)

# Save K-Means clustering labels
joblib.dump(kmeans_labels, "D:/work/code/ML2Proj/3. Exploratory data analysis/kmeans_labels.pkl")


# Save results using Joblib
joblib.dump({"distortions": distortions, "silhouette_scores": silhouette_scores}, 
            "D:/work/code/ML2Proj/3. Exploratory data analysis/kmeans_metrics.pkl")

# Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(2, len(distortions) + 2), distortions, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion')
plt.title('Elbow Method for K-Means')
plt.savefig("D:/work/code/ML2Proj/3. Exploratory data analysis/kmeans_elbow.png")
plt.close()

# Plot Silhouette Score
plt.figure(figsize=(8, 5))
plt.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker='s', linestyle='-', color='r')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for K-Means')
plt.savefig("D:/work/code/ML2Proj/3. Exploratory data analysis/kmeans_silhouette.png")
plt.close()

# Step 2: DBSCAN Clustering
dbscan = DBSCAN(eps=15, min_samples=8)  
dbscan_labels = dbscan.fit_predict(features)

# Save DBSCAN results
joblib.dump(dbscan_labels, "D:/work/code/ML2Proj/3. Exploratory data analysis/dbscan_labels.pkl")

kmeans_labels = joblib.load("D:/work/code/ML2Proj/3. Exploratory data analysis/kmeans_labels.pkl")

pca = PCA(n_components=2)
features_pca = pca.fit_transform(features)

df_pca = pd.DataFrame(features_pca, columns=["PC1", "PC2"])
df_pca["Cluster"] = kmeans_labels

plt.figure(figsize=(8, 6))
sns.scatterplot(x="PC1", y="PC2", hue="Cluster", palette="viridis", data=df_pca)
plt.title("K-Means Clusters (PCA Projection)")

# Save the figure
plt.savefig("D:/work/code/ML2Proj/3. Exploratory data analysis/kmeans_pca.png")

#t-SNE
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
features_tsne = tsne.fit_transform(features)

df_tsne = pd.DataFrame(features_tsne, columns=["TSNE1", "TSNE2"])
df_tsne["Cluster"] = kmeans_labels

plt.figure(figsize=(8, 6))
sns.scatterplot(x="TSNE1", y="TSNE2", hue="Cluster", palette="viridis", data=df_tsne)
plt.title("K-Means Clusters (t-SNE Projection)")

# Save the figure
plt.savefig("D:/work/code/ML2Proj/3. Exploratory data analysis/kmeans_tsne.png")
plt.close()

#save DBSCAN result
import numpy as np

dbscan_labels = joblib.load("D:/work/code/ML2Proj/3. Exploratory data analysis/dbscan_labels.pkl")
unique, counts = np.unique(dbscan_labels, return_counts=True)

dbscan_cluster_distribution = {label: count for label, count in zip(unique, counts)}

# Save as text file
with open("D:/work/code/ML2Proj/3. Exploratory data analysis/dbscan_cluster_distribution.txt", "w") as f:
    for label, count in dbscan_cluster_distribution.items():
        f.write(f"Cluster {label}: {count} samples\n")



print("All done")