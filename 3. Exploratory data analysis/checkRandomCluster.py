import joblib
import os
from collections import Counter

# Load DBSCAN labels
dbscan_labels = joblib.load("D:/work/code/ML2Proj/3. Exploratory data analysis/dbscan_final_labels.pkl")

# Load feature data (features and corresponding image paths)
features, image_paths = joblib.load("D:/work/code/ML2Proj/3. Exploratory data analysis/animals_features.pkl")

# Find images in Cluster 1
cluster_1_images = [image_paths[i] for i in range(len(dbscan_labels)) if dbscan_labels[i] == 1]

# Save the list of Cluster 1 images
output_file = "D:/work/code/ML2Proj/3. Exploratory data analysis/cluster_1_images.txt"
with open(output_file, "w") as f:
    for img in cluster_1_images:
        f.write(img + "\n")

# Find images in Cluster 1
cluster_2_images = [image_paths[i] for i in range(len(dbscan_labels)) if dbscan_labels[i] == 2]

# Save the list of Cluster 1 images
output_file = "D:/work/code/ML2Proj/3. Exploratory data analysis/cluster_2_images.txt"
with open(output_file, "w") as f:
    for img in cluster_2_images:
        f.write(img + "\n")

print(f"Cluster 1 images saved to {output_file}")

# Analyze class distribution in Cluster 1
cluster_1_classes = [os.path.basename(os.path.dirname(img)) for img in cluster_1_images]
class_counts = Counter(cluster_1_classes)

# Save class distribution to a text file
dist_output = "D:/work/code/ML2Proj/3. Exploratory data analysis/cluster_1_class_distribution.txt"
with open(dist_output, "w") as f:
    for class_name, count in class_counts.items():
        f.write(f"{class_name}: {count}\n")

print(f"Cluster 1 class distribution saved to {dist_output}")
