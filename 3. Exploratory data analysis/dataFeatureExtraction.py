import os
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from PIL import Image
import joblib  # To save extracted features

# Define feature extraction functions
def extract_color_histogram(image, bins=(8, 8, 8)):
    """Extract color histogram features from an image."""
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_edges(image):
    """Extract edge features using Canny edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.mean(edges)  # Using mean edge intensity as a feature

def extract_texture_features(image):
    """Extract texture features using Gray-Level Co-occurrence Matrix (GLCM)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return [contrast, energy, homogeneity]

def extract_features(image_path):
    """Extract all features from an image."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    color_hist = extract_color_histogram(image)
    edge_feature = [extract_edges(image)]
    texture_features = extract_texture_features(image)

    return np.hstack([color_hist, edge_feature, texture_features])

# Process all images
dataset_path = "D:/work/code/ML2Proj/1. Data collection, Animals Processed"
output_path = "D:/work/code/ML2Proj/3. Exploratory data analysis/animals_features.pkl"

features = []
labels = []

for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            feature_vector = extract_features(img_path)
            features.append(feature_vector)
            labels.append(class_name)

# Save extracted features
features = np.array(features)
labels = np.array(labels)
joblib.dump((features, labels), output_path)

print("Feature extraction completed! Features saved to:", output_path)
