import os
import numpy as np
import cv2
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from skimage.feature import graycomatrix, graycoprops

# Load extracted features and labels
data_path = "D:/work/code/ML2Proj/3. Exploratory data analysis/animals_features.pkl"
features, labels = joblib.load(data_path)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

# Feature extraction functions
def extract_color_histogram(image, bins=(8, 8, 8)):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.mean(edges)

def extract_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return [contrast, energy, homogeneity]

def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    color_hist = extract_color_histogram(image)
    edge_feature = [extract_edges(image)]
    texture_features = extract_texture_features(image)
    return np.hstack([color_hist, edge_feature, texture_features])

def predict_multiple_images(image_paths, model_path):
    model = joblib.load(model_path)
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        image_features = extract_features(image_path).reshape(1, -1)
        prediction = model.predict(image_features)[0]
        print(f"Image: {os.path.basename(image_path)} -> Predicted label: {prediction}")

# Example usage
image_folder = "D:/work/code/ML2Proj/5. Model Deployment/test_images"  # Folder containing test images
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

image_paths = sorted(
    [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.lower().endswith(('png', 'jpg', 'jpeg'))],
    key=natural_sort_key
)
predict_multiple_images(image_paths, "D:/work/code/ML2Proj/4. Model Building/decision_tree_model.pkl")
