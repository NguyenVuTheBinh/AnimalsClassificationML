import numpy as np
import joblib
import os
import cv2
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load extracted features and labels
data_path = "D:/work/code/ML2Proj/3. Exploratory data analysis/animals_features.pkl"
features, labels = joblib.load(data_path)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

# Train Decision Tree classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Make predictions
y_pred = dt.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
output_model_path = "D:/work/code/ML2Proj/4. Model Building/decision_tree_model.pkl"
joblib.dump(dt, output_model_path)

# Plot and save confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(labels), yticklabels=np.unique(labels))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix for Decision Tree")
plt.savefig("D:/work/code/ML2Proj/4. Model Building/decision_tree_confusion_matrix.png")
plt.close()

# Functions to preprocess new images
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

def preprocess_image(image_path):
    """Extract features from an image (same method used in training)."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    color_hist = extract_color_histogram(image)
    edge_feature = [extract_edges(image)]
    texture_features = extract_texture_features(image)

    return np.hstack([color_hist, edge_feature, texture_features]).reshape(1, -1)  # Match training shape


# Function to predict new image
def predict_new_image(image_path, model_path):
    model = joblib.load(model_path)
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    print(f"Predicted label: {prediction[0]}")
    return prediction[0]

new_image_path = "D:/work/code/ML2Proj/5. Model Deployment/test_image.jpg" 
predict_new_image(new_image_path, output_model_path)