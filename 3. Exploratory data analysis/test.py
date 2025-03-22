import joblib
import numpy as np

# Load extracted features
features_path = "D:/work/code/ML2Proj/3. Exploratory data analysis/animals_features.pkl"
features = joblib.load(features_path)

# Check the shape of each feature vector
for i, feature in enumerate(features[:5]):  # Checking the first 5 elements
    print(f"Feature {i} shape:", np.array(feature).shape)
