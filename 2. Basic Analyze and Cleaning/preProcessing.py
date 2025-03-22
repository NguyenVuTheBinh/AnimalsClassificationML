import os
from collections import Counter
from PIL import Image

# Dataset path
dataset_path = "D:/work/animals/raw-img"

# Class name translations (Italian ↔ English)
class_translation = {
    "cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly",
    "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep",
    "scoiattolo": "squirrel", "ragno": "spider"
}

# Reverse mapping for lookup
reverse_translation = {v: k for k, v in class_translation.items()}

# Get class names from folder structure
classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

# Count images and check for corrupt files
image_counts = {}
corrupt_images = []

for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)
    image_files = [f for f in os.listdir(class_path) if f.endswith(('jpg', 'png', 'jpeg'))]

    # Use English names if available
    display_name = class_translation.get(class_name, class_name)
    image_counts[display_name] = len(image_files)

    # Check for corrupt images
    for image_file in image_files:
        img_path = os.path.join(class_path, image_file)
        try:
            with Image.open(img_path) as img:
                img.verify()  # Check if image can be opened
        except Exception:
            corrupt_images.append(img_path)

# Print class distribution
print("Image distribution per class:")
for class_name, count in image_counts.items():
    print(f"{class_name}: {count} images")

# Print corrupt images
if corrupt_images:
    print("\nCorrupt images detected:")
    for img_path in corrupt_images:
        print(img_path)
else:
    print("\nNo corrupt images found.")
