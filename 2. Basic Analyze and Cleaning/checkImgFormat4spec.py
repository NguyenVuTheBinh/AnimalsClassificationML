import os
from collections import Counter
from PIL import Image

# Dataset path
dataset_path = "D:/work/animals/raw-img"

# Selected classes for analysis
selected_classes = ["chicken", "dog", "horse", "squirrel"]

# Store format and size details
formats = Counter()
sizes = Counter()

# Analyze only selected classes
for class_name in selected_classes:
    class_path = os.path.join(dataset_path, class_name)
    if not os.path.exists(class_path):
        print(f"Class folder '{class_name}' not found, skipping.")
        continue

    image_files = [f for f in os.listdir(class_path) if f.endswith(('jpg', 'png', 'jpeg'))]

    for image_file in image_files:
        img_path = os.path.join(class_path, image_file)
        try:
            with Image.open(img_path) as img:
                formats[img.format] += 1  # Count image formats
                sizes[img.size] += 1  # Count image sizes
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Print results
print("\nImage Formats:")
for fmt, count in formats.items():
    print(f"{fmt}: {count} images")

print("\nTop 10 Image Sizes:")
for size, count in sizes.most_common(50):
    print(f"{size}: {count} images")

print(f"\nTotal unique resolutions: {len(sizes)}")
