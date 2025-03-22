import os
from collections import Counter
from PIL import Image

# Dataset path
dataset_path = "D:/work/animals/raw-img"

# Store format and size details
formats = Counter()
sizes = Counter()

# Get all class folders
classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)
    image_files = [f for f in os.listdir(class_path) if f.endswith(('jpg', 'png', 'jpeg'))]

    for image_file in image_files:
        img_path = os.path.join(class_path, image_file)
        try:
            with Image.open(img_path) as img:
                formats[img.format] += 1  # Count image formats
                sizes[img.size] += 1  # Count image sizes
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Print format distribution
print("\nImage Formats:")
for fmt, count in formats.items():
    print(f"{fmt}: {count} images")

# Print top 10 most common sizes
print("\nTop 10 Image Sizes:")
for size, count in sizes.most_common(10):
    print(f"{size}: {count} images")

# Print number of unique resolutions
print(f"\nTotal unique resolutions: {len(sizes)}")
